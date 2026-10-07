"""Train the model.

    python train.py --data data/prepared --out runs/v1
"""
import argparse, math, os, time
from functools import partial
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import tokenizer as T
from data import Dataset, fixed_batches
from model import Config, Pianist


def loss_fn(model, x, y, mask):
    """Average cross-entropy over the positions where mask is 1."""
    logits, _ = model(x)
    ce = nn.losses.cross_entropy(logits, y, reduction="none")
    return (ce * mask).sum() / mask.sum()


def evaluate(model, batches):
    """Mean loss over fixed batches, with dropout off."""
    model.eval()
    total = sum(loss_fn(model, *map(mx.array, b)).item() for b in batches)
    model.train()
    return total / len(batches)


def main():
    """Parse arguments, build or resume the model, and run the training loop."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/prepared")
    ap.add_argument("--out", default="runs/v1")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train = Dataset(os.path.join(args.data, "train.npz"))
    val = fixed_batches(Dataset(os.path.join(args.data, "validation.npz")),
                        args.eval_batches, args.batch, args.ctx)

    last = os.path.join(args.out, "last")
    start, best = 0, float("inf")
    if args.resume and os.path.exists(last + ".json"):
        model, info = Pianist.load(last)
        model.train()
        start, best = info["step"], info["best"]
        print(f"resumed from step {start}")
    else:
        model = Pianist(Config(vocab_size=T.VOCAB_SIZE, ctx=args.ctx, dim=args.dim,
                               n_layers=args.layers, n_heads=args.heads,
                               dropout=args.dropout))
    mx.eval(model.parameters())
    mx.random.seed(args.seed + start)           # fresh random numbers after a resume
    rng = np.random.default_rng(args.seed + start)
    print(f"{model.n_params() / 1e6:.1f} M parameters, {len(train)} training pieces")

    # learning rate: linear warm-up, then a cosine curve down to 10 %
    warm = optim.linear_schedule(0.0, args.lr, args.warmup)
    cosine = optim.cosine_decay(args.lr, max(args.steps - args.warmup, 1), args.lr * 0.1)
    schedule = optim.join_schedules([warm, cosine], [args.warmup])
    optimizer = optim.AdamW(learning_rate=schedule, betas=[0.9, 0.95], weight_decay=0.01)
    optimizer.init(model.trainable_parameters())
    optimizer.state["step"] = mx.array(start, dtype=mx.uint64)   # resume the schedule

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    # everything the compiled step reads and changes besides its arguments
    state = [model.state, optimizer.state, mx.random.state]

    def step(x, y, mask):
        """One optimiser update on one batch; returns the loss."""
        loss, grads = loss_and_grad(model, x, y, mask)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(model, grads)
        return loss

    if not args.no_compile:          # build the whole step into one fast graph
        step = partial(mx.compile, inputs=state, outputs=state)(step)

    log = open(os.path.join(args.out, "log.csv"), "a")
    if start == 0:
        log.write("step,train_loss,val_loss,tokens_per_s\n")
    running, t0, seen = [], time.time(), 0
    for it in range(start + 1, args.steps + 1):
        x, y, mask = train.batch(rng, args.batch, args.ctx)
        loss = step(mx.array(x), mx.array(y), mx.array(mask))
        mx.eval(state, loss)                         # MLX is lazy: run it now
        running.append(loss.item())
        seen += x.size
        if not math.isfinite(running[-1]):
            raise SystemExit("loss is not finite: lower --lr")

        if it % 50 == 0:
            print(f"step {it:6d}  loss {np.mean(running[-50:]):.3f}  "
                  f"{seen / (time.time() - t0):,.0f} tokens/s", flush=True)
        if it % args.eval_every == 0 or it == args.steps:
            v = evaluate(model, val)
            tps = seen / (time.time() - t0)
            log.write(f"{it},{np.mean(running):.4f},{v:.4f},{tps:.0f}\n"); log.flush()
            flag = ""
            if v < best:
                best, flag = v, "  (best, saved)"
                model.save(os.path.join(args.out, "best"), step=it, best=best)
            model.save(last, step=it, best=best)
            print(f"== step {it}: train {np.mean(running):.3f}  val {v:.3f}{flag}", flush=True)
            running, t0, seen = [], time.time(), 0


if __name__ == "__main__":
    main()
