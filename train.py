"""Train the model.

    python train.py --data data/prepared --out runs/v1
"""
# argparse: command-line options; math: isfinite check; os: paths/folders; time: throughput timing
import argparse, math, os, time
# partial: pre-fills mx.compile's keyword arguments so it can wrap the step function
from functools import partial
# numpy: random number generator for batches, and averaging logged losses
import numpy as np
# mlx.core: Apple's array library (arrays, lazy evaluation, compile, random seed)
import mlx.core as mx
# mlx.nn: loss functions and value_and_grad (computes loss and gradients together)
import mlx.nn as nn
# mlx.optimizers: AdamW, learning-rate schedules and gradient clipping
import mlx.optimizers as optim
# tokenizer: provides the vocabulary size for building the model
import tokenizer as T
# Dataset: random training batches; fixed_batches: a constant validation set
from data import Dataset, fixed_batches
# Config: model hyperparameters; Pianist: the GPT-style model
from model import Config, Pianist


def loss_fn(model, x, y, mask):
    """Average cross-entropy over the positions where mask is 1."""
    # forward pass: logits [B, T, vocab] are scores for every possible next token (cache ignored)
    logits, _ = model(x)
    # cross-entropy per position: low when the model gave the true next token y high probability;
    # "none" keeps one value per position instead of averaging
    ce = nn.losses.cross_entropy(logits, y, reduction="none")
    # zero out padding and tag positions, then average over the music positions only
    return (ce * mask).sum() / mask.sum()


def evaluate(model, batches):
    """Mean loss over fixed batches, with dropout off."""
    # evaluation mode: disables dropout so results are deterministic
    model.eval()
    # loss on each batch (numpy arrays converted to MLX arrays), .item() pulls the number out
    total = sum(loss_fn(model, *map(mx.array, b)).item() for b in batches)
    # back to training mode (dropout on) for the training loop
    model.train()
    # average of the per-batch losses
    return total / len(batches)


def main():
    """Parse arguments, build or resume the model, and run the training loop."""
    # command-line options
    ap = argparse.ArgumentParser()
    # folder holding train.npz and validation.npz from prepare.py
    ap.add_argument("--data", default="data/prepared")
    # folder for checkpoints and the log
    ap.add_argument("--out", default="runs/v1")
    # total number of optimiser updates
    ap.add_argument("--steps", type=int, default=20000)
    # passages per batch
    ap.add_argument("--batch", type=int, default=32)
    # tokens per training sequence (the model's context length)
    ap.add_argument("--ctx", type=int, default=1024)
    # width of each token vector inside the model
    ap.add_argument("--dim", type=int, default=512)
    # number of transformer layers
    ap.add_argument("--layers", type=int, default=6)
    # number of attention heads per layer
    ap.add_argument("--heads", type=int, default=8)
    # fraction of activations randomly zeroed during training, to reduce overfitting
    ap.add_argument("--dropout", type=float, default=0.1)
    # peak learning rate (reached after warm-up)
    ap.add_argument("--lr", type=float, default=3e-4)
    # steps over which the learning rate ramps up from 0 to --lr
    ap.add_argument("--warmup", type=int, default=1000)
    # run validation and save checkpoints every this many steps
    ap.add_argument("--eval-every", type=int, default=500)
    # number of fixed validation batches
    ap.add_argument("--eval-batches", type=int, default=20)
    # continue from the "last" checkpoint in --out if there is one
    ap.add_argument("--resume", action="store_true")
    # skip mx.compile (slower, but easier to debug)
    ap.add_argument("--no-compile", action="store_true")
    # base random seed for batch sampling and dropout
    ap.add_argument("--seed", type=int, default=0)
    # read the options from the command line
    args = ap.parse_args()
    # create the output folder (no error if it already exists)
    os.makedirs(args.out, exist_ok=True)

    # the training split, from which random passages are drawn every step
    train = Dataset(os.path.join(args.data, "train.npz"))
    # a fixed list of validation batches (same every evaluation, so losses are comparable)
    val = fixed_batches(Dataset(os.path.join(args.data, "validation.npz")),
                        args.eval_batches, args.batch, args.ctx)

    # path prefix of the most recent checkpoint (last.json + last.safetensors)
    last = os.path.join(args.out, "last")
    # step to continue after (0 = fresh run) and the best validation loss seen so far
    start, best = 0, float("inf")
    # resume only if asked and a previous checkpoint actually exists
    if args.resume and os.path.exists(last + ".json"):
        # rebuild the model from its saved config and weights; info holds step and best loss
        model, info = Pianist.load(last)
        # load() returns the model in eval mode; switch dropout back on for training
        model.train()
        # continue counting steps and tracking the best loss from where the last run stopped
        start, best = info["step"], info["best"]
        # tell the user
        print(f"resumed from step {start}")
    # otherwise build a brand-new model with random weights from the command-line sizes
    else:
        # vocab_size comes from the tokenizer; the rest from the options above
        model = Pianist(Config(vocab_size=T.VOCAB_SIZE, ctx=args.ctx, dim=args.dim,
                               n_layers=args.layers, n_heads=args.heads,
                               dropout=args.dropout))
    # MLX is lazy: force the weights to actually be created/loaded in memory now
    mx.eval(model.parameters())
    # seed MLX's random numbers (dropout); adding start gives fresh random numbers after a resume
    mx.random.seed(args.seed + start)           # fresh random numbers after a resume
    # numpy generator for choosing passages and augmentations, seeded the same way
    rng = np.random.default_rng(args.seed + start)
    # report model size (in millions of parameters) and number of training pieces
    print(f"{model.n_params() / 1e6:.1f} M parameters, {len(train)} training pieces")

    # learning rate: linear warm-up, then a cosine curve down to 10 %
    # warm-up: rise linearly from 0 to --lr over the first --warmup steps (avoids unstable early updates)
    warm = optim.linear_schedule(0.0, args.lr, args.warmup)
    # then a smooth cosine decay from --lr to 10 % of it over the remaining steps (at least 1)
    cosine = optim.cosine_decay(args.lr, max(args.steps - args.warmup, 1), args.lr * 0.1)
    # glue the two: use `warm` until step --warmup, then `cosine`
    schedule = optim.join_schedules([warm, cosine], [args.warmup])
    # AdamW optimiser using that schedule; betas control momentum averaging, weight_decay
    # gently shrinks weights each step to discourage overfitting
    optimizer = optim.AdamW(learning_rate=schedule, betas=[0.9, 0.95], weight_decay=0.01)
    # create the optimiser's per-parameter state (momentum buffers) now, so it can be compiled
    optimizer.init(model.trainable_parameters())
    # set the optimiser's step counter so the schedule continues where a resumed run left off
    # (note: only the counter is restored; the momentum buffers start fresh)
    optimizer.state["step"] = mx.array(start, dtype=mx.uint64)   # resume the schedule

    # wrap loss_fn so one call returns both the loss and its gradients w.r.t. the model's weights
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    # everything the compiled step reads and changes besides its arguments:
    # model weights, optimiser state and the random generator state (for dropout); mx.compile
    # must be told about them so updates made inside the compiled graph are kept
    state = [model.state, optimizer.state, mx.random.state]

    def step(x, y, mask):
        """One optimiser update on one batch; returns the loss."""
        # forward and backward pass: the loss and the gradient of every weight
        loss, grads = loss_and_grad(model, x, y, mask)
        # if the gradients' overall size (L2 norm) exceeds 1.0, scale them all down to 1.0,
        # so one unusual batch cannot make a huge destabilising update
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        # apply AdamW: adjust the weights in place using the gradients and current learning rate
        optimizer.update(model, grads)
        # hand the loss back for logging
        return loss

    # by default, compile the step: MLX traces it once and fuses it into one optimised graph;
    # inputs/outputs=state tells it which outside arrays the step reads and modifies
    if not args.no_compile:          # build the whole step into one fast graph
        # replace step with its compiled version
        step = partial(mx.compile, inputs=state, outputs=state)(step)

    # open the CSV log in append mode, so a resumed run continues the same file
    log = open(os.path.join(args.out, "log.csv"), "a")
    # a fresh run starts the file with a header row
    if start == 0:
        # column names
        log.write("step,train_loss,val_loss,tokens_per_s\n")
    # recent training losses, timer start, and tokens processed since the last report
    running, t0, seen = [], time.time(), 0
    # step numbers are 1-based; a resumed run continues from start + 1
    for it in range(start + 1, args.steps + 1):
        # draw a fresh random batch (numpy arrays) with augmentation and tag dropout
        x, y, mask = train.batch(rng, args.batch, args.ctx)
        # run one optimisation step on the batch (converted to MLX arrays)
        loss = step(mx.array(x), mx.array(y), mx.array(mask))
        # MLX is lazy: the step above only built a computation; evaluate it (and the updated
        # weights/optimiser state) now, so work does not pile up across iterations
        mx.eval(state, loss)                         # MLX is lazy: run it now
        # store the loss as a plain Python float
        running.append(loss.item())
        # count tokens processed (batch * (ctx - 1)) for the throughput figure
        seen += x.size
        # stop if training diverged (loss became NaN or infinite)
        if not math.isfinite(running[-1]):
            # exit with a hint on how to fix it
            raise SystemExit("loss is not finite: lower --lr")

        # every 50 steps print a progress line
        if it % 50 == 0:
            # mean of the last 50 losses, and tokens per second since the last evaluation
            print(f"step {it:6d}  loss {np.mean(running[-50:]):.3f}  "
                  f"{seen / (time.time() - t0):,.0f} tokens/s", flush=True)
        # periodically (and on the final step) validate and save checkpoints
        if it % args.eval_every == 0 or it == args.steps:
            # loss on the fixed validation batches
            v = evaluate(model, val)
            # training throughput since the last evaluation
            tps = seen / (time.time() - t0)
            # append a row to the CSV and flush it to disk immediately
            log.write(f"{it},{np.mean(running):.4f},{v:.4f},{tps:.0f}\n"); log.flush()
            # note appended to the printed line when a new best is saved
            flag = ""
            # a new lowest validation loss: remember it and save it as the "best" checkpoint
            if v < best:
                # update the record and set the note
                best, flag = v, "  (best, saved)"
                # write best.safetensors and best.json (with the step and best loss)
                model.save(os.path.join(args.out, "best"), step=it, best=best)
            # always save the latest state too, so --resume can continue from here
            model.save(last, step=it, best=best)
            # summary line: mean training loss since last evaluation, and validation loss
            print(f"== step {it}: train {np.mean(running):.3f}  val {v:.3f}{flag}", flush=True)
            # reset the loss list, timer and token count for the next interval
            running, t0, seen = [], time.time(), 0


# run main() only when executed as a script, not when imported
if __name__ == "__main__":
    # start training
    main()
