"""Fine-tune a small Qwen to continue a caption with piano music.

    uv run python qwen/train_qwen.py --data data/maestro_prepared --out runs/qwen
"""
import argparse, math, os, time
import numpy as np
import torch
from music_lm import load, make_batch, music_loss
from data import Dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--data", default="data/maestro_prepared")
    ap.add_argument("--out", default="runs/qwen")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accumulate", type=int, default=1, help="batches per update")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    last = os.path.join(args.out, "last")
    # resume only if a previous run left an optimiser state behind
    resume = args.resume and os.path.exists(os.path.join(last, "state.pt"))

    # The weights are kept in 32-bit floats for training. Qwen ships in
    # bfloat16, which is too coarse to absorb small optimiser updates.
    model, tok, base = load(last if resume else args.base, device, torch.float32)
    train = Dataset(os.path.join(args.data, "train.npz"))
    val_ds = Dataset(os.path.join(args.data, "validation.npz"))
    val_rng = np.random.default_rng(1234)
    # fixed validation batches, built once, so every evaluation sees the same music
    val = [make_batch(val_ds, tok, base, val_rng, args.batch, args.ctx,
                      augment=False, tag_dropout=False) for _ in range(args.eval_batches)]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params / 1e6:.0f} M parameters, {next(model.parameters()).dtype}, on {device}")

    # No weight decay on the embedding table: decay would slowly shrink the
    # thousands of text rows that this training never touches.
    embed = model.get_input_embeddings().weight
    rest = [p for p in model.parameters() if p is not embed]
    opt = torch.optim.AdamW([{"params": [embed], "weight_decay": 0.0},
                             {"params": rest, "weight_decay": 0.01}],
                            lr=args.lr, betas=(0.9, 0.95))
    start, best = 0, float("inf")
    if resume:
        state = torch.load(os.path.join(last, "state.pt"), map_location=device)
        opt.load_state_dict(state["optimizer"])
        start, best = state["step"], state["best"]
        print(f"resumed from step {start}")
    torch.manual_seed(args.seed + start)
    rng = np.random.default_rng(args.seed + start)

    def lr_at(step):                 # linear warm-up, cosine down to 10 %
        if step < args.warmup:
            return args.lr * step / args.warmup
        t = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))

    def run(batch):
        x, att, y = (t.to(device) for t in batch)
        # on a GPU compute in bfloat16 while the weights stay in float32
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            return music_loss(model, base, x, att, y)

    running, t0, seen = [], time.time(), 0
    log = open(os.path.join(args.out, "log.csv"), "a")
    model.train()
    for it in range(start + 1, args.steps + 1):
        for group in opt.param_groups:
            group["lr"] = lr_at(it)
        for _ in range(args.accumulate):
            batch = make_batch(train, tok, base, rng, args.batch, args.ctx)
            loss = run(batch)
            (loss / args.accumulate).backward()
            seen += int(batch[1].sum())         # real (non-padding) tokens
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        running.append(loss.item())

        if it % 20 == 0:
            print(f"step {it:6d}  loss {np.mean(running[-20:]):.3f}  "
                  f"{seen / (time.time() - t0):,.0f} tokens/s", flush=True)
        if it % args.eval_every == 0 or it == args.steps:
            model.eval()
            with torch.no_grad():
                v = float(np.mean([run(b).item() for b in val]))
            model.train()
            log.write(f"{it},{np.mean(running):.4f},{v:.4f}\n"); log.flush()
            flag = ""
            if v < best:
                best, flag = v, "  (best, saved)"
                model.save_pretrained(os.path.join(args.out, "best"))
                tok.save_pretrained(os.path.join(args.out, "best"))
            model.save_pretrained(last)
            tok.save_pretrained(last)
            torch.save({"optimizer": opt.state_dict(), "step": it, "best": best},
                       os.path.join(last, "state.pt"))
            print(f"== step {it}: train {np.mean(running):.3f}  val {v:.3f}{flag}", flush=True)
            running, t0, seen = [], time.time(), 0


if __name__ == "__main__":
    main()
