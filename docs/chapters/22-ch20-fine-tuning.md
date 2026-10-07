# 20. Fine-tuning

::: filename
qwen/train_qwen.py
:::

``` python
"""Fine-tune a small Qwen to continue a caption with piano music.

    python qwen/train_qwen.py --data data/prepared --out runs/qwen
"""
import argparse, math, os, time
import numpy as np
import torch
from music_lm import load, make_batch, music_loss
from data import Dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--data", default="data/prepared")
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
    resume = args.resume and os.path.exists(os.path.join(last, "state.pt"))

    # The weights are kept in 32-bit floats for training. Qwen ships in
    # bfloat16, which is too coarse to absorb small optimiser updates.
    model, tok, base = load(last if resume else args.base, device, torch.float32)
    train = Dataset(os.path.join(args.data, "train.npz"))
    val_ds = Dataset(os.path.join(args.data, "validation.npz"))
    val_rng = np.random.default_rng(1234)
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
            seen += int(batch[1].sum())
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
```

The structure mirrors `train.py` from Part I. The differences are these.

**Every parameter is trained.** Lighter methods such as LoRA train a small add-on and leave the original weights frozen. They work well when the new task resembles the old one. Producing note events is unlike anything in Qwen's pre-training, and the new token rows start from nothing, so the whole model needs to adapt. With 128 GB of memory, full fine-tuning of a model this size is comfortable.

**A gentler learning rate.** The peak is 0.0001, a third of what the small model used. Qwen's body is already well trained and large steps would damage what it knows.

**No weight decay on the embeddings.** Weight decay shrinks every weight a little on every step. The embedding table has about 152,000 rows and almost none of them are ever used in our training. Decay would steadily erode those unused text rows and with them Qwen's vocabulary.

**Number formats.** Qwen's published weights are stored as 16-bit numbers (`bfloat16`). That format is fine for running a model and too coarse for training one: many of the optimiser's small updates would be rounded away to nothing. `load` is therefore told to convert the weights to 32-bit, and the script prints the format at start-up so you can confirm it says `torch.float32`. The computation inside `torch.autocast` still uses 16-bit arithmetic where that is safe, which saves memory and time. For playing, in the next chapter, the model is loaded in 16-bit.

**Resuming.** Every evaluation saves the model, the tokeniser and the optimiser's state to `last`. After an interruption, repeat the command with `--resume` added.

## 20.1 Run it

Do a two-minute check first:

``` bash
python qwen/train_qwen.py --data data/prepared --out runs/qwen_test \
    --steps 60 --batch 4 --warmup 10 --eval-every 20 --eval-batches 2
```

The first line must say `torch.float32`. The first loss printed should be around 6: the natural logarithm of 401 is 5.99, and the new tokens start out nearly equally likely. It should be clearly lower by step 60. On the stand-in model used to test this code the loss began at 5.99.

Then start the real run:

``` bash
nohup python qwen/train_qwen.py --data data/prepared --out runs/qwen > qwen.log 2>&1 &
tail -f qwen.log
```

`nohup` keeps the run going if your connection to the Spark drops.

## 20.2 Batch size, memory and time

Start with the default batch of 16. Watch memory with `nvidia-smi` or `free -h` during the first few hundred steps. If there is plenty of room, 32 gives a steadier gradient. If the machine is under pressure, drop to 8 and add `--accumulate 2`, which adds up the gradients of two small batches before each update and so behaves like a batch of 16.

As in Part I, read the `tokens/s` figure after the first hundred steps and work out the total:

> hours = steps × batch × accumulate × 1,024 ÷ (tokens per second) ÷ 3,600

::: {.admonition .warning}
No measured figure

This was not run on a DGX Spark. Expect each step to cost several tens of times more computation than the small model's, since the model is about thirty times larger. Plan for the full run to take a day or more, and decide after the first evaluation whether to continue, shorten the run, or rent a faster GPU for this one job.
:::

## 20.3 What to watch

The validation loss measures the same thing as the small model's, loss per music token on MAESTRO's validation split, so the small model's best figure is the number to beat. The two scripts draw different random passages, so treat a gap of less than about 0.05 as a tie.

Expect the early part of the curve to look different from Part I. The small model started from random weights everywhere. Qwen starts with a capable body and 402 token rows that all look alike, so the first stretch of training is mostly the embeddings finding their places.

Overfitting arrives sooner with a larger model on the same data. If the validation loss turns upwards early, the two effective responses are more data (the deduplicated Aria-MIDI subset is the obvious source) and a lower learning rate.
