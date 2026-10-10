# 11. Training on MAESTRO and Aria-MIDI

## 11.1 The training loop

::: filename
train.py
:::

``` python
"""Train the model.

    python train.py --data data/maestro_prepared --out runs/v1
    python train.py --data data/maestro_prepared data/aria_prepared --out runs/v2

With several --data folders, training draws from all of their train.npz
files; validation uses the first folder only, so losses stay comparable.
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
    ap.add_argument("--data", nargs="+", default=["data/maestro_prepared"])
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

    train = Dataset([os.path.join(d, "train.npz") for d in args.data])
    val = fixed_batches(Dataset(os.path.join(args.data[0], "validation.npz")),
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
```

The loop does the same four things at every step: build a batch, compute the loss and its gradients, clip the gradients, and update the parameters. Every 500 steps it measures the loss on the validation batches and saves the model.

`--data` takes one or more prepared folders. Training draws from all of their `train.npz` files as one dataset (chapter 8). Validation uses only the first folder's `validation.npz`. We put MAESTRO first, so the validation loss always measures classical playing on the cleanest data, and it stays comparable with a model trained on MAESTRO alone. Aria's own validation and test sets are used in chapter 15.

The settings and the reasons for them:

| Setting | Value | Why |
|----|----|----|
| Batch size | 32 sequences by default; 8 on a 16 GB Mac | Large enough for a steady gradient. See below if memory is tight. |
| Context | 1,024 tokens | About 250 notes, or 25 seconds of typical playing. Long enough to hold a phrase. |
| Optimiser | AdamW, betas 0.9 and 0.95 | The standard choice for transformers. |
| Peak learning rate | 0.0003 | A safe value for a model of this size. |
| Warm-up | 1,000 steps | The learning rate rises from zero. Large early steps on a random model cause instability. |
| Decay | Cosine, down to 10% of peak | Smaller steps late in training let the model settle. |
| Weight decay | 0.01 | A light pull of the weights towards zero, which discourages memorisation. |
| Gradient clipping | Norm 1.0 | Caps the size of any single update, so one unusual batch cannot derail training. |
| Dropout | 0.1 | See chapter 9. |

Two files are written on every evaluation. `last` is the most recent state, used for resuming. `best` is the state with the lowest validation loss so far, and it is the one you use afterwards.

The call to `mx.compile` turns the whole step into a single optimised graph. The `state` list tells MLX which things the step changes as a side effect: the model's parameters, the optimiser's internal averages and the random number generator used by dropout.

## 11.2 Start training

``` bash
python train.py --data data/maestro_prepared data/aria_prepared --out runs/v2 \
    --batch 8 --steps 60000
```

The first lines on an M1 Pro with 16 GB:

``` text
19.1 M parameters, 38645 training pieces
step     50  loss 6.022  13,669 tokens/s
```

The 38,645 pieces are MAESTRO's 962 and Aria's 37,683. The first 50 steps are a little slow because `mx.compile` builds the step once; the speed then settles at about 14,500 tokens per second.

The run names `v1` and `v2` are used through the rest of the book. `v1` is the same model trained on MAESTRO alone (`--data data/maestro_prepared --out runs/v1`); its results in chapter 15 are the baseline that shows what the second dataset adds.

## 11.3 Measure your speed first

The `tokens/s` figure tells you how long the run will take. Each step processes batch × 1,023 tokens, so:

> hours for the full run = steps × batch × 1,023 ÷ (tokens per second) ÷ 3,600

Measured on an M1 Pro with 16 GB at batch 8:

| Run | Steps | Tokens per second | Time |
|----|---:|---:|---:|
| `v1`, MAESTRO alone | 25,500 | 14,400 | 4.9 hours |
| `v2`, MAESTRO and Aria | 60,000 | 14,500 | about 9.5 hours |

The speed is the same for both, because it depends on the model and the batch, not on how much data there is. Read your own number after the first hundred steps. If the full run would take longer than you want, see "If training is too slow" below.

Keep the Mac plugged in and stop it from sleeping while training runs. Running the command through `caffeinate` does that:

``` bash
caffeinate -i python train.py --data data/maestro_prepared data/aria_prepared --out runs/v2 \
    --batch 8 --steps 60000
```

If training is interrupted, continue from the last save with:

``` bash
python train.py --data data/maestro_prepared data/aria_prepared --out runs/v2 \
    --batch 8 --steps 60000 --resume
```

Use the same `--batch`, `--steps` and learning-rate settings as the original run: resuming takes them from the command line, and changing them changes the learning-rate schedule part-way through. Resuming restores the model and the position in the learning-rate schedule. It does not restore the optimiser's running averages, which rebuild themselves within a few hundred steps. You may see the loss tick up briefly after a resume for that reason.

## 11.4 Reading the loss

Two numbers matter: the training loss, printed every 50 steps, and the validation loss, printed every 500.

**The start.** Both begin between 6 and 7. Blind guessing over 465 tokens gives 6.1, and a freshly initialised model is a little worse than that because its random scores are not perfectly even. Within the first few hundred steps the loss drops quickly as the model learns the grammar: that a Vel follows a Pitch, and so on. That alone removes most of the uncertainty about which *kind* of token comes next.

**The long middle.** The loss then falls slowly for thousands of steps while the model learns harmony, voicing and rhythm.

**The turn.** At some point the validation loss stops falling and begins to rise, while the training loss keeps going down. From there on the model is memorising the training pieces instead of learning things that carry over to new music. This is *overfitting*. The `best` checkpoint is the model at the lowest point, and that is why the script keeps it separately.

With about 227 million tokens of data and 8,184 tokens per step, one "pass" through both datasets is roughly 28,000 steps, so 60,000 steps is a little over two passes. With that much data the turn may not come at all. On MAESTRO alone, 22 million tokens, a pass is only about 2,700 steps, and the turn comes much sooner. Either way, you can stop the run with Ctrl-C once the validation loss has risen at three evaluations in a row; nothing later will beat `best`.

Two things look different from a MAESTRO-only run. First, the validation loss trails `v1`'s at the same step for most of the run: validation is on MAESTRO, and `v2` sees far less MAESTRO per step while it also learns five more styles. It catches up as the learning rate decays. Second, the training loss sits further below the validation loss. That is not overfitting. The training loss is mostly Aria and the validation loss is MAESTRO, so they measure different music.

::: {.admonition .note}
What final loss to expect

The MAESTRO-only run `v1` reached a validation loss of 2.27 at step 25,500 and was still falling slowly (about 0.01 per 1,000 steps). Its measured results are in chapter 15. As a guide to judge your own result: a validation loss still above 3 after several thousand steps means something is wrong; a loss around 2.2 to 2.3 is what this model reaches on MAESTRO. The tests in chapter 15 are a better judge than any single loss value.
:::

Every evaluation also appends a line to `runs/v2/log.csv` with the step, both losses and the speed, which you can open in any spreadsheet to draw the curves.

## 11.5 If training is too slow, or memory runs out

**Out of memory, or the Mac becomes sluggish.** The attention step is the hungry part: at batch 32 each layer builds tables of 32 × 8 × 1,023 × 1,023 numbers, about 1 GB, and they are kept until the gradients have been computed. On an M1 Pro with 16 GB, the default batch of 32 swapped so heavily that it never printed its first progress line; batch 8 needs about 4 GB and runs at full speed. With 32 GB or more the default should fit. Both runs in this book kept the learning rate at 0.0003 with batch 8 without trouble. If a smaller batch makes the loss unstable, lower it to `--lr 2e-4`. Activity Monitor's memory pressure graph should stay green.

**Too slow.** There are three ways to trade quality for time.

| Option | Command | Effect |
|----|----|----|
| A shorter run | `--steps 6000` | A third of the time. Enough to hear clearly musical output, and to run the tests in chapter 15. |
| A smaller model | `--dim 384 --heads 6` | About 10.8 million parameters and a little over half the computation per step. |
| A shorter context | `--ctx 512` | Half the tokens per step. The model remembers about 12 seconds instead of 25. |

A short first run is a good idea in any case. Train for 3,000 steps, generate a file, listen. If it sounds like a piano being played by something that has an idea of harmony, the pipeline is right and the long run is worth starting.

**Rent a GPU, or use the DGX Spark?** This model does not need either. If you want to run the Part I code on an NVIDIA machine anyway, MLX publishes a CUDA build for Linux (`uv pip install "mlx[cuda]"`), which was not tested for this book.
