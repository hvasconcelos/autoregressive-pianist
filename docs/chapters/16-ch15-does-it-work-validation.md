# 15. Does it work? Validation

Part II is only worth starting if the small model works. This chapter defines "works" as four tests with pass marks.

## 15.1 The evaluation script

::: filename
evaluate.py
:::

``` python
"""Measure a trained model: loss on held-out music, and whether it obeys tags.

    python evaluate.py --model runs/v1/best --data data/prepared
"""
import argparse, os
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import tokenizer as T
from data import Dataset, fixed_batches
from model import Pianist
from performer import MLXBackend, Performer
from tags import compute_tags

FAMILIES = {"pitch": (T.PITCH0, T.VEL0), "velocity": (T.VEL0, T.DUR0),
            "duration": (T.DUR0, T.SHIFT0), "shift": (T.SHIFT0, T.VOCAB_SIZE)}


def held_out_loss(model, batches):
    """Loss per music token, overall and split by token family."""
    total = {k: 0.0 for k in FAMILIES}
    count = {k: 0 for k in FAMILIES}
    for x, y, mask in batches:
        logits, _ = model(mx.array(x))
        ce = np.array(nn.losses.cross_entropy(logits, mx.array(y), reduction="none"))
        for k, (lo, hi) in FAMILIES.items():
            sel = (y >= lo) & (y < hi)
            total[k] += ce[sel].sum()
            count[k] += sel.sum()
    out = {k: total[k] / count[k] for k in FAMILIES}
    out["all"] = sum(total.values()) / sum(count.values())
    return out


def tag_adherence(model, category, n_samples=8, n_notes=120, seed=0):
    """Ask for each value of one tag, measure the tag of what comes out.

    Returns (exact match rate, rate within one step for ordered tags)."""
    values = T.TAG_VALUES[category]
    exact = near = n = 0
    for vi, value in enumerate(values):
        for s in range(n_samples):
            p = Performer(MLXBackend(model), {category: value}, seed=seed + 1000 * vi + s)
            notes = p.notes(n_notes)
            if len(notes) < 16:
                continue
            got = compute_tags(notes)[category]
            exact += got == value
            near += abs(values.index(got) - vi) <= 1
            n += 1
    return exact / n, near / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/v1/best")
    ap.add_argument("--data", default="data/prepared")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--samples", type=int, default=8, help="generations per tag value")
    ap.add_argument("--keys", action="store_true", help="also test all 24 keys (slow)")
    args = ap.parse_args()

    model, info = Pianist.load(args.model)
    print(f"model from step {info.get('step')}")
    ds = Dataset(os.path.join(args.data, f"{args.split}.npz"))
    losses = held_out_loss(model, fixed_batches(ds, args.batches, args.batch, model.config.ctx))
    print(f"\n{args.split} loss per token (lower is better; perplexity = e^loss)")
    for k, v in losses.items():
        print(f"  {k:<9} {v:.3f}   perplexity {np.exp(v):6.1f}")

    print("\ntag adherence (generate with one tag, measure the result)")
    print(f"  {'tag':<9} {'exact':>6} {'within 1':>9} {'chance':>7}")
    for cat in ["density", "dynamics", "register"] + (["key"] if args.keys else []):
        exact, near = tag_adherence(model, cat, args.samples)
        chance = 1 / len(T.TAG_VALUES[cat])
        near_s = f"{near:9.0%}" if cat != "key" else f"{'-':>9}"
        print(f"  {cat:<9} {exact:6.0%} {near_s} {chance:7.0%}")


if __name__ == "__main__":
    main()
```

``` bash
python evaluate.py --model runs/v1/best --data data/prepared
```

It prints two tables.

## 15.2 Test 1: loss on unseen music

The first table is the loss on the validation split, overall and for each token family.

The split by family is a useful diagnostic, because the four families are not equally hard.

| Family | What a low loss means | Typical difficulty |
|----|----|----|
| Pitch | The model knows which note comes next | Hardest. This is where harmony and melody live. |
| Velocity | It predicts how hard the note is struck | Moderate. Neighbouring bins are nearly interchangeable. |
| Duration | It predicts how long the note sounds | Moderate, and affected by the pedal. |
| Shift | It predicts the wait to the next note | Usually the easiest, because rhythm is regular. |

**Pass mark:** the overall validation loss is far below 6, and the pitch perplexity is well under 88, the number of keys. A pitch perplexity of 88 would mean the model has learned nothing about which notes follow which.

Run the same command with `--split test` once, at the very end, for a figure on data that played no part in any decision.

## 15.3 Test 2: tag adherence

The second table answers the question that matters most for a request-driven system: does the model do what it is asked?

For each value of a tag, the script generates several short performances with only that one tag set (the hardest case, since the model has the least to go on), measures the tag of the result with the same code that labelled the training data, and compares.

- **exact** is how often the measured value equals the requested one.
- **within 1** allows a miss by one step, for example `p` when `pp` was asked for.
- **chance** is what random playing would score.

**Pass mark:** "within 1" above 80% for density and dynamics, and "exact" well above chance for register. Add `--keys` to test all 24 keys as well; expect key adherence to be lower than the others, since the key estimate is itself imperfect and related keys are easily confused.

If adherence is at chance while the loss is good, the model has learned music but is ignoring the tags. The usual causes are a bug that drops the tags from the training sequences, or tag dropout set far too high.

## 15.4 Test 3: real-time headroom

Run `sample.py` with a dense request and read the last line.

``` bash
python sample.py --model runs/v1/best --tags density=very_dense --notes 600
```

**Pass mark:** generation is at least three times faster than the music needs. Then run `play.py` for five minutes and confirm that `underruns` is 0 or 1.

## 15.5 Test 4: listening

Numbers cannot tell you whether the result is musical. Generate ten files across a range of requests and listen to each for a minute, using the table in chapter 12.

**Pass mark:** most of them sound like plausible piano playing for at least 20 to 30 seconds at a stretch, and you can tell the requests apart with your eyes closed.

## 15.6 The decision

| Result | What to do |
|----|----|
| All four pass | The approach works. Go on to Part II. |
| Loss is poor | Train longer if the validation loss was still falling. If it had turned, the model is overfitting: add data (Aria-MIDI) or raise dropout to 0.2. |
| Loss is good, tags are ignored | Fix the conditioning before anything else. Part II inherits the same data pipeline, and Qwen will not rescue broken tags. |
| Too slow for real time | Unlikely for this model. Check that nothing else is using the GPU. |
| Musical for 5 seconds, then nonsense | Under-trained, or sampling too hot. Try `--temperature 0.9 --top-p 0.9`. |

## 15.7 A checklist before moving on

- [ ] The toy run of chapter 10 passes.
- [ ] `stats.py` shows no nearly empty tag value.
- [ ] The validation loss fell, turned, and `best` was saved before the turn.
- [ ] Tag adherence passes for density, dynamics and register.
- [ ] Five minutes of real-time playing with no underruns.
- [ ] You have listened, and it sounds like a piano being played.

Keep `runs/v1/best` and the output of `evaluate.py`. They are the baseline that Qwen has to beat.
