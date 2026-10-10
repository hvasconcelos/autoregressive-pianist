# 15. Does it work? Validation

Part II is only worth starting if the small model works. This chapter defines "works" as four tests with pass marks, and applies them to the model trained on both datasets (`runs/v2`). Where this book has measured figures, they are from `runs/v1`, the same model trained on MAESTRO alone, which is the baseline `v2` has to match on classical music.

## 15.1 The evaluation script

::: filename
evaluate.py
:::

``` python
"""Measure a trained model: loss on held-out music, and whether it obeys tags.

    python evaluate.py --model runs/v1/best --data data/maestro_prepared
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

# token id range [lo, hi) of each family of music tokens
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


def close_keys(key):
    """The key itself, its relative major/minor and the keys a fifth up and down.

    These share all but one note of their scale, so the key estimate often
    confuses them."""
    mode, tonic = divmod(T.KEYS.index(key), 12)    # KEYS: 12 majors, then 12 minors
    relative = (1 - mode) * 12 + (tonic + (9 if mode == 0 else 3)) % 12
    return {key, T.KEYS[relative],
            T.KEYS[mode * 12 + (tonic + 7) % 12], T.KEYS[mode * 12 + (tonic + 5) % 12]}


def tag_adherence(model, category, n_samples=8, n_notes=120, seed=0):
    """Ask for each value of one tag, measure the tag of what comes out.

    Returns (exact match rate, near rate): near is within one step for
    ordered tags, and a close key (see close_keys) for the key tag."""
    values = T.TAG_VALUES[category]
    exact = near = n = 0
    for vi, value in enumerate(values):
        for s in range(n_samples):
            p = Performer(MLXBackend(model), {category: value}, seed=seed + 1000 * vi + s)
            notes = p.notes(n_notes)
            if len(notes) < 16:                     # too short to measure
                continue
            got = compute_tags(notes)[category]
            exact += got == value
            if category == "key":
                near += got in close_keys(value)
            else:
                near += abs(values.index(got) - vi) <= 1
            n += 1
    return exact / n, near / n


def main():
    """Print held-out loss per token family, then tag adherence."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/v1/best")
    ap.add_argument("--data", default="data/maestro_prepared")
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
    print(f"  {'tag':<9} {'exact':>6} {'near':>6} {'chance':>7}")
    for cat in ["density", "dynamics", "register"] + (["key"] if args.keys else []):
        exact, near = tag_adherence(model, cat, args.samples)
        chance = 1 / len(T.TAG_VALUES[cat])
        print(f"  {cat:<9} {exact:6.0%} {near:6.0%} {chance:7.0%}")
    print("  near: within one step; for key, the same, relative or a fifth-related key"
          + (f" (chance {4 / len(T.KEYS):.0%})" if args.keys else ""))


if __name__ == "__main__":
    main()
```

``` bash
python evaluate.py --model runs/v2/best --data data/maestro_prepared
python evaluate.py --model runs/v2/best --data data/aria_prepared
```

It prints two tables. With two datasets, run it on each. The loss table changes with the data. The tag adherence table doesn't, because it generates fresh music and never reads the dataset; it is the same in both runs.

## 15.2 Test 1: loss on unseen music

The first table is the loss on the validation split, overall and for each token family.

The split by family is a useful diagnostic, because the four families are not equally hard.

| Family | What a low loss means | `v1` on MAESTRO's test split |
|----|----|----|
| Pitch | The model knows which note comes next | 1.78, perplexity 5.9: the easiest, once harmony is learned |
| Velocity | It predicts how hard the note is struck | 2.01, perplexity 7.5. Neighbouring bins are nearly interchangeable. |
| Duration | It predicts how long the note sounds | 2.62, perplexity 13.8: the hardest, because the pedal decides it |
| Shift | It predicts the wait to the next note | 2.14, perplexity 8.5. Rubato and spread chords are hard to predict. |
| **All** | | **2.14, perplexity 8.5** (validation split: 2.25) |

Some uncertainty never goes away. A pianist's touch and timing vary from one performance to the next, so velocity and timing can't be predicted exactly however well the model is trained.

**Pass mark:** the overall validation loss is far below 6, and the pitch perplexity is well under 88, the number of keys. A pitch perplexity of 88 would mean the model has learned nothing about which notes follow which.

Run the same commands with `--split test` once, at the very end, for figures on data that played no part in any decision:

``` bash
python evaluate.py --model runs/v2/best --data data/maestro_prepared --split test --keys
python evaluate.py --model runs/v2/best --data data/aria_prepared --split test
```

The first answers whether the second dataset helped, or at least didn't hurt, on classical music: compare it with `v1`'s 2.14. A difference under 0.05 is noise. The second measures the new styles, on Aria recordings the model never trained on. `v1` can't be compared fairly there, because it never saw those styles.

## 15.3 Test 2: tag adherence

The second table answers the question that matters most for a request-driven system: does the model do what it is asked?

For each value of a tag, the script generates several short performances with only that one tag set (the hardest case, since the model has the least to go on), measures the tag of the result with the same code that labelled the training data, and compares.

- **exact** is how often the measured value equals the requested one.
- **near** allows a miss by one step, for example `p` when `pp` was asked for. For keys, which have no order, it counts the requested key, its relative major or minor and the keys a fifth up and down. For C major those are C major, A minor, G major and F major. They share all but one note of their scale, so the key estimate often confuses them.
- **chance** is what random playing would score. For key, the near chance is 4 keys in 24, or 17%.

**Pass mark:** "near" above 80% for density and dynamics, and "exact" well above chance for register. Add `--keys` to test all 24 keys as well; expect key adherence to be lower than the others, since the key estimate is itself imperfect.

`v1` measured:

``` text
  tag        exact   near  chance
  density      48%    82%     20%
  dynamics     68%    95%     20%
  register     62%    96%     33%
  key          23%    51%      4%
```

All three pass marks hold, density only just. For key, 23% of generations were in exactly the requested key, 28% in a neighbouring key and 49% in an unrelated one. Each rate comes from 40 to 192 generations, so allow about ±10%; use `--samples 32` when comparing two models.

Genre, era and composer are not in the table, because they can't be measured from the notes. Genre is judged by ear in test 4.

If adherence is at chance while the loss is good, the model has learned music but is ignoring the tags. The usual causes are a bug that drops the tags from the training sequences, or tag dropout set far too high.

## 15.4 Test 3: real-time headroom

Run `sample.py` with a dense request and read the last line.

``` bash
python sample.py --model runs/v2/best --tags density=very_dense --notes 600
```

**Pass mark:** generation is at least three times faster than the music needs. `v1` generated 368 tokens per second against 71 needed for 18 notes per second, 5.2 times faster. `v2` is the same size, so it runs at the same speed. Then run `play.py` for five minutes and confirm that `underruns` is 0 or 1.

## 15.5 Test 4: listening

Numbers cannot tell you whether the result is musical. Generate ten files across a range of requests and listen to each for a minute, using the table in chapter 12.

Include every genre. The same request in each style is the clearest test of the genre tag:

``` bash
for g in classical jazz pop film ragtime; do
  python sample.py --model runs/v2/best --tags genre=$g,density=medium --out $g.mid
done
```

**Pass mark:** most of them sound like plausible piano playing for at least 20 to 30 seconds at a stretch, you can tell the requests apart with your eyes closed, and you can tell the genres apart.

## 15.6 The decision

| Result | What to do |
|----|----|
| All four pass | The approach works. Go on to Part II. |
| Loss is poor | Train longer if the validation loss was still falling. If it had turned, the model is overfitting: add more Aria-MIDI (raise `--notes` in chapter 7) or raise dropout to 0.2. |
| Classical loss is clearly worse than `v1` | The Aria share is drowning MAESTRO out. Lower `--notes`, or train longer so the learning rate has time to decay. |
| Loss is good, tags are ignored | Fix the conditioning before anything else. Part II inherits the same data pipeline, and Qwen will not rescue broken tags. |
| Too slow for real time | Unlikely for this model. Check that nothing else is using the GPU. |
| Musical for 5 seconds, then nonsense | Under-trained, or sampling too hot. Try `--temperature 0.9 --top-p 0.9`. |

## 15.7 A checklist before moving on

- [ ] The toy run of chapter 10 passes.
- [ ] Both datasets are prepared, and `stats.py` shows no nearly empty tag value in either.
- [ ] The validation loss fell and `best` was saved at its lowest point.
- [ ] Tag adherence passes for density, dynamics and register.
- [ ] Five minutes of real-time playing with no underruns.
- [ ] You have listened, it sounds like a piano being played, and the genres sound different.

Keep `runs/v2/best` and the output of `evaluate.py` on both test splits. They are the baseline that Qwen has to beat.
