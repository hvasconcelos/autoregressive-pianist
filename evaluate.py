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


def tag_adherence(model, category, n_samples=8, n_notes=120, seed=0):
    """Ask for each value of one tag, measure the tag of what comes out.

    Returns (exact match rate, rate within one step for ordered tags)."""
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
            near += abs(values.index(got) - vi) <= 1
            n += 1
    return exact / n, near / n


def main():
    """Print held-out loss per token family, then tag adherence."""
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
