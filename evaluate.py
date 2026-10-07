"""Measure a trained model: loss on held-out music, and whether it obeys tags.

    python evaluate.py --model runs/v1/best --data data/prepared
"""
# argparse: command-line options; os: building file paths
import argparse, os
# numpy: per-token loss bookkeeping and selecting tokens by family
import numpy as np
# mlx.core: Apple's array library; turns numpy batches into MLX arrays for the model
import mlx.core as mx
# mlx.nn: provides the cross-entropy loss function
import mlx.nn as nn
# tokenizer: token id ranges and the list of possible values for each tag
import tokenizer as T
# Dataset / fixed_batches: load a prepared split and draw reproducible evaluation batches
from data import Dataset, fixed_batches
# Pianist: the model class, used here to load a saved checkpoint
from model import Pianist
# MLXBackend / Performer: generate notes from the model given a tag request
from performer import MLXBackend, Performer
# compute_tags: measure the tags of the generated notes, to compare with what was asked
from tags import compute_tags

# token id range [lo, hi) of each family of music tokens; the vocabulary is laid out
# Pitch, Vel, Dur, Shift in that order, so each family ends where the next begins
FAMILIES = {"pitch": (T.PITCH0, T.VEL0), "velocity": (T.VEL0, T.DUR0),
            "duration": (T.DUR0, T.SHIFT0), "shift": (T.SHIFT0, T.VOCAB_SIZE)}


def held_out_loss(model, batches):
    """Loss per music token, overall and split by token family."""
    # running sum of loss for each family
    total = {k: 0.0 for k in FAMILIES}
    # running count of target tokens for each family
    count = {k: 0 for k in FAMILIES}
    # each batch is (inputs, targets, mask) numpy arrays; the mask is not needed here because
    # the family ranges below already exclude padding, tags and other special tokens
    for x, y, mask in batches:
        # run the model on the inputs; logits are scores over the vocabulary at every position
        logits, _ = model(mx.array(x))
        # cross-entropy (how surprised the model was by the true next token) at every position,
        # not averaged ("none"), then converted back to numpy for indexing
        ce = np.array(nn.losses.cross_entropy(logits, mx.array(y), reduction="none"))
        # split the losses by which family the target token belongs to
        for k, (lo, hi) in FAMILIES.items():
            # boolean mask of positions whose target token id falls in this family's range
            sel = (y >= lo) & (y < hi)
            # add up the loss at those positions
            total[k] += ce[sel].sum()
            # and count how many there were
            count[k] += sel.sum()
    # average loss per token within each family
    out = {k: total[k] / count[k] for k in FAMILIES}
    # overall average across all music tokens (weighted by how many tokens each family has)
    out["all"] = sum(total.values()) / sum(count.values())
    # dict like {"pitch": 2.1, "velocity": 1.8, ..., "all": 1.9}
    return out


def tag_adherence(model, category, n_samples=8, n_notes=120, seed=0):
    """Ask for each value of one tag, measure the tag of what comes out.

    Returns (exact match rate, rate within one step for ordered tags)."""
    # all possible values of this tag, e.g. ["pp", "p", "mf", "f", "ff"] for dynamics
    values = T.TAG_VALUES[category]
    # exact = hits, near = hits within one step, n = number of samples actually measured
    exact = near = n = 0
    # request each value in turn (vi is its position in the ordered list)
    for vi, value in enumerate(values):
        # several generations per value, to average out randomness
        for s in range(n_samples):
            # a performer asked for only this one tag, with a distinct reproducible seed per sample
            p = Performer(MLXBackend(model), {category: value}, seed=seed + 1000 * vi + s)
            # generate n_notes notes as an array of (onset_ms, pitch, velocity, dur_ms)
            notes = p.notes(n_notes)
            # skip generations that ended (<eos>) too early for the tags to be meaningful
            if len(notes) < 16:                     # too short to measure
                continue
            # measure the requested tag on what was actually played
            got = compute_tags(notes)[category]
            # count an exact match (True adds 1, False adds 0)
            exact += got == value
            # count a match within one step of the request (e.g. asked "p", got "mf");
            # only meaningful for ordered tags, so the caller ignores it for keys
            near += abs(values.index(got) - vi) <= 1
            # one more measured sample
            n += 1
    # fractions of measured samples that matched exactly / approximately
    return exact / n, near / n


def main():
    """Print held-out loss per token family, then tag adherence."""
    # command-line options
    ap = argparse.ArgumentParser()
    # checkpoint path prefix (without .json / .safetensors) of the model to evaluate
    ap.add_argument("--model", default="runs/v1/best")
    # folder holding the prepared .npz splits
    ap.add_argument("--data", default="data/prepared")
    # which split to compute the loss on (e.g. validation or test)
    ap.add_argument("--split", default="validation")
    # how many batches to average the loss over
    ap.add_argument("--batches", type=int, default=20)
    # passages per batch
    ap.add_argument("--batch", type=int, default=16)
    # how many generations to make for each tag value
    ap.add_argument("--samples", type=int, default=8, help="generations per tag value")
    # also test the key tag (24 values, so much slower)
    ap.add_argument("--keys", action="store_true", help="also test all 24 keys (slow)")
    # read the options from the command line
    args = ap.parse_args()

    # load the weights and the saved info (config, training step, best loss); comes back in eval mode
    model, info = Pianist.load(args.model)
    # report which training step this checkpoint came from
    print(f"model from step {info.get('step')}")
    # load the chosen data split
    ds = Dataset(os.path.join(args.data, f"{args.split}.npz"))
    # compute per-family loss on fixed, un-augmented batches sized to the model's context length
    losses = held_out_loss(model, fixed_batches(ds, args.batches, args.batch, model.config.ctx))
    # header for the loss table
    print(f"\n{args.split} loss per token (lower is better; perplexity = e^loss)")
    # one row per family plus "all"
    for k, v in losses.items():
        # perplexity e^loss is roughly "how many tokens the model was choosing between"
        print(f"  {k:<9} {v:.3f}   perplexity {np.exp(v):6.1f}")

    # header for the tag adherence table
    print("\ntag adherence (generate with one tag, measure the result)")
    # column titles, padded to line up with the rows below
    print(f"  {'tag':<9} {'exact':>6} {'within 1':>9} {'chance':>7}")
    # test the measurable tags; key only when --keys is given since it has 24 values
    for cat in ["density", "dynamics", "register"] + (["key"] if args.keys else []):
        # generate with each value of this tag and measure how often the result matches
        exact, near = tag_adherence(model, cat, args.samples)
        # the exact-match rate a model ignoring the tag would get by guessing uniformly
        chance = 1 / len(T.TAG_VALUES[cat])
        # "within 1" means nothing for keys (they are not ordered), so print a dash instead
        near_s = f"{near:9.0%}" if cat != "key" else f"{'-':>9}"
        # one row of the table, rates shown as percentages
        print(f"  {cat:<9} {exact:6.0%} {near_s} {chance:7.0%}")


# run main() only when executed as a script, not when imported
if __name__ == "__main__":
    # start the evaluation
    main()
