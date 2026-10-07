"""Show how the tags are distributed in a prepared split.

    python stats.py --data data/prepared

Use it to check that every tag value has a healthy share of the data
before you train. If one value is almost empty, move the thresholds
in tags.py.
"""
import argparse, collections, os
import numpy as np
import tokenizer as T
from data import Dataset


def main():
    """Sample passages the way training does (no augmentation) and count tags."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/prepared")
    ap.add_argument("--split", default="train")
    ap.add_argument("--passages", type=int, default=2000)
    args = ap.parse_args()
    ds = Dataset(os.path.join(args.data, f"{args.split}.npz"))
    rng = np.random.default_rng(0)
    counts = {k: collections.Counter() for k in T.TAG_ORDER}
    lengths, per_note = [], []
    for _ in range(args.passages):
        # 1016 = a 1024-token context minus the 8-token tag prefix
        tags, music = ds.passage(rng, 1016, augment=False, tag_dropout=False)
        ids = T.encode_tags(tags) + music
        music = [t for t in ids if t >= T.MUSIC0]
        lengths.append(len(ids))
        per_note.append(len(music) / max(sum(T.kind(t) == "pitch" for t in music), 1))
        for t in ids:
            if T.kind(t) == "tag":
                k, v = T.VOCAB[t][1:-1].split(":")
                counts[k][v] += 1
    print(f"{len(ds)} pieces, {len(ds.notes):,} notes")
    print(f"tokens per note: {np.mean(per_note):.2f}   "
          f"-> about {len(ds.notes) * np.mean(per_note) / 1e6:.1f} M music tokens")
    print(f"sequence length: mean {np.mean(lengths):.0f}, max {max(lengths)}")
    for k in T.TAG_ORDER:
        print(f"\n{k}")
        for v in T.TAG_VALUES[k]:
            share = counts[k][v] / args.passages
            print(f"  {v:<13} {share:6.1%}  {'#' * int(round(share * 50))}")


if __name__ == "__main__":
    main()
