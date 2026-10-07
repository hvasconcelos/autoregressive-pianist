"""Show how the tags are distributed in a prepared split.

    uv run python stats.py --data data/prepared

Use it to check that every tag value has a healthy share of the data
before you train. If one value is almost empty, move the thresholds
in tags.py.
"""
# argparse: command-line options; collections: Counter for tallying tag values; os: paths
import argparse, collections, os
# numpy: random generator and means
import numpy as np
# tokenizer: vocabulary, tag order and token helpers (encode_tags, kind, VOCAB)
import tokenizer as T
# Dataset: loads a prepared .npz split and samples training passages from it
from data import Dataset


def main():
    """Sample passages the way training does (no augmentation) and count tags."""
    # command-line options:
    ap = argparse.ArgumentParser()
    # --data: folder with the prepared .npz files
    ap.add_argument("--data", default="data/prepared")
    # --split: which split to inspect (train / validation / test)
    ap.add_argument("--split", default="train")
    # --passages: how many random passages to sample
    ap.add_argument("--passages", type=int, default=2000)
    args = ap.parse_args()
    # load the chosen split
    ds = Dataset(os.path.join(args.data, f"{args.split}.npz"))
    # fixed seed so repeated runs give the same numbers
    rng = np.random.default_rng(0)
    # one counter per tag category, e.g. counts["dynamics"]["p"] = passages tagged p
    counts = {k: collections.Counter() for k in T.TAG_ORDER}
    # lengths: tokens per sequence; per_note: music tokens per note, per passage
    lengths, per_note = [], []
    for _ in range(args.passages):
        # 1016 = a 1024-token context minus the 8-token tag prefix
        # (prefix = <bos> + up to 6 tags + <sep>); no augmentation and no tag dropout here.
        tags, music = ds.passage(rng, 1016, augment=False, tag_dropout=False)
        # the full sequence as the model sees it: tag prefix followed by the music tokens
        ids = T.encode_tags(tags) + music
        # only the music tokens (pitch/vel/dur/shift ids are all >= MUSIC0; drops tags/specials)
        music = [t for t in ids if t >= T.MUSIC0]
        lengths.append(len(ids))
        # each note has exactly one Pitch token, so music tokens / pitch tokens = tokens per note
        # (max(..., 1) avoids dividing by zero).
        per_note.append(len(music) / max(sum(T.kind(t) == "pitch" for t in music), 1))
        # count each tag token in the sequence
        for t in ids:
            if T.kind(t) == "tag":
                # token name like "<key:Cmaj>": strip the "<" and ">" and split on ":"
                k, v = T.VOCAB[t][1:-1].split(":")
                counts[k][v] += 1
    # Dataset size: number of pieces and total notes.
    print(f"{len(ds)} pieces, {len(ds.notes):,} notes")
    # average tokens per note, and from it the estimated total music tokens in the split
    print(f"tokens per note: {np.mean(per_note):.2f}   "
          f"-> about {len(ds.notes) * np.mean(per_note) / 1e6:.1f} M music tokens")
    # average and longest sequence length in tokens
    print(f"sequence length: mean {np.mean(lengths):.0f}, max {max(lengths)}")
    # for each category, show what share of passages got each value
    for k in T.TAG_ORDER:
        print(f"\n{k}")
        for v in T.TAG_VALUES[k]:
            # fraction of sampled passages with this value (shares can sum below 100 % when a
            # tag such as composer is missing for some pieces).
            share = counts[k][v] / args.passages
            # value name, percentage, and a text bar of up to 50 '#' characters
            print(f"  {v:<13} {share:6.1%}  {'#' * int(round(share * 50))}")


# run main() only when executed as a script
if __name__ == "__main__":
    main()
