"""Turn a balanced, high-quality slice of Aria-MIDI into note arrays.

    uv run python prepare_aria.py --archive data/aria/aria-midi-v1-deduped-ext.tar.gz \
        --out data/aria_prepared

Reads the .tar.gz directly, without unpacking it. Files are picked by audio
score, best first, taking one from each genre in turn until --notes is
reached, so small genres like ragtime are not crowded out. Of the picked
recordings, one in --hold-out goes to validation and another one in
--hold-out to test, chosen by recording id so the split never changes.
Writes train.npz, validation.npz and test.npz in the same format as
prepare.py.
"""
import argparse, collections, io, json, os, tarfile
from multiprocessing import Pool
import numpy as np
from midi_io import load_notes
from tags import aria_tags

BYTES_PER_NOTE = 7.2        # measured on the deduped subset; only used to budget


def members(archive):
    """Stream the archive once, yielding (TarInfo, archive stream)."""
    with tarfile.open(archive, "r|gz") as tf:
        for m in tf:
            yield m, tf


def parse(item):
    """(name, MIDI bytes) -> (name, float32 note array or None)."""
    name, data = item
    try:
        notes = load_notes(io.BytesIO(data))
    except Exception:                       # a few transcriptions are malformed
        return name, None
    notes = notes[(notes[:, 1] >= 21) & (notes[:, 1] <= 108)]
    return name, notes.astype(np.float32) if len(notes) >= 64 else None


def select(meta, sizes, budget, hold_out):
    """Round-robin over genres, best audio score first, until the budget."""
    queues = collections.defaultdict(list)
    for name, size in sizes.items():
        file_id, seg = os.path.basename(name)[:-4].rsplit("_", 1)
        entry = meta.get(file_id, {})
        tags = aria_tags(entry.get("metadata", {}))
        if tags:
            score = entry.get("audio_scores", {}).get(seg, 0.0)
            queues[tags["genre"]].append((score, name, file_id, tags))
    for q in queues.values():
        q.sort()                                 # best score last, so pop() takes it
    picked, total = {}, 0.0
    while total < budget and any(queues.values()):
        for genre in sorted(queues):
            if queues[genre] and total < budget:
                score, name, file_id, tags = queues[genre].pop()
                split = {0: "validation", 1: "test"}.get(int(file_id) % hold_out, "train")
                picked[name] = {**tags, "file_id": file_id, "score": score, "split": split}
                total += sizes[name] / BYTES_PER_NOTE
    return picked


def main():
    """Pick files, parse them in parallel and write one .npz per split."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="data/aria/aria-midi-v1-deduped-ext.tar.gz")
    ap.add_argument("--out", default="data/aria_prepared")
    ap.add_argument("--notes", type=float, default=55e6, help="approximate note budget")
    ap.add_argument("--hold-out", type=int, default=100,
                    help="1 in N recordings is validation, another 1 in N is test")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # pass 1: metadata.json comes first in the archive, then the file sizes
    meta, sizes = None, {}
    for m, tf in members(args.archive):
        if m.name.endswith("metadata.json"):
            meta = json.load(tf.extractfile(m))
        elif m.name.endswith(".mid"):
            sizes[m.name] = m.size
    picked = select(meta, sizes, args.notes, args.hold_out)
    print(f"picked {len(picked):,} of {len(sizes):,} files")

    # pass 2: read only the picked files and parse them on all cores
    def wanted():
        for m, tf in members(args.archive):
            if m.name in picked:
                yield m.name, tf.extractfile(m).read()

    pieces = {"train": [], "validation": [], "test": []}
    with Pool(args.workers) as pool:
        for name, notes in pool.imap_unordered(parse, wanted(), chunksize=32):
            if notes is not None:
                pieces[picked[name]["split"]].append((name, notes))

    for split, items in pieces.items():
        items.sort()                                         # same order every run
        meta_out = [{k: v for k, v in picked[n].items() if k != "split"} for n, _ in items]
        notes = [p for _, p in items]
        offsets = np.cumsum([0] + [len(p) for p in notes])
        np.savez_compressed(os.path.join(args.out, f"{split}.npz"),
                            notes=np.concatenate(notes), offsets=offsets,
                            meta=json.dumps(meta_out))
        hours = sum(p[-1, 0] for p in notes) / 3.6e6         # last onset, ms -> h
        genres = collections.Counter(m["genre"] for m in meta_out)
        print(f"{split:<11} {len(notes):6,d} pieces  {offsets[-1]:11,d} notes  {hours:7.1f} h")
        print("            " + "  ".join(f"{g} {c:,}" for g, c in sorted(genres.items())))


if __name__ == "__main__":
    main()
