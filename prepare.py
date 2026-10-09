"""Turn the MAESTRO MIDI files into three compact files of note arrays.

    uv run python prepare.py --maestro data/maestro/maestro-v3.0.0 --out data/maestro_prepared

Writes train.npz, validation.npz and test.npz. Each holds every note of the
split in one big array, plus where each piece starts and its fixed tags
(genre, era, composer).
"""
import argparse, csv, json, os
import numpy as np
from midi_io import load_notes
from tags import composer_tags


def main():
    """Read the MAESTRO csv, load every MIDI file and write one .npz per split."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--maestro", default="data/maestro/maestro-v3.0.0",
                    help="folder with maestro-v3.0.0.csv")
    ap.add_argument("--out", default="data/maestro_prepared")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    csv_path = [f for f in os.listdir(args.maestro) if f.endswith(".csv")][0]
    rows = list(csv.DictReader(open(os.path.join(args.maestro, csv_path), encoding="utf-8")))
    unknown = set()

    for split in ("train", "validation", "test"):
        pieces, meta = [], []
        for row in rows:
            if row["split"] != split:
                continue
            notes = load_notes(os.path.join(args.maestro, row["midi_filename"]))
            # keep only the 88 piano keys, and skip fragments too short to use
            keep = (notes[:, 1] >= 21) & (notes[:, 1] <= 108)
            notes = notes[keep]
            if len(notes) < 64:
                continue
            tags = {"genre": "classical", **composer_tags(row["canonical_composer"])}
            if "era" not in tags:
                unknown.add(row["canonical_composer"])
            pieces.append(notes.astype(np.float32))
            meta.append({**tags, "title": row["canonical_title"],
                         "composer_name": row["canonical_composer"]})
        offsets = np.cumsum([0] + [len(p) for p in pieces])   # where each piece starts
        np.savez_compressed(os.path.join(args.out, f"{split}.npz"),
                            notes=np.concatenate(pieces), offsets=offsets,
                            meta=json.dumps(meta))
        hours = sum(p[-1, 0] for p in pieces) / 3.6e6        # last onset, ms -> h
        print(f"{split:<11} {len(pieces):5d} pieces  {offsets[-1]:9,d} notes  {hours:6.1f} h")

    if unknown:
        print("composers with no era tag:", sorted(unknown))


if __name__ == "__main__":
    main()
