"""Turn the MAESTRO MIDI files into three compact files of note arrays.

    python prepare.py --maestro data/maestro-v3.0.0 --out data/prepared

Writes train.npz, validation.npz and test.npz. Each holds every note of the
split in one big array, plus where each piece starts and its composer tags.
"""
# argparse: command-line options; csv: read MAESTRO's metadata table;
# json: store per-piece metadata as text; os: paths and folders
import argparse, csv, json, os
# numpy: note arrays and the compressed .npz output
import numpy as np
# load_notes: MIDI file -> (onset_ms, pitch, velocity, dur_ms) array, with pedal handling
from midi_io import load_notes
# composer_tags: composer name -> {"era": ..., "composer": ...}
from tags import composer_tags


def main():
    """Read the MAESTRO csv, load every MIDI file and write one .npz per split."""
    # command-line options:
    ap = argparse.ArgumentParser()
    # --maestro: the unpacked MAESTRO dataset folder (must contain the .csv index)
    ap.add_argument("--maestro", required=True, help="folder with maestro-v3.0.0.csv")
    # --out: where the .npz files are written
    ap.add_argument("--out", default="data/prepared")
    args = ap.parse_args()
    # create the output folder (no error if it already exists)
    os.makedirs(args.out, exist_ok=True)

    # find the metadata .csv in the dataset folder (takes the first one found)
    csv_path = [f for f in os.listdir(args.maestro) if f.endswith(".csv")][0]
    # read every row of the csv as a dict keyed by column name (split, midi_filename, ...)
    rows = list(csv.DictReader(open(os.path.join(args.maestro, csv_path), encoding="utf-8")))
    # composer names we couldn't assign an era to, reported at the end
    unknown = set()

    # build each of MAESTRO's three predefined splits in turn
    for split in ("train", "validation", "test"):
        # pieces: one note array per piece; meta: one dict of tags/title per piece
        pieces, meta = [], []
        for row in rows:
            # skip pieces belonging to other splits
            if row["split"] != split:
                continue
            # read the performance's MIDI file into a note array
            notes = load_notes(os.path.join(args.maestro, row["midi_filename"]))
            # keep only the 88 piano keys, and skip fragments too short to use
            # MIDI pitch 21 = A0 (lowest piano key), 108 = C8 (highest); boolean mask of rows.
            keep = (notes[:, 1] >= 21) & (notes[:, 1] <= 108)
            notes = notes[keep]
            # pieces with fewer than 64 notes are dropped
            if len(notes) < 64:
                continue
            # era/composer tags from MAESTRO's normalised composer name
            tags = composer_tags(row["canonical_composer"])
            # note composers that got no era so the ERA table can be extended
            if "era" not in tags:
                unknown.add(row["canonical_composer"])
            # float32 halves the file size; ms precision is plenty
            pieces.append(notes.astype(np.float32))
            # store the tags plus title and full composer name for this piece
            meta.append({**tags, "title": row["canonical_title"],
                         "composer_name": row["canonical_composer"]})
        # running total of note counts, e.g. pieces of 100, 50 notes -> [0, 100, 150];
        # piece i is rows offsets[i]:offsets[i+1] of the big array, offsets[-1] = total notes.
        offsets = np.cumsum([0] + [len(p) for p in pieces])   # where each piece starts
        # write one compressed file: all notes stacked into one array, the offsets, and the
        # metadata list as a JSON string.
        np.savez_compressed(os.path.join(args.out, f"{split}.npz"),
                            notes=np.concatenate(pieces), offsets=offsets,
                            meta=json.dumps(meta))
        # total hours of music, using each piece's last onset (3.6e6 ms in an hour)
        hours = sum(p[-1, 0] for p in pieces) / 3.6e6        # last onset, ms -> h
        # summary line: split name, number of pieces, number of notes, hours
        print(f"{split:<11} {len(pieces):5d} pieces  {offsets[-1]:9,d} notes  {hours:6.1f} h")

    # list any composers that got no era tag
    if unknown:
        print("composers with no era tag:", sorted(unknown))


# run main() only when executed as a script
if __name__ == "__main__":
    main()
