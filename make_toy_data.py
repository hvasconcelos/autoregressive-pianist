"""Write a tiny fake dataset laid out exactly like MAESTRO, for smoke tests.

    python make_toy_data.py --out data/toy

The "music" is a random walk on a scale over simple chords. It is not good
music; it has an obvious key, loudness, speed and register, so we can check
that every stage of the pipeline works before spending hours on real data.
"""
import argparse, csv, os
import numpy as np
from midi_io import save_notes

MAJ, MIN = [0, 2, 4, 5, 7, 9, 11], [0, 2, 3, 5, 7, 8, 10]
COMPOSERS = ["Johann Sebastian Bach", "Wolfgang Amadeus Mozart", "Frédéric Chopin",
             "Claude Debussy", "Franz Schubert / Franz Liszt"]


def toy_piece(rng, seconds=60):
    """One random piece: a melody wandering on a scale, sometimes over a chord.
    Key, speed, loudness and register are fixed per piece."""
    tonic, scale = rng.integers(0, 12), (MAJ, MIN)[rng.integers(0, 2)]
    step_ms = rng.choice([60, 90, 140, 220, 450])         # speed of the melody
    loud = rng.choice([35, 52, 64, 76, 95])               # overall loudness
    centre = rng.choice([50, 62, 76])                     # register
    notes, t, degree = [], 0.0, 0
    while t < seconds * 1000:
        # step up or down the scale by at most two degrees
        degree = int(np.clip(degree + rng.integers(-2, 3), -7, 7))
        pitch = centre + tonic + 12 * (degree // 7) + scale[degree % 7]
        vel = int(np.clip(loud + rng.normal(0, 5), 1, 127))
        notes.append((t, pitch, vel, step_ms * rng.uniform(0.6, 1.1)))
        if rng.random() < 0.12:                           # a triad underneath
            for d in (0, 2, 4):
                low = centre - 12 + tonic + scale[d]
                if low != pitch:
                    notes.append((t, low, max(vel - 10, 1), step_ms * 3))
        # usually one step, sometimes two, with a little timing jitter
        t += step_ms * rng.choice([1, 1, 1, 2]) + rng.normal(0, 4)
    notes = np.array(notes)
    notes[:, 1] = np.clip(notes[:, 1], 21, 108)
    notes[:, 0] = np.maximum(notes[:, 0], 0)
    return notes


def main():
    """Write the toy MIDI files and a MAESTRO-style csv listing them."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/toy")
    ap.add_argument("--pieces", type=int, default=120)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    # MAESTRO keeps files in one folder per year; mimic that
    os.makedirs(os.path.join(args.out, "2004"), exist_ok=True)
    rows = []
    for i in range(args.pieces):
        name = f"2004/toy_{i:04d}.midi"
        save_notes(toy_piece(rng), os.path.join(args.out, name))
        # 80 % train, 10 % validation, 10 % test
        split = "train" if i % 10 < 8 else ("validation" if i % 10 == 8 else "test")
        rows.append({"canonical_composer": COMPOSERS[i % len(COMPOSERS)],
                     "canonical_title": f"Toy piece {i}", "split": split, "year": 2004,
                     "midi_filename": name, "audio_filename": "", "duration": 60})
    path = os.path.join(args.out, "maestro-v3.0.0.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {args.pieces} toy pieces to {args.out}")


if __name__ == "__main__":
    main()
