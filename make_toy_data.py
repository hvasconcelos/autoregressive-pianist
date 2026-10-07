"""Write a tiny fake dataset laid out exactly like MAESTRO, for smoke tests.

    uv run python make_toy_data.py --out data/toy

The "music" is a random walk on a scale over simple chords. It is not good
music; it has an obvious key, loudness, speed and register, so we can check
that every stage of the pipeline works before spending hours on real data.
"""
# argparse: command-line options; csv: write the MAESTRO-style index; os: paths and folders
import argparse, csv, os
# numpy: random numbers and the note array
import numpy as np
# save_notes: note array -> MIDI file
from midi_io import save_notes

# semitone offsets of the 7 notes of a major and a natural minor scale, from the tonic
# (e.g. major on C: C D E F G A B).
MAJ, MIN = [0, 2, 4, 5, 7, 9, 11], [0, 2, 3, 5, 7, 8, 10]
# composer names in MAESTRO's format; the last one tests the "arrangement" case
COMPOSERS = ["Johann Sebastian Bach", "Wolfgang Amadeus Mozart", "Frédéric Chopin",
             "Claude Debussy", "Franz Schubert / Franz Liszt"]


def toy_piece(rng, seconds=60):
    """One random piece: a melody wandering on a scale, sometimes over a chord.
    Key, speed, loudness and register are fixed per piece."""
    # random tonic pitch class (0 = C .. 11 = B) and random major or minor scale.
    tonic, scale = rng.integers(0, 12), (MAJ, MIN)[rng.integers(0, 2)]
    # milliseconds between melody notes; picked from 5 values to spread across density buckets
    step_ms = rng.choice([60, 90, 140, 220, 450])         # speed of the melody
    # base MIDI velocity; one value per dynamics bucket (pp .. ff).
    loud = rng.choice([35, 52, 64, 76, 95])               # overall loudness
    # base MIDI pitch: 50 (low), 62 (around middle C), 76 (high)
    centre = rng.choice([50, 62, 76])                     # register
    # notes: collected rows; t: current time in ms; degree: position on the scale
    notes, t, degree = [], 0.0, 0
    # keep adding notes until the piece is `seconds` long
    while t < seconds * 1000:
        # step up or down the scale by at most two degrees
        # (rng.integers(-2, 3) gives -2..2; clip keeps the melody within one octave each way)
        degree = int(np.clip(degree + rng.integers(-2, 3), -7, 7))
        # scale degree -> MIDI pitch: degree // 7 is the octave (floors, so -1 -> octave -1),
        # degree % 7 picks the scale note within it, e.g. degree -1 = 7th note an octave down.
        pitch = centre + tonic + 12 * (degree // 7) + scale[degree % 7]
        # velocity: base loudness plus a little random variation, kept in MIDI's 1..127
        vel = int(np.clip(loud + rng.normal(0, 5), 1, 127))
        # add the melody note; it lasts 60-110 % of the step so notes slightly detach/overlap
        notes.append((t, pitch, vel, step_ms * rng.uniform(0.6, 1.1)))
        if rng.random() < 0.12:                           # a triad underneath
            # scale degrees 0, 2, 4 = root, third, fifth: the tonic chord
            for d in (0, 2, 4):
                # chord note one octave below the melody's centre
                low = centre - 12 + tonic + scale[d]
                # skip a chord note that would duplicate the melody pitch
                if low != pitch:
                    # chord notes are a bit softer and last three steps
                    notes.append((t, low, max(vel - 10, 1), step_ms * 3))
        # usually one step, sometimes two, with a little timing jitter
        t += step_ms * rng.choice([1, 1, 1, 2]) + rng.normal(0, 4)
    # list of tuples -> (N, 4) array
    notes = np.array(notes)
    # keep pitches on the 88-key piano range (21 = A0 .. 108 = C8).
    notes[:, 1] = np.clip(notes[:, 1], 21, 108)
    # no negative onsets
    notes[:, 0] = np.maximum(notes[:, 0], 0)
    return notes


def main():
    """Write the toy MIDI files and a MAESTRO-style csv listing them."""
    # command-line options:
    ap = argparse.ArgumentParser()
    # --out: folder to create the fake dataset in
    ap.add_argument("--out", default="data/toy")
    # --pieces: how many toy pieces to generate
    ap.add_argument("--pieces", type=int, default=120)
    args = ap.parse_args()
    # fixed seed, so the toy dataset is the same every run
    rng = np.random.default_rng(0)
    # MAESTRO keeps files in one folder per year; mimic that
    os.makedirs(os.path.join(args.out, "2004"), exist_ok=True)
    # one csv row per piece
    rows = []
    for i in range(args.pieces):
        # path relative to the dataset folder, as in MAESTRO's midi_filename column
        name = f"2004/toy_{i:04d}.midi"
        # generate a piece and write it as MIDI
        save_notes(toy_piece(rng), os.path.join(args.out, name))
        # 80 % train, 10 % validation, 10 % test
        # (i % 10 in 0..7 -> train, 8 -> validation, 9 -> test)
        split = "train" if i % 10 < 8 else ("validation" if i % 10 == 8 else "test")
        # same columns as the real MAESTRO csv; composers cycle through the list
        rows.append({"canonical_composer": COMPOSERS[i % len(COMPOSERS)],
                     "canonical_title": f"Toy piece {i}", "split": split, "year": 2004,
                     "midi_filename": name, "audio_filename": "", "duration": 60})
    # use MAESTRO's csv file name so prepare.py finds it
    path = os.path.join(args.out, "maestro-v3.0.0.csv")
    # newline="" is what the csv module needs to avoid blank lines on Windows
    with open(path, "w", newline="", encoding="utf-8") as f:
        # column names taken from the keys of the first row
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        # header line, then all the rows
        w.writeheader(); w.writerows(rows)
    print(f"wrote {args.pieces} toy pieces to {args.out}")


# run main() only when executed as a script
if __name__ == "__main__":
    main()
