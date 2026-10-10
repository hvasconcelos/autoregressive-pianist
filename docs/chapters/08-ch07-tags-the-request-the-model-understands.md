# 7. Tags: the request the model understands

A tag is a label attached to a passage of music. For the model to learn what `<dynamics:p>` means, every training passage must carry its correct tags. Labelling by hand is out of the question. Fortunately most useful tags can be *measured* from the notes.

## 7.1 The seven tags

| Tag | Values | How it is obtained |
|----|----|----|
| `density` | very_sparse, sparse, medium, dense, very_dense | Notes per second in the passage |
| `dynamics` | pp, p, mf, f, ff | Average velocity |
| `register` | low, mid, high | Average pitch |
| `key` | 24 keys, C major to B minor | Estimated from which pitches are used |
| `era` | baroque, classical, romantic, modern | Looked up from the composer |
| `composer` | 14 well-represented composers | From the dataset's metadata |
| `genre` | classical, jazz, pop, film, ragtime, other | `classical` for MAESTRO; from Aria-MIDI's metadata |

The first four are measured on each passage separately. That matters: a sonata has loud and quiet sections, and a tag for the whole piece would be wrong for most of its passages. The last three come from the piece's metadata and are the same for every passage of it.

::: {.admonition .note}
Why there is no tempo tag

Tempo means beats per minute, and a recorded performance does not come with beats. Estimating them from expressive playing is a research problem of its own. Notes per second is easy to measure exactly, and it captures most of what people mean by "fast" and "slow" when they ask for music.
:::

## 7.2 Estimating the key

The key is estimated with the Krumhansl-Schmuckler method, which is simple and works well on tonal music.

1.  Count how much each of the twelve pitch classes (C, C sharp, D, ...) is used in the passage, weighting each note by its duration. This gives a list of twelve numbers.
2.  Compare that list with a reference profile for a major key and one for a minor key. The profiles come from listening experiments and say how strongly each pitch class "belongs" in a key. The tonic and the fifth score highest.
3.  Rotate each profile to all twelve possible tonics, giving 24 candidates, and choose the one with the highest correlation.

The method confuses closely related keys now and then, such as C major and A minor, which share all their notes. For our purpose that is acceptable. The model learns from thousands of passages, and occasional near-miss labels do little harm.

## 7.3 The code

::: filename
tags.py
:::

``` python
"""Compute the request tags of a passage directly from its notes."""
import numpy as np
from tokenizer import KEYS, TAG_VALUES

# Krumhansl-Kessler key profiles: how strongly each of the 12 pitch classes
# belongs to a major / minor key whose tonic is pitch class 0.
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

DENSITY_EDGES = [4, 7, 11, 16]       # notes per second
DYNAMICS_EDGES = [45, 58, 70, 82]    # mean MIDI velocity
REGISTER_EDGES = [57, 69]            # mean MIDI pitch (57 = A3, 69 = A4)

ERA = {"baroque": ["bach", "handel", "scarlatti", "rameau", "couperin", "purcell",
                   "fischer"],
       "classical": ["haydn", "mozart", "beethoven", "clementi", "soler"],
       "romantic": ["schubert", "chopin", "schumann", "liszt", "mendelssohn",
                    "brahms", "grieg", "tchaikovsky", "franck", "balakirev",
                    "mussorgsky", "wagner", "verdi", "glinka", "weber", "bizet",
                    "strauss", "rimsky"],
       "modern": ["rachmaninoff", "scriabin", "debussy", "ravel", "prokofiev",
                  "bartok", "berg", "medtner", "janacek", "albeniz", "stravinsky",
                  "shostakovich", "busoni", "kapustin", "szymanowski"]}

# Aria-MIDI's genre labels -> our genre tag. Labels not listed get no genre.
GENRE = {"classical": "classical", "atonal": "classical", "pop": "pop", "rock": "pop",
         "soundtrack": "film", "jazz": "jazz", "blues": "jazz", "ragtime": "ragtime",
         "folk": "other", "ambient": "other"}
# Aria-MIDI's music_period -> era, used when the composer gives no era.
# "contemporary" is left out: it labels new music of any style, not an era.
PERIOD = {"baroque": "baroque", "classical": "classical", "romantic": "romantic",
          "impressionist": "modern", "modern": "modern"}


def estimate_key(notes):
    """Krumhansl-Schmuckler: correlate the passage's pitch-class histogram
    (weighted by duration) with the 24 rotated key profiles."""
    hist = np.zeros(12)
    np.add.at(hist, notes[:, 1].astype(int) % 12, np.minimum(notes[:, 3], 2000))
    best, best_r = None, -2.0
    for mode, profile in enumerate((MAJOR, MINOR)):
        for tonic in range(12):
            r = np.corrcoef(hist, np.roll(profile, tonic))[0, 1]
            if r > best_r:
                best, best_r = KEYS[mode * 12 + tonic], r
    return best


def compute_tags(notes):
    """Tags that can be measured from the notes of one passage.

    Each measurement is bucketed by the *_EDGES thresholds above; the
    bucket index picks the tag value."""
    span_s = max((notes[-1, 0] - notes[0, 0]) / 1000.0, 1.0)   # at least 1 s
    d = int(np.searchsorted(DENSITY_EDGES, len(notes) / span_s, side="right"))
    v = int(np.searchsorted(DYNAMICS_EDGES, notes[:, 2].mean(), side="right"))
    r = int(np.searchsorted(REGISTER_EDGES, notes[:, 1].mean(), side="right"))
    return {"density": TAG_VALUES["density"][d],
            "dynamics": TAG_VALUES["dynamics"][v],
            "register": TAG_VALUES["register"][r],
            "key": estimate_key(notes)}


def composer_tags(name):
    """'Frédéric Chopin' -> {'composer': 'chopin', 'era': 'romantic'}.
    For arrangements ('Franz Schubert / Franz Liszt') the first name wins.
    Composers outside TAG_VALUES get only an era; unknown ones get {}."""
    import re, unicodedata
    # strip accents so "Frédéric" matches "frederic", then lower-case
    plain = unicodedata.normalize("NFKD", name.split("/")[0])
    plain = "".join(c for c in plain if not unicodedata.combining(c)).lower()
    out = {}
    for era, names in ERA.items():
        for n in names:
            if re.search(rf"\b{n}\b", plain):
                out["era"] = era
                if n in TAG_VALUES["composer"]:
                    out["composer"] = n
                return out
    return out


def aria_tags(metadata):
    """Aria-MIDI metadata dict -> fixed tags {'genre', 'era', 'composer'}.

    Era and composer are only set for classical music, where they mean
    what they mean for MAESTRO."""
    genre = GENRE.get(metadata.get("genre"))
    if genre is None:
        return {}
    out = {"genre": genre}
    if genre == "classical":
        out.update(composer_tags(metadata.get("composer", "")))
        if "era" not in out and metadata.get("music_period") in PERIOD:
            out["era"] = PERIOD[metadata["music_period"]]
    return out
```

The three `_EDGES` lists are the thresholds between tag values. For example, a passage with between 7 and 11 notes per second is `medium`. The thresholds were chosen around MAESTRO's overall average of about 9.8 notes per second (7.04 million notes in 198.7 hours). They are reasonable starting values and you should check them against your data with `stats.py` below.

`composer_tags` matches surnames against the `ERA` table. It strips accents first, so "Janáček" matches "janacek". When a performance is an arrangement, MAESTRO writes both names separated by a slash, and we use the first.

`aria_tags` does the same job for Aria-MIDI's metadata. Aria uses more genre labels than we want as tags, so `GENRE` maps them onto our six:

| Aria label | `genre` tag |
|----|----|
| classical, atonal | `classical` |
| pop, rock | `pop` |
| soundtrack | `film` |
| jazz, blues | `jazz` |
| ragtime | `ragtime` |
| folk, ambient | `other` |

Files without a genre label get no tags at all and are left out, since they have nothing to teach about genre. For classical files, era and composer come from the composer's name exactly as for MAESTRO. When the name gives no era, Aria's `music_period` is used through `PERIOD`. Other genres get neither tag, because "romantic era" means nothing for a pop song.

## 7.4 Convert MAESTRO

`prepare.py` reads every MIDI file once and stores the notes in a compact form, so that training never has to parse MIDI. Every MAESTRO piece is tagged `genre: classical`.

::: filename
prepare.py
:::

``` python
"""Turn the MAESTRO MIDI files into three compact files of note arrays.

    python prepare.py --maestro data/maestro/maestro-v3.0.0 --out data/maestro_prepared

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
```

Run it:

``` bash
python prepare.py --maestro data/maestro/maestro-v3.0.0 --out data/maestro_prepared
```

It prints one line per split with the number of pieces, notes and hours. Compare them with the table in chapter 5. The hours should be close to MAESTRO's figures, and the note counts close to 5.66, 0.64 and 0.74 million.

If the script ends with a line starting `composers with no era tag:`, those composers are missing from the `ERA` table in `tags.py`. Their pieces are still used; they simply carry no era tag. Add the surnames to the table and run the script again if you want them labelled.

## 7.5 Convert Aria-MIDI

`prepare_aria.py` does the same for Aria-MIDI, with three differences.

- **It reads the archive directly.** `metadata.json` comes first in the `.tar.gz`, so a first pass reads it along with every file's size. A second pass reads only the chosen files and parses them on every core.
- **It chooses a slice.** Within each genre, files are taken in order of audio score, best first. A higher score means the classifier was more confident the recording is clean solo piano, so the transcription is more likely to be good. The genres take turns, one file each, until the note budget (`--notes`, 55 million by default, about ten times MAESTRO) is reached. Taken purely by score, classical and pop would fill most of the budget. Ragtime has only 3,319 files, so it runs out first and the other genres share the rest. The budget is counted from file sizes, at about 7.2 bytes per note, measured on a sample. It is only used to decide when to stop.
- **It makes its own split.** Aria has no train, validation and test sets, so the script makes them. Of the chosen recordings, one in 100 goes to validation and another one in 100 to test, decided by the recording's ID number. The split is therefore the same every time the script runs, every genre appears in all three sets, and no recording is split across two sets.

::: filename
prepare_aria.py
:::

``` python
"""Turn a balanced, high-quality slice of Aria-MIDI into note arrays.

    python prepare_aria.py --archive data/aria/aria-midi-v1-deduped-ext.tar.gz \
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
```

Run it:

``` bash
python prepare_aria.py --archive data/aria/aria-midi-v1-deduped-ext.tar.gz --out data/aria_prepared
```

It takes a few minutes on an M1 Pro and prints:

``` text
picked 38,464 of 371,053 files
train       37,683 pieces   53,623,616 notes   2184.5 h
            classical 6,995  film 7,014  jazz 6,996  other 7,022  pop 6,994  ragtime 2,662
validation     392 pieces      555,827 notes     22.7 h
            classical 75  film 73  jazz 75  other 70  pop 75  ragtime 24
test           389 pieces      559,958 notes     22.5 h
            classical 78  film 61  jazz 77  other 56  pop 78  ragtime 39
```

## 7.6 Check the tags

::: filename
stats.py
:::

``` python
"""Show how the tags are distributed in a prepared split.

    python stats.py --data data/maestro_prepared

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
    ap.add_argument("--data", default="data/maestro_prepared")
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
```

``` bash
python stats.py --data data/maestro_prepared
python stats.py --data data/aria_prepared
```

The script draws 2,000 random passages and prints, for each tag, what share of them got each value, as a bar of `#` characters. Look for two things.

- **No value should be nearly empty.** If `very_sparse` is at 1%, the model will see too few examples to learn it. Move the corresponding threshold in `tags.py` so that the smallest group has at least 5% or so, then re-run.
- **The tokens-per-note figure** should be a little under 4: 3.83 for MAESTRO, about 21.7 million tokens in all, and 3.81 for Aria, about 204 million.

For Aria, the genre section should show every genre between about 10% and 20%. Ragtime's share of passages is larger than its share of files, because ragtime pieces are long and dense, and passages are drawn in proportion to notes. Era and composer are rare in Aria, because only its classical files carry them.

Composer tags will be very uneven, which is expected: MAESTRO has far more of some composers than others. A rare composer tag will have a weaker effect, and a composer who does not appear in your copy of the data at all will show 0% here. A tag with no examples is never trained and does nothing; the list of fourteen in `tokenizer.py` was written from general knowledge of the repertoire and was not checked against the real files, so treat this output as the authority.
