# Appendix D. Adding Aria-MIDI and a genre tag

MAESTRO is clean but small, and it is all classical. Aria-MIDI adds about ten times as much music, and with it a new tag: **genre**, so a request can ask for jazz, pop, film music or ragtime as well as classical. This appendix prepares an Aria slice with its own train, validation and test sets, adds the genre tag, and trains a second small model (`runs/v2`) on MAESTRO and Aria together.

Everything else in the pipeline stays as it is. Batches, augmentation, the model and the Performer don't change.

## D.1 The data layout

Each dataset has a folder for the raw download and a `_prepared` folder beside it for the `.npz` files:

``` text
data/
  maestro/            maestro-v3.0.0-midi.zip and the unzipped maestro-v3.0.0/
  maestro_prepared/   train.npz, validation.npz, test.npz
  aria/               aria-midi-v1-deduped-ext.tar.gz (read without unpacking)
  aria_prepared/      train.npz, validation.npz, test.npz
  toy/                fake MAESTRO-style data from make_toy_data.py
  toy_prepared/       train.npz, validation.npz, test.npz
```

## D.2 Download

Use the deduplicated subset: 371,053 files and a 2.0 GB download. It's the subset the dataset's authors filtered most heavily for training generative models. Aria-MIDI is published on Hugging Face under CC BY-NC-SA 4.0, the same licence as MAESTRO.

``` bash
hf download loubb/aria-midi aria-midi-v1-deduped-ext.tar.gz \
    --repo-type dataset --local-dir data/aria
```

There's no need to unpack it. Unpacked, it is several gigabytes spread over 371,053 small files. `prepare_aria.py` reads the archive directly instead.

Inside the archive, `metadata.json` comes first. For each recording it holds the labels that were extracted from the recording's title and description, and an audio score from the classifier that judged whether the recording is solo piano:

``` text
"2": {"metadata": {"composer": "strauss", "form": "waltz", "performer": "cziffra",
                   "genre": "classical", "music_period": "classical"},
      "audio_scores": {"0": 0.9902}}
```

The files themselves are named `<recording id>_<segment>.mid`.

::: {.admonition .warning}
How much memory and disk this takes

The full deduplicated subset is about 100 times MAESTRO. Its note arrays would not fit in 16 GB of memory, and one pass over it would take days on an M1 Pro. A slice of about ten times MAESTRO is what this appendix uses. It needs about 1 GB of memory for training and 350 MB of disk once prepared.
:::

## D.3 The genre tag

The genre tag is one more category in `TAG_VALUES` in `tokenizer.py`:

::: filename
tokenizer.py
:::

``` python
    # MAESTRO is all classical; the rest come from Aria-MIDI's metadata (tags.aria_tags)
    "genre":    ["classical", "jazz", "pop", "film", "ragtime", "other"],
```

It adds six tokens, so the vocabulary grows from 459 to 465. A model trained before this change can't be loaded by the new code, so the genre tag means training a new model.

Unlike density or key, genre can't be measured from the notes. Like era and composer, it is fixed per piece and stored in the `meta` of the `.npz` files. `data.py` copies it into each passage's tags, and tag dropout removes it at random like any other tag. All MAESTRO pieces are tagged `classical`, so run `prepare.py` once more after adding the tag.

## D.4 From Aria's labels to tags

Aria uses more genre labels than we want as tags. Counted over the deduplicated subset, and mapped onto the six values:

| Aria label | Files | `genre` tag |
|----|---:|----|
| classical | 112,946 | `classical` |
| *(none)* | 94,105 | not used |
| pop | 70,024 | `pop` |
| soundtrack | 54,912 | `film` |
| jazz | 18,839 | `jazz` |
| rock | 6,666 | `pop` |
| folk | 5,299 | `other` |
| ambient | 3,561 | `other` |
| ragtime | 3,319 | `ragtime` |
| blues | 1,248 | `jazz` |
| atonal | 134 | `classical` |

Files without a genre label are skipped, since they have nothing to teach about genre. For classical files, era and composer come from the composer's name, exactly as for MAESTRO. When the name gives no era, Aria's `music_period` is used. Other genres get neither tag, because "romantic era" means nothing for a pop song.

::: filename
tags.py
:::

``` python
# Aria-MIDI's genre labels -> our genre tag. Labels not listed get no genre.
GENRE = {"classical": "classical", "atonal": "classical", "pop": "pop", "rock": "pop",
         "soundtrack": "film", "jazz": "jazz", "blues": "jazz", "ragtime": "ragtime",
         "folk": "other", "ambient": "other"}
# Aria-MIDI's music_period -> era, used when the composer gives no era.
# "contemporary" is left out: it labels new music of any style, not an era.
PERIOD = {"baroque": "baroque", "classical": "classical", "romantic": "romantic",
          "impressionist": "modern", "modern": "modern"}


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

## D.5 Choosing the files

Two things decide which files go in.

- **Quality first.** Within each genre, files are taken in order of audio score, best first. A higher score means the classifier was more confident that the recording is clean solo piano, so the transcription is more likely to be good.
- **Genres in turn.** Files are taken round-robin, one from each genre in turn, until the note budget is reached. Taken purely by score, classical and pop would fill most of the budget and leave little jazz or ragtime. Ragtime has only 3,319 files, so it runs out first and the other genres share the rest.

The budget is counted in notes, estimated from each file's size: about 7.2 bytes per note, measured on a sample. The estimate is only used to decide when to stop.

## D.6 Train, validation and test

MAESTRO comes with its own split. Aria doesn't, so `prepare_aria.py` makes one. Of the picked recordings, one in 100 goes to validation and another one in 100 to test, decided by the recording's ID number. This means:

- the split is the same every time the script runs;
- every genre appears in all three sets, in about the same proportion as in training;
- a recording is never split across two sets.

When training on both datasets, validation during training uses **MAESTRO's** validation set, so the loss stays comparable with the model of chapter 11. Aria's validation and test sets are for measuring the new genres afterwards (D.9).

::: filename
prepare_aria.py
:::

``` python
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
```

## D.7 Preparing the data

``` bash
python prepare.py --maestro data/maestro/maestro-v3.0.0 --out data/maestro_prepared
python prepare_aria.py --archive data/aria/aria-midi-v1-deduped-ext.tar.gz --out data/aria_prepared
python stats.py --data data/aria_prepared
```

`prepare_aria.py` reads the archive twice: once for the metadata and file sizes, then again to parse only the picked files on every core. On an M1 Pro it takes a few minutes. With the default budget of 55M notes it prints:

``` text
picked 38,464 of 371,053 files
train       37,683 pieces   53,623,616 notes   2184.5 h
            classical 6,995  film 7,014  jazz 6,996  other 7,022  pop 6,994  ragtime 2,662
validation     392 pieces      555,827 notes     22.7 h
            classical 75  film 73  jazz 75  other 70  pop 75  ragtime 24
test           389 pieces      559,958 notes     22.5 h
            classical 78  film 61  jazz 77  other 56  pop 78  ragtime 39
```

`stats.py` now shows a genre section too. Ragtime's share of passages is larger than its share of files, because ragtime pieces are long and dense, and passages are drawn in proportion to notes.

## D.8 Training on both

`--data` takes several folders. Training draws from all of their `train.npz` files, and validation uses the first folder only:

``` bash
caffeinate -i python train.py --data data/maestro_prepared data/aria_prepared \
    --out runs/v2 --batch 8 --steps 60000
```

Together, the two datasets hold about 59 million notes, roughly 230 million tokens and ten times what chapter 11 trained on. MAESTRO is now less than a tenth of the training music. Keep the model the same size as v1, so any difference comes from the data and not from the model. 60,000 steps is about two passes over the data and takes about nine and a half hours on an M1 Pro.

Expect the validation loss to trail v1's at the same step for most of the run. Validation is on MAESTRO, and v2 sees far less MAESTRO per step while it also learns four new genres. It catches up as the learning rate decays.

## D.9 Measuring it

The model has two kinds of data to be judged on now.

``` bash
# classical, against v1's numbers in chapter 15
python evaluate.py --model runs/v2/best --data data/maestro_prepared --split test --keys
# the new genres
python evaluate.py --model runs/v2/best --data data/aria_prepared --split test
```

The first line answers whether the extra data helped, or at least didn't hurt, on MAESTRO. Compare its loss with v1's: a difference under 0.05 is noise. The second gives a loss for Aria's test set, which v1 can't be compared on fairly, because it never saw those genres.

Genre itself can't be scored by `evaluate.py`, because it can't be measured from the notes. Listen instead. Generate the same request with each genre and check that you can tell them apart:

``` bash
for g in classical jazz pop film ragtime; do
  python sample.py --model runs/v2/best --tags genre=$g,density=medium --out $g.mid
done
```

Plain-words requests work too: `request.py` maps words such as "jazzy", "bebop", "soundtrack", "cinematic", "stride" and "ragtime" to the genre tag.
