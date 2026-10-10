# 5. The datasets

We train on two datasets that complement each other. MAESTRO is small and clean, and it is all classical. Aria-MIDI is large and varied, but less clean. Together they give the model accurate classical playing and a range of other styles.

## 5.1 MAESTRO

MAESTRO is a collection of piano performances recorded at the International Piano-e-Competition over ten years. The pianists played Yamaha Disklavier concert grands, which are acoustic pianos with sensors that record every key and pedal movement as MIDI. The result is real virtuoso playing captured key by key. (The dataset also includes the audio, aligned to the MIDI to within about 3 milliseconds; we use only the MIDI.)

The figures for version 3.0.0, the current one:

| Split      | Performances | Hours     | Notes (millions) |
|------------|--------------|-----------|------------------|
| Train      | 962          | 159.2     | 5.66             |
| Validation | 137          | 19.4      | 0.64             |
| Test       | 177          | 20.0      | 0.74             |
| **Total**  | **1,276**    | **198.7** | **7.04**         |

The repertoire is classical, from the 17th to the early 20th century, with a lot of Chopin, Liszt, Beethoven, Schubert and Bach. The splits are arranged so that the same composition never appears in more than one of them, which means the validation loss measures how the model handles music it has not seen, and not just a different recording of a piece it has memorised.

The licence is Creative Commons Attribution Non-Commercial Share-Alike 4.0. You may use it freely for research and personal projects. A commercial product would need different data.

## 5.2 Aria-MIDI

Aria-MIDI is about 100,000 hours of solo piano: 1.19 million files, transcribed from recordings by a model that turns audio into MIDI. The recordings come from many sources and every genre, so it brings jazz, pop, film music and ragtime that MAESTRO lacks. The price is accuracy. A transcription has occasional wrong notes, and its velocities are estimated from the sound, not measured at the key.

We use the **deduplicated subset**, which the dataset's authors filtered most heavily for training generative models: 371,053 files and a 2.0 GB download. Each file is one stretch of solo piano from one recording, named `<recording id>_<segment>.mid`. A file called `metadata.json` describes every recording, with labels extracted from its title and description, and an audio score from the classifier that judged whether the recording is clean solo piano:

``` text
"2": {"metadata": {"composer": "strauss", "form": "waltz", "performer": "cziffra",
                   "genre": "classical", "music_period": "classical"},
      "audio_scores": {"0": 0.9902}}
```

The genre labels across the subset:

| Aria genre | Files |
|----|---:|
| classical | 112,946 |
| *(none)* | 94,105 |
| pop | 70,024 |
| soundtrack | 54,912 |
| jazz | 18,839 |
| rock | 6,666 |
| folk | 5,299 |
| ambient | 3,561 |
| ragtime | 3,319 |
| blues | 1,248 |
| atonal | 134 |

We don't use all of it. The whole subset is about 100 times MAESTRO. Its note arrays would not fit in 16 GB of memory, and one pass over it would take days on an M1 Pro. Instead, chapter 7 picks a slice of about ten times MAESTRO: the best-scoring files of each genre, in equal numbers where the genre has enough. It also makes Aria's train, validation and test sets, which the dataset doesn't provide.

Aria-MIDI has the same licence as MAESTRO, CC BY-NC-SA 4.0.

## 5.3 Download them

Each dataset gets a folder for the raw download, and later a `_prepared` folder beside it for the converted files:

``` text
data/
  maestro/            maestro-v3.0.0-midi.zip and the unzipped maestro-v3.0.0/
  maestro_prepared/   train.npz, validation.npz, test.npz   (chapter 7)
  aria/               aria-midi-v1-deduped-ext.tar.gz
  aria_prepared/      train.npz, validation.npz, test.npz   (chapter 7)
  toy/                fake data for the smoke test          (chapter 10)
  toy_prepared/       train.npz, validation.npz, test.npz   (chapter 10)
```

For MAESTRO you only need the MIDI files, which are a 56 MB download:

``` bash
mkdir -p data/maestro && cd data/maestro
curl -O https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
unzip -q maestro-v3.0.0-midi.zip
cd ../..
```

This creates `data/maestro/maestro-v3.0.0/`, with one folder per competition year and a file called `maestro-v3.0.0.csv` that describes every performance. The columns we use are:

| Column | Example | Used for |
|----|----|----|
| `canonical_composer` | Frédéric Chopin | The composer and era tags |
| `canonical_title` | Ballade No. 1 in G Minor, Op. 23 | Reference only |
| `split` | train | Which file the performance goes into |
| `midi_filename` | 2004/MIDI-Unprocessed\_...\_wav.midi | Where the notes are |

Aria-MIDI is published on Hugging Face. Download the deduplicated subset with the `hf` tool:

``` bash
hf download loubb/aria-midi aria-midi-v1-deduped-ext.tar.gz \
    --repo-type dataset --local-dir data/aria
```

Don't unpack it. Unpacked, it is several gigabytes spread over 371,053 small files. Chapter 7 reads the archive directly.

## 5.4 Other datasets you may want later

| Dataset | Size | What it is | Licence |
|----|----|----|----|
| Aria-MIDI, pruned or full | 820,944 or 1,186,253 files | The larger Aria-MIDI subsets: less filtered and with more near-duplicates. For a bigger model on bigger hardware. | CC BY-NC-SA 4.0 |
| MidiCaps | 168,000 files | Multi-instrument MIDI files from the Lakh collection, each with a written description, genre, mood, key and tempo. Useful for Part II. These are scores for bands, not piano performances. | CC BY-SA 4.0 |

## 5.5 Reading and writing MIDI

The first file of the project converts between MIDI files and note arrays.

::: filename
midi_io.py
:::

``` python
"""Read a MIDI file into a note array, and write a note array back.

Note array: one row per note, columns (onset_ms, pitch, velocity, dur_ms).
"""
import mido
import numpy as np


def load_notes(path, use_pedal=True):
    """MIDI file (a path or an open binary file) -> note array. With
    use_pedal, notes released while the sustain pedal is down keep sounding
    until the pedal comes up, so the duration is the time the string
    actually rings."""
    now = 0.0
    pedal = False
    active = {}        # pitch -> (onset, velocity): key is held down
    sustained = {}     # pitch -> (onset, velocity): key up, pedal holding it
    notes = []

    def end(store, pitch):
        """Finish the note on `pitch` in `store` (if any) at the current time."""
        if pitch in store:
            on, vel = store.pop(pitch)
            notes.append((on, pitch, vel, max(now - on, 0.01)))

    midi = mido.MidiFile(path) if isinstance(path, str) else mido.MidiFile(file=path)
    for msg in midi:                           # msg.time is seconds since last
        now += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            end(active, msg.note); end(sustained, msg.note)   # re-struck key
            active[msg.note] = (now, msg.velocity)
        elif msg.type in ("note_off", "note_on"):             # note_on vel 0
            if pedal and use_pedal and msg.note in active:
                sustained[msg.note] = active.pop(msg.note)
            else:
                end(active, msg.note)
        elif msg.type == "control_change" and msg.control == 64:
            pedal = msg.value >= 64
            if not pedal:
                for p in list(sustained):
                    end(sustained, p)
    for store in (active, sustained):
        for p in list(store):
            end(store, p)

    arr = np.array(notes, dtype=np.float64).reshape(-1, 4)
    arr[:, [0, 3]] *= 1000.0                                  # seconds -> ms
    return arr[np.lexsort((arr[:, 1], arr[:, 0]))]


def save_notes(notes, path):
    """Note array -> a one-track MIDI file (120 bpm, 480 ticks per beat,
    so one tick is 1/960 s)."""
    notes = np.array(notes, dtype=np.float64).reshape(-1, 4)
    notes = notes[np.argsort(notes[:, 0], kind="stable")]
    end = notes[:, 0] + notes[:, 3]
    last = {}                                   # pitch -> index of its last note
    for i, pitch in enumerate(notes[:, 1]):
        j = last.get(pitch)
        if j is not None and end[j] > notes[i, 0]:
            end[j] = notes[i, 0]                # key struck again: cut the old note
        last[pitch] = i
    events = []
    for (on, pitch, vel, _), off in zip(notes, end):
        events.append((on, 1, int(pitch), int(vel)))
        events.append((off, 0, int(pitch), 0))
    events.sort()                               # note-offs before note-ons
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    last = 0
    for t_ms, is_on, pitch, vel in events:
        tick = int(round(t_ms * 0.96))         # 960 ticks per second
        # MIDI times are deltas from the previous event
        track.append(mido.Message("note_on" if is_on else "note_off",
                                  note=pitch, velocity=vel, time=tick - last))
        last = tick
    mid.save(path)
```

`load_notes` accepts either a path or an open file. MAESTRO's files are read by path; Aria's are read straight out of the archive as bytes.

It walks through the file's events while keeping a running clock. Iterating over a `mido.MidiFile` gives each message a `time` field that is the number of seconds since the previous message, with the file's tempo already taken into account, so adding them up gives absolute time.

Two dictionaries track what is sounding. `active` holds keys that are physically down. `sustained` holds keys that have been released but are still ringing because the pedal is down. A note ends, and is written to the list, in one of three ways: the key is released with the pedal up, the pedal comes up while the note is in `sustained`, or the same key is struck again.

`save_notes` does the reverse. The only subtle part is the first loop: if the same pitch is struck again before the earlier note has finished, the earlier note is cut short at that moment. Without this, the earlier note's `note_off` would arrive in the middle of the new note and silence it. MIDI has no way to say which of two overlapping notes on the same key a `note_off` belongs to.
