# 5. The dataset

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

## 5.2 Download it

You only need the MIDI files, which are a 56 MB download:

``` bash
mkdir -p data && cd data
curl -O https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
unzip -q maestro-v3.0.0-midi.zip
cd ..
```

This creates `data/maestro-v3.0.0/`, with one folder per competition year and a file called `maestro-v3.0.0.csv` that describes every performance. The columns we use are:

| Column | Example | Used for |
|----|----|----|
| `canonical_composer` | Frédéric Chopin | The composer and era tags |
| `canonical_title` | Ballade No. 1 in G Minor, Op. 23 | Reference only |
| `split` | train | Which file the performance goes into |
| `midi_filename` | 2004/MIDI-Unprocessed\_...\_wav.midi | Where the notes are |

## 5.3 Other datasets you may want later

Start with MAESTRO alone. When you want more variety, these are the natural next steps.

| Dataset | Size | What it is | Licence |
|----|----|----|----|
| MAESTRO v3 | 199 hours, 1,276 performances | Recorded directly from concert grands. The cleanest timing and velocity data available. | CC BY-NC-SA 4.0 |
| Aria-MIDI | About 100,000 hours, 1.19 million files | Piano recordings of all genres, turned into MIDI by a transcription model. Much larger and more varied, with some transcription errors. A deduplicated subset of 371,053 files is provided for training generative models. | CC BY-NC-SA 4.0 |
| MidiCaps | 168,000 files | Multi-instrument MIDI files from the Lakh collection, each with a written description, genre, mood, key and tempo. Useful for Part II. These are scores for bands, not piano performances. | CC BY-SA 4.0 |

## 5.4 Reading and writing MIDI

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
    """MIDI file -> note array. With use_pedal, notes released while the
    sustain pedal is down keep sounding until the pedal comes up, so the
    duration is the time the string actually rings."""
    now = 0.0
    pedal = False
    active = {}        # pitch -> (onset, velocity): key is held down
    sustained = {}     # pitch -> (onset, velocity): key up, pedal holding it
    notes = []

    def end(store, pitch):
        if pitch in store:
            on, vel = store.pop(pitch)
            notes.append((on, pitch, vel, max(now - on, 0.01)))

    for msg in mido.MidiFile(path):            # msg.time is seconds since last
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
        tick = int(round(t_ms * 0.96))
        track.append(mido.Message("note_on" if is_on else "note_off",
                                  note=pitch, velocity=vel, time=tick - last))
        last = tick
    mid.save(path)
```

`load_notes` walks through the file's events while keeping a running clock. Iterating over a `mido.MidiFile` gives each message a `time` field that is the number of seconds since the previous message, with the file's tempo already taken into account, so adding them up gives absolute time.

Two dictionaries track what is sounding. `active` holds keys that are physically down. `sustained` holds keys that have been released but are still ringing because the pedal is down. A note ends, and is written to the list, in one of three ways: the key is released with the pedal up, the pedal comes up while the note is in `sustained`, or the same key is struck again.

`save_notes` does the reverse. The only subtle part is the first loop: if the same pitch is struck again before the earlier note has finished, the earlier note is cut short at that moment. Without this, the earlier note's `note_off` would arrive in the middle of the new note and silence it. MIDI has no way to say which of two overlapping notes on the same key a `note_off` belongs to.
