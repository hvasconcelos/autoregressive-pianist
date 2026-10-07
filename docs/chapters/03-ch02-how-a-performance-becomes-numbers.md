# 2. How a performance becomes numbers

Before writing any code it helps to be exact about what the data is.

## 2.1 MIDI in five minutes

MIDI is the language keyboards use to talk to each other. It carries no sound. It carries *events*: "key 60 went down with force 80", "key 60 came up". A MIDI file is a list of such events with the time between them.

The events that matter for piano are:

| Event | Fields | Meaning |
|----|----|----|
| `note_on` | pitch 0 to 127, velocity 1 to 127 | A key was struck. Velocity is how hard. |
| `note_off` | pitch | The key was released. (A `note_on` with velocity 0 means the same thing.) |
| `control_change` 64 | value 0 to 127 | The sustain pedal moved. 64 or more means down. |

**Pitch** is a key number. Middle C is 60, and each step of 1 is one key to the right, black keys included. An 88-key piano spans 21 (the lowest A) to 108 (the highest C). Twelve steps make an octave, so pitch 72 is the C above middle C.

**Velocity** is the speed of the key when it hit, measured by the instrument. It controls both loudness and tone. Values in the 30s are very quiet, the 60s are a comfortable medium and the 100s are forceful.

## 2.2 A performance is not a score

A score says "a quarter note, then two eighth notes". A performance says "this key went down at 12.431 seconds and came up at 12.902". The difference is the whole point.

A human never plays exactly on the grid. Notes of a chord land a few milliseconds apart, the melody is a little louder than the accompaniment, the tempo breathes. Those small deviations are what make a recording sound like a person. A model trained on scores plays like a music box. A model trained on performances learns the deviations along with the notes.

This is why the choice of dataset matters so much, and why MAESTRO is the right starting point: it was captured from real concert grands fitted with sensors that record every key movement.

## 2.3 The note array

Everything in this project passes through one simple structure, a table with a row per note and four columns:

| Column   | Unit                        | Example |
|----------|-----------------------------|---------|
| onset    | milliseconds from the start | 1500    |
| pitch    | MIDI key number             | 72      |
| velocity | 1 to 127                    | 92      |
| duration | milliseconds                | 500     |

Reading a MIDI file means pairing each `note_on` with its `note_off` to get a row. Writing one means doing the reverse. The tokeniser turns rows into tokens, and the player turns tokens back into rows and then into key presses.

## 2.4 What to do about the pedal

The sustain pedal lifts the dampers, so notes keep ringing after the finger leaves the key. A pianist uses it almost constantly. There are two ways to handle it.

The first is to give the model pedal tokens, so it learns to press and release the pedal itself. This is the most faithful option, and it adds a second stream of events the model has to coordinate with the notes.

The second is to fold the pedal into the note durations: if a key is released while the pedal is down, the note is treated as lasting until the pedal comes up. The duration then describes how long the string actually sounds. This is the approach used in several published piano transcription systems, and it is the one this book takes, because it keeps every note at four tokens and needs no extra machinery.

The cost is small but real. On playback the model holds each key for the full sounding time instead of pressing the pedal, so the sympathetic resonance that a real pedal adds is missing. Chapter 23 describes how to add pedal tokens later.
