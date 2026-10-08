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
        """Finish the note on `pitch` in `store` (if any) at the current time."""
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
        tick = int(round(t_ms * 0.96))         # 960 ticks per second
        # MIDI times are deltas from the previous event
        track.append(mido.Message("note_on" if is_on else "note_off",
                                  note=pitch, velocity=vel, time=tick - last))
        last = tick
    mid.save(path)
