"""Read a MIDI file into a note array, and write a note array back.

Note array: one row per note, columns (onset_ms, pitch, velocity, dur_ms).
"""
# mido: reads and writes MIDI files and messages
import mido
# numpy: holds the note array and sorts it
import numpy as np


def load_notes(path, use_pedal=True):
    """MIDI file -> note array. With use_pedal, notes released while the
    sustain pedal is down keep sounding until the pedal comes up, so the
    duration is the time the string actually rings."""
    # current time in seconds as we walk through the file
    now = 0.0
    # is the sustain pedal currently pressed?
    pedal = False
    active = {}        # pitch -> (onset, velocity): key is held down
    sustained = {}     # pitch -> (onset, velocity): key up, pedal holding it
    # finished notes as (onset_s, pitch, velocity, duration_s) tuples
    notes = []

    def end(store, pitch):
        """Finish the note on `pitch` in `store` (if any) at the current time."""
        # only if that pitch is actually sounding in this store
        if pitch in store:
            # remove it and get its start time and velocity
            on, vel = store.pop(pitch)
            # record it; duration is at least 10 ms so no note has zero length
            notes.append((on, pitch, vel, max(now - on, 0.01)))

    # iterating a MidiFile merges all tracks in time order and converts ticks to seconds
    # (using the file's tempo changes).
    for msg in mido.MidiFile(path):            # msg.time is seconds since last
        # advance the clock by this message's delta time
        now += msg.time
        # key pressed (a note_on with velocity 0 means release, so it's excluded here)
        if msg.type == "note_on" and msg.velocity > 0:
            # if this pitch is still sounding (held or pedalled), end it before restarting it
            end(active, msg.note); end(sustained, msg.note)   # re-struck key
            # start the new note: remember when and how hard it was struck
            active[msg.note] = (now, msg.velocity)
        # key released: a real note_off, or (falling through from above) note_on with velocity 0
        elif msg.type in ("note_off", "note_on"):             # note_on vel 0
            # pedal down: the string keeps ringing, so move the note to `sustained` instead of
            # ending it; it will end when the pedal lifts or the key is struck again.
            if pedal and use_pedal and msg.note in active:
                sustained[msg.note] = active.pop(msg.note)
            # pedal up (or pedal ignored): the note ends now
            else:
                end(active, msg.note)
        # controller 64 is the sustain (damper) pedal
        elif msg.type == "control_change" and msg.control == 64:
            # values 64-127 mean pedal down, 0-63 mean pedal up
            pedal = msg.value >= 64
            # pedal lifted: every note it was holding stops now. Keys still held stay in `active`.
            if not pedal:
                # list(...) copies the keys because end() removes entries while we loop
                for p in list(sustained):
                    end(sustained, p)
    # end of file: close any notes still sounding at the final time
    for store in (active, sustained):
        for p in list(store):
            end(store, p)

    # make an (N, 4) float array; reshape keeps the 4 columns even when there are no notes
    arr = np.array(notes, dtype=np.float64).reshape(-1, 4)
    # convert onset (column 0) and duration (column 3) from seconds to milliseconds
    arr[:, [0, 3]] *= 1000.0                                  # seconds -> ms
    # sort rows by onset, then by pitch for notes starting together (lexsort's last key is
    # the primary one), so chords come out low to high.
    return arr[np.lexsort((arr[:, 1], arr[:, 0]))]


def save_notes(notes, path):
    """Note array -> a one-track MIDI file (120 bpm, 480 ticks per beat,
    so one tick is 1/960 s)."""
    # accept any list/array of rows; make it an (N, 4) float array
    notes = np.array(notes, dtype=np.float64).reshape(-1, 4)
    # sort by onset; "stable" keeps the original order of notes with equal onsets
    notes = notes[np.argsort(notes[:, 0], kind="stable")]
    # each note's end time in ms (onset + duration)
    end = notes[:, 0] + notes[:, 3]
    last = {}                                   # pitch -> index of its last note
    # A MIDI key can only sound once at a time, so overlapping notes on one pitch must be cut.
    for i, pitch in enumerate(notes[:, 1]):
        # previous note on the same pitch, if any
        j = last.get(pitch)
        # it is still sounding when this one starts...
        if j is not None and end[j] > notes[i, 0]:
            end[j] = notes[i, 0]                # key struck again: cut the old note
        # this note is now the latest one on its pitch
        last[pitch] = i
    # MIDI events as (time_ms, is_on, pitch, velocity).
    events = []
    # pair each note row with its (possibly shortened) end time; the original duration is unused
    for (on, pitch, vel, _), off in zip(notes, end):
        # note-on event at the onset
        events.append((on, 1, int(pitch), int(vel)))
        # note-off event at the end
        events.append((off, 0, int(pitch), 0))
    # sort by time; at equal times is_on=0 sorts first, so a cut note ends before its re-strike
    events.sort()                               # note-offs before note-ons
    # new MIDI file with 480 ticks per quarter-note beat
    mid = mido.MidiFile(ticks_per_beat=480)
    # A single track holds all the events.
    track = mido.MidiTrack()
    mid.tracks.append(track)
    # tempo in microseconds per beat: 500000 us = 0.5 s per beat = 120 bpm
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    # absolute tick time of the previous event
    last = 0
    for t_ms, is_on, pitch, vel in events:
        # 480 ticks per beat * 2 beats per second = 960 ticks per second = 0.96 ticks per ms.
        tick = int(round(t_ms * 0.96))         # 960 ticks per second
        # MIDI times are deltas from the previous event
        track.append(mido.Message("note_on" if is_on else "note_off",
                                  note=pitch, velocity=vel, time=tick - last))
        # remember this event's absolute time for the next delta
        last = tick
    # write the .mid file to disk
    mid.save(path)
