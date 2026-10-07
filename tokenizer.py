"""Performance tokeniser: notes <-> integer tokens.

A note is four numbers: onset time (ms), pitch (21-108), velocity (1-127)
and duration (ms). The token stream for a note is

    [Shift ...]  Pitch  Vel  Dur

where the optional Shift tokens say how long to wait since the previous
onset. Notes that start together (a chord) have no Shift between them.
"""
# NumPy: vectorised maths on note arrays (rounding, sorting, binning, masks)
import numpy as np

# ---------------------------------------------------------------- vocabulary
# Control tokens: padding (fills short training sequences), begin-of-sequence,
# end-of-sequence (piece is over) and separator (end of the tag prefix, music follows)
SPECIALS = ["<pad>", "<bos>", "<eos>", "<sep>"]
# their integer ids, which are simply their positions at the start of VOCAB
PAD, BOS, EOS, SEP = 0, 1, 2, 3

# the 24 musical keys, e.g. "Cmaj", "Dbmaj", ..., "Bmin": every note name with "maj", then with "min"
KEYS = [f"{n}{m}" for m in ("maj", "min")
        for n in ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")]
# conditioning tags: for each tag category, the values it can take. A request such as
# "slow and quiet in D minor" becomes a few of these, placed before the music
TAG_VALUES = {
    # how many notes per second, from very few to very many
    "density":  ["very_sparse", "sparse", "medium", "dense", "very_dense"],
    # overall loudness, using musical dynamics marks (pianissimo ... fortissimo)
    "dynamics": ["pp", "p", "mf", "f", "ff"],
    # which part of the keyboard the music mostly sits in
    "register": ["low", "mid", "high"],
    # the musical key (one of the 24 above)
    "key":      KEYS,
    # stylistic period of the piece
    "era":      ["baroque", "classical", "romantic", "modern"],
    # composer of the piece (the ones well represented in MAESTRO)
    "composer": ["bach", "haydn", "mozart", "beethoven", "schubert", "chopin",
                 "schumann", "liszt", "mendelssohn", "brahms", "rachmaninoff",
                 "scriabin", "debussy", "ravel"],
}
# category names in dict order ("density", "dynamics", ...); tags always appear in this order
TAG_ORDER = list(TAG_VALUES)
# one token name per (category, value) pair, e.g. "<key:Cmaj>", "<dynamics:p>"
TAGS = [f"<{k}:{v}>" for k in TAG_ORDER for v in TAG_VALUES[k]]

# lowest and highest MIDI pitch: A0 (21) to C8 (108), the 88 keys of a piano
PITCH_MIN, PITCH_MAX = 21, 108
# velocity (loudness 1-127) is quantised into 32 bins, 4 MIDI steps each
N_VEL = 32
# time resolution: all onsets are rounded to multiples of 10 ms
STEP_MS = 10
# possible single waits: 100 values, 10 ms ... 1 s (longer waits chain several Shifts)
SHIFT_GRID = np.arange(10, 1001, 10)
# possible note durations: fine steps for short notes, coarser for long ones (180 values)
DUR_GRID = np.concatenate([
    # 10 ms steps up to 1 s (100 values)
    np.arange(10, 1001, 10),
    # 50 ms steps from 1.05 s up to 4 s (60 values)
    np.arange(1050, 4001, 50),
    # 200 ms steps from 4.2 s up to 8 s (20 values)
    np.arange(4200, 8001, 200),
])

# the full vocabulary: list index = token id. Blocks are laid out in this order,
# which the *0 offsets below and kind() rely on
VOCAB = (SPECIALS + TAGS
         # 88 pitch tokens: "Pitch_21" ... "Pitch_108"
         + [f"Pitch_{p}" for p in range(PITCH_MIN, PITCH_MAX + 1)]
         # 32 velocity tokens, named after the centre of their bin: "Vel_2", "Vel_6", ... "Vel_126"
         + [f"Vel_{v * 4 + 2}" for v in range(N_VEL)]
         # 180 duration tokens: "Dur_10" ... "Dur_8000" (ms)
         + [f"Dur_{d}" for d in DUR_GRID]
         # 100 time-shift tokens: "Shift_10" ... "Shift_1000" (Shift_380 = wait 380 ms)
         + [f"Shift_{s}" for s in SHIFT_GRID])
# reverse lookup: token name -> token id, e.g. TOKEN_ID["<key:Cmaj>"]
TOKEN_ID = {name: i for i, name in enumerate(VOCAB)}
# total number of distinct tokens (459); the model's output layer has this many scores
VOCAB_SIZE = len(VOCAB)

# token id where the tag block starts (right after the 4 specials)
TAG0 = len(SPECIALS)
# token id of "Pitch_21", the first pitch token
PITCH0 = TAG0 + len(TAGS)
# token id of the first velocity token (after the 88 pitches)
VEL0 = PITCH0 + (PITCH_MAX - PITCH_MIN + 1)
# token id of the first duration token (after the 32 velocities)
DUR0 = VEL0 + N_VEL
# token id of the first time-shift token (after the 180 durations); shifts run to the end
SHIFT0 = DUR0 + len(DUR_GRID)
# every id >= MUSIC0 is a music token (pitch, velocity, duration or shift)
MUSIC0 = PITCH0

# midpoints between neighbouring durations: searchsorted on them rounds a raw
# duration to the index of the nearest DUR_GRID value (e.g. 14 ms -> 10, 16 ms -> 20)
_DUR_EDGES = (DUR_GRID[:-1] + DUR_GRID[1:]) / 2


def kind(tok):
    """Which family a token id belongs to."""
    # check block boundaries from the top down; the first start offset the id reaches wins
    if tok >= SHIFT0: return "shift"
    if tok >= DUR0:   return "dur"
    if tok >= VEL0:   return "vel"
    if tok >= PITCH0: return "pitch"
    if tok >= TAG0:   return "tag"
    # anything below TAG0 is one of <pad>, <bos>, <eos>, <sep>
    return "special"


# ------------------------------------------------------------------ encoding
def encode_tags(tags):
    """{'key': 'Cmaj', 'dynamics': 'p'} -> [BOS, tag ids..., SEP]."""
    # look up the id of each tag that is set, in the fixed TAG_ORDER; missing or empty ones are skipped
    ids = [TOKEN_ID[f"<{k}:{tags[k]}>"] for k in TAG_ORDER if tags.get(k)]
    # wrap them: <bos> opens the sequence, <sep> marks where the music begins
    return [BOS] + ids + [SEP]


def encode_notes(notes):
    """notes: array of rows (onset_ms, pitch, velocity, dur_ms) -> token ids."""
    # accept a list of tuples or an array; force shape [n_notes, 4] (works for zero notes too)
    notes = np.asarray(notes, dtype=np.float64).reshape(-1, 4)
    # onset times rounded to whole 10 ms steps (integers)
    onset = np.round(notes[:, 0] / STEP_MS).astype(int)
    # sort indices by onset, then pitch (lexsort uses the LAST key as primary), so chords go low to high
    order = np.lexsort((notes[:, 1], onset))
    # reorder the notes and their rounded onsets into that order
    notes, onset = notes[order], onset[order]
    # pitch clamped to the piano range, then shifted so Pitch_21 -> 0 (an offset from PITCH0)
    pitch = np.clip(notes[:, 1], PITCH_MIN, PITCH_MAX).astype(int) - PITCH_MIN
    # velocity clamped to 1-127 and integer-divided by 4 into one of 32 bins (0-31)
    vel = np.clip(notes[:, 2], 1, 127).astype(int) // 4
    # index of the nearest DUR_GRID value for each duration (beyond 8 s clamps to the last one)
    dur = np.searchsorted(_DUR_EDGES, notes[:, 3])

    # output token list, and the "clock" (in 10 ms steps) at the last emitted onset.
    # Start the clock at the first onset, so the stream does not open with a wait
    out, now = [], onset[0] if len(onset) else 0
    # emit tokens note by note
    for i in range(len(notes)):
        # how many 10 ms steps to wait since the previous note's onset (0 within a chord)
        gap = int(onset[i] - now)
        # A wait longer than 1 s is spelled as several Shift tokens chained together
        while gap > 0:
            # take as much of the gap as one Shift can hold (at most 100 steps = 1 s)
            s = min(gap, len(SHIFT_GRID))
            # shift of s * 10 ms: Shift_10 is SHIFT0, so s steps is SHIFT0 + s - 1
            out.append(SHIFT0 + s - 1)
            # whatever is left still has to be waited
            gap -= s
        # the clock is now at this note's onset
        now = onset[i]
        # the note itself: its Pitch, Vel and Dur tokens (block start + offset within the block)
        out += [PITCH0 + int(pitch[i]), VEL0 + int(vel[i]), DUR0 + int(dur[i])]
    # the full token id list for the notes (no tags, no <bos>/<sep>)
    return out


# ------------------------------------------------------------------ decoding
def decode_notes(tokens):
    """Token ids -> array of rows (onset_ms, pitch, velocity, dur_ms).

    Ignores specials and tags, and drops a trailing incomplete note."""
    # finished notes; current time in ms; the note being assembled ([pitch] then [pitch, vel])
    notes, now, cur = [], 0, []
    # walk the tokens like a small state machine
    for t in tokens:
        # which family this token is from
        k = kind(t)
        # A Shift advances the clock by its value in ms; drop a half-built note
        if k == "shift":
            now += int(SHIFT_GRID[t - SHIFT0]); cur = []
        # A Pitch starts a new note (convert the id back to a MIDI pitch number)
        elif k == "pitch":
            cur = [t - PITCH0 + PITCH_MIN]
        # A Vel is only valid right after a Pitch; use the centre of the bin (bin 20 -> 82)
        elif k == "vel" and len(cur) == 1:
            cur.append((t - VEL0) * 4 + 2)
        # A Dur after Pitch+Vel completes the note: record it at the current time, then reset
        elif k == "dur" and len(cur) == 2:
            notes.append((now, cur[0], cur[1], int(DUR_GRID[t - DUR0]))); cur = []
    # as a float array of shape [n_notes, 4] (shape [0, 4] if there were no notes)
    return np.array(notes, dtype=np.float64).reshape(-1, 4)


def describe(tokens):
    """Human-readable token names, for printing."""
    # map each id to its name and join with spaces, e.g. "<bos> <sep> Pitch_67 Vel_82 Dur_380"
    return " ".join(VOCAB[t] for t in tokens)


# ------------------------------------------------------------------- grammar
def allowed_next(prev):
    """Boolean mask over the vocabulary: which tokens may follow `prev`.

    Used at generation time so the model can never emit a malformed note."""
    # start with nothing allowed: one True/False per vocabulary entry
    m = np.zeros(VOCAB_SIZE, dtype=bool)
    # the rule depends only on the family of the previous token
    k = kind(prev)
    # after a Pitch the only valid continuation is a Vel
    if k == "pitch":   m[VEL0:DUR0] = True
    # after a Vel, a Dur must follow
    elif k == "vel":   m[DUR0:SHIFT0] = True
    # after a Dur the note is complete: start another note (chord), wait (Shift), or end the piece
    elif k == "dur":   m[PITCH0:VEL0] = True; m[SHIFT0:] = True; m[EOS] = True
    # after a Shift: another Shift (longer wait) or a new note; no ending right after a wait
    elif k == "shift": m[PITCH0:VEL0] = True; m[SHIFT0:] = True
    # right after <sep> (or any other non-music token) the music must open with a Pitch
    else:              m[PITCH0:VEL0] = True
    # mask that the sampler uses to rule out everything else
    return m


# run only when this file is executed directly (python tokenizer.py), not when imported
if __name__ == "__main__":
    # demo: encode a short tune, print the tokens, and decode it back
    print("vocabulary size:", VOCAB_SIZE)
    # the opening of "Happy Birthday": onset, pitch, velocity, duration
    hb = [(0, 67, 80, 375), (375, 67, 72, 125), (500, 69, 84, 500),
          (1000, 67, 84, 500), (1500, 72, 92, 500), (2000, 71, 88, 1000)]
    # tag prefix (<bos> <key:Cmaj> <dynamics:f> <sep>) followed by the encoded notes
    ids = encode_tags({"key": "Cmaj", "dynamics": "f"}) + encode_notes(hb)
    # the token names, to see the Shift/Pitch/Vel/Dur pattern
    print(describe(ids))
    # the raw integer ids, as the model sees them
    print(ids)
    # decode back to notes: should match hb up to rounding (10 ms grid, velocity bins)
    print(decode_notes(ids))
