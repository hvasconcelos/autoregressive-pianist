"""Performance tokeniser: notes <-> integer tokens.

A note is four numbers: onset time (ms), pitch (21-108), velocity (1-127)
and duration (ms). The token stream for a note is

    [Shift ...]  Pitch  Vel  Dur

where the optional Shift tokens say how long to wait since the previous
onset. Notes that start together (a chord) have no Shift between them.
"""
import numpy as np

# ---------------------------------------------------------------- vocabulary
SPECIALS = ["<pad>", "<bos>", "<eos>", "<sep>"]
PAD, BOS, EOS, SEP = 0, 1, 2, 3

KEYS = [f"{n}{m}" for m in ("maj", "min")
        for n in ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")]
TAG_VALUES = {
    "density":  ["very_sparse", "sparse", "medium", "dense", "very_dense"],
    "dynamics": ["pp", "p", "mf", "f", "ff"],
    "register": ["low", "mid", "high"],
    "key":      KEYS,
    "era":      ["baroque", "classical", "romantic", "modern"],
    "composer": ["bach", "haydn", "mozart", "beethoven", "schubert", "chopin",
                 "schumann", "liszt", "mendelssohn", "brahms", "rachmaninoff",
                 "scriabin", "debussy", "ravel"],
    # MAESTRO is all classical; the rest come from Aria-MIDI's metadata (tags.aria_tags)
    "genre":    ["classical", "jazz", "pop", "film", "ragtime", "other"],
}
TAG_ORDER = list(TAG_VALUES)            # tags always appear in this order
TAGS = [f"<{k}:{v}>" for k in TAG_ORDER for v in TAG_VALUES[k]]

PITCH_MIN, PITCH_MAX = 21, 108          # the 88 keys of a piano
N_VEL = 32                              # velocity bins, 4 MIDI steps each
STEP_MS = 10                            # time resolution
SHIFT_GRID = np.arange(10, 1001, 10)    # 100 values: 10 ms ... 1 s
DUR_GRID = np.concatenate([
    np.arange(10, 1001, 10),            # 10 ms steps up to 1 s
    np.arange(1050, 4001, 50),          # 50 ms steps up to 4 s
    np.arange(4200, 8001, 200),         # 200 ms steps up to 8 s
])                                      # 180 values

VOCAB = (SPECIALS + TAGS
         + [f"Pitch_{p}" for p in range(PITCH_MIN, PITCH_MAX + 1)]
         + [f"Vel_{v * 4 + 2}" for v in range(N_VEL)]
         + [f"Dur_{d}" for d in DUR_GRID]
         + [f"Shift_{s}" for s in SHIFT_GRID])
TOKEN_ID = {name: i for i, name in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

TAG0 = len(SPECIALS)
PITCH0 = TAG0 + len(TAGS)
VEL0 = PITCH0 + (PITCH_MAX - PITCH_MIN + 1)
DUR0 = VEL0 + N_VEL
SHIFT0 = DUR0 + len(DUR_GRID)
MUSIC0 = PITCH0                         # every id >= MUSIC0 is a music token

# midpoints between grid values: searchsorted on them rounds to the nearest duration
_DUR_EDGES = (DUR_GRID[:-1] + DUR_GRID[1:]) / 2


def kind(tok):
    """Which family a token id belongs to."""
    if tok >= SHIFT0: return "shift"
    if tok >= DUR0:   return "dur"
    if tok >= VEL0:   return "vel"
    if tok >= PITCH0: return "pitch"
    if tok >= TAG0:   return "tag"
    return "special"


# ------------------------------------------------------------------ encoding
def encode_tags(tags):
    """{'key': 'Cmaj', 'dynamics': 'p'} -> [BOS, tag ids..., SEP]."""
    ids = [TOKEN_ID[f"<{k}:{tags[k]}>"] for k in TAG_ORDER if tags.get(k)]
    return [BOS] + ids + [SEP]


def encode_notes(notes):
    """notes: array of rows (onset_ms, pitch, velocity, dur_ms) -> token ids."""
    notes = np.asarray(notes, dtype=np.float64).reshape(-1, 4)
    onset = np.round(notes[:, 0] / STEP_MS).astype(int)     # in 10 ms steps
    order = np.lexsort((notes[:, 1], onset))                # by onset, then pitch
    notes, onset = notes[order], onset[order]
    pitch = np.clip(notes[:, 1], PITCH_MIN, PITCH_MAX).astype(int) - PITCH_MIN
    vel = np.clip(notes[:, 2], 1, 127).astype(int) // 4   # 32 bins of 4
    dur = np.searchsorted(_DUR_EDGES, notes[:, 3])          # nearest DUR_GRID index

    # start the clock at the first onset, so the stream does not open with a wait
    out, now = [], onset[0] if len(onset) else 0
    for i in range(len(notes)):
        gap = int(onset[i] - now)
        while gap > 0:                                      # long waits chain
            s = min(gap, len(SHIFT_GRID))
            out.append(SHIFT0 + s - 1)                      # Shift of s * 10 ms
            gap -= s
        now = onset[i]
        out += [PITCH0 + int(pitch[i]), VEL0 + int(vel[i]), DUR0 + int(dur[i])]
    return out


# ------------------------------------------------------------------ decoding
def decode_notes(tokens):
    """Token ids -> array of rows (onset_ms, pitch, velocity, dur_ms).

    Ignores specials and tags, and drops a trailing incomplete note."""
    notes, now, cur = [], 0, []
    for t in tokens:
        k = kind(t)
        if k == "shift":
            now += int(SHIFT_GRID[t - SHIFT0]); cur = []    # drop a half-built note
        elif k == "pitch":
            cur = [t - PITCH0 + PITCH_MIN]
        elif k == "vel" and len(cur) == 1:
            cur.append((t - VEL0) * 4 + 2)                  # centre of the bin
        elif k == "dur" and len(cur) == 2:
            notes.append((now, cur[0], cur[1], int(DUR_GRID[t - DUR0]))); cur = []
    return np.array(notes, dtype=np.float64).reshape(-1, 4)


def describe(tokens):
    """Human-readable token names, for printing."""
    return " ".join(VOCAB[t] for t in tokens)


# ------------------------------------------------------------------- grammar
def allowed_next(prev):
    """Boolean mask over the vocabulary: which tokens may follow `prev`.

    Used at generation time so the model can never emit a malformed note."""
    m = np.zeros(VOCAB_SIZE, dtype=bool)
    k = kind(prev)
    if k == "pitch":   m[VEL0:DUR0] = True
    elif k == "vel":   m[DUR0:SHIFT0] = True
    elif k == "dur":   m[PITCH0:VEL0] = True; m[SHIFT0:] = True; m[EOS] = True
    elif k == "shift": m[PITCH0:VEL0] = True; m[SHIFT0:] = True
    else:              m[PITCH0:VEL0] = True      # right after <sep>
    return m


if __name__ == "__main__":
    # demo: encode a short tune, print the tokens, and decode it back
    print("vocabulary size:", VOCAB_SIZE)
    # the opening of "Happy Birthday": onset, pitch, velocity, duration
    hb = [(0, 67, 80, 375), (375, 67, 72, 125), (500, 69, 84, 500),
          (1000, 67, 84, 500), (1500, 72, 92, 500), (2000, 71, 88, 1000)]
    ids = encode_tags({"key": "Cmaj", "dynamics": "f"}) + encode_notes(hb)
    print(describe(ids))
    print(ids)
    print(decode_notes(ids))
