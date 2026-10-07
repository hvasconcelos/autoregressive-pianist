# 12. Generating and listening

## 12.1 The generation engine

Everything that produces notes goes through one class, `Performer`. It samples tokens, enforces the grammar, assembles tokens into notes and manages the model's limited memory.

::: filename
performer.py
:::

``` python
"""The generation engine: turns a model into an endless stream of notes.

A Performer does not care which model is underneath. It talks to a
"backend" with two methods:

    start(request, music_tokens) -> logits for the next token
    step(token)                  -> logits for the token after that

Logits are always a NumPy vector over OUR 459-token vocabulary, so the same
Performer drives the small MLX model (Part I) and Qwen (Part II).
"""
import numpy as np
import tokenizer as T


class MLXBackend:
    """Backend for the small model in model.py. `request` is a tags dict."""

    def __init__(self, model):
        import mlx.core as mx
        self.mx, self.model, self.ctx = mx, model, model.config.ctx
        self.cache = None

    def start(self, request, music_tokens):
        ids = T.encode_tags(request) + list(music_tokens)
        logits, self.cache = self.model(self.mx.array([ids]))   # one full pass
        return np.array(logits[0, -1])

    def step(self, token):
        logits, self.cache = self.model(self.mx.array([[token]]), self.cache)
        return np.array(logits[0, -1])


class Performer:
    def __init__(self, backend, request, temperature=1.0, top_p=0.95,
                 endless=True, seed=None):
        self.backend, self.request = backend, request
        self.temperature, self.top_p, self.endless = temperature, top_p, endless
        self.rng = np.random.default_rng(seed)
        self.music = []            # every music token still in the context
        self.n_tokens = 0          # tokens generated so far
        self.shift_ms = 0          # wait accumulated since the last note
        self._begin()

    def _begin(self):
        """(Re)build the model's context: the request, then recent music."""
        self.logits = self.backend.start(self.request, self.music)
        self.prev = self.music[-1] if self.music else T.SEP
        self.note = []

    def set_request(self, request):
        """Change what is being asked for, in the middle of a performance.
        The most recent notes stay in the context so the music carries on."""
        self.request = request
        self._trim(keep=self.backend.ctx // 4)
        self._begin()

    def _trim(self, keep):
        """Keep only the last `keep` music tokens, starting on a whole note."""
        m = self.music[-keep:]
        while m and T.kind(m[0]) != "pitch":
            m.pop(0)
        while m and T.kind(m[-1]) not in ("dur", "shift"):
            m.pop()
        self.music = m

    def _sample(self):
        mask = T.allowed_next(self.prev)
        if self.endless:
            mask[T.EOS] = False
        z = np.where(mask, self.logits.astype(np.float64), -np.inf) / max(self.temperature, 1e-6)
        p = np.exp(z - z.max())
        p /= p.sum()
        order = np.argsort(-p)                      # nucleus (top-p) sampling
        cut = np.searchsorted(np.cumsum(p[order]), self.top_p) + 1
        keep = order[:cut]
        return int(self.rng.choice(keep, p=p[keep] / p[keep].sum()))

    def step(self):
        """Generate exactly one token. Returns a finished note as
        (wait_ms, pitch, velocity, dur_ms), "end" at <eos>, or None."""
        full = len(self.music) + len(T.TAG_ORDER) + 6 >= self.backend.ctx
        if full and T.kind(self.prev) in ("dur", "shift"):  # between two notes
            self._trim(keep=self.backend.ctx // 2)          # forget the oldest half
            self._begin()
        tok = self._sample()
        self.n_tokens += 1
        if tok == T.EOS:
            return "end"
        self.music.append(tok)
        self.prev = tok
        self.logits = self.backend.step(tok)

        k = T.kind(tok)
        if k == "shift":
            self.shift_ms += int(T.SHIFT_GRID[tok - T.SHIFT0])
        elif k == "pitch":
            self.note = [tok - T.PITCH0 + T.PITCH_MIN]
        elif k == "vel":
            self.note.append((tok - T.VEL0) * 4 + 2)
        elif k == "dur":
            out = (self.shift_ms, self.note[0], self.note[1], int(T.DUR_GRID[tok - T.DUR0]))
            self.shift_ms = 0
            return out
        return None

    def notes(self, n):
        """Generate n notes (fewer if the piece ends) as a note array."""
        out, now = [], 0
        while len(out) < n:
            r = self.step()
            if r == "end":
                break
            if r:
                now += r[0]
                out.append((now, r[1], r[2], r[3]))
        return np.array(out, dtype=np.float64).reshape(-1, 4)
```

`MLXBackend` is the thin layer between the engine and the model. `start` runs the request and any existing music through the model in one pass, which fills the key/value cache. `step` then feeds one token at a time.

`Performer.step` is the heart of the system. Each call generates exactly one token, and returns a finished note whenever that token was a Dur. Generating one token per call, and not one note per call, keeps each call short, so the player in the next chapter can look at its buffer and check for a new request between tokens.

`_sample` applies, in order, the grammar mask, the temperature and the top-p cut described in chapter 3.

## 12.2 An endless performance with a finite memory

The model was trained on windows of 1,024 tokens. A performance lasts much longer than that, so the window has to move.

When the window is nearly full, and the model is between two notes, `step` does the following.

1.  It throws away the oldest half of the music tokens, trimming so that what remains begins on a whole note.
2.  It rebuilds the context as the original request followed by the remaining music.
3.  It runs that context through the model in one pass to rebuild the cache.

The request is put back at the front every time. This is the important detail. If the window simply slid forward, the tags would eventually fall out of it and the model would forget what it had been asked for. Pinning them keeps the request in force for the whole performance.

The rebuild is a single pass over about 500 tokens. The real-time player in the next chapter sends notes from a separate thread, so notes already in its buffer keep playing on time while the rebuild runs.

The same mechanism handles a change of request. `set_request` keeps the last quarter-window of music, swaps the tags and rebuilds. The model then continues from what it was just playing, under the new instructions, so the music changes over the following bars instead of jumping.

## 12.3 Generate a file

::: filename
sample.py
:::

``` python
"""Generate a performance into a MIDI file (not in real time).

    python sample.py --model runs/v1/best --request "slow and quiet in D minor" --out out.mid
"""
import argparse, time
from model import Pianist
from performer import MLXBackend, Performer
from midi_io import save_notes
from request import parse_request, parse_tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/v1/best")
    ap.add_argument("--request", default="", help="plain words")
    ap.add_argument("--tags", default="", help="exact tags, e.g. key=Dmin,dynamics=p")
    ap.add_argument("--notes", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="out.mid")
    args = ap.parse_args()

    tags = parse_tags(args.tags) if args.tags else parse_request(args.request)
    print("tags:", tags)
    model, info = Pianist.load(args.model)
    p = Performer(MLXBackend(model), tags, args.temperature, args.top_p, seed=args.seed)
    t0 = time.time()
    notes = p.notes(args.notes)
    dt = time.time() - t0
    save_notes(notes, args.out)
    music_s = (notes[-1, 0] + notes[-1, 3]) / 1000
    print(f"{len(notes)} notes, {music_s:.1f} s of music -> {args.out}")
    print(f"generated {p.n_tokens / dt:.0f} tokens/s; "
          f"this music needs {p.n_tokens / music_s:.0f} tokens/s")


if __name__ == "__main__":
    main()
```

``` bash
python sample.py --model runs/v1/best --request "slow and quiet in D minor" --out slow.mid
python sample.py --model runs/v1/best --tags density=dense,dynamics=ff,key=Cmaj --out loud.mid
```

The script prints the tags it used, the number of notes, and two speeds:

``` text
tags: {'density': 'sparse', 'dynamics': 'p', 'key': 'Dmin'}
400 notes, ... s of music -> slow.mid
generated ... tokens/s; this music needs ... tokens/s
```

The last line is the real-time budget. The first number is how fast your Mac generates. The second is how fast this particular music consumes tokens. Real-time playing needs the first to be comfortably larger than the second, and for a model this small it should be larger by a wide margin.

## 12.4 Listening critically

Generate several files with different requests and seeds, and listen for specific things.

| Listen for | Good sign | Bad sign and likely cause |
|----|----|----|
| Harmony | Chords make sense for a few bars at a time | Random-sounding clusters: under-trained, or temperature too high |
| Touch | Melody stands out from accompaniment; dynamics rise and fall | Every note the same loudness: velocity handling is broken somewhere |
| Timing | Slight, natural unevenness | Machine-gun regularity, or notes piling up: check the Shift tokens |
| Request | Quiet requests sound quiet, dense ones busy | No difference between requests: see chapter 15 |
| Repetition | Ideas return with variation | The same bar looping for ever: temperature too low |

Then adjust the two sampling settings and listen again:

- `--temperature 0.9` makes the playing more conservative and more accurate.
- `--temperature 1.1` makes it more adventurous and more error-prone.
- `--top-p 0.9` removes more of the unlikely choices.
