"""The generation engine: turns a model into an endless stream of notes.

A Performer does not care which model is underneath. It talks to a
"backend" with two methods:

    start(request, music_tokens) -> logits for the next token
    step(token)                  -> logits for the token after that

Logits are always a NumPy vector over OUR 465-token vocabulary, so the same
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
        """Fresh context: tags, then the given music. Resets the key/value cache."""
        ids = T.encode_tags(request) + list(music_tokens)
        logits, self.cache = self.model(self.mx.array([ids]))   # one full pass
        return np.array(logits[0, -1])

    def step(self, token):
        """Feed one token; reuses the cache, so only the new token is computed."""
        logits, self.cache = self.model(self.mx.array([[token]]), self.cache)
        return np.array(logits[0, -1])


class Performer:
    """Samples tokens one at a time and assembles them into notes.

    The grammar in tokenizer.allowed_next keeps every note well formed.
    With `endless`, <eos> is never sampled, and when the context fills up
    the oldest music is dropped so generation can go on forever."""

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
        self.prev = self.music[-1] if self.music else T.SEP   # for the grammar
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
        """Pick the next token: grammar mask, temperature, then top-p."""
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
        # room for the tag prefix (<bos>, tags, <sep>) plus a few spare tokens
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
            self.note.append((tok - T.VEL0) * 4 + 2)       # centre of the bin
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
