"""The generation engine: turns a model into an endless stream of notes.

A Performer does not care which model is underneath. It talks to a
"backend" with two methods:

    start(request, music_tokens) -> logits for the next token
    step(token)                  -> logits for the token after that

Logits are always a NumPy vector over OUR 459-token vocabulary, so the same
Performer drives the small MLX model (Part I) and Qwen (Part II).
"""
# NumPy: probability maths for sampling (masking, softmax, top-p) and the random generator
import numpy as np
# our tokenizer: vocabulary offsets, grammar mask, tag encoding, token families
import tokenizer as T


class MLXBackend:
    """Backend for the small model in model.py. `request` is a tags dict."""

    def __init__(self, model):
        # MLX array library, imported here so this file can be used without MLX installed
        import mlx.core as mx
        # keep MLX, the model, and its maximum context length (how many tokens it can attend to)
        self.mx, self.model, self.ctx = mx, model, model.config.ctx
        # key/value cache: the model's saved attention state for tokens already fed in
        self.cache = None

    def start(self, request, music_tokens):
        """Fresh context: tags, then the given music. Resets the key/value cache."""
        # full prompt: <bos>, tag tokens, <sep>, then any music already played
        ids = T.encode_tags(request) + list(music_tokens)
        # one full forward pass over the prompt (batch of 1); the returned cache replaces the old one
        logits, self.cache = self.model(self.mx.array([ids]))
        # scores at the last position = prediction for the next token; converted to a NumPy vector
        return np.array(logits[0, -1])

    def step(self, token):
        """Feed one token; reuses the cache, so only the new token is computed."""
        # shape [1, 1] input (one sequence, one token) plus the cache of everything before it
        logits, self.cache = self.model(self.mx.array([[token]]), self.cache)
        # scores for the token after this one, as a NumPy vector over the vocabulary
        return np.array(logits[0, -1])


class Performer:
    """Samples tokens one at a time and assembles them into notes.

    The grammar in tokenizer.allowed_next keeps every note well formed.
    With `endless`, <eos> is never sampled, and when the context fills up
    the oldest music is dropped so generation can go on forever."""

    def __init__(self, backend, request, temperature=1.0, top_p=0.95,
                 endless=True, seed=None):
        # the model wrapper to query, and the tags dict describing what to play
        self.backend, self.request = backend, request
        # sampling settings: temperature (randomness), top_p (nucleus size), never-ending mode
        self.temperature, self.top_p, self.endless = temperature, top_p, endless
        # random generator; a fixed seed makes a performance reproducible
        self.rng = np.random.default_rng(seed)
        self.music = []            # every music token still in the context
        self.n_tokens = 0          # tokens generated so far
        self.shift_ms = 0          # wait accumulated since the last note
        # feed the initial prompt to the model and get the first prediction
        self._begin()

    def _begin(self):
        """(Re)build the model's context: the request, then recent music."""
        # restart the backend with tags + remembered music; keep the next-token scores
        self.logits = self.backend.start(self.request, self.music)
        # previous token, which decides what the grammar allows next; <sep> if no music yet
        self.prev = self.music[-1] if self.music else T.SEP
        # the note being assembled from Pitch/Vel tokens (empty: none in progress)
        self.note = []

    def set_request(self, request):
        """Change what is being asked for, in the middle of a performance.
        The most recent notes stay in the context so the music carries on."""
        # store the new tags
        self.request = request
        # keep only the last quarter of the context's worth of music, so the new tags dominate
        self._trim(keep=self.backend.ctx // 4)
        # rebuild the context with the new tags in front of that music
        self._begin()

    def _trim(self, keep):
        """Keep only the last `keep` music tokens, starting on a whole note."""
        # the most recent `keep` tokens (a copy, so popping does not touch self.music yet)
        m = self.music[-keep:]
        # drop leading tokens until the slice starts at a Pitch (no half note or stray Shift at the front)
        while m and T.kind(m[0]) != "pitch":
            m.pop(0)
        # drop trailing tokens until it ends after a complete note or a wait (removes a half-built note)
        while m and T.kind(m[-1]) not in ("dur", "shift"):
            m.pop()
        # that becomes the remembered music
        self.music = m

    def _sample(self):
        """Pick the next token: grammar mask, temperature, then top-p."""
        # which tokens are grammatically allowed after the previous one
        mask = T.allowed_next(self.prev)
        # in endless mode the piece may never end, so forbid <eos>
        if self.endless:
            mask[T.EOS] = False
        # forbidden tokens get score -inf (probability 0); dividing by temperature sharpens (<1)
        # or flattens (>1) the distribution; max(..., 1e-6) avoids dividing by zero
        z = np.where(mask, self.logits.astype(np.float64), -np.inf) / max(self.temperature, 1e-6)
        # softmax, step 1: exponentiate; subtracting the max first avoids overflow
        p = np.exp(z - z.max())
        # softmax, step 2: normalise so the probabilities sum to 1
        p /= p.sum()
        # token ids from most to least likely (nucleus / top-p sampling)
        order = np.argsort(-p)
        # how many top tokens are needed for their cumulative probability to reach top_p
        # (searchsorted finds the first position where the running total hits top_p; +1 includes it)
        cut = np.searchsorted(np.cumsum(p[order]), self.top_p) + 1
        # the "nucleus": the smallest set of most likely tokens covering top_p of the probability
        keep = order[:cut]
        # draw one of them, with probabilities renormalised over the nucleus; return a plain int
        return int(self.rng.choice(keep, p=p[keep] / p[keep].sum()))

    def step(self):
        """Generate exactly one token. Returns a finished note as
        (wait_ms, pitch, velocity, dur_ms), "end" at <eos>, or None."""
        # is the context nearly full? Leave room for the tag prefix (<bos>, tags, <sep>)
        # plus a few spare tokens
        full = len(self.music) + len(T.TAG_ORDER) + 6 >= self.backend.ctx
        # only trim between two notes (after a Dur or Shift), never in the middle of one
        if full and T.kind(self.prev) in ("dur", "shift"):
            # forget the oldest half, keeping the most recent half of the context
            self._trim(keep=self.backend.ctx // 2)
            # re-run the model on the shortened context (rebuilds the cache)
            self._begin()
        # choose the next token from the current scores
        tok = self._sample()
        # count it, for the tokens-per-second report
        self.n_tokens += 1
        # the model chose to finish the piece (only possible when not endless)
        if tok == T.EOS:
            return "end"
        # remember the token as part of the context
        self.music.append(tok)
        # it is now the previous token for the grammar
        self.prev = tok
        # feed it to the model and get scores for the token after it
        self.logits = self.backend.step(tok)

        # update the note being assembled, depending on the token family
        k = T.kind(tok)
        # A Shift adds its wait (ms) to the time since the last note
        if k == "shift":
            self.shift_ms += int(T.SHIFT_GRID[tok - T.SHIFT0])
        # A Pitch starts a new note with this MIDI pitch
        elif k == "pitch":
            self.note = [tok - T.PITCH0 + T.PITCH_MIN]
        # A Vel adds the velocity (centre of the bin)
        elif k == "vel":
            self.note.append((tok - T.VEL0) * 4 + 2)
        # A Dur completes the note: package (wait before it, pitch, velocity, duration in ms)
        elif k == "dur":
            out = (self.shift_ms, self.note[0], self.note[1], int(T.DUR_GRID[tok - T.DUR0]))
            # the next note's wait is counted from this note's onset
            self.shift_ms = 0
            # hand the finished note back to the caller
            return out
        # no note finished on this token
        return None

    def notes(self, n):
        """Generate n notes (fewer if the piece ends) as a note array."""
        # finished notes with absolute onsets; running clock in ms
        out, now = [], 0
        # keep generating tokens until we have n notes
        while len(out) < n:
            # one token; may or may not complete a note
            r = self.step()
            # the piece ended early
            if r == "end":
                break
            # A note was completed (None means not yet)
            if r:
                # turn the relative wait into an absolute onset time
                now += r[0]
                # store as (onset_ms, pitch, velocity, dur_ms), the tokenizer's note format
                out.append((now, r[1], r[2], r[3]))
        # as a float array of shape [n_notes, 4]
        return np.array(out, dtype=np.float64).reshape(-1, 4)
