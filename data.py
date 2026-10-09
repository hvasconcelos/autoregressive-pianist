"""Batches for training: random passages, augmented, tagged and tokenised."""
import json
import numpy as np
import tokenizer as T
from tags import compute_tags


def drop_tags(tags, rng):
    """Randomly remove tags, so the model learns to cope with partial requests.

    One example in ten keeps no tags at all. For the rest, each tag is
    removed with a probability that is itself random, between 0 and 0.8, so
    training sees everything from complete requests to a single tag."""
    if rng.random() < 0.1:
        return {}
    p = rng.uniform(0.0, 0.8)
    return {k: v for k, v in tags.items() if rng.random() >= p}


class Dataset:
    """One or more splits written by prepare.py / prepare_aria.py, as one array.

    Piece i is notes[offsets[i]:offsets[i + 1]]; meta[i] holds its fixed tags
    (genre, era, composer). `paths` is one .npz path or a list of them."""

    def __init__(self, paths):
        notes, sizes, self.meta = [], [], []
        for path in [paths] if isinstance(paths, str) else paths:
            z = np.load(path)
            notes.append(z["notes"].astype(np.float32))       # float32 halves the RAM
            sizes.append(np.diff(z["offsets"]))
            self.meta += json.loads(str(z["meta"]))
        self.notes = np.concatenate(notes)
        self.sizes = np.concatenate(sizes)    # notes per piece
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])

    def __len__(self):
        return len(self.sizes)

    def piece(self, i):
        """Note array of piece i."""
        return self.notes[self.offsets[i]:self.offsets[i + 1]]

    def passage(self, rng, budget, augment=True, tag_dropout=True):
        """A random passage whose music fits in `budget` tokens.

        Returns (tags, music token ids). The ids end with <eos> when the
        passage runs to the end of its piece."""
        # longer pieces are picked more often, so every note is equally likely
        i = rng.choice(len(self), p=self.sizes / self.sizes.sum())
        piece = self.piece(i)
        n = budget // 3                            # a note is at least 3 tokens
        last_start = max(len(piece) - n, 0)
        start = last_start if rng.random() < 0.03 else rng.integers(0, last_start + 1)
        at_end = start == last_start               # 3 % of passages are endings
        notes = piece[start:start + n].astype(np.float64)
        if augment:                                # small random variations
            lo, hi = notes[:, 1].min(), notes[:, 1].max()
            # up to 3 semitones either way, staying on the keyboard
            shift = rng.integers(max(-3, T.PITCH_MIN - lo), min(3, T.PITCH_MAX - hi) + 1)
            notes[:, 1] += shift                               # transpose
            notes[:, [0, 3]] *= rng.uniform(0.9, 1.1)          # faster / slower
            notes[:, 2] = np.clip(notes[:, 2] + rng.integers(-6, 7), 1, 127)

        music = T.encode_notes(notes)
        room = budget - 1 if at_end else budget    # an ending needs room for <eos>
        if len(music) > room:
            if at_end:                             # keep the last notes
                music = music[-room:]
                while T.kind(music[0]) != "pitch":
                    music.pop(0)
            else:                                  # keep the first notes
                music = music[:room]
                while T.kind(music[-1]) != "dur":
                    music.pop()
            n_kept = sum(T.kind(t) == "pitch" for t in music)
            notes = notes[-n_kept:] if at_end else notes[:n_kept]

        tags = compute_tags(notes)                 # measured on what is kept
        tags.update({k: self.meta[i][k] for k in ("genre", "era", "composer") if k in self.meta[i]})
        if tag_dropout:
            tags = drop_tags(tags, rng)
        return tags, music + ([T.EOS] if at_end else [])

    def batch(self, rng, batch_size, ctx, **kw):
        """(inputs, targets, mask) as int32/float32 arrays of shape [B, ctx-1].

        The mask is 1 where the target is a music token (or <eos>): the loss
        ignores padding and does not ask the model to predict the tags."""
        x = np.full((batch_size, ctx), T.PAD, dtype=np.int32)
        prefix = len(T.TAG_ORDER) + 2              # <bos>, one tag per category, <sep>
        for b in range(batch_size):
            tags, music = self.passage(rng, ctx - prefix, **kw)
            ids = T.encode_tags(tags) + music
            x[b, :len(ids)] = ids                  # the rest stays <pad>
        inputs, targets = x[:, :-1], x[:, 1:]       # predict the next token
        mask = ((targets >= T.MUSIC0) | (targets == T.EOS)).astype(np.float32)
        return inputs, targets, mask


def fixed_batches(ds, n_batches, batch_size, ctx, seed=0):
    """The same un-augmented, fully tagged batches every time, for validation."""
    rng = np.random.default_rng(seed)
    return [ds.batch(rng, batch_size, ctx, augment=False, tag_dropout=False)
            for _ in range(n_batches)]
