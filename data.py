"""Batches for training: random passages, augmented, tagged and tokenised."""
# json: decodes the per-piece metadata (composer/era tags) stored as a JSON string in the .npz
import json
# numpy: holds the note arrays and does all random sampling and batch assembly
import numpy as np
# tokenizer: turns notes and tags into integer token ids, and knows the vocabulary layout
import tokenizer as T
# compute_tags: measures density/dynamics/register/key tags from a passage's notes
from tags import compute_tags


def drop_tags(tags, rng):
    """Randomly remove tags, so the model learns to cope with partial requests.

    One example in ten keeps no tags at all. For the rest, each tag is
    removed with a probability that is itself random, between 0 and 0.8, so
    training sees everything from complete requests to a single tag."""
    # 10 % of the time, return an empty request: the model must also play with no guidance
    if rng.random() < 0.1:
        # no tags at all
        return {}
    # pick this example's drop probability, anywhere from "keep all" (0) to "drop most" (0.8)
    p = rng.uniform(0.0, 0.8)
    # keep each tag independently when its coin flip lands at or above p (so it survives with prob 1-p)
    return {k: v for k, v in tags.items() if rng.random() >= p}


class Dataset:
    """One split written by prepare.py: every note of every piece in one array.

    Piece i is notes[offsets[i]:offsets[i + 1]]; meta[i] holds its composer tags."""

    def __init__(self, path):
        # open the .npz archive written by prepare.py (a zip of named numpy arrays)
        z = np.load(path)
        # all notes of all pieces stacked into one [N, 4] array of (onset_ms, pitch, velocity, dur_ms);
        # converted to float64 so augmentation (time stretching) can produce fractional values
        self.notes = z["notes"].astype(np.float64)
        # start index of each piece in self.notes, plus a final end index (so len = pieces + 1)
        self.offsets = z["offsets"]
        # list of dicts, one per piece, e.g. {"composer": "chopin", "era": "romantic"}
        self.meta = json.loads(str(z["meta"]))
        # number of notes in each piece = difference between consecutive offsets
        self.sizes = np.diff(self.offsets)    # notes per piece

    def __len__(self):
        # the dataset's length is the number of pieces
        return len(self.sizes)

    def piece(self, i):
        """Note array of piece i."""
        # slice piece i's notes out of the big array (a view, not a copy)
        return self.notes[self.offsets[i]:self.offsets[i + 1]]

    def passage(self, rng, budget, augment=True, tag_dropout=True):
        """A random passage whose music fits in `budget` tokens.

        Returns (tags, music token ids). The ids end with <eos> when the
        passage runs to the end of its piece."""
        # pick a piece at random, weighted by its note count: longer pieces are picked
        # more often, so every note is equally likely to end up in training
        i = rng.choice(len(self), p=self.sizes / self.sizes.sum())
        # the chosen piece's notes
        piece = self.piece(i)
        # a note is at least 3 tokens (Pitch, Vel, Dur), so the budget holds at most budget // 3 notes
        n = budget // 3                            # a note is at least 3 tokens
        # the latest start index that still leaves n notes before the end (0 if the piece is short)
        last_start = max(len(piece) - n, 0)
        # 3 % of the time force the passage to end at the piece's end (so the model learns endings);
        # otherwise start anywhere from 0 to last_start inclusive
        start = last_start if rng.random() < 0.03 else rng.integers(0, last_start + 1)
        # the passage reaches the final note exactly when it starts at last_start
        # (forced 3 % of the time, or by chance, or always for short pieces)
        at_end = start == last_start               # 3 % of passages are endings
        # take up to n notes; copy so augmentation does not modify the dataset itself
        notes = piece[start:start + n].copy()
        # data augmentation: random small variations so the model sees more varied music
        if augment:                                # small random variations
            # lowest and highest pitch in the passage (column 1 is pitch)
            lo, hi = notes[:, 1].min(), notes[:, 1].max()
            # transpose by up to 3 semitones either way, but only as far as keeps every note
            # on the 88-key keyboard (integers' upper bound is exclusive, hence the +1)
            shift = rng.integers(max(-3, T.PITCH_MIN - lo), min(3, T.PITCH_MAX - hi) + 1)
            # apply the transposition to every pitch
            notes[:, 1] += shift                               # transpose
            # stretch onsets (column 0) and durations (column 3) by one random factor in
            # [0.9, 1.1], i.e. play up to 10 % faster or slower
            notes[:, [0, 3]] *= rng.uniform(0.9, 1.1)          # faster / slower
            # shift all velocities (column 2) by the same random amount in -6..+6 (louder or
            # softer), clipped to MIDI's valid range 1..127
            notes[:, 2] = np.clip(notes[:, 2] + rng.integers(-6, 7), 1, 127)

        # turn the notes into music token ids ([Shift...] Pitch Vel Dur per note)
        music = T.encode_notes(notes)
        # tokens we may spend: an ending needs one slot reserved for the <eos> token
        room = budget - 1 if at_end else budget    # an ending needs room for <eos>
        # shift tokens can push the stream past the budget; if so, trim it, keeping whole notes only
        if len(music) > room:
            # for an ending we must keep the end of the piece, so cut from the front
            if at_end:                             # keep the last notes
                # keep the last `room` tokens
                music = music[-room:]
                # the cut may land mid-note (e.g. on a Vel or Dur token, or a Shift belonging to
                # the first kept note); drop tokens until the stream starts on a Pitch
                while T.kind(music[0]) != "pitch":
                    # remove the leading token
                    music.pop(0)
            # otherwise keep the start of the passage, so cut from the back
            else:                                  # keep the first notes
                # keep the first `room` tokens
                music = music[:room]
                # drop trailing tokens until the stream ends on a Dur, i.e. a complete note
                while T.kind(music[-1]) != "dur":
                    # remove the last token
                    music.pop()
            # count the notes that survived (every note has exactly one Pitch token)
            n_kept = sum(T.kind(t) == "pitch" for t in music)
            # trim the note array the same way, so the tags below describe exactly the kept music
            notes = notes[-n_kept:] if at_end else notes[:n_kept]

        # measure density/dynamics/register/key from the notes actually kept
        tags = compute_tags(notes)                 # measured on what is kept
        # add the piece's era and composer tags from the metadata, when it has them
        tags.update({k: self.meta[i][k] for k in ("era", "composer") if k in self.meta[i]})
        # optionally remove some tags at random (see drop_tags)
        if tag_dropout:
            # replace the full tag set with a random subset
            tags = drop_tags(tags, rng)
        # return the tags and the music tokens, with <eos> appended if this passage ends the piece
        return tags, music + ([T.EOS] if at_end else [])

    def batch(self, rng, batch_size, ctx, **kw):
        """(inputs, targets, mask) as int32/float32 arrays of shape [B, ctx-1].

        The mask is 1 where the target is a music token (or <eos>): the loss
        ignores padding and does not ask the model to predict the tags."""
        # a [batch_size, ctx] grid of token ids, pre-filled with <pad> so short rows are padded
        x = np.full((batch_size, ctx), T.PAD, dtype=np.int32)
        # tokens reserved at the start of each row for the request: <bos>, up to six tags, <sep>
        prefix = len(T.TAG_ORDER) + 2              # <bos>, up to six tags, <sep>
        # fill each row of the batch with one random passage
        for b in range(batch_size):
            # get tags and music sized so that prefix + music fits in ctx (kw passes augment/tag_dropout)
            tags, music = self.passage(rng, ctx - prefix, **kw)
            # full sequence: [<bos>, tag tokens..., <sep>] followed by the music tokens
            ids = T.encode_tags(tags) + music
            # write it at the start of row b
            x[b, :len(ids)] = ids                  # the rest stays <pad>
        # shift by one: at each position the model sees tokens up to t and must predict token t+1
        # (e.g. inputs [a, b, c] -> targets [b, c, d])
        inputs, targets = x[:, :-1], x[:, 1:]       # predict the next token
        # 1.0 where the target is a music token or <eos>, 0.0 for <pad>, <bos>, <sep> and tags,
        # so the loss only scores the music the model should generate
        mask = ((targets >= T.MUSIC0) | (targets == T.EOS)).astype(np.float32)
        # hand back the three numpy arrays
        return inputs, targets, mask


def fixed_batches(ds, n_batches, batch_size, ctx, seed=0):
    """The same un-augmented, fully tagged batches every time, for validation."""
    # a random generator with a fixed seed, so the same passages are drawn on every call
    rng = np.random.default_rng(seed)
    # build n_batches batches with no augmentation and no tag dropout, for a stable comparison
    return [ds.batch(rng, batch_size, ctx, augment=False, tag_dropout=False)
            for _ in range(n_batches)]
