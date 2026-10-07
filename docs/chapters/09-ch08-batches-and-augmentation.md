# 8. Batches and augmentation

Training consumes *batches*: a block of 32 token sequences, each up to 1,024 tokens long. This chapter builds them.

## 8.1 Random passages, not fixed chunks

A simple approach would cut every piece into consecutive chunks once and reuse them. We do something better. Every time a batch is needed, each of its 32 sequences is cut fresh from a random place in a random piece. Over a long training run the model sees every possible alignment of every piece, and never the same chunk twice.

Pieces are chosen with probability proportional to their length, so that every note in the dataset is equally likely to be used.

## 8.2 Augmentation

Augmentation means altering the training data in ways that keep it valid music, so that the model sees more variety than the dataset contains. With 22 million tokens and 19 million parameters, it is the main defence against memorisation. Each passage gets three random changes.

| Change | Range | Why it is safe |
|----|----|----|
| Transpose | Up or down by 0 to 3 semitones | A piece played a tone higher is still the same piece. The key tag is measured afterwards, so it stays correct. |
| Stretch time | 0.9× to 1.1× | A slightly faster or slower performance is equally plausible. |
| Shift velocity | Add −6 to +6 to every note | The same performance on a slightly louder or softer day. |

The ranges are deliberately modest. Transposing by an octave would move music into registers where it would never be written.

## 8.3 Tag dropout

When you make a request you will rarely specify all six tags. "Something quiet" sets only one. The model must therefore be comfortable with any subset, and it only becomes comfortable with what it sees in training.

`drop_tags` removes tags at random from each training passage, in two stages. One passage in ten loses all its tags, which teaches the model to play with no request at all. For the others, a removal probability between 0 and 0.8 is drawn first, and each tag is then dropped with that probability. Some passages therefore keep everything, some keep a single tag, and the rest fall in between. A simpler scheme that drops each tag with a fixed small probability would almost never produce a one-tag example, and one-tag requests are the most common kind.

## 8.4 Endings

`<eos>` should follow the last note of a piece and nothing else. A passage cut from a random place almost never includes the last note, so the model would hardly ever see an ending. To fix that, 3% of passages are deliberately taken from the very end of their piece and finish with `<eos>`.

## 8.5 What the model is scored on

The loss is computed only where the target is a music token or `<eos>`. Predicting the tags themselves is not part of the job, and padding is meaningless, so both are masked out. This also means the loss values in this book are "per music token", which makes them comparable between the small model and Qwen later.

## 8.6 The code

::: filename
data.py
:::

``` python
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
    def __init__(self, path):
        z = np.load(path)
        self.notes = z["notes"].astype(np.float64)
        self.offsets = z["offsets"]
        self.meta = json.loads(str(z["meta"]))
        self.sizes = np.diff(self.offsets)

    def __len__(self):
        return len(self.sizes)

    def piece(self, i):
        return self.notes[self.offsets[i]:self.offsets[i + 1]]

    def passage(self, rng, budget, augment=True, tag_dropout=True):
        """A random passage whose music fits in `budget` tokens.

        Returns (tags, music token ids). The ids end with <eos> when the
        passage runs to the end of its piece."""
        i = rng.choice(len(self), p=self.sizes / self.sizes.sum())
        piece = self.piece(i)
        n = budget // 3                            # a note is at least 3 tokens
        last_start = max(len(piece) - n, 0)
        start = last_start if rng.random() < 0.03 else rng.integers(0, last_start + 1)
        at_end = start == last_start               # 3 % of passages are endings
        notes = piece[start:start + n].copy()
        if augment:
            lo, hi = notes[:, 1].min(), notes[:, 1].max()
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
        tags.update({k: self.meta[i][k] for k in ("era", "composer") if k in self.meta[i]})
        if tag_dropout:
            tags = drop_tags(tags, rng)
        return tags, music + ([T.EOS] if at_end else [])

    def batch(self, rng, batch_size, ctx, **kw):
        """(inputs, targets, mask) as int32/float32 arrays of shape [B, ctx-1].

        The mask is 1 where the target is a music token (or <eos>): the loss
        ignores padding and does not ask the model to predict the tags."""
        x = np.full((batch_size, ctx), T.PAD, dtype=np.int32)
        prefix = len(T.TAG_ORDER) + 2              # <bos>, up to six tags, <sep>
        for b in range(batch_size):
            tags, music = self.passage(rng, ctx - prefix, **kw)
            ids = T.encode_tags(tags) + music
            x[b, :len(ids)] = ids
        inputs, targets = x[:, :-1], x[:, 1:]
        mask = ((targets >= T.MUSIC0) | (targets == T.EOS)).astype(np.float32)
        return inputs, targets, mask


def fixed_batches(ds, n_batches, batch_size, ctx, seed=0):
    """The same un-augmented, fully tagged batches every time, for validation."""
    rng = np.random.default_rng(seed)
    return [ds.batch(rng, batch_size, ctx, augment=False, tag_dropout=False)
            for _ in range(n_batches)]
```

`passage` builds the music for one sequence. It takes more notes than can possibly fit (a note is never fewer than three tokens), encodes them, and cuts the token list at a note boundary to fit the budget. Ordinary passages keep their first notes; endings keep their last. The tags are computed *after* the cut, on exactly the notes the model will see.

`batch` stacks 32 sequences into an array, padding the short ones. The inputs are every token but the last; the targets are every token but the first. Position *i* of the input is paired with position *i* + 1 of the sequence, which is the "predict the next token" exercise from chapter 3.

`fixed_batches` produces the validation batches. It uses a fixed random seed and switches off augmentation and tag dropout, so every validation passage carries its full set of tags and the validation loss is measured on the same data every time and changes only because the model changed.
