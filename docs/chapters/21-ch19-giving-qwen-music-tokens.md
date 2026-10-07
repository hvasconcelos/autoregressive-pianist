# 19. Giving Qwen music tokens

::: filename
qwen/music_lm.py
:::

``` python
"""Shared pieces for the Qwen version: vocabulary, model loading, the
music-only output layer, and batches of caption + music.

Run from the project folder:  python qwen/train_qwen.py ...
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import tokenizer as T
from captions import caption

MUSIC_NAMES = T.VOCAB[T.MUSIC0:]                     # Pitch_21 ... Shift_1000
NEW_TOKENS = [f"<|{n}|>" for n in MUSIC_NAMES] + ["<|music_end|>", "<|music|>"]
N_MUSIC = len(MUSIC_NAMES)
N_OUT = N_MUSIC + 1                                  # music tokens + <|music_end|>


def load(name, device, dtype=None):
    """Load Qwen and give it our music tokens.

    Returns (model, tokenizer, base) where `base` is the id of the first
    music token: our token `t` lives at Qwen id `base + t - MUSIC0`."""
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    is_new = tok.convert_tokens_to_ids(NEW_TOKENS[0]) in (None, tok.unk_token_id)
    if is_new:
        n_text = len(tok)                            # tokens Qwen already defines
        tok.add_tokens(NEW_TOKENS, special_tokens=True)
        if len(tok) > model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(tok), mean_resizing=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = tok.convert_tokens_to_ids(NEW_TOKENS[0])
    assert tok.convert_tokens_to_ids(NEW_TOKENS[-1]) == base + N_MUSIC + 1
    if is_new:
        # start every music row at the average text embedding, plus a little
        # noise so that the rows are not identical
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            text = emb[:n_text].float()
            noise = torch.randn(len(NEW_TOKENS), emb.shape[1]) * text.std(0) * 0.1
            emb[base:base + len(NEW_TOKENS)] = (text.mean(0) + noise).to(emb.dtype)
    return model.to(device), tok, base


def to_qwen(ids, base):
    """Our music token ids -> Qwen ids (<eos> becomes <|music_end|>)."""
    return [base + N_MUSIC if t == T.EOS else base + t - T.MUSIC0 for t in ids]


def music_logits(model, hidden, base):
    """Scores for the next token over the music tokens only.

    Qwen's normal output layer scores all ~152,000 tokens. We multiply the
    hidden state by just our rows of the (tied) embedding table instead."""
    rows = model.get_input_embeddings().weight[base:base + N_OUT]
    return hidden @ rows.T


def hidden_states(model, **inputs):
    """Run the transformer body without its full-vocabulary output layer."""
    return model.model(**inputs).last_hidden_state


def make_batch(ds, tok, base, rng, batch_size, ctx, augment=True, tag_dropout=True):
    """Tensors (input_ids, attention_mask, targets). `targets` holds an index
    into the music-only output (0 ... N_OUT-1), or -100 where no loss applies."""
    rows = []
    for _ in range(batch_size):
        tags, music = ds.passage(rng, ctx - 48, augment, tag_dropout)   # 48: caption
        text = tok(caption(tags, rng), add_special_tokens=False)["input_ids"][:46]
        ids = text + [base + N_MUSIC + 1] + to_qwen(music, base)
        rows.append((ids, len(text) + 1))            # (sequence, where music begins)
    width = max(len(ids) for ids, _ in rows)
    x = torch.full((batch_size, width), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((batch_size, width), dtype=torch.long)
    y = torch.full((batch_size, width), -100, dtype=torch.long)
    for b, (ids, start) in enumerate(rows):
        x[b, :len(ids)] = torch.tensor(ids)
        att[b, :len(ids)] = 1
        # the target at position i is the token at i + 1, as a music index
        y[b, start - 1:len(ids) - 1] = torch.tensor(ids[start:]) - base
    return x, att, y


def music_loss(model, base, x, att, y):
    """Cross-entropy over music tokens, at the positions that have a target."""
    hidden = hidden_states(model, input_ids=x, attention_mask=att)
    sel = y >= 0
    logits = music_logits(model, hidden[sel], base)
    return F.cross_entropy(logits.float(), y[sel])
```

This file holds the four ideas that turn a text model into a music model.

## 19.1 1. New tokens

`load` adds 402 tokens to Qwen's tokeniser: the 400 music tokens, a `<|music_end|>` marker and a `<|music|>` marker that plays the role `<sep>` played in Part I. They are added as *special* tokens, which guarantees the tokeniser treats each one as a single unit and never splits it into pieces.

The embedding table then needs a row for each. Qwen's table has 151,936 rows, somewhat more than its tokeniser uses, so some of our tokens land on spare rows that already exist and the rest need new rows, which `resize_token_embeddings` adds. To treat all 402 alike, the code then overwrites every one of them with the same starting value: the average of all the text embeddings, plus a little random noise so that no two rows are identical. They begin as "average-looking" tokens and not as outliers. Because the table is tied, the same rows serve as the output layer.

The new ids are consecutive. That lets the code convert between our numbering and Qwen's with one addition: our token `t` is Qwen's token `base + t − MUSIC0`.

## 19.2 2. The sequence

A training sequence is a caption, the `<|music|>` marker, and the music:

``` text
A slow, quiet piano piece, in D minor. <|music|> <|Pitch_62|> <|Vel_46|> <|Dur_800|> ...
```

The caption is ordinary text and takes around 20 tokens. To the model this is one continuous sequence, and the music is simply how this kind of sentence continues.

## 19.3 3. A music-only output layer

This is the most important optimisation in Part II.

Qwen's output layer scores every one of its roughly 152,000 tokens at every position. We only ever want one of 401. Computing the other 151,500 scores is wasted work, and in training it is a great deal of wasted memory:

| Output layer | Scores per batch (16 × 1,024 positions) | Memory at 4 bytes each |
|----|----|----|
| Full vocabulary | 16 × 1,024 × 151,936 ≈ 2.5 billion | About 10 GB |
| Music tokens only | 16 × 1,024 × 401 ≈ 6.6 million | About 26 MB |

`music_logits` therefore skips Qwen's output layer. It takes the final hidden vectors and multiplies them by only our 401 rows of the embedding table. Since the table is tied, this gives exactly the scores the full layer would have given for those tokens.

It has a second benefit. The loss is now a choice among music tokens only, the same kind of choice the small model makes, so the two models' losses can be compared closely. (The small model's output also covers its 59 tag and special tokens, but it learns within a few hundred steps that those never occur in music, so the difference is negligible.)

## 19.4 4. Scoring only the music

`make_batch` builds a `targets` array that is −100 wherever no loss should apply: over the caption, the marker and the padding. `music_loss` selects only the positions that have a real target before computing anything, so the caption costs nothing in the output layer.

The model is not trained to write captions. It is trained to read them.
