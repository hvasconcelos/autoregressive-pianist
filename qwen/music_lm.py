"""Shared pieces for the Qwen version: vocabulary, model loading, the
music-only output layer, and batches of caption + music.

Run from the project folder:  uv run python qwen/train_qwen.py ...
"""
import os, sys
# make the project folder importable, so the Part I modules (tokenizer, data, ...) load
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import tokenizer as T
from captions import caption

MUSIC_NAMES = T.VOCAB[T.MUSIC0:]                     # Pitch_21 ... Shift_1000
# one Qwen token per music token, then <|music_end|> (our <eos>) and <|music|>,
# which separates the caption from the music. Added in this order, they get
# consecutive ids: music at base ... base+N_MUSIC-1, then base+N_MUSIC, base+N_MUSIC+1
NEW_TOKENS = [f"<|{n}|>" for n in MUSIC_NAMES] + ["<|music_end|>", "<|music|>"]
N_MUSIC = len(MUSIC_NAMES)
N_OUT = N_MUSIC + 1                                  # music tokens + <|music_end|>


def load(name, device, dtype=None):
    """Load Qwen and give it our music tokens.

    Returns (model, tokenizer, base) where `base` is the id of the first
    music token: our token `t` lives at Qwen id `base + t - MUSIC0`."""
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    # a fine-tuned checkpoint already has the tokens; only the base model needs them added
    is_new = tok.convert_tokens_to_ids(NEW_TOKENS[0]) in (None, tok.unk_token_id)
    if is_new:
        n_text = len(tok)                            # tokens Qwen already defines
        tok.add_tokens(NEW_TOKENS, special_tokens=True)
        # Qwen pads its embedding table, so there may already be room for them
        if len(tok) > model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(tok), mean_resizing=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = tok.convert_tokens_to_ids(NEW_TOKENS[0])
    # the id arithmetic everywhere else relies on the tokens being contiguous
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
        # reserve 48 positions for the caption (at most 46 tokens) and <|music|>
        tags, music = ds.passage(rng, ctx - 48, augment, tag_dropout)
        text = tok(caption(tags, rng), add_special_tokens=False)["input_ids"][:46]
        ids = text + [base + N_MUSIC + 1] + to_qwen(music, base)   # caption <|music|> music
        rows.append((ids, len(text) + 1))            # (sequence, where music begins)
    width = max(len(ids) for ids, _ in rows)
    x = torch.full((batch_size, width), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((batch_size, width), dtype=torch.long)
    y = torch.full((batch_size, width), -100, dtype=torch.long)
    for b, (ids, start) in enumerate(rows):
        x[b, :len(ids)] = torch.tensor(ids)
        att[b, :len(ids)] = 1
        # the target at position i is the token at i + 1, as a music index;
        # the caption is never a target, so the model only learns to play
        y[b, start - 1:len(ids) - 1] = torch.tensor(ids[start:]) - base
    return x, att, y


def music_loss(model, base, x, att, y):
    """Cross-entropy over music tokens, at the positions that have a target."""
    hidden = hidden_states(model, input_ids=x, attention_mask=att)
    sel = y >= 0
    logits = music_logits(model, hidden[sel], base)
    return F.cross_entropy(logits.float(), y[sel])
