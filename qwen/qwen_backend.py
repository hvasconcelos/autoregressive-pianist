"""Make the fine-tuned Qwen look like the small model to the Performer."""
import numpy as np
import torch
from music_lm import load, to_qwen, music_logits, N_MUSIC   # also sets the import path
import tokenizer as T


class QwenBackend:
    """`request` is a sentence, e.g. "A slow, quiet piano piece in D minor."."""

    def __init__(self, path, ctx=1024, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else None
        self.model, self.tok, self.base = load(path, self.device, dtype)
        self.model.eval()
        self.ctx = ctx - 48                     # the caption uses part of the window
        self.past = None

    def _logits(self, ids):
        """Feed `ids` after everything already in the cache; return logits for the next token."""
        with torch.no_grad():
            out = self.model.model(input_ids=torch.tensor([ids], device=self.device),
                                   past_key_values=self.past, use_cache=True)
            self.past = out.past_key_values     # the key/value cache
            z = music_logits(self.model, out.last_hidden_state[0, -1], self.base)
        z = z.float().cpu().numpy()
        full = np.full(T.VOCAB_SIZE, -np.inf)   # back to our 465-token layout
        full[T.MUSIC0:] = z[:N_MUSIC]
        full[T.EOS] = z[N_MUSIC]                # <|music_end|> plays the part of <eos>
        return full

    def start(self, request, music_tokens):
        # same layout as in training: caption (at most 46 tokens), <|music|>, music
        text = self.tok(request, add_special_tokens=False)["input_ids"][:46]
        self.past = None                        # a new prompt: start a fresh cache
        return self._logits(text + [self.base + N_MUSIC + 1] + to_qwen(music_tokens, self.base))

    def step(self, token):
        return self._logits(to_qwen([token], self.base))
