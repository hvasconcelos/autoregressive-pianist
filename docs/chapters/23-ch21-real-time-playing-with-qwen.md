# 21. Real-time playing with Qwen

## 21.1 The backend

Because the `Performer` talks to its model through two methods, plugging Qwen in takes one small class.

::: filename
qwen/qwen_backend.py
:::

``` python
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
```

`start` tokenises the request sentence, appends the `<|music|>` marker and any music already played, and runs the lot through the model. `step` feeds one token. Both go through `_logits`, which uses the music-only output layer and then copies the 401 scores into a vector laid out like the small model's 465-token vocabulary, with minus infinity in the slots that have no Qwen equivalent. The `Performer` cannot tell which model it is driving.

`past_key_values` is the Transformers library's name for the key/value cache.

::: filename
qwen/play_qwen.py
:::

``` python
"""Real-time playing with the fine-tuned Qwen. The request is plain text.

    python qwen/play_qwen.py --model runs/qwen/best --port "..." \
        --request "A slow, quiet piano piece in D minor, like Chopin."
"""
import argparse, sys
from qwen_backend import QwenBackend   # first: it puts the project folder on the import path
from performer import Performer
from play import MidiPort, PrintPort, play, stdin_commands
from midi_io import save_notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/qwen/best")
    ap.add_argument("--request", default="A calm piano piece.")
    ap.add_argument("--port", default=None)
    ap.add_argument("--out", default=None, help="write a MIDI file instead of playing")
    ap.add_argument("--notes", type=int, default=400)
    ap.add_argument("--lookahead", type=float, default=2.0)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    performer = Performer(QwenBackend(args.model), args.request,
                          args.temperature, args.top_p, seed=args.seed)
    if args.out:
        save_notes(performer.notes(args.notes), args.out)
        return
    port = MidiPort(args.port) if args.port else PrintPort()
    stats = play(performer, port, args.lookahead, args.seconds,
                 commands=stdin_commands(), to_request=lambda text: text)   # typed text is the request as is
    print(stats, file=sys.stderr)


if __name__ == "__main__":
    main()
```

## 21.2 Try it

Generate a file first:

``` bash
python qwen/play_qwen.py --model runs/qwen/best --out test.mid \
    --request "A calm, quiet piano piece in F major."
```

Then play live, either on an instrument connected to the Spark or with events printed to the screen:

``` bash
python qwen/play_qwen.py --model runs/qwen/best --port "<port name>" \
    --request "Something stormy and virtuosic, low on the keyboard."
```

As before, typing a new sentence and pressing Enter changes the request.

## 21.3 Will it be fast enough?

This is the open question of Part II, and the reason the small model was built first.

The requirement has not changed: the model must generate tokens faster than the music uses them. From chapter 7, MAESTRO averages about 9.8 notes per second, or roughly 40 tokens per second, with dense passages needing 100 or more.

A 0.6-billion-parameter model generating one token at a time in plain PyTorch spends much of its time on per-step overhead in Python and in launching many small GPU operations, and less on the arithmetic itself. Whether that reaches 100 tokens per second on the Spark has to be measured. Add a timing line around `performer.notes` in `play_qwen.py`, or simply play with events printed and read the `underruns` count.

If it is fast enough, you are done. If it is not, work through these in order.

| Step | What to do | Expected gain |
|----|----|----|
| 1 | Raise `--lookahead` to 4 or more | None in speed, but bursts and context rebuilds are absorbed. Enough if only dense passages stutter. |
| 2 | Shorten the window: build `QwenBackend(path, ctx=512)` | Each token attends to fewer positions, and the periodic rebuild is twice as quick. |
| 3 | Compile the model with `torch.compile` | Removes much of the per-step Python overhead. Try it on its own first, since compilation behaviour varies between platforms. |
| 4 | Export the model to an optimised inference runtime | The largest gain, and the most work. |
| 5 | Request lower density | Sparse music needs a fraction of the tokens. |
| 6 | Keep the small model for playing | See chapter 22. |

One structural fact is on your side. A note is three or four tokens, but only the first, the Pitch, is a hard musical decision. A later refinement, beyond the scope of this book, is to let the large model choose pitches and waits while a tiny second network fills in velocity and duration.
