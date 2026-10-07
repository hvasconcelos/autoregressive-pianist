# How to use this book

This book walks through building a small neural network that plays the piano. You type a request such as "slow and quiet, in D minor", and the model produces a stream of notes, one at a time, that are sent to a keyboard as they are generated.

The book has two parts.

**Part I** builds everything from nothing on a MacBook Pro with Apple's MLX framework: the data pipeline, a tokeniser, a transformer with about 19 million parameters, the training loop, a real-time player and a set of tests that tell you whether the model works. By the end of Part I you have a working pianist and a clear yes-or-no answer to "does this approach work?".

**Part II** starts only after that answer is yes. It replaces the small model with a small Qwen language model, fine-tuned on an NVIDIA DGX Spark, so that the request can be free text instead of a fixed set of tags. Almost everything from Part I is reused.

Read the chapters in order the first time. Most of them add one file to the project, explain the ideas behind it and give you a command to run.

## The code

All the code is in the `pianist` folder that comes with this book as a zip file. Every listing in the book is printed directly from those files, so the two cannot disagree. You do not need to type anything in from the page.

``` text
pianist/
  tokenizer.py       notes <-> tokens, the vocabulary, the grammar      (chapter 6)
  midi_io.py         read and write MIDI files                          (chapter 5)
  tags.py            measure tags from notes                            (chapter 7)
  prepare.py         MAESTRO -> compact note arrays                     (chapter 7)
  stats.py           check the tag distribution                         (chapter 7)
  data.py            training batches and augmentation                  (chapter 8)
  model.py           the transformer, in MLX                            (chapter 9)
  make_toy_data.py   a fake dataset for a quick end-to-end test         (chapter 10)
  train.py           the training loop                                  (chapter 11)
  performer.py       the generation engine                              (chapter 12)
  sample.py          generate a MIDI file                               (chapter 12)
  play.py            play in real time                                  (chapter 13)
  request.py         plain words -> tags                                (chapter 14)
  evaluate.py        loss and tag-adherence tests                       (chapter 15)
  qwen/
    captions.py      tags -> a sentence                                 (chapter 18)
    music_lm.py      Qwen with music tokens                             (chapter 19)
    train_qwen.py    fine-tuning                                        (chapter 20)
    qwen_backend.py  Qwen behind the same player                        (chapter 21)
    play_qwen.py     real-time playing with Qwen                        (chapter 21)
```

## What was tested, and what was not

You should know exactly how far to trust each part before you spend a night of compute on it.

**Tested by running it while writing this book:**

- Every file in Part I ran end to end with MLX 0.32.3: preparing data, tokenising, training, saving and resuming, generating, the real-time scheduler and the evaluation. This was done on a Linux machine using MLX's CPU build, on the small synthetic dataset of chapter 10, with a deliberately tiny model. The numbers in chapter 10 are from that run.
- The Part II code ran end to end with PyTorch 2.14.1 and Transformers 5.19.0: adding the music tokens, the music-only output layer, the training loop, saving, resuming, reloading, and generating through the same player. This used a tiny, randomly initialised model with Qwen3's architecture, because the machine had no access to the real Qwen weights. The stand-in was saved in the same 16-bit format as Qwen and given spare embedding rows like Qwen's, but it ran on a CPU, so the GPU mixed-precision path was not exercised.
- The real-time scheduler was tested with a simulated performer that stalls for 0.3 seconds at intervals. Events were still sent within about a millisecond of their due time.
- An independent review pass read all the code and text and re-ran the pieces separately. It found, among other things, a number-format problem that would have quietly weakened the Qwen training, which has been fixed. Its checks are the reason some claims in this book are more cautious than they would otherwise be.

**Not tested:**

- Nothing was run on Apple silicon or on a DGX Spark. Speeds and training times for those machines are estimates, and each one is marked as such. Chapter 11 shows how to measure your own speed in the first minute.
- The full MAESTRO dataset could not be downloaded to the test machine. The preparation script was run against a folder with the same layout and column names. If MAESTRO's real files differ from the documented layout, `prepare.py` is where it will show.
- No model was trained to convergence on real music, so this book cannot show you a loss value or a sound sample from the final system. Chapter 15 says what to expect and what counts as a pass.
- Sending notes to a physical MIDI port, the optional language-model request parser in chapter 14, and loading the real Qwen3 weights were written but not run.

Where a claim comes from a published source rather than from a test, the source is listed at the end of the book.
