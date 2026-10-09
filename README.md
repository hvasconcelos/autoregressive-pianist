# Autoregressive Pianist

A small GPT-style model that learns to play solo piano from the
[MAESTRO](https://magenta.tensorflow.org/datasets/maestro) dataset. You can steer it
with tags (key, dynamics, density, register, era, composer) or plain words like
"slow and quiet in D minor". It can write MIDI files or play live on a MIDI
instrument.

Training runs on Apple Silicon with [MLX](https://github.com/ml-explore/mlx).

## Pipeline at a glance

```
MAESTRO MIDI ──prepare.py──▶ data/prepared/*.npz ──train.py──▶ runs/v1/best.*
                                   │                               │
                               stats.py                 evaluate.py / sample.py / play.py
```

| Script             | What it does                                                    |
|--------------------|-----------------------------------------------------------------|
| `make_toy_data.py` | Writes a small fake MAESTRO-style dataset for smoke tests        |
| `prepare.py`       | Converts MAESTRO MIDI into `train/validation/test.npz`           |
| `stats.py`         | Shows how tag values are spread across a prepared split          |
| `train.py`         | Trains the model and saves `best` and `last` checkpoints         |
| `evaluate.py`      | Reports held-out loss per token family and how well tags are followed |
| `sample.py`        | Generates a performance into a `.mid` file                      |
| `play.py`          | Plays in real time on a MIDI port and takes new requests as you type |
| `request.py`       | Turns a plain-words request into tags (handy for checking)       |

## 1. Setup

You need a Mac with Apple Silicon, [uv](https://docs.astral.sh/uv/) and Python 3.10
or newer (developed on 3.12).

```bash
uv venv --python 3.12
uv pip install mlx numpy mido python-rtmidi
```

`python-rtmidi` is only needed by `play.py` to open real MIDI ports. Every command
below runs through `uv run`, which uses the `.venv` in the project folder, so there
is no environment to activate.

## 2. Smoke test on toy data (about 5 minutes)

Before spending hours on MAESTRO, run the whole pipeline on a tiny fake dataset.
This checks that every stage works.

```bash
# 120 random "pieces" laid out like MAESTRO
uv run python make_toy_data.py --out data/toy

# MIDI -> .npz
uv run python prepare.py --maestro data/toy --out data/toy_prepared

# a tiny model, trained briefly
uv run python train.py --data data/toy_prepared --out runs/toy \
    --steps 1500 --ctx 192 --dim 96 --layers 2 --heads 4 --eval-every 250

# check it end to end
uv run python evaluate.py --model runs/toy/best --data data/toy_prepared
uv run python sample.py --model runs/toy/best --request "fast and loud" --out toy.mid
```

Validation loss should drop from about 2.5 to about 1.9.

## 3. Get MAESTRO

Download the MIDI-only version of MAESTRO v3.0.0 (about 57 MB) and unzip it into `data/`:

```bash
mkdir -p data
curl -L -o data/maestro-v3.0.0-midi.zip \
  https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
unzip -q data/maestro-v3.0.0-midi.zip -d data
```

You should now have `data/maestro-v3.0.0/maestro-v3.0.0.csv` and one folder per
year (`2004/`, `2006/`, ...).

## 4. Prepare the data

```bash
uv run python prepare.py --maestro data/maestro-v3.0.0 --out data/prepared
```

This reads every performance, keeps the 88 piano keys, drops fragments with fewer
than 64 notes, attaches composer and era tags, and writes `train.npz`,
`validation.npz` and `test.npz` to `data/prepared/`. It uses MAESTRO's own
train/validation/test split. Any composers it can't place in an era are printed
at the end; you can add them to the table in `tags.py`.

Optional: check that every tag value has a fair share of the data.

```bash
uv run python stats.py --data data/prepared --split train
```

If one value is almost empty, adjust the bucket thresholds in `tags.py`. These
tags are measured when batches are built, so you don't need to run `prepare.py`
again (you do if you change the `ERA` table).

## 5. Train

The defaults are the full model: 6 layers, width 512, 8 heads, 1024-token context,
batch 32, 20,000 steps.

```bash
uv run python train.py --data data/prepared --out runs/v1
```

### Macs with 16 GB of memory or less: use `--batch 8`

The default batch (32 sequences × 1024 tokens) needs more than 16 GB during
training, because attention keeps a 1024×1024 grid per head and layer for
backprop. On a 16 GB Mac the run starts swapping and never prints a progress
line. Use a smaller batch instead:

```bash
uv run python train.py --data data/prepared --out runs/v1 --batch 8
```

This needs roughly 4 GB. `--batch 16` (about 7 GB) may work if you close other
apps. Signs that you're out of memory: no `step 50` line after a few minutes,
and swap use climbing in Activity Monitor.

### How long it takes

Each progress line shows `tokens/s`. Estimate the total time with:

```
hours ≈ steps × batch × 1023 / tokens_per_s / 3600
```

As a rough guide, an M1 Pro runs at about 10–15k tokens/s on the default model,
so 20,000 steps at `--batch 8` take about 3–5 hours. The first progress line
takes a minute or two because MLX compiles the training step once. To shorten
the run, lower `--steps` (for example `--steps 10000`). `best.safetensors` is
saved whenever validation improves, so you can stop at any point and still
have a usable model.

What happens while it runs:

- A progress line every 50 steps (`loss`, `tokens/s`).
- Every 500 steps it computes validation loss on fixed batches and writes a row
  to `runs/v1/log.csv` (`step,train_loss,val_loss,tokens_per_s`).
- It saves `runs/v1/last.{json,safetensors}` at every evaluation, and
  `runs/v1/best.{json,safetensors}` when validation loss reaches a new low.

The learning rate warms up linearly for `--warmup` steps, then follows a cosine
curve down to 10% of `--lr`. AdamW is used with gradient clipping at 1.0.

### Resuming

If training stops, run the same command again with `--resume`. Training picks up
from `runs/v1/last` and continues the step count, LR schedule and log file:

```bash
uv run python train.py --data data/prepared --out runs/v1 --resume
```

Only the optimiser's step counter is restored; its momentum buffers start fresh.

### Training options

| Flag             | Default         | Meaning                                         |
|------------------|-----------------|-------------------------------------------------|
| `--data`         | `data/prepared` | Folder with `train.npz` and `validation.npz`    |
| `--out`          | `runs/v1`       | Folder for checkpoints and `log.csv`            |
| `--steps`        | `20000`         | Total optimiser updates                         |
| `--batch`        | `32`            | Passages per batch                              |
| `--ctx`          | `1024`          | Tokens per sequence (context length)            |
| `--dim`          | `512`           | Model width                                     |
| `--layers`       | `6`             | Transformer layers                              |
| `--heads`        | `8`             | Attention heads                                 |
| `--dropout`      | `0.1`           | Dropout rate                                    |
| `--lr`           | `3e-4`          | Peak learning rate                              |
| `--warmup`       | `1000`          | Warm-up steps                                   |
| `--eval-every`   | `500`           | Validate and checkpoint every N steps           |
| `--eval-batches` | `20`            | Number of fixed validation batches              |
| `--resume`       | off             | Continue from `<out>/last`                      |
| `--no-compile`   | off             | Turn off `mx.compile` (slower, easier to debug) |
| `--seed`         | `0`             | Random seed                                     |

### Troubleshooting

- **`loss is not finite: lower --lr`**: training diverged. Restart with a smaller
  `--lr`, for example `1e-4`.
- **No progress lines, heavy swapping**: you're out of memory. Lower `--batch`
  (8 on a 16 GB Mac) or `--ctx` (for example 512). See "Macs with 16 GB of memory" above.
- **Odd errors in the training step**: add `--no-compile` to get readable
  tracebacks.

## 6. Evaluate

```bash
uv run python evaluate.py --model runs/v1/best --data data/prepared
```

This prints:

1. **Held-out loss** for each token family (pitch, velocity, duration, shift) and
   overall, with perplexity. Use `--split test` for the final number.
2. **Tag adherence**: the model generates with each value of `density`, `dynamics`
   and `register`, measures what it actually played, and reports two rates next
   to the chance rate:
   - **exact**: the measured value equals the one requested.
   - **near**: the measured value is within one step of the request (for
     example `p` when `pp` was asked for).

   Add `--keys` to also test all 24 keys (slow). For keys, **near** counts the
   requested key, its relative major or minor, and the keys a fifth up and down.
   For C major, that's C major, A minor, G major and F major. These keys share
   all but one note of their scale, so the key estimate in `tags.py` often
   confuses them. Chance is 4% for exact and 17% for near. A high near rate with
   a low exact rate means the model plays in the right key family. A low near
   rate means it isn't following the key.

   The tag test generates fresh music with fixed seeds and doesn't use the
   dataset, so `--split` changes only the loss, never these rates. Each value is
   tried `--samples` times (default 8), so expect ±10% noise. Use `--samples 32`
   when comparing two models.

## 7. Generate music

To write a MIDI file:

```bash
uv run python sample.py --model runs/v1/best --request "slow and quiet in D minor" --out out.mid
# or with exact tags
uv run python sample.py --model runs/v1/best --tags key=Dmin,dynamics=p,density=sparse --out out.mid
```

Useful flags: `--notes` (default 400), `--temperature` (default 1.0),
`--top-p` (default 0.95) and `--seed`.

To see which tags a request becomes:

```bash
uv run python request.py "something slow and quiet in D minor, like Chopin"
```

Available tag values:

| Tag        | Values |
|------------|--------|
| `density`  | `very_sparse`, `sparse`, `medium`, `dense`, `very_dense` |
| `dynamics` | `pp`, `p`, `mf`, `f`, `ff` |
| `register` | `low`, `mid`, `high` |
| `key`      | `Cmaj` … `Bmaj`, `Cmin` … `Bmin` (flats spelled `Db`, `Eb`, `Gb`, `Ab`, `Bb`) |
| `era`      | `baroque`, `classical`, `romantic`, `modern` |
| `composer` | `bach`, `haydn`, `mozart`, `beethoven`, `schubert`, `chopin`, `schumann`, `liszt`, `mendelssohn`, `brahms`, `rachmaninoff`, `scriabin`, `debussy`, `ravel` |

## 8. Play in real time

```bash
uv run python play.py --list                                   # show MIDI output ports
uv run python play.py --port "IAC Driver Bus 1" --request "calm, quiet, in F major"
```

While it plays, type a new request and press Enter to change the music. Ctrl-C
stops and releases every key. Without `--port`, the note events are printed
instead of sent, which is handy for testing. On macOS, turn on the IAC Driver in
Audio MIDI Setup to send notes to a DAW or software piano.

`sample.py` prints how fast generation runs compared with playback. Generation
has to be faster than the music for `play.py` to keep up. If it isn't, raise
`--lookahead` (default 1.0 s).

## Full run, start to finish

```bash
uv venv --python 3.12
uv pip install mlx numpy mido python-rtmidi

curl -L -o data/maestro-v3.0.0-midi.zip --create-dirs \
  https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
unzip -q data/maestro-v3.0.0-midi.zip -d data

uv run python prepare.py  --maestro data/maestro-v3.0.0 --out data/prepared
uv run python stats.py    --data data/prepared
uv run python train.py    --data data/prepared --out runs/v1 --batch 8  # add --resume to continue
uv run python evaluate.py --model runs/v1/best --data data/prepared --split test
uv run python sample.py   --model runs/v1/best --request "slow and quiet in D minor" --out out.mid
```

`data/` and `runs/` are git-ignored.
