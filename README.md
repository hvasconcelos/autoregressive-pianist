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

You need a Mac with Apple Silicon and Python 3.10 or newer (developed on 3.12).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install mlx numpy mido python-rtmidi
```

`python-rtmidi` is only needed by `play.py` to open real MIDI ports.

## 2. Smoke test on toy data (about 5 minutes)

Before spending hours on MAESTRO, run the whole pipeline on a tiny fake dataset.
This checks that every stage works.

```bash
# 120 random "pieces" laid out like MAESTRO
python make_toy_data.py --out data/toy

# MIDI -> .npz
python prepare.py --maestro data/toy --out data/toy_prepared

# a tiny model, trained briefly
python train.py --data data/toy_prepared --out runs/toy \
    --steps 1500 --ctx 192 --dim 96 --layers 2 --heads 4 --eval-every 250

# check it end to end
python evaluate.py --model runs/toy/best --data data/toy_prepared
python sample.py --model runs/toy/best --request "fast and loud" --out toy.mid
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
python prepare.py --maestro data/maestro-v3.0.0 --out data/prepared
```

This reads every performance, keeps the 88 piano keys, drops fragments with fewer
than 64 notes, attaches composer and era tags, and writes `train.npz`,
`validation.npz` and `test.npz` to `data/prepared/`. It uses MAESTRO's own
train/validation/test split. Any composers it can't place in an era are printed
at the end; you can add them to the table in `tags.py`.

Optional: check that every tag value has a fair share of the data.

```bash
python stats.py --data data/prepared --split train
```

If one value is almost empty, adjust the bucket thresholds in `tags.py` and run
`prepare.py` again.

## 5. Train

The defaults are the full model: 6 layers, width 512, 8 heads, 1024-token context,
batch 32, 20,000 steps.

```bash
python train.py --data data/prepared --out runs/v1
```

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
python train.py --data data/prepared --out runs/v1 --resume
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
- **Out of memory**: lower `--batch` (for example 16) or `--ctx` (for example 512).
- **Odd errors in the training step**: add `--no-compile` to get readable
  tracebacks.

## 6. Evaluate

```bash
python evaluate.py --model runs/v1/best --data data/prepared
```

This prints:

1. **Held-out loss** for each token family (pitch, velocity, duration, shift) and
   overall, with perplexity. Use `--split test` for the final number.
2. **Tag adherence**: the model generates with each value of `density`, `dynamics`
   and `register`, measures what it actually played, and reports the exact-match
   rate and within-one rate next to the chance rate. Add `--keys` to also test all
   24 keys (slow).

## 7. Generate music

To write a MIDI file:

```bash
python sample.py --model runs/v1/best --request "slow and quiet in D minor" --out out.mid
# or with exact tags
python sample.py --model runs/v1/best --tags key=Dmin,dynamics=p,density=sparse --out out.mid
```

Useful flags: `--notes` (default 400), `--temperature` (default 1.0),
`--top-p` (default 0.95) and `--seed`.

To see which tags a request becomes:

```bash
python request.py "something slow and quiet in D minor, like Chopin"
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
python play.py --list                                   # show MIDI output ports
python play.py --port "IAC Driver Bus 1" --request "calm, quiet, in F major"
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
python3 -m venv .venv && source .venv/bin/activate
pip install mlx numpy mido python-rtmidi

curl -L -o data/maestro-v3.0.0-midi.zip --create-dirs \
  https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
unzip -q data/maestro-v3.0.0-midi.zip -d data

python prepare.py  --maestro data/maestro-v3.0.0 --out data/prepared
python stats.py    --data data/prepared
python train.py    --data data/prepared --out runs/v1          # add --resume to continue
python evaluate.py --model runs/v1/best --data data/prepared --split test
python sample.py   --model runs/v1/best --request "slow and quiet in D minor" --out out.mid
```

`data/` and `runs/` are git-ignored.
