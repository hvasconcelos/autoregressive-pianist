# Autoregressive Pianist

A small GPT-style model that learns to play solo piano from two datasets:
[MAESTRO](https://magenta.tensorflow.org/datasets/maestro) (about 200 hours of
classical piano recorded on concert grands) and a genre-balanced slice of
[Aria-MIDI](https://huggingface.co/datasets/loubb/aria-midi) (about 2,200 hours of
transcribed classical, jazz, pop, film music, ragtime and more). You can steer it
with tags (key, dynamics, density, register, era, composer, genre) or plain words
like "slow and quiet in D minor" or "a jazzy ballad". It can write MIDI files or play live on a MIDI
instrument.

Training runs on Apple Silicon with [MLX](https://github.com/ml-explore/mlx).
A second version fine-tunes a small Qwen on the same two datasets so that any
sentence works as a request; it needs an NVIDIA GPU (section 9).

## Pipeline at a glance

```
data/maestro ──prepare.py──────▶ data/maestro_prepared ─┐
data/aria    ──prepare_aria.py──▶ data/aria_prepared ────┴─train.py──▶ runs/v2/best.*
                                        │                                  │
                                    stats.py                 evaluate.py / sample.py / play.py
```

| Script             | What it does                                                    |
|--------------------|-----------------------------------------------------------------|
| `make_toy_data.py` | Writes a small fake MAESTRO-style dataset for smoke tests        |
| `prepare.py`       | Converts MAESTRO MIDI into `train/validation/test.npz`           |
| `prepare_aria.py`  | Picks a balanced, high-quality slice of Aria-MIDI and converts it the same way |
| `stats.py`         | Shows how tag values are spread across a prepared split          |
| `train.py`         | Trains the model and saves `best` and `last` checkpoints         |
| `evaluate.py`      | Reports held-out loss per token family and how well tags are followed |
| `sample.py`        | Generates a performance into a `.mid` file                      |
| `play.py`          | Plays in real time on a MIDI port and takes new requests as you type |
| `request.py`       | Turns a plain-words request into tags (handy for checking)       |

Each dataset has a raw folder and a prepared folder next to it in `data/`:

```
data/
  maestro/            maestro-v3.0.0-midi.zip and the unzipped maestro-v3.0.0/
  maestro_prepared/   train.npz, validation.npz, test.npz
  aria/               aria-midi-v1-deduped-ext.tar.gz (read without unpacking)
  aria_prepared/      train.npz, validation.npz, test.npz
  toy/                fake MAESTRO-style data from make_toy_data.py
  toy_prepared/       train.npz, validation.npz, test.npz
```

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

## 3. Get the data

Download the MIDI-only version of MAESTRO v3.0.0 (about 57 MB) and unzip it into `data/maestro/`:

```bash
mkdir -p data/maestro
curl -L -o data/maestro/maestro-v3.0.0-midi.zip \
  https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
unzip -q data/maestro/maestro-v3.0.0-midi.zip -d data/maestro
```

You should now have `data/maestro/maestro-v3.0.0/maestro-v3.0.0.csv` and one folder per
year (`2004/`, `2006/`, ...).

Then download Aria-MIDI's deduplicated subset (2.0 GB, 371,053 files, CC BY-NC-SA 4.0)
into `data/aria/`. Don't unpack it: `prepare_aria.py` reads the archive directly.

```bash
hf download loubb/aria-midi aria-midi-v1-deduped-ext.tar.gz --repo-type dataset --local-dir data/aria
```

## 4. Prepare the data

### MAESTRO

```bash
uv run python prepare.py --maestro data/maestro/maestro-v3.0.0 --out data/maestro_prepared
```

This reads every performance, keeps the 88 piano keys, drops fragments with fewer
than 64 notes, attaches genre (`classical`), composer and era tags, and writes `train.npz`,
`validation.npz` and `test.npz` to `data/maestro_prepared/`. It uses MAESTRO's own
train/validation/test split. Any composers it can't place in an era are printed
at the end; you can add them to the table in `tags.py`.

### Aria-MIDI

```bash
uv run python prepare_aria.py --archive data/aria/aria-midi-v1-deduped-ext.tar.gz --out data/aria_prepared
```

It picks files using each recording's audio-quality score, best first. It
takes one file from each genre in turn, so small genres aren't crowded out, and
stops at `--notes` (default 55M, about 10× MAESTRO). Of the picked
recordings, one in `--hold-out` (100) goes to `validation.npz` and another
one in 100 to `test.npz`. Both are chosen by recording ID, so the split is
the same on every run and every genre is represented. Files without a genre
label are skipped. It takes a few minutes and gives 37,683 train, 392 validation and
389 test pieces (53.6M training notes).

Aria's genre labels map onto the `genre` tag like this:

| Aria label | `genre` tag |
|---|---|
| classical, atonal | `classical` |
| pop, rock | `pop` |
| soundtrack | `film` |
| jazz, blues | `jazz` |
| ragtime | `ragtime` |
| folk, ambient | `other` |

For classical Aria files, era and
composer come from the composer's name as for MAESTRO, falling back to Aria's
`music_period`. Other genres get no era or composer. The mapping is in `tags.py`
(`GENRE`, `PERIOD`, `aria_tags`).

### Check the tags

Check that every tag value has a fair share of each dataset:

```bash
uv run python stats.py --data data/maestro_prepared
uv run python stats.py --data data/aria_prepared
```

If one value is almost empty, adjust the bucket thresholds in `tags.py`. These
tags are measured when batches are built, so you don't need to prepare again
(you do if you change `ERA`, `GENRE` or `PERIOD`, which are stored at prepare
time).

The genre tag made the vocabulary 465 tokens (459 before), so checkpoints from
before it can't be loaded by the current code.

## 5. Train

Train on both datasets. List MAESTRO first: training uses every folder's
`train.npz`, but validation uses only the first folder's, so the validation loss
measures classical playing on the cleanest data.

```bash
caffeinate -i uv run python train.py --data data/maestro_prepared data/aria_prepared \
    --out runs/v2 --batch 8 --steps 60000
```

The model is 6 layers, width 512, 8 heads and a 1024-token context, 19.1M
parameters. The two datasets are about 227M tokens. 60,000 steps at batch 8 is a
little over two passes and takes about 9.5 hours on an M1 Pro. `caffeinate`
keeps the Mac awake.

For a MAESTRO-only baseline to compare against, train
`--data data/maestro_prepared --out runs/v1 --batch 8`.

### Macs with 16 GB of memory or less: use `--batch 8`

`train.py`'s default batch (32 sequences × 1024 tokens) needs more than 16 GB during
training, because attention keeps a 1024×1024 grid per head and layer for
backprop. On a 16 GB Mac the run starts swapping and never prints a progress
line. Use a smaller batch instead:

```bash
uv run python train.py --data data/maestro_prepared data/aria_prepared --out runs/v2 --batch 8
```

This needs roughly 4 GB, plus about 1 GB for the two datasets' notes. `--batch 16` (about 7 GB) may work if you close other
apps. Signs that you're out of memory: no `step 50` line after a few minutes,
and swap use climbing in Activity Monitor.

### How long it takes

Each progress line shows `tokens/s`. Estimate the total time with:

```
hours ≈ steps × batch × 1023 / tokens_per_s / 3600
```

Measured on an M1 Pro: about 14.5k tokens/s at `--batch 8`, so 60,000 steps take
about 9.5 hours and 20,000 about 3.2 hours. The first progress line
takes a minute or two because MLX compiles the training step once. To shorten
the run, lower `--steps` (for example `--steps 10000`). `best.safetensors` is
saved whenever validation improves, so you can stop at any point and still
have a usable model.

What happens while it runs:

- A progress line every 50 steps (`loss`, `tokens/s`).
- Every 500 steps it computes validation loss on fixed batches and writes a row
  to `runs/v2/log.csv` (`step,train_loss,val_loss,tokens_per_s`).
- It saves `runs/v2/last.{json,safetensors}` at every evaluation, and
  `runs/v2/best.{json,safetensors}` when validation loss reaches a new low.

The training loss sits below the validation loss partly because it is mostly
Aria music, while validation is MAESTRO only. A gap is not by itself
overfitting; watch whether validation keeps falling.

The learning rate warms up linearly for `--warmup` steps, then follows a cosine
curve down to 10% of `--lr`. AdamW is used with gradient clipping at 1.0.

### Resuming

If training stops, run the same command again with `--resume`. Training picks up
from `runs/v2/last` and continues the step count, LR schedule and log file:

```bash
caffeinate -i uv run python train.py --data data/maestro_prepared data/aria_prepared \
    --out runs/v2 --batch 8 --steps 60000 --resume
```

Keep `--batch`, `--steps`, `--lr` and `--warmup` the same as the original run.

Only the optimiser's step counter is restored; its momentum buffers start fresh.

### Training options

| Flag             | Default         | Meaning                                         |
|------------------|-----------------|-------------------------------------------------|
| `--data`         | `data/maestro_prepared` | One or more folders. Training uses all their `train.npz` files; validation uses the first folder's `validation.npz` |
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

Run it on each dataset's test split:

```bash
uv run python evaluate.py --model runs/v2/best --data data/maestro_prepared --split test --keys
uv run python evaluate.py --model runs/v2/best --data data/aria_prepared --split test
```

This prints:

1. **Held-out loss** for each token family (pitch, velocity, duration, shift) and
   overall, with perplexity. On MAESTRO it compares directly with a MAESTRO-only
   model; the MAESTRO-only baseline `v1` scored 2.139 on the test split
   (`docs/results/small-model-v1.md`). On Aria it measures the new genres.
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

   Genre, era and composer can't be measured from notes, so they aren't in this
   table. Judge genre by ear: generate the same request with each `genre=` value
   and listen.

## 7. Generate music

To write a MIDI file:

```bash
uv run python sample.py --model runs/v2/best --request "slow and quiet in D minor" --out out.mid
uv run python sample.py --model runs/v2/best --request "a jazzy ballad in F major" --out jazz.mid
# or with exact tags
uv run python sample.py --model runs/v2/best --tags key=Dmin,dynamics=p,genre=film --out film.mid
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
| `genre`    | `classical`, `jazz`, `pop`, `film`, `ragtime`, `other` |
| `composer` | `bach`, `haydn`, `mozart`, `beethoven`, `schubert`, `chopin`, `schumann`, `liszt`, `mendelssohn`, `brahms`, `rachmaninoff`, `scriabin`, `debussy`, `ravel` |

## 8. Play in real time

```bash
uv run python play.py --list                                   # show MIDI output ports
uv run python play.py --model runs/v2/best --port "IAC Driver Bus 1" --request "calm, quiet, in F major"
```

While it plays, type a new request and press Enter to change the music. Ctrl-C
stops and releases every key. Without `--port`, the note events are printed
instead of sent, which is handy for testing. On macOS, turn on the IAC Driver in
Audio MIDI Setup to send notes to a DAW or software piano.

`sample.py` prints how fast generation runs compared with playback. Generation
has to be faster than the music for `play.py` to keep up. If it isn't, raise
`--lookahead` (default 1.0 s).

## 9. The Qwen version (Part II of the book)

`qwen/` fine-tunes `Qwen/Qwen3-0.6B-Base` to continue a plain-text request with
music, so any sentence works as a request. It uses PyTorch and `transformers` on an
NVIDIA GPU (the book uses a DGX Spark), not MLX. It trains on the same prepared
data as the small model: copy `data/maestro_prepared` and `data/aria_prepared`
(about 400 MB) to the GPU machine. The raw downloads are not needed there.

```bash
uv pip install torch transformers accelerate safetensors numpy mido python-rtmidi

# two-minute check, then the real run; validation uses the first --data folder
uv run python qwen/train_qwen.py --data data/maestro_prepared data/aria_prepared --out runs/qwen_test \
    --steps 60 --batch 4 --warmup 10 --eval-every 20 --eval-batches 2
nohup uv run python qwen/train_qwen.py --data data/maestro_prepared data/aria_prepared --out runs/qwen > qwen.log 2>&1 &
# add --resume to continue

uv run python qwen/play_qwen.py --model runs/qwen/best --out test.mid \
    --request "A slow, quiet piano piece in D minor, like Chopin."
```

Every passage gets a caption written from its measured tags, including the genre
(`qwen/captions.py`). Because validation is on MAESTRO only, the `val` figure can be
compared directly with the small model's validation loss.

## Full run, start to finish

```bash
uv venv --python 3.12
uv pip install mlx numpy mido python-rtmidi

curl -L -o data/maestro/maestro-v3.0.0-midi.zip --create-dirs \
  https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
unzip -q data/maestro/maestro-v3.0.0-midi.zip -d data/maestro
hf download loubb/aria-midi aria-midi-v1-deduped-ext.tar.gz --repo-type dataset --local-dir data/aria

uv run python prepare.py      --maestro data/maestro/maestro-v3.0.0 --out data/maestro_prepared
uv run python prepare_aria.py --archive data/aria/aria-midi-v1-deduped-ext.tar.gz --out data/aria_prepared
uv run python stats.py        --data data/aria_prepared

caffeinate -i uv run python train.py --data data/maestro_prepared data/aria_prepared \
    --out runs/v2 --batch 8 --steps 60000                                   # add --resume to continue

uv run python evaluate.py --model runs/v2/best --data data/maestro_prepared --split test --keys
uv run python evaluate.py --model runs/v2/best --data data/aria_prepared --split test
uv run python sample.py   --model runs/v2/best --request "a jazzy ballad in F major" --out out.mid
```

`data/` and `runs/` are git-ignored.
