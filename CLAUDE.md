# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small GPT-style model, trained on the MAESTRO piano dataset, that generates expressive solo piano as MIDI. It is conditioned on tags (density, dynamics, register, key, era, composer). It is written in MLX, so it runs on Apple Silicon only. Every script is a flat top-level module with an `argparse` CLI. There is no package, no `pyproject.toml`, no test suite and no linter. Dependencies are `mlx numpy mido` (plus `python-rtmidi` for live ports in `play.py`, and `mlx_lm` only for `request.py --llm`). The user runs the scripts through `uv run python ...` with a local `.venv`.

## Commands

```bash
# end-to-end smoke test on fake data (minutes); use this to check changes
uv run python make_toy_data.py --out data/toy
uv run python prepare.py --maestro data/toy --out data/toy_prepared
uv run python train.py --data data/toy_prepared --out runs/toy --steps 1500 --ctx 192 --dim 96 --layers 2 --heads 4 --eval-every 250
uv run python evaluate.py --model runs/toy/best --data data/toy_prepared
uv run python sample.py --model runs/toy/best --request "fast and loud" --out toy.mid

# quick self-checks
uv run python tokenizer.py                          # encode/decode round trip, prints vocab size
uv run python request.py "slow and quiet in D minor" # shows which tags a request becomes
uv run python stats.py --data data/prepared          # tag value distribution in a split

# real run (MAESTRO unzipped to data/maestro-v3.0.0)
uv run python prepare.py --maestro data/maestro-v3.0.0 --out data/prepared
uv run python train.py --data data/prepared --out runs/v1 --batch 8   # add --resume to continue
```

On the user's 16 GB M1 Pro, the default `--batch 32` with `--ctx 1024` swaps heavily and makes no visible progress. Use `--batch 8`. Training prints a line only every 50 steps, and the first one is delayed further because `mx.compile` traces the training step once. `--no-compile` gives readable tracebacks.

## Architecture

The data flow is: MIDI → note arrays → (`prepare.py`) `.npz` per split → (`data.py`) tokens built on the fly per batch → `train.py` → checkpoint → `Performer` → notes → MIDI or a live port.

- **Note array** is the common currency everywhere: a float64 `(N, 4)` array of `(onset_ms, pitch, velocity, dur_ms)`. `midi_io.py` converts to and from MIDI and folds the sustain pedal into note durations.
- **Prepared data stores notes, not tokens.** Each `.npz` has `notes` (all pieces concatenated), `offsets` (piece boundaries) and `meta` (JSON with per-piece `era`/`composer`). As a result, tokenizer changes and changes to the density/dynamics/register/key thresholds in `tags.py` do **not** require re-running `prepare.py`. Changes to `ERA` or `composer_tags` **do**, because those values are written into `meta` at prepare time.
- **Tokenizer (`tokenizer.py`)**: each note is `[Shift…] Pitch Vel Dur`. Shift tokens encode the wait since the previous onset in 10 ms steps up to 1 s, repeated for longer gaps. Chord notes have no Shift between them. The vocabulary (459 tokens) is laid out in fixed contiguous ranges: specials, tags, Pitch, Vel, Dur, Shift. Code tells token families apart by comparing ids against `TAG0/PITCH0/VEL0/DUR0/SHIFT0` (`kind()`), and `evaluate.py` splits losses by those ranges. Keep that ordering if you add tokens. `allowed_next()` is the note grammar that is enforced at sampling time.
- **Sequence format**: `<bos> tags… <sep> music… [<eos>]`. Tags always follow `TAG_ORDER`, and any of them may be missing.
- **Tags are measured, not labelled** (`data.py` `Dataset.passage`). Each training passage is sampled, augmented (±3 semitones of transposition, ±10% tempo, a velocity offset) and trimmed to whole notes. Then `tags.compute_tags` measures density, dynamics, register and key (Krumhansl-Schmuckler) on the notes actually kept, and era/composer are added from `meta`. `drop_tags` randomly removes tags so the model learns to handle partial requests. The loss mask covers only music tokens and `<eos>`, never tags or padding. Validation uses `fixed_batches`, with no augmentation and no tag dropout.
- **Model (`model.py`)**: `Pianist` is a pre-norm transformer with rotary positions and a per-layer key/value cache. With a cache it accepts **one token at a time** only. `Pianist.save/load` write `<path>.safetensors` plus `<path>.json`, which holds the `Config` and extra info (`step`, `best`). A checkpoint is referred to by its path prefix without extension (`runs/v1/best`). `train.py` writes `best` (on a new lowest validation loss) and `last` (at every evaluation) to `--out`, and appends to `log.csv`.
- **Generation (`performer.py`)**: `Performer` is independent of the model. It talks to a backend with `start(request, music_tokens)` / `step(token)`, both returning NumPy logits over this vocabulary. `MLXBackend` wraps `Pianist`, and the docstring anticipates a second (Qwen) backend. Sampling applies the grammar mask, then temperature, then top-p. In `endless` mode `<eos>` is masked out, and when the context fills the oldest music is trimmed on note boundaries and the context is rebuilt. `set_request()` changes tags in the middle of a performance while keeping recent music. `sample.py` (offline) and `play.py` (real-time scheduler thread, typed requests on stdin) are both thin wrappers around it.
- **Qwen version (`qwen/`)**: fine-tunes `Qwen/Qwen3-0.6B-Base` in PyTorch/`transformers` (not MLX) to continue a plain-text caption with music. `captions.py` turns measured tags into varied sentences. `music_lm.py` adds the music tokens to Qwen's vocabulary as one contiguous id block and scores only those rows. `qwen_backend.py` is the second `Performer` backend. Run the scripts from the project folder (`uv run python qwen/train_qwen.py`); `music_lm.py` puts the project folder on `sys.path` so they can import the Part I modules.
- **Evaluation (`evaluate.py`)**: reports held-out loss per token family, then tag adherence. For tag adherence it generates with one tag at a time (fixed seeds, no dataset, so `--split` doesn't affect it) and measures the result with `compute_tags`. It reports `exact` and `near`. For ordered tags, `near` means within one step. For key, `near` means the key is in `close_keys()`: the same key, its relative major/minor, or a key a fifth away.
- **Requests (`request.py`)**: `parse_request` maps plain words to tags with a keyword table (the longest match wins) plus key/mood regexes. `parse_tags` parses exact `key=value` specs. Tag values must come from `tokenizer.TAG_VALUES`, which is the single source of truth for every tag category.

## Conventions

- Keep comments sparse: a docstring per module and function, plus short inline comments only at the non-obvious points (units, grid layouts, tricky indexing, why a step exists). Don't comment lines that read clearly on their own.
- `data/` and `runs/` are git-ignored, so never commit datasets or checkpoints.
