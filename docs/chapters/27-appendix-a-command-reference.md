# Appendix A. Command reference

## Part I, on the Mac

``` bash
# once
uv venv --python 3.12 && source .venv/bin/activate
uv pip install mlx numpy mido python-rtmidi

# data
mkdir -p data && cd data
curl -O https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
unzip -q maestro-v3.0.0-midi.zip && cd ..
python prepare.py --maestro data/maestro-v3.0.0 --out data/prepared
python stats.py --data data/prepared

# smoke test
python make_toy_data.py --out data/toy
python prepare.py --maestro data/toy --out data/toy_prepared
python train.py --data data/toy_prepared --out runs/toy --steps 1500 --batch 8 \
    --ctx 192 --dim 96 --layers 2 --heads 4 --warmup 100 --eval-every 250 \
    --eval-batches 4 --lr 1e-3
python evaluate.py --model runs/toy/best --data data/toy_prepared --batches 4 --batch 8

# train
caffeinate -i python train.py --data data/prepared --out runs/v1
python train.py --data data/prepared --out runs/v1 --resume

# use
python sample.py --model runs/v1/best --request "slow and quiet in D minor" --out out.mid
python play.py --list
python play.py --model runs/v1/best --port "IAC Driver Bus 1" --request "calm, in F major"
python evaluate.py --model runs/v1/best --data data/prepared
```

## Part II, on the DGX Spark

``` bash
# all commands are run inside the pianist folder
uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
uv pip install transformers accelerate safetensors numpy mido python-rtmidi

python qwen/train_qwen.py --data data/prepared --out runs/qwen_test \
    --steps 60 --batch 4 --warmup 10 --eval-every 20 --eval-batches 2
nohup python qwen/train_qwen.py --data data/prepared --out runs/qwen > qwen.log 2>&1 &
python qwen/train_qwen.py --data data/prepared --out runs/qwen --resume

python qwen/play_qwen.py --model runs/qwen/best --out test.mid \
    --request "A calm, quiet piano piece in F major."
python qwen/play_qwen.py --model runs/qwen/best --port "<port name>" \
    --request "Something stormy and virtuosic."
```

## Options of train.py

| Option | Default | Meaning |
|----|----|----|
| `--data` | `data/prepared` | Folder with `train.npz` and `validation.npz` |
| `--out` | `runs/v1` | Where checkpoints and the log go |
| `--steps` | 20000 | Number of parameter updates |
| `--batch` | 32 | Sequences per step |
| `--ctx` | 1024 | Tokens per sequence |
| `--dim` | 512 | Vector width |
| `--layers` | 6 | Number of transformer blocks |
| `--heads` | 8 | Attention heads; must divide `--dim` |
| `--dropout` | 0.1 | Dropout rate |
| `--lr` | 3e-4 | Peak learning rate |
| `--warmup` | 1000 | Warm-up steps |
| `--eval-every` | 500 | Steps between evaluations and saves |
| `--eval-batches` | 20 | Validation batches per evaluation |
| `--resume` | off | Continue from `last` in the output folder |
| `--no-compile` | off | Do not use `mx.compile` |
