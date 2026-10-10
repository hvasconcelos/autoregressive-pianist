# 17. Setting up the DGX Spark

## 17.1 The machine

The DGX Spark is a small desktop computer with an NVIDIA GB10 chip, 128 GB of memory shared between CPU and GPU, and an ARM processor. Two of those facts shape the setup.

**It is ARM, not x86.** Python packages that contain compiled code need an ARM64 Linux build with CUDA support. A plain `uv pip install torch` may fetch a build without GPU support.

**Memory is unified.** There is no separate GPU memory. A job that uses too much does not fail with a tidy "CUDA out of memory" error; it can make the whole machine unresponsive. Start with small batches and increase.

## 17.2 Install the software

The commands below follow a community setup guide for this machine (listed in the sources) and were not run for this book. NVIDIA also publishes its own setup instructions and ready-made containers for the Spark; if anything here fails, use those.

First copy the project folder from the Mac, including `data/maestro_prepared` and `data/aria_prepared`. Those six `.npz` files, about 400 MB, are all the data the Spark needs; the raw downloads in `data/maestro` and `data/aria` can stay behind. Run this on the Mac, from the folder that contains `pianist`, with `spark` replaced by the Spark's network name:

``` bash
rsync -av pianist/ spark:pianist/ --exclude .venv --exclude runs \
    --exclude data/maestro --exclude data/aria
```

Then, on the Spark:

``` bash
cd pianist
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
uv pip install transformers accelerate safetensors numpy mido python-rtmidi
```

If `uv` is not yet on the Spark, install it first with the one-line installer on uv's website. Every command in Part II is run from inside the `pianist` folder with this environment active. If `python-rtmidi` fails to build, it is only needed for sending notes to a MIDI port from the Spark; leave it out and install the ALSA development package (`libasound2-dev` on Ubuntu) before trying again.

Check that PyTorch sees the GPU:

``` bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The code in this part was tested with Transformers 5.19; install a current version and not an old pinned one.

The second value printed by the check must be `True`. PyTorch may print a warning about the GPU's capability level; the guide reports that it is safe to ignore.

Download the model once, so that later runs work offline:

``` bash
python -c "from transformers import AutoModelForCausalLM as M, AutoTokenizer as T; \
n='Qwen/Qwen3-0.6B-Base'; T.from_pretrained(n); M.from_pretrained(n)"
```
