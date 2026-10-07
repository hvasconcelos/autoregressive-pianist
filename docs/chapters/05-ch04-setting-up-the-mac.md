# 4. Setting up the Mac

## 4.1 What you need

- A MacBook Pro with Apple silicon (an M-series chip). MLX does not run on Intel Macs.
- A recent macOS. MLX's installation page lists the minimum version it supports.
- `uv`, a fast tool that creates Python environments and installs packages. It also downloads a suitable Python for you, so you do not need to install one yourself. Install it with `brew install uv`, or with the one-line installer on uv's website.
- About 2 GB of free disk space.

## 4.2 Create the project

Unzip the code that came with this book, then make a private Python environment inside it so that the libraries do not interfere with anything else on the machine:

``` bash
cd pianist
uv venv --python 3.12
source .venv/bin/activate
uv pip install mlx numpy mido python-rtmidi
```

The first command creates the environment in a folder called `.venv`, the second switches this Terminal window to it, and the third installs the libraries into it. After the second command, plain `python` means the environment's Python.

MLX needs a native Apple-silicon Python. `uv` picks one automatically on an M-series Mac. To confirm, run `python -c "import platform; print(platform.processor())"`, which must print `arm`. If it prints `i386`, your Terminal is running under Rosetta and MLX will not install.

| Library | What it does here |
|----|----|
| `mlx` | Apple's array and neural-network framework. It runs the model on the Mac's GPU. |
| `numpy` | Arrays on the CPU, used for data preparation. |
| `mido` | Reads and writes MIDI files and messages. |
| `python-rtmidi` | Lets `mido` send messages to real MIDI ports. |

Run `source .venv/bin/activate` again in each new Terminal window before using the project.

Check the installation:

``` bash
python model.py
```

Expected output:

``` text
19.1 M parameters
logits: (2, 16, 459)
```

## 4.3 A few words about MLX

MLX will feel familiar if you have used NumPy or PyTorch. Three things are different and worth knowing before you read the code.

**It is lazy.** Writing `c = a + b` does not compute anything. It records that `c` is the sum of `a` and `b`. The computation happens when you ask for the result, for example by printing it, calling `.item()`, or calling `mx.eval(c)`. The training loop calls `mx.eval` once per step to force the work to happen.

**Memory is shared.** On Apple silicon the CPU and GPU use the same memory, so there is no copying of arrays "to the GPU". The amount of memory you have for training is simply the Mac's RAM, minus what everything else is using.

**Gradients are functions.** You do not call `.backward()` on a loss. You wrap the loss function with `nn.value_and_grad`, and the wrapped function returns both the loss and the gradients.

## 4.4 Set up something to play on

The model sends MIDI. Something has to turn that into sound. You have two options.

**A digital piano or keyboard.** Connect it by USB. It will appear as a MIDI port with the instrument's name.

**A software piano on the Mac.** macOS has a built-in virtual MIDI cable called the IAC Driver. To switch it on:

1.  Open the **Audio MIDI Setup** app (in Applications, Utilities).
2.  From the **Window** menu choose **Show MIDI Studio**.
3.  Double-click **IAC Driver** and tick **Device is online**.

A port named "IAC Driver Bus 1" now exists. Open GarageBand, create a project with a software instrument track and choose a piano. GarageBand listens to every MIDI input, including the IAC bus, so anything sent to that port plays through the piano.

List the ports Python can see:

``` bash
python play.py --list
```

You should see the IAC bus, your keyboard, or both. Note the exact name; you will pass it to `play.py` in chapter 13.

::: {.admonition .note}
Listening to MIDI files

`sample.py` writes `.mid` files. To hear one, drag it into a GarageBand project, or open it in any other program that plays MIDI. QuickTime Player does not.
:::
