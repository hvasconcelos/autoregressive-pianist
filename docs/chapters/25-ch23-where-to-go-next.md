# 23. Where to go next

## 23.1 More data

The single most effective improvement to either model is still more music. The Aria-MIDI slice in chapter 7 uses 38,464 of the deduplicated subset's 371,053 files, chosen to fit 16 GB of memory and a night of training. On a machine with more memory and time, raise `--notes`; past what fits in memory, `Dataset` would need to read notes from disk instead of loading them all. The pruned subset (820,944 files) goes further still, at some cost in cleanliness.

With much more data, raise the model size as well: more layers and a wider vector, in that order. As a guide, 20 tokens of training per parameter is a good balance, so the 227 million tokens used here suit a model of about 10 to 20 million parameters, which is what we have.

## 23.2 Longer memory

The wandering quality of long performances comes from the 25-second window. Doubling the context to 2,048 tokens doubles the memory at roughly four times the attention cost. A cheaper trick is to add a coarse "plan" to the prefix: tags for the section the music is in (opening, build, climax, close) that your code advances on a timer.

## 23.3 The pedal

To give the model a real sustain pedal, add two tokens, `Pedal_down` and `Pedal_up`, call `load_notes` with `use_pedal=False` so that durations are the physical key-hold times, and insert the pedal events into the token stream at their times. The grammar gains one rule, and the player sends control change 64.

## 23.4 Synthesiser parameters

The original idea behind this project was a model that plays any instrument, sending both notes and synthesiser settings. The piano version is the foundation for that, and the extension follows the same pattern as the pedal.

- Add a token family per continuous parameter, for example 32 levels each of filter cutoff, resonance and envelope times.
- Insert parameter-change tokens between notes, with the same Shift tokens providing their timing.
- Put the starting patch in the prefix, as tags or as words in the caption.
- In the player, send those tokens as MIDI control changes or as OSC messages.

The hard part is data. There is no MAESTRO for synthesiser performances. The practical route is to record your own: play the target synthesiser with its controls mapped to MIDI controllers, record everything, and fine-tune a model that has already learned notes from piano data. A few hours of your own playing, on top of a model that already understands music, goes a long way.
