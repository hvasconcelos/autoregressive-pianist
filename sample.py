"""Generate a performance into a MIDI file (not in real time).

    python sample.py --model runs/v1/best --request "slow and quiet in D minor" --out out.mid
"""
# argparse: command-line options; time: measure generation speed
import argparse, time
# the trained GPT-style model class (used to load a checkpoint)
from model import Pianist
# the generation engine and its adapter for the MLX model
from performer import MLXBackend, Performer
# writes a note array out as a .mid file
from midi_io import save_notes
# turn plain-English requests or exact "key=value" specs into a tags dict
from request import parse_request, parse_tags


def main():
    """Generate --notes notes for the request and save them as MIDI."""
    # command-line interface
    ap = argparse.ArgumentParser()
    # checkpoint path without extension (loads <path>.safetensors and <path>.json)
    ap.add_argument("--model", default="runs/v1/best")
    # what to play, in plain words, e.g. "slow and quiet in D minor"
    ap.add_argument("--request", default="", help="plain words")
    # exact tags instead of plain words; takes priority over --request when given
    ap.add_argument("--tags", default="", help="exact tags, e.g. key=Dmin,dynamics=p")
    # how many notes to generate
    ap.add_argument("--notes", type=int, default=400)
    # sampling randomness: lower = safer and more repetitive, higher = more adventurous
    ap.add_argument("--temperature", type=float, default=1.0)
    # nucleus size: sample only from the most likely tokens covering this much probability
    ap.add_argument("--top-p", type=float, default=0.95)
    # random seed for reproducible output (None = different every run)
    ap.add_argument("--seed", type=int, default=None)
    # output MIDI file
    ap.add_argument("--out", default="out.mid")
    # read the options from the command line
    args = ap.parse_args()

    # build the tags dict: from exact --tags if given, otherwise by interpreting the plain-words request
    tags = parse_tags(args.tags) if args.tags else parse_request(args.request)
    # show what the model is actually being conditioned on
    print("tags:", tags)
    # load weights and config; `info` (extra saved metadata) is not used here
    model, info = Pianist.load(args.model)
    # A Performer driving this model with the tags and sampling settings (endless by default)
    p = Performer(MLXBackend(model), tags, args.temperature, args.top_p, seed=args.seed)
    # start the stopwatch
    t0 = time.time()
    # generate the notes as an array of (onset_ms, pitch, velocity, dur_ms)
    notes = p.notes(args.notes)
    # seconds spent generating
    dt = time.time() - t0
    # write them to the MIDI file
    save_notes(notes, args.out)
    # length of the music in seconds: when the last-starting note ends (onset + duration, ms -> s)
    music_s = (notes[-1, 0] + notes[-1, 3]) / 1000
    # summary: note count, music length, output file
    print(f"{len(notes)} notes, {music_s:.1f} s of music -> {args.out}")
    # for real-time play (play.py) the first number must beat the second:
    # generation speed vs. how many tokens per second of music this piece contains
    print(f"generated {p.n_tokens / dt:.0f} tokens/s; "
          f"this music needs {p.n_tokens / music_s:.0f} tokens/s")


# run main() only when executed as a script, not when imported
if __name__ == "__main__":
    main()
