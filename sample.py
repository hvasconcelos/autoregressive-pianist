"""Generate a performance into a MIDI file (not in real time).

    python sample.py --model runs/v1/best --request "slow and quiet in D minor" --out out.mid
"""
import argparse, time
from model import Pianist
from performer import MLXBackend, Performer
from midi_io import save_notes
from request import parse_request, parse_tags


def main():
    """Generate --notes notes for the request and save them as MIDI."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/v1/best")
    ap.add_argument("--request", default="", help="plain words")
    ap.add_argument("--tags", default="", help="exact tags, e.g. key=Dmin,dynamics=p")
    ap.add_argument("--notes", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="out.mid")
    args = ap.parse_args()

    tags = parse_tags(args.tags) if args.tags else parse_request(args.request)
    print("tags:", tags)
    model, info = Pianist.load(args.model)
    p = Performer(MLXBackend(model), tags, args.temperature, args.top_p, seed=args.seed)
    t0 = time.time()
    notes = p.notes(args.notes)
    dt = time.time() - t0
    save_notes(notes, args.out)
    music_s = (notes[-1, 0] + notes[-1, 3]) / 1000
    print(f"{len(notes)} notes, {music_s:.1f} s of music -> {args.out}")
    # for real-time play (play.py) the first number must beat the second
    print(f"generated {p.n_tokens / dt:.0f} tokens/s; "
          f"this music needs {p.n_tokens / music_s:.0f} tokens/s")


if __name__ == "__main__":
    main()
