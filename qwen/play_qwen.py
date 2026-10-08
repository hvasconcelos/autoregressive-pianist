"""Real-time playing with the fine-tuned Qwen. The request is plain text.

    uv run python qwen/play_qwen.py --model runs/qwen/best --port "..." \
        --request "A slow, quiet piano piece in D minor, like Chopin."
"""
import argparse, sys
from qwen_backend import QwenBackend   # first: it puts the project folder on the import path
from performer import Performer
from play import MidiPort, PrintPort, play, stdin_commands
from midi_io import save_notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/qwen/best")
    ap.add_argument("--request", default="A calm piano piece.")
    ap.add_argument("--port", default=None)
    ap.add_argument("--out", default=None, help="write a MIDI file instead of playing")
    ap.add_argument("--notes", type=int, default=400)
    ap.add_argument("--lookahead", type=float, default=2.0)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    performer = Performer(QwenBackend(args.model), args.request,
                          args.temperature, args.top_p, seed=args.seed)
    if args.out:
        save_notes(performer.notes(args.notes), args.out)
        return
    port = MidiPort(args.port) if args.port else PrintPort()
    stats = play(performer, port, args.lookahead, args.seconds,
                 commands=stdin_commands(), to_request=lambda text: text)   # typed text is the request as is
    print(stats, file=sys.stderr)


if __name__ == "__main__":
    main()
