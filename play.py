"""Play in real time on a MIDI instrument.

    python play.py --list
    python play.py --port "IAC Driver Bus 1" --request "calm, quiet, in F major"

While it plays, type a new request and press Enter to change the music.
Ctrl-C stops (and silences every key).
"""
import argparse, heapq, queue, sys, threading, time
from performer import Performer
from request import parse_request, parse_tags


class PrintPort:
    """Stands in for a MIDI port: prints what would be sent."""
    def send(self, kind, pitch, velocity):
        print(f"{time.monotonic() % 1000:8.3f}  {kind:<8} pitch {pitch:3d}  vel {velocity:3d}")
    def close(self):
        pass


class MidiPort:
    """A real MIDI output port, opened by name (see --list)."""
    def __init__(self, name):
        # imported here so the rest of the file works without mido installed
        import mido
        self.mido, self.port = mido, mido.open_output(name)
    def send(self, kind, pitch, velocity):
        self.port.send(self.mido.Message(kind, note=pitch, velocity=velocity))
    def close(self):
        self.port.reset()          # all notes off
        self.port.close()


class Scheduler(threading.Thread):
    """Holds generated notes and sends each event at its exact time.

    It runs on its own thread, so sending never waits for the model. Times
    are "music time": seconds since the start of the performance."""

    def __init__(self, port):
        super().__init__(daemon=True)
        self.port, self.lock = port, threading.Lock()
        # heap of (time, order, count, kind, pitch, vel, id). At equal times
        # order puts note-offs (0) before note-ons (1); count breaks ties.
        self.events = []
        self.last = {}         # pitch -> (id, off time) of its most recent note
        self.cut = set()       # ids of notes that were cut short by a re-strike
        self.count = 0
        self.worst = 0.0       # largest lateness of any sent event, in seconds
        self.running = True
        self.t0 = time.monotonic() + 0.25          # small run-up before bar one

    def now(self):
        """Current music time, in seconds."""
        return time.monotonic() - self.t0

    def pause(self, seconds):
        """Move the whole timeline later (used when generation falls behind)."""
        with self.lock:
            self.t0 += seconds

    def add_note(self, on, pitch, vel, dur):
        """Schedule a note-on at music time `on` and its note-off `dur` seconds later."""
        with self.lock:
            self.count += 1
            nid = self.count
            old = self.last.get(pitch)
            if old and old[1] > on:                # key is still down: release it
                self.cut.add(old[0])               # first, and ignore the old note's
                self._push(on, 0, "note_off", pitch, 0, None)   # own note-off later
            self.last[pitch] = (nid, on + dur)
            self._push(on, 1, "note_on", pitch, vel, nid)
            self._push(on + dur, 0, "note_off", pitch, 0, nid)

    def _push(self, t, order, kind, pitch, vel, nid):
        """Add one event to the heap. Caller must hold the lock."""
        self.count += 1
        heapq.heappush(self.events, (t, order, self.count, kind, pitch, vel, nid))

    def pending(self):
        """Number of events not yet sent."""
        return len(self.events)

    def run(self):
        """Thread loop: every millisecond, send all events whose time has come."""
        while self.running:
            due = []
            with self.lock:
                now = self.now()
                while self.events and self.events[0][0] <= now:
                    t, _, _, kind, pitch, vel, nid = heapq.heappop(self.events)
                    if kind == "note_off" and nid in self.cut:
                        self.cut.discard(nid)      # already released early
                        continue
                    due.append((kind, pitch, vel))
                    self.worst = max(self.worst, now - t)
            for event in due:                      # send outside the lock
                self.port.send(*event)
            time.sleep(0.001)


def play(performer, port, lookahead=1.0, seconds=None, commands=None, to_request=None):
    """Keep `lookahead` seconds of music generated ahead of the clock.

    Runs until the model emits "end", `seconds` of music are generated, or
    Ctrl-C. Lines arriving on the `commands` queue are turned into a new
    request with `to_request`. Returns stats: notes played, how many times
    generation fell behind, and the worst send lateness in ms."""
    sched = Scheduler(port)
    sched.start()
    head = 0.0         # music time of the latest generated note onset
    notes = late = 0
    try:
        while seconds is None or head < seconds:
            if commands is not None and not commands.empty():
                performer.set_request(to_request(commands.get()))   # new request typed
            if head - sched.now() >= lookahead:
                time.sleep(0.002)                  # the buffer is full: rest
                continue
            r = performer.step()                   # generate one token
            # r is "end", None (no note finished on this token),
            # or a finished note (wait_ms, pitch, vel, dur_ms)
            if r == "end":
                break
            if r:
                wait_ms, pitch, vel, dur_ms = r
                head += wait_ms / 1000
                behind = sched.now() - head
                if behind > 0:                     # generation fell behind:
                    sched.pause(behind + 0.05)     # pause the music, don't rush it
                    late += 1
                sched.add_note(head, pitch, vel, dur_ms / 1000)
                notes += 1
        while sched.pending():                     # let the last notes finish
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        sched.running = False
        sched.join()
        port.close()
    return {"notes": notes, "underruns": late,
            "worst_lateness_ms": round(sched.worst * 1000, 1)}


def stdin_commands():
    """Lines typed on the keyboard arrive in a queue, without blocking."""
    q = queue.Queue()
    def reader():
        for line in sys.stdin:
            if line.strip():
                q.put(line.strip())
    threading.Thread(target=reader, daemon=True).start()
    return q


def main():
    """Parse arguments, load the model and play until stopped."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list MIDI output ports")
    ap.add_argument("--model", default="runs/v1/best")
    ap.add_argument("--port", default=None, help="MIDI port; omit to print events")
    ap.add_argument("--request", default="")
    ap.add_argument("--tags", default="")
    ap.add_argument("--lookahead", type=float, default=1.0, help="seconds generated ahead")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this much music")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.list:
        import mido
        print("\n".join(mido.get_output_names()) or "no MIDI output ports found")
        return

    # imported late so --list works without MLX
    from model import Pianist
    from performer import MLXBackend
    # --tags gives the conditioning tags directly; otherwise parse them from --request
    tags = parse_tags(args.tags) if args.tags else parse_request(args.request)
    print("tags:", tags, file=sys.stderr)
    model, _ = Pianist.load(args.model)
    performer = Performer(MLXBackend(model), tags, args.temperature, args.top_p, seed=args.seed)
    port = MidiPort(args.port) if args.port else PrintPort()
    stats = play(performer, port, args.lookahead, args.seconds,
                 commands=stdin_commands(), to_request=parse_request)
    print(stats, file=sys.stderr)


if __name__ == "__main__":
    main()
