"""Play in real time on a MIDI instrument.

    python play.py --list
    python play.py --port "IAC Driver Bus 1" --request "calm, quiet, in F major"

While it plays, type a new request and press Enter to change the music.
Ctrl-C stops (and silences every key).
"""
# argparse: command-line options; heapq: time-ordered event queue; queue: thread-safe inbox
# for typed requests; sys: stdin/stderr; threading: scheduler + keyboard threads; time: clock
import argparse, heapq, queue, sys, threading, time
# Performer: samples tokens from the model and assembles them into notes
from performer import Performer
# parse_request: free text -> tags dict; parse_tags: "key=value" spec -> tags dict
from request import parse_request, parse_tags


# fake output used when no --port is given, handy for testing without an instrument
class PrintPort:
    """Stands in for a MIDI port: prints what would be sent."""
    # same interface as MidiPort.send: kind is "note_on" or "note_off"
    def send(self, kind, pitch, velocity):
        # print a timestamp (seconds, wrapped at 1000 to stay short), event kind, pitch, velocity
        print(f"{time.monotonic() % 1000:8.3f}  {kind:<8} pitch {pitch:3d}  vel {velocity:3d}")
    # nothing to release for a printer
    def close(self):
        pass  # no-op


# wrapper around a real MIDI output (hardware synth, DAW, or a virtual bus)
class MidiPort:
    """A real MIDI output port, opened by name (see --list)."""
    # open the port called `name`
    def __init__(self, name):
        # imported here so the rest of the file works without mido installed
        import mido
        # keep the mido module (to build messages later) and the opened output port
        self.mido, self.port = mido, mido.open_output(name)
    # send one note event to the instrument
    def send(self, kind, pitch, velocity):
        # build a MIDI message ("note_on"/"note_off", key number, loudness) and send it
        self.port.send(self.mido.Message(kind, note=pitch, velocity=velocity))
    # shut the port down cleanly
    def close(self):
        self.port.reset()          # all notes off, so no key is left hanging
        self.port.close()          # release the port


# a background thread that fires note events at the right moment
class Scheduler(threading.Thread):
    """Holds generated notes and sends each event at its exact time.

    It runs on its own thread, so sending never waits for the model. Times
    are "music time": seconds since the start of the performance."""

    # set up the empty timeline for output `port`
    def __init__(self, port):
        # daemon=True: this thread will not keep the program alive after main exits
        super().__init__(daemon=True)
        # the output, and a lock guarding the shared state below (used by two threads)
        self.port, self.lock = port, threading.Lock()
        # heap of (time, order, count, kind, pitch, vel, id). At equal times
        # order puts note-offs (0) before note-ons (1); count breaks ties.
        # Tuples compare field by field, so the heap always pops the earliest event first;
        # releasing before striking lets a repeated key sound again instead of being cut.
        self.events = []
        self.last = {}         # pitch -> (id, off time) of its most recent note
        self.cut = set()       # ids of notes that were cut short by a re-strike
        self.count = 0         # ever-increasing counter for note ids and heap tie-breaks
        self.worst = 0.0       # largest lateness of any sent event, in seconds
        self.running = True    # cleared by play() to stop the thread loop
        self.t0 = time.monotonic() + 0.25          # small run-up before bar one

    # convert the wall clock into music time
    def now(self):
        """Current music time, in seconds."""
        # seconds elapsed since the (possibly shifted) start point t0; negative during run-up
        return time.monotonic() - self.t0

    # stretch the timeline when the model can't keep up
    def pause(self, seconds):
        """Move the whole timeline later (used when generation falls behind)."""
        # take the lock so the scheduler thread never sees a half-updated clock
        with self.lock:
            # moving the start later makes now() smaller, so every pending event (and the
            # music clock itself) is delayed by `seconds`; the music waits rather than rushing
            self.t0 += seconds

    # called by play() for every generated note
    def add_note(self, on, pitch, vel, dur):
        """Schedule a note-on at music time `on` and its note-off `dur` seconds later."""
        # the scheduler thread reads these structures, so modify them under the lock
        with self.lock:
            # give this note a unique id
            self.count += 1
            nid = self.count
            # the previous note on the same key, if any: (id, off time)
            old = self.last.get(pitch)
            if old and old[1] > on:                # key is still down: release it
                self.cut.add(old[0])               # first, and ignore the old note's
                self._push(on, 0, "note_off", pitch, 0, None)   # own note-off later
            # this note is now the latest on this key
            self.last[pitch] = (nid, on + dur)
            # strike the key at `on` (order 1: after any note-off at the same instant)
            self._push(on, 1, "note_on", pitch, vel, nid)
            # release it `dur` seconds later, tagged with the id so it can be cancelled if cut
            self._push(on + dur, 0, "note_off", pitch, 0, nid)

    # low-level insert into the event heap
    def _push(self, t, order, kind, pitch, vel, nid):
        """Add one event to the heap. Caller must hold the lock."""
        # a fresh count per event makes every tuple unique, so ties never compare kind/None
        self.count += 1
        # insert keeping the heap ordered by (time, order, count)
        heapq.heappush(self.events, (t, order, self.count, kind, pitch, vel, nid))

    # used by play() to wait for the tail of the performance
    def pending(self):
        """Number of events not yet sent."""
        # read without the lock: len() of a list is atomic enough for a polling check
        return len(self.events)

    # the body of the thread, started by sched.start()
    def run(self):
        """Thread loop: every millisecond, send all events whose time has come."""
        # keep going until play() sets running = False
        while self.running:
            # events to send in this tick
            due = []
            # hold the lock only while touching the heap
            with self.lock:
                # snapshot the music time once for this tick
                now = self.now()
                # pop every event whose time is at or before now (heap[0] is the earliest)
                while self.events and self.events[0][0] <= now:
                    # take it off the heap; the order and count fields are not needed now
                    t, _, _, kind, pitch, vel, nid = heapq.heappop(self.events)
                    # this note-off belongs to a note already released by a re-strike
                    if kind == "note_off" and nid in self.cut:
                        self.cut.discard(nid)      # already released early
                        continue                   # skip it so it doesn't silence the new note
                    # queue it for sending
                    due.append((kind, pitch, vel))
                    # track how late we are compared with the scheduled time
                    self.worst = max(self.worst, now - t)
            for event in due:                      # send outside the lock
                # (MIDI I/O can be slow; holding the lock would block add_note/pause)
                self.port.send(*event)
            # wait ~1 ms before checking again (sets the timing resolution)
            time.sleep(0.001)


# the main real-time loop: generate notes just ahead of the playback clock
def play(performer, port, lookahead=1.0, seconds=None, commands=None, to_request=None):
    """Keep `lookahead` seconds of music generated ahead of the clock.

    Runs until the model emits "end", `seconds` of music are generated, or
    Ctrl-C. Lines arriving on the `commands` queue are turned into a new
    request with `to_request`. Returns stats: notes played, how many times
    generation fell behind, and the worst send lateness in ms."""
    # create the scheduler for this port
    sched = Scheduler(port)
    # launch its thread; from now on it sends events as their time comes
    sched.start()
    head = 0.0         # music time of the latest generated note onset
    notes = late = 0   # notes scheduled so far, and times generation fell behind
    # try/finally so the scheduler and port are always shut down, even on Ctrl-C
    try:
        # loop forever, or until `seconds` of music have been generated
        while seconds is None or head < seconds:
            # a line was typed on the keyboard since the last check
            if commands is not None and not commands.empty():
                performer.set_request(to_request(commands.get()))   # new request typed
            # we are already `lookahead` seconds ahead of playback
            if head - sched.now() >= lookahead:
                time.sleep(0.002)                  # the buffer is full: rest
                continue                           # check again
            r = performer.step()                   # generate one token
            # r is "end", None (no note finished on this token),
            # or a finished note (wait_ms, pitch, vel, dur_ms)
            # the model chose to finish the piece
            if r == "end":
                break
            # a complete note came out
            if r:
                # unpack: gap since the previous onset, key, loudness, length (ms)
                wait_ms, pitch, vel, dur_ms = r
                # advance the onset time of the newest note (ms -> seconds)
                head += wait_ms / 1000
                # how far playback has already passed this note's onset (positive = too late)
                behind = sched.now() - head
                if behind > 0:                     # generation fell behind:
                    # shift the timeline so playback is 50 ms before this note again
                    sched.pause(behind + 0.05)     # pause the music, don't rush it
                    late += 1                      # count the underrun
                # hand the note to the scheduler (duration converted to seconds)
                sched.add_note(head, pitch, vel, dur_ms / 1000)
                notes += 1
        while sched.pending():                     # let the last notes finish
            time.sleep(0.01)
    # Ctrl-C: stop quietly and fall through to the cleanup
    except KeyboardInterrupt:
        pass  # no traceback
    # runs on every exit path
    finally:
        sched.running = False                      # ask the scheduler loop to exit
        sched.join()                               # wait until its thread has finished
        port.close()                               # silence everything and close the port
    # summary: notes played, underrun count, worst lateness converted to ms
    return {"notes": notes, "underruns": late,
            "worst_lateness_ms": round(sched.worst * 1000, 1)}


# lets the user type new requests while music keeps playing
def stdin_commands():
    """Lines typed on the keyboard arrive in a queue, without blocking."""
    # thread-safe queue; play() polls it with empty()/get()
    q = queue.Queue()
    # runs on its own thread because reading stdin blocks until Enter is pressed
    def reader():
        # iterate over lines as they are typed (ends when stdin closes)
        for line in sys.stdin:
            # ignore empty lines
            if line.strip():
                # pass the trimmed text to the main loop
                q.put(line.strip())
    # daemon thread, so a pending stdin read never stops the program from exiting
    threading.Thread(target=reader, daemon=True).start()
    # play() reads requests from this queue
    return q


# command-line entry point
def main():
    """Parse arguments, load the model and play until stopped."""
    # define the command-line options
    ap = argparse.ArgumentParser()
    # just print the available MIDI outputs and quit
    ap.add_argument("--list", action="store_true", help="list MIDI output ports")
    # checkpoint path prefix (without .json/.safetensors)
    ap.add_argument("--model", default="runs/v1/best")
    # name of the MIDI output to play on
    ap.add_argument("--port", default=None, help="MIDI port; omit to print events")
    # plain-English description of the music wanted, e.g. "calm, in F major"
    ap.add_argument("--request", default="")
    # conditioning tags given explicitly instead of via --request
    ap.add_argument("--tags", default="")
    # how far ahead of playback to keep generating (bigger = safer, slower to react)
    ap.add_argument("--lookahead", type=float, default=1.0, help="seconds generated ahead")
    # optional length limit, in seconds of music
    ap.add_argument("--seconds", type=float, default=None, help="stop after this much music")
    # sampling randomness: <1 safer and more repetitive, >1 more adventurous
    ap.add_argument("--temperature", type=float, default=1.0)
    # nucleus sampling: only sample from the most likely tokens covering 95% of probability
    ap.add_argument("--top-p", type=float, default=0.95)
    # random seed for reproducible performances
    ap.add_argument("--seed", type=int, default=None)
    # read sys.argv into `args`
    args = ap.parse_args()

    # --list mode
    if args.list:
        # imported here so the normal path only needs mido when a port is used
        import mido
        # one port name per line, or a message if there are none
        print("\n".join(mido.get_output_names()) or "no MIDI output ports found")
        return  # done; don't load the model

    # imported late so --list works without MLX
    from model import Pianist
    # backend that lets the Performer drive the MLX model with a KV cache
    from performer import MLXBackend
    # --tags gives the conditioning tags directly; otherwise parse them from --request
    tags = parse_tags(args.tags) if args.tags else parse_request(args.request)
    # show the tags on stderr so they don't mix with printed events on stdout
    print("tags:", tags, file=sys.stderr)
    # load the trained weights and config (the saved info dict is not needed)
    model, _ = Pianist.load(args.model)
    # wrap the model in a Performer with the chosen tags and sampling settings
    performer = Performer(MLXBackend(model), tags, args.temperature, args.top_p, seed=args.seed)
    # real MIDI output if a port was named, otherwise print events to the terminal
    port = MidiPort(args.port) if args.port else PrintPort()
    # play until stopped; typed lines become new requests through parse_request
    stats = play(performer, port, args.lookahead, args.seconds,
                 commands=stdin_commands(), to_request=parse_request)
    # report notes played, underruns and worst lateness
    print(stats, file=sys.stderr)


# only run main() when executed as a script, not when imported
if __name__ == "__main__":
    main()
