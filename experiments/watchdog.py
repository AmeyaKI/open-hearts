"""Phase 7 Task F0: a watchdog that only ever LOGS, never kills anything.

WHY THIS EXISTS, PLAINLY (Session Lessons 1+3, PHASE6_PLAN.md). Long runs used
to be watched by an ad-hoc shell loop that sampled RSS in a terminal and
nothing else. Twice that wasn't enough: once a laptop was put to sleep mid-run
and sat 13 hours at zero progress before anyone noticed, and separately a
25-deal block could pin one worker alone for 40 minutes with no way to tell,
from the outside, whether it was almost done or simply frozen. This script
watches one or more block-checkpoint partial files (the `name@block J{...}`
files `blockdriver.run_blocks` and `run_exploit_eval.run_row` append to) from
OUTSIDE the run -- a separate process, so it survives the run being paused,
resumed, or killed -- and raises its voice in the log when either failure mode
is happening. It never sends a signal to anything and never deletes a file:
its only output is lines of text, because deciding whether a slow run is
"frozen" or "just doing the last, hard block" needs a human, not a script.

TWO ALARMS, BOTH SCOPED PER WATCHED FILE (a deliberate reading of the F0
brief, which described the first-block deadline as a condition over "every
watched partial" collectively -- see the design note below).
  FIRST-BLOCK DEADLINE. If `k * block_eta` seconds pass, counted from when
  THIS WATCHDOG started, and a given watched partial has not grown by even
  one line, something is almost certainly wrong with THAT row before it has
  produced any checkpoint at all (the frozen-laptop case) -- fires once per
  file.
  STALL. Once a file HAS produced at least one block, if `stall_after`
  seconds pass with no further growth, that row has stopped making progress
  after a good start -- fires once per file. Defaults to `k * block_eta` if
  not given explicitly, so one number (the projected per-block time) covers
  both alarms unless the caller wants to tune them separately.

DESIGN NOTE / DEVIATION FROM THE LITERAL BRIEF. The brief phrased the
first-block deadline as firing when the deadline elapses "with zero new
lines in every watched partial" -- read most literally, a single global
alarm that requires ALL watched files to be stuck before it makes a sound,
so one live row would silence the alarm for a dead sibling row in the same
`--partial` invocation. This implementation fires the deadline (and the
stall alarm) independently per file instead: a row that never produces its
first block is worth reporting on its own, even while another row in the
same invocation is progressing normally. Each file's firing is still
independently counted, so the exit summary's totals reflect exactly how
many rows tripped which alarm.

Both alarms are LOUD (printed AND appended to the log) and BOTH COUNT THEIR
OWN FIRINGS -- an alarm that can pass a test by silently never running is a
bug, so `--log` and the exit summary line always report `checks=`,
`first_block_firings=`, `stall_firings=`, `mem_alerts=` so a reviewer can see
the watchdog actually looked, not just that it printed nothing alarming.

Usage:
    python experiments/watchdog.py \\
        --partial results/exploit_R1v_partial.txt \\
        --block-eta 90 --k 3 --interval 15
"""
import argparse
import glob
import os
import subprocess
import sys
import time

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def line_count(path):
    """Non-blank, non-comment lines in a partial file. Missing file -> 0,
    not an error: the file may not exist yet when the watchdog starts."""
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f
                  if line.strip() and not line.strip().startswith("#"))


def total_rss_gb(pattern=".venv/bin/python"):
    """Sum RSS (GB) over every process whose command line contains `pattern`.

    Deliberately NOT the `ps -g <pgid>` trick `run_exploit_eval.total_rss_gb`
    uses: that only sees the caller's own process group, but this watchdog
    runs as its OWN separate process (that is the whole point -- it must
    keep watching across a pause/resume of the experiment, which changes
    the experiment's pgid). Pattern-matching on the venv interpreter path
    instead finds every worker regardless of which group it landed in.
    """
    out = subprocess.run(["ps", "-axo", "rss=,command="],
                         capture_output=True, text=True).stdout
    total_kb = 0
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        rss_str, cmd = parts
        if pattern in cmd:
            try:
                total_kb += int(rss_str)
            except ValueError:
                continue
    return total_kb / (1024 ** 2)


class WatchState:
    """Per-partial-file bookkeeping. Plain object (not a dict) so tests can
    construct and inspect it directly without touching the filesystem."""

    def __init__(self, path):
        self.path = path
        self.last_count = 0
        self.last_growth_t = None       # None until the file first grows
        self.first_block_fired = False
        self.stall_fired = False


def check_once(states, now, start_t, k, block_eta, stall_after, mem,
               mem_limit_gb, counters, log_fn, count_fn=line_count):
    """One watchdog tick -- the testable loop body.

    Pure w.r.t. the clock and the alarm math (`now`, `start_t`, `mem` are all
    injected, never read from `time.time()`/`total_rss_gb()` here), so a test
    can fire the deadline or the stall alarm without sleeping or touching a
    real file. The one impurity is `count_fn`, which reads a partial file's
    line count; tests inject a fake there too.

    Mutates `states` (per-file growth bookkeeping) and `counters` (firing
    totals -- keys "checks", "first_block_firings", "stall_firings",
    "mem_alerts") in place. Calls `log_fn(line)` once per line this tick
    wants recorded: always the progress line, plus one line per alarm that
    fires. Returns nothing; the caller reads `counters` for the exit summary.
    """
    counters["checks"] += 1
    for st in states:
        n = count_fn(st.path)
        if n > st.last_count:
            st.last_count = n
            st.last_growth_t = now

    elapsed = now - start_t
    deadline = k * block_eta
    if elapsed >= deadline:
        for st in states:
            if st.last_growth_t is None and not st.first_block_fired:
                st.first_block_fired = True
                counters["first_block_firings"] += 1
                log_fn(f"WATCHDOG DEADLINE FIRED (first block): "
                      f"{st.path} produced no block in {elapsed:.0f}s "
                      f"(deadline {deadline:.0f}s = {k}x{block_eta}s "
                      f"block-eta) -- the run may be frozen")

    for st in states:
        if st.last_growth_t is None:
            continue
        since = now - st.last_growth_t
        if since >= stall_after and not st.stall_fired:
            st.stall_fired = True
            counters["stall_firings"] += 1
            log_fn(f"WATCHDOG STALL FIRED: {st.path} produced no new "
                  f"block for {since:.0f}s (stall-after {stall_after:.0f}s) "
                  f"after starting fine -- the run may have stalled")

    if mem > mem_limit_gb:
        counters["mem_alerts"] += 1
        log_fn(f"WATCHDOG MEM ALERT: total RSS {mem:.1f}GB over limit "
              f"{mem_limit_gb:.1f}GB (alert-only -- nothing is being killed)")

    def _since_growth(st):
        if st.last_growth_t is None:
            return "n/a"
        return f"{now - st.last_growth_t:.0f}s"

    progress = " ".join(
        f"{os.path.basename(st.path)}: lines={st.last_count} "
        f"since_growth={_since_growth(st)}"
        for st in states)
    log_fn(f"{time.strftime('%H:%M:%S', time.localtime(now))} "
          f"mem={mem:.2f}GB {progress}")


def run_watchdog(paths, block_eta, k, stall_after, interval, mem_limit_gb,
                 log_path, sleep_fn=time.sleep, now_fn=time.time,
                 mem_fn=total_rss_gb, count_fn=line_count, max_checks=None):
    """The real, infinite (unless `max_checks` is given) loop. Never called
    from tests -- `check_once` above carries all the logic tests exercise."""
    states = [WatchState(p) for p in paths]
    counters = {"checks": 0, "first_block_firings": 0,
               "stall_firings": 0, "mem_alerts": 0}
    start_t = now_fn()
    log_f = open(log_path, "a")
    stop = {"flag": False}

    def log_fn(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    def _request_stop(_signum, _frame):
        # Set-a-flag, don't act here: PEP 475 means time.sleep() just
        # resumes for the rest of its duration if the handler returns
        # normally, so the loop notices within one `interval` -- fine for
        # the 15s-scale intervals this watchdog runs at, and far simpler
        # than making sleep itself interruptible.
        stop["flag"] = True

    import signal
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    log_fn(f"watchdog start: watching {[st.path for st in states]} "
          f"block_eta={block_eta}s k={k} stall_after={stall_after}s "
          f"interval={interval}s mem_limit_gb={mem_limit_gb}")
    try:
        while not stop["flag"]:
            if max_checks is not None and counters["checks"] >= max_checks:
                break
            check_once(states, now_fn(), start_t, k, block_eta, stall_after,
                      mem_fn(), mem_limit_gb, counters, log_fn,
                      count_fn=count_fn)
            sleep_fn(interval)
    finally:
        log_fn(f"watchdog exit: checks={counters['checks']} "
              f"first_block_firings={counters['first_block_firings']} "
              f"stall_firings={counters['stall_firings']} "
              f"mem_alerts={counters['mem_alerts']}")
        log_f.close()
    return counters


def _expand(patterns):
    """Glob each pattern; a pattern that matches nothing is kept literally
    (the file may simply not exist yet -- a fresh run hasn't written its
    first block, which is exactly the case the first-block deadline covers)."""
    paths = []
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        paths.extend(matches if matches else [pat])
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--partial", nargs="+", required=True,
                    help="partial file path(s) or glob(s) to watch")
    ap.add_argument("--block-eta", type=float, required=True,
                    help="projected seconds per block")
    ap.add_argument("--k", type=float, default=3.0,
                    help="deadline multiplier (default 3.0)")
    ap.add_argument("--stall-after", type=float, default=None,
                    help="seconds of no growth (after the first block) "
                         "before a STALL fires; default k*block-eta")
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--mem-limit-gb", type=float, default=110.0,
                    help="alert-only -- nothing is ever killed")
    ap.add_argument("--log", default=os.path.join(RESULTS, "watchdog.log"),
                    help="append-only log path")
    args = ap.parse_args()
    stall_after = (args.stall_after if args.stall_after is not None
                  else args.k * args.block_eta)
    paths = _expand(args.partial)
    run_watchdog(paths, args.block_eta, args.k, stall_after, args.interval,
                args.mem_limit_gb, args.log)


if __name__ == "__main__":
    sys.exit(main() or 0)
