"""Task F0 straggler-tail probe (PHASE7_PLAN.md, gate ii): does shrinking the
checkpoint block actually shrink idle-core tail time?

WHY, PLAINLY. Session Lesson 1 said a 25-deal block let one worker run 40
minutes alone with seven others idle, because the LAST block dispatched is
whichever one drew the slowest items, and a bigger block means a bigger
"whichever one" to wait for. Shrinking the block doesn't remove the
variance in per-item cost, but it shrinks the SIZE of the one block that
ends up straggling, which should shrink the idle tail. This script measures
that directly rather than assuming it: same total items, same worker count,
two block sizes, and it prints the wall clock plus a completion timeline
plus how much of the wall clock happened after the LAST block was dispatched
(the tail-idle window, when every other worker has already drained).

SYNTHETIC MODE (default, and the only mode this task's tests rely on): a
fake per-item cost, seeded so it's reproducible but deliberately uneven
(uniform jitter on top of a baseline) -- exactly the shape of cost variance
that produces a straggler. No games are played, no JIT, no belief search;
this only exercises the checkpoint/dispatch pattern's timing, not any
correctness claim (that's `tests/test_blockdriver.py`'s job).

--real MODE wraps a genuine `run_exploit_eval.run_block("R0", ...)` call per
item (one real deal per item index, same DEAL_SEED_BASE+900_000 offset the
script's own `--probe` uses) instead of sleeping. Much slower, correctness-
irrelevant here too (R0 is just a convenient real, expensive workload) --
this mode is what the LEAD runs for the real headline number; this script's
job is only to make sure --real works, not to report its result.

Usage:
    python experiments/probe_blocks.py                          # synthetic
    python experiments/probe_blocks.py --items 200 --workers 8 --blocks 25,5
    python experiments/probe_blocks.py --real --workers 8 --blocks 25,5
"""
import argparse
import concurrent.futures as cf
import os
import sys
import time

import numpy as np

#: Synthetic per-item cost = BASE_S + uniform(0, JITTER_S), seeded by item
#: index so re-running with the same --items reproduces the same timeline.
BASE_S = 0.02
JITTER_S = 0.12


def _synthetic_cost(item):
    rng = np.random.default_rng(item)
    return BASE_S + float(rng.uniform(0, JITTER_S))


def _synthetic_worker(block_idx, item_indices):
    # `time.time()` calls happen INSIDE the worker process, so the first one
    # marks when this block actually started RUNNING -- after however long
    # it sat queued waiting for a free worker, which is the number that
    # matters for the tail-idle estimate (submission to the pool is near-
    # instantaneous no matter how many blocks queue behind it, so measuring
    # "submitted at" instead of "started at" would hide exactly the queuing
    # effect this probe exists to show).
    t_start_abs = time.time()
    for item in item_indices:
        time.sleep(_synthetic_cost(item))
    return block_idx, len(item_indices), t_start_abs, time.time()


def _real_worker(block_idx, item_indices):
    """One real R0 game per item index. Only reachable with --real."""
    sys.path.insert(0, os.path.dirname(__file__))
    import run_exploit_eval as X  # noqa: E402 (sets up its own src/ sys.path)
    t_start_abs = time.time()
    seeds = [X.DEAL_SEED_BASE + 900_000 + i for i in item_indices]
    X.run_block("R0", block_idx, seeds)
    return block_idx, len(item_indices), t_start_abs, time.time()


def run_probe(n_items, block_size, workers, worker_fn):
    """Dispatch `n_items` split into `block_size`-sized blocks across
    `workers` processes; return (wall_s, timeline, tail_idle_s).

    `timeline` is a list of (block_idx, start_s, done_s, n_in_block,
    worker_reported_s) sorted by completion time, offsets from this
    function's own start. `tail_idle_s` is the wall clock elapsed AFTER the
    LAST-STARTED block began running (not merely submitted -- with more
    blocks than workers, later blocks queue and only start once a worker
    frees up) -- the window in which every worker but the straggler
    finishing that block has nothing left to do. Same-machine epoch time is
    shared between the main process and the spawned workers, so comparing
    timestamps across processes is safe here.
    """
    n_blocks = -(-n_items // block_size)
    blocks = [(b, list(range(b * block_size, min(n_items, (b + 1) * block_size))))
             for b in range(n_blocks)]
    t0_epoch = time.time()
    timeline = []
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(worker_fn, b, items): (b, len(items))
               for b, items in blocks}
        for fut in cf.as_completed(futs):
            b, n = futs[fut]
            _b2, _n2, start_abs, end_abs = fut.result()
            timeline.append((b, start_abs - t0_epoch, end_abs - t0_epoch,
                             n, end_abs - start_abs))
    wall = time.time() - t0_epoch
    timeline.sort(key=lambda row: row[2])
    last_started_s = max(row[1] for row in timeline)
    tail_idle = wall - last_started_s
    return wall, timeline, tail_idle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--blocks", default="25,5",
                    help="comma-separated block sizes to compare")
    ap.add_argument("--real", action="store_true",
                    help="wrap genuine R0 games instead of the synthetic "
                         "sleep-jittered workload (much slower; LEAD-only "
                         "for the real headline measurement)")
    args = ap.parse_args()
    worker_fn = _real_worker if args.real else _synthetic_worker
    mode = "REAL R0 smoke" if args.real else "synthetic"

    for block_size in (int(x) for x in args.blocks.split(",") if x):
        wall, timeline, tail_idle = run_probe(
            args.items, block_size, args.workers, worker_fn)
        print(f"\n=== block={block_size}  workers={args.workers}  "
              f"items={args.items}  ({mode}) ===")
        print(f"wall time: {wall:.2f}s")
        print(f"tail idle (wall clock after the LAST block STARTED running, "
              f"i.e. every worker but the straggler finishing that block "
              f"sat idle for this long): {tail_idle:.2f}s "
              f"({100 * tail_idle / wall:.1f}% of wall time)")
        print("completion timeline (block, start_s, done_s, n_items, "
              "worker_reported_s):")
        for b, start_s, done_s, n, worker_s in timeline:
            print(f"  block {b:>3} | start={start_s:7.2f}s done={done_s:7.2f}s "
                  f"n={n:3d} worker_s={worker_s:7.2f}s")


if __name__ == "__main__":
    main()
