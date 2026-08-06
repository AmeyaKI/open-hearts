"""Parallel driver for the ablation: same experiment, many cores.

This changes ORCHESTRATION ONLY. All measured code paths (rotated_match,
SearchPlayer, bootstrap_ci, write_outputs) are imported unchanged from
run_ablation.py. Games are chunked by deal and run in worker processes.

Determinism: every game's rng is seeded from (config_id, deal_seed, rotation)
— run_ablation._game_seed — and each worker builds its factory over its own
chunk's seed list, so per-game seeds are identical to the serial run and the
assembled per-deal arrays are bitwise identical regardless of chunking or
worker count (verified by test mode: `--verify`).

Safety: worker count is capped (default 12 of the machine's cores) and the
parent polls total RSS of the process tree every few seconds, aborting the
run if it exceeds MEM_LIMIT_GB (default 100).
"""
import argparse
import concurrent.futures as cf
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import run_ablation as ra  # noqa: E402

WORKERS = 12
CHUNK = 25
MEM_LIMIT_GB = 100.0


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield i, seq[i:i + size]


def worker(name, kind, config_id, level_name, n_samples, start, chunk_seeds):
    """Run one chunk of deals for one configuration in a worker process."""
    from openhearts.belief.table import Level
    from openhearts.eval.harness import rotated_match
    from openhearts.players.heuristic import HeuristicPlayer

    tally = {}
    if kind == "search":
        factory = ra.search_factory(config_id, Level[level_name], n_samples,
                                    chunk_seeds, tally)
    elif kind == "heuristic":
        factory = lambda: HeuristicPlayer()  # noqa: E731
    elif kind == "random":
        factory = ra.random_factory(config_id, chunk_seeds)
    else:
        raise ValueError(kind)
    per_deal = rotated_match(chunk_seeds, factory, lambda: HeuristicPlayer())
    bots = tally.get("bots", [])
    return (name, start, per_deal,
            sum(b.fallbacks for b in bots),
            sum(b.failed_samples for b in bots))


def total_rss_gb():
    """Total resident memory (GB) of this process and its children."""
    out = subprocess.run(["ps", "-o", "rss=", "-g", str(os.getpgrp())],
                         capture_output=True, text=True).stdout
    return sum(int(x) for x in out.split()) / (1024 ** 2)


CONFIGS = (
    [(f"search-{name}-n{ra.MAIN_N_SAMPLES}", "search", cid, level.name,
      ra.MAIN_N_SAMPLES)
     for cid, (name, level) in enumerate(ra.MAIN_LEVELS)]
    + [("heuristic", "heuristic", 80, None, None),
       ("random-legal", "random", 90, None, None)]
    + [(f"sweep-FULL-n{n}", "search", cid, "FULL", n)
       for cid, n in enumerate(ra.SWEEP_N, start=10) if n != ra.MAIN_N_SAMPLES]
)


def run_all(deal_seeds, workers):
    results = {name: np.zeros(len(deal_seeds)) for name, *_ in CONFIGS}
    tallies = {name: [0, 0] for name, *_ in CONFIGS}
    jobs = []
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        for name, kind, cid, level_name, n in CONFIGS:
            for start, chunk in _chunks(deal_seeds, CHUNK):
                jobs.append(pool.submit(worker, name, kind, cid, level_name,
                                        n, start, chunk))
        done = 0
        t0 = time.time()
        for fut in cf.as_completed(jobs):
            name, start, per_deal, fb, fs = fut.result()
            results[name][start:start + len(per_deal)] = per_deal
            tallies[name][0] += fb
            tallies[name][1] += fs
            done += 1
            mem = total_rss_gb()
            print(f"[{done}/{len(jobs)}] {name} deals {start}-"
                  f"{start + len(per_deal) - 1} done | mem={mem:.1f}GB | "
                  f"{time.time() - t0:.0f}s elapsed", flush=True)
            ra._append_partial(f"{name}@{start}", per_deal)
            if mem > MEM_LIMIT_GB:
                print(f"ABORT: memory {mem:.1f}GB exceeds "
                      f"{MEM_LIMIT_GB}GB limit", flush=True)
                for j in jobs:
                    j.cancel()
                raise MemoryError("memory limit exceeded")
    return results, tallies


def as_rows(results, tallies, deal_seeds, elapsed):
    from openhearts.eval.stats import bootstrap_ci
    main_rows, sweep_rows = [], []
    for name, kind, cid, level_name, n in CONFIGS:
        per_deal = results[name]
        mean, lo, hi = bootstrap_ci(per_deal)
        fb, fs = tallies[name]
        row = {"name": name, "per_deal": per_deal, "mean": mean, "lo": lo,
               "hi": hi, "seconds": elapsed,
               "fallbacks": fb if kind == "search" else None,
               "failed_samples": fs if kind == "search" else None}
        if name.startswith("sweep-"):
            sweep_rows.append({**row, "n_samples": n, "reused": False})
        else:
            main_rows.append(row)
    # insert the reused n=100 point in sweep order
    full_row = next(r for r in main_rows
                    if r["name"] == f"search-FULL-n{ra.MAIN_N_SAMPLES}")
    sweep_rows.append({**full_row, "n_samples": ra.MAIN_N_SAMPLES,
                       "reused": True})
    sweep_rows.sort(key=lambda r: r["n_samples"])
    return main_rows, sweep_rows


def verify():
    """Chunked parallel results must be bitwise identical to serial."""
    seeds = ra.DEAL_SEEDS[:8]
    serial = {}
    for name, kind, cid, level_name, n in CONFIGS[:3]:
        from openhearts.belief.table import Level
        tally = {}
        factory = ra.search_factory(cid, Level[level_name], n, seeds, tally)
        from openhearts.eval.harness import rotated_match
        from openhearts.players.heuristic import HeuristicPlayer
        serial[name] = rotated_match(seeds, factory, lambda: HeuristicPlayer())
    global CHUNK
    CHUNK = 4  # force 2 chunks per config
    results, _ = run_all(seeds, workers=6)
    for name in serial:
        assert np.array_equal(serial[name], results[name]), f"MISMATCH {name}"
        print(f"[verify] {name}: parallel == serial (bitwise)", flush=True)
    print("[verify] OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="check parallel==serial on 8 deals, then exit")
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()
    if args.verify:
        verify()
        return
    os.makedirs(ra.RESULTS, exist_ok=True)
    probe = ra.timing_probe()
    with open(ra.PARTIAL, "w") as f:
        f.write(f"# parallel run start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    t0 = time.time()
    results, tallies = run_all(ra.DEAL_SEEDS, args.workers)
    elapsed = time.time() - t0
    main_rows, sweep_rows = as_rows(results, tallies, ra.DEAL_SEEDS, elapsed)
    ra.write_outputs(main_rows, sweep_rows, probe)
    print(f"[done] wall time {elapsed/60:.1f} min with {args.workers} workers",
          flush=True)


if __name__ == "__main__":
    main()
