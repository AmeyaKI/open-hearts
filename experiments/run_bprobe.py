"""Version B diagnostic: ablation vs randomized (unpredictable) opponents.

Purpose (PHASE23_PLAN Task 2): isolate whether opponent *predictability* --
not search blindness -- explains Phase 1's flat ablation (UNIFORM-noleak
3.37 / VOIDS 3.14 / FULL 3.27, overlapping CIs). This script reruns the
identical three Phase-1 search rows (SearchPlayer, n_samples=100, same 500
deal seeds) but replaces the three heuristic opponents with
`RandomizedHeuristic(epsilon=0.1)`, plus a randomized-mirror row (all four
seats RandomizedHeuristic) that should land near the 6.5 symmetric
break-even, confirming the harness/seeding still behaves normally.

If belief levels separate here but not in Phase 1's ablation.txt,
predictability was a factor; if flat both here too, search blindness (fixed
by Task 3's HonestSearchPlayer) is confirmed as the bottleneck.

Determinism / opponent-factory contract (see eval/harness.rotated_match's
documented iteration order): `bot_factory()` is called once per game,
`opp_factory()` exactly 3 times per game right after it, once for each
non-bot seat in ascending seat order (0..3, skipping the bot's rotation
seat). Each RandomizedHeuristic opponent therefore gets its own rng seeded
from (config_id, deal_seed, rotation, seat) -- reruns reproduce every
opponent's exact card choices, and no two seats/games ever share a stream.
The randomized-mirror row's bot (rotation seat) is seeded the same way, with
seat=rotation, so its formula is a strict superset of the opponent formula.

Reuses run_ablation's `_game_seed`-style per-game seeding pattern and
run_ablation_parallel's chunked/12-worker/partial-file/memory-watchdog
orchestration; no library code is added, only new player-factory glue.

Usage:
    python experiments/run_bprobe.py --smoke   # 4 deals per row, then exit
    python experiments/run_bprobe.py           # full 500-deal run (LEAD ONLY)
"""
import argparse
import itertools
import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_ablation as ra  # noqa: E402

from openhearts.belief.table import Level  # noqa: E402
from openhearts.eval.harness import rotated_match  # noqa: E402
from openhearts.eval.stats import bootstrap_ci  # noqa: E402
from openhearts.players.randomized import RandomizedHeuristic  # noqa: E402
from openhearts.search.decision import SearchPlayer  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
PARTIAL = os.path.join(RESULTS, "bprobe_partial.txt")

EPSILON = 0.1
MAIN_N_SAMPLES = 100
N_DEALS = 500
DEAL_SEEDS = ra.DEAL_SEEDS  # identical 500 seeds (100000..100499) as Phase 1
WORKERS = 12
CHUNK = 25
MEM_LIMIT_GB = 100.0

# Fresh config_ids: not previously used by run_ablation.py (0,1,2 = main
# levels; 10.. = sweep; 80 = heuristic; 90 = random-legal; 999 = probe).
MAIN_LEVELS = [("UNIFORM-noleak", Level.UNIFORM, False, 200),
               ("VOIDS", Level.VOIDS, True, 201),
               ("FULL", Level.FULL, True, 202)]
MIRROR_CONFIG_ID = 203


def _append_partial(name, values):
    os.makedirs(RESULTS, exist_ok=True)
    with open(PARTIAL, "a") as f:
        f.write(name + " " + " ".join(f"{v:.6f}" for v in values) + "\n")
        f.flush()


def randomized_opp_factory(config_id, deal_seeds):
    """3 calls/game (harness contract): one RandomizedHeuristic per non-bot
    seat, in ascending seat order, seeded from (config_id, deal seed,
    rotation, seat)."""
    state = {"game_idx": 0, "call_in_game": 0}
    def factory():
        i = state["game_idx"]
        seed, rotation = deal_seeds[i // 4], i % 4
        call = state["call_in_game"]
        seats = [s for s in range(4) if s != rotation]
        seat = seats[call]
        state["call_in_game"] += 1
        if state["call_in_game"] == 3:
            state["call_in_game"] = 0
            state["game_idx"] += 1
        rng = np.random.default_rng([config_id, seed, rotation, seat])
        return RandomizedHeuristic(rng, epsilon=EPSILON)
    return factory


def randomized_bot_factory(config_id, deal_seeds):
    """1 call/game (harness contract): the bot occupies the rotation seat,
    seeded with the same (config_id, deal seed, rotation, seat) formula as
    the opponent factory (seat=rotation) -- so mirror-row seeds are a strict
    superset of the opponent seeds used elsewhere."""
    counter = itertools.count()
    def factory():
        i = next(counter)
        seed, rotation = deal_seeds[i // 4], i % 4
        rng = np.random.default_rng([config_id, seed, rotation, rotation])
        return RandomizedHeuristic(rng, epsilon=EPSILON)
    return factory


def search_vs_randomized_factory(config_id, level, sampler_respects_voids,
                                 n_samples, deal_seeds, tally=None):
    """SearchPlayer bot (1 call/game), seeded like run_ablation.search_factory."""
    counter = itertools.count()
    def factory():
        i = next(counter)
        seed, rotation = deal_seeds[i // 4], i % 4
        bot = SearchPlayer(level, n_samples, ra._game_seed(config_id, seed, rotation),
                           sampler_respects_voids=sampler_respects_voids)
        if tally is not None:
            tally.setdefault("bots", []).append(bot)
        return bot
    return factory


CONFIGS = (
    [(f"search-{name}-n{MAIN_N_SAMPLES}", "search", cid, level.name, respects,
      MAIN_N_SAMPLES)
     for name, level, respects, cid in MAIN_LEVELS]
    + [("randomized-mirror", "mirror", MIRROR_CONFIG_ID, None, None, None)]
)


def _progress(label):
    def on_deal(done, total):
        if done % 25 == 0 or done == total:
            print(f"  [{label}] {done}/{total} deals", flush=True)
    return on_deal


def run_config_serial(name, kind, config_id, level_name, respects, n_samples,
                      deal_seeds):
    """Serial fallback (used by --smoke); mirrors run_ablation.run_config."""
    tally = {} if kind == "search" else None
    if kind == "search":
        bot_factory = search_vs_randomized_factory(
            config_id, Level[level_name], respects, n_samples, deal_seeds, tally)
        opp_factory = randomized_opp_factory(config_id, deal_seeds)
    elif kind == "mirror":
        bot_factory = randomized_bot_factory(config_id, deal_seeds)
        opp_factory = randomized_opp_factory(config_id, deal_seeds)
    else:
        raise ValueError(kind)
    print(f"[config] {name}: {len(deal_seeds)} deals x 4 rotations", flush=True)
    t0 = time.time()
    per_deal = rotated_match(deal_seeds, bot_factory, opp_factory,
                             on_deal=_progress(name))
    elapsed = time.time() - t0
    mean, lo, hi = bootstrap_ci(per_deal)
    print(f"[config] {name}: mean={mean:.3f} CI=({lo:.3f},{hi:.3f}) "
          f"in {elapsed:.1f}s", flush=True)
    fallbacks = failed = None
    if tally is not None:
        bots = tally.get("bots", [])
        fallbacks = sum(b.fallbacks for b in bots)
        failed = sum(b.failed_samples for b in bots)
        print(f"[config] {name}: heuristic fallbacks={fallbacks}, "
              f"failed sample draws={failed}", flush=True)
    _append_partial(name, per_deal)
    return {"name": name, "per_deal": per_deal, "mean": mean, "lo": lo,
            "hi": hi, "seconds": elapsed, "fallbacks": fallbacks,
            "failed_samples": failed}


# ---------------------------------------------------------------- parallel
def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield i, seq[i:i + size]


def worker(name, kind, config_id, level_name, respects, n_samples, start,
          chunk_seeds):
    """Run one chunk of deals for one configuration in a worker process."""
    tally = {} if kind == "search" else None
    if kind == "search":
        bot_factory = search_vs_randomized_factory(
            config_id, Level[level_name], respects, n_samples, chunk_seeds, tally)
        opp_factory = randomized_opp_factory(config_id, chunk_seeds)
    elif kind == "mirror":
        bot_factory = randomized_bot_factory(config_id, chunk_seeds)
        opp_factory = randomized_opp_factory(config_id, chunk_seeds)
    else:
        raise ValueError(kind)
    per_deal = rotated_match(chunk_seeds, bot_factory, opp_factory)
    fb = fs = 0
    if tally is not None:
        bots = tally.get("bots", [])
        fb = sum(b.fallbacks for b in bots)
        fs = sum(b.failed_samples for b in bots)
    return (name, start, per_deal, fb, fs)


def total_rss_gb():
    out = subprocess.run(["ps", "-o", "rss=", "-g", str(os.getpgrp())],
                         capture_output=True, text=True).stdout
    return sum(int(x) for x in out.split()) / (1024 ** 2)


def run_all(deal_seeds, workers):
    import concurrent.futures as cf
    results = {name: np.zeros(len(deal_seeds)) for name, *_ in CONFIGS}
    tallies = {name: [0, 0] for name, *_ in CONFIGS}
    jobs = []
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        for name, kind, cid, level_name, respects, n in CONFIGS:
            for start, chunk in _chunks(deal_seeds, CHUNK):
                jobs.append(pool.submit(worker, name, kind, cid, level_name,
                                        respects, n, start, chunk))
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
            _append_partial(f"{name}@{start}", per_deal)
            if mem > MEM_LIMIT_GB:
                print(f"ABORT: memory {mem:.1f}GB exceeds "
                      f"{MEM_LIMIT_GB}GB limit", flush=True)
                for j in jobs:
                    j.cancel()
                raise MemoryError("memory limit exceeded")
    return results, tallies


def as_rows(results, tallies, elapsed):
    rows = []
    for name, kind, cid, level_name, respects, n in CONFIGS:
        per_deal = results[name]
        mean, lo, hi = bootstrap_ci(per_deal)
        fb, fs = tallies[name]
        rows.append({"name": name, "per_deal": per_deal, "mean": mean,
                     "lo": lo, "hi": hi, "seconds": elapsed,
                     "fallbacks": fb if kind == "search" else None,
                     "failed_samples": fs if kind == "search" else None})
    return rows


# ---------------------------------------------------------------- outputs
def write_outputs(rows, n_deals, deal_seeds, out_name="bprobe.txt"):
    os.makedirs(RESULTS, exist_ok=True)
    lines = [
        "open-hearts Version B diagnostic (Task 2, PHASE23_PLAN)",
        "Same three Phase-1 SearchPlayer belief-level rows (n_samples=100) "
        "but opponents are RandomizedHeuristic instead of plain HeuristicPlayer.",
        f"epsilon={EPSILON} for every RandomizedHeuristic opponent/mirror seat.",
        "Lower points per hand is better. 26 points are dealt out per hand,",
        "so 6.5 is the symmetric break-even.",
        "",
        f"deals: {n_deals} (seeds {deal_seeds[0]}..{deal_seeds[-1]}), "
        f"each played 4x with the bot rotated through every seat",
        "identical deals across rows; CIs are 10,000 bootstrap resamples "
        "over DEALS (not games).",
        "randomized-mirror: all four seats are RandomizedHeuristic(epsilon="
        f"{EPSILON}) -- expected mean near 6.5, sanity-checking the harness "
        "and seeding rather than measuring the search.",
        "",
        f"{'config':<28}{'mean':>8}{'lo95':>9}{'hi95':>9}{'secs':>10}"
        f"{'fallbk':>9}{'failsmp':>9}",
    ]
    for r in rows:
        fb = "-" if r.get("fallbacks") is None else str(r["fallbacks"])
        fs = "-" if r.get("failed_samples") is None else str(r["failed_samples"])
        lines.append(f"{r['name']:<28}{r['mean']:>8.3f}{r['lo']:>9.3f}"
                     f"{r['hi']:>9.3f}{r['seconds']:>10.1f}{fb:>9}{fs:>9}")
    with open(os.path.join(RESULTS, out_name), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="4 deals per row, serial, then exit (mechanics check)")
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    if args.smoke:
        smoke_seeds = DEAL_SEEDS[:4]
        with open(PARTIAL, "w") as f:
            f.write(f"# smoke run start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        rows = []
        for name, kind, cid, level_name, respects, n in CONFIGS:
            rows.append(run_config_serial(name, kind, cid, level_name,
                                          respects, n, smoke_seeds))
        write_outputs(rows, len(smoke_seeds), smoke_seeds,
                     out_name="bprobe_smoke.txt")
        print("[smoke] done (mechanics check only -- means are not "
              "meaningful at 4 deals)", flush=True)
        return

    with open(PARTIAL, "w") as f:
        f.write(f"# full run start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    t0 = time.time()
    results, tallies = run_all(DEAL_SEEDS, args.workers)
    elapsed = time.time() - t0
    rows = as_rows(results, tallies, elapsed)
    write_outputs(rows, N_DEALS, DEAL_SEEDS)
    print(f"[done] wall time {elapsed/60:.1f} min with {args.workers} workers",
          flush=True)


if __name__ == "__main__":
    main()
