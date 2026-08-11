"""Task 6: world-survival measurement for choice filtering.

For the first 300 records of results/heuristic_games.txt, at every
completed-trick boundary (entering tricks 2..13 -- entering trick 1 has no
evidence yet, so it is skipped) and for EACH of the 4 observer seats, this
runs `WeightedPosterior.from_view(view, Level.FULL, HeuristicPlayer(),
epsilon=0.0, n_worlds=100, rng, max_draws=50000)` and records how many
candidate worlds had to be drawn before 100 survived (or how many died
trying, up to the cap).

This measures the cost of strict (epsilon=0) rejection-sampling filtering
against a deterministic heuristic policy: if the policy's decisions are
highly constraining, few candidate worlds will replay legally/plausibly, and
`draws_used` will run high (or hit `max_draws` and give up early -- an
"exhausted" call) or every draw dies (a "raised" call, total weight zero).
Results are bucketed by trick number and written to results/survival.txt,
with a --smoke flag for a small subset for a quick sanity/timing check.

Deterministic rng: each (record index, trick, observer) triple gets its own
`np.random.default_rng` seeded from a fixed hash of those three integers, so
the measurement is exactly reproducible regardless of parallelization/chunking.

Parallelized across records with the ProcessPoolExecutor chunk pattern from
run_ablation_parallel.py.
"""
import argparse
import os
import subprocess
import sys
import time
import concurrent.futures as cf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from openhearts.belief.table import Level  # noqa: E402
from openhearts.belief.weighted import WeightedPosterior  # noqa: E402
from openhearts.engine import cards  # noqa: E402
from openhearts.engine.state import GameState  # noqa: E402
from openhearts.eval.records import read_records  # noqa: E402
from openhearts.players.heuristic import HeuristicPlayer  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
RECORDS_PATH = os.path.join(RESULTS, "heuristic_games.txt")
OUT_PATH = os.path.join(RESULTS, "survival.txt")
SMOKE_OUT_PATH = os.path.join(RESULTS, "survival_smoke.txt")

NUM_RECORDS = 300
NUM_TRICKS = 13
EPSILON = 0.0
N_WORLDS = 100
MAX_DRAWS = 50000

WORKERS = 12
CHUNK = 25

SMOKE_RECORDS = 8
SMOKE_WORKERS = 2


def _seed_for(rec_idx, trick, observer):
    """Deterministic rng seed from (record index, trick, observer)."""
    return int(
        np.random.default_rng(
            [rec_idx, trick, observer]
        ).integers(0, 2**63 - 1)
    )


def _call_for_boundary(state, rec_idx, trick, policy):
    """Run the WeightedPosterior call for all 4 observers at this boundary.

    Returns a list of 4 dicts, one per observer, each with
    draws_used / n_worlds_used / n_effective / exhausted / raised.
    """
    out = []
    for seat in range(4):
        rng = np.random.default_rng(_seed_for(rec_idx, trick, seat))
        view = state.view_for(seat)
        try:
            post = WeightedPosterior.from_view(
                view, Level.FULL, policy, epsilon=EPSILON,
                n_worlds=N_WORLDS, rng=rng, max_draws=MAX_DRAWS,
            )
            exhausted = (post.n_worlds_used < N_WORLDS
                        and post.draws_used >= MAX_DRAWS)
            out.append({
                "draws_used": post.draws_used,
                "n_worlds_used": post.n_worlds_used,
                "n_effective": post.n_effective,
                "exhausted": exhausted,
                "raised": False,
            })
        except AssertionError:
            out.append({
                "draws_used": MAX_DRAWS,
                "n_worlds_used": 0,
                "n_effective": 0.0,
                "exhausted": False,
                "raised": True,
            })
    return out


def process_record(rec):
    """Replay one record; return per-trick lists of per-observer stat dicts.

    `results[k]` (k in 0..11, trick = k+2) is a list of stat dicts collected
    across the 4 observers at the boundary entering trick k+2.
    """
    results = {trick: [] for trick in range(2, NUM_TRICKS + 1)}
    policy = HeuristicPlayer()

    state = GameState(hands=list(rec.hands))
    state.to_play = next(
        s for s in range(4) if rec.hands[s] & cards.bit(cards.TWO_CLUBS)
    )
    assert state.to_play == rec.plays[0][0], "record leader is not 2c holder"

    for seat, card in rec.plays:
        assert seat == state.to_play, (
            f"replay desync: recorded seat {seat} != {state.to_play}"
        )
        state.play(card)
        if not state.current_trick and state.trick_number < NUM_TRICKS:
            trick = state.trick_number + 1  # entering this trick
            if trick >= 2:
                results[trick].extend(
                    _call_for_boundary(state, rec.seed, trick, policy)
                )
    assert state.is_over(), "record did not replay to a finished game"
    return results


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield i, seq[i:i + size]


def worker(chunk_records):
    """Process a chunk of records; return per-trick accumulated stat lists."""
    acc = {trick: [] for trick in range(2, NUM_TRICKS + 1)}
    for rec in chunk_records:
        per_rec = process_record(rec)
        for trick in acc:
            acc[trick].extend(per_rec[trick])
    return acc


def total_rss_gb():
    out = subprocess.run(["ps", "-o", "rss=", "-g", str(os.getpgrp())],
                         capture_output=True, text=True).stdout
    return sum(int(x) for x in out.split()) / (1024 ** 2)


def run_all(records, workers, chunk_size, progress_every=25):
    acc = {trick: [] for trick in range(2, NUM_TRICKS + 1)}
    jobs = []
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        for start, chunk in _chunks(records, chunk_size):
            jobs.append(pool.submit(worker, chunk))
        done = 0
        t0 = time.time()
        for fut in cf.as_completed(jobs):
            chunk_acc = fut.result()
            for trick in acc:
                acc[trick].extend(chunk_acc[trick])
            done += 1
            n_records_done = min(done * chunk_size, len(records))
            if n_records_done % progress_every < chunk_size or done == len(jobs):
                mem = total_rss_gb()
                print(f"[{done}/{len(jobs)}] chunks done "
                      f"(~{n_records_done}/{len(records)} records) | "
                      f"mem={mem:.1f}GB | {time.time() - t0:.0f}s elapsed",
                      flush=True)
    return acc, time.time() - t0


def summarize(acc):
    """Per-trick summary rows from the accumulated per-observer stat dicts."""
    rows = []
    for trick in range(2, NUM_TRICKS + 1):
        stats = acc[trick]
        n = len(stats)
        assert n > 0, f"no calls recorded for trick {trick}"
        draws = np.array([s["draws_used"] for s in stats], dtype=float)
        nworlds = np.array([s["n_worlds_used"] for s in stats], dtype=float)
        neff = np.array([s["n_effective"] for s in stats], dtype=float)
        exhausted_frac = np.mean([s["exhausted"] for s in stats])
        raised_frac = np.mean([s["raised"] for s in stats])
        rows.append({
            "trick": trick,
            "n_calls": n,
            "mean_draws": float(draws.mean()),
            "median_draws": float(np.median(draws)),
            "mean_n_worlds": float(nworlds.mean()),
            "median_n_worlds": float(np.median(nworlds)),
            "mean_n_effective": float(neff.mean()),
            "frac_exhausted": float(exhausted_frac),
            "frac_raised": float(raised_frac),
        })
    return rows


def write_table(path, rows, num_records, elapsed, workers, smoke):
    with open(path, "w") as f:
        f.write(f"# survival measurement (Task 6)\n")
        f.write(f"# records={num_records} (first {num_records} of "
                f"{RECORDS_PATH}), master_seed=2026 (from heuristic_games.txt)\n")
        f.write(f"# level=FULL policy=HeuristicPlayer epsilon={EPSILON} "
                f"n_worlds={N_WORLDS} max_draws={MAX_DRAWS}\n")
        f.write(f"# tricks: entering trick k boundary, k=2..13 "
                f"(entering trick 1 has no evidence, skipped)\n")
        f.write(f"# observers: all 4 seats per boundary\n")
        f.write(f"# wall_time_s={elapsed:.1f} workers={workers} smoke={smoke}\n")
        f.write("trick n_calls mean_draws median_draws mean_n_worlds "
                "median_n_worlds mean_n_effective frac_exhausted frac_raised\n")
        for r in rows:
            f.write(f"{r['trick']} {r['n_calls']} "
                    f"{r['mean_draws']:.3f} {r['median_draws']:.3f} "
                    f"{r['mean_n_worlds']:.3f} {r['median_n_worlds']:.3f} "
                    f"{r['mean_n_effective']:.3f} "
                    f"{r['frac_exhausted']:.6f} {r['frac_raised']:.6f}\n")


def print_table(rows):
    print(f"\n{'trick':>5} {'n':>5} {'mean_draws':>11} {'med_draws':>10} "
          f"{'mean_nw':>8} {'med_nw':>7} {'mean_neff':>10} "
          f"{'exhausted':>10} {'raised':>8}")
    for r in rows:
        print(f"{r['trick']:>5} {r['n_calls']:>5} {r['mean_draws']:>11.1f} "
              f"{r['median_draws']:>10.1f} {r['mean_n_worlds']:>8.1f} "
              f"{r['median_n_worlds']:>7.1f} {r['mean_n_effective']:>10.2f} "
              f"{r['frac_exhausted']:>10.4f} {r['frac_raised']:>8.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="first 8 records, 2 workers, separate output file")
    args = ap.parse_args()

    all_records = read_records(RECORDS_PATH)

    if args.smoke:
        records = all_records[:SMOKE_RECORDS]
        workers = SMOKE_WORKERS
        chunk_size = SMOKE_RECORDS  # single chunk is fine at this size
        out_path = SMOKE_OUT_PATH
    else:
        records = all_records[:NUM_RECORDS]
        workers = WORKERS
        chunk_size = CHUNK
        out_path = OUT_PATH

    os.makedirs(RESULTS, exist_ok=True)
    print(f"processing {len(records)} records with {workers} workers "
          f"(chunk={chunk_size})", flush=True)

    acc, elapsed = run_all(records, workers, chunk_size)
    rows = summarize(acc)
    write_table(out_path, rows, len(records), elapsed, workers, args.smoke)
    print_table(rows)
    print(f"\nwrote {out_path} in {elapsed:.1f}s "
          f"({len(records)} records, {workers} workers)")


if __name__ == "__main__":
    main()
