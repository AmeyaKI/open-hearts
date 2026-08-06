"""Sampler failure-rate measurement.

Reuses the 2,000 recorded games from results/heuristic_games.txt (written by
run_guessing.py); this script uses only the FIRST 500 of those records to
keep runtime reasonable. Each game is replayed with the engine (to_play is
set to the 2-of-clubs holder before replay, exactly as
eval/guessing.evaluate_record does, since GameRecord does not store to_play).

At every point a trick is about to be played -- the fresh deal (entering
trick 1) and after every completed trick (entering trick 2..13) -- a FULL
BeliefTable is built for the seat about to play, and sample_arrangement is
called 20 times. We record, bucketed by the upcoming trick number:
  - attempts-per-success (mean number of restarts used by successful calls)
  - failure rate (fraction of the 20 calls that returned None)

Writes results/sampler_stats.txt (trick, mean attempts, failure rate) and
prints a summary table.
"""
import os
import time

import numpy as np

from openhearts.belief.table import BeliefTable, Level
from openhearts.engine import cards
from openhearts.engine.state import GameState
from openhearts.eval.records import read_records
from openhearts.sampler.sampler import sample_arrangement

NUM_TRICKS = 13
NUM_RECORDS_USED = 500
SAMPLES_PER_POINT = 20
SAMPLER_SEED = 4242

RESULTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
RECORDS_PATH = os.path.join(RESULTS, "heuristic_games.txt")
STATS_PATH = os.path.join(RESULTS, "sampler_stats.txt")

FAILURE_RATE_WARN = 0.01
MEAN_ATTEMPTS_WARN = 5.0


def _sample_stats(view, rng, attempts_by_trick, failures_by_trick, calls_by_trick):
    trick = view.trick_number + 1  # trick about to be played, 1-indexed
    table = BeliefTable.from_view(view, Level.FULL)
    for _ in range(SAMPLES_PER_POINT):
        result = sample_arrangement(table, rng)
        calls_by_trick[trick] += 1
        if result is None:
            failures_by_trick[trick] += 1
        else:
            _, attempts = result
            attempts_by_trick[trick].append(attempts)


def measure(records, rng):
    attempts_by_trick = {k: [] for k in range(1, NUM_TRICKS + 1)}
    failures_by_trick = {k: 0 for k in range(1, NUM_TRICKS + 1)}
    calls_by_trick = {k: 0 for k in range(1, NUM_TRICKS + 1)}

    for rec in records:
        state = GameState(hands=list(rec.hands))
        state.to_play = next(
            s for s in range(4) if rec.hands[s] & cards.bit(cards.TWO_CLUBS)
        )
        assert state.to_play == rec.plays[0][0], "record leader is not 2c holder"

        # entering trick 1: the fresh deal
        _sample_stats(state.view_for(state.to_play), rng,
                      attempts_by_trick, failures_by_trick, calls_by_trick)

        for seat, card in rec.plays:
            assert seat == state.to_play, (
                f"replay desync: recorded seat {seat} != {state.to_play}"
            )
            state.play(card)
            if not state.current_trick and state.trick_number < NUM_TRICKS:
                _sample_stats(state.view_for(state.to_play), rng,
                              attempts_by_trick, failures_by_trick,
                              calls_by_trick)
        assert state.is_over(), "record did not replay to a finished game"

    return attempts_by_trick, failures_by_trick, calls_by_trick


def write_table(path, attempts_by_trick, failures_by_trick, calls_by_trick,
                num_records, warnings):
    with open(path, "w") as f:
        f.write(f"# records_used={num_records} (first {NUM_RECORDS_USED} of "
                f"{RECORDS_PATH})\n")
        f.write(f"# samples_per_point={SAMPLES_PER_POINT} "
                f"sampler_seed={SAMPLER_SEED}\n")
        f.write("# trick k = FULL table for the seat about to play trick k "
                "(k=1 is the fresh deal)\n")
        f.write("trick mean_attempts failure_rate\n")
        for k in range(1, NUM_TRICKS + 1):
            succ = attempts_by_trick[k]
            mean_att = float(np.mean(succ)) if succ else float("nan")
            fail_rate = failures_by_trick[k] / calls_by_trick[k]
            f.write(f"{k} {mean_att:.6f} {fail_rate:.6f}\n")
        if warnings:
            f.write("\n# WARNING\n")
            for w in warnings:
                f.write(f"# {w}\n")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    records = read_records(RECORDS_PATH)[:NUM_RECORDS_USED]
    rng = np.random.default_rng(SAMPLER_SEED)

    t0 = time.time()
    attempts_by_trick, failures_by_trick, calls_by_trick = measure(records, rng)
    print(f"measured {len(records)} records in {time.time() - t0:.1f}s",
          flush=True)

    warnings = []
    print("\ntrick  mean_attempts  failure_rate")
    for k in range(1, NUM_TRICKS + 1):
        succ = attempts_by_trick[k]
        mean_att = float(np.mean(succ)) if succ else float("nan")
        fail_rate = failures_by_trick[k] / calls_by_trick[k]
        print(f"{k:>5}  {mean_att:>13.4f}  {fail_rate:>12.4f}")
        if fail_rate > FAILURE_RATE_WARN:
            warnings.append(
                f"trick {k}: failure_rate={fail_rate:.4f} exceeds "
                f"~{FAILURE_RATE_WARN}"
            )
        if mean_att == mean_att and mean_att > MEAN_ATTEMPTS_WARN:
            warnings.append(
                f"trick {k}: mean_attempts={mean_att:.4f} exceeds "
                f"~{MEAN_ATTEMPTS_WARN}"
            )

    write_table(STATS_PATH, attempts_by_trick, failures_by_trick,
                calls_by_trick, len(records), warnings)
    print(f"\nwrote {STATS_PATH}")

    if warnings:
        print("\nWARNING: sampler failure rate / attempts exceed thresholds:")
        for w in warnings:
            print(f"  {w}")
        print("Per the plan: do not silently implement fixes. Consult the "
              "project owner about which planned fix to apply "
              "(most-constrained-first is the first choice).")
    else:
        print("\nNo threshold violations: failure rate stayed under "
              f"~{FAILURE_RATE_WARN} and mean attempts stayed under "
              f"~{MEAN_ATTEMPTS_WARN} at every trick.")


if __name__ == "__main__":
    main()
