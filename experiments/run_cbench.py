"""C-bench 1: our frozen champion (honest-FULL) vs OpenSpiel's generic
ISMCTS bot in OpenSpiel's own hearts(pass_cards=False) state machine.

MUST be run from a venv with `pyspiel` installed (NOT the project .venv --
see experiments/cbench/RULES_ALIGNMENT.md for how that venv is built). The
project .venv never gains an OpenSpiel dependency.

Design (see experiments/cbench/RULES_ALIGNMENT.md and adapter.py for the
full rationale):
  - OpenSpiel's pyspiel.State drives the game; the ISMCTS bot is native,
    unmodified OpenSpiel, and only ever sees that state.
  - Our bot is wrapped: a harness-side mirror GameState (holding the true
    deal) replays the same action sequence; our bot only ever receives a
    PlayerView via mirror.view_for(seat).
  - After every action, the mirror's legal moves are asserted to agree with
    OpenSpiel's own legal_actions() (translated) -- the live rules-alignment
    tripwire. Aborts loudly (raises) on disagreement; never swallowed.
  - Scoring is always OUR rescoring from the played-card history (no
    moon-shoot rule on our side, see RULES_ALIGNMENT.md sec 3) -- OpenSpiel's
    own returns() is not used for the reported result.
  - Deals: forced identical via our own deal(seed) pushed through OpenSpiel's
    chance nodes (adapter.force_deal).
  - Each deal is played with the "minority" bot rotated through all 4 seats
    (like eval/harness.py's rotated_match), one direction per run:
      --direction ours-minority  (1 ours vs 3 ISMCTS, default)
      --direction ours-majority  (3 ours vs 1 ISMCTS)
  - Checkpointed, resumable, multiprocessing worker pool (--workers).
  - Per-decision wall-clock timing recorded for BOTH bot types.

Output: results/cbench_<direction>.txt (final) and
results/cbench_<direction>_partial.txt (checkpoint, resumable).
"""
import argparse
import concurrent.futures as cf
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from cbench import adapter  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from openhearts.belief.table import Level  # noqa: E402
from openhearts.engine import cards  # noqa: E402
from openhearts.search.honest import HonestSearchPlayer  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

SEED_BASE = 100000
N_OUTER_DEFAULT = 50
N_INNER_DEFAULT = 20
MAX_SIMULATIONS_DEFAULT = 1000


def build_our_bot(rng, n_outer, n_inner):
    return HonestSearchPlayer(Level.FULL, n_outer, n_inner, rng,
                               sampler_respects_voids=True,
                               posterior_factory=None)


def build_ismcts_bot(seed, max_simulations):
    import pyspiel
    from open_spiel.python.algorithms import ismcts, evaluate_bots  # noqa: F401
    from open_spiel.python.algorithms.mcts import RandomRolloutEvaluator

    game = pyspiel.load_game(adapter.GAME_STRING)
    evaluator = RandomRolloutEvaluator(n_rollouts=1, random_state=np.random.RandomState(seed))
    bot = ismcts.ISMCTSBot(
        game=game,
        evaluator=evaluator,
        uct_c=2.0,
        max_simulations=max_simulations,
        random_state=np.random.RandomState(seed),
    )
    return bot


def play_one_game(deal_seed, our_seat_positions, n_outer, n_inner, max_simulations,
                   config_id):
    """Play one full deal, with our bot occupying `our_seat_positions` (a set
    of seats) and the ISMCTS bot occupying the rest. Returns:
      (points_by_seat: list[4], our_decision_times: list[float],
       ismcts_decision_times: list[float])
    """
    import pyspiel

    game = pyspiel.load_game(adapter.GAME_STRING)
    os_state = game.new_initial_state()
    mirror = adapter.force_deal(os_state, seed=deal_seed)

    our_bots = {}
    ismcts_bots = {}
    for seat in range(4):
        seed = hash((config_id, deal_seed, seat)) & 0xFFFFFFFF
        if seat in our_seat_positions:
            our_bots[seat] = build_our_bot(np.random.default_rng(seed), n_outer, n_inner)
        else:
            ismcts_bots[seat] = build_ismcts_bot(seed, max_simulations)

    our_times, ismcts_times = [], []
    while not os_state.is_terminal():
        adapter.assert_legal_agreement(os_state, mirror)
        seat = mirror.to_play
        t0 = time.perf_counter()
        if seat in our_seat_positions:
            view = mirror.view_for(seat)
            card_ours = our_bots[seat].choose(view)
            our_times.append(time.perf_counter() - t0)
        else:
            action_os = ismcts_bots[seat].step(os_state)
            ismcts_times.append(time.perf_counter() - t0)
            card_ours = adapter.os_to_ours(action_os)
        adapter.apply_both(os_state, mirror, card_ours)

    points = adapter.rescore(mirror)
    assert sum(points) == 26
    return points, our_times, ismcts_times


def play_one_deal_rotated(deal_seed, direction, n_outer, n_inner, max_simulations):
    """4 rotations of one deal, minority bot rotated through every seat.
    Returns (our_avg_points, ismcts_avg_points, our_times, ismcts_times)."""
    our_total, ismcts_total = 0.0, 0.0
    our_times, ismcts_times = [], []
    for rotation in range(4):
        if direction == "ours-minority":
            our_seats = {rotation}
        elif direction == "ours-majority":
            our_seats = set(range(4)) - {rotation}
        else:
            raise ValueError(direction)
        config_id = (direction, rotation)
        points, ot, it = play_one_game(deal_seed, our_seats, n_outer, n_inner,
                                       max_simulations, config_id)
        our_pts = sum(points[s] for s in our_seats) / len(our_seats)
        ismcts_seats = set(range(4)) - our_seats
        ismcts_pts = sum(points[s] for s in ismcts_seats) / len(ismcts_seats)
        our_total += our_pts
        ismcts_total += ismcts_pts
        our_times.extend(ot)
        ismcts_times.extend(it)
    return our_total / 4.0, ismcts_total / 4.0, our_times, ismcts_times


def _partial_paths(direction, n_outer=N_OUTER_DEFAULT, n_inner=N_INNER_DEFAULT):
    """Partial/final paths, CONFIG-QUALIFIED for non-default search sizes.

    Phase 7 Task A3 runs this script at several (n_outer, n_inner) operating
    points. The original paths were keyed by direction alone; letting a
    100x20 run resume from the banked 50x20 partial would silently mix two
    different bots' deals into one file -- and the banked partials are
    append-only reference data for the published C-bench numbers. The
    default config keeps its legacy names so those banked files stay
    resumable; every other config gets its own pair.
    """
    cfg = ""
    if (n_outer, n_inner) != (N_OUTER_DEFAULT, N_INNER_DEFAULT):
        cfg = f"_{n_outer}x{n_inner}"
    cfg += _PARTIAL_TAG
    final = os.path.join(RESULTS, f"cbench_{direction}{cfg}.txt")
    partial = os.path.join(RESULTS, f"cbench_{direction}{cfg}_partial.txt")
    return final, partial


#: Optional path suffix (set from --partial-tag) so a DEFAULT-config run can
#: be routed away from the banked legacy files -- e.g. A3's same-environment
#: incumbent control, which must not resume from (or append to) the published
#: C-bench partial. Empty for every historical invocation.
_PARTIAL_TAG = ""


def _load_partial(partial_path):
    done = {}
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
                key, val = parts
                done[key] = float(val)
    return done


def _append_partial(partial_path, key, val):
    with open(partial_path, "a") as f:
        f.write(f"{key} {val}\n")


def run(n_deals, workers, direction, n_outer, n_inner, max_simulations, seed_base):
    final_path, partial_path = _partial_paths(direction, n_outer, n_inner)
    # Loud guard: if the partial's header records a different config than
    # this run, refuse -- resuming across configs would mix two bots' deals.
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            first = f.readline()
        want = f"n_outer={n_outer} n_inner={n_inner}"
        if first.startswith("#") and want not in first:
            raise AssertionError(
                f"{partial_path} header does not match this run's config "
                f"({want}): refusing to resume across configs. Header: "
                f"{first.strip()}")
    banked = _load_partial(partial_path)
    deal_seeds = [seed_base + i for i in range(n_deals)]

    header = (
        f"# cbench direction={direction} n_deals={n_deals} workers={workers} "
        f"n_outer={n_outer} n_inner={n_inner} max_simulations={max_simulations} "
        f"seed_base={seed_base} game={adapter.GAME_STRING}\n"
    )
    if not os.path.exists(partial_path):
        with open(partial_path, "w") as f:
            f.write(header)

    to_run = [s for s in deal_seeds if f"our@{s}" not in banked]
    print(f"[cbench] {len(banked)//2 if banked else 0} deals already banked; "
          f"{len(to_run)} to run with {workers} worker(s)")

    our_times_all, ismcts_times_all = [], []
    our_pts_all, ismcts_pts_all = [], []

    # Pre-existing banked results (for the final report)
    for s in deal_seeds:
        if f"our@{s}" in banked and f"ismcts@{s}" in banked:
            our_pts_all.append(banked[f"our@{s}"])
            ismcts_pts_all.append(banked[f"ismcts@{s}"])

    t_start = time.perf_counter()
    if workers <= 1:
        for s in to_run:
            our_pts, ismcts_pts, ot, it = play_one_deal_rotated(
                s, direction, n_outer, n_inner, max_simulations)
            _append_partial(partial_path, f"our@{s}", our_pts)
            _append_partial(partial_path, f"ismcts@{s}", ismcts_pts)
            our_pts_all.append(our_pts)
            ismcts_pts_all.append(ismcts_pts)
            our_times_all.extend(ot)
            ismcts_times_all.extend(it)
            print(f"[cbench] deal {s}: our={our_pts:.2f} ismcts={ismcts_pts:.2f}")
    else:
        with cf.ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(play_one_deal_rotated, s, direction, n_outer,
                                n_inner, max_simulations): s for s in to_run}
            for fut in cf.as_completed(futs):
                s = futs[fut]
                our_pts, ismcts_pts, ot, it = fut.result()
                _append_partial(partial_path, f"our@{s}", our_pts)
                _append_partial(partial_path, f"ismcts@{s}", ismcts_pts)
                our_pts_all.append(our_pts)
                ismcts_pts_all.append(ismcts_pts)
                our_times_all.extend(ot)
                ismcts_times_all.extend(it)
                print(f"[cbench] deal {s}: our={our_pts:.2f} ismcts={ismcts_pts:.2f}")
    elapsed = time.perf_counter() - t_start

    def summarize(times):
        if not times:
            return {}
        s = sorted(times)
        return {
            "mean": statistics.mean(s),
            "median": statistics.median(s),
            "p95": s[int(0.95 * (len(s) - 1))],
            "n": len(s),
        }

    our_summary = summarize(our_times_all)
    ismcts_summary = summarize(ismcts_times_all)

    lines = [header]
    lines.append(f"wall_time_s={elapsed:.2f} deals_this_run={len(to_run)}\n")
    lines.append(f"our_avg_points={statistics.mean(our_pts_all):.4f} "
                 f"n={len(our_pts_all)}\n")
    lines.append(f"ismcts_avg_points={statistics.mean(ismcts_pts_all):.4f} "
                 f"n={len(ismcts_pts_all)}\n")
    lines.append(f"our_decision_time_s mean={our_summary.get('mean', float('nan')):.5f} "
                 f"median={our_summary.get('median', float('nan')):.5f} "
                 f"p95={our_summary.get('p95', float('nan')):.5f} n={our_summary.get('n', 0)}\n")
    lines.append(f"ismcts_decision_time_s mean={ismcts_summary.get('mean', float('nan')):.5f} "
                 f"median={ismcts_summary.get('median', float('nan')):.5f} "
                 f"p95={ismcts_summary.get('p95', float('nan')):.5f} n={ismcts_summary.get('n', 0)}\n")
    with open(final_path, "w") as f:
        f.writelines(lines)
    print("".join(lines))
    return our_pts_all, ismcts_pts_all, our_summary, ismcts_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--direction", choices=["ours-minority", "ours-majority"],
                    default="ours-minority")
    ap.add_argument("--n-outer", type=int, default=N_OUTER_DEFAULT)
    ap.add_argument("--n-inner", type=int, default=N_INNER_DEFAULT)
    ap.add_argument("--max-simulations", type=int, default=MAX_SIMULATIONS_DEFAULT)
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    ap.add_argument("--partial-tag", default="",
                    help="suffix for the partial/final filenames (e.g. "
                         "'_envcheck') so a default-config run never touches "
                         "the banked legacy files")
    args = ap.parse_args()

    global _PARTIAL_TAG
    _PARTIAL_TAG = args.partial_tag

    run(args.deals, args.workers, args.direction, args.n_outer, args.n_inner,
        args.max_simulations, args.seed_base)


if __name__ == "__main__":
    main()
