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
import json
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


BOT_KINDS = ("honest", "expert-rollout")


def build_our_bot(rng, n_outer, n_inner, bot_kind="honest"):
    """`honest` = the frozen champion (C-bench 1 protocol, unfused).
    `expert-rollout` = Phase 7C's ExpertRolloutSearchPlayer (train-side
    experts as the opponent playout policy; unfused by construction)."""
    if bot_kind == "expert-rollout":
        from openhearts.players.expert_population import train_ids
        from openhearts.search.expert_rollout import ExpertRolloutSearchPlayer
        return ExpertRolloutSearchPlayer(Level.FULL, n_outer, n_inner, rng,
                                         rollout_ids=train_ids())
    assert bot_kind == "honest", bot_kind
    return HonestSearchPlayer(Level.FULL, n_outer, n_inner, rng,
                               sampler_respects_voids=True,
                               posterior_factory=None)


# ---- B1: the ISMCTS rung (ROADMAP post-Phase-7 B1; PHASE7_PLAN.md TASK B1) ----
# The stock rung is OpenSpiel's ISMCTS exactly as C-bench 1 ran it. Other rungs
# plug our heuristic into OpenSpiel's evaluator hooks and/or retune its knobs.
# The config travels to spawned workers through the environment (JSON), like
# the other harnesses; at the defaults the header and every game are unchanged.
ISMCTS_RUNGS = ("stock", "tuned", "heur-rollout", "heur-prior")
ISMCTS_DEFAULTS = {"rung": "stock", "uct_c": 2.0, "world_samples": -1,
                   "final_policy": "MAX_VISIT_COUNT", "child_selection": "UCT",
                   "prior_eps": None}
_ISMCTS_ENV = "CBENCH_ISMCTS_CFG"


def ismcts_cfg():
    cfg = dict(ISMCTS_DEFAULTS)
    raw = os.environ.get(_ISMCTS_ENV)
    if raw:
        cfg.update(json.loads(raw))
    return cfg


def ismcts_cfg_tag(cfg=None):
    """"" at the stock defaults (banked headers stay byte-identical), else a
    header token naming every knob -- the resume guard compares it."""
    cfg = cfg or ismcts_cfg()
    if all(cfg[k] == v for k, v in ISMCTS_DEFAULTS.items()):
        return ""
    return (f" ismcts={cfg['rung']}:c={cfg['uct_c']}:ws={cfg['world_samples']}"
            f":fp={cfg['final_policy']}:sel={cfg['child_selection']}:eps={cfg['prior_eps']}")


def build_ismcts_bot(seed, max_simulations):
    import pyspiel
    from open_spiel.python.algorithms import ismcts, evaluate_bots  # noqa: F401
    from open_spiel.python.algorithms.mcts import RandomRolloutEvaluator

    cfg = ismcts_cfg()
    game = pyspiel.load_game(adapter.GAME_STRING)
    if cfg["rung"] in ("stock", "tuned"):
        evaluator = RandomRolloutEvaluator(n_rollouts=1, random_state=np.random.RandomState(seed))
    else:
        from cbench.evaluators import HeuristicRolloutEvaluator
        evaluator = HeuristicRolloutEvaluator(
            prior_eps=cfg["prior_eps"] if cfg["rung"] == "heur-prior" else None)
    bot = ismcts.ISMCTSBot(
        game=game,
        evaluator=evaluator,
        uct_c=float(cfg["uct_c"]),
        max_simulations=max_simulations,
        max_world_samples=int(cfg["world_samples"]),
        random_state=np.random.RandomState(seed),
        final_policy_type=ismcts.ISMCTSFinalPolicyType[cfg["final_policy"]],
        child_selection_policy=ismcts.ChildSelectionPolicy[cfg["child_selection"]],
    )
    return bot


def play_one_game(deal_seed, our_seat_positions, n_outer, n_inner, max_simulations,
                   config_id, bot_kind="honest"):
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
        if bot_kind == "honest":
            # B0 (2026-09-26): stable seat seed (openhearts.eval.seeds) in place of
            # the per-process-salted hash() the banked rows were drawn with.
            # The same seed also seeds the ISMCTS bot -> a C-bench game is replayable.
            from openhearts.eval.seeds import HARNESS_CBENCH, bench_seat_seed
            seed = bench_seat_seed(HARNESS_CBENCH,
                                   0 if direction_of(config_id) == "ours-minority" else 1,
                                   config_id[1], deal_seed, seat)
        else:
            from openhearts.players.expert_population import derive_seed, DOMAIN_SEAT_ROTATION
            seed = derive_seed(DOMAIN_SEAT_ROTATION, 7300, deal_seed, seat,
                               0 if direction_of(config_id) == "ours-minority" else 1,
                               config_id[1]) & 0xFFFFFFFF
        if seat in our_seat_positions:
            our_bots[seat] = build_our_bot(np.random.default_rng(seed), n_outer, n_inner, bot_kind)
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
    diag = {"n_evaluate": 0, "n_inconsistent": 0, "n_illegal_replays": 0}   # B1 report-only
    for b in ismcts_bots.values():
        ev = getattr(b, "_evaluator", None)
        if ev is not None and hasattr(ev, "n_inconsistent_worlds"):
            diag["n_evaluate"] += ev.n_evaluate
            diag["n_inconsistent"] += ev.n_inconsistent_worlds
            diag["n_illegal_replays"] += ev.n_illegal_replays
    return points, our_times, ismcts_times, diag


def direction_of(config_id):
    return config_id[0]


def play_one_deal_rotated(deal_seed, direction, n_outer, n_inner, max_simulations, bot_kind="honest"):
    """4 rotations of one deal, minority bot rotated through every seat.
    Returns (our_avg_points, ismcts_avg_points, our_times, ismcts_times)."""
    our_total, ismcts_total = 0.0, 0.0
    our_times, ismcts_times = [], []
    diag = {"n_evaluate": 0, "n_inconsistent": 0, "n_illegal_replays": 0}
    for rotation in range(4):
        if direction == "ours-minority":
            our_seats = {rotation}
        elif direction == "ours-majority":
            our_seats = set(range(4)) - {rotation}
        else:
            raise ValueError(direction)
        config_id = (direction, rotation)
        points, ot, it, dg = play_one_game(deal_seed, our_seats, n_outer, n_inner,
                                           max_simulations, config_id, bot_kind)
        for k in diag:
            diag[k] += dg[k]
        our_pts = sum(points[s] for s in our_seats) / len(our_seats)
        ismcts_seats = set(range(4)) - our_seats
        ismcts_pts = sum(points[s] for s in ismcts_seats) / len(ismcts_seats)
        our_total += our_pts
        ismcts_total += ismcts_pts
        our_times.extend(ot)
        ismcts_times.extend(it)
    return our_total / 4.0, ismcts_total / 4.0, our_times, ismcts_times, diag


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


def run(n_deals, workers, direction, n_outer, n_inner, max_simulations, seed_base,
        bot_kind="honest"):
    final_path, partial_path = _partial_paths(direction, n_outer, n_inner)
    # Loud guard: if the partial's header records a different config than
    # this run, refuse -- resuming across configs would mix two bots' deals.
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            first = f.readline()
        want = f"n_outer={n_outer} n_inner={n_inner}"
        tag = ismcts_cfg_tag()
        if first.startswith("#") and ((" ismcts=" in first) != bool(tag) or (tag and tag.strip() not in first)):
            raise AssertionError(
                f"{partial_path} was banked under a different ISMCTS rung/knobs than this run "
                f"({tag.strip() or 'stock defaults'}): refusing to resume. Header: {first.strip()}")
        if first.startswith("#") and want not in first:
            raise AssertionError(
                f"{partial_path} header does not match this run's config "
                f"({want}): refusing to resume across configs. Header: "
                f"{first.strip()}")
        # B0 guard: an honest run under stable seeds may only resume a partial
        # that was written under stable seeds. The banked pre-B0 files carry no
        # seeds token, so this refuses them -- use a fresh --partial-tag.
        from openhearts.eval.seeds import SEEDS_TOKEN
        if bot_kind == "honest" and first.startswith("#") and SEEDS_TOKEN not in first:
            raise AssertionError(
                f"{partial_path} was banked BEFORE the B0 seed repair (header lacks "
                f"'{SEEDS_TOKEN}'): a stable-seed run must not append to it. "
                f"Use a new --partial-tag. Header: {first.strip()}")
    banked = _load_partial(partial_path)
    deal_seeds = [seed_base + i for i in range(n_deals)]

    header = (
        f"# cbench direction={direction} n_deals={n_deals} workers={workers} "
        f"n_outer={n_outer} n_inner={n_inner} max_simulations={max_simulations} "
        f"seed_base={seed_base} game={adapter.GAME_STRING}"
        + (f" bot={bot_kind}" if bot_kind != "honest" else "")
        + (" seeds=stable-v1" if bot_kind == "honest" else "")
        + ismcts_cfg_tag() + "\n"
    )
    if not os.path.exists(partial_path):
        with open(partial_path, "w") as f:
            f.write(header)

    to_run = [s for s in deal_seeds if f"our@{s}" not in banked]
    print(f"[cbench] {len(banked)//2 if banked else 0} deals already banked; "
          f"{len(to_run)} to run with {workers} worker(s)")

    our_times_all, ismcts_times_all = [], []
    our_pts_all, ismcts_pts_all = [], []
    diag_all = {"n_evaluate": 0, "n_inconsistent": 0, "n_illegal_replays": 0}   # B1 report-only diagnostic

    # Pre-existing banked results (for the final report)
    for s in deal_seeds:
        if f"our@{s}" in banked and f"ismcts@{s}" in banked:
            our_pts_all.append(banked[f"our@{s}"])
            ismcts_pts_all.append(banked[f"ismcts@{s}"])

    t_start = time.perf_counter()
    if workers <= 1:
        for s in to_run:
            our_pts, ismcts_pts, ot, it, dg = play_one_deal_rotated(
                s, direction, n_outer, n_inner, max_simulations, bot_kind)
            for k in diag_all:
                diag_all[k] += dg[k]
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
                                n_inner, max_simulations, bot_kind): s for s in to_run}
            for fut in cf.as_completed(futs):
                s = futs[fut]
                our_pts, ismcts_pts, ot, it, dg = fut.result()
                for k in diag_all:
                    diag_all[k] += dg[k]
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
    if diag_all["n_evaluate"]:
        lines.append(f"ismcts_worlds_inconsistent_with_history={diag_all['n_inconsistent']} of "
                     f"{diag_all['n_evaluate']} evaluate calls "
                     f"({100.0 * diag_all['n_inconsistent'] / diag_all['n_evaluate']:.1f}%), "
                     f"illegal_recorded_plays={diag_all['n_illegal_replays']}  [B1 report-only diagnostic]\n")
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
    ap.add_argument("--bot", choices=list(BOT_KINDS), default="honest",
                    help="7C: expert-rollout = train experts as the opponent playout "
                         "policy (requires --partial-tag; stable derive_seed seat seeds)")
    ap.add_argument("--rung", choices=list(ISMCTS_RUNGS), default="stock",
                    help="B1 ISMCTS rung: stock (C-bench 1), tuned (random rollout, retuned knobs), "
                         "heur-rollout (our heuristic as the rollout), heur-prior (+ heuristic prior, PUCT)")
    ap.add_argument("--uct-c", type=float, default=ISMCTS_DEFAULTS["uct_c"])
    ap.add_argument("--world-samples", type=int, default=ISMCTS_DEFAULTS["world_samples"],
                    help="max_world_samples (-1 = unlimited)")
    ap.add_argument("--final-policy", default=ISMCTS_DEFAULTS["final_policy"],
                    choices=["MAX_VISIT_COUNT", "MAX_VALUE", "NORMALIZED_VISITED_COUNT"])
    ap.add_argument("--child-selection", default=None, choices=["UCT", "PUCT"],
                    help="default UCT; heur-prior defaults to PUCT")
    ap.add_argument("--prior-eps", type=float, default=None,
                    help="heur-prior only: (1-eps) on the heuristic's card, eps/|legal| on each legal card")
    ap.add_argument("--partial-tag", default="",
                    help="suffix for the partial/final filenames (e.g. "
                         "'_envcheck') so a default-config run never touches "
                         "the banked legacy files")
    args = ap.parse_args()

    global _PARTIAL_TAG
    _PARTIAL_TAG = args.partial_tag

    if args.bot != "honest":
        assert args.partial_tag, "--bot expert-rollout requires --partial-tag (banked files stay untouched)"
    cfg = dict(ISMCTS_DEFAULTS)
    cfg.update(rung=args.rung, uct_c=args.uct_c, world_samples=args.world_samples,
               final_policy=args.final_policy,
               child_selection=args.child_selection or ("PUCT" if args.rung == "heur-prior" else "UCT"),
               prior_eps=args.prior_eps if args.rung == "heur-prior" else None)
    if args.rung == "heur-prior":
        assert args.prior_eps is not None, "--rung heur-prior requires --prior-eps"
    os.environ[_ISMCTS_ENV] = json.dumps(cfg)
    if ismcts_cfg_tag(cfg):
        assert args.partial_tag, "a non-stock ISMCTS rung/knob requires --partial-tag (banked files stay untouched)"
    run(args.deals, args.workers, args.direction, args.n_outer, args.n_inner,
        args.max_simulations, args.seed_base, args.bot)


if __name__ == "__main__":
    main()
