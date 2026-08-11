"""Task 5 probe: does a value-truncated search agree with full playouts, and
how much cheaper is it?

This is a DECISION-QUALITY + SPEED probe, run before any strength ablation.
It answers two questions on the same sampled decisions:

(a) agreement -- how often `horizon=k` picks the same card as `horizon=None`
    (the full-playout search that produced every published row);
(b) cost -- mean seconds per decision, and one full game's wall clock.

Why the timing half is as load-bearing as the agreement half: the value net v1
FAILED its pre-registered MSE gate against a single heuristic playout (Task 3
results block), and the owner is proceeding under option (c) -- the remaining
case for the net is SPEED at equal compute, i.e. the same wall clock spent on
more worlds. A horizon that agrees 85% of the time at 10x the speed is a good
trade; one that agrees 85% at 1.5x is not. So read the two columns together.

Method
------
Decisions come from `results/heuristic_games.txt` (the established records,
replayed exactly as run_guessing2.py does it), taking every position where the
seat to play has more than one legal move, then stride-sampling across the
whole file so early, middle and late tricks are all represented.

Every horizon sees the SAME decision with the SAME player rng seed, so the
outer worlds and the inner re-determinization worlds are the identical
arrangements. The only difference between rows is how a world is scored. That
is what makes "agreement" mean what it says.

Config (plain honest search -- the CHOICE posterior is deliberately NOT used
here; this probe isolates the evaluator, and the posterior is an orthogonal
knob measured in Task 6).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from openhearts.belief.table import Level  # noqa: E402
from openhearts.engine import cards, kernel  # noqa: E402
from openhearts.engine.features import FEATURES_V  # noqa: E402
from openhearts.engine.game import deal, play_game  # noqa: E402
from openhearts.engine.state import GameState  # noqa: E402
from openhearts.eval.records import read_records  # noqa: E402
from openhearts.players.heuristic import HeuristicPlayer  # noqa: E402
from openhearts.search.valuesearch import ValueSearchPlayer  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
RECORDS_PATH = os.path.join(RESULTS, "heuristic_games.txt")
OUT_PATH = os.path.join(RESULTS, "value_probe.txt")
SMOKE_OUT_PATH = os.path.join(RESULTS, "value_probe_smoke.txt")
WEIGHTS = os.path.join(ROOT, "models", "value_v1.npz")

N_DECISIONS = 300
SMOKE_DECISIONS = 20
N_OUTER = 50
N_INNER = 20
LEVEL = Level.FULL
HORIZONS = [0, 1, 2, None]
REFERENCE = None            # the row everything is compared against
MASTER_SEED = 40000         # player rng seeds: MASTER_SEED + decision index
GAME_SEED = 700001          # full-game timing deal (NOT an ablation seed:
                            # the 100000+ range is reserved for paired
                            # strength rows, and this is a stopwatch)
N_RECORDS = 60              # records scanned for decisions (>>300 available)

PREREG = ("horizon 1-2 agrees with full playouts on a large majority of "
          "decisions at a fraction of the cost; horizon 0 is fastest and "
          "least accurate.")
MAJORITY = 0.75             # the reading of "large majority", fixed pre-run


def _label(h):
    return "None" if h is None else str(h)


def collect_decisions(n_wanted, n_records):
    """Views at real decisions, stride-sampled across records and tricks."""
    recs = read_records(RECORDS_PATH)[:n_records]
    views = []
    for rec in recs:
        state = GameState(hands=list(rec.hands))
        state.to_play = next(s for s in range(4)
                             if rec.hands[s] & cards.bit(cards.TWO_CLUBS))
        assert state.to_play == rec.plays[0][0], "record leader is not 2c"
        for seat, card in rec.plays:
            assert seat == state.to_play, "replay desync"
            view = state.view_for(seat)
            if len(cards.cards_in(view.legal_moves)) > 1:
                views.append(view)
            state.play(card)
        assert state.is_over(), "record did not replay to a finished game"
    assert len(views) >= n_wanted, (
        f"only {len(views)} decisions in {n_records} records")
    step = max(1, len(views) // n_wanted)
    return views[::step][:n_wanted]


def make_player(horizon, seed):
    return ValueSearchPlayer(LEVEL, n_outer=N_OUTER, n_inner=N_INNER,
                             rng=np.random.default_rng(seed),
                             horizon=horizon, weights_path=WEIGHTS)


def run_decisions(views):
    """{horizon: (choices, seconds_in_choose, value_calls)}

    Only `choose` is timed. Player construction is deliberately OUTSIDE the
    clock: the constructor reads the weights `.npz`, and a sub-millisecond
    file read is a several-percent tax on the horizon=0 row (whole decisions
    there cost single-digit milliseconds) while being pure noise on the
    horizon=None row -- i.e. it would bias the comparison against precisely
    the row whose speed is the headline. One discarded warm-up `choose` per
    horizon keeps a cold numba compile out of whichever row runs first.
    """
    out = {}
    for h in HORIZONS:
        make_player(h, MASTER_SEED).choose(views[0])   # warm-up, discarded
        choices, calls, secs = [], 0, 0.0
        for i, view in enumerate(views):
            p = make_player(h, MASTER_SEED + i)
            t0 = time.perf_counter()
            choices.append(p.choose(view))
            secs += time.perf_counter() - t0
            calls += 0 if h is None else p.value_calls
        out[h] = (choices, secs, calls)
    return out


def run_full_game(horizon):
    rng = np.random.default_rng(GAME_SEED)
    state0 = deal(rng)
    bot = make_player(horizon, MASTER_SEED)
    players = [bot] + [HeuristicPlayer() for _ in range(3)]
    t0 = time.perf_counter()
    final = play_game(state0, players)
    dt = time.perf_counter() - t0
    assert sum(final.scores) == 26, "illegal game"
    return dt, final.scores[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n = SMOKE_DECISIONS if args.smoke else N_DECISIONS
    out_path = SMOKE_OUT_PATH if args.smoke else OUT_PATH

    views = collect_decisions(n, 8 if args.smoke else N_RECORDS)
    tricks = [v.trick_number for v in views]
    t_start = time.perf_counter()
    res = run_decisions(views)
    games = {h: run_full_game(h) for h in HORIZONS}
    wall = time.perf_counter() - t_start

    ref_choices = res[REFERENCE][0]
    ref_secs = res[REFERENCE][1] / len(views)

    lines = []
    add = lines.append
    add("Task 5 probe: value-truncated search vs full playouts")
    add("=" * 62)
    add(f"decisions           : {len(views)} "
        f"(records {RECORDS_PATH}, first "
        f"{8 if args.smoke else N_RECORDS}, stride-sampled)")
    add(f"trick spread        : min {min(tricks)} median "
        f"{int(np.median(tricks))} max {max(tricks)}")
    add(f"search config       : Level.{LEVEL.name}, n_outer={N_OUTER}, "
        f"n_inner={N_INNER}, jit_sampler=True")
    add("posterior           : NONE (plain honest search -- this probe "
        "isolates the evaluator)")
    add(f"player rng seed     : {MASTER_SEED} + decision index (IDENTICAL "
        "across horizons, so worlds are identical and only scoring differs)")
    add(f"weights             : {os.path.relpath(WEIGHTS, ROOT)} "
        f"(FEATURES_V={FEATURES_V})")
    add(f"jit                 : {kernel.jit_enabled()}")
    add(f"full-game deal seed : {GAME_SEED}, bot at seat 0 vs 3 "
        "HeuristicPlayers")
    add(f"smoke               : {args.smoke}")
    add("")
    add(f"{'horizon':>8} {'agree w/ None':>14} {'s/decision':>11} "
        f"{'speedup':>8} {'net calls/dec':>14} {'s/game':>8} {'pts':>5}")
    for h in HORIZONS:
        choices, secs, calls = res[h]
        agree = np.mean([a == b for a, b in zip(choices, ref_choices)])
        per = secs / len(views)
        gsec, gpts = games[h]
        add(f"{_label(h):>8} {agree:>14.3f} {per:>11.4f} "
            f"{ref_secs / per:>8.2f}x {calls / len(views):>14.1f} "
            f"{gsec:>8.2f} {gpts:>5d}")
    add("")
    add("agreement is with horizon=None on the SAME decision and the same "
        "sampled worlds; 1.000 for the None row by construction.")
    add("PRE-REGISTERED EXPECTATION (PHASE4_PLAN.md, Task 5, verbatim):")
    add(f'  "{PREREG}"')
    add("Reported against it clause by clause, nulls included. "
        f'"Large majority" is read as agreement >= {MAJORITY:.2f}, fixed here '
        "before the run.")
    agr = {h: float(np.mean([a == b for a, b in
                             zip(res[h][0], ref_choices)])) for h in HORIZONS}
    spd = {h: ref_secs / (res[h][1] / len(views)) for h in HORIZONS}
    for text, ok in [
        (f"horizon 1-2 agree on a large majority (>= {MAJORITY:.2f})",
         agr[1] >= MAJORITY and agr[2] >= MAJORITY),
        ("horizon 1-2 cost a fraction of full playouts (speedup > 1)",
         spd[1] > 1.0 and spd[2] > 1.0),
        ("horizon 0 is the fastest of {0,1,2}",
         spd[0] == max(spd[0], spd[1], spd[2])),
        ("horizon 0 is the least accurate of {0,1,2}",
         agr[0] == min(agr[0], agr[1], agr[2])),
    ]:
        add(f"  [{'PASS' if ok else 'FAIL'}] {text}")
    add("")
    add("horizon=0 reaches no imagined decision of our own, so the honest "
        "re-determinization never fires: that row is Phase-1 determinized "
        "search with a learned evaluator, NOT honest search. Label it so.")
    add("'net calls/dec' counts value_forward evaluations across both search "
        "stages; it is 0 whenever every imagined hand ended before the "
        "horizon (terminal positions score by actual points, never the net).")
    add(f"total wall time     : {wall:.1f}s")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
