"""Task 8: robustness -- what strict choice-filtering costs against
imperfect opponents.

Task 7 measured choice-aware guessing against the opponents the model assumes:
4x deterministic `HeuristicPlayer`. That is the best case, and it is not the
case that will hold against humans. This experiment breaks the assumption on
purpose. The games are played by 4x `RandomizedHeuristic(epsilon_true=0.1)`,
which plays the heuristic's card 90% of the time and a uniformly random OTHER
legal card 10% of the time; the guessers still assume a plain `HeuristicPlayer`.
Three guessers are compared at every completed-trick boundary:

  FULL          constraint-only `BeliefTable.from_view(view, Level.FULL)` --
                voids + hand sizes only, so it is policy-independent and
                cannot be wrong about the opponents.
  CHOICE-strict `WeightedPosterior(..., HeuristicPlayer(), epsilon=0.0)` --
                deliberately wrong: it treats every observed play as if the
                heuristic must have produced it, so a single deviation kills
                the true world outright.
  CHOICE-soft   the same with epsilon=0.1 -- uniform-over-all-legal-moves
                smoothing, a crude "my opponent model may be wrong" knob.

Coincidence worth naming: soft's epsilon = 0.1 numerically equals the
opponents' epsilon_true = 0.1, but they are NOT the same noise model.
`WeightedPosterior` smooths uniformly over ALL legal moves (including the
heuristic's own choice); `RandomizedHeuristic` spreads its epsilon over the
num_legal - 1 OTHER moves. Soft is therefore still a wrong model of these
opponents, just a forgiving one. The variable under test is strict's
epsilon = 0.

Pre-registered expectations (PHASE23_PLAN.md, Task 8), quoted verbatim:
"strict beats constraint-only early but degrades or hard-fails as deviations
accumulate; soft epsilon=0.1 dominates strict against these opponents and
never confidently excludes the truth."

--------------------------------------------------------------------------
Honesty rules -- exactly how each metric handles truth zeros and collapse
--------------------------------------------------------------------------
Two units of aggregation appear below and are never mixed:

  * boundary-observer -- one (game, trick, observer seat) triple. There are
    n_games * 12 tricks * 4 observers of them per guesser.
  * card -- one truth card inside one boundary-observer (39 - 3*(k-1) of them
    entering trick k).

COLLAPSE. `WeightedPosterior.from_view` raises "no candidate world survived"
when every one of its up-to-max_draws candidate worlds gets weight 0. That is
a real failure of the guesser, not an error: it is caught (BY MESSAGE -- every
other AssertionError in weighted.py signals a structural bug and is
re-raised). Such a boundary-observer produces NO probabilities at all, so:
  - collapse_frac  = collapsed boundary-observers / ALL boundary-observers.
  - every other metric EXCLUDES collapsed boundary-observers. Their common
    denominator is n_scored (boundary-observers) or the cards inside those
    (n_cards). Nothing is imputed as zero.
  - FULL's collapse_frac is 0 BY CONSTRUCTION, not by measurement: a
    constraint table has no candidate worlds that could die. FULL is not
    wrapped in try/except at all -- any exception from it is a bug and
    propagates.

SURVIVORSHIP. Because collapsed calls are excluded, a guesser's meanP is
CONDITIONAL on it having produced an answer. For strict, whose collapse_frac
is large late, that number flatters it. `meanP_uncond = meanP * (1 -
collapse_frac)` scores a collapse as zero information and is the column that
actually adjudicates "degrades or hard-fails". FULL's two meanP columns are
identical by construction.

TRUTH ZEROS. A true card whose marginal is exactly 0.0. NOT floored, NOT
smoothed, NOT excluded: it counts as 0.0 in meanP, as a miss in top1, and as
confidently-wrong. `truth_zero_frac` = such cards / n_cards. There is no NLL
column in this experiment, so the exclusion run_guessing2 had to make does not
arise. For FULL a truth zero is impossible (voids and hand sizes are hard
policy-independent facts) and is asserted against, exactly as Phase 1's
`eval.guessing.metrics_for` does.

CONFIDENTLY WRONG. `confidently_wrong_frac` = cards whose true holder got
marginal < 0.01 / n_cards -- the same per-card denominator as
truth_zero_frac, over the same non-collapsed boundary-observers. Truth zeros
are a subset of it.

TRUE-WORLD DIAGNOSTIC. `true_world_zero_frac` (strict/soft only) is the
fraction of boundary-observers at which the TRUE world itself audits to
weight 0.0 under that guesser's epsilon. This is ground truth touched only
for diagnostics, never fed to a guesser. For strict it is the direct
mechanistic cause of collapse and of confidently-wrong: it says "by now the
opponents have deviated at least once, so the model has excluded reality".
For soft it must be 0.0 everywhere -- the true world always replays legally
and every policy factor is >= epsilon/num_legal > 0 -- and the smoke run
asserts a positive true-world soft weight at every boundary-observer, which
is this experiment's replacement for run_guessing2's `weight == 1.0`
invariant (that invariant is FALSE here by design: deviations are real).

SAMPLING HEALTH. `n_eff` and `draws` are reported per guesser. Soft's weights
are products of ~30 factors that are either ~0.9 or ~epsilon/num_legal, so
100 positive-weight worlds can still mean n_effective ~ 1-3; without this
column degenerate soft marginals look inexplicable. A truth zero under soft
at a late trick is far more likely "never sampled" than "excluded by the
model", and the two are distinguished by reading n_eff alongside
true_world_zero_frac.

--------------------------------------------------------------------------
Determinism
--------------------------------------------------------------------------
Master seed 3026. Game i is dealt from `default_rng([3026, i])`; seat s of
game i plays with its own `default_rng([3026, i, s])` (four distinct rngs,
the `[config, seed, ..., seat]` convention of run_bprobe.py). Games are
simulated in the parent process and written to results/robustness_games.txt
in the standard `eval.records` v1 format, so the whole run is replayable.
Belief rngs use run_survival.py / run_guessing2.py's scheme,
`default_rng([rec.seed, trick, observer])` -> seed; strict and soft get the
SAME seed at each boundary-observer so their proposal streams start paired.
"""
import argparse
import concurrent.futures as cf
import os
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from openhearts.belief.table import BeliefTable, Level  # noqa: E402
from openhearts.belief.weighted import (  # noqa: E402
    WeightedPosterior, world_weight,
)
from openhearts.engine import cards  # noqa: E402
from openhearts.engine.game import deal, play_game  # noqa: E402
from openhearts.engine.state import GameState  # noqa: E402
from openhearts.eval import guessing  # noqa: E402
from openhearts.eval.records import record_from, write_records  # noqa: E402
from openhearts.players.heuristic import HeuristicPlayer  # noqa: E402
from openhearts.players.randomized import RandomizedHeuristic  # noqa: E402

RESULTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
OUT_PATH = os.path.join(RESULTS, "robustness.txt")
PLOT_PATH = os.path.join(RESULTS, "robustness.png")
PARTIAL_PATH = os.path.join(RESULTS, "robustness_partial.txt")
GAMES_PATH = os.path.join(RESULTS, "robustness_games.txt")
SMOKE_OUT_PATH = os.path.join(RESULTS, "robustness_smoke.txt")
SMOKE_PLOT_PATH = os.path.join(RESULTS, "robustness_smoke.png")
SMOKE_GAMES_PATH = os.path.join(RESULTS, "robustness_games_smoke.txt")

MASTER_SEED = 3026
NUM_GAMES = 300
EPSILON_TRUE = 0.1
NUM_TRICKS = guessing.NUM_TRICKS
FIRST_TRICK = 2  # entering trick 1 there is no evidence yet
LEVEL = Level.FULL
EPS_STRICT = 0.0
EPS_SOFT = 0.1
N_WORLDS = 100
MAX_DRAWS = 50000
CONFIDENT = 0.01

WORKERS = 12
CHUNK = 25

SMOKE_GAMES = 8
SMOKE_WORKERS = 2
SMOKE_CHUNK = 4

GUESSERS = ("FULL", "CHOICE-strict", "CHOICE-soft")
STYLE = {
    "FULL": dict(color="tab:blue", marker="o"),
    "CHOICE-strict": dict(color="crimson", marker="s"),
    "CHOICE-soft": dict(color="tab:green", marker="^"),
}
COLLAPSE_MSG = "no candidate world survived"


# --------------------------------------------------------------------------
# game generation
# --------------------------------------------------------------------------
def simulate_games(n_games):
    """n_games of 4x RandomizedHeuristic(epsilon_true), as GameRecords."""
    recs = []
    for i in range(n_games):
        state = deal(np.random.default_rng([MASTER_SEED, i]))
        initial_hands = list(state.hands)  # play_game mutates state.hands
        players = [
            RandomizedHeuristic(np.random.default_rng([MASTER_SEED, i, s]),
                                epsilon=EPSILON_TRUE)
            for s in range(4)
        ]
        final = play_game(state, players)
        assert sum(final.scores) == 26, "simulated game did not score 26"
        recs.append(record_from(i, initial_hands, final))
    return recs


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def _seed_for(rec_idx, trick, observer):
    """Deterministic rng seed -- same scheme as run_survival/run_guessing2."""
    return int(
        np.random.default_rng([rec_idx, trick, observer]).integers(
            0, 2**63 - 1
        )
    )


def score_probs(probs, truth):
    """Score a (3, 52) posterior against truth. Nothing is floored.

    Returns (meanP, top1, n_cards, n_zero, n_confident_wrong), where a truth
    zero counts as P = 0.0, as a top-1 miss, and as confidently wrong.
    """
    assert truth, "no cards to score"
    ps = []
    hits = []
    for c, i in truth.items():
        p = float(probs[i, c])
        assert np.isfinite(p) and p >= 0.0, f"bad probability {p}"
        ps.append(p)
        hits.append(1.0 if int(np.argmax(probs[:, c])) == i else 0.0)
    ps = np.asarray(ps, dtype=float)
    return (
        float(ps.mean()),
        float(np.mean(hits)),
        int(ps.size),
        int((ps == 0.0).sum()),
        int((ps < CONFIDENT).sum()),
    )


def _true_world_hands(state, opponent_seats):
    """Opponents' CURRENT true hands in opponent_seats order (diagnostic)."""
    return [int(state.hands[s]) for s in opponent_seats]


def _boundary(state, rec, trick, policy, verify_soft):
    """All 3 guessers x 4 observers at one completed-trick boundary."""
    out = {g: [] for g in GUESSERS}
    for seat in range(4):
        view = state.view_for(seat)

        # ---- FULL: constraint-only. No try/except by design (see docstring).
        table = BeliefTable.from_view(view, LEVEL)
        truth = guessing._truth_for(seat, rec, table)
        meanp, top1, n_cards, n_zero, n_cw = score_probs(table.probs, truth)
        assert n_zero == 0, (
            "constraint table zeroed a true holder -- voids and hand sizes "
            "are policy-independent facts, so this is a bug, not noise"
        )
        out["FULL"].append({
            "collapsed": False, "meanP": meanp, "top1": top1,
            "n_cards": n_cards, "n_zero": n_zero, "n_cw": n_cw,
            "n_effective": float("nan"), "draws_used": 0,
            "true_world_zero": 0,
        })

        # ---- the two choice guessers, paired on one proposal seed.
        for name, eps in (("CHOICE-strict", EPS_STRICT),
                          ("CHOICE-soft", EPS_SOFT)):
            rng = np.random.default_rng(_seed_for(rec.seed, trick, seat))
            # ground-truth diagnostic, never fed to the guesser
            tw = world_weight(
                view,
                _true_world_hands(state, [(seat + 1 + i) % 4
                                          for i in range(3)]),
                policy, eps,
            )
            if verify_soft and eps > 0.0:
                assert tw > 0.0, (
                    f"true world has weight {tw} at epsilon={eps}: with "
                    f"positive epsilon the true world always replays legally "
                    f"and every factor is >= eps/num_legal, so this means a "
                    f"replay or seat-mapping bug (record {rec.seed} trick "
                    f"{trick} observer {seat})"
                )
            try:
                post = WeightedPosterior.from_view(
                    view, LEVEL, policy, epsilon=eps, n_worlds=N_WORLDS,
                    rng=rng, max_draws=MAX_DRAWS,
                )
            except AssertionError as e:
                if COLLAPSE_MSG not in str(e):
                    raise  # structural bug in weighted.py, not a collapse
                out[name].append({"collapsed": True,
                                  "true_world_zero": int(tw == 0.0)})
                continue
            truth = guessing._truth_for(seat, rec, post)
            meanp, top1, n_cards, n_zero, n_cw = score_probs(post.probs, truth)
            out[name].append({
                "collapsed": False, "meanP": meanp, "top1": top1,
                "n_cards": n_cards, "n_zero": n_zero, "n_cw": n_cw,
                "n_effective": post.n_effective,
                "draws_used": post.draws_used,
                "true_world_zero": int(tw == 0.0),
            })
    return out


def process_record(rec, verify_soft=False):
    """Replay one record; per-guesser, per-trick lists of observer dicts."""
    results = {g: {t: [] for t in range(FIRST_TRICK, NUM_TRICKS + 1)}
               for g in GUESSERS}
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
            if trick >= FIRST_TRICK:
                per = _boundary(state, rec, trick, policy, verify_soft)
                for g in GUESSERS:
                    results[g][trick].extend(per[g])
    assert state.is_over(), "record did not replay to a finished game"
    return results


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def worker(args):
    chunk_records, verify_soft = args
    acc = {g: {t: [] for t in range(FIRST_TRICK, NUM_TRICKS + 1)}
           for g in GUESSERS}
    for rec in chunk_records:
        per_rec = process_record(rec, verify_soft)
        for g in GUESSERS:
            for t in acc[g]:
                acc[g][t].extend(per_rec[g][t])
    return acc


def total_rss_gb():
    out = subprocess.run(["ps", "-o", "rss=", "-g", str(os.getpgrp())],
                         capture_output=True, text=True).stdout
    return sum(int(x) for x in out.split()) / (1024 ** 2)


def run_all(records, workers, chunk_size, verify_soft, partial_path):
    acc = {g: {t: [] for t in range(FIRST_TRICK, NUM_TRICKS + 1)}
           for g in GUESSERS}
    chunks = list(_chunks(records, chunk_size))
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(worker, (c, verify_soft)) for c in chunks]
        done = 0
        for fut in cf.as_completed(jobs):
            chunk_acc = fut.result()
            for g in GUESSERS:
                for t in acc[g]:
                    acc[g][t].extend(chunk_acc[g][t])
            done += 1
            print(f"[{done}/{len(jobs)}] chunks done | "
                  f"mem={total_rss_gb():.1f}GB | "
                  f"{time.time() - t0:.0f}s elapsed", flush=True)
            if partial_path:
                _write_partial(partial_path, summarize(acc), done, len(jobs))
    return acc, time.time() - t0


def summarize(acc):
    rows = []
    for guesser in GUESSERS:
        for trick in range(FIRST_TRICK, NUM_TRICKS + 1):
            stats = acc[guesser][trick]
            if not stats:
                continue
            n_calls = len(stats)
            ok = [s for s in stats if not s["collapsed"]]
            collapse_frac = 1.0 - len(ok) / n_calls
            tw_zero = sum(s["true_world_zero"] for s in stats) / n_calls
            if not ok:
                rows.append({
                    "guesser": guesser, "trick": trick, "n_calls": n_calls,
                    "n_scored": 0, "meanP": float("nan"),
                    "meanP_uncond": 0.0, "top1": float("nan"),
                    "truth_zero_frac": float("nan"),
                    "cw_frac": float("nan"),
                    "collapse_frac": collapse_frac,
                    "true_world_zero_frac": tw_zero,
                    "n_eff": float("nan"), "draws": float("nan"),
                })
                continue
            meanp = float(np.mean([s["meanP"] for s in ok]))
            n_cards = sum(s["n_cards"] for s in ok)
            rows.append({
                "guesser": guesser,
                "trick": trick,
                "n_calls": n_calls,
                "n_scored": len(ok),
                "meanP": meanp,
                "meanP_uncond": meanp * (1.0 - collapse_frac),
                "top1": float(np.mean([s["top1"] for s in ok])),
                "truth_zero_frac": sum(s["n_zero"] for s in ok) / n_cards,
                "cw_frac": sum(s["n_cw"] for s in ok) / n_cards,
                "collapse_frac": collapse_frac,
                "true_world_zero_frac": tw_zero,
                "n_eff": float(np.mean([s["n_effective"] for s in ok])),
                "draws": float(np.mean([s["draws_used"] for s in ok])),
            })
    return rows


HEADER_COLS = ("guesser trick n_calls n_scored meanP meanP_uncond top1 "
               "truth_zero_frac confidently_wrong_frac collapse_frac "
               "true_world_zero_frac mean_n_effective mean_draws_used\n")


def _row_line(r):
    return (f"{r['guesser']} {r['trick']} {r['n_calls']} {r['n_scored']} "
            f"{r['meanP']:.6f} {r['meanP_uncond']:.6f} {r['top1']:.6f} "
            f"{r['truth_zero_frac']:.6f} {r['cw_frac']:.6f} "
            f"{r['collapse_frac']:.6f} {r['true_world_zero_frac']:.6f} "
            f"{r['n_eff']:.3f} {r['draws']:.1f}\n")


def _write_partial(path, rows, done, total):
    with open(path, "w") as f:
        f.write(f"# PARTIAL: {done}/{total} chunks complete\n")
        f.write(HEADER_COLS)
        for r in rows:
            f.write(_row_line(r))


def write_table(path, rows, n_games, elapsed, workers, smoke, games_path):
    with open(path, "w") as f:
        f.write("# Task 8: robustness of choice filtering vs imperfect "
                "opponents\n")
        f.write(f"# games: {n_games} FRESH games, all 4 seats "
                f"RandomizedHeuristic(epsilon_true={EPSILON_TRUE}); "
                f"master_seed={MASTER_SEED}\n")
        f.write(f"#   deal i <- default_rng([{MASTER_SEED}, i]); seat s of "
                f"game i <- default_rng([{MASTER_SEED}, i, s]); records "
                f"written to {os.path.basename(games_path)} (v1 format)\n")
        f.write("# guessers, all built from the observer's PlayerView only:\n")
        f.write(f"#   FULL          BeliefTable.from_view(view, "
                f"{LEVEL.value}) -- constraint-only (voids + hand sizes)\n")
        f.write(f"#   CHOICE-strict WeightedPosterior.from_view(view, "
                f"{LEVEL.value}, HeuristicPlayer, epsilon={EPS_STRICT}, "
                f"n_worlds={N_WORLDS}, max_draws={MAX_DRAWS})\n")
        f.write(f"#   CHOICE-soft   same with epsilon={EPS_SOFT}\n")
        f.write("# BOTH choice guessers assume a plain deterministic "
                "HeuristicPlayer, which is DELIBERATELY the wrong model of "
                "these opponents. strict's epsilon=0 is the variable under "
                "test.\n")
        f.write(f"# COINCIDENCE, not a matched model: soft's epsilon="
                f"{EPS_SOFT} equals epsilon_true={EPSILON_TRUE} numerically, "
                f"but WeightedPosterior smooths uniformly over ALL legal "
                f"moves while RandomizedHeuristic spreads epsilon over the "
                f"num_legal-1 OTHER moves. Soft is a forgiving wrong model, "
                f"not a correct one.\n")
        f.write("# rng: default_rng([rec.seed, trick, observer]) -> seed "
                "(run_survival/run_guessing2 scheme); strict and soft get "
                "the SAME seed at each boundary-observer.\n")
        f.write(f"# trick k = belief state entering trick k, k={FIRST_TRICK}"
                f"..{NUM_TRICKS} (entering trick 1 there is no evidence, so "
                f"all three guessers coincide and it is skipped)\n")
        f.write("# observers: all 4 seats per boundary.\n")
        f.write("#\n# DENOMINATORS (never mixed):\n")
        f.write("#   n_calls  = boundary-observers at this trick "
                "(games * 4 observers)\n")
        f.write("#   n_scored = non-collapsed boundary-observers; meanP and "
                "top1 are the mean over cards within an observer, then over "
                "these observers\n")
        f.write("#   cards    = truth cards inside the n_scored observers; "
                "truth_zero_frac and confidently_wrong_frac are per-CARD "
                "over exactly that set\n")
        f.write("# collapse: from_view raised 'no candidate world survived' "
                "(caught BY MESSAGE; any other AssertionError from "
                "weighted.py is a structural bug and is re-raised). "
                "collapse_frac = collapsed / n_calls. Collapsed "
                "boundary-observers are EXCLUDED from every other metric, "
                "never imputed as zero.\n")
        f.write("#   FULL's collapse_frac is 0 BY CONSTRUCTION, not by "
                "measurement: a constraint table has no candidate worlds "
                "that could die. FULL is not wrapped in try/except at all.\n")
        f.write("# meanP is therefore CONDITIONAL on the guesser having "
                "answered. meanP_uncond = meanP * (1 - collapse_frac) scores "
                "a collapse as zero information and is the column that "
                "adjudicates 'degrades or hard-fails'. Wherever "
                "collapse_frac > 0, compare meanP_uncond, not meanP.\n")
        f.write("# truth zeros: a true card with marginal exactly 0.0. NOT "
                "floored/smoothed/excluded -- counted as 0.0 in meanP, as a "
                "top-1 miss, and as confidently wrong. For FULL a truth zero "
                "is impossible (voids and hand sizes are policy-independent "
                "facts) and is asserted against.\n")
        f.write(f"# confidently_wrong_frac: cards whose true holder got "
                f"marginal < {CONFIDENT}. Truth zeros are a subset.\n")
        f.write("# true_world_zero_frac: fraction of ALL n_calls "
                "boundary-observers at which the TRUE world itself audits to "
                "weight 0.0 under that guesser's epsilon (ground truth, "
                "diagnostic only, never fed to a guesser). For strict this "
                "is the mechanism behind collapse and confident wrongness: "
                "by then the opponents have deviated at least once. For soft "
                "it is 0 everywhere and is asserted, replacing "
                "run_guessing2's weight==1.0 invariant, which is FALSE here "
                "by design.\n")
        f.write("# mean_n_effective / mean_draws_used: sampling health over "
                "the n_scored observers (NaN/0 for FULL, which samples "
                "nothing). Soft's weights are products of many small factors, "
                "so 100 positive-weight worlds can still mean n_eff ~ 1-3; a "
                "late soft truth zero is usually 'never sampled', not "
                "'excluded by the model'.\n")
        f.write("#\n# PRE-REGISTERED EXPECTATIONS (PHASE23_PLAN.md Task 8, "
                "verbatim):\n")
        f.write("#   \"strict beats constraint-only early but degrades or "
                "hard-fails as deviations accumulate; soft epsilon=0.1 "
                "dominates strict against these opponents and never "
                "confidently excludes the truth. This experiment is the "
                "honesty bridge to future human play: it quantifies the cost "
                "of a wrong opponent model.\"\n")
        f.write(f"# wall_time_s={elapsed:.1f} workers={workers} "
                f"smoke={smoke}\n")
        f.write(HEADER_COLS)
        for r in rows:
            f.write(_row_line(r))


def print_table(rows):
    print(f"\n{'guesser':>14} {'trick':>5} {'n':>6} {'meanP':>8} "
          f"{'meanPu':>8} {'top1':>8} {'tzero':>8} {'cwrong':>8} "
          f"{'collapse':>9} {'twzero':>8} {'neff':>8} {'draws':>9}")
    for r in rows:
        print(f"{r['guesser']:>14} {r['trick']:>5} {r['n_scored']:>6} "
              f"{r['meanP']:>8.4f} {r['meanP_uncond']:>8.4f} "
              f"{r['top1']:>8.4f} {r['truth_zero_frac']:>8.4f} "
              f"{r['cw_frac']:>8.4f} {r['collapse_frac']:>9.4f} "
              f"{r['true_world_zero_frac']:>8.4f} {r['n_eff']:>8.2f} "
              f"{r['draws']:>9.1f}")


PANELS = [("meanP", "mean P(truth)  [conditional on answering]"),
          ("top1", "top-1 accuracy"),
          ("cw_frac", "fraction of cards with P(truth) < 0.01"),
          ("collapse_frac", "fraction of boundary-observers that collapsed")]


def plot(path, rows, n_games, smoke):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (key, label) in zip(axes.ravel(), PANELS):
        for g in GUESSERS:
            sel = [r for r in rows if r["guesser"] == g]
            ax.plot([r["trick"] for r in sel], [r[key] for r in sel],
                    lw=2, alpha=0.85, label=g, **STYLE[g])
        if key == "meanP":
            for g in ("CHOICE-strict", "CHOICE-soft"):
                sel = [r for r in rows if r["guesser"] == g]
                ax.plot([r["trick"] for r in sel],
                        [r["meanP_uncond"] for r in sel], ls="--", lw=1.2,
                        color=STYLE[g]["color"],
                        label=f"{g} (uncond.)")
        ax.set_xlabel("trick number (belief state entering trick)")
        ax.set_ylabel(label)
        ax.set_title(key)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"Cost of a wrong opponent model: guessing vs 4x "
        f"RandomizedHeuristic(eps_true={EPSILON_TRUE}), {n_games} games"
        + ("  [SMOKE]" if smoke else "")
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help=f"{SMOKE_GAMES} games, {SMOKE_WORKERS} workers, "
                         f"separate outputs")
    ap.add_argument("--games", type=int, default=None,
                    help="override the game count")
    args = ap.parse_args()

    if args.smoke:
        n = args.games or SMOKE_GAMES
        workers, chunk = SMOKE_WORKERS, SMOKE_CHUNK
        out_path, plot_path, partial = SMOKE_OUT_PATH, SMOKE_PLOT_PATH, None
        games_path = SMOKE_GAMES_PATH
    else:
        n = args.games or NUM_GAMES
        workers, chunk = WORKERS, CHUNK
        out_path, plot_path, partial = OUT_PATH, PLOT_PATH, PARTIAL_PATH
        games_path = GAMES_PATH
    # The true-world soft-weight assert costs one extra audit per
    # boundary-observer against the hundreds already done there (~0.01%), so
    # it is always on, as in run_guessing2.
    verify_soft = True

    os.makedirs(RESULTS, exist_ok=True)
    print(f"simulating {n} games (master_seed={MASTER_SEED}, "
          f"epsilon_true={EPSILON_TRUE})", flush=True)
    records = simulate_games(n)
    write_records(games_path, records)
    print(f"wrote {games_path}; evaluating with {workers} workers "
          f"(chunk={chunk})", flush=True)

    acc, elapsed = run_all(records, workers, chunk, verify_soft, partial)
    rows = summarize(acc)
    write_table(out_path, rows, n, elapsed, workers, args.smoke, games_path)
    plot(plot_path, rows, n, args.smoke)
    print_table(rows)
    print(f"\nwrote {out_path}, {plot_path} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
