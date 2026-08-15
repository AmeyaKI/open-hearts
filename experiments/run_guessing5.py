"""Phase 5 Task 4: guessing curves OFF HOME TURF (held-out personalities).

Phase 3's headline (results/guessing2.txt) measured choice-aware reading
against the opponents it assumes: 4x deterministic `HeuristicPlayer`. That is
home turf, and Phase 4's league showed the crown does not travel. This run
asks the same question against players the bot has NEVER seen -- HELD-OUT
personalities from Task 1's split, whose games never entered a training shard
(the held-out wall) -- and compares five likelihoods in the Phase-3 socket:

  FULL           constraint-only BeliefTable (exact, no sampling): the floor
                 that reads nothing about HOW opponents choose.
  CHOICE-strict  heuristic-match, eps=0.0: Phase 3's champion. Expected to
                 collapse -- a personality deviating even once kills the true
                 world for every observer at every later boundary.
  CHOICE-soft    heuristic-match, eps=0.1: the current best hedge, THE
                 BASELINE TO BEAT.
  PROFILER       the Task-3 GENERIC net's P(observed card | replayed view).
  PROFILER-ORACLE the CONDITIONED net + each seat's TRUE personality params.
                 DEPLOYMENT-IMPOSSIBLE, simulation-cheap: the reading ceiling
                 (PHASE5_PLAN redesign 2026-08-13).

GAMES AND SEEDS.  Games are played here, not read from a record file: seeds
960000+ (a FRESH block, distinct from Task 3's 950000+ held-out evaluation
games and from the 700000+ training range; it sits inside the Global-
Constraints 100000+ evaluation range). Each seed deterministically draws 4
DISTINCT HELD-OUT personalities via `default_rng([seed, TABLE_SALT])` -- the
same table-draw scheme as the generator, pointed at the held-out pool. No
anchors: anchors are TRAIN-side by Task 1's contract, and a held-out claim
must not lean on them.

METRICS (house convention, unchanged from run_guessing2 except one addition):
mean P(truth) and top-1 over ALL truth cards (a zero counts as 0.0 / a miss);
NLL over p>0 cards ONLY with `truth_zero_frac` reported separately;
`collapse_frac` for boundary-observers whose posterior raised
`PosteriorCollapse` (EXCLUDED from means, never imputed). ADDED here:
`truth_lt01_frac`, the fraction of truth cards with p < 0.01 -- the
pre-registered "never confidently excludes the truth" number, which
`truth_zero_frac` alone does not answer.

THREE READING WARNINGS, stated up front because they decide how the table is
read:
 1. CHOICE-strict's surviving means are SURVIVOR-BIASED. Its true world dies
    the moment any opponent deviates from the script, so the boundaries that
    survive are exactly the ones where the personalities happened to play
    heuristically. Judge CHOICE-strict by `collapse_frac` and
    `truth_lt01_frac`, NOT by meanP.
 2. The draw budget FAVOURS the collapsing curves. PROFILER and CHOICE-soft
    accept nearly every drawn world, so they spend exactly `n_worlds` draws;
    CHOICE-strict may spend up to `max_draws` hunting for 100 survivors. Any
    finding against CHOICE-strict is therefore conservative.
 3. `n_effective` is a load-bearing caveat for PROFILER, not decoration: its
    weights are products of ~30 smooth factors, so a late-trick posterior can
    concentrate on a handful of worlds. Check the column before believing a
    late-trick delta.

Usage:
  .venv/bin/python experiments/run_guessing5.py --smoke      # 20 games
  .venv/bin/python experiments/run_guessing5.py --full       # 500 games
"""
import argparse
import concurrent.futures as cf
import os
import subprocess
import sys
import time
from collections import namedtuple

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from openhearts.belief.table import BeliefTable, Level  # noqa: E402
from openhearts.belief.weighted import (  # noqa: E402
    PosteriorCollapse, WeightedPosterior)
from openhearts.engine.game import deal  # noqa: E402
from openhearts.engine.state import GameState  # noqa: E402
from openhearts.eval import guessing  # noqa: E402
from openhearts.opponent.params import param_vector  # noqa: E402
from openhearts.players.heuristic import HeuristicPlayer  # noqa: E402
from openhearts.players.personality import (  # noqa: E402
    PersonalityPlayer, make_population, sample_personality)
from openhearts.search.profiled import (  # noqa: E402
    load_profiler_likelihood, profiler_posterior)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
OUT_PATH = os.path.join(RESULTS, "guessing5.txt")
PLOT_PATH = os.path.join(RESULTS, "guessing5.png")
PARTIAL_PATH = os.path.join(RESULTS, "guessing5_partial.txt")
SMOKE_OUT_PATH = os.path.join(RESULTS, "guessing5_smoke.txt")
SMOKE_PLOT_PATH = os.path.join(RESULTS, "guessing5_smoke.png")

GENERIC_NPZ = os.path.join(ROOT, "models", "profiler_v1.npz")
# The CONDITIONED net is simulation-only (ORACLE row) and therefore lives with
# the training outputs, not in models/ -- it is regenerable, not shipped.
CONDITIONED_NPZ = os.path.join(RESULTS, "profiler_train",
                               "profiler_conditioned.npz")

SEED_BASE = 960000            # FRESH block; see module docstring
TABLE_SALT = 777              # same salt as the generator's table draw
MASTER_SEED = 314159          # Task 1 population split (frozen)
N_TRAIN, N_HELDOUT = 200, 50
NUM_GAMES = 500
SMOKE_GAMES = 20
NUM_TRICKS = guessing.NUM_TRICKS
LEVEL = Level.FULL
N_WORLDS = 100
MAX_DRAWS = 50000             # Phase-3 conventions
EPS_SOFT = 0.1
WORKERS = 8
CHUNK = 5
SMOKE_WORKERS = 4

CURVES = ("FULL", "CHOICE-strict", "CHOICE-soft", "PROFILER",
          "PROFILER-ORACLE")

PRE_REGISTERED = [
    "PROFILER beats FULL at every trick (meanP)",
    "PROFILER beats CHOICE-soft overall (mean over tricks, meanP)",
    "CHOICE-strict collapses (high collapse_frac off home turf)",
    "PROFILER never confidently excludes truth "
    "(fraction of truth-P<0.01 cards ~ 0)",
]

_TRAIN_IDS, _HELDOUT_IDS = make_population(N_TRAIN, N_HELDOUT, MASTER_SEED)
_HELDOUT_IDS = list(_HELDOUT_IDS)

Rec = namedtuple("Rec", "seed hands plays pids")

_LIK = {}


def _likelihoods():
    """Per-process lazy load of the two nets (workers are separate processes).

    ORACLE's `seat_params` is per-GAME (it depends on who is at the table), so
    only the WEIGHTS are cached here; the likelihood object is rebuilt per
    game in `process_game`.
    """
    if not _LIK:
        _LIK["generic"] = load_profiler_likelihood(GENERIC_NPZ)
        from openhearts.opponent.infer import load_profiler
        _LIK["cond_w"] = load_profiler(CONDITIONED_NPZ)
    return _LIK


def table_for_seed(seed):
    """4 DISTINCT held-out personality ids; sampled order IS the seat order."""
    rng = np.random.default_rng([seed, TABLE_SALT])
    idx = rng.choice(len(_HELDOUT_IDS), size=4, replace=False)
    return [int(_HELDOUT_IDS[i]) for i in idx]


def play_game(seed):
    """Play one held-out-personality game -> Rec (original hands + plays)."""
    pids = table_for_seed(seed)
    players = [PersonalityPlayer(np.random.default_rng([seed, s, 0xA1CE]),
                                 sample_personality(p))
               for s, p in enumerate(pids)]
    state = deal(np.random.default_rng(seed))
    hands = list(state.hands)
    plays = []
    while not state.is_over():
        seat = state.to_play
        card = players[seat].choose(state.view_for(seat))
        plays.append((seat, card))
        state.play(card)
    assert sum(state.scores) == 26, "scores must sum to 26"
    return Rec(seed, hands, plays, pids)


def _seed_for(seed, trick, observer):
    """Deterministic rng seed; same scheme as run_guessing2/run_survival."""
    return int(np.random.default_rng([seed, trick, observer]).integers(
        0, 2 ** 63 - 1))


def choice_metrics(probs, truth):
    """Score a (3,52) posterior against truth. Nothing is floored.

    Returns (meanP, nll_sum, n_positive, top1, n_cards, n_zero, n_lt01).
    """
    assert truth, "no cards to score"
    ps, hits = [], []
    for c, i in truth.items():
        p = float(probs[i, c])
        assert np.isfinite(p) and p >= 0.0, f"bad probability {p}"
        ps.append(p)
        hits.append(1.0 if int(np.argmax(probs[:, c])) == i else 0.0)
    ps = np.asarray(ps, dtype=float)
    pos = ps[ps > 0.0]
    return (float(ps.mean()),
            float((-np.log(pos)).sum()) if pos.size else 0.0,
            int(pos.size), float(np.mean(hits)), int(ps.size),
            int(ps.size - pos.size), int((ps < 0.01).sum()))


def _posterior(curve, view, rng, lik_generic, lik_oracle, policy):
    if curve == "FULL":
        t = BeliefTable.from_view(view, LEVEL)
        return t, 0.0, 0
    if curve == "CHOICE-strict":
        return (WeightedPosterior.from_view(
            view, LEVEL, policy, epsilon=0.0, n_worlds=N_WORLDS, rng=rng,
            max_draws=MAX_DRAWS), None, None)
    if curve == "CHOICE-soft":
        return (WeightedPosterior.from_view(
            view, LEVEL, policy, epsilon=EPS_SOFT, n_worlds=N_WORLDS, rng=rng,
            max_draws=MAX_DRAWS), None, None)
    lik = lik_generic if curve == "PROFILER" else lik_oracle
    return (profiler_posterior(view, LEVEL, lik, N_WORLDS, rng, MAX_DRAWS),
            None, None)


def _boundary(state, rec, trick, lik_generic, lik_oracle, policy, timings):
    """All 4 observers x all 5 curves at one completed-trick boundary."""
    out = {c: [] for c in CURVES}
    for seat in range(4):
        view = state.view_for(seat)
        for curve in CURVES:
            rng = np.random.default_rng(_seed_for(rec.seed, trick, seat))
            t0 = time.time()
            try:
                post, _a, _b = _posterior(curve, view, rng, lik_generic,
                                          lik_oracle, policy)
            except PosteriorCollapse:
                out[curve].append({"collapsed": True})
                timings[curve].append(time.time() - t0)
                continue
            timings[curve].append(time.time() - t0)
            truth = guessing._truth_for(seat, rec, post)
            mp, nll_sum, n_pos, top1, n_cards, n_zero, n_lt01 = \
                choice_metrics(post.probs, truth)
            out[curve].append({
                "collapsed": False, "meanP": mp, "nll_sum": nll_sum,
                "n_pos": n_pos, "top1": top1, "n_cards": n_cards,
                "n_zero": n_zero, "n_lt01": n_lt01,
                "n_effective": float(getattr(post, "n_effective", 0.0)),
                "draws_used": int(getattr(post, "draws_used", 0)),
            })
    return out


def process_game(seed):
    """Play + score one game: {curve: {trick: [per-observer dicts]}}."""
    rec = play_game(seed)
    liks = _likelihoods()
    lik_generic = liks["generic"]
    from openhearts.search.profiled import ProfilerLikelihood
    seat_params = {s: param_vector(p) for s, p in enumerate(rec.pids)}
    lik_oracle = ProfilerLikelihood(liks["cond_w"][0], seat_params,
                                    liks["cond_w"][1])
    policy = HeuristicPlayer()
    timings = {c: [] for c in CURVES}
    acc = {c: {t: [] for t in range(1, NUM_TRICKS + 1)} for c in CURVES}

    state = GameState(hands=list(rec.hands))
    state.to_play = rec.plays[0][0]
    res = _boundary(state, rec, 1, lik_generic, lik_oracle, policy, timings)
    for c in CURVES:
        acc[c][1].extend(res[c])
    for seat, card in rec.plays:
        assert seat == state.to_play, "replay desync"
        state.play(card)
        if not state.current_trick and state.trick_number < NUM_TRICKS:
            trick = state.trick_number + 1
            res = _boundary(state, rec, trick, lik_generic, lik_oracle,
                            policy, timings)
            for c in CURVES:
                acc[c][trick].extend(res[c])
    assert state.is_over()
    return acc, timings


def worker(seeds):
    acc = {c: {t: [] for t in range(1, NUM_TRICKS + 1)} for c in CURVES}
    timings = {c: [] for c in CURVES}
    for s in seeds:
        a, tm = process_game(s)
        for c in CURVES:
            for t in acc[c]:
                acc[c][t].extend(a[c][t])
            timings[c].extend(tm[c])
    return acc, timings


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def total_rss_gb():
    out = subprocess.run(["ps", "-o", "rss=", "-g", str(os.getpgrp())],
                         capture_output=True, text=True).stdout
    return sum(int(x) for x in out.split()) / (1024 ** 2)


def run_all(seeds, workers, chunk_size, partial_path):
    acc = {c: {t: [] for t in range(1, NUM_TRICKS + 1)} for c in CURVES}
    timings = {c: [] for c in CURVES}
    chunks = list(_chunks(seeds, chunk_size))
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(worker, c) for c in chunks]
        done = 0
        for fut in cf.as_completed(jobs):
            a, tm = fut.result()
            for c in CURVES:
                for t in acc[c]:
                    acc[c][t].extend(a[c][t])
                timings[c].extend(tm[c])
            done += 1
            print(f"[{done}/{len(jobs)}] chunks | mem={total_rss_gb():.1f}GB "
                  f"| {time.time() - t0:.0f}s", flush=True)
            if partial_path:
                _write_partial(partial_path, summarize(acc), done, len(jobs))
    return acc, timings, time.time() - t0


def summarize(acc):
    rows = []
    for curve in CURVES:
        for trick in range(1, NUM_TRICKS + 1):
            stats = acc[curve][trick]
            if not stats:
                continue
            ok = [s for s in stats if not s["collapsed"]]
            collapse_frac = 1.0 - len(ok) / len(stats)
            if not ok:
                rows.append({"curve": curve, "trick": trick,
                             "n_calls": len(stats), "n_scored": 0,
                             "meanP": float("nan"), "NLL": float("nan"),
                             "n_nll": 0, "top1": float("nan"),
                             "truth_zero_frac": float("nan"),
                             "truth_lt01_frac": float("nan"),
                             "collapse_frac": collapse_frac,
                             "mean_n_effective": float("nan"),
                             "mean_draws_used": float("nan")})
                continue
            with_pos = [s for s in ok if s["n_pos"] > 0]
            n_cards = sum(s["n_cards"] for s in ok)
            rows.append({
                "curve": curve, "trick": trick, "n_calls": len(stats),
                "n_scored": len(ok),
                "meanP": float(np.mean([s["meanP"] for s in ok])),
                "NLL": (float(np.mean([s["nll_sum"] / s["n_pos"]
                                       for s in with_pos]))
                        if with_pos else float("nan")),
                "n_nll": len(with_pos),
                "top1": float(np.mean([s["top1"] for s in ok])),
                "truth_zero_frac": sum(s["n_zero"] for s in ok) / n_cards,
                "truth_lt01_frac": sum(s["n_lt01"] for s in ok) / n_cards,
                "collapse_frac": collapse_frac,
                "mean_n_effective": float(np.mean([s["n_effective"]
                                                   for s in ok])),
                "mean_draws_used": float(np.mean([s["draws_used"]
                                                  for s in ok])),
            })
    return rows


HEADER_COLS = ("curve trick n_calls n_scored meanP NLL n_nll_obs top1 "
               "truth_zero_frac truth_lt01_frac collapse_frac "
               "mean_n_effective mean_draws_used\n")


def _row_line(r):
    return (f"{r['curve']} {r['trick']} {r['n_calls']} {r['n_scored']} "
            f"{r['meanP']:.6f} {r['NLL']:.6f} {r['n_nll']} {r['top1']:.6f} "
            f"{r['truth_zero_frac']:.6f} {r['truth_lt01_frac']:.6f} "
            f"{r['collapse_frac']:.6f} {r['mean_n_effective']:.3f} "
            f"{r['mean_draws_used']:.3f}\n")


def _write_partial(path, rows, done, total):
    with open(path, "w") as f:
        f.write(f"# PARTIAL: {done}/{total} chunks complete\n")
        f.write(HEADER_COLS)
        for r in rows:
            f.write(_row_line(r))


def _by_curve(rows):
    return {c: {r["trick"]: r for r in rows if r["curve"] == c}
            for c in CURVES}


def verdicts(rows):
    """Pre-registered expectations, answered verbatim. -> [(text, ok, note)]"""
    bc = _by_curve(rows)
    out = []

    beats = [(t, bc["PROFILER"][t]["meanP"] - bc["FULL"][t]["meanP"])
             for t in sorted(bc["PROFILER"])]
    losses = [(t, d) for t, d in beats if not (d > 0)]
    out.append((PRE_REGISTERED[0], not losses,
                "per-trick delta " +
                " ".join(f"t{t}:{d:+.4f}" for t, d in beats) +
                ("; LOSING: " + ", ".join(f"t{t}({d:+.4f})"
                                          for t, d in losses)
                 if losses else "") +
                "  [trick 1 has NO observed opponent plies: every candidate "
                "world has weight 1 there, so PROFILER-vs-FULL at trick 1 is "
                "100-world sampling noise against exact marginals, not "
                "reading]"))

    pm = float(np.mean([r["meanP"] for r in rows
                        if r["curve"] == "PROFILER" and r["n_scored"]]))
    cm = float(np.mean([r["meanP"] for r in rows
                        if r["curve"] == "CHOICE-soft" and r["n_scored"]]))
    out.append((PRE_REGISTERED[1], pm > cm,
                f"PROFILER {pm:.4f} vs CHOICE-soft {cm:.4f} "
                f"(delta {pm - cm:+.4f})"))

    cs = [bc["CHOICE-strict"][t]["collapse_frac"]
          for t in sorted(bc["CHOICE-strict"])]
    out.append((PRE_REGISTERED[2], max(cs) > 0.5,
                f"collapse_frac max {max(cs):.3f}, mean {np.mean(cs):.3f}, "
                f"trick 13 {cs[-1]:.3f} (judge this curve by collapse, NOT "
                f"by its survivor-biased meanP)"))

    lt = [bc["PROFILER"][t]["truth_lt01_frac"] for t in sorted(bc["PROFILER"])]
    out.append((PRE_REGISTERED[3], max(lt) < 0.05,
                f"truth-P<0.01 fraction: max {max(lt):.4f} over tricks, "
                f"mean {np.mean(lt):.4f} (threshold for '~0' read as <0.05)"))
    return out


def gaps(rows):
    """ORACLE headroom + reading-value ceiling, the Task-6 context numbers."""
    m = {c: float(np.mean([r["meanP"] for r in rows
                           if r["curve"] == c and r["n_scored"]]))
         for c in CURVES}
    return m, (m["PROFILER-ORACLE"] - m["PROFILER"],
               m["PROFILER-ORACLE"] - m["CHOICE-soft"])


def write_table(path, rows, n_games, elapsed, workers, smoke, timings,
                verdict_rows, means, gap):
    with open(path, "w") as f:
        f.write("# Phase 5 Task 4: guessing curves vs HELD-OUT personalities\n")
        f.write(f"# games={n_games} seeds={SEED_BASE}..{SEED_BASE+n_games-1} "
                f"(FRESH block; Task 3 eval used 950000+)\n")
        f.write(f"# population: make_population({N_TRAIN}, {N_HELDOUT}, "
                f"{MASTER_SEED}); tables = 4 DISTINCT HELD-OUT ids per seed "
                f"via default_rng([seed, {TABLE_SALT}]); no anchors\n")
        f.write(f"# posteriors: level={LEVEL.value} n_worlds={N_WORLDS} "
                f"max_draws={MAX_DRAWS}; CHOICE eps: strict=0.0 "
                f"soft={EPS_SOFT}\n")
        f.write(f"# models: GENERIC={os.path.basename(GENERIC_NPZ)} "
                f"CONDITIONED={os.path.basename(CONDITIONED_NPZ)} "
                f"(ORACLE = CONDITIONED + TRUE per-seat params: "
                f"DEPLOYMENT-IMPOSSIBLE, ceiling only)\n")
        f.write("# rng: default_rng([seed, trick, observer]) per curve -- "
                "same seed derivation and the same FULL-level proposal "
                "DISTRIBUTION for every curve, but the realized draw streams "
                "differ: the fused CHOICE audit and the profiler audit "
                "consume the generator differently (and CHOICE-strict "
                "re-chunks hunting survivors).\n")
        f.write("# trick 1 has NO observed opponent plies: every candidate "
                "world has weight 1, so the three sampled curves are the "
                "FULL-level SAMPLER anchor there while the FULL column is "
                "the EXACT table. Trick-1 differences are estimator noise, "
                "not reading.\n")
        f.write("# FULL is the exact constraint-only BeliefTable (no "
                "sampling; n_effective/draws_used are 0 by construction)\n")
        f.write("# metrics: meanP/top1 over ALL truth cards (zeros count as "
                "0.0 / a miss); NLL over p>0 cards ONLY; truth_zero_frac and "
                "truth_lt01_frac (p<0.01) reported separately\n")
        f.write("# collapse: PosteriorCollapse raised -> that boundary-"
                "observer is EXCLUDED from means (never imputed as 0)\n")
        f.write("# WARNING 1: CHOICE-strict's means are SURVIVOR-BIASED -- "
                "judge it by collapse_frac and truth_lt01_frac.\n")
        f.write("# WARNING 2: the draw budget FAVOURS CHOICE-strict (it may "
                "spend max_draws hunting survivors); findings against it are "
                "conservative.\n")
        f.write("# WARNING 3: PROFILER weights are log-space products of ~30 "
                "smooth factors; read mean_n_effective before believing a "
                "late-trick delta. PROFILER's total_weight is relative (see "
                "search/profiled.py contract 2) and is not reported.\n")
        f.write(f"# wall_time_s={elapsed:.1f} workers={workers} "
                f"smoke={smoke}\n")
        for c in CURVES:
            if timings[c]:
                f.write(f"# cost {c}: mean {1000*np.mean(timings[c]):.2f} ms/"
                        f"posterior over {len(timings[c])} calls\n")
        f.write("#\n# PRE-REGISTERED EXPECTATIONS (verbatim):\n")
        for text, ok, note in verdict_rows:
            f.write(f"#   [{'PASS' if ok else 'FAIL'}] {text}\n"
                    f"#          {note}\n")
        f.write("#\n# overall meanP by curve: " +
                "  ".join(f"{c}={means[c]:.4f}" for c in CURVES) + "\n")
        f.write(f"# ORACLE - PROFILER (reading-identity headroom): "
                f"{gap[0]:+.4f} meanP\n")
        f.write(f"# ORACLE - CHOICE-soft (reading-value ceiling, Task-6 "
                f"context): {gap[1]:+.4f} meanP\n")
        f.write(HEADER_COLS)
        for r in rows:
            f.write(_row_line(r))


def print_table(rows):
    print(f"\n{'curve':>16} {'trick':>5} {'n':>6} {'meanP':>8} {'NLL':>8} "
          f"{'top1':>7} {'tzero':>7} {'t<.01':>7} {'coll':>7} {'neff':>7}")
    for r in rows:
        print(f"{r['curve']:>16} {r['trick']:>5} {r['n_scored']:>6} "
              f"{r['meanP']:>8.4f} {r['NLL']:>8.4f} {r['top1']:>7.4f} "
              f"{r['truth_zero_frac']:>7.4f} {r['truth_lt01_frac']:>7.4f} "
              f"{r['collapse_frac']:>7.4f} {r['mean_n_effective']:>7.2f}")


METRICS = [("meanP", "mean P(truth)"), ("NLL", "mean -ln P(truth)"),
           ("top1", "top-1 accuracy")]


def plot(path, rows, n_games, smoke):
    bc = _by_curve(rows)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for j, (short, label) in enumerate(METRICS):
        ax = axes[j]
        for c in CURVES:
            ts = sorted(bc[c])
            ys = [bc[c][t][short] for t in ts]
            ax.plot(ts, ys, marker="o", lw=1.8, label=c)
        ax.set_xlabel("trick number (belief state entering trick)")
        ax.set_ylabel(label)
        ax.set_title(short + (" (p>0 cards only)" if short == "NLL" else ""))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Reading held-out personalities: {n_games} games"
                 + ("  [SMOKE]" if smoke else ""))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--games", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    if not (args.smoke or args.full or args.games):
        ap.error("pass --smoke or --full (or --games N)")

    # Anything that is not an explicit --full run writes to the SMOKE
    # outputs, so a partial/manual --games run can never overwrite the
    # headline table.
    smoke = not args.full
    n_games = args.games or (SMOKE_GAMES if smoke else NUM_GAMES)
    workers = args.workers or (SMOKE_WORKERS if smoke else WORKERS)
    chunk = max(1, min(CHUNK, n_games // max(1, workers)))
    seeds = [SEED_BASE + i for i in range(n_games)]
    out_path = SMOKE_OUT_PATH if smoke else OUT_PATH
    plot_path = SMOKE_PLOT_PATH if smoke else PLOT_PATH
    partial = None if smoke else PARTIAL_PATH

    for p in (GENERIC_NPZ, CONDITIONED_NPZ):
        assert os.path.exists(p), f"missing model: {p}"
    os.makedirs(RESULTS, exist_ok=True)
    print(f"{n_games} held-out-personality games, {workers} workers, "
          f"chunk={chunk}, curves={list(CURVES)}", flush=True)

    acc, timings, elapsed = run_all(seeds, workers, chunk, partial)
    rows = summarize(acc)
    vr = verdicts(rows)
    means, gap = gaps(rows)
    write_table(out_path, rows, n_games, elapsed, workers, smoke, timings,
                vr, means, gap)
    plot(plot_path, rows, n_games, smoke)
    print_table(rows)

    print("\ncost per posterior (ms):")
    per_game = 0.0
    for c in CURVES:
        if timings[c]:
            ms = 1000 * float(np.mean(timings[c]))
            per_game += ms * len(timings[c]) / n_games
            print(f"  {c:>16}: {ms:8.2f}  over {len(timings[c])} calls")
    proj = per_game * NUM_GAMES / 1000.0
    print(f"\nprojection for a {NUM_GAMES}-game run: "
          f"{proj:.0f}s serial = {proj/3600:.2f}h; "
          f"/{WORKERS} workers = {proj/WORKERS/3600:.2f}h")

    print("\nPRE-REGISTERED EXPECTATIONS (verbatim):")
    for text, ok, note in vr:
        print(f"  [{'PASS' if ok else 'FAIL'}] {text}\n         {note}")
    print("\noverall meanP: " + "  ".join(f"{c}={means[c]:.4f}"
                                          for c in CURVES))
    print(f"ORACLE - PROFILER   (reading-identity headroom): {gap[0]:+.4f}")
    print(f"ORACLE - CHOICE-soft (reading-value ceiling)   : {gap[1]:+.4f}")
    print(f"\nwrote {out_path}, {plot_path} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
