"""Headline result 3: choice-aware guessing curves (Task 7).

Phase 1's belief table extracts everything the CONSTRAINTS say and tops out at
mean P(truth) = 0.5877 entering trick 13 -- the "constraint-evidence ceiling".
This experiment adds the evidence the table cannot see: HOW the opponents
chose. For every completed-trick boundary of a recorded heuristic game and
each of the 4 observer seats, it builds a `WeightedPosterior` (epsilon = 0,
n_worlds = 100, FULL-level proposal, plain `HeuristicPlayer` as the assumed
opponent policy -- which is exactly what generated the records) and scores its
marginals against the true deal.

The three Phase-1 curves (UNIFORM / VOIDS / FULL) are NOT recomputed; they are
re-read from results/guessing.txt and re-plotted alongside the CHOICE curve.

Honesty rules implemented here (all plan-specified):

* Truth zeros are NOT floored. With finitely many sampled worlds a true card
  can end up with sampled marginal exactly 0.0 without that being a bug. Such
  cards are counted (`truth_zero_frac`) and still count in meanP (as 0.0) and
  top-1 (as a miss); they are EXCLUDED from NLL, which is therefore reported
  over p > 0 cards only. `eval.guessing.metrics_for` is deliberately not used
  and not modified: its `assert p > 0` is correct for exact tables.
* Collapse (`from_view` raising "no candidate world survived") is counted per
  trick (`collapse_frac`) and that boundary-observer is EXCLUDED from the
  metric means -- not imputed as zero.
* Sampling health (`n_effective`, `draws_used`) is reported per trick.

Truth-world invariant: the records are 4x deterministic `HeuristicPlayer`, so
at epsilon = 0 the TRUE world must have weight exactly 1.0 at every boundary.
`--smoke` asserts this at every boundary-observer; it is the sharpest available
check against a replay/seat bug silently faking the whole result.

Deterministic rng: seeded from (rec.seed, trick, observer) with exactly the
scheme used by run_survival.py, so survival statistics are cross-checkable.
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

from openhearts.belief.table import Level  # noqa: E402
from openhearts.belief.weighted import (  # noqa: E402
    WeightedPosterior, world_weight,
)
from openhearts.engine import cards  # noqa: E402
from openhearts.engine.state import GameState  # noqa: E402
from openhearts.eval import guessing  # noqa: E402
from openhearts.eval.records import read_records  # noqa: E402
from openhearts.players.heuristic import HeuristicPlayer  # noqa: E402

RESULTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
RECORDS_PATH = os.path.join(RESULTS, "heuristic_games.txt")
PHASE1_TABLE = os.path.join(RESULTS, "guessing.txt")
OUT_PATH = os.path.join(RESULTS, "guessing2.txt")
PLOT_PATH = os.path.join(RESULTS, "guessing2.png")
PARTIAL_PATH = os.path.join(RESULTS, "guessing2_partial.txt")
SMOKE_OUT_PATH = os.path.join(RESULTS, "guessing2_smoke.txt")
SMOKE_PLOT_PATH = os.path.join(RESULTS, "guessing2_smoke.png")

NUM_RECORDS = 500  # pre-approved reduction from 2000 (plan, Task 7)
NUM_TRICKS = guessing.NUM_TRICKS
LEVEL = Level.FULL
EPSILON = 0.0
N_WORLDS = 100
MAX_DRAWS = 50000
MASTER_SEED = 2026

WORKERS = 12
CHUNK = 25

SMOKE_RECORDS = 8
SMOKE_WORKERS = 2

CEILING = 0.587699  # Phase-1 FULL meanP entering trick 13


def _seed_for(rec_idx, trick, observer):
    """Deterministic rng seed -- identical scheme to run_survival.py."""
    return int(
        np.random.default_rng([rec_idx, trick, observer]).integers(
            0, 2**63 - 1
        )
    )


def choice_metrics(probs, truth):
    """Score a (3, 52) posterior against truth without flooring anything.

    Returns (meanP, nll_sum, n_positive, top1, n_cards, n_zero):
      * meanP / top1 average over ALL truth cards (a zero counts as P = 0 and
        as a top-1 miss),
      * nll_sum / n_positive describe ONLY the p > 0 cards; the caller forms
        the observer's NLL as nll_sum / n_positive when n_positive > 0.
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
    pos = ps[ps > 0.0]
    return (
        float(ps.mean()),
        float((-np.log(pos)).sum()) if pos.size else 0.0,
        int(pos.size),
        float(np.mean(hits)),
        int(ps.size),
        int(ps.size - pos.size),
    )


def _true_world_hands(state, opponent_seats):
    """The opponents' CURRENT true hands, in opponent_seats order.

    Ground truth, read from the replayed state; used only by the smoke
    invariant check, never by the posterior.
    """
    return [int(state.hands[s]) for s in opponent_seats]


def _boundary(state, rec, trick, policy, verify_truth):
    """Run all 4 observers at one completed-trick boundary."""
    out = []
    for seat in range(4):
        rng = np.random.default_rng(_seed_for(rec.seed, trick, seat))
        view = state.view_for(seat)
        try:
            post = WeightedPosterior.from_view(
                view, LEVEL, policy, epsilon=EPSILON, n_worlds=N_WORLDS,
                rng=rng, max_draws=MAX_DRAWS,
            )
        except AssertionError:
            out.append({"collapsed": True})
            continue

        if verify_truth:
            w = world_weight(
                view, _true_world_hands(state, post.opponent_seats),
                policy, 0.0,
            )
            assert w == 1.0, (
                f"TRUE world has weight {w} != 1.0 at record {rec.seed} "
                f"trick {trick} observer {seat}: the replay or the seat "
                f"mapping is wrong and every curve below is invalid"
            )

        truth = guessing._truth_for(seat, rec, post)
        meanp, nll_sum, n_pos, top1, n_cards, n_zero = choice_metrics(
            post.probs, truth
        )
        out.append({
            "collapsed": False,
            "meanP": meanp,
            "nll_sum": nll_sum,
            "n_pos": n_pos,
            "top1": top1,
            "n_cards": n_cards,
            "n_zero": n_zero,
            "n_effective": post.n_effective,
            "draws_used": post.draws_used,
        })
    return out


def process_record(rec, verify_truth=False):
    """Replay one record; per-trick lists of per-observer result dicts."""
    results = {t: [] for t in range(1, NUM_TRICKS + 1)}
    policy = HeuristicPlayer()

    state = GameState(hands=list(rec.hands))
    state.to_play = next(
        s for s in range(4) if rec.hands[s] & cards.bit(cards.TWO_CLUBS)
    )
    assert state.to_play == rec.plays[0][0], "record leader is not 2c holder"

    results[1].extend(_boundary(state, rec, 1, policy, verify_truth))
    for seat, card in rec.plays:
        assert seat == state.to_play, (
            f"replay desync: recorded seat {seat} != {state.to_play}"
        )
        state.play(card)
        if not state.current_trick and state.trick_number < NUM_TRICKS:
            trick = state.trick_number + 1  # entering this trick
            results[trick].extend(
                _boundary(state, rec, trick, policy, verify_truth)
            )
    assert state.is_over(), "record did not replay to a finished game"
    return results


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def worker(args):
    chunk_records, verify_truth = args
    acc = {t: [] for t in range(1, NUM_TRICKS + 1)}
    for rec in chunk_records:
        per_rec = process_record(rec, verify_truth)
        for t in acc:
            acc[t].extend(per_rec[t])
    return acc


def total_rss_gb():
    out = subprocess.run(["ps", "-o", "rss=", "-g", str(os.getpgrp())],
                         capture_output=True, text=True).stdout
    return sum(int(x) for x in out.split()) / (1024 ** 2)


def run_all(records, workers, chunk_size, verify_truth, partial_path):
    acc = {t: [] for t in range(1, NUM_TRICKS + 1)}
    chunks = list(_chunks(records, chunk_size))
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(worker, (c, verify_truth)) for c in chunks]
        done = 0
        for fut in cf.as_completed(jobs):
            chunk_acc = fut.result()
            for t in acc:
                acc[t].extend(chunk_acc[t])
            done += 1
            print(f"[{done}/{len(jobs)}] chunks done | "
                  f"mem={total_rss_gb():.1f}GB | "
                  f"{time.time() - t0:.0f}s elapsed", flush=True)
            if partial_path:
                _write_partial(partial_path, summarize(acc), done, len(jobs))
    return acc, time.time() - t0


def summarize(acc):
    rows = []
    for trick in range(1, NUM_TRICKS + 1):
        stats = acc[trick]
        if not stats:
            continue
        n_calls = len(stats)
        ok = [s for s in stats if not s["collapsed"]]
        collapse_frac = 1.0 - len(ok) / n_calls
        assert ok, f"every call collapsed at trick {trick}"
        # per-observer means, then mean over boundary-observers (Phase-1
        # nesting: mean over cards inside an observer, then over observers)
        meanp = float(np.mean([s["meanP"] for s in ok]))
        top1 = float(np.mean([s["top1"] for s in ok]))
        with_pos = [s for s in ok if s["n_pos"] > 0]
        nll = (float(np.mean([s["nll_sum"] / s["n_pos"] for s in with_pos]))
               if with_pos else float("nan"))
        n_cards = sum(s["n_cards"] for s in ok)
        n_zero = sum(s["n_zero"] for s in ok)
        rows.append({
            "trick": trick,
            "n_calls": n_calls,
            "n_scored": len(ok),
            "meanP": meanp,
            "NLL": nll,
            "n_nll": len(with_pos),
            "top1": top1,
            "truth_zero_frac": n_zero / n_cards,
            "collapse_frac": collapse_frac,
            "mean_n_effective": float(np.mean([s["n_effective"]
                                               for s in ok])),
            "mean_draws_used": float(np.mean([s["draws_used"] for s in ok])),
        })
    return rows


HEADER_COLS = ("trick n_calls n_scored meanP NLL n_nll_obs top1 "
               "truth_zero_frac collapse_frac mean_n_effective "
               "mean_draws_used\n")


def _row_line(r):
    return (f"{r['trick']} {r['n_calls']} {r['n_scored']} "
            f"{r['meanP']:.6f} {r['NLL']:.6f} {r['n_nll']} "
            f"{r['top1']:.6f} {r['truth_zero_frac']:.6f} "
            f"{r['collapse_frac']:.6f} {r['mean_n_effective']:.3f} "
            f"{r['mean_draws_used']:.3f}\n")


def _write_partial(path, rows, done, total):
    with open(path, "w") as f:
        f.write(f"# PARTIAL: {done}/{total} chunks complete\n")
        f.write(HEADER_COLS)
        for r in rows:
            f.write(_row_line(r))


def write_table(path, rows, num_records, elapsed, workers, smoke):
    with open(path, "w") as f:
        f.write("# choice-aware guessing curves (Task 7, headline 3)\n")
        f.write(f"# CHOICE curve: first {num_records} records of "
                f"{os.path.basename(RECORDS_PATH)} "
                f"(master_seed={MASTER_SEED}).\n")
        f.write("#   500 of the 2000 records is the plan's pre-approved "
                "reduction: the CHOICE curve is far more expensive than the "
                "table-based curves.\n")
        f.write("#   The UNIFORM/VOIDS/FULL curves in guessing2.png are NOT "
                "recomputed here; they are re-read from guessing.txt and are "
                "over all 2000 records.\n")
        f.write(f"# posterior: WeightedPosterior.from_view(level={LEVEL.value},"
                f" policy=HeuristicPlayer, epsilon={EPSILON}, "
                f"n_worlds={N_WORLDS}, max_draws={MAX_DRAWS})\n")
        f.write("# rng: default_rng([rec.seed, trick, observer]) -> seed "
                "(same scheme as run_survival.py)\n")
        f.write("# trick k = belief state entering trick k (k=1 is the fresh "
                "deal; with no observed plays every world has weight 1, so "
                "trick 1 is the FULL-table sampler anchor ~0.3333)\n")
        f.write("# observers: all 4 seats per boundary; metrics are the mean "
                "over truth cards within an observer, then the mean over "
                "boundary-observers (same nesting as Phase 1)\n")
        f.write("# truth zeros: a true card whose sampled marginal is exactly "
                "0.0 (possible with finite worlds; NOT a floor and NOT "
                "smoothed). Counted in truth_zero_frac; included in meanP "
                "(as 0.0) and top1 (as a miss); EXCLUDED from NLL.\n")
        f.write("# NLL is therefore computed over p>0 cards ONLY; n_nll_obs "
                "is how many of the n_scored observers had >=1 such card.\n")
        f.write("#   CONSEQUENCE for the 4-line NLL panel: the UNIFORM/VOIDS/"
                "FULL curves are exact tables (no truth zeros possible, NLL "
                "over ALL cards), while CHOICE drops its truth-zero cards -- "
                "exactly the hardest ones. The strictly comparable CHOICE NLL "
                "is infinite wherever truth_zero_frac > 0, so the CHOICE NLL "
                "line is optimistically biased vs the other three. meanP and "
                "top1 are unaffected (zeros count as 0.0 and as misses) and "
                "are clean comparisons.\n")
        f.write("# collapse: from_view raised (no candidate world survived). "
                "Those boundary-observers are counted in collapse_frac and "
                "EXCLUDED from all means (not imputed as zero); n_scored is "
                "the surviving denominator.\n")
        f.write("# truth-world invariant: at every boundary-observer below, "
                "world_weight(view, TRUE world, HeuristicPlayer, eps=0) was "
                "asserted == 1.0 exactly (the records are 4x deterministic "
                "HeuristicPlayer, so anything else means a replay or seat-"
                "mapping bug). The run completing means it held everywhere.\n")
        f.write(f"# wall_time_s={elapsed:.1f} workers={workers} "
                f"smoke={smoke}\n")
        f.write(HEADER_COLS)
        for r in rows:
            f.write(_row_line(r))


def print_table(rows):
    print(f"\n{'trick':>5} {'n':>6} {'meanP':>9} {'NLL':>9} {'top1':>9} "
          f"{'tzero':>9} {'collapse':>9} {'neff':>8} {'draws':>10}")
    for r in rows:
        print(f"{r['trick']:>5} {r['n_scored']:>6} {r['meanP']:>9.4f} "
              f"{r['NLL']:>9.4f} {r['top1']:>9.4f} "
              f"{r['truth_zero_frac']:>9.5f} {r['collapse_frac']:>9.5f} "
              f"{r['mean_n_effective']:>8.2f} {r['mean_draws_used']:>10.1f}")


def read_phase1(path):
    """Re-read the Phase-1 curves: level -> (13, 3) array [meanP, NLL, top1]."""
    out = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("level"):
                continue
            level, k, p, nll, top1 = line.split()
            out.setdefault(level, np.zeros((NUM_TRICKS, 3)))[int(k) - 1] = (
                float(p), float(nll), float(top1)
            )
    for level, arr in out.items():
        assert (arr != 0).any(axis=1).all(), f"missing tricks for {level}"
    return out


METRICS = [("meanP", "mean P(truth)"), ("NLL", "mean -ln P(truth)"),
           ("top1", "top-1 accuracy")]


def plot(path, rows, phase1, num_records, smoke):
    tricks = np.array([r["trick"] for r in rows])
    choice = np.array([[r["meanP"], r["NLL"], r["top1"]] for r in rows])
    all_tricks = np.arange(1, NUM_TRICKS + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for j, (short, label) in enumerate(METRICS):
        ax = axes[j]
        for level in ("uniform", "voids", "full"):
            ax.plot(all_tricks, phase1[level][:, j], marker="o", alpha=0.75,
                    label=f"{level} (2000 games)")
        ax.plot(tricks, choice[:, j], marker="s", color="crimson", lw=2,
                label=f"choice (eps=0, {num_records} games)")
        if short == "meanP":
            ax.axhline(CEILING, ls="--", color="gray", lw=1)
            ax.annotate("constraint-evidence ceiling\n"
                        f"(full, trick 13 = {CEILING:.3f})",
                        xy=(13, CEILING), xytext=(5.2, CEILING + 0.03),
                        fontsize=8, color="gray",
                        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
        ax.set_xlabel("trick number (belief state entering trick)")
        ax.set_ylabel(label)
        ax.set_title(short + (" (NLL over p>0 cards only)"
                              if short == "NLL" else ""))
        ax.grid(alpha=0.3)
        ax.legend(title="evidence level", fontsize=8)
    fig.suptitle(
        "Guessing quality by evidence level: constraints vs choices"
        + ("  [SMOKE]" if smoke else "")
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help=f"first {SMOKE_RECORDS} records, {SMOKE_WORKERS} "
                         f"workers, separate outputs, true-world assert on")
    ap.add_argument("--records", type=int, default=None,
                    help="override the record count (default 500)")
    args = ap.parse_args()

    all_records = read_records(RECORDS_PATH)
    if args.smoke:
        n = args.records or SMOKE_RECORDS
        records, workers, chunk = all_records[:n], SMOKE_WORKERS, 4
        out_path, plot_path, partial = SMOKE_OUT_PATH, SMOKE_PLOT_PATH, None
        verify_truth = True
    else:
        n = args.records or NUM_RECORDS
        records, workers, chunk = all_records[:n], WORKERS, CHUNK
        out_path, plot_path, partial = OUT_PATH, PLOT_PATH, PARTIAL_PATH
        # Always on: one extra world_weight call against the hundreds-to-tens-
        # of-thousands already done per boundary-observer is ~0.01% overhead,
        # and it converts "no seat/replay bug on 8 smoke records" into "the
        # true world had weight exactly 1.0 at every boundary-observer of the
        # headline run".
        verify_truth = True

    os.makedirs(RESULTS, exist_ok=True)
    print(f"processing {len(records)} records with {workers} workers "
          f"(chunk={chunk}, verify_truth={verify_truth})", flush=True)

    acc, elapsed = run_all(records, workers, chunk, verify_truth, partial)
    rows = summarize(acc)
    write_table(out_path, rows, len(records), elapsed, workers, args.smoke)
    plot(plot_path, rows, read_phase1(PHASE1_TABLE), len(records), args.smoke)
    print_table(rows)

    full = read_phase1(PHASE1_TABLE)["full"]
    print("\nCHOICE vs FULL (meanP):")
    for r in rows:
        d = r["meanP"] - full[r["trick"] - 1, 0]
        print(f"  trick {r['trick']:>2}: choice {r['meanP']:.4f}  "
              f"full {full[r['trick'] - 1, 0]:.4f}  delta {d:+.4f}")
    print(f"\nwrote {out_path}, {plot_path} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
