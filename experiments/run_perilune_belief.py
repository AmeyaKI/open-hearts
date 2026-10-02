"""B6 Tier 0: a LEARNED belief head against an EXACT belief table, on identical positions.

WHAT THIS MEASURES, PLAINLY.  Both projects must answer the same question before they can
search: "who is holding each card I cannot see?"  We answer with arithmetic -- a table built
from the rules alone (voids, hand sizes), `BeliefTable` Level.FULL.  Perilune answers with a
neural network head trained on self-play to predict the three opponents' hands.  Nobody has
put the two side by side.  This script replays whole games and, at EVERY decision, asks both
for their probabilities and scores them against who really held each card.

    A card example.  West discarded on a club lead, so West has no clubs.  Our table gives
    West probability 0 for every club and spreads the clubs over North and East.  The learned
    head was shown the same void flag and has to have LEARNED that rule -- and it may also
    have learned things the table cannot know, such as "a player who ducked that trick
    probably does not hold the queen".  Which effect is bigger is the measurement.

POSITIONS (observer = the seat about to play; all 52 decisions of every game):
    --table heuristic   4x our beginner heuristic       (Phase-3 home-turf convention)
    --table ours        4x honest-FULL 50x20 fused      (our shipped bot's own games)
    --table perilune    4x Perilune raw net             (THEIR play: in-distribution for the head)
Games use seeds 1,200,000+ (a fresh block; smokes 1,290,000+).

CURVES:
    uniform / voids / exact   our BeliefTable at Level.UNIFORM / VOIDS / FULL (exact = shipped)
    ours-sampler              Monte-Carlo marginals of OUR world sampler on the exact table
    perilune-raw              sigmoid(belief logits), as trained (independent per-cell)
    perilune-normalized       raw renormalised over the three opponents per card
    perilune-deployed         the per-card weights their sampler uses (floor 1e-4, their void
                              tracker, renormalised)
    perilune-sampler          Monte-Carlo marginals of a port of THEIR determinization sampler
The two *-sampler curves are the like-for-like "as deployed" comparison; they cost a Python
loop per draw and run only when --sampler-draws > 0.

METRICS (house convention, run_guessing2/5): over every unseen card, P(truth), top-1 hit, and
-ln P(truth) over cards with P>0 ONLY, with `zero` (P = 0 on the true holder) and `lt01`
(P < 0.01) reported separately -- nothing is floored or clipped.  Added here: `cellBCE`, the
binary cross-entropy over all 3 x unseen (opponent, card) cells -- Perilune's own training
loss, so the raw reading is also judged on the yardstick it was optimised for (cells whose
loss is infinite are counted in `cellInf` and excluded, never smoothed).

This file, like eval/guessing.py, is a place where ground truth meets belief output; nothing
here may be imported by player, belief or search code.

Usage (nothing runs without an explicit flag):
    --smoke                         3 games per table, 2 workers, 8 sampler draws
    --run --workers N [--games G]   the registered run (resumable; rerun the same command)
    --report                        re-print from the partials
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from openhearts.belief.table import BeliefTable, Level  # noqa: E402
from openhearts.engine import cards  # noqa: E402
from openhearts.engine.game import deal, play_game  # noqa: E402
from openhearts.engine.state import GameState  # noqa: E402
from openhearts.eval import blockdriver  # noqa: E402
from openhearts.eval.seeds import HARNESS_PERILUNE, SEEDS_TOKEN, bench_seat_seed  # noqa: E402
from openhearts.eval.stats import bootstrap_ci  # noqa: E402
from openhearts.players.heuristic import HeuristicPlayer  # noqa: E402
from openhearts.sampler.sampler import sample_arrangement  # noqa: E402
from openhearts.search.honest import HonestSearchPlayer  # noqa: E402
from perilune import adapter  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
SEED_BASE = 1_200_000
SMOKE_SEED_BASE = 1_290_000
BLOCK_SIZE = blockdriver.DEFAULT_BLOCK_SIZE
N_OUTER, N_INNER = 50, 20
TABLES = {"heuristic": 1, "ours": 2, "perilune": 3}
ANALYTIC = ("uniform", "voids", "exact", "perilune-raw", "perilune-normalized",
            "perilune-deployed")
SAMPLED = ("ours-sampler", "perilune-sampler")
CURVES = ANALYTIC + SAMPLED
# per (curve, trick) accumulator columns
COLS = ("n", "sum_p", "hits", "zero", "lt01", "sum_nll", "n_nll", "sum_bce", "n_bce", "inf_bce")
NUM_TRICKS = 13


def score(probs, truth, unseen):
    """One (3, 52) table against truth {card: opponent index}. Returns a COLS-ordered list."""
    acc = dict.fromkeys(COLS, 0.0)
    for c in unseen:
        k = truth[c]
        p = float(probs[k, c])
        acc["n"] += 1
        acc["sum_p"] += p
        acc["hits"] += 1.0 if int(np.argmax(probs[:, c])) == k else 0.0
        if p <= 0.0:
            acc["zero"] += 1
        else:
            acc["sum_nll"] += -np.log(p)
            acc["n_nll"] += 1
        if p < 0.01:
            acc["lt01"] += 1
        for j in range(3):
            q = min(max(float(probs[j, c]), 0.0), 1.0)     # IPF can overshoot 1 by ~1e-9
            y = 1.0 if j == k else 0.0
            wrong = (1.0 - q) if y else q                  # probability mass on the wrong answer
            if wrong >= 1.0:
                acc["inf_bce"] += 1
            else:
                acc["sum_bce"] += -np.log1p(-wrong)
                acc["n_bce"] += 1
    return [acc[k] for k in COLS]


def _truth(state, seat, unseen):
    out = {}
    for c in unseen:
        holder = next(s for s in range(4) if state.hands[s] & cards.bit(c))
        assert holder != seat
        out[c] = (holder - seat - 1) % 4
    return out


def _ours_sampler_marginals(table, rng, n_draws):
    counts = np.zeros((3, 52))
    got = 0
    for _ in range(n_draws):
        res = sample_arrangement(table, rng)
        if res is None:
            continue
        hands, _ = res
        got += 1
        for k in range(3):
            for c in cards.cards_in(hands[k]):
                counts[k, c] += 1.0
    return (counts / got if got else counts), n_draws - got


def curves_at(view, net, sampler_draws, rng):
    """{curve: (3, 52) probs} for one view, plus sampler diagnostics."""
    out, diag = {}, {}
    for name, level in (("uniform", Level.UNIFORM), ("voids", Level.VOIDS), ("exact", Level.FULL)):
        table = BeliefTable.from_view(view, level)
        out[name] = table.probs
    r = adapter.belief_readings(net, view)
    out["perilune-raw"], out["perilune-normalized"] = r["raw"], r["normalized"]
    out["perilune-deployed"] = r["deployed"]
    if sampler_draws > 0:
        out["ours-sampler"], failed = _ours_sampler_marginals(table, rng, sampler_draws)
        out["perilune-sampler"], unweighted = adapter.sampler_marginals(
            net, view, rng, n_draws=sampler_draws)
        diag = {"ours_failed_draws": failed, "perilune_unweighted_frac": unweighted}
    return out, diag


def make_table(table, net, seed, size=(N_OUTER, N_INNER)):
    if table == "heuristic":
        return [HeuristicPlayer()] * 4
    if table == "perilune":
        return [adapter.PeriluneRawPlayer(net)] * 4
    if table == "ours":
        return [HonestSearchPlayer(
            Level.FULL, size[0], size[1],
            np.random.default_rng(bench_seat_seed(HARNESS_PERILUNE, 100 + TABLES[table], 0,
                                                  seed, 0, s)), fused=True) for s in range(4)]
    raise ValueError(table)


def process_game(table, net, seed, sampler_draws, size=(N_OUTER, N_INNER)):
    """Play one game at `table`, then replay it scoring every curve at every decision."""
    start = deal(np.random.default_rng(seed))
    initial_hands, leader = list(start.hands), start.to_play
    final = play_game(start, make_table(table, net, seed, size))
    plays = list(final.history)
    assert len(plays) == 52 and sum(final.scores) == 26

    state = GameState(hands=list(initial_hands))
    state.to_play = leader
    names = CURVES if sampler_draws > 0 else ANALYTIC
    acc = {name: np.zeros((NUM_TRICKS, len(COLS))) for name in names}
    diag = {"ours_failed_draws": 0, "perilune_unweighted": 0.0, "positions": 0}
    for j, (seat, card) in enumerate(plays):
        assert seat == state.to_play, "replay desync"
        view = state.view_for(seat)
        unseen = cards.cards_in(adapter.unseen_mask(view))
        truth = _truth(state, seat, unseen)
        rng = np.random.default_rng([seed, j, 777])
        probs, d = curves_at(view, net, sampler_draws, rng)
        for name in names:
            acc[name][state.trick_number] += score(probs[name], truth, unseen)
        diag["positions"] += 1
        diag["ours_failed_draws"] += d.get("ours_failed_draws", 0)
        diag["perilune_unweighted"] += d.get("perilune_unweighted_frac", 0.0)
        state.play(card)
    assert state.is_over()
    return {name: a.tolist() for name, a in acc.items()}, diag


def worker(name, block_idx, item_indices):
    cfg = json.loads(os.environ["PERILUNE_BELIEF_CFG"])
    adapter.single_thread()
    t0 = time.time()
    games, diags = [], []
    for idx in item_indices:
        g, d = process_game(name, cfg["net"], cfg["seed_base"] + idx, cfg["sampler_draws"],
                            tuple(cfg["size"]))
        games.append(g)
        diags.append(d)
    return {"block_size": BLOCK_SIZE, "games": games, "diag": diags,
            "seconds": time.time() - t0}


def partial_path(table, net, tag="", kind=""):
    return os.path.join(RESULTS, f"perilune_belief_{net}_{table}{kind}{tag}_partial.txt")


def report_path(net, tag=""):
    return os.path.join(RESULTS, f"perilune_belief_{net}{tag}.txt")


def header_line(table, net, sampler_draws, size=(N_OUTER, N_INNER)):
    return (f"# {SEEDS_TOKEN} harness=perilune-belief table={table} net={net} "
            f"pin={adapter.PIN_COMMIT[:12]} sha256={adapter.NETS[net][1][:16]} "
            f"sampler_draws={sampler_draws} ours={size[0]}x{size[1]}")


def ensure_header(path, want):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path) as f:
            first = f.readline().rstrip("\n")
        if first != want:
            raise SystemExit(
                f"refusing to resume {path}: header mismatch\n  have: {first}\n  want: {want}")
        return
    with open(path, "a") as f:
        f.write(want + "\n")


def run_tables(tables, net, n_games, workers, seed_base, sampler_draws, tag="", kind="",
               size=(N_OUTER, N_INNER)):
    os.environ["PERILUNE_BELIEF_CFG"] = json.dumps(
        {"net": net, "seed_base": seed_base, "sampler_draws": sampler_draws, "size": list(size)})
    os.makedirs(RESULTS, exist_ok=True)
    for table in tables:
        path = partial_path(table, net, tag, kind)
        want = header_line(table, net, sampler_draws, size)
        ensure_header(path, want)
        print(want, flush=True)
        t0 = time.time()
        blockdriver.run_blocks(table, n_games, BLOCK_SIZE, worker, workers, path)
        print(f"[{table}] {n_games} games in {time.time() - t0:.0f} s", flush=True)


def load_games(table, net, n_games, tag="", kind=""):
    """{game index: {curve: (13, len(COLS)) array}} from the banked blocks."""
    banked = blockdriver.load_partial(partial_path(table, net, tag, kind), table)
    games, diag = {}, {"positions": 0, "ours_failed_draws": 0, "perilune_unweighted": 0.0}
    for b, p in banked.items():
        bs = int(p["block_size"])
        for j, g in enumerate(p["games"]):
            if b * bs + j < n_games:
                games[b * bs + j] = {name: np.array(a) for name, a in g.items()}
                for k in diag:
                    diag[k] += p["diag"][j][k]
    return games, diag


def _col(a, name):
    return a[..., COLS.index(name)]


def per_game(games, curve, metric):
    """One number per game (every game scores the same number of cards, so games weigh equally)."""
    out = []
    for i in sorted(games):
        a = games[i][curve].sum(axis=0)
        n = _col(a, "n")
        if metric == "meanP":
            out.append(_col(a, "sum_p") / n)
        elif metric == "top1":
            out.append(_col(a, "hits") / n)
        elif metric == "NLL":
            out.append(_col(a, "sum_nll") / max(_col(a, "n_nll"), 1.0))
        elif metric == "cellBCE":
            out.append(_col(a, "sum_bce") / max(_col(a, "n_bce"), 1.0))
        else:
            raise ValueError(metric)
    return np.array(out)


def report(net, n_games, tag="", kind="", out_path=None):
    lines = [f"perilune_belief REPORT net={net} {time.strftime('%Y-%m-%d %H:%M')} "
             f"pin={adapter.PIN_COMMIT[:12]}  (observer = seat to act; all 52 decisions/game)"]
    for table in TABLES:
        games, diag = load_games(table, net, n_games, tag, kind)
        if not games:
            continue
        curves = [c for c in CURVES if c in next(iter(games.values()))]
        lines.append(f"\n== table: {table}   games={len(games)}   positions={diag['positions']}"
                     + (f"   ours-sampler failed draws={diag['ours_failed_draws']}   "
                        f"perilune-sampler unweighted-fallback frac="
                        f"{diag['perilune_unweighted'] / max(diag['positions'], 1):.4f}"
                        if "perilune-sampler" in curves else ""))
        lines.append(f"{'curve':22s} {'meanP':>22s} {'NLL(p>0)':>22s} {'top1':>7s} "
                     f"{'zero':>8s} {'lt01':>8s} {'cellBCE':>9s} {'cellInf':>8s}")
        for c in curves:
            tot = sum(g[c].sum(axis=0) for g in games.values())
            n = _col(tot, "n")
            mp = bootstrap_ci(per_game(games, c, "meanP"))
            nl = bootstrap_ci(per_game(games, c, "NLL"))
            cells = _col(tot, "n_bce") + _col(tot, "inf_bce")
            lines.append(
                f"{c:22s} {mp[0]:.4f} ({mp[1]:.4f},{mp[2]:.4f}) {nl[0]:.4f} ({nl[1]:.4f},{nl[2]:.4f}) "
                f"{_col(tot, 'hits') / n:7.4f} {_col(tot, 'zero') / n:8.5f} {_col(tot, 'lt01') / n:8.5f} "
                f"{_col(tot, 'sum_bce') / max(_col(tot, 'n_bce'), 1):9.5f} "
                f"{_col(tot, 'inf_bce') / max(cells, 1):8.6f}")
        lines.append("  paired per game, curve minus exact (NLL and cellBCE: negative = curve "
                     "locates cards better than the exact table; meanP: positive = better):")
        for c in curves:
            if c == "exact":
                continue
            parts = []
            for metric in ("NLL", "cellBCE", "meanP"):
                d = per_game(games, c, metric) - per_game(games, "exact", metric)
                m, lo, hi = bootstrap_ci(d)
                parts.append(f"{metric} {m:+.4f} ({lo:+.4f}, {hi:+.4f})")
            lines.append(f"    {c:22s} " + "   ".join(parts))
        lines.append("  by trick (NLL over p>0; pooled over games):")
        lines.append("    trick " + " ".join(f"{t + 1:>7d}" for t in range(NUM_TRICKS)))
        for c in curves:
            tot = sum(g[c] for g in games.values())
            nll = _col(tot, "sum_nll") / np.maximum(_col(tot, "n_nll"), 1.0)
            lines.append(f"    {c[:20]:20s} " + " ".join(f"{x:7.4f}" for x in nll))
    text = "\n".join(lines)
    print(text)
    if out_path:
        with open(out_path, "a") as f:
            f.write(text + "\n")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", choices=sorted(adapter.NETS), default="v5")
    ap.add_argument("--tables", nargs="+", default=list(TABLES), choices=list(TABLES))
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--sampler-draws", type=int, default=0,
                    help="Monte-Carlo draws per position for the two *-sampler curves (0 = off)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if not adapter.available():
        raise SystemExit("Perilune checkout/weights absent: run scripts/build_perilune.sh")
    if args.smoke:
        for table in args.tables:
            p = partial_path(table, args.net, args.tag, "_smoke")
            if os.path.exists(p):
                os.remove(p)
        run_tables(args.tables, args.net, 3, 2, SMOKE_SEED_BASE, 8, args.tag, "_smoke")
        report(args.net, 3, args.tag, "_smoke")
        return
    if args.run:
        assert args.workers, "owner-approved worker count required: --workers"
        run_tables(args.tables, args.net, args.games, args.workers, SEED_BASE,
                   args.sampler_draws, args.tag)
        report(args.net, args.games, args.tag, out_path=report_path(args.net, args.tag))
        return
    if args.report:
        report(args.net, args.games, args.tag, out_path=report_path(args.net, args.tag))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
