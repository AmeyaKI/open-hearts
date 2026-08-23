"""How well does the exploiter's NESTED MODEL imitate the real champion?

PLAINLY. Inside its imagination the exploiter replaces the beginner heuristic
with a miniature copy of the champion. The miniature is the same algorithm at
fewer samples, so it sometimes plays a different card than the real champion
would. This script measures how often it plays the SAME card, on positions the
real champion actually meets, split by stage of the hand.

It is the committed version of the C1.2 "seed-book probe" (PHASE6_C0_NOTES.md),
extended for the 2026-08-23 escalation with an arbitrary list of model sizes:

    .venv/bin/python experiments/probe_model_agreement.py \
        --games 10 --sizes 5x2,20x5

METHOD (identical to C1.2, so the numbers are comparable with the banked ones):
  * harvest every REAL decision (>1 legal move) from `--games` CHAMPION-vs-
    champion games, deal seeds 950000+, champion = HonestSearchPlayer(FULL,
    50x20) in all four seats;
  * re-ask each position of a FRESHLY BUILT player, changing nothing but the
    size and the rng seed;
  * reference = 50x20 with seed 1. Each model size is asked with seed 1 too
    ("same seed"). A second 50x20 with seed 2 gives the champion's agreement
    with ITSELF -- the ceiling any model can reach, since two runs of the real
    bot under different dice already disagree that often.

Fusion (Phase 2.8) is used everywhere for speed. It is bitwise -- the same
cards either way -- so it changes the wall clock of this probe and nothing it
reports.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openhearts.belief.table import Level  # noqa: E402
from openhearts.engine import cards  # noqa: E402
from openhearts.engine.game import deal  # noqa: E402
from openhearts.search.honest import HonestSearchPlayer  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

LEVEL = Level.FULL
CHAMP = (50, 20)
HARVEST_SEED_BASE = 950_000
REF_SEED = 1
ALT_SEED = 2

BUCKETS = (("1-4 (early)", 0, 4), ("5-9 (middle)", 4, 9),
           ("10-13 (endgame)", 9, 13))


def _player(n_outer, n_inner, seed, level=LEVEL):
    return HonestSearchPlayer(level, n_outer, n_inner,
                              np.random.default_rng(seed), fused=True)


def harvest(n_games):
    """Real decisions from champion-vs-champion play, with their trick number."""
    out = []
    for g in range(n_games):
        state = deal(np.random.default_rng(HARVEST_SEED_BASE + g))
        players = [_player(*CHAMP, seed=HARVEST_SEED_BASE + g * 4 + s)
                   for s in range(4)]
        while not state.is_over():
            seat = state.to_play
            v = state.view_for(seat)
            if len(cards.cards_in(v.legal_moves)) > 1:
                out.append((v, int(v.trick_number)))
            state.play(players[seat].choose(v))
        print(f"[harvest] game {g + 1}/{n_games}: {len(out)} decisions",
              flush=True)
    return out


def ask(views, n_outer, n_inner, seed, level=LEVEL):
    """One freshly built player per position, as C1.2 did."""
    t0 = time.time()
    cards_out = [_player(n_outer, n_inner, seed, level).choose(v)
                 for v, _t in views]
    print(f"[ask] {n_outer}x{n_inner} {level.name} seed {seed}: "
          f"{len(views)} decisions in "
          f"{time.time() - t0:.1f}s", flush=True)
    return cards_out


def agree(a, b, views, lo=None, hi=None):
    idx = [i for i, (_v, t) in enumerate(views)
           if lo is None or (lo <= t < hi)]
    if not idx:
        return float("nan"), 0
    return sum(a[i] == b[i] for i in idx) / len(idx), len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--sizes", default="5x2,20x5",
                    help="comma-separated n_outer x n_inner")
    ap.add_argument("--model-level", default=LEVEL.name,
                    help="belief level for the MODEL sizes only (the "
                         "reference champion is always FULL). Prototyping "
                         "the cheaper-nested-posterior lever.")
    ap.add_argument("--min-decisions", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(RESULTS,
                                                  "model_agreement.txt"))
    args = ap.parse_args()
    sizes = [tuple(int(x) for x in s.split("x"))
             for s in args.sizes.split(",") if s]

    views = harvest(args.games)
    assert len(views) >= args.min_decisions, (
        f"corpus too small: {len(views)} < {args.min_decisions}")

    ref = ask(views, *CHAMP, seed=REF_SEED)
    cols = [(f"{CHAMP[0]}x{CHAMP[1]} s1 vs s2",
             ask(views, *CHAMP, seed=ALT_SEED))]
    mlevel = Level[args.model_level]
    tag = "" if mlevel is LEVEL else f" {mlevel.name}"
    for (o, i) in sizes:
        cols.append((f"{o}x{i}{tag} vs {CHAMP[0]}x{CHAMP[1]}",
                     ask(views, o, i, seed=REF_SEED, level=mlevel)))

    lines = []
    a = lines.append
    a("Nested-model agreement with the real champion")
    a("=" * 64)
    a(f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    a(f"champion: HonestSearchPlayer({LEVEL.name}, {CHAMP[0]}x{CHAMP[1]}), "
      f"fused (bitwise)")
    a(f"corpus: {len(views)} real decisions from {args.games} "
      f"champion-vs-champion games, deal seeds {HARVEST_SEED_BASE}+")
    a("")
    head = ["tricks"] + [c[0] for c in cols] + ["n"]
    a(" | ".join(head))
    a(" | ".join("---" for _ in head))
    row = ["ALL"]
    for _name, col in cols:
        row.append(f"{agree(ref, col, views)[0]:.3f}")
    row.append(str(len(views)))
    a(" | ".join(row))
    for label, lo, hi in BUCKETS:
        row = [label]
        n = 0
        for _name, col in cols:
            v, n = agree(ref, col, views, lo, hi)
            row.append(f"{v:.3f}")
        row.append(str(n))
        a(" | ".join(row))
    text = "\n".join(lines)
    print("\n" + text)
    with open(args.out, "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
