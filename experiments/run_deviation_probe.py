"""Task A2 (PHASE7_PLAN.md): WHERE does the exploiter earn its +0.198?

WHY, PLAINLY. The R1v verdict says a student of our bot takes +0.198
pts/hand off it, and the points ARRIVE in tricks 10-13 (the endgame trade).
But the C2 diagnostics hinted the edge is "a better middle game feeding the
amplified endgame trade" -- i.e. the DECISIONS that cause the endgame bleed
may happen in tricks 5-9 (win an innocuous trick at trick 8 so the champion
is on lead at trick 11 holding only losers). The planned mixing fix (7D-3b)
randomizes tricks 10-13; if the causal decisions live earlier, that window
is misaimed. This probe measures the location directly.

HOW. The exploiter with its model off is bitwise honest-FULL INCLUDING its
rng stream (exploiter.py's constructor contract), and all of a decision's
randomness flows through the one `self.rng` generator (honest.py shares it
with the inner player; the heuristic draws nothing; the nested model uses
its own separate stream by contract 2). So at every real exploiter decision
along a real R1v game we can snapshot the rng state, let the model-ON
exploiter choose, then rewind a model-OFF twin to the same state and ask
what plain honest-FULL would have played from the identical view and the
IDENTICAL sampled worlds. Any difference is the model's doing -- not the
39% Monte-Carlo self-disagreement noise, which same-state pairing removes
by construction.

IMPACT WEIGHT. When the two disagree, the deviation's weight is the
exploiter's own imagined-score gap between the two cards (its `last_avgs`
entry for the twin's card minus its chosen card's). Honest caveat, stated
here and in the output: that is the ATTACKER'S BELIEF about the value of
deviating, not ground-truth points -- good enough to locate where the model
changes behaviour most, which is all A2 asks.

SANITY GATE (--gate, run before any real probe; the alarm counts its own
checks). Tracked seat = the SAME probe subclass with champion_model=None
(bitwise honest-FULL) + the same twin. Every decision must match exactly:
one mismatch means the same-rng-state pairing is broken and every number
this script could produce would be noise. gate_checks is asserted > 0.

CONFIG. R1v recipe exactly (1 probe seat + 3 champions, nested 20x5,
2 plies, VOIDS nested posterior, --fast switches), deal seeds 100000+i --
the same deals the +0.198 verdict was measured on, so the diagnostic is
in-distribution with the number it explains. Checkpointing via
`openhearts.eval.blockdriver` (Task F0). No banked partial is touched:
this writes its own results/deviation_probe_partial.txt.

Usage:
    .venv/bin/python experiments/run_deviation_probe.py --gate
    .venv/bin/python experiments/run_deviation_probe.py --deals 40 --workers 8
"""
import argparse
import copy
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import run_exploit_eval as ree  # noqa: E402  (import-safe: __main__ guarded)
from openhearts.belief.table import Level  # noqa: E402
from openhearts.engine import cards  # noqa: E402
from openhearts.engine.game import deal  # noqa: E402
from openhearts.eval import blockdriver  # noqa: E402
from openhearts.players.heuristic import HeuristicPlayer  # noqa: E402
from openhearts.search.exploiter import (ExploiterSearchPlayer,  # noqa: E402
                                         champion_model_factory)
from openhearts.search.honest import HonestSearchPlayer  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
PARTIAL = os.path.join(RESULTS, "deviation_probe_partial.txt")
OUT = os.path.join(RESULTS, "deviation_probe.txt")

ROW = "R1v"                     # seed bookkeeping matches the banked attacker
NESTED_OUTER, NESTED_INNER = 20, 5
NESTED_LEVEL = "VOIDS"
NESTED_PLIES = 2
DEAL_SEED_BASE = ree.DEAL_SEED_BASE      # 100000+, the verdict's own deals

#: --fast equivalents, applied to ree.OPTS in every process (workers spawn
#: fresh, so the worker re-applies them; bitwise switches only, plus the
#: R1v attacker definition).
FAST_OPTS = {"fused_nested": True, "fused_champions": True,
             "batched_model": True,
             "nested_outer": NESTED_OUTER, "nested_inner": NESTED_INNER,
             "nested_level": NESTED_LEVEL, "nested_plies": NESTED_PLIES}


class DeviationProbe(ExploiterSearchPlayer):
    """R1v exploiter that also asks, at every real decision, what its own
    model-OFF self (bitwise honest-FULL) would have played from the same rng
    state -- then plays its model-ON card so the game IS an R1v game."""

    def __init__(self, *args, twin=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert twin is not None, "DeviationProbe needs its model-off twin"
        self.twin = twin
        self.records = []        # dicts: t, role, on, off, imp
        self.decisions = 0       # multi-card decisions actually compared
        self.singles = 0         # single-legal-card plays (nothing to compare)

    @staticmethod
    def _role(view):
        if not view.current_trick:
            return "lead"
        led = cards.suit(view.current_trick[0][1])
        return "follow" if view.hand & cards.SUIT_MASK[led] else "discard"

    def choose(self, view):
        legal = cards.cards_in(view.legal_moves)
        if len(legal) == 1:
            self.singles += 1
            return super().choose(view)

        snap = copy.deepcopy(self.rng.bit_generator.state)
        card_on = super().choose(view)
        avgs = np.asarray(self.last_avgs, dtype=float)
        cand = list(self.last_candidates)

        self.twin.rng.bit_generator.state = copy.deepcopy(snap)
        card_off = self.twin.choose(view)
        self.decisions += 1

        if card_off != card_on:
            # Both cards are in `cand`: grouping is off, so candidates ==
            # legal for probe and twin alike. A KeyError here would mean the
            # pairing is broken -- let it raise.
            imp = float(avgs[cand.index(card_off)] - avgs[cand.index(card_on)])
            self.records.append({
                "t": len(view.history) // 4 + 1,     # 1-based trick number
                "role": self._role(view),
                "on": int(card_on), "off": int(card_off),
                "imp": imp,
            })
        return card_on


def _build_probe_table(deal_seed, rotation, model_on):
    """R1v seating (probe tracked + 3 champions), seeds exactly as
    run_exploit_eval.build_table derives them."""
    tracked = rotation
    champ_seats = [s for s in range(4) if s != tracked]
    players = [None, None, None, None]
    for s in champ_seats:
        players[s] = HonestSearchPlayer(
            ree.LEVEL, ree.N_OUTER, ree.N_INNER,
            np.random.default_rng(ree.seat_seed(deal_seed, rotation, s)),
            fused=ree.OPTS["fused_champions"])

    twin = HonestSearchPlayer(
        ree.LEVEL, ree.N_OUTER, ree.N_INNER,
        np.random.default_rng(0))   # state is overwritten before every use
    trng = np.random.default_rng(ree.seat_seed(deal_seed, rotation, tracked))
    if model_on:
        players[tracked] = DeviationProbe(
            ree.LEVEL, ree.N_OUTER, ree.N_INNER, trng,
            champion_model=champion_model_factory(
                Level[NESTED_LEVEL], NESTED_OUTER, NESTED_INNER,
                fused=ree.OPTS["fused_nested"]),
            champion_seats=champ_seats,
            seed_mode="unknown",
            model_seed=ree.model_seed(ROW, deal_seed, rotation),
            nested_size=(NESTED_OUTER, NESTED_INNER),
            nested_level=Level[NESTED_LEVEL],
            nested_plies=NESTED_PLIES,
            batched_model=ree.OPTS["batched_model"],
            measuring_instrument=True,
            twin=twin)
    else:
        # The gate configuration: model OFF, so probe seat is bitwise
        # honest-FULL and must agree with the twin on every decision.
        players[tracked] = DeviationProbe(
            ree.LEVEL, ree.N_OUTER, ree.N_INNER, trng,
            champion_model=None, twin=twin)
    return players, tracked


def _play(deal_seed, rotation, model_on):
    state = deal(np.random.default_rng(deal_seed))
    players, tracked = _build_probe_table(deal_seed, rotation, model_on)
    while not state.is_over():
        seat = state.to_play
        state.play(players[seat].choose(state.view_for(seat)))
    assert sum(state.scores) == 26, "engine invariant broken"
    p = players[tracked]
    return {"records": p.records, "decisions": p.decisions,
            "singles": p.singles, "tracked_pts": int(state.scores[tracked])}


def worker(name, block_idx, item_indices):
    ree.OPTS.update(FAST_OPTS)   # spawned process: re-apply the R1v config
    records, decisions, singles, pts = [], 0, 0, 0
    for i in item_indices:
        for rotation in range(4):
            g = _play(DEAL_SEED_BASE + i, rotation, model_on=True)
            records.extend(g["records"])
            decisions += g["decisions"]
            singles += g["singles"]
            pts += g["tracked_pts"]
    return {"records": records, "decisions": decisions,
            "singles": singles, "tracked_pts": pts,
            "games": 4 * len(item_indices)}


def gate(n_deals=2):
    """Model-OFF probe vs twin: zero deviations or this whole file is noise."""
    ree.OPTS.update(FAST_OPTS)
    checks, deviations = 0, 0
    for i in range(n_deals):
        for rotation in range(4):
            g = _play(DEAL_SEED_BASE + i, rotation, model_on=False)
            checks += g["decisions"]
            deviations += len(g["records"])
    assert checks > 0, "gate ran zero comparisons -- it proved nothing"
    print(f"GATE: {checks} model-off decisions compared, "
          f"{deviations} deviations (must be 0)")
    assert deviations == 0, (
        "model-off probe disagreed with its twin: the same-rng-state pairing "
        "is broken; every deviation this script reports would be noise")
    print("GATE PASSED")


BUCKETS = (("1-4", 1, 4), ("5-9", 5, 9), ("10-13", 10, 13))


def report(all_records, decisions, singles, games, pts, wall, workers):
    lines = []
    a = lines.append
    a("Phase 7 Task A2 -- exploiter deviation-location diagnostic")
    a("=" * 64)
    a(f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    a(f"attacker: R1v recipe (nested {NESTED_OUTER}x{NESTED_INNER}, "
      f"{NESTED_PLIES} plies, {NESTED_LEVEL} nested posterior, batched+fused)")
    a(f"deals: seeds {DEAL_SEED_BASE}..{DEAL_SEED_BASE + games // 4 - 1} "
      f"x 4 rotations = {games} games  |  workers: {workers}  |  "
      f"wall: {wall:.0f}s")
    a(f"probe seat pts/hand (sanity vs R1v's ~6.0-6.2 tracked side): "
      f"{pts / games:.3f}")
    a(f"multi-card decisions compared: {decisions}  |  single-card plays "
      f"(skipped, nothing to compare): {singles}")
    n_dev = len(all_records)
    tot_imp = sum(r["imp"] for r in all_records) or 1e-12
    a(f"deviations (model-on card != model-off card, SAME rng state, same "
      f"worlds): {n_dev} ({n_dev / max(decisions, 1):.1%} of decisions)")
    a("impact = exploiter's own imagined-score gap (its BELIEF about the "
      "deviation's value, not ground truth) -- see file docstring")
    a("")
    a(f"{'tricks':>8} | {'decisions':>9} | {'devs':>5} | {'rate':>6} | "
      f"{'impact sum':>10} | {'impact share':>12}")
    dec_by_t = {}
    # decisions-per-trick denominator is not tracked per record; rates per
    # bucket use deviation counts over TOTAL decisions and are labeled so.
    for label, lo, hi in BUCKETS:
        rs = [r for r in all_records if lo <= r["t"] <= hi]
        imp = sum(r["imp"] for r in rs)
        a(f"{label:>8} | {'---':>9} | {len(rs):>5} | "
          f"{len(rs) / max(n_dev, 1):>6.1%} | {imp:>10.3f} | "
          f"{imp / tot_imp:>12.1%}")
    a("(rate column = share of all deviations, not per-bucket decision rate)")
    a("")
    a("by decision type:")
    for role in ("lead", "follow", "discard"):
        rs = [r for r in all_records if r["role"] == role]
        imp = sum(r["imp"] for r in rs)
        a(f"  {role:>8}: {len(rs):>5} devs | impact {imp:>8.3f} "
          f"({imp / tot_imp:.1%})")
    a("")
    mid = sum(r["imp"] for r in all_records if 5 <= r["t"] <= 9) / tot_imp
    end = sum(r["imp"] for r in all_records if 10 <= r["t"] <= 13) / tot_imp
    a(f"PRE-REGISTERED QUESTION (plan A2): impact-weighted share in tricks "
      f"5-9 = {mid:.1%}, tricks 10-13 = {end:.1%}")
    a(f"DECISION RULE: >50% in tricks 5-9 -> 7D-3b window/knobs are "
      f"redesigned before its pre-registration is frozen. Verdict input: "
      f"{'REDESIGN' if mid > 0.5 else '3b window stands (pending owner read)'}")
    text = "\n".join(lines)
    print(text)
    with open(OUT, "w") as f:
        f.write(text + "\n")
    print(f"\nwritten: {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--deals", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--block", type=int, default=None)
    args = ap.parse_args()
    ree.OPTS.update(FAST_OPTS)

    if args.gate:
        gate()
        return

    os.makedirs(RESULTS, exist_ok=True)
    all_records, tot = [], {"decisions": 0, "singles": 0,
                            "tracked_pts": 0, "games": 0}

    def on_block_done(_b, _items, payload):
        all_records.extend(payload["records"])
        for k in tot:
            tot[k] += payload[k]

    t0 = time.time()
    blockdriver.run_blocks(
        "A2", args.deals,
        args.block if args.block else blockdriver.DEFAULT_BLOCK_SIZE,
        worker, args.workers, PARTIAL,
        on_block_done=on_block_done,
        explicit_block=args.block is not None)
    report(all_records, tot["decisions"], tot["singles"], tot["games"],
           tot["tracked_pts"], time.time() - t0, args.workers)


if __name__ == "__main__":
    main()
