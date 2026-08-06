"""Question A: how well do the beliefs locate hidden cards?

This is the ONLY file in the project where ground truth and belief output
meet. Belief code itself sees nothing but a PlayerView; here we replay a
recorded game (whose record stores the original deal) and score the resulting
tables against who really held what. Nothing in this module may be imported
by player, belief, or search code.

Snapshots are taken at completed-trick boundaries only. Row k-1 of the
returned array is the belief state ENTERING trick k: the fresh deal for k=1,
and after k-1 completed tricks otherwise. Mid-trick views are deliberately
not evaluated. At these boundaries an observer has 39 - 3*(k-1) unseen cards,
so every row 1..13 is non-empty (row 13 scores the last 3 unseen cards).
"""
import numpy as np

from openhearts.belief.table import BeliefTable, Level
from openhearts.engine import cards
from openhearts.engine.state import GameState

NUM_TRICKS = 13
NUM_METRICS = 3


def metrics_for(probs, truth):
    """Score a (3, 52) table against truth: card -> true opponent index.

    Returns (mean P(truth), mean -ln P(truth), mean top-1 accuracy).
    P(truth) is never floored or clipped: a zero on the true holder is a
    catastrophic, unrecoverable bug and must crash rather than be smoothed.
    """
    assert truth, "no cards to score"
    ps = []
    hits = []
    for c, i in truth.items():
        p = probs[i, c]
        assert p > 0.0, f"true holder {i} zeroed for card {c}"
        ps.append(p)
        hits.append(1.0 if int(np.argmax(probs[:, c])) == i else 0.0)
    ps = np.asarray(ps, dtype=float)
    return float(ps.mean()), float((-np.log(ps)).mean()), float(np.mean(hits))


def _truth_for(observer_seat, rec, table):
    """Map each card unseen by this observer to its true opponent index.

    The truth is the ORIGINAL deal: an unseen card is by definition one that
    has not been played, so whoever was dealt it still holds it. The seat is
    then converted to this observer's opponent index via table.opponent_seats.
    """
    truth = {}
    for c in cards.cards_in(table.unseen_mask):
        holder = next(s for s in range(4) if rec.hands[s] & cards.bit(c))
        assert holder != observer_seat, (
            f"card {c} counted as unseen but observer {observer_seat} was "
            f"dealt it"
        )
        truth[c] = table.opponent_seats.index(holder)
    return truth


def _snapshot(state, rec, level):
    """Mean of the three metrics over all 4 observer seats at this boundary."""
    rows = []
    for seat in range(4):
        table = BeliefTable.from_view(state.view_for(seat), level)
        rows.append(metrics_for(table.probs, _truth_for(seat, rec, table)))
    return np.mean(np.array(rows, dtype=float), axis=0)


def evaluate_record(rec, level: Level) -> np.ndarray:
    """Replay one recorded game; return (13, 3) metrics per trick boundary."""
    state = GameState(hands=list(rec.hands))
    # The record does not store to_play; the 2-of-clubs holder leads trick 1.
    # Without this, a replay where seat 0 is not the leader would desync
    # seats from the recorded game and quietly score the wrong observers.
    state.to_play = next(
        s for s in range(4) if rec.hands[s] & cards.bit(cards.TWO_CLUBS)
    )
    assert state.to_play == rec.plays[0][0], "record leader is not 2c holder"

    out = np.zeros((NUM_TRICKS, NUM_METRICS))
    out[0] = _snapshot(state, rec, level)  # entering trick 1: the fresh deal
    for seat, card in rec.plays:
        assert seat == state.to_play, (
            f"replay desync: recorded seat {seat} != {state.to_play}"
        )
        state.play(card)  # play() asserts the recorded card was legal
        if not state.current_trick and state.trick_number < NUM_TRICKS:
            out[state.trick_number] = _snapshot(state, rec, level)
    assert state.is_over(), "record did not replay to a finished game"
    return out


def run(records, level: Level) -> np.ndarray:
    """Mean (13, 3) metrics across games."""
    acc = np.zeros((NUM_TRICKS, NUM_METRICS))
    n = 0
    for rec in records:
        acc += evaluate_record(rec, level)
        n += 1
    assert n > 0, "no records"
    return acc / n
