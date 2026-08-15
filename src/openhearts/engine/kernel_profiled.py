"""Phase 5 Task 6: model-driven playouts (Organ 2's kernel).

`kernel.py` plays imagined futures with the hand-coded heuristic in all four
seats. Phase 4's league showed why that is the wrong rehearsal against anyone
who is not the script. This module adds NEW playout entries BESIDE the Phase-2.5
ones -- `kernel.py` is byte-unchanged -- in which:

* OUR seat still chooses with the heuristic (`kernel._choose`), unchanged;
* every OPPONENT seat SAMPLES from the profiler's masked softmax over its own
  legal moves, with the features built IN-KERNEL from the playout state.

WHAT THE OPPONENT SEES (the information boundary, in-kernel)
------------------------------------------------------------
`_ply_features` builds exactly `opponent.obsfeat.observer_features`: a
`FEATURES_V=1` row over a hands array in which ONLY the acting seat's hand is
set, so the rotated hand blocks r=1..3 are structurally zero. The scratch array
is cleared in full every ply -- writing only `fh[seat]` and reusing the buffer
would leak the previous ply's actor into the row (both an information leak and
a silent divergence from `observer_features`). `tests/test_profiled_playout.py`
pins the in-kernel row against the Python helper for EXACT equality over >=1000
(state, seat) pairs, and the in-kernel probability vector against
`infer.profiler_probs` likewise, so the features->distribution link is pinned
too, not just the features.

RNG, and why the stream is not the Python one (Phase 2.6 precedent)
-------------------------------------------------------------------
Sampling uses numba's `np.random` generator, seeded ONCE PER DECISION by
`seed_playouts(seed)` with a single draw from the caller's `numpy.Generator`
(`ProfiledSearchPlayer.choose`). This is the same honesty note
`kernel.sample_arrangements` carries: results are fully deterministic given the
caller's rng state, but the stream is NOT the one a per-card Python sampler
would consume, and JIT vs NO_JIT are two different generators. Nothing here is
bitwise-pinned across those boundaries; the tests pin DISTRIBUTIONS (3-sigma
frequency checks) instead, per the plan's statistical-pinning precedent.

Under `OPENHEARTS_NO_JIT=1` the identical Python source runs, driven by numpy's
GLOBAL `np.random` state. That state is saved and restored around every entry
point (`_enter`/`_exit` below) and the module's own stream is carried in a
private variable, so a Python-path playout cannot perturb any other
`np.random.*` consumer in the process. Under the JIT, numba's generator is
separate from numpy's and no such care is needed.

THE REDUCTION MODE (`mode=MODE_HEURISTIC_ONEHOT`)
-------------------------------------------------
The plan pre-registers: "profiler-playout with the profiler replaced by a
one-hot heuristic-match distribution must reproduce heuristic-playout scores on
identical worlds". `mode=1` fills the SAME `probs[52]` buffer with a one-hot at
`kernel._choose`'s card and runs the SAME cumulative-sample step; a one-hot
cumulative lands on its single atom for any draw, so the result is bitwise
`kernel.playout_to_end` / `playout_until_decision`. Scope, stated honestly: this
pins the LOOP and its bookkeeping (scores, hearts_broken, trick rollover,
stop-seat semantics, play log), which is what could break; the net path is
pinned by the feature/probability equality tests and the sampling-frequency
test instead. No single test covers the whole chain and none pretends to.
"""
import numpy as np

from .features import NF
from .kernel import (HAVE_NUMBA, _choose, _legal, _popcount, _trick_head,
                     jit_enabled, njit)

MODE_PROFILER = 0
MODE_HEURISTIC_ONEHOT = 1

if HAVE_NUMBA:
    from .features import _featurize_njit as _featurize_k
    from ..opponent.infer import _profiler_probs_njit as _probs_k
else:  # pragma: no cover - only without numba installed
    from .features import _featurize_py as _featurize_k
    from ..opponent.infer import _profiler_probs_py as _probs_k

_ONE = np.int64(1)
_QS = 36


# --------------------------------------------------------------------------
# per-ply pieces (shared by the loop and by the test probe, so the tests pin
# the code that actually runs)
# --------------------------------------------------------------------------
def _ply_features_src(fh, hands, seat, played_mask, trick_cards, trick_seats,
                      trick_len, hearts_broken, trick_number, scores):
    """`obsfeat.observer_features` for `seat`, built from kernel arrays."""
    for r in range(4):
        fh[r] = 0
    fh[seat] = hands[seat]
    led_suit, _wr, win_seat = _trick_head(trick_cards, trick_seats, trick_len)
    return _featurize_k(fh, played_mask, trick_cards, trick_seats, trick_len,
                        led_suit, win_seat, hearts_broken, trick_number,
                        scores, seat)


def _ply_probs_src(fh, hands, seat, played_mask, trick_cards, trick_seats,
                   trick_len, hearts_broken, trick_number, scores, legal,
                   W1, b1, W2, b2, W3, b3):
    """The opponent's masked choice distribution at this ply -> float64[52]."""
    f = _ply_features(fh, hands, seat, played_mask, trick_cards,
                      trick_seats, trick_len, hearts_broken, trick_number,
                      scores)
    return _probs_k(W1, b1, W2, b2, W3, b3, f, legal)


_ply_features = njit(cache=True)(_ply_features_src) if HAVE_NUMBA \
    else _ply_features_src
_ply_probs = njit(cache=True)(_ply_probs_src) if HAVE_NUMBA else _ply_probs_src


def _run_profiled_src(hands, to_play, played_mask, trick_cards, trick_seats,
                      trick_len, hearts_broken, trick_number, scores,
                      our_seat, stop_seat, mode, W1, b1, W2, b2, W3, b3,
                      out_cards, out_seats):
    """`kernel._run` with profiler-sampled opponents; same return contract.

    Returns (status, to_play, played_mask, trick_len, hearts_broken,
    trick_number, n_plays); status 1 = stopped before `stop_seat`'s first
    >1-legal decision, 0 = hand over.
    """
    n = 0
    fh = np.zeros(4, dtype=np.int64)
    while True:
        if trick_len == 0:
            over = True
            for s in range(4):
                if hands[s] != 0:
                    over = False
                    break
            if over:
                return (0, to_play, played_mask, trick_len, hearts_broken,
                        trick_number, n)
        led_suit, win_rank, win_seat = _trick_head(trick_cards, trick_seats,
                                                   trick_len)
        seat = to_play
        hand = hands[seat]
        legal = _legal(hand, led_suit, trick_len, hearts_broken, trick_number)
        if seat == stop_seat and _popcount(legal) > 1:
            return (1, to_play, played_mask, trick_len, hearts_broken,
                    trick_number, n)
        if seat == our_seat or _popcount(legal) == 1:
            # Our seat keeps the heuristic. A forced move is also taken
            # directly: the masked softmax is exactly 1.0 there, so querying
            # the net would cost a forward pass to learn nothing -- and the
            # profiler was never trained on forced plies (Task 2 emitted
            # n_legal > 1 rows only).
            card = _choose(hand, legal, led_suit, win_rank, trick_len,
                           played_mask | hand)
        else:
            if mode == MODE_PROFILER:
                probs = _ply_probs(fh, hands, seat, played_mask,
                                   trick_cards, trick_seats, trick_len,
                                   hearts_broken, trick_number, scores,
                                   legal, W1, b1, W2, b2, W3, b3)
            else:
                probs = np.zeros(52, dtype=np.float64)
                probs[_choose(hand, legal, led_suit, win_rank, trick_len,
                              played_mask | hand)] = 1.0
            # The cumulative draw is INLINE, not a helper. A helper would
            # have to be the njit build under NO_JIT too (a plain-Python
            # global cannot be called from a compiled parent), and numba's
            # generator is a DIFFERENT stream from numpy's global one -- so
            # the fallback would sample off a generator `seed_playouts` never
            # seeded. Inlining keeps one source and one stream per mode.
            u = np.random.random()
            acc = 0.0
            card = -1
            for c in range(52):
                if (legal >> c) & 1:
                    acc += probs[c]
                    card = c
                    if u < acc:
                        break
        bit = _ONE << card
        hands[seat] = hand & ~bit
        played_mask |= bit
        if card // 13 == 3:
            hearts_broken = True
        trick_cards[trick_len] = card
        trick_seats[trick_len] = seat
        trick_len += 1
        out_cards[n] = card
        out_seats[n] = seat
        n += 1
        if trick_len == 4:
            _l, _r, win_seat = _trick_head(trick_cards, trick_seats, trick_len)
            pts = 0
            for i in range(4):
                c = trick_cards[i]
                if c // 13 == 3:
                    pts += 1
                elif c == _QS:
                    pts += 13
            scores[win_seat] += pts
            trick_len = 0
            trick_number += 1
            to_play = win_seat
        else:
            to_play = (seat + 1) % 4


def _seed_src(seed):
    np.random.seed(seed)


_run_profiled = njit(cache=True)(_run_profiled_src) if HAVE_NUMBA \
    else _run_profiled_src
_seed = njit(cache=True)(_seed_src) if HAVE_NUMBA else _seed_src


def _to_end_src(hands, to_play, played_mask, trick_cards, trick_seats,
                trick_len, hearts_broken, trick_number, scores, our_seat,
                mode, W1, b1, W2, b2, W3, b3, out_cards, out_seats):
    return _run_profiled_src(hands, to_play, played_mask, trick_cards,
                             trick_seats, trick_len, hearts_broken,
                             trick_number, scores, our_seat, -1, mode,
                             W1, b1, W2, b2, W3, b3, out_cards, out_seats)


def _until_src(hands, to_play, played_mask, trick_cards, trick_seats,
               trick_len, hearts_broken, trick_number, scores, our_seat,
               stop_seat, mode, W1, b1, W2, b2, W3, b3, out_cards, out_seats):
    return _run_profiled_src(hands, to_play, played_mask, trick_cards,
                             trick_seats, trick_len, hearts_broken,
                             trick_number, scores, our_seat, stop_seat, mode,
                             W1, b1, W2, b2, W3, b3, out_cards, out_seats)


if HAVE_NUMBA:
    def _to_end_njit_src(hands, to_play, played_mask, trick_cards, trick_seats,
                         trick_len, hearts_broken, trick_number, scores,
                         our_seat, mode, W1, b1, W2, b2, W3, b3, out_cards,
                         out_seats):
        return _run_profiled(hands, to_play, played_mask, trick_cards,
                             trick_seats, trick_len, hearts_broken,
                             trick_number, scores, our_seat, -1, mode,
                             W1, b1, W2, b2, W3, b3, out_cards, out_seats)

    def _until_njit_src(hands, to_play, played_mask, trick_cards, trick_seats,
                        trick_len, hearts_broken, trick_number, scores,
                        our_seat, stop_seat, mode, W1, b1, W2, b2, W3, b3,
                        out_cards, out_seats):
        return _run_profiled(hands, to_play, played_mask, trick_cards,
                             trick_seats, trick_len, hearts_broken,
                             trick_number, scores, our_seat, stop_seat, mode,
                             W1, b1, W2, b2, W3, b3, out_cards, out_seats)

    playout_to_end_profiled = njit(cache=True)(_to_end_njit_src)
    playout_until_decision_profiled = njit(cache=True)(_until_njit_src)
else:  # pragma: no cover
    playout_to_end_profiled = _to_end_src
    playout_until_decision_profiled = _until_src


# --------------------------------------------------------------------------
# NO_JIT stream isolation (see module docstring)
# --------------------------------------------------------------------------
_PY_STATE = None


def _enter():
    """Swap numpy's global RNG state for this module's private one."""
    global _PY_STATE
    outer = np.random.get_state()
    if _PY_STATE is not None:
        np.random.set_state(_PY_STATE)
    return outer


def _exit(outer):
    global _PY_STATE
    _PY_STATE = np.random.get_state()
    np.random.set_state(outer)


def seed_playouts(seed) -> None:
    """Seed the playout sampler. Called ONCE PER DECISION by the player."""
    s = int(seed) % (2 ** 32)
    if jit_enabled():
        _seed(np.int64(s))
        return
    outer = _enter()
    try:
        _seed_src(s)
    finally:
        _exit(outer)


# --------------------------------------------------------------------------
# GameState adapters (mirror kernel.run_playout / run_playout_until_decision)
# --------------------------------------------------------------------------
def _call(fn_jit, fn_py, args):
    if jit_enabled():
        return fn_jit(*args)
    outer = _enter()
    try:
        return fn_py(*args)
    finally:
        _exit(outer)


def run_playout_profiled(state, our_seat: int, weights,
                         mode: int = MODE_PROFILER) -> None:
    """Kernel-equivalent of `kernel.run_playout` with profiled opponents."""
    from . import kernel  # local: reuse the array adapters, no cycle worry

    hands, scores, tc, ts, tl, played = kernel._to_arrays(state)
    out_cards = np.zeros(52, dtype=np.int64)
    out_seats = np.zeros(52, dtype=np.int64)
    args = (hands, np.int64(state.to_play), played, tc, ts, np.int64(tl),
            bool(state.hearts_broken), np.int64(state.trick_number), scores,
            np.int64(our_seat), np.int64(mode)) + tuple(weights) + \
        (out_cards, out_seats)
    (_status, to_play, _pm, tl, hb, tn, n) = _call(
        playout_to_end_profiled, _to_end_src, args)
    kernel._write_back(state, hands, scores, tc, ts, tl, to_play, hb, tn,
                       out_cards, out_seats, n)


def run_playout_until_decision_profiled(state, our_seat: int, stop_seat: int,
                                        weights,
                                        mode: int = MODE_PROFILER) -> bool:
    """`kernel.run_playout_until_decision` with profiled opponents.

    Scratch buffers are LOCAL, not module-level: the caller runs a Python
    inner search between this call and the finishing one, so a shared buffer
    would have to survive across it.
    """
    from . import kernel

    hands, scores, tc, ts, tl, played = kernel._to_arrays(state)
    out_cards = np.zeros(52, dtype=np.int64)
    out_seats = np.zeros(52, dtype=np.int64)
    args = (hands, np.int64(state.to_play), played, tc, ts, np.int64(tl),
            bool(state.hearts_broken), np.int64(state.trick_number), scores,
            np.int64(our_seat), np.int64(stop_seat), np.int64(mode)) + \
        tuple(weights) + (out_cards, out_seats)
    (status, to_play, _pm, tl, hb, tn, n) = _call(
        playout_until_decision_profiled, _until_src, args)
    kernel._write_back(state, hands, scores, tc, ts, tl, to_play, hb, tn,
                       out_cards, out_seats, n)
    return status == 1


def ply_probs_for_state(state, seat: int, legal: int, weights):
    """TEST PROBE: the in-kernel features and probs for one (state, seat).

    Exposes exactly the two values `_run_profiled` computes at an opponent
    ply, so `tests/test_profiled_playout.py` pins the code that runs rather
    than a re-implementation of it.
    """
    from . import kernel

    hands, scores, tc, ts, tl, _played = kernel._to_arrays(state)
    played = np.int64(0)
    for _s, c in state.history:
        played |= np.int64(1) << np.int64(c)
    for _s, c in state.current_trick:
        played |= np.int64(1) << np.int64(c)
    fh = np.zeros(4, dtype=np.int64)
    fa = (fh, hands, np.int64(seat), played, tc, ts, np.int64(tl),
          bool(state.hearts_broken), np.int64(state.trick_number), scores)
    if jit_enabled():
        feats = _ply_features(*fa)
        probs = _ply_probs(*(fa + (np.int64(legal),) + tuple(weights)))
    else:
        feats = _ply_features_src(*fa)
        probs = _ply_probs_src(*(fa + (np.int64(legal),) + tuple(weights)))
    assert feats.shape[0] == NF
    return feats, probs
