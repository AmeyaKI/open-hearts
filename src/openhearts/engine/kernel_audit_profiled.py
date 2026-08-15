"""Phase 5.5b: the PROFILER AUDIT, fused into one numba kernel.

`search.profiled.profiler_world_logweight` replays the observed game inside a
candidate world in PYTHON, builds an observer-legal feature row per opponent
multi-legal ply, batches those rows through the profiler net, and sums
`log p(observed card)`. Task 4 recorded that deviation from the plan ("an njit
replay would have to re-port `featurize` + the audit loop into one compiled
function") and pre-registered the condition for revisiting it: "If Task 6's
playouts need more, that is the moment to compile the loop, with a profile in
hand." Task 6 needed more; this module is that compilation.

WHAT IS FUSED, AND WHY EACH PIECE IS EXACT BY CONSTRUCTION
----------------------------------------------------------
Per candidate world, ONE kernel call does replay + featurize + net forward +
log accumulation. Nothing here is a re-implementation of the numerics:

* REPLAY. The play-application block (hand clear, played_mask, hearts_broken,
  trick rollover, `scores[win_seat] += pts`, `to_play`) is copied VERBATIM from
  `kernel_profiled._run_profiled_src`, which in turn mirrors
  `engine.state.GameState.play`. Legality comes from `kernel._legal`, whose
  equivalence to the Python `game.legal_moves` is INHERITED from Phase 3.5's
  `tests/test_jit_audit.py` pin (`audit_world` replays with the same `_legal`
  and is pinned bitwise against `belief.weighted`'s Python replay) -- this
  module does not newly assume it.
* FEATURES. `kernel_profiled._ply_features`, unchanged and un-ported. Phase
  5's `tests/test_profiled_playout.py` pins it EXACTLY against
  `opponent.obsfeat.observer_features` over >=1000 (state, seat) pairs.
* NET FORWARD. `opponent.infer._profiler_probs_njit`, the SAME compiled
  function `ProfilerLikelihood.batch_probs` reaches through
  `profiler_probs_batch`. Same numerics because it is the same code, not
  because it was checked. Deliberately NOT specialized to the one output card
  we need: a single-output forward would be much faster and would forfeit
  exactly this argument. If a profile later says the forward dominates, that
  is a separate task with the profile in hand.
* REDUCTION ORDER. The Python path accumulates `total += np.log(p)` in a plain
  sequential loop over plies (`profiled.py`) -- there is no `np.sum`, so there
  is no pairwise reduction to emulate. The kernel runs the identical
  sequential loop in the identical ply order, and `profiled.py`'s summation is
  BYTE-UNCHANGED.

THE `np.log` QUESTION, DECIDED BY MEASUREMENT NOT BY FAITH
----------------------------------------------------------
The one operation that is not literally shared source is the scalar log:
numpy's `np.log` on a float64 and numba's compiled `np.log` are different code
paths that need not agree to the last ULP on every platform. So the kernel
returns BOTH: the in-kernel `logsum`, and `out_probs`, the gathered
`p(observed card)` per counted ply in ply order. `LOG_IN_KERNEL = False` makes
the adapter re-run the Python `total += np.log(p)` loop over those bitwise-
identical probabilities, which is exact by construction at the cost of ~30
scalar logs per world (a cost the Python path already pays today, so the
fallback is free relative to the baseline being compared against).

On this machine the in-kernel log was measured bitwise-identical to numpy's
over 900k float64s spanning 1e-12..1.0 plus denormal and near-1.0 edges, and
gate 1 re-confirms it over the real audit corpus, so `LOG_IN_KERNEL` defaults
to True. Flip it if a future platform's gate 1 disagrees.

STATUS CODES, because -1.0 is a perfectly good log-weight
---------------------------------------------------------
`kernel.audit_world` can use -1.0 as its desync sentinel (weights are >= 0).
Log-weights are signed, so this kernel returns an explicit status instead:
0 ok, 1 observed card illegal in this world (-inf), 2 replay desync,
3 non-positive probability on a legal card. The adapter turns 2 and 3 into the
same loud assertions the Python path raises.
"""
import numpy as np

from .features import NF
from .kernel import (HAVE_NUMBA, _legal, _popcount, _trick_head, jit_enabled,
                     njit)
# `_ply_features` is the njit object whenever numba is installed, and is
# called as such from BOTH builds -- the same thing `kernel_profiled`'s own
# NO_JIT fallback (`_until_src` -> `_run_profiled_src` -> `_ply_probs`) does.
# One featurizer, one set of numerics, no fallback-only copy to drift.
from .kernel_profiled import _ply_features

if HAVE_NUMBA:
    from ..opponent.infer import _profiler_probs_njit as _probs_k
else:  # pragma: no cover - only without numba installed
    from ..opponent.infer import _profiler_probs_py as _probs_k

#: Use the compiled `np.log` (True) or re-run numpy's in Python over the
#: gathered probabilities (False). See the module docstring.
LOG_IN_KERNEL = True

_ONE = np.int64(1)
_QS = 36

ST_OK = 0
ST_ILLEGAL = 1
ST_DESYNC = 2
ST_BAD_PROB = 3


def _audit_profiled_src(hands, plays_cards, plays_seats, observer,
                        include_forced, n_in, params,
                        W1, b1, W2, b2, W3, b3, out_probs):
    """Replay one candidate world; return (status, n_counted, logsum).

    `hands` int64[4] are the four ORIGINAL hands (mutated in place -- the
    adapter hands over a private array, exactly as `kernel.audit_world_weight`
    does). `params` is float64[4, pd]: the per-seat personality block appended
    to each feature row for the CONDITIONED/ORACLE variant, `pd == 0` for the
    GENERIC model. `out_probs` float64[52] receives p(observed card) at each
    COUNTED ply, in ply order.
    """
    n = plays_cards.shape[0]
    trick_cards = np.zeros(4, dtype=np.int64)
    trick_seats = np.zeros(4, dtype=np.int64)
    scores = np.zeros(4, dtype=np.int64)
    fh = np.zeros(4, dtype=np.int64)
    row = np.zeros(n_in, dtype=np.float64)
    pd = params.shape[1]
    trick_len = 0
    trick_number = 0
    hearts_broken = False
    played_mask = np.int64(0)
    to_play = plays_seats[0]
    total = 0.0
    n_counted = 0

    for k in range(n):
        seat = plays_seats[k]
        card = plays_cards[k]
        if seat != to_play:
            return (ST_DESYNC, n_counted, total)
        led_suit, _win_rank, _win_seat = _trick_head(trick_cards, trick_seats,
                                                     trick_len)
        hand = hands[seat]
        legal = _legal(hand, led_suit, trick_len, hearts_broken, trick_number)
        if ((legal >> card) & 1) == 0:
            return (ST_ILLEGAL, n_counted, total)
        if seat != observer:
            if _popcount(legal) > 1 or include_forced:
                f = _ply_features(fh, hands, seat, played_mask, trick_cards,
                                  trick_seats, trick_len, hearts_broken,
                                  trick_number, scores)
                # `ProfilerLikelihood.row`: the GENERIC model consumes the
                # NF-wide row as is; the CONDITIONED one appends this seat's
                # personality block. Copying is exact either way.
                for j in range(NF):
                    row[j] = f[j]
                for j in range(pd):
                    row[NF + j] = params[seat, j]
                probs = _probs_k(W1, b1, W2, b2, W3, b3, row, legal)
                p = probs[card]
                if not (p > 0.0):
                    return (ST_BAD_PROB, n_counted, total)
                out_probs[n_counted] = p
                n_counted += 1
                total += np.log(p)

        # ---- apply the play (verbatim from kernel_profiled._run_profiled_src)
        bit = _ONE << card
        hands[seat] = hand & ~bit
        played_mask |= bit
        if card // 13 == 3:
            hearts_broken = True
        trick_cards[trick_len] = card
        trick_seats[trick_len] = seat
        trick_len += 1
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
    return (ST_OK, n_counted, total)


_audit_profiled = njit(cache=True)(_audit_profiled_src) if HAVE_NUMBA \
    else _audit_profiled_src


def audit_world_logweight(hands, all_plays, observer, weights, n_in,
                          params=None, include_forced: bool = False):
    """Adapter: fused profiler audit for one reconstructed candidate world.

    `hands` are the four ORIGINAL hands and `all_plays` the observed
    ((seat, card), ...) sequence, both from
    `belief.weighted._reconstruct_original_hands` -- every structural assert
    stays on the Python side of this boundary, exactly as for
    `kernel.audit_world_weight`.

    Returns the log-weight, or `-inf` when the observed card is illegal in
    this world.
    """
    if not all_plays:
        return 0.0
    arr_hands = np.array(hands, dtype=np.int64)
    seats = np.array([s for s, _c in all_plays], dtype=np.int64)
    cs = np.array([c for _s, c in all_plays], dtype=np.int64)
    if params is None:
        params = np.zeros((4, 0), dtype=np.float64)
    out_probs = np.zeros(52, dtype=np.float64)
    fn = _audit_profiled if jit_enabled() else _audit_profiled_src
    status, n_counted, logsum = fn(
        arr_hands, cs, seats, np.int64(observer), bool(include_forced),
        np.int64(n_in), np.ascontiguousarray(params, dtype=np.float64),
        *weights, out_probs)
    status = int(status)
    if status == ST_ILLEGAL:
        return -np.inf
    if status == ST_DESYNC:
        raise AssertionError(
            "replay desync: observed seat order does not match the engine's "
            "turn order in this world")
    if status == ST_BAD_PROB:
        # The Python path's assertion, verbatim in meaning: a masked softmax
        # over a mask that PROVABLY contains the card is strictly positive
        # there. If this fires, the likelihood -- not the world -- is broken.
        raise AssertionError(
            "profiler assigned non-positive probability to a LEGAL observed "
            "card; masked softmax must never be exactly zero on the mask")
    if LOG_IN_KERNEL:
        total = float(logsum)
    else:
        total = 0.0
        for i in range(int(n_counted)):
            total += np.log(out_probs[i])
        total = float(total)
    assert np.isfinite(total), f"bad log weight {total}"
    return total


def audit_world_probs(hands, all_plays, observer, weights, n_in, params=None,
                      include_forced: bool = False):
    """TEST PROBE: the per-ply p(observed card) the kernel gathers, in order.

    Lets gate 1 compare the two log-reduction variants (in-kernel vs numpy)
    over bitwise-identical inputs, and lets a test pin the PROBABILITIES
    against the Python path's `batch_probs` output independently of the sum.
    Returns `None` for a world that replays illegally.
    """
    if not all_plays:
        return np.zeros(0, dtype=np.float64)
    arr_hands = np.array(hands, dtype=np.int64)
    seats = np.array([s for s, _c in all_plays], dtype=np.int64)
    cs = np.array([c for _s, c in all_plays], dtype=np.int64)
    if params is None:
        params = np.zeros((4, 0), dtype=np.float64)
    out_probs = np.zeros(52, dtype=np.float64)
    fn = _audit_profiled if jit_enabled() else _audit_profiled_src
    status, n_counted, _logsum = fn(
        arr_hands, cs, seats, np.int64(observer), bool(include_forced),
        np.int64(n_in), np.ascontiguousarray(params, dtype=np.float64),
        *weights, out_probs)
    if int(status) == ST_ILLEGAL:
        return None
    assert int(status) == ST_OK, f"kernel audit status {int(status)}"
    return out_probs[:int(n_counted)].copy()
