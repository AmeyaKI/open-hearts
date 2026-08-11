"""Value-truncated honest search (Phase 4, Task 5).

`ValueSearchPlayer` is `HonestSearchPlayer` with one extra knob, `horizon`,
controlling how an imagined world is SCORED after a candidate play:

* ``horizon=None`` (or ``float('inf')``) -- full heuristic playout to the end
  of the hand. This path is `HonestSearchPlayer` itself, called through
  ``super()``: same rng stream, same arrangements, same chosen card.
  `tests/test_value_search.py` pins that on 200+ decisions.
* ``horizon=k`` (k >= 0) -- play forward with the heuristic kernel until `k`
  tricks have COMPLETED counting from the position immediately after the
  candidate play, then stop and score the stopped position with the learned
  value net.

`honest.py` is not modified; this module only adds.

What "k completed tricks" means precisely
-----------------------------------------
Let ``tn0`` be ``trick_number`` immediately AFTER the candidate card is played
into the imagined world. The playout runs while ``trick_number < tn0 + k``,
checked before every card. Consequences, stated so nobody has to re-derive
them:

* ``k = 0`` stops immediately -- no playout at all, and the position handed to
  the net may be mid-trick (the candidate's own trick unfinished).
* If the candidate card COMPLETED a trick, that completion is already in
  ``tn0``; ``k`` counts tricks after it.
* Otherwise the first thing ``k >= 1`` buys is finishing the candidate's own
  trick -- so "one completed trick" can be as little as one to three more
  cards. That is the honest reading of a trick-counted horizon in a game whose
  decisions land mid-trick; the alternative (count only whole tricks started
  after the candidate) makes ``k=1`` cost between 4 and 7 cards depending on
  seat position, which is worse, not better.

The scoring accounting (the thing to get right)
-----------------------------------------------
`HonestSearchPlayer` scores a world as ``state.scores[seat] - base_score``
after a FULL playout, where ``base_score = view.scores[view.seat]`` at the
real decision point: points our seat takes from here to the end of the hand.
The truncated path must estimate the same quantity, so it is a sum of two
parts:

    score(world) = (scores[seat] - base_score)     # ACCRUED: points our seat
                                                   # actually took in the
                                                   # imagined world between
                                                   # the decision point and
                                                   # the stopping position
                 + v[0]                            # ESTIMATED: the net's
                                                   # remaining points for the
                                                   # evaluated seat from the
                                                   # stopping position on

``v = value_forward(..., featurize(stopped position, seat=our seat))``, and
index 0 is the evaluated seat because the featurizer rotates the position to
it (features.py layout v1). The net's four outputs are remaining-points targets
trained exactly on "points this seat takes from this ply onward", which is why
the two parts add without a scale factor. Nothing is clamped: the net can
return a small negative number and that is used as-is, because clamping would
bias the comparison between candidates asymmetrically.

TERMINAL POSITIONS ARE NEVER ESTIMATED. If the imagined hand ends before the
horizon, the run stops with the hand over and the score is the accrued points
alone -- i.e. exactly the full-playout number, bitwise. So ``horizon=k`` on a
position with <= k tricks left IS `HonestSearchPlayer`. The `value_calls`
counter exposes this (it must be 0 in that regime) and a test pins both.

Both stages
-----------
The truncation applies at the OUTER candidate evaluation and at the INNER
re-determinization: the inner player is `_ValueInnerSearchPlayer`, a
`SearchPlayer` subclass whose world scoring is the same accrued+estimated sum
with its own fresh ``tn0`` (the inner decision point), never a leftover budget
from the outer call. Each imagined decision therefore gets the same depth of
imagination, which is what makes the horizon a single interpretable knob.

CONSEQUENCE AT ``horizon=0``, stated loudly because it is easy to miss: with
no plies played after the candidate, our seat's next imagined decision is
never reached, so the honest re-determinization never fires and `n_inner` is
silently irrelevant. ``horizon=0`` is therefore not "honest search with a
cheap evaluator" -- it is Phase-1 determinized search with a cheap evaluator.
Any row run at horizon 0 must be labelled that way.

Where the fusion boundary is (efficiency discipline, PHASE4_PLAN (d))
---------------------------------------------------------------------
``_eval_worlds`` is a single ``@njit`` entry that, for ONE candidate card,
takes the WHOLE batch of imagined worlds and does apply-card -> truncated
playout -> featurize -> value_forward -> score for every one of them without
returning to Python. Scoring a world never crosses the boundary per call.
That fused path serves:

* every INNER re-determinization (`_ValueInnerSearchPlayer`), which is the hot
  one -- it runs once per (outer world x outer candidate);
* the OUTER evaluation whenever ``n_inner == 0``.

When ``n_inner > 0`` the outer evaluation cannot be one call per batch, because
the interception in the middle of each world's playout IS a Python search. That
path keeps the per-world crossing structure `honest.py` already has today
(run-to-decision, Python inner choose, finish-and-score), so it is no worse
than the code it extends, and the inner search underneath it is fully fused.

``_run_horizon`` deliberately calls `kernel._legal` / `_choose` / `_trick_head`
rather than copying them: the heuristic POLICY has exactly one implementation
and cannot drift. Only the loop scaffolding (the horizon stop) is new, and
`test_run_horizon_kernel_matches_playout_to_end` pins it against
`kernel.playout_to_end` with the horizon disabled.

Python fallback (`OPENHEARTS_NO_JIT=1`) is an independent reference: GameState
+ `HeuristicPlayer` + the dispatching `featurize`/`value_forward` wrappers.
Both modes are tested to choose the same card.
"""
import os

import numpy as np

from openhearts.engine import cards, kernel
from openhearts.engine.features import NF, featurize
from openhearts.engine.kernel import HAVE_NUMBA, njit
from openhearts.search.decision import SearchPlayer, state_from_view
from openhearts.search.honest import HonestSearchPlayer
from openhearts.value.infer import load_weights, value_forward
from openhearts.value.infer import _value_forward_njit

# Resolved against the repo root, not the cwd: an experiment run from
# anywhere must load the same weights.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_WEIGHTS = os.path.join(_REPO_ROOT, "models", "value_v1.npz")

_ONE = np.int64(1)
_QS = 36


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------
def _apply_card_src(hands, to_play, played_mask, trick_cards, trick_seats,
                    trick_len, hearts_broken, trick_number, scores, card):
    """Play `card` for seat `to_play`; same bookkeeping as `kernel._run`.

    Returns (to_play, played_mask, trick_len, hearts_broken, trick_number).
    Legality is the caller's business (the caller took the card from the
    view's legal mask), exactly as in the existing kernel path.
    """
    seat = to_play
    bit = _ONE << card
    hands[seat] = hands[seat] & ~bit
    played_mask = played_mask | bit
    if card // 13 == 3:
        hearts_broken = True
    trick_cards[trick_len] = card
    trick_seats[trick_len] = seat
    trick_len += 1
    if trick_len == 4:
        _, _, win_seat = kernel._trick_head(trick_cards, trick_seats,
                                            trick_len)
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
    return to_play, played_mask, trick_len, hearts_broken, trick_number


_apply_card = njit(cache=True)(_apply_card_src) if HAVE_NUMBA \
    else _apply_card_src


def _run_horizon_src(hands, to_play, played_mask, trick_cards, trick_seats,
                     trick_len, hearts_broken, trick_number, scores,
                     stop_seat, stop_trick, out_cards, out_seats):
    """`kernel._run` plus a trick-count horizon.

    `stop_trick >= 0` stops (before playing) as soon as
    ``trick_number >= stop_trick``; `stop_trick < 0` disables the horizon, in
    which case this is `kernel.playout_to_end` / `playout_until_decision`
    exactly (pinned by test).

    Returns (status, to_play, played_mask, trick_len, hearts_broken,
    trick_number, n_plays); status 0 = hand over, 1 = stopped at `stop_seat`'s
    decision, 2 = stopped at the horizon. The hand-over check comes FIRST, so
    a finished hand is never reported as a horizon stop -- that ordering is
    what makes terminal positions score by actual points.
    """
    n = 0
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
        if stop_trick >= 0 and trick_number >= stop_trick:
            return (2, to_play, played_mask, trick_len, hearts_broken,
                    trick_number, n)
        led_suit, win_rank, win_seat = kernel._trick_head(
            trick_cards, trick_seats, trick_len)
        seat = to_play
        hand = hands[seat]
        legal = kernel._legal(hand, led_suit, trick_len, hearts_broken,
                              trick_number)
        if seat == stop_seat and kernel._popcount(legal) > 1:
            return (1, to_play, played_mask, trick_len, hearts_broken,
                    trick_number, n)
        card = kernel._choose(hand, legal, led_suit, win_rank, trick_len,
                              played_mask | hand)
        out_cards[n] = card
        out_seats[n] = seat
        n += 1
        (to_play, played_mask, trick_len, hearts_broken,
         trick_number) = _apply_card(hands, seat, played_mask, trick_cards,
                                     trick_seats, trick_len, hearts_broken,
                                     trick_number, scores, card)


_run_horizon = njit(cache=True)(_run_horizon_src) if HAVE_NUMBA \
    else _run_horizon_src


def _score_position_njit_src(hands, played_mask, trick_cards, trick_seats,
                             trick_len, hearts_broken, trick_number, scores,
                             seat, W1, b1, W2, b2, W3, b3):
    """Net estimate of `seat`'s remaining points at a stopped position."""
    led_suit, _win_rank, win_seat = kernel._trick_head(trick_cards,
                                                       trick_seats, trick_len)
    f = kernel_featurize_njit(hands, played_mask, trick_cards, trick_seats,
                              trick_len, led_suit, win_seat, hearts_broken,
                              trick_number, scores, seat)
    v = _value_forward_njit(W1, b1, W2, b2, W3, b3, f)
    return v[0]


def _eval_worlds_njit_src(worlds, to_play0, played0, tc0, ts0, tl0, hb0, tn0,
                          scores0, card, seat, horizon, base_score,
                          W1, b1, W2, b2, W3, b3, out, flags):
    """FUSED: score every world in `worlds` for one candidate `card`.

    `worlds` is int64[n, 4] of full imagined hands (our seat included) at the
    decision point; the rest is the shared position. `out[w]` receives the
    world's points-from-here for `seat` (accrued + estimated) and `flags[w]`
    is 1 iff the net was consulted. `horizon < 0` means full playout.

    One Python->kernel crossing per (candidate, world batch).
    """
    n = worlds.shape[0]
    out_cards = np.zeros(52, dtype=np.int64)
    out_seats = np.zeros(52, dtype=np.int64)
    for w in range(n):
        hands = worlds[w].copy()
        tc = tc0.copy()
        ts = ts0.copy()
        scores = scores0.copy()
        (to_play, played, tl, hb, tn) = _apply_card(
            hands, to_play0, played0, tc, ts, tl0, hb0, tn0, scores, card)
        stop_trick = -1
        if horizon >= 0:
            stop_trick = tn + horizon
        (status, _tp, played, tl, hb, tn, _n) = _run_horizon(
            hands, to_play, played, tc, ts, tl, hb, tn, scores, -1,
            stop_trick, out_cards, out_seats)
        v = 0.0
        if status != 0:
            v = _score_position_njit(hands, played, tc, ts, tl, hb, tn,
                                     scores, seat, W1, b1, W2, b2, W3, b3)
            flags[w] = 1
        out[w] = (scores[seat] - base_score) + v


if HAVE_NUMBA:
    from openhearts.engine.features import _featurize_njit as \
        kernel_featurize_njit
    _score_position_njit = njit(cache=True)(_score_position_njit_src)
    _eval_worlds_njit = njit(cache=True)(_eval_worlds_njit_src)
else:  # pragma: no cover - exercised only without numba installed
    from openhearts.engine.features import _featurize_py as \
        kernel_featurize_njit
    _score_position_njit = _score_position_njit_src
    _eval_worlds_njit = _eval_worlds_njit_src


# --------------------------------------------------------------------------
# Python reference path (OPENHEARTS_NO_JIT=1)
# --------------------------------------------------------------------------
def _state_arrays(state):
    """kernel._to_arrays, plus led_suit/win_seat for the featurizer."""
    hands, scores, tc, ts, tl, played = kernel._to_arrays(state)
    led_suit, _wr, win_seat = kernel._trick_head(tc, ts, tl) if tl else \
        (-1, -1, -1)
    return hands, scores, tc, ts, tl, played, led_suit, win_seat


class _ValueScorer:
    """Shared world-scoring logic for the outer and inner players."""

    def __init__(self, horizon, weights, heuristic):
        self.horizon = horizon          # int >= 0 (None handled by callers)
        self.weights = weights
        self._heuristic = heuristic
        self.value_calls = 0

    # ---------------------------------------------------------------- python
    def _playout_horizon_python(self, state, stop_trick, our_seat=-1,
                                inner=None):
        """Heuristic playout stopping at `stop_trick` completed tricks.

        `inner` (when not None) is the one honest re-determinization: at
        `our_seat`'s first real decision the move comes from `inner.choose`
        instead of the heuristic, exactly as `HonestSearchPlayer` does.
        """
        intercepted = inner is None
        while not state.is_over():
            if stop_trick >= 0 and state.trick_number >= stop_trick:
                return False
            seat = state.to_play
            view = state.view_for(seat)
            if not intercepted and seat == our_seat:
                if len(cards.cards_in(view.legal_moves)) > 1:
                    state.play(inner.choose(view))
                    intercepted = True
                    continue
            state.play(self._heuristic.choose(view))
        return True

    def _estimate_python(self, state, seat):
        hands, scores, tc, ts, tl, played, led_suit, win_seat = \
            _state_arrays(state)
        f = featurize(hands, played, tc, ts, tl, led_suit, win_seat,
                      state.hearts_broken, state.trick_number, scores, seat)
        assert f.shape[0] == NF
        v = value_forward(*self.weights, f)
        self.value_calls += 1
        return float(v[0])

    def _score_world_python(self, view, hands, card, inner):
        """One world, one candidate: accrued + estimated (see module doc)."""
        state = state_from_view(view, hands)
        base_score = view.scores[view.seat]
        state.play(card)
        stop_trick = -1 if self.horizon is None else \
            state.trick_number + self.horizon
        over = self._playout_horizon_python(state, stop_trick, view.seat,
                                            inner)
        accrued = state.scores[view.seat] - base_score
        if over:
            return float(accrued)
        return accrued + self._estimate_python(state, view.seat)

    # ------------------------------------------------------------------ jit
    def _score_worlds_fused(self, view, arrangements, card):
        """One kernel call for the whole world batch (no inner search)."""
        n = len(arrangements)
        worlds = np.empty((n, 4), dtype=np.int64)
        for i, hands in enumerate(arrangements):
            worlds[i, view.seat] = view.hand
            for j in range(3):
                worlds[i, (view.seat + 1 + j) % 4] = hands[j]
        base = _state_arrays_from_view(view)
        out = np.zeros(n, dtype=np.float64)
        flags = np.zeros(n, dtype=np.int64)
        horizon = -1 if self.horizon is None else self.horizon
        _eval_worlds_njit(worlds, np.int64(view.seat), base["played"],
                          base["tc"], base["ts"], np.int64(base["tl"]),
                          bool(view.hearts_broken),
                          np.int64(view.trick_number), base["scores"],
                          np.int64(card), np.int64(view.seat),
                          np.int64(horizon),
                          np.int64(view.scores[view.seat]),
                          *self.weights, out, flags)
        self.value_calls += int(flags.sum())
        return float(out.mean())


def _state_arrays_from_view(view):
    """Kernel arrays for the position a view describes (our seat's cards are
    supplied per world, so `hands` is not built here)."""
    played = 0
    for _seat, card in view.history:
        played |= 1 << card
    tc = np.zeros(4, dtype=np.int64)
    ts = np.zeros(4, dtype=np.int64)
    for i, (seat, card) in enumerate(view.current_trick):
        tc[i] = card
        ts[i] = seat
        played |= 1 << card
    return {"played": np.int64(played), "tc": tc, "ts": ts,
            "tl": len(view.current_trick),
            "scores": np.array(view.scores, dtype=np.int64)}


# --------------------------------------------------------------------------
# players
# --------------------------------------------------------------------------
class _ValueInnerSearchPlayer(SearchPlayer):
    """Phase-1 `SearchPlayer` with value-truncated world scoring.

    Identical to `SearchPlayer` in every other respect -- same sampler, same
    rng consumption, same fallback rule -- so with `horizon=None` it would be
    `SearchPlayer` exactly (the outer player never builds it in that case; it
    builds a plain `SearchPlayer`, so the reduction is trivially exact).
    """

    def __init__(self, level, n_samples, rng, sampler_respects_voids,
                 jit_sampler, horizon, weights):
        super().__init__(level, n_samples, rng, sampler_respects_voids,
                         jit_sampler)
        self.scorer = _ValueScorer(horizon, weights, self._heuristic)

    def choose(self, view):
        legal = cards.cards_in(view.legal_moves)
        if len(legal) == 1:
            return legal[0]
        from openhearts.belief.table import BeliefTable
        table = BeliefTable.from_view(view, self.level)
        if not self.sampler_respects_voids:
            table = BeliefTable(table.probs, [set(), set(), set()],
                                table.hand_sizes, table.opponent_seats,
                                table.unseen_mask)
        arrangements = self._sample(table, self.n_samples)
        if len(arrangements) * 2 < self.n_samples:
            self.fallbacks += 1
            return self._heuristic.choose(view)

        best_card, best_avg = None, None
        use_jit = kernel.jit_enabled()
        for card in legal:
            if use_jit:
                avg = self.scorer._score_worlds_fused(view, arrangements, card)
            else:
                total = 0.0
                for hands in arrangements:
                    total += self.scorer._score_world_python(view, hands,
                                                             card, None)
                avg = total / len(arrangements)
            if best_avg is None or avg < best_avg:
                best_card, best_avg = card, avg
        return best_card


class ValueSearchPlayer(HonestSearchPlayer):
    """Honest search whose imagined worlds are scored by the value net after
    `horizon` completed tricks. See the module docstring for the accounting.

    `horizon=None` (the default) is `HonestSearchPlayer`, delegated to
    `super()` so the reduction is exact rather than merely intended.
    """

    def __init__(self, level, n_outer, n_inner, rng,
                 sampler_respects_voids=True, jit_sampler=True,
                 posterior_factory=None, horizon=None,
                 weights_path=DEFAULT_WEIGHTS):
        super().__init__(level, n_outer, n_inner, rng, sampler_respects_voids,
                         jit_sampler, posterior_factory)
        if horizon is not None and horizon == float("inf"):
            horizon = None
        if horizon is not None:
            horizon = int(horizon)
            assert horizon >= 0, "horizon must be >= 0 (or None for full)"
        self.horizon = horizon
        self.weights_path = weights_path
        # Loaded once, in the constructor; the arrays are passed straight into
        # the kernel, never re-read per decision.
        self.weights = load_weights(weights_path)
        self.scorer = _ValueScorer(horizon, self.weights, self._heuristic)
        if horizon is not None and n_inner > 0:
            # Replace honest's plain inner search with the truncated one. It
            # shares our rng, exactly as honest's does, so the stream is the
            # same shape.
            self._inner = _ValueInnerSearchPlayer(
                level, n_inner, rng, sampler_respects_voids, jit_sampler,
                horizon, self.weights)

    @property
    def value_calls(self) -> int:
        """Net evaluations across both stages (0 proves a run was terminal)."""
        inner = self._inner.scorer.value_calls \
            if isinstance(self._inner, _ValueInnerSearchPlayer) else 0
        return self.scorer.value_calls + inner

    def choose(self, view):
        if self.horizon is None:
            return super().choose(view)

        legal = cards.cards_in(view.legal_moves)
        if len(legal) == 1:
            return legal[0]

        arrangements = None
        if self.posterior_factory is not None:
            arrangements = self._posterior_worlds(view)

        if arrangements is None:
            from openhearts.belief.table import BeliefTable
            table = BeliefTable.from_view(view, self.level)
            if not self.sampler_respects_voids:
                table = BeliefTable(table.probs, [set(), set(), set()],
                                    table.hand_sizes, table.opponent_seats,
                                    table.unseen_mask)
            arrangements = self._sample(table, self.n_outer)
            if len(arrangements) * 2 < self.n_outer:
                self.fallbacks += 1
                return self._heuristic.choose(view)

        fused = kernel.jit_enabled() and self.n_inner <= 0
        best_card, best_avg = None, None
        for card in legal:
            if fused:
                avg = self.scorer._score_worlds_fused(view, arrangements, card)
            else:
                total = 0.0
                for hands in arrangements:
                    total += self._score_world(view, hands, card)
                avg = total / len(arrangements)
            if best_avg is None or avg < best_avg:
                best_card, best_avg = card, avg
        return best_card

    def _score_world(self, view, hands, card):
        """One imagined world, honest (interception) + truncated (net)."""
        inner = self._inner if self.n_inner > 0 else None
        if not kernel.jit_enabled():
            return self.scorer._score_world_python(view, hands, card, inner)
        return self._score_world_jit(view, hands, card, inner)

    def _score_world_jit(self, view, hands, card, inner):
        # Per-world crossings, unavoidable here: the interception in the
        # middle IS a Python search. Structure mirrors
        # HonestSearchPlayer._playout_jit, with the horizon added.
        state = state_from_view(view, hands)
        base_score = view.scores[view.seat]
        state.play(card)
        stop_trick = -1 if self.horizon is None else \
            state.trick_number + self.horizon
        status = _run_state(state, view.seat if inner is not None else -1,
                            stop_trick)
        if status == 1:
            state.play(inner.choose(state.view_for(view.seat)))
            status = _run_state(state, -1, stop_trick)
        accrued = state.scores[view.seat] - base_score
        if status == 0:
            return float(accrued)
        h, sc, tc, ts, tl, played, led, win = _state_arrays(state)
        v = _score_position_njit(h, played, tc, ts, tl, state.hearts_broken,
                                 state.trick_number, sc, view.seat,
                                 *self.weights)
        self.scorer.value_calls += 1
        return accrued + float(v)


def _run_state(state, stop_seat, stop_trick):
    """Adapter: advance `state` with `_run_horizon`; returns its status."""
    hands, scores, tc, ts, tl, played = kernel._to_arrays(state)
    out_cards, out_seats = np.zeros(52, np.int64), np.zeros(52, np.int64)
    (status, to_play, _pm, tl, hb, tn, n) = _run_horizon(
        hands, state.to_play, played, tc, ts, tl, state.hearts_broken,
        state.trick_number, scores, stop_seat, stop_trick, out_cards,
        out_seats)
    kernel._write_back(state, hands, scores, tc, ts, tl, to_play, hb, tn,
                       out_cards, out_seats, n)
    return status
