"""Numba-compiled playout kernel: an exact port of the heuristic playout.

Profiling one honest-search game showed ~90% of runtime inside the pure-Python
playout loop (`HeuristicPlayer.choose` + `GameState.play` + `view_for`, ~1M
calls each per game). This module reimplements that loop over int64 bitmasks
and small int arrays so numba can compile it.

The policy here MUST stay bit-for-bit identical to
`openhearts.players.heuristic.HeuristicPlayer` combined with
`openhearts.engine.game.legal_moves`/`trick_winner`/`trick_points` and
`GameState.play`. `tests/test_kernel_equivalence.py` pins that across
thousands of generated states; if you touch either side, run it.

Nothing here ever sees information a playout would not have: the kernel is
handed a fully-resolved *imagined* world (the same one the Python playout gets
from the sampler) and simply plays it out. The view boundary is enforced one
level up, where that world is sampled from a `PlayerView`.

Fallback: set `OPENHEARTS_NO_JIT=1` (or run without numba installed) and
callers use the original Python playout instead. Both paths stay tested.
"""
import os

import numpy as np

try:  # pragma: no cover - exercised by whichever env the suite runs in
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def wrap(fn):
            return fn
        return wrap(args[0]) if args and callable(args[0]) else wrap


_QS = 36
_SUIT_MASKS = np.array([((1 << 13) - 1) << (13 * s) for s in range(4)],
                       dtype=np.int64)
_HEARTS_MASK = np.int64(((1 << 13) - 1) << 39)
_POINTS_MASK = np.int64((((1 << 13) - 1) << 39) | (1 << _QS))
_ONE = np.int64(1)


# --------------------------------------------------------------------------
# bit helpers
# --------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _popcount(mask):
    n = 0
    m = mask
    while m:
        m &= m - 1
        n += 1
    return n


@njit(cache=True, inline="always")
def _lowest(mask):
    """Index of the lowest set bit (mask must be non-zero)."""
    low = mask & -mask
    c = 0
    while low > 1:
        low >>= 1
        c += 1
    return c


@njit(cache=True, inline="always")
def _highest(mask):
    """Index of the highest set bit (mask must be non-zero)."""
    c = -1
    m = mask
    while m:
        m >>= 1
        c += 1
    return c


# --------------------------------------------------------------------------
# engine rules
# --------------------------------------------------------------------------
@njit(cache=True)
def _legal(hand, led_suit, trick_len, hearts_broken, trick_number):
    """Port of engine.game.legal_moves."""
    if trick_len > 0:
        follow = hand & _SUIT_MASKS[led_suit]
        if follow:
            if trick_number == 0:
                safe = follow & ~_POINTS_MASK
                return safe if safe else follow
            return follow
        if trick_number == 0:
            safe = hand & ~_POINTS_MASK
            return safe if safe else hand
        return hand
    if trick_number == 0:
        return _ONE  # the two of clubs
    if hearts_broken:
        return hand
    nh = hand & ~_HEARTS_MASK
    return nh if nh else hand


@njit(cache=True)
def _trick_head(trick_cards, trick_seats, trick_len):
    """Led suit and current winner, derived only from the trick so far.

    Single source of truth: nothing else in the kernel tracks these, so a
    mid-trick entry and a mid-trick continuation cannot disagree.
    """
    if trick_len == 0:
        return -1, -1, -1
    led = trick_cards[0] // 13
    win_rank = trick_cards[0] % 13
    win_seat = trick_seats[0]
    for i in range(1, trick_len):
        c = trick_cards[i]
        if c // 13 == led and c % 13 > win_rank:
            win_rank = c % 13
            win_seat = trick_seats[i]
    return led, win_rank, win_seat


# --------------------------------------------------------------------------
# the heuristic policy (exact port of players/heuristic.py)
# --------------------------------------------------------------------------
@njit(cache=True)
def _choose(hand, legal, led_suit, win_rank, trick_len, seen):
    if _popcount(legal) == 1:
        return _lowest(legal)
    if trick_len == 0:
        return _lead(hand, legal, seen)
    if legal & _SUIT_MASKS[led_suit]:
        # `suit(legal[0]) == led` in the Python version: when the seat can
        # follow, every legal card is in the led suit, and when it is void
        # none are -- so the mask test is equivalent.
        return _follow(legal, win_rank)
    return _discard(legal)


@njit(cache=True)
def _lead(hand, legal, seen):
    # avoid leading spades while holding Qs with Ks or As still unseen,
    # but only when some other suit is available
    n_suits = 0
    for s in range(4):
        if legal & _SUIT_MASKS[s]:
            n_suits += 1
    skip_spades = False
    if (legal & _SUIT_MASKS[2]) and n_suits > 1 and ((hand >> _QS) & 1):
        if ((seen >> 37) & 1) == 0 or ((seen >> 38) & 1) == 0:
            skip_spades = True
    # min over suits by (length, suit index)
    best_suit = -1
    best_len = 99
    for s in range(4):
        if s == 2 and skip_spades:
            continue
        sm = legal & _SUIT_MASKS[s]
        if sm == 0:
            continue
        ln = _popcount(sm)
        if ln < best_len:  # strict: ties keep the lower suit index
            best_len = ln
            best_suit = s
    return _lowest(legal & _SUIT_MASKS[best_suit])


@njit(cache=True)
def _follow(legal, win_rank):
    # every legal card is in the led suit here, so card order == rank order
    losers = np.int64(0)
    winners = np.int64(0)
    m = legal
    while m:
        low = m & -m
        c = _lowest(low)
        if c % 13 < win_rank:
            losers |= low
        else:  # an equal rank is impossible: the card is already played
            winners |= low
        m ^= low
    if winners == 0:
        if (losers >> _QS) & 1:
            return _QS
        return _highest(losers)
    if losers:
        return _highest(losers)  # duck as high as possible, no Qs preference
    return _lowest(winners)


@njit(cache=True)
def _discard(legal):
    if (legal >> _QS) & 1:
        return _QS
    hm = legal & _HEARTS_MASK
    if hm:
        return _highest(hm)
    # longest suit, ties to the LOWEST suit index (the Python `(len, -s)` max)
    best_suit = -1
    best_len = -1
    for s in range(4):
        ln = _popcount(legal & _SUIT_MASKS[s])
        if ln > best_len:  # strict: ties keep the lower suit index
            best_len = ln
            best_suit = s
    return _highest(legal & _SUIT_MASKS[best_suit])


# --------------------------------------------------------------------------
# the playout loop
# --------------------------------------------------------------------------
@njit(cache=True)
def _run(hands, to_play, played_mask, trick_cards, trick_seats, trick_len,
         hearts_broken, trick_number, scores, stop_seat,
         out_cards, out_seats):
    """Play the hand out heuristically; mutate hands/scores/trick arrays.

    `stop_seat >= 0` stops (before playing) at that seat's first decision with
    more than one legal move -- the honest player's interception point. A
    forced move at `stop_seat` is played normally and does not stop the run.

    Returns (status, to_play, played_mask, trick_len, hearts_broken,
    trick_number, n_plays); status 1 = stopped at a decision, 0 = hand over.
    Every play is appended to out_cards/out_seats, so the caller can rebuild
    history exactly.
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
        led_suit, win_rank, win_seat = _trick_head(trick_cards, trick_seats,
                                                   trick_len)
        seat = to_play
        hand = hands[seat]
        legal = _legal(hand, led_suit, trick_len, hearts_broken, trick_number)
        if seat == stop_seat and _popcount(legal) > 1:
            return (1, to_play, played_mask, trick_len, hearts_broken,
                    trick_number, n)
        # "seen" for the leading rule is the seat's own hand plus every card
        # in completed tricks; mid-trick cards cannot matter because the rule
        # is only consulted when the trick is empty.
        card = _choose(hand, legal, led_suit, win_rank, trick_len,
                       played_mask | hand)
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
            _, _, win_seat = _trick_head(trick_cards, trick_seats, trick_len)
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


@njit(cache=True)
def playout_to_end(hands, to_play, played_mask, trick_cards, trick_seats,
                   trick_len, hearts_broken, trick_number, scores,
                   out_cards, out_seats):
    """Finish the hand with the heuristic in all four seats (mid-trick ok)."""
    return _run(hands, to_play, played_mask, trick_cards, trick_seats,
                trick_len, hearts_broken, trick_number, scores, -1,
                out_cards, out_seats)


@njit(cache=True)
def playout_until_decision(hands, to_play, played_mask, trick_cards,
                           trick_seats, trick_len, hearts_broken,
                           trick_number, scores, stop_seat,
                           out_cards, out_seats):
    """Play heuristically until `stop_seat` faces >1 legal move (return 1),
    stopping BEFORE that card is played, or until the hand ends (return 0)."""
    return _run(hands, to_play, played_mask, trick_cards, trick_seats,
                trick_len, hearts_broken, trick_number, scores, stop_seat,
                out_cards, out_seats)


# --------------------------------------------------------------------------
# GameState <-> arrays adapter
# --------------------------------------------------------------------------
_ENABLED = None


def jit_enabled() -> bool:
    """True when playouts should use the compiled kernel.

    Looked up once and cached; `reset_jit_enabled()` re-reads the environment
    (used by the JIT-vs-no-JIT equality test).
    """
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = HAVE_NUMBA and os.environ.get("OPENHEARTS_NO_JIT") != "1"
    return _ENABLED


def reset_jit_enabled() -> None:
    global _ENABLED
    _ENABLED = None


# Scratch buffers reused across playouts. Safe because every buffer is
# written and consumed inside a single adapter call, and playouts never nest.
_OUT_CARDS = np.zeros(52, dtype=np.int64)
_OUT_SEATS = np.zeros(52, dtype=np.int64)


def _to_arrays(state):
    hands = np.array(state.hands, dtype=np.int64)
    scores = np.array(state.scores, dtype=np.int64)
    trick_cards = np.zeros(4, dtype=np.int64)
    trick_seats = np.zeros(4, dtype=np.int64)
    played = 0
    for _seat, card in state.history:
        played |= 1 << card
    for i, (seat, card) in enumerate(state.current_trick):
        trick_cards[i] = card
        trick_seats[i] = seat
        played |= 1 << card
    return (hands, scores, trick_cards, trick_seats,
            len(state.current_trick), np.int64(played))


def _write_back(state, hands, scores, trick_cards, trick_seats, trick_len,
                to_play, hearts_broken, trick_number, out_cards, out_seats,
                n_plays):
    # history/current_trick are rebuilt exactly, so the state stays a valid
    # GameState (view_for included) after a kernel playout.
    n_before = len(state.current_trick)
    plays = list(zip(out_seats[:n_plays].tolist(),
                     out_cards[:n_plays].tolist()))
    prefix = list(state.current_trick)
    state.history.extend((prefix + plays)[:n_before + n_plays - trick_len])
    state.current_trick = list(zip(trick_seats[:trick_len].tolist(),
                                   trick_cards[:trick_len].tolist()))
    state.hands = hands.tolist()
    state.scores = scores.tolist()
    state.to_play = int(to_play)
    state.hearts_broken = bool(hearts_broken)
    state.trick_number = int(trick_number)


def run_playout(state) -> None:
    """Kernel equivalent of the pure-Python `_playout` loop."""
    hands, scores, tc, ts, tl, played = _to_arrays(state)
    out_cards, out_seats = _OUT_CARDS, _OUT_SEATS
    (_status, to_play, _pm, tl, hb, tn, n) = playout_to_end(
        hands, state.to_play, played, tc, ts, tl, state.hearts_broken,
        state.trick_number, scores, out_cards, out_seats)
    _write_back(state, hands, scores, tc, ts, tl, to_play, hb, tn,
                out_cards, out_seats, n)


def run_playout_until_decision(state, stop_seat: int) -> bool:
    """Advance `state` to `stop_seat`'s first >1-legal-move decision.

    Returns True if such a decision was reached (state is left exactly at that
    moment, card unplayed), False if the hand finished first.
    """
    hands, scores, tc, ts, tl, played = _to_arrays(state)
    out_cards, out_seats = _OUT_CARDS, _OUT_SEATS
    (status, to_play, _pm, tl, hb, tn, n) = playout_until_decision(
        hands, state.to_play, played, tc, ts, tl, state.hearts_broken,
        state.trick_number, scores, stop_seat, out_cards, out_seats)
    _write_back(state, hands, scores, tc, ts, tl, to_play, hb, tn,
                out_cards, out_seats, n)
    return status == 1
