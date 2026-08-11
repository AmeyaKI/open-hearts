"""Position featurizer for the learned value function (layout v1).

This is a FOREVER CONTRACT. Every training shard and every `models/value_*.npz`
records `FEATURES_V`; loaders assert it matches. If the layout below ever
changes, bump `FEATURES_V` -- never re-purpose an offset.

Information status (recorded so nobody "fixes" it as a leak): this featurizer
runs INSIDE search on fully-determinized *imagined* worlds produced by the
belief sampler -- exactly the information a heuristic playout has today. Real
player code still receives only a `PlayerView`. See PHASE4_PLAN.md,
"Architecture unchanged".

Rotation
--------
Everything seat-indexed is rotated to the evaluated seat: rotated slot ``r``
means absolute seat ``(seat + r) % 4``, so slot 0 is always the evaluated seat
and slots 1..3 are the following seats in play order. One network then serves
all four seats. Note the evaluated seat is NOT always the seat to act: the
data generator featurizes all four seats at every ply.

Layout (NF = 333, all values in [0, 1])
---------------------------------------
=========  =====  ==================================================
offset     size   contents
=========  =====  ==================================================
0          208    OFF_HANDS: 4x52 binary, block r = card c is in
                  rotated seat r's CURRENT hand (row-major: index
                  ``r * 52 + c``).
208         52    OFF_PLAYED: card c has already been played
                  (completed tricks AND the current trick -- this is
                  the kernel's `played_mask` convention).
260         52    OFF_TRICK: card c is one of the cards played to the
                  CURRENT (incomplete) trick.
312          4    OFF_WIN_SEAT: one-hot, rotated seat currently
                  winning the trick. All zeros when trick_len == 0.
316          4    OFF_LED: one-hot led suit. All zeros when
                  trick_len == 0 (i.e. the evaluated seat's side is
                  on lead / the trick is empty).
320         13    OFF_SCALARS, in order:
                    +0  trick_number / 13        (1.0 at hand end)
                    +1  trick_len / 4
                    +2  hearts_broken (0/1)
                    +3  points in the current trick so far / 26
                        (max 16: Qs + 3 hearts)
                    +4..+7  scores[rotated seat r] / 26 (per-hand
                        scores, which sum to 26)
                    +8  Q-spades already played (0/1)
                    +9..+12  cards of suit s still UNPLAYED (i.e. in
                        the deck across all four hands, not just the
                        evaluated hand) / 13
=========  =====  ==================================================

Deviation from the plan text (recorded with reasoning). The plan's v1 spec
says "52 + 4: current-trick cards one-hot with (rotated) seat tags". That line
is underdetermined -- 4 slots cannot tag 52 cards. The two readings are
(a) "which rotated seats have played to this trick" and (b) "which rotated
seat is winning it". We chose (b), because (a) is always recoverable from the rest of
the vector, for every rotated seat, whether or not it is the one to act: each
seat plays exactly one card per trick, so rotated seat r has ``13 -
trick_number`` cards if it has not yet played to the current trick and one
fewer if it has -- and both the hand-block counts and `trick_number` are in
the vector. The winner is NOT recoverable: the
trick block is an unordered SET of cards, so the winning card is identifiable
but its player is not. Corroborating: the plan passes `win_seat` explicitly in
the signature even though the kernel derives it, and specs pass derived values
because they are meant to be consumed. Full card->seat attribution inside the
trick would cost 3x52 more features and blow the ~330 target; v1 is
"deliberately simple, extend only with evidence".

Both paths
----------
`featurize` dispatches on `kernel.jit_enabled()` between the compiled kernel
and the identical pure-Python source (`_featurize_py`), so the fallback under
`OPENHEARTS_NO_JIT=1` cannot drift from the compiled one by construction.
`tests/test_features.py` still pins them against each other in one process.
"""
import numpy as np

from .kernel import HAVE_NUMBA, jit_enabled, njit

FEATURES_V = 1

OFF_HANDS = 0
OFF_PLAYED = 208
OFF_TRICK = 260
OFF_WIN_SEAT = 312
OFF_LED = 316
OFF_SCALARS = 320
N_SCALARS = 13
NF = 333


def _featurize_py(hands, played_mask, trick_cards, trick_seats, trick_len,
                  led_suit, win_seat, hearts_broken, trick_number, scores,
                  seat):
    """Numba-compatible featurizer (see module docstring for the layout).

    Written once and compiled by numba below; this exact source is also the
    `OPENHEARTS_NO_JIT=1` fallback.
    """
    out = np.zeros(NF, dtype=np.float64)
    pm = np.int64(played_mask)

    # 4x52 rotated hands + 52 played
    for r in range(4):
        h = hands[(seat + r) % 4]
        base = OFF_HANDS + r * 52
        for c in range(52):
            if (h >> c) & 1:
                out[base + c] = 1.0
    n_unplayed = np.zeros(4, dtype=np.int64)
    for c in range(52):
        if (pm >> c) & 1:
            out[OFF_PLAYED + c] = 1.0
        else:
            n_unplayed[c // 13] += 1

    # current trick: card set, winner tag, led suit, points so far
    trick_pts = 0
    for i in range(trick_len):
        c = trick_cards[i]
        out[OFF_TRICK + c] = 1.0
        if c // 13 == 3:
            trick_pts += 1
        elif c == 36:
            trick_pts += 13
    if trick_len > 0:
        # guarded: led_suit / win_seat are -1 on an empty trick, and numba
        # does not bounds-check, so an unguarded write would silently land in
        # the preceding block.
        out[OFF_LED + led_suit] = 1.0
        out[OFF_WIN_SEAT + ((win_seat - seat) % 4)] = 1.0

    out[OFF_SCALARS + 0] = trick_number / 13.0
    out[OFF_SCALARS + 1] = trick_len / 4.0
    if hearts_broken:
        out[OFF_SCALARS + 2] = 1.0
    out[OFF_SCALARS + 3] = trick_pts / 26.0
    for r in range(4):
        out[OFF_SCALARS + 4 + r] = scores[(seat + r) % 4] / 26.0
    if (pm >> 36) & 1:
        out[OFF_SCALARS + 8] = 1.0
    for s in range(4):
        out[OFF_SCALARS + 9 + s] = n_unplayed[s] / 13.0
    return out


_featurize_njit = njit(cache=True)(_featurize_py) if HAVE_NUMBA \
    else _featurize_py


def featurize(hands, played_mask, trick_cards, trick_seats, trick_len,
              led_suit, win_seat, hearts_broken, trick_number, scores, seat):
    """Feature vector for a determinized position, rotated to `seat`.

    Arguments are the kernel's array representation: `hands` int64[4] (current
    hands), `played_mask` int64 (completed tricks AND the current trick),
    `trick_cards`/`trick_seats` int64[4] with `trick_len` entries valid,
    `led_suit`/`win_seat` as returned by `kernel._trick_head` (-1 when the
    trick is empty), `scores` int64[4] (per-hand points taken so far).

    `trick_seats` is accepted for signature compatibility with the plan and is
    deliberately NOT read: per-card attribution inside the trick is carried by
    `win_seat` plus hand-size parity (see the module docstring). Do not wire it
    in without bumping FEATURES_V.

    Caching hazard: the offsets above are module globals frozen into numba's
    on-disk cache (`njit(cache=True)`, house pattern). If any offset ever
    moves, clear `__pycache__` as well as bumping FEATURES_V.
    """
    fn = _featurize_njit if jit_enabled() else _featurize_py
    return fn(np.asarray(hands, dtype=np.int64), np.int64(played_mask),
              np.asarray(trick_cards, dtype=np.int64),
              np.asarray(trick_seats, dtype=np.int64), np.int64(trick_len),
              np.int64(led_suit), np.int64(win_seat), bool(hearts_broken),
              np.int64(trick_number), np.asarray(scores, dtype=np.int64),
              np.int64(seat))


# --------------------------------------------------------------------------
# batch variant (Task 2 addition, per Task 1's implementation note): the
# LAYOUT above is the forever contract; this out-parameter batch entry point
# is new surface, not a change to it. It featurizes every (ply, seat) pair of
# one game -- P plies x 4 seats -- in a single dispatch, so a data generator
# pays Python/numba-boundary overhead once per game instead of once per row.
# --------------------------------------------------------------------------
def _featurize_batch_py(hands, played_mask, trick_cards, trick_seats,
                        trick_len, led_suit, win_seat, hearts_broken,
                        trick_number, scores, out):
    n = hands.shape[0]
    for p in range(n):
        for seat in range(4):
            out[p, seat, :] = _featurize_py(
                hands[p], played_mask[p], trick_cards[p], trick_seats[p],
                trick_len[p], led_suit[p], win_seat[p], hearts_broken[p],
                trick_number[p], scores[p], seat)


def _featurize_batch_njit_src(hands, played_mask, trick_cards, trick_seats,
                              trick_len, led_suit, win_seat, hearts_broken,
                              trick_number, scores, out):
    n = hands.shape[0]
    for p in range(n):
        for seat in range(4):
            out[p, seat, :] = _featurize_njit(
                hands[p], played_mask[p], trick_cards[p], trick_seats[p],
                trick_len[p], led_suit[p], win_seat[p], hearts_broken[p],
                trick_number[p], scores[p], seat)


_featurize_batch_njit = njit(cache=True)(_featurize_batch_njit_src) \
    if HAVE_NUMBA else _featurize_batch_njit_src


def featurize_batch(hands, played_mask, trick_cards, trick_seats, trick_len,
                    led_suit, win_seat, hearts_broken, trick_number, scores,
                    out):
    """Featurize every ply of one game, all 4 seats, in a single call.

    Same per-argument semantics as `featurize`, but each argument carries a
    leading ply axis (P plies): `hands` int64[P,4], `played_mask` int64[P],
    `trick_cards`/`trick_seats` int64[P,4], `trick_len`/`led_suit`/`win_seat`/
    `trick_number` int64[P], `hearts_broken` bool[P], `scores` int64[P,4].

    `out` is a preallocated float64[P,4,NF] buffer; `out[p, seat]` is filled
    in place with `featurize(..., seat)` for ply `p` (LAYOUT identical to the
    per-row call -- the signature is new, the contract is not). Returns `out`
    for convenience.
    """
    fn = _featurize_batch_njit if jit_enabled() else _featurize_batch_py
    fn(np.asarray(hands, dtype=np.int64),
       np.asarray(played_mask, dtype=np.int64),
       np.asarray(trick_cards, dtype=np.int64),
       np.asarray(trick_seats, dtype=np.int64),
       np.asarray(trick_len, dtype=np.int64),
       np.asarray(led_suit, dtype=np.int64),
       np.asarray(win_seat, dtype=np.int64),
       np.asarray(hearts_broken, dtype=np.bool_),
       np.asarray(trick_number, dtype=np.int64),
       np.asarray(scores, dtype=np.int64),
       out)
    return out
