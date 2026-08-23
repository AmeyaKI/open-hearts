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
_FULL_DECK = np.int64((1 << 52) - 1)


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
# candidate-world audit (Phase 3.5)
# --------------------------------------------------------------------------
@njit(cache=True)
def _audit(hands, plays_cards, plays_seats, observer, epsilon, early_exit):
    """Replay the observed sequence in a candidate world; return its weight.

    Exact port of `belief.weighted.world_weight`'s replay loop with
    `HeuristicPlayer` as the policy: at each OPPONENT ply multiply in
    (1-eps)*[heuristic would play this card] + eps/n_legal; the observer's
    own plies contribute 1.0; an observed card that is illegal in this world
    returns 0.0. With `early_exit` a zero policy factor returns 0.0
    immediately (the eps=0 rejection fast path); without it the zero is
    multiplied through, which is the same answer more slowly.

    `hands` is the four ORIGINAL 13-card hands (mutated in place -- the
    adapter hands over a private array). Turn order is derived here exactly
    as the engine does; a mismatch against the observed seat returns the
    sentinel -1.0, which the adapter turns into the Python path's loud
    "replay desync" AssertionError.
    """
    n = plays_cards.shape[0]
    trick_cards = np.zeros(4, dtype=np.int64)
    trick_seats = np.zeros(4, dtype=np.int64)
    trick_len = 0
    trick_number = 0
    hearts_broken = False
    played_mask = np.int64(0)
    to_play = plays_seats[0]
    weight = 1.0
    for k in range(n):
        seat = plays_seats[k]
        card = plays_cards[k]
        if seat != to_play:
            return -1.0
        led_suit, win_rank, _ws = _trick_head(trick_cards, trick_seats,
                                              trick_len)
        hand = hands[seat]
        legal = _legal(hand, led_suit, trick_len, hearts_broken, trick_number)
        if ((legal >> card) & 1) == 0:
            return 0.0
        if seat != observer:
            n_legal = _popcount(legal)
            # `seen` for the leading rule: everything played so far plus this
            # seat's own remaining hand -- the same quantity the Python
            # policy builds from view.history and view.hand (mid-trick cards
            # cannot matter; the rule is only consulted when the trick is
            # empty).
            choice = _choose(hand, legal, led_suit, win_rank, trick_len,
                             played_mask | hand)
            hit = 1.0 if choice == card else 0.0
            factor = (1.0 - epsilon) * hit + epsilon / n_legal
            if factor == 0.0 and early_exit:
                return 0.0
            weight *= factor
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
            trick_len = 0
            trick_number += 1
            to_play = win_seat
        else:
            to_play = (seat + 1) % 4
    return weight


@njit(cache=True)
def audit_world(orig_hands, plays_cards, plays_seats, observer, epsilon):
    """Weight of a candidate world, with the rung-1 early exit."""
    return _audit(orig_hands, plays_cards, plays_seats, observer, epsilon,
                  True)


@njit(cache=True)
def audit_world_no_early_exit(orig_hands, plays_cards, plays_seats, observer,
                              epsilon):
    """Same weight, multiplying every policy factor through (test oracle)."""
    return _audit(orig_hands, plays_cards, plays_seats, observer, epsilon,
                  False)


def audit_world_weight(hands, all_plays, observer, epsilon,
                       early_exit: bool = True) -> float:
    """Adapter: compiled audit for reconstructed hands + observed plays.

    `hands` are the four ORIGINAL hands and `all_plays` the observed
    ((seat, card), ...) sequence, both produced by
    `belief.weighted._reconstruct_original_hands` -- every structural assert
    stays on the Python side of this boundary.
    """
    if not all_plays:
        return 1.0  # no evidence yet: every world is equally consistent
    arr_hands = np.array(hands, dtype=np.int64)
    seats = np.array([s for s, _c in all_plays], dtype=np.int64)
    cs = np.array([c for _s, c in all_plays], dtype=np.int64)
    fn = audit_world if early_exit else audit_world_no_early_exit
    w = float(fn(arr_hands, cs, seats, np.int64(observer), float(epsilon)))
    if w == -1.0:
        raise AssertionError(
            "replay desync: observed seat order does not match the engine's "
            "turn order in this world"
        )
    assert np.isfinite(w) and w >= 0.0, f"bad world weight {w}"
    return w


# --------------------------------------------------------------------------
# batch arrangement sampler (Phase 2.6)
# --------------------------------------------------------------------------
@njit(cache=True)
def _sample_arrangements_core(probs, void_suit_masks, hand_sizes,
                              unseen_cards, n_samples, max_restarts,
                              hands_out, attempts_out, remaining, w, hands):
    """The draw loop, WITHOUT seeding and WITHOUT allocating (Phase 2.8).

    Split out of `sample_arrangements_batch` so the fused honest-search kernel
    can seed numba's generator ONCE per decision and then draw a continuous
    stream through every inner sample, reusing scratch buffers instead of
    re-initialising MT19937 and reallocating for each of the (candidate x
    world) inner re-determinizations. The 2.6 entry point below is unchanged
    in behaviour -- it seeds, allocates, and calls this -- so every existing
    bitwise pin on the batch sampler still holds.

    Only the first `n_success` rows of `hands_out` / `attempts_out` are
    written, so a reused buffer needs no clearing between calls.
    """
    k = unseen_cards.shape[0]
    n_success = 0
    for _s in range(n_samples):
        ok = False
        used = 0
        for attempt in range(1, max_restarts + 1):
            for i in range(3):
                remaining[i] = hand_sizes[i]
                hands[i] = 0
            dead = False
            for ci in range(k):
                c = unseen_cards[ci]
                suit = c // 13
                total = 0.0
                for i in range(3):
                    wi = probs[i, c]
                    if remaining[i] == 0 or ((void_suit_masks[i] >> suit) & 1):
                        wi = 0.0
                    w[i] = wi
                    total += wi
                if total <= 0.0:
                    dead = True
                    break
                u = np.random.random() * total
                acc = 0.0
                pick = -1
                for i in range(3):
                    acc += w[i]
                    if u < acc and w[i] > 0.0:
                        pick = i
                        break
                if pick < 0:  # float rounding at the top of the range
                    for i in range(2, -1, -1):
                        if w[i] > 0.0:
                            pick = i
                            break
                hands[pick] |= _ONE << c
                remaining[pick] -= 1
            if not dead and remaining[0] == 0 and remaining[1] == 0 \
                    and remaining[2] == 0:
                ok = True
                used = attempt
                break
        if ok:
            hands_out[n_success, 0] = hands[0]
            hands_out[n_success, 1] = hands[1]
            hands_out[n_success, 2] = hands[2]
            attempts_out[n_success] = used
            n_success += 1
    return n_success


@njit(cache=True)
def sample_arrangements_batch(probs, void_suit_masks, hand_sizes,
                              unseen_cards, n_samples, max_restarts, seed):
    """Draw `n_samples` arrangements of `unseen_cards` among 3 opponents.

    Exact algorithmic port of `sampler.sample_arrangement`, batched: walk the
    unseen cards in ascending order, pick a holder with probability
    proportional to `probs[:, c]` after zeroing opponents who are full or void
    in that suit, restart the whole arrangement on a dead end, give up after
    `max_restarts`. `void_suit_masks[i]` has bit s set when opponent i is void
    in suit s (kept separate from `probs` because UNIFORM tables do not encode
    voids, and one experiment disables void-respecting sampling entirely).

    Returns (hands[n_samples, 3], attempts[n_samples], n_success); only the
    first `n_success` rows are filled. Failures are simply not emitted, which
    is what the callers' `failed_samples` counter counts.
    """
    np.random.seed(seed)
    hands_out = np.zeros((n_samples, 3), dtype=np.int64)
    attempts_out = np.zeros(n_samples, dtype=np.int64)
    remaining = np.zeros(3, dtype=np.int64)
    w = np.zeros(3, dtype=np.float64)
    hands = np.zeros(3, dtype=np.int64)
    n_success = _sample_arrangements_core(
        probs, void_suit_masks, hand_sizes, unseen_cards, n_samples,
        max_restarts, hands_out, attempts_out, remaining, w, hands)
    return hands_out, attempts_out, n_success


# --------------------------------------------------------------------------
# fused draw + reconstruct + audit + accumulate (Phase 3.6)
# --------------------------------------------------------------------------
@njit(cache=True)
def draw_audit_batch(probs_tab, void_masks, hand_sizes, unseen_cards,
                     n_candidates, max_restarts, seed,
                     obs_hand, opp_seats, plays_cards, plays_seats,
                     observer, epsilon, out_probs, worlds, weights,
                     total_w, total_w2):
    """Draw `n_candidates` worlds, audit each, accumulate the survivors.

    This is Phase 3.5's `sample_arrangements` + Python
    `_reconstruct_original_hands` + `audit_world` + `account` loop, fused into
    one compiled call. It CALLS `sample_arrangements_batch` with the same
    arguments the 3.5 adapter passes, so the RNG stream and the drawn worlds
    are unchanged by construction.

    Reconstruction (exact port of `belief.weighted._reconstruct_original_hands`):
    `worlds[j]` holds opponent i's CURRENT hand in `BeliefTable.opponent_seats`
    order, so seat `opp_seats[i]`; the observer keeps `obs_hand`; then every
    observed play is added back to the seat that made it. All four seats are
    ASSIGNED before any bit is OR-ed back, so the scratch array needs no reset.

    The per-candidate structural asserts of the Python reconstruction (13 cards
    per seat, disjoint hands, full deck, observer containment) are NOT repeated
    here: they are subsumed by a once-per-call frame check on the Python side
    (`fused_audit_context`) plus the sampler's own exit condition. See that
    function's docstring for the argument.

    Accumulation order matches the Python `account()` exactly -- `kept`,
    `total_w`, `total_w2`, then `out_probs[i, c] += w` over i ascending and
    c ascending -- so the float sums are bitwise identical.

    `total_w` / `total_w2` come in as the caller's RUNNING sums and come back
    updated. Threading them through (rather than summing per chunk and adding)
    keeps the float association order identical to the one-at-a-time 3.5 loop
    across chunk boundaries -- without it the totals differ in the last bits.

    Returns `(n_drawn, n_kept, total_w, total_w2, desync)`. `n_drawn` is
    always `n_candidates` (failed draws still count as draws, as in 3.5).
    `worlds` / `weights` are filled in for the first `n_kept` survivors.
    `desync` is True if the observed seat order is impossible in the engine's
    turn order, which the adapter turns into a loud AssertionError.
    """
    hands_out, _attempts, n_ok = sample_arrangements_batch(
        probs_tab, void_masks, hand_sizes, unseen_cards, n_candidates,
        max_restarts, seed)
    n_plays = plays_cards.shape[0]
    orig = np.zeros(4, dtype=np.int64)
    n_kept = 0
    for j in range(n_ok):
        orig[observer] = obs_hand
        for i in range(3):
            orig[opp_seats[i]] = hands_out[j, i]
        for k in range(n_plays):
            orig[plays_seats[k]] |= _ONE << plays_cards[k]
        if n_plays == 0:
            w = 1.0  # no evidence yet: every world is equally consistent
        else:
            w = _audit(orig, plays_cards, plays_seats, observer, epsilon,
                       True)
            if w < 0.0:
                return n_candidates, n_kept, total_w, total_w2, True
        if w <= 0.0:
            continue
        for i in range(3):
            worlds[n_kept, i] = hands_out[j, i]
        weights[n_kept] = w
        n_kept += 1
        total_w += w
        total_w2 += w * w
        for i in range(3):
            h = hands_out[j, i]
            for c in range(52):
                if (h >> c) & 1:
                    out_probs[i, c] += w
    return n_candidates, n_kept, total_w, total_w2, False


# --------------------------------------------------------------------------
# belief rebalance (Phase 2.7)
# --------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _pairwise_sum(a):
    """numpy's float64 reduction order for a contiguous 1-D block.

    NOT a naive loop. numpy sums a contiguous float64 axis with the
    8-accumulator pairwise scheme below, and a naive left-to-right loop
    disagrees in the last bits on roughly half of realistic belief rows
    (measured: 11100/20000). Since Phase 2.7 demands BITWISE equality with the
    Python reference, the order is reproduced exactly. numba's own `a.sum()`
    is naive and must not be substituted here.

    Only the `n <= 128` branch of numpy's `pairwise_sum` is reproduced: rows
    here are always 52 long and columns always 3. `_rebalance_kernel` asserts
    the row length so a future caller cannot silently escape that range.
    """
    n = a.shape[0]
    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[i]
        return res
    r0 = a[0]
    r1 = a[1]
    r2 = a[2]
    r3 = a[3]
    r4 = a[4]
    r5 = a[5]
    r6 = a[6]
    r7 = a[7]
    i = 8
    lim = n - (n % 8)
    while i < lim:
        r0 += a[i]
        r1 += a[i + 1]
        r2 += a[i + 2]
        r3 += a[i + 3]
        r4 += a[i + 4]
        r5 += a[i + 5]
        r6 += a[i + 6]
        r7 += a[i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[i]
        i += 1
    return res


@njit(cache=True)
def _rebalance_kernel(probs, sizes, unseen_cols, tol, coarse_tol,
                      coarse_after, max_iters):
    """Exact port of `belief.table._rebalance`; see that docstring.

    Returns `(probs, status, dev, n_iters)`. status 0 = converged,
    1 = a column collapsed to zero, 2 = did not converge, 3 = the final
    [0, 1+1e-9] bound was violated. The adapter turns 1/2/3 into the same
    AssertionErrors the Python version raises.

    Operation order is load-bearing and mirrors the reference line for line:
    row sums -> row scale -> multiply -> unseen column sums -> divide ->
    deviation. No reassociation, no fastmath.
    """
    n_rows = probs.shape[0]
    n_cols = probs.shape[1]
    k_unseen = unseen_cols.shape[0]
    dev = 0.0
    n_iters = 0
    row = np.zeros(n_rows, dtype=np.float64)
    col = np.zeros(k_unseen, dtype=np.float64)
    converged = False
    for k in range(max_iters):
        n_iters = k + 1
        for i in range(n_rows):
            row[i] = _pairwise_sum(probs[i])
        for i in range(n_rows):
            if row[i] > 0.0:
                scale = sizes[i] / row[i]
            else:
                scale = 1.0
            for c in range(n_cols):
                probs[i, c] = probs[i, c] * scale
        for j in range(k_unseen):
            c = unseen_cols[j]
            # 3-row reduction: numpy's axis-0 order is sequential by row
            s = probs[0, c]
            for i in range(1, n_rows):
                s += probs[i, c]
            col[j] = s
            if not (s > 0.0):
                return probs, 1, 0.0, n_iters
        for j in range(k_unseen):
            c = unseen_cols[j]
            for i in range(n_rows):
                probs[i, c] = probs[i, c] / col[j]
        dev = 0.0
        for i in range(n_rows):
            d = abs(_pairwise_sum(probs[i]) - sizes[i])
            if d > dev:
                dev = d
        for j in range(k_unseen):
            c = unseen_cols[j]
            s = probs[0, c]
            for i in range(1, n_rows):
                s += probs[i, c]
            d = abs(s - 1.0)
            if d > dev:
                dev = d
        if dev < tol or (k + 1 >= coarse_after and dev < coarse_tol):
            converged = True
            break
    if not converged:
        return probs, 2, dev, n_iters
    for i in range(n_rows):
        for c in range(n_cols):
            if probs[i, c] < 0.0 or probs[i, c] > 1.0 + 1e-9:
                return probs, 3, dev, n_iters
    return probs, 0, dev, n_iters


def _rebalance_call(probs, hand_sizes, unseen_mask, tol=1e-9,
                    coarse_tol=1e-4, coarse_after=2000, max_iters=100000):
    from openhearts.engine import cards as _cards  # local: avoids cycle

    sizes = np.array(hand_sizes, dtype=float)
    unseen_cols = np.array(_cards.cards_in(unseen_mask), dtype=np.int64)
    if unseen_cols.size == 0:
        return probs, 0, 0.0, 0  # matches the reference's early return
    assert probs.shape == (3, 52)
    # the reference rebinds `probs = probs * scale[:, None]` and so never
    # writes through to the caller's array; the kernel scales in place, so
    # copy first to keep that contract.
    work = np.array(probs, dtype=np.float64, order="C", copy=True)
    out, status, dev, n_iters = _rebalance_kernel(
        work, sizes, unseen_cols, float(tol), float(coarse_tol),
        int(coarse_after), int(max_iters))
    if status == 1:
        raise AssertionError("column collapsed to zero during rebalance")
    if status == 2:
        raise AssertionError(f"rebalance did not converge (dev={dev})")
    if status == 3:
        raise AssertionError()
    return out, status, dev, n_iters


def rebalance(probs, hand_sizes, unseen_mask, tol=1e-9, coarse_tol=1e-4,
              coarse_after=2000, max_iters=100000):
    """Compiled `belief.table._rebalance`, bitwise-identical to it."""
    out, _s, _d, n = _rebalance_call(probs, hand_sizes, unseen_mask, tol,
                                     coarse_tol, coarse_after, max_iters)
    return probs if n == 0 else out


def rebalance_iters(probs, hand_sizes, unseen_mask, tol=1e-9, coarse_tol=1e-4,
                    coarse_after=2000, max_iters=100000):
    """Passes the kernel needed to converge (diagnostic; used by tests)."""
    return _rebalance_call(probs, hand_sizes, unseen_mask, tol, coarse_tol,
                           coarse_after, max_iters)[3]


# --------------------------------------------------------------------------
# fused honest-search decision loop (Phase 2.8)
# --------------------------------------------------------------------------
# Level codes, matching `belief.table.Level`. Passed as an int so the kernel
# never sees a Python enum.
LEVEL_UNIFORM = 0
LEVEL_VOIDS = 1
LEVEL_FULL = 2

# Evidence / decision status codes. 0 is success; everything else is turned
# into the SAME AssertionError the Python path raises, never swallowed.
_ST_OK = 0
_ST_NO_HOLDER = 1        # table._build_raw's "card has no possible holder"
_ST_COL_COLLAPSE = 2     # _rebalance status 1
_ST_NO_CONVERGE = 3      # _rebalance status 2
_ST_BOUNDS = 4           # _rebalance status 3
_ST_BAD_STOP = 5         # stopped at a seat with <=1 legal move (impossible)
_ST_SEEDS = 6            # ran out of pre-drawn seeds (upper bound was wrong)


@njit(cache=True)
def _evidence(obs_hand, plays_cards, plays_seats, n_plays, observer, level,
              respects_voids, probs, hand_sizes, void_masks, unseen_cards,
              sizes):
    """Port of `belief.table._build_raw` + `from_view`'s normalisation.

    Reads NOTHING but the observer's own hand and the flat play sequence --
    i.e. exactly the contents of a `PlayerView`. The fused loop holds a
    fully-resolved imagined world in `hands[]`, and this function is
    deliberately not given it: the view boundary is enforced by the argument
    list. `tests/test_jit_honest_evidence.py` pins the result bitwise against
    `BeliefTable.from_view` over imagined mid-playout states.

    Trick grouping: `plays_cards/plays_seats[0:n_plays]` is the whole hand's
    play sequence in order, so consecutive runs of 4 are exactly the tricks --
    which is what makes this equivalent to the reference's
    `history + current_trick` scan (history only ever extends by whole
    tricks; the test asserts that rather than assuming it).

    Writes `probs` (3x52), `hand_sizes` (3), `void_masks` (3) and
    `unseen_cards` (52, ascending); `sizes` is float64 scratch for the
    rebalance. Returns `(n_unseen, status)`.
    """
    # ---- seen / unseen ---------------------------------------------------
    seen = obs_hand
    for k in range(n_plays):
        seen |= _ONE << plays_cards[k]
    unseen_mask = _FULL_DECK & ~seen

    # ---- hand sizes ------------------------------------------------------
    played_count0 = 0
    played_count1 = 0
    played_count2 = 0
    played_count3 = 0
    for k in range(n_plays):
        s = plays_seats[k]
        if s == 0:
            played_count0 += 1
        elif s == 1:
            played_count1 += 1
        elif s == 2:
            played_count2 += 1
        else:
            played_count3 += 1
    for i in range(3):
        s = (observer + 1 + i) % 4
        if s == 0:
            hand_sizes[i] = 13 - played_count0
        elif s == 1:
            hand_sizes[i] = 13 - played_count1
        elif s == 2:
            hand_sizes[i] = 13 - played_count2
        else:
            hand_sizes[i] = 13 - played_count3

    # ---- voids: failure to follow the LED suit ---------------------------
    # The observer's own failures are skipped (`if s in opponent_seats` in the
    # reference); seat -> opponent index is (s - observer - 1) mod 4, written
    # as (s - observer + 3) % 4 so it never goes negative.
    for i in range(3):
        void_masks[i] = 0
    tl = 0
    led = -1
    for k in range(n_plays):
        c = plays_cards[k]
        s = plays_seats[k]
        if tl > 0:
            if c // 13 != led:
                idx = (s - observer + 3) % 4
                if idx < 3:
                    void_masks[idx] |= _ONE << led
        else:
            led = c // 13
        tl += 1
        if tl == 4:
            tl = 0

    # ---- raw probs -------------------------------------------------------
    for i in range(3):
        for c in range(52):
            probs[i, c] = 0.0
    n_unseen = 0
    for c in range(52):
        if (unseen_mask >> c) & 1:
            unseen_cards[n_unseen] = c
            n_unseen += 1
            probs[0, c] = 1.0
            probs[1, c] = 1.0
            probs[2, c] = 1.0

    if level == LEVEL_VOIDS or level == LEVEL_FULL:
        for i in range(3):
            for s in range(4):
                if (void_masks[i] >> s) & 1:
                    for c in range(13 * s, 13 * s + 13):
                        probs[i, c] = 0.0

    # guard: every unseen card must have at least one possible holder
    for j in range(n_unseen):
        c = unseen_cards[j]
        if probs[0, c] + probs[1, c] + probs[2, c] <= 0.0:
            return n_unseen, _ST_NO_HOLDER

    # ---- level-specific normalisation ------------------------------------
    if level == LEVEL_FULL:
        if n_unseen > 0:  # reference `_rebalance` returns early when empty
            for i in range(3):
                sizes[i] = hand_sizes[i]
            _p, status, _dev, _n = _rebalance_kernel(
                probs, sizes, unseen_cards[:n_unseen],
                1e-9, 1e-4, 2000, 100000)
            if status == 1:
                return n_unseen, _ST_COL_COLLAPSE
            if status == 2:
                return n_unseen, _ST_NO_CONVERGE
            if status == 3:
                return n_unseen, _ST_BOUNDS
    else:
        # reference: probs /= np.where(col > 0, col, 1.0) over ALL columns
        for c in range(52):
            col = probs[0, c] + probs[1, c] + probs[2, c]
            if col > 0.0:
                probs[0, c] = probs[0, c] / col
                probs[1, c] = probs[1, c] / col
                probs[2, c] = probs[2, c] / col

    # The sampler's void masks are a SEPARATE axis from the probs zeroing:
    # `sampler_respects_voids=False` empties the void sets only (that is what
    # HonestSearchPlayer.choose does to the table), leaving the probs of a
    # VOIDS/FULL table zeroed as built.
    if not respects_voids:
        for i in range(3):
            void_masks[i] = 0
    return n_unseen, _ST_OK


@njit(cache=True)
def _play_into(hands, scores, tc, ts, tl, hb, tn, seat, card, played,
               plays_cards, plays_seats, n_plays):
    """One ply, mutating the array-form state; port of `GameState.play`.

    Also appends to the flat play sequence, which is what `_evidence` reads.
    Returns `(to_play, played, tl, hb, tn, n_plays)`.
    """
    bit = _ONE << card
    hands[seat] = hands[seat] & ~bit
    played |= bit
    if card // 13 == 3:
        hb = True
    tc[tl] = card
    ts[tl] = seat
    tl += 1
    plays_cards[n_plays] = card
    plays_seats[n_plays] = seat
    n_plays += 1
    if tl == 4:
        _l, _r, win_seat = _trick_head(tc, ts, 4)
        pts = 0
        for i in range(4):
            c = tc[i]
            if c // 13 == 3:
                pts += 1
            elif c == _QS:
                pts += 13
        scores[win_seat] += pts
        tl = 0
        tn += 1
        to_play = win_seat
    else:
        to_play = (seat + 1) % 4
    return to_play, played, tl, hb, tn, n_plays


@njit(cache=True)
def _inner_best(obs_hand, legal, opp_seats, inner_hands, n_ok, observer,
                played, tc, ts, tl, hb, tn, scores,
                ihands, iscores, itc, its, iout_cards, iout_seats):
    """Phase-1 `SearchPlayer.choose`'s evaluation loop, in-kernel.

    Legal cards are walked in ASCENDING order and the comparison is strict
    `<`, so ties keep the lowest card index -- identical to the Python loop.
    The per-world total is accumulated as an int64 and divided once, matching
    `total / len(arrangements)` exactly, so no float drift enters the
    comparison.
    """
    base = scores[observer]
    best_card = -1
    best_avg = 0.0
    first = True
    m = legal
    while m:
        low = m & -m
        card = _lowest(low)
        m ^= low
        total = 0
        for j in range(n_ok):
            ihands[observer] = obs_hand
            for i in range(3):
                ihands[opp_seats[i]] = inner_hands[j, i]
            for s in range(4):
                iscores[s] = scores[s]
            for i in range(4):
                itc[i] = tc[i]
                its[i] = ts[i]
            p = played
            ltl = tl
            lhb = hb
            ltn = tn
            (tp, p, ltl, lhb, ltn, _np) = _play_into(
                ihands, iscores, itc, its, ltl, lhb, ltn, observer, card, p,
                iout_cards, iout_seats, 0)
            (_st, tp, p, ltl, lhb, ltn, _n) = _run(
                ihands, tp, p, itc, its, ltl, lhb, ltn, iscores, -1,
                iout_cards, iout_seats)
            total += iscores[observer] - base
        avg = total / n_ok
        if first or avg < best_avg:
            best_card = card
            best_avg = avg
            first = False
    return best_card


@njit(cache=True)
def _honest_core(obs_hand, opp_seats, real_cards, real_seats,
                 n_real, trick_cards0, trick_seats0, trick_len0,
                 hearts_broken0, trick_number0, scores0, observer,
                 arrangements, n_worlds, legal_cards, n_cand, n_inner, level,
                 respects_voids, max_restarts, seeds, seed_off, avgs):
    """Steps 3-4 of `HonestSearchPlayer.choose`, entirely in one crossing.

    For each candidate x outer world: rebuild the imagined state, play the
    candidate, run the heuristic playout to our next real decision,
    re-determinize THERE (evidence extraction -> 2.7 rebalance -> 2.6 batch
    sampler -> 2.5 playouts for the inner candidates), play the inner choice,
    finish the hand; average the points delta; take the min with ties going to
    the lowest card index -- the same order and the same tie-break as the
    Python loop.

    RNG -- PRE-DRAWN SEEDS (the Phase 2.8 redesign). `seeds` holds numbers the
    CALLER already drew from its numpy Generator with the same scalar
    `rng.integers(2**63)` call the unfused path uses. This kernel consumes
    them strictly in order, one immediately before each inner sample, calling
    `np.random.seed` exactly where `sample_arrangements_batch` calls it on the
    unfused path. Therefore the fused path is BITWISE identical to the unfused
    JIT path -- same worlds, same playouts, same means, same card -- and the
    number actually consumed is returned so the caller can leave its Generator
    in the identical state.

    `seeds` must hold at least `n_cand * n_worlds` entries (the exact upper
    bound: each (candidate, world) playout intercepts at most once); running
    out is a loud `_ST_SEEDS`, never a silent reseed.

    Returns `(best_card, n_fallbacks, n_failed, n_seeds_used, status)` and
    writes `avgs[ci]` = candidate `ci`'s mean points-from-here.

    Phase 6C split this body out of `honest_decision_kernel` (now a thin
    wrapper below) so the batched exploiter kernel can run a whole NESTED
    champion decision in-kernel. The extra parameters are all about being
    callable from inside another kernel: `n_worlds`/`n_cand` because the
    caller passes reused scratch buffers whose shape is an upper bound rather
    than the live count; `seed_off` because the caller owns one long
    pre-drawn seed array that many nested decisions consume in order (the
    returned count is RELATIVE to `seed_off`); and `avgs` because a numba
    callee should not allocate a return array per call.
    """
    base_score = scores0[observer]

    # outer scratch
    hands = np.zeros(4, dtype=np.int64)
    scores = np.zeros(4, dtype=np.int64)
    tc = np.zeros(4, dtype=np.int64)
    ts = np.zeros(4, dtype=np.int64)
    plays_cards = np.zeros(52, dtype=np.int64)
    plays_seats = np.zeros(52, dtype=np.int64)
    out_cards = np.zeros(52, dtype=np.int64)
    out_seats = np.zeros(52, dtype=np.int64)
    # evidence scratch
    probs = np.zeros((3, 52), dtype=np.float64)
    hand_sizes = np.zeros(3, dtype=np.int64)
    void_masks = np.zeros(3, dtype=np.int64)
    unseen_cards = np.zeros(52, dtype=np.int64)
    sizes = np.zeros(3, dtype=np.float64)
    # inner sampler scratch
    inner_hands = np.zeros((n_inner, 3), dtype=np.int64)
    inner_att = np.zeros(n_inner, dtype=np.int64)
    remaining = np.zeros(3, dtype=np.int64)
    wbuf = np.zeros(3, dtype=np.float64)
    hbuf = np.zeros(3, dtype=np.int64)
    # inner playout scratch
    ihands = np.zeros(4, dtype=np.int64)
    iscores = np.zeros(4, dtype=np.int64)
    itc = np.zeros(4, dtype=np.int64)
    its = np.zeros(4, dtype=np.int64)
    iout_cards = np.zeros(52, dtype=np.int64)
    iout_seats = np.zeros(52, dtype=np.int64)

    n_fallbacks = 0
    n_failed = 0
    n_seeds = seed_off
    best_card = -1
    best_avg = 0.0
    first = True
    stop_seat = observer if n_inner > 0 else -1

    for ci in range(n_cand):
        card = legal_cards[ci]
        total = 0
        for j in range(n_worlds):
            # ---- rebuild the imagined state (port of state_from_view) ----
            hands[observer] = obs_hand
            for i in range(3):
                hands[opp_seats[i]] = arrangements[j, i]
            for s in range(4):
                scores[s] = scores0[s]
            played = np.int64(0)
            for k in range(n_real):
                plays_cards[k] = real_cards[k]
                plays_seats[k] = real_seats[k]
                played |= _ONE << real_cards[k]
            n_plays = n_real
            for i in range(4):
                tc[i] = trick_cards0[i]
                ts[i] = trick_seats0[i]
            tl = trick_len0
            hb = hearts_broken0
            tn = trick_number0

            # ---- our candidate -------------------------------------------
            (to_play, played, tl, hb, tn, n_plays) = _play_into(
                hands, scores, tc, ts, tl, hb, tn, observer, card, played,
                plays_cards, plays_seats, n_plays)

            # ---- forward to our next real decision ------------------------
            (status, to_play, played, tl, hb, tn, n) = _run(
                hands, to_play, played, tc, ts, tl, hb, tn, scores,
                stop_seat, out_cards, out_seats)
            for k in range(n):
                plays_cards[n_plays + k] = out_cards[k]
                plays_seats[n_plays + k] = out_seats[k]
            n_plays += n

            if status == 1:
                # ---- re-determinize HERE, from the imagined VIEW ----------
                obs_now = hands[observer]
                led0, _wr0, _ws0 = _trick_head(tc, ts, tl)
                if _popcount(_legal(obs_now, led0, tl, hb, tn)) <= 1:
                    # `_run` promises to stop only at >1 legal move, and the
                    # unfused path's inner `choose` would draw NOTHING at a
                    # forced move. A divergence here would silently shift the
                    # seed count, so it is loud instead.
                    return (-1, n_fallbacks, n_failed, n_seeds - seed_off,
                            _ST_BAD_STOP)
                n_unseen, ev = _evidence(
                    obs_now, plays_cards, plays_seats, n_plays, observer,
                    level, respects_voids, probs, hand_sizes, void_masks,
                    unseen_cards, sizes)
                if ev != _ST_OK:
                    return -1, n_fallbacks, n_failed, n_seeds - seed_off, ev
                if n_seeds >= seeds.shape[0]:
                    return (-1, n_fallbacks, n_failed, n_seeds - seed_off,
                            _ST_SEEDS)
                # Exactly where `sample_arrangements_batch` seeds on the
                # unfused path: one pre-drawn seed per inner decision.
                np.random.seed(seeds[n_seeds])
                n_seeds += 1
                n_ok = _sample_arrangements_core(
                    probs, void_masks, hand_sizes, unseen_cards[:n_unseen],
                    n_inner, max_restarts, inner_hands, inner_att, remaining,
                    wbuf, hbuf)
                n_failed += n_inner - n_ok
                led, win_rank, _ws = _trick_head(tc, ts, tl)
                legal = _legal(obs_now, led, tl, hb, tn)
                if n_ok * 2 < n_inner:
                    n_fallbacks += 1
                    chosen = _choose(obs_now, legal, led, win_rank, tl,
                                     played | obs_now)
                else:
                    chosen = _inner_best(
                        obs_now, legal, opp_seats, inner_hands, n_ok,
                        observer, played, tc, ts, tl, hb, tn, scores,
                        ihands, iscores, itc, its, iout_cards, iout_seats)
                (to_play, played, tl, hb, tn, n_plays) = _play_into(
                    hands, scores, tc, ts, tl, hb, tn, observer, chosen,
                    played, plays_cards, plays_seats, n_plays)
                # ---- finish the hand --------------------------------------
                (_st, to_play, played, tl, hb, tn, n2) = _run(
                    hands, to_play, played, tc, ts, tl, hb, tn, scores, -1,
                    out_cards, out_seats)
            total += scores[observer] - base_score
        avg = total / n_worlds
        avgs[ci] = avg
        if first or avg < best_avg:
            best_card = card
            best_avg = avg
            first = False
    return (best_card, n_fallbacks, n_failed, n_seeds - seed_off,
            _ST_OK)


@njit(cache=True)
def honest_decision_kernel(obs_hand, opp_seats, real_cards, real_seats,
                           n_real, trick_cards0, trick_seats0, trick_len0,
                           hearts_broken0, trick_number0, scores0, observer,
                           arrangements, legal_cards, n_inner, level,
                           respects_voids, max_restarts, seeds):
    """Phase 2.8 entry point. Behaviour unchanged; body moved to `_honest_core`.

    The split changed no line of the loop itself, and the 2.8 bitwise pins run
    against this entry point, so they verify the split as well.
    """
    n_cand = legal_cards.shape[0]
    avgs = np.zeros(n_cand, dtype=np.float64)
    best_card, n_fb, n_fs, n_used, status = _honest_core(
        obs_hand, opp_seats, real_cards, real_seats, n_real, trick_cards0,
        trick_seats0, trick_len0, hearts_broken0, trick_number0, scores0,
        observer, arrangements, arrangements.shape[0], legal_cards, n_cand,
        n_inner, level, respects_voids, max_restarts, seeds, 0, avgs)
    return best_card, n_fb, n_fs, n_used, avgs, status


def _raise_status(status: int) -> None:
    """Turn a kernel status code into the Python path's own exception."""
    if status == _ST_NO_HOLDER:
        raise AssertionError("card has no possible holder (bad zeroing)")
    if status == _ST_COL_COLLAPSE:
        raise AssertionError("column collapsed to zero during rebalance")
    if status == _ST_NO_CONVERGE:
        raise AssertionError("rebalance did not converge")
    if status == _ST_BOUNDS:
        raise AssertionError()
    if status == _ST_BAD_STOP:
        raise AssertionError(
            "fused decision stopped at a forced move: the kernel's `_legal` "
            "and the engine's `legal_moves` disagree")
    if status == _ST_SEEDS:
        raise AssertionError(
            "fused decision ran out of pre-drawn seeds: the n_cand x n_worlds "
            "upper bound is wrong")


def honest_evidence(obs_hand, all_plays, observer, level, respects_voids):
    """Test/debug adapter for the in-kernel evidence extraction.

    Returns `(probs, voids, hand_sizes, unseen_mask)` in exactly the shapes
    `BeliefTable` carries, so gate 1 can compare them field by field.
    """
    plays_cards = np.zeros(52, dtype=np.int64)
    plays_seats = np.zeros(52, dtype=np.int64)
    for k, (s, c) in enumerate(all_plays):
        plays_cards[k] = c
        plays_seats[k] = s
    probs = np.zeros((3, 52), dtype=np.float64)
    hand_sizes = np.zeros(3, dtype=np.int64)
    void_masks = np.zeros(3, dtype=np.int64)
    unseen_cards = np.zeros(52, dtype=np.int64)
    sizes = np.zeros(3, dtype=np.float64)
    n_unseen, status = _evidence(
        np.int64(obs_hand), plays_cards, plays_seats, len(all_plays),
        np.int64(observer), int(level), bool(respects_voids), probs,
        hand_sizes, void_masks, unseen_cards, sizes)
    _raise_status(status)
    unseen_mask = 0
    for j in range(n_unseen):
        unseen_mask |= 1 << int(unseen_cards[j])
    voids = [{s for s in range(4) if (int(void_masks[i]) >> s) & 1}
             for i in range(3)]
    return probs, voids, [int(x) for x in hand_sizes], unseen_mask


def honest_decision(view, arrangements, observer: int, n_inner: int,
                    level: int, respects_voids: bool, rng,
                    max_restarts: int = 200, candidates=None):
    """Adapter for `honest_decision_kernel`.

    `arrangements` is the OUTER worlds (a list of 3 bitmasks each, in
    `BeliefTable.opponent_seats` order) -- so the choice-aware posterior path
    fuses exactly as the constraint-sampler path does.

    PRE-DRAWN SEEDS. The unfused path draws one `rng.integers(2**63)` per
    inner re-determinization, and how many of those there are is only known
    once the playouts have been run. So: snapshot the Generator, pre-draw the
    exact upper bound `n_cand * n_worlds` with the SAME scalar call, let the
    kernel consume `m` of them in order, then restore the snapshot and re-draw
    exactly `m`. The Generator therefore ends in the state the unfused path
    would leave it in, having produced the same values in the same order, and
    every world/playout/mean/card downstream is bitwise identical.

    `candidates` (optional) replaces the legal-move list with a shorter,
    still-ASCENDING one -- the equivalence-class representatives from
    `search/grouping.py`. Ascending order is load-bearing: the kernel's
    tie-break keeps the lowest card index. `None` keeps the exact 2.8
    behaviour, so every bitwise pin holds with grouping off. NOTE that with
    grouping ON the pre-drawn seed count `n_cand * n_worlds` shrinks, so the
    rng end state legitimately differs from the ungrouped path -- grouping is
    NOT bitwise-compatible with existing rows.

    Returns `(best_card, inner_fallbacks, inner_failed_samples, avgs)`, where
    `avgs` is one mean score per candidate in ascending card order.
    """
    all_plays = list(view.history) + list(view.current_trick)
    n_real = len(all_plays)
    real_cards = np.zeros(52, dtype=np.int64)
    real_seats = np.zeros(52, dtype=np.int64)
    trick_cards0 = np.zeros(4, dtype=np.int64)
    trick_seats0 = np.zeros(4, dtype=np.int64)
    for k, (s, c) in enumerate(all_plays):
        real_cards[k] = c
        real_seats[k] = s
    for i, (s, c) in enumerate(view.current_trick):
        trick_cards0[i] = c
        trick_seats0[i] = s

    from openhearts.engine import cards as _cards  # local: avoids cycle

    if candidates is None:
        candidates = _cards.cards_in(view.legal_moves)
    legal_cards = np.array(candidates, dtype=np.int64)
    arr = np.array([[int(h) for h in w] for w in arrangements],
                   dtype=np.int64).reshape(len(arrangements), 3)
    opp_seats = np.array([(observer + 1 + i) % 4 for i in range(3)],
                         dtype=np.int64)
    n_max = int(legal_cards.shape[0]) * int(arr.shape[0])
    state0 = rng.bit_generator.state
    seeds = np.empty(n_max, dtype=np.int64)
    for k in range(n_max):
        seeds[k] = rng.integers(2**63)

    best, n_fb, n_fs, n_used, avgs, status = honest_decision_kernel(
        np.int64(view.hand), opp_seats, real_cards, real_seats, n_real,
        trick_cards0, trick_seats0, len(view.current_trick),
        bool(view.hearts_broken), int(view.trick_number),
        np.array(view.scores, dtype=np.int64), np.int64(observer),
        arr, legal_cards, int(n_inner), int(level), bool(respects_voids),
        int(max_restarts), seeds)

    # Rewind to before the pre-draw and advance by exactly the number the
    # unfused path would have drawn. Restoring first and re-drawing (rather
    # than trying to "un-draw") is what makes the end state exact.
    rng.bit_generator.state = state0
    for _ in range(int(n_used)):
        rng.integers(2**63)

    _raise_status(status)
    assert best >= 0, "fused honest decision returned no card"
    return int(best), int(n_fb), int(n_fs), avgs


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


def sample_arrangements(table, rng, n_samples: int, max_restarts: int = 200):
    """Batch-sample a decision's arrangements with the compiled sampler.

    Returns `(hands_list, n_failed)` where each entry of `hands_list` is a
    `[int, int, int]` bitmask triple in `table.opponent_seats` order -- the
    same shape `sampler.sample_arrangement` returns, so callers just swap the
    loop for one call.

    HONESTY NOTE: this is NOT bitwise-compatible with the Python sampler. It
    takes ONE draw from the caller's Generator per decision and uses it to
    seed numba's internal RNG, instead of consuming the Generator once per
    card. Results are fully deterministic given the caller's rng state, but
    they are a different stream. That is why `jit_sampler` defaults to False
    on SearchPlayer: every existing bitwise-pinned row keeps the Python path.
    """
    from openhearts.engine import cards as _cards  # local: avoids import cycle

    seed = int(rng.integers(2**63))
    unseen = np.array(_cards.cards_in(table.unseen_mask), dtype=np.int64)
    void_masks = np.array(
        [sum(1 << s for s in table.voids[i]) for i in range(3)],
        dtype=np.int64)
    hand_sizes = np.array(table.hand_sizes, dtype=np.int64)
    probs = np.ascontiguousarray(table.probs, dtype=np.float64)
    hands, _attempts, n_ok = sample_arrangements_batch(
        probs, void_masks, hand_sizes, unseen, n_samples, max_restarts, seed)
    out = [[int(hands[j, 0]), int(hands[j, 1]), int(hands[j, 2])]
           for j in range(n_ok)]
    return out, n_samples - n_ok


class FusedAuditContext:
    """Everything `draw_audit_batch` needs that is constant across chunks.

    In the hard zone `from_view` runs hundreds of chunks; hoisting these array
    builds out of the loop is what makes the fusion pay. Holding them fixed
    also keeps the chunk loop bitwise identical to Phase 3.5's.
    """

    __slots__ = ("probs", "void_masks", "hand_sizes", "unseen", "obs_hand",
                 "opp_seats", "plays_cards", "plays_seats", "observer",
                 "epsilon", "max_restarts")

    def __init__(self, probs, void_masks, hand_sizes, unseen, obs_hand,
                 opp_seats, plays_cards, plays_seats, observer, epsilon,
                 max_restarts):
        self.probs = probs
        self.void_masks = void_masks
        self.hand_sizes = hand_sizes
        self.unseen = unseen
        self.obs_hand = obs_hand
        self.opp_seats = opp_seats
        self.plays_cards = plays_cards
        self.plays_seats = plays_seats
        self.observer = observer
        self.epsilon = epsilon
        self.max_restarts = max_restarts


def fused_audit_context(view, table, observer: int, epsilon: float,
                        max_restarts: int = 200) -> FusedAuditContext:
    """Build the per-call context AND run the once-per-call structural checks.

    WHY ONCE PER CALL IS ENOUGH (Phase 3.6). The Python path
    (`belief.weighted._reconstruct_original_hands`) asserted, for every
    candidate world, that each reconstructed seat holds exactly 13 cards, that
    the four hands are disjoint, that their union is the full deck, and that
    the observer's current hand is contained in its reconstructed hand. Every
    one of those is a property of the FRAME (observer hand, played cards,
    unseen mask, hand sizes) combined with the sampler's contract, not of the
    particular world:

    * checked here, once: `view.hand`, the played cards and `unseen_mask` are
      pairwise disjoint and union to `FULL_DECK`; `sum(hand_sizes)` equals the
      number of unseen cards; and for every seat, (cards it has played) +
      (its remaining hand size) == 13.
    * guaranteed by `sample_arrangements_batch`: an emitted arrangement is a
      partition of exactly `unseen_cards` among the three opponents with
      exactly `hand_sizes[i]` cards each (it only emits when no card was left
      unplaced and every `remaining[i]` hit 0).

    Together those give disjointness, the 13-card counts, the full-deck union
    and observer containment for EVERY candidate -- so the per-candidate
    asserts are subsumed, not dropped.

    The 2-clubs check (`_assert_two_clubs_leads`) is likewise hoisted: the
    first observed play is always the 2 of clubs, so reconstruction always
    hands 2C to `all_plays[0][0]`, making the check world-independent. We
    assert the load-bearing fact (`all_plays[0][1] == TWO_CLUBS`) here rather
    than assume it.
    """
    from openhearts.engine import cards as _cards  # local: avoids import cycle

    all_plays = list(view.history) + list(view.current_trick)
    played = 0
    played_counts = [0, 0, 0, 0]
    for s, c in all_plays:
        b = _cards.bit(c)
        assert not (played & b), f"card {c} appears twice in the observed play"
        played |= b
        played_counts[s] += 1

    unseen_mask = int(table.unseen_mask)
    assert not (view.hand & played), (
        "observer still holds an already-played card")
    assert not (view.hand & unseen_mask), (
        "observer's own hand overlaps the unseen mask")
    assert not (played & unseen_mask), (
        "an already-played card is marked unseen")
    assert (view.hand | played | unseen_mask) == _cards.FULL_DECK, (
        "observer hand + played + unseen is not a full deck")

    unseen = np.array(_cards.cards_in(unseen_mask), dtype=np.int64)
    hand_sizes = np.array(table.hand_sizes, dtype=np.int64)
    assert int(hand_sizes.sum()) == unseen.shape[0], (
        "opponent hand sizes do not account for every unseen card")
    opp_seats = [int(s) for s in table.opponent_seats]
    assert opp_seats == [(observer + 1 + i) % 4 for i in range(3)], (
        "opponent_seats is not in (observer+1+i) %% 4 order")
    for i, s in enumerate(opp_seats):
        assert played_counts[s] + int(hand_sizes[i]) == 13, (
            f"seat {s} played {played_counts[s]} + holds {hand_sizes[i]} "
            f"!= 13")
    assert played_counts[observer] + _popcount_py(view.hand) == 13, (
        "observer's played + held cards do not make 13")

    if all_plays:
        assert all_plays[0][1] == _cards.TWO_CLUBS, (
            f"observed game does not open on the 2 of clubs "
            f"(first play {all_plays[0]})")

    void_masks = np.array(
        [sum(1 << s for s in table.voids[i]) for i in range(3)],
        dtype=np.int64)
    return FusedAuditContext(
        probs=np.ascontiguousarray(table.probs, dtype=np.float64),
        void_masks=void_masks,
        hand_sizes=hand_sizes,
        unseen=unseen,
        obs_hand=np.int64(view.hand),
        opp_seats=np.array(opp_seats, dtype=np.int64),
        plays_cards=np.array([c for _s, c in all_plays], dtype=np.int64),
        plays_seats=np.array([s for s, _c in all_plays], dtype=np.int64),
        observer=np.int64(observer),
        epsilon=float(epsilon),
        max_restarts=max_restarts,
    )


def _popcount_py(mask: int) -> int:
    return bin(mask).count("1")


def draw_audit_chunk(ctx: FusedAuditContext, rng, n_candidates: int,
                     out_probs, total_w: float = 0.0,
                     total_w2: float = 0.0):
    """One fused chunk: draw, audit, accumulate into `out_probs`.

    Takes exactly ONE `rng.integers(2**63)` draw, seeding the compiled
    sampler -- identical rng consumption to Phase 3.5's
    `kernel.sample_arrangements` for the same chunk.

    `total_w` / `total_w2` are the caller's running sums (see
    `draw_audit_batch`); the updated values are returned.

    Returns `(worlds, weights, n_kept, total_w, total_w2)`; only the first
    `n_kept` rows of `worlds` / `weights` are meaningful.
    """
    seed = int(rng.integers(2**63))
    worlds = np.zeros((n_candidates, 3), dtype=np.int64)
    weights = np.zeros(n_candidates, dtype=np.float64)
    (_n_drawn, n_kept, total_w, total_w2, desync) = draw_audit_batch(
        ctx.probs, ctx.void_masks, ctx.hand_sizes, ctx.unseen,
        n_candidates, ctx.max_restarts, seed, ctx.obs_hand, ctx.opp_seats,
        ctx.plays_cards, ctx.plays_seats, ctx.observer, ctx.epsilon,
        out_probs, worlds, weights, float(total_w), float(total_w2))
    if desync:
        raise AssertionError(
            "replay desync: observed seat order does not match the engine's "
            "turn order in this world"
        )
    return worlds, weights, int(n_kept), float(total_w), float(total_w2)


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
