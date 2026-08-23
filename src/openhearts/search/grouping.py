"""Equivalence-class card grouping for honest search (ScrofaZero adoption (i)).

WHAT IT IS, IN CARDS. Say we hold 5d and 6d and nobody else can hold anything
between them (there is nothing between them). Playing the 5 and playing the 6
lead to the same hand: whatever happens after the 5, the identical thing
happens after the 6 with the two cards' roles swapped, and we score exactly
the same. Evaluating both is imagination spent twice on one question. So the
search evaluates the class ONCE, at its lowest member, and credits the answer
to every member of the class. Roughly 20-30% of candidates disappear, most of
them late in the hand where our remaining cards are long runs.

--------------------------------------------------------------------------
THE CONDITION
--------------------------------------------------------------------------
Fix a suit. Call a card LIVE-OTHER if it is neither in our hand nor in a
COMPLETED trick -- i.e. it is still in somebody else's hand, or it is sitting
in the current trick. (Current-trick cards must count as live: if the 8d is
already on the table, our 7d loses the trick and our 9d wins it -- they are
emphatically not interchangeable.)

Two cards of our hand that are CONSECUTIVE among our own cards of that suit
are joined when:
  (a) no live-other card of that suit has a rank strictly between them, and
  (b) neither of them is the queen of spades.
Classes are the connected runs of that joining relation, computed over our
whole HAND and then intersected with the legal mask.

Both clauses are load-bearing:

(a) is the actual interchangeability. A live-other card between them would
    order them differently against that card, so one wins tricks the other
    loses.

(b) is the points clause, and it is subtler than "Qs is worth 13 so never
    group Qs". Suppose we hold Js, Qs, Ks with nothing else live between. The
    isomorphism below maps the Ks-branch's hand onto the Js-branch's hand by
    shifting our cards in the interval DOWN one slot -- Qs would map to Js and
    Ks to Qs. That map does not preserve point value (13 -> 0), so the two
    branches score differently: in one we later eat the queen, in the other we
    play a worthless jack at that moment. Making the queen break every chain
    it touches kills {Js,Ks} for free, without a separate rule.

WHY THE CHAIN IS BUILT OVER THE HAND, NOT OVER THE LEGAL MASK. On the first
trick the points filter makes Qs illegal while Js and Ks stay legal. Chaining
over legal cards only would let the queen silently vanish from between them
and wrongly join Js to Ks. So: chain over the hand, then intersect.

(The intersection never splits a class: every legality filter in
`game.legal_moves` keys on suit and on point-bearing-ness only, and a class is
single-suit and uniform in point value -- it contains no Qs, and hearts are
all worth 1. The code asserts this rather than trusting it.)

--------------------------------------------------------------------------
THE PROOF (why the outcome is identical, not merely similar)
--------------------------------------------------------------------------
Let C be a class and X < Y two of its members. Compare the two real games:
one where we play X now, one where we play Y now. Define phi, the
ORDER-PRESERVING relabelling from "our hand after playing Y" to "our hand
after playing X": it fixes every card below X and above Y, and shifts each of
our cards in the interval (X, Y] down one slot within the class. (The naive
transposition X <-> Y is NOT the right map when we hold cards in between --
it is not order-preserving.)

phi preserves everything the game and the policies can see:

1. SUIT -- it moves cards only within one suit.
2. POINT VALUE -- a class is single-suit with no Qs, so every member is worth
   the same (0 for clubs/diamonds/non-queen spades, 1 for hearts). This is
   exactly the clause-(b) requirement.
3. ORDER AGAINST EVERY OTHER LIVE CARD -- phi moves cards only inside the
   interval [X, Y], and by clause (a) no live-other card of the suit lies
   inside it. So every rank comparison in `game.trick_winner` and in the
   heuristic's `_follow` (`c % 13 < win_rank`) gives the same answer.
4. ORDER AMONG OUR OWN CARDS -- phi is order-preserving by construction, so
   `_lowest` / `_highest` / suit-length popcounts inside `_choose`, `_lead`,
   `_follow`, `_discard` pick corresponding cards.
5. LEGALITY -- `legal_moves` is built from SUIT_MASK and POINTS_MASK plus
   trick/hearts-broken flags, all preserved by 1-2.

Hence the two games are isomorphic under phi: the opponents play LITERALLY
the same cards (phi fixes every card that is not ours), the same tricks are
won by the same seats, and our seat's score is identical. Zero accuracy loss,
by theorem.

The one policy input that is not a pure order/suit question is the heuristic's
lead rule, which skips leading spades while holding Qs with Ks or As still
unseen -- and `seen` is `played_mask | hand`, which differs between the two
branches for the OPPONENTS (one of X, Y is in `played` instead of our hand).
Three cases, all safe:
  - Qs is dead (played, or ours-and-already-gone... precisely: not in the
    acting seat's hand): the rule is gated on `(hand >> QS) & 1`, so it never
    fires for that seat at all.
  - Qs is OURS: only we consult the rule, and OUR `seen = played | hand` is
    invariant under the swap -- X and Y are both in `played u hand` either
    way, so the set is literally identical.
  - Qs is an OPPONENT'S: then Qs is a live-other card, so by clause (a) any
    class containing Ks or As collapses to a singleton or to exactly {Ks, As}
    (both ours, nothing live between). For {Ks, As}, whichever we play the
    other stays in our hand, so "Ks unseen OR As unseen" is TRUE in both
    branches, for every seat. Same decision.
This is why NO extra restriction on Ks/As is needed, and why e.g. {Js, As} is
groupable once Qs and Ks are both played.

RE-DETERMINIZATION IS ALSO SAFE (this is what makes the theorem survive
honest search rather than just plain playouts). At the interception point the
imagined view's UNSEEN set is `52 - our hand - played`; both X and Y are in
`hand u played` in either branch, so the unseen set, the hand sizes and the
void evidence are IDENTICAL. Same belief table, same sampled inner worlds
given the same seed, same inner decision. See gate A in
`tests/test_equivalence_gate.py`, which asserts bit-identical means.

--------------------------------------------------------------------------
WHICH MEMBER WE PLAY, AND THE ONE CAVEAT
--------------------------------------------------------------------------
Representative = the LOWEST member of the class. Combined with the search's
existing tie-break (strict `<` over ascending card index, so ties go to the
lowest index), this makes the grouped choice equal to the ungrouped choice
whenever both see the same scores.

CAVEAT, DOCUMENTED NOT SOLVED: which member we play cannot change anything in
our simulator, but it is information at a real table. Always playing the
lowest of a run is a habit a human could read ("it never plays the 9 from
9-8-7, so when it does play the 9, it has no 8"). Grouping also deletes the
accidental randomness the C0 probe measured on exactly these
provably-indifferent choices. Neither is a bug; both are for the lead to weigh
before this is ever enabled in a measured row.

--------------------------------------------------------------------------
The njit twin below is an exact port kept for the house Python/kernel-pin
pattern. It is NOT on the hot path: grouping is computed once per decision in
Python (microseconds against a ~30 ms fused decision) and passed to the fused
kernel as a shortened candidate list.
"""
import numpy as np

from openhearts.engine import cards
from openhearts.engine.kernel import njit

_QS = cards.QUEEN_SPADES
_FULL = cards.FULL_DECK


def dead_mask_from_view(view) -> int:
    """Cards in COMPLETED tricks. Current-trick cards are deliberately NOT
    dead: a card on the table still orders our cards against it."""
    mask = 0
    for _seat, card in view.history:
        mask |= 1 << card
    return mask


def _classes_py(hand: int, legal: int, dead: int):
    """Reference implementation. Returns `(rep_mask, rep_of)` where
    `rep_of[c]` is the representative card of `c`'s class for every legal
    card `c`, and -1 elsewhere."""
    live_other = _FULL & ~hand & ~dead
    rep_of = np.full(52, -1, dtype=np.int64)
    rep_mask = 0
    for s in range(4):
        suit_hand = hand & cards.SUIT_MASK[s]
        own = cards.cards_in(suit_hand)   # ascending
        if not own:
            continue
        start = 0
        for i in range(len(own) + 1):
            cut = i == len(own)
            if not cut:
                if i == start:
                    continue
                a, b = own[i - 1], own[i]
                between = 0
                for r in range(a + 1, b):
                    between |= 1 << r
                joined = (between & live_other) == 0 and a != _QS and b != _QS
                cut = not joined
            if cut:
                members = own[start:i]
                start = i
                mask = 0
                for c in members:
                    mask |= 1 << c
                lm = mask & legal
                if lm == 0:
                    continue
                assert lm == mask, (
                    "legality split an equivalence class -- impossible if "
                    "legal_moves keys only on suit and point-bearing-ness")
                rep = cards.cards_in(lm)[0]
                rep_mask |= 1 << rep
                for c in members:
                    rep_of[c] = rep
    return rep_mask, rep_of


@njit(cache=True)
def _classes_kernel(hand, legal, dead):
    """njit twin of `_classes_py` (pinned by tests, not on the hot path)."""
    live_other = np.int64((1 << 52) - 1) & ~hand & ~dead
    rep_of = np.full(52, -1, dtype=np.int64)
    rep_mask = np.int64(0)
    for s in range(4):
        prev = -1                    # previous card of OUR hand in this suit
        run_mask = np.int64(0)
        for r in range(14):          # 13 ranks + one flush pass
            flush = r == 13          # end of suit always flushes
            c = s * 13 + r
            if r < 13 and ((hand >> c) & 1) == 1:
                joined = False
                if prev >= 0:
                    between = np.int64(0)
                    for k in range(prev + 1, c):
                        between |= np.int64(1) << k
                    joined = ((between & live_other) == 0 and prev != _QS
                              and c != _QS)
                flush = not joined
                if joined:
                    run_mask |= np.int64(1) << c
            if flush and run_mask != 0:
                lm = run_mask & legal
                if lm != 0:
                    rep = -1
                    for k in range(52):
                        if (lm >> k) & 1:
                            rep = k
                            break
                    rep_mask |= np.int64(1) << rep
                    for k in range(52):
                        if (run_mask >> k) & 1:
                            rep_of[k] = rep
                run_mask = np.int64(0)
            if r < 13 and ((hand >> c) & 1) == 1:
                if run_mask == 0:
                    run_mask = np.int64(1) << c
                prev = c
    return rep_mask, rep_of


def equivalence_classes(hand: int, legal: int, dead: int, use_jit: bool = False):
    """Equivalence classes of the LEGAL cards.

    `hand`   our 52-bit hand mask.
    `legal`  the legal-move mask (a submask of `hand`).
    `dead`   cards in COMPLETED tricks (see `dead_mask_from_view`).

    Returns `(rep_mask, rep_of)`: the mask of representatives to evaluate, and
    a length-52 array giving each legal card's representative (-1 elsewhere).
    """
    if use_jit:
        rep_mask, rep_of = _classes_kernel(np.int64(hand), np.int64(legal),
                                           np.int64(dead))
        return int(rep_mask), rep_of
    return _classes_py(hand, legal, dead)


def class_members(rep_of, rep: int) -> list:
    """The legal cards credited to representative `rep`, ascending."""
    return [c for c in range(52) if rep_of[c] == rep]
