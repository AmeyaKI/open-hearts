"""ExpertPlayer — Phase 7A reference implementation of EXPERT_RULEBOOK_DRAFT.md.

WHAT THIS IS.  A rule-following Hearts player whose decisions come from the
owner-approved expert catalogue: mandatory legality/public-fact/queen
reflexes that every expert shares, then bounded, weighted style preferences
compared stage by stage, then a total fallback that always returns a legal
card.  The catalogue is the specification; PHASE7_PLAN.md's 7A-1 spec (§2)
pins every interpretation the catalogue leaves open.  Where a comment below
says *interpretation*, that is a pinned choice, numbered in the plan.

WHAT THIS IS NOT.  Not a search player, not a belief model, not a claim of
human fidelity or measured strength.  "Expert" names a repertoire.

INFORMATION BOUNDARY.  `action_distribution(view, params)` is a pure function
of the eight `PlayerView` fields and the parameter record.  It keeps no state
between calls: every public fact (played cards, opponents' outstanding cards
per suit, queen state, public voids) is recomputed from the view.  Two full
game states that induce the same view produce the same distribution --
enforced by tests/test_expert.py.  *Interpretation:* the catalogue's
"carry masks incrementally, reset at branch boundaries" describes the future
compiled 7C kernel; this Python reference is stateless on purpose so there is
no branch-local state to get wrong, and the kernel later gates bitwise
against it.

DETERMINISM.  `choose` draws from the player's rng EXACTLY ONCE per decision,
and only when `tie_mode == "seeded_uniform"` and the final tie set has more
than one card; otherwise it draws nothing.  No noise crosses a stage.

STAGES (catalogue §3; lexicographic -- a later stage never overturns an
earlier one; at each scoring stage all maximizers survive):
  0  legality, forced move, E5 all-26-captured, F3 position dispatch
  1  mandatory queen reflexes: Q5, then Q3/F2's queen branch
  2  mandatory current-point avoidance
  3A L1 opening club; D1 exposed-high-spade disposal
  3B F5 refuse a stranded clean win
  3C D3/E3 exit retention (FOLLOWS AND DISCARDS ONLY -- owner ruling R1,
     2026-09-05; on leads the same feature votes at 3D)
  3D weighted planning: L2(xX2,E4) L4 L5 Q1 Q2/Q4 F1(Q6/H3) F2 F4 X1 H4 H5
     D1-ordinary D2(E4) and lead-side exit retention
  4  total fallback, then tie_mode
"""
from dataclasses import replace

import numpy as np

from openhearts.engine import cards
from openhearts.engine.game import legal_moves, trick_points
from openhearts.engine.state import PlayerView
from openhearts.players.expert_population import ExpertParams, REFERENCE_PARAMS

CLUBS, DIAMONDS, SPADES, HEARTS = cards.CLUBS, cards.DIAMONDS, cards.SPADES, cards.HEARTS
QS = cards.QUEEN_SPADES          # 36
KS, AS = 37, 38
KH, AH = 50, 51
_HIGH_SPADES = cards.bit(KS) | cards.bit(AS)
_LOW_SPADES = cards.SUIT_MASK[SPADES] & (cards.bit(QS) - 1)   # 2..J of spades
_SUIT_NAMES = ("clubs", "diamonds", "spades", "hearts")

STAGE_NAMES = ("0", "1", "2", "3A", "3B", "3C", "3D", "4", "tie")


# ---------------------------------------------------------------- bit helpers

def _count(mask: int) -> int:
    return bin(mask).count("1")


def _above(mask: int, card: int) -> int:
    """Cards in `mask` of `card`'s suit ranked strictly above it."""
    return mask & cards.SUIT_MASK[cards.suit(card)] & ~((1 << (card + 1)) - 1)


def _below(mask: int, card: int) -> int:
    return mask & cards.SUIT_MASK[cards.suit(card)] & ((1 << card) - 1)


def _suit_len(hand: int, s: int) -> int:
    return _count(hand & cards.SUIT_MASK[s])


def _lowest(cs):
    return min(cs, key=cards.rank)


def _highest(cs):
    return max(cs, key=cards.rank)


def fixed_order_key(card: int):
    """tie_mode='fixed': rank ascending, then C/D/S/H."""
    return (cards.rank(card), cards.suit(card))


# ---------------------------------------------------------------- public facts

def public_voids(view: PlayerView):
    """E1: per-seat bitmask of suits that seat has publicly failed to follow,
    from completed tricks AND the partial current trick.  Nothing else is a
    void: never having failed means unknown."""
    voids = [0, 0, 0, 0]
    plays = list(view.history) + list(view.current_trick)
    for i in range(0, len(plays), 4):
        trick = plays[i:i + 4]
        led = cards.suit(trick[0][1])
        for seat, c in trick[1:]:
            if cards.suit(c) != led:
                voids[seat] |= 1 << led
    return voids


def played_mask(view: PlayerView) -> int:
    m = 0
    for _, c in view.history:
        m |= cards.bit(c)
    for _, c in view.current_trick:
        m |= cards.bit(c)
    return m


def outstanding(hand: int, played: int):
    """E2: O[s] -- suit s minus own hand minus played cards.  Identities
    outside our hand, NOT their seat allocation."""
    unseen = ~(hand | played)
    return [cards.SUIT_MASK[s] & unseen & cards.FULL_DECK for s in range(4)]


def queen_state(hand: int, view: PlayerView) -> str:
    """Exactly one of owned / on_table / captured / unseen (Q7 switches at
    capture, i.e. when Q♠ enters COMPLETED history)."""
    if hand & cards.bit(QS):
        return "owned"
    for _, c in view.current_trick:
        if c == QS:
            return "on_table"
    for _, c in view.history:
        if c == QS:
            return "captured"
    return "unseen"


def is_exit(card: int, hand: int, played: int, hearts_broken: bool,
            trick_number: int) -> bool:
    """Exit predicate: legal to lead, at least one opponent card in the suit
    remains, and EVERY remaining opponent card in the suit is higher.
    *Interpretation:* hypothetical-lead legality uses `max(t, 1)` so the
    trick-1 snapshot means "as a lead at the next opportunity", not the
    forced 2♣."""
    lead = legal_moves(hand, (), hearts_broken, max(trick_number, 1))
    if not lead & cards.bit(card):
        return False
    o = cards.SUIT_MASK[cards.suit(card)] & ~(hand | played) & cards.FULL_DECK
    if o == 0:
        return False
    return (o & ((1 << (card + 1)) - 1)) == 0


def snapshot_exits(hand: int, played: int, hearts_broken: bool,
                   trick_number: int) -> int:
    """Bitmask of exit-predicate cards in `hand` right now."""
    m = 0
    for c in cards.cards_in(hand):
        if is_exit(c, hand, played, hearts_broken, trick_number):
            m |= cards.bit(c)
    return m


class Facts:
    """Everything the catalogue's §2 defines, computed once per decision from
    the view alone."""

    def __init__(self, view: PlayerView):
        self.view = view
        self.hand = view.hand
        self.played = played_mask(view)
        self.O = outstanding(self.hand, self.played)
        self.queen = queen_state(self.hand, view)
        self.queen_live = self.queen != "captured"
        self.queen_unseen = self.queen == "unseen"
        self.voids = public_voids(view)
        self.t = view.trick_number
        self.hearts_broken = view.hearts_broken
        self.points_remain = sum(view.scores) < 26
        self.legal = cards.cards_in(view.legal_moves)
        self.n_low_spades = _count(self.hand & _LOW_SPADES)
        self.has_high_spade = bool(self.hand & _HIGH_SPADES)
        self.trick = view.current_trick
        self.trick_mask = 0
        for _, c in self.trick:
            self.trick_mask |= cards.bit(c)
        if self.trick:
            self.led = cards.suit(self.trick[0][1])
            self.position = len(self.trick) + 1
            self.win_rank = max(cards.rank(c) for _, c in self.trick
                                if cards.suit(c) == self.led)
            self.void_in_led = (self.hand & cards.SUIT_MASK[self.led]) == 0
            if self.void_in_led:
                self.kind = "discard"
            else:
                self.kind = "follow"
        else:
            self.led = None
            self.position = 1
            self.win_rank = None
            self.void_in_led = False
            self.kind = "lead"
        self._snap = None
        ranks = [cards.rank(c) for c in cards.cards_in(self.hand)]
        self.mean_rank = float(np.mean(ranks)) if ranks else 0.0
        self.min_rank = min(ranks) if ranks else 0

    # -- derived predicates --------------------------------------------
    @property
    def snap(self) -> int:
        """Snapshot exits over the full current hand (computed lazily)."""
        if self._snap is None:
            self._snap = snapshot_exits(self.hand, self.played,
                                        self.hearts_broken, self.t)
        return self._snap

    def retained(self, c: int) -> int:
        return _count(self.snap & ~cards.bit(c))

    def is_loser(self, c: int) -> bool:
        return (self.kind == "follow" and cards.rank(c) < self.win_rank)

    def is_winner(self, c: int) -> bool:
        return (self.kind == "follow" and cards.rank(c) > self.win_rank)

    def final_cost(self, c: int) -> int:
        """Fourth-hand candidate-inclusive trick points."""
        return trick_points(list(self.trick) + [(self.view.seat, c)])

    def projected_exits_after_win(self, c: int) -> int:
        hand2 = self.hand & ~cards.bit(c)
        played2 = self.played | cards.bit(c)
        hb2 = self.hearts_broken or cards.suit(c) == HEARTS or bool(
            self.trick_mask & cards.HEARTS_MASK)
        return snapshot_exits(hand2, played2, hb2, self.t + 1)

    def opp_void_count(self, s: int) -> int:
        return sum(1 for seat in range(4)
                   if seat != self.view.seat and self.voids[seat] & (1 << s))

    def is_exposed_high_spade(self, c: int) -> bool:
        return (c in (KS, AS) and self.queen_unseen
                and self.n_low_spades == 0)

    def is_guarded_high_spade(self, c: int) -> bool:
        return (c in (KS, AS) and self.queen_unseen
                and self.n_low_spades >= 1)

    def useful_void(self, c: int) -> bool:
        s = cards.suit(c)
        return (self.t <= 11 and _suit_len(self.hand, s) == 1
                and self.O[s] != 0)

    def e4_active(self, p: ExpertParams) -> bool:
        return (self.mean_rank >= p.high_hand_mean
                and self.min_rank > p.small_rank)


# ---------------------------------------------------------------- stage engine

def _keep_max(cands, score):
    best = max(score[c] for c in cands)
    return [c for c in cands if score[c] == best]


def _apply(stage, cands, contribs, explain):
    """`contribs` = list of (rule, {card: delta}).  Sum per card in the fixed
    rule order (so equal rule sets give bit-identical sums), keep maximizers,
    record what fired."""
    score = {c: 0.0 for c in cands}
    fired = []
    for rule, delta in contribs:
        hit = {c: d for c, d in delta.items() if c in score and d != 0.0}
        if not hit:
            continue
        fired.append((rule, hit))
        for c, d in hit.items():
            score[c] += d
    survivors = _keep_max(cands, score) if cands else cands
    explain["stages"].append((stage, list(survivors), fired))
    if len(survivors) == 1 and explain["returned_at"] is None:
        explain["returned_at"] = stage
    return survivors


def _decide(view: PlayerView, p: ExpertParams):
    """Run the ladder.  Returns (tie_set sorted by fixed order, explain)."""
    explain = {"stages": [], "kind": None, "returned_at": None}
    legal = cards.cards_in(view.legal_moves)
    assert legal, "ExpertPlayer.choose called at a terminal/illegal state"
    if len(legal) == 1:
        explain["kind"] = "forced"
        explain["returned_at"] = "0"
        return list(legal), explain

    f = Facts(view)
    explain["kind"] = f.kind

    # ---------------- stage 0: E5
    if not f.points_remain:
        explain["returned_at"] = "0/E5"
        return sorted(legal, key=fixed_order_key), explain

    cands = list(legal)

    # ---------------- stage 1: mandatory queen reflexes (successive filters)
    if QS in cands:
        # Q5: a free legal queen disposal -- void in the led suit, or a
        # played A♠/K♠ already beats it on a spade trick.
        free = (f.kind == "discard") or (
            f.kind == "follow" and f.led == SPADES
            and bool(f.trick_mask & _HIGH_SPADES))
        if free:
            explain["stages"].append(("1", [QS], [("Q5", {QS: 0.0})]))
            explain["returned_at"] = "1/Q5"
            return [QS], explain
        # F2's queen branch: never win with own Q♠ when a non-queen follow exists.
        if f.kind == "follow" and f.led == SPADES and f.is_winner(QS) \
                and len(cands) > 1:
            cands = [c for c in cands if c != QS]
            explain["stages"].append(("1", list(cands), [("F2q", {})]))
            if len(cands) == 1:
                explain["returned_at"] = "1"
    if f.kind == "follow" and f.led == SPADES and f.queen_unseen \
            and f.position < 4:
        # Q3 (follow): prefer sub-queen spades to A♠/K♠ while a seat can still act.
        sub = [c for c in cands if cards.rank(c) < cards.rank(QS)]
        if sub and len(sub) < len(cands):
            cands = sub
            explain["stages"].append(("1", list(cands), [("Q3f", {})]))
            if len(cands) == 1:
                explain["returned_at"] = "1"
    if f.kind == "lead" and f.queen_unseen:
        # Q3 (lead): avoid voluntary A♠/K♠ leads while the queen is out.
        alt = [c for c in cands if c not in (KS, AS)]
        if alt and len(alt) < len(cands):
            cands = alt
            explain["stages"].append(("1", list(cands), [("Q3l", {})]))
            if len(cands) == 1:
                explain["returned_at"] = "1"

    # ---------------- stage 2: mandatory current-point avoidance
    if f.kind == "follow":
        losers = [c for c in cands if f.is_loser(c)]
        winners = [c for c in cands if f.is_winner(c)]
        if losers and any(f.final_cost(w) > 0 for w in winners):
            cands = losers
            explain["stages"].append(("2", list(cands), [("S2", {})]))
            if len(cands) == 1:
                explain["returned_at"] = "2"

    if len(cands) == 1:
        explain["returned_at"] = "1-2" if explain["returned_at"] is None else explain["returned_at"]
        return list(cands), explain

    losers = [c for c in cands if f.is_loser(c)]
    winners = [c for c in cands if f.is_winner(c)]

    # ---------------- stage 3A: L1, D1-exposed
    contribs = []
    if f.kind == "follow" and f.t == 0 and f.led == CLUBS and p.w_L1 > 0:
        contribs.append(("L1", {_highest(cands): p.w_L1}))
    if f.kind == "discard" and p.w_exposed > 0:
        exposed = [c for c in cands if f.is_exposed_high_spade(c)]
        if exposed:
            contribs.append(("D1x", {_highest(exposed): p.w_exposed}))
    cands = _apply("3A", cands, contribs, explain)
    if len(cands) == 1:
        return list(cands), explain

    # ---------------- stage 3B: F5
    contribs = []
    if f.kind == "follow" and f.position == 4 and losers and p.w_F5 > 0:
        delta = {}
        for w in winners:
            if w in cands and f.final_cost(w) == 0 \
                    and f.projected_exits_after_win(w) == 0:
                delta[w] = -p.w_F5
        contribs.append(("F5", delta))
    cands = _apply("3B", cands, contribs, explain)
    if len(cands) == 1:
        return list(cands), explain

    # ---------------- stage 3C: exit retention on follows and discards (R1)
    contribs = []
    if f.kind in ("follow", "discard") and p.w_exit > 0 and f.snap:
        delta = {c: p.w_exit * min(f.retained(c), p.exit_count) / p.exit_count
                 for c in cands}
        if f.kind == "follow":
            ls = [c for c in cands if f.is_loser(c)]
            if ls:
                cap = max(delta[c] for c in ls)
                for c in cands:
                    if f.is_winner(c) and delta[c] > cap:
                        delta[c] = cap
        contribs.append(("EXIT", delta))
    cands = _apply("3C", cands, contribs, explain)
    if len(cands) == 1:
        return list(cands), explain

    # ---------------- stage 3D: weighted planning
    contribs = []
    e4 = f.e4_active(p)

    def boosted(w):
        return min(1.0, w * (1.0 + p.escape_boost)) if e4 else w

    if f.kind == "lead":
        # L2 x X2 (x E4)
        if p.w_L2 > 0 and f.t >= 1:
            elig = {}
            for c in cands:
                s = cards.suit(c)
                if f.O[s] == 0 or (s == SPADES and f.queen_live):
                    continue
                elig.setdefault(s, []).append(c)
            if elig:
                mult = max(0.0, 1.0 - (f.t - 1) / (p.void_window - 1))
                shortest = min(_suit_len(f.hand, s) for s in elig)
                delta = {_lowest(cs): boosted(p.w_L2) * mult
                         for s, cs in elig.items()
                         if _suit_len(f.hand, s) == shortest}
                contribs.append(("L2", delta))
        # exit retention on leads (R1: 3D)
        if p.w_exit > 0 and f.snap:
            contribs.append(("EXITl", {
                c: p.w_exit * min(f.retained(c), p.exit_count) / p.exit_count
                for c in cands}))
        # L4: cannot be overtaken
        if p.w_L4 > 0:
            contribs.append(("L4", {c: -p.w_L4 for c in cands
                                    if _above(f.O[cards.suit(c)], c) == 0}))
        # L5: void-exposed non-exit
        if p.w_L5 > 0:
            contribs.append(("L5", {
                c: -p.w_L5 for c in cands
                if f.opp_void_count(cards.suit(c)) >= p.void_lead_threshold
                and not is_exit(c, f.hand, f.played, f.hearts_broken, f.t)}))
        # Q1: hunt with low spades
        if (p.w_Q1 > 0 and f.queen_unseen
                and 1 <= f.t <= p.queen_hunt_until
                and _suit_len(f.hand, SPADES) >= p.hunt_min_spades
                and (f.hand & cards.SUIT_MASK[SPADES]) == (f.hand & _LOW_SPADES)):
            sp = [c for c in cands if cards.suit(c) == SPADES]
            if sp:
                contribs.append(("Q1", {_lowest(sp): p.w_Q1}))
        # H5: lowest heart within the heart candidates
        if p.w_H5 > 0:
            hs = [c for c in cands if cards.suit(c) == HEARTS]
            if len(hs) > 1:
                lo = _lowest(hs)
                contribs.append(("H5", {c: -p.w_H5 for c in hs if c != lo}))

    # Q2/Q4 shared guard feature (leads, discards; losers only on follows)
    if p.w_guard > 0 and f.queen_live and (f.hand & (cards.bit(QS) | _HIGH_SPADES)):
        if f.kind == "follow":
            targets = [c for c in cands if f.is_loser(c)]
        else:
            targets = list(cands)
        if targets:
            need = min(f.n_low_spades, p.guard_target)
            delta = {}
            for c in targets:
                low_after = f.n_low_spades - (1 if cards.bit(c) & _LOW_SPADES else 0)
                # Literal catalogue predicate: any play that keeps the guard
                # count earns the bonus -- including playing the Q♠/K♠/A♠
                # itself.  A stricter "liability still held" reading was tried
                # and rejected: it made GUARD fight D1's guarded-spade-first
                # ladder on the catalogue's own K♠/2♠/A♥ example.  Leading Q♠
                # is kept off by the fallback's queen-live avoidance.
                a = low_after >= need
                b = (f.kind == "lead" and (f.hand & cards.bit(QS)) and
                     (cards.suit(c) != SPADES
                      or is_exit(c, f.hand, f.played, f.hearts_broken, f.t)))
                if a or b:
                    delta[c] = p.w_guard
            contribs.append(("GUARD", delta))

    if f.kind == "follow":
        ls = [c for c in cands if f.is_loser(c)]
        ws = [c for c in cands if f.is_winner(c)]
        if ls:
            w = {SPADES: p.w_F1_spades, HEARTS: p.w_F1_hearts}.get(f.led, p.w_F1)
            if w > 0:
                contribs.append(("F1", {_highest(ls): w}))
        else:
            if p.w_F2 > 0:
                contribs.append(("F2", {_lowest(cands): p.w_F2}))
        if f.position == 4:
            clean = [w for w in ws if f.final_cost(w) == 0]
            if p.w_F4 > 0:
                contribs.append(("F4", {w: p.w_F4 for w in clean
                                        if f.projected_exits_after_win(w) != 0}))
            if p.w_X1 > 0 and f.led != HEARTS and clean:
                contribs.append(("X1", {_highest(clean): p.w_X1}))
            if p.w_H4 > 0 and f.led == HEARTS and not ls:
                top = None
                if f.hand & cards.bit(AH):
                    top = AH
                elif (f.hand & cards.bit(KH)) and (f.played & cards.bit(AH)):
                    top = KH
                if top is not None and top in cands \
                        and f.final_cost(top) <= p.ace_release_cost:
                    contribs.append(("H4", {top: p.w_H4}))

    if f.kind == "discard":
        # D1 ordinary ladder
        if p.w_disposal > 0:
            guarded = [c for c in cands if f.is_guarded_high_spade(c)]
            hearts = [c for c in cands if cards.suit(c) == HEARTS]
            order = [guarded, hearts] if p.guarded_spade_first else [hearts, guarded]
            cd = {}
            for c in cands:
                s = cards.suit(c)
                if s in (CLUBS, DIAMONDS) and f.O[s] != 0:
                    cd.setdefault(s, []).append(c)
            if cd:
                shortest = min(_suit_len(f.hand, s) for s in cd)
                cd_cat = [c for s, cs in cd.items() for c in cs
                          if _suit_len(f.hand, s) == shortest]
            else:
                cd_cat = []
            order += [cd_cat, list(cands)]
            for cat in order:
                if cat:
                    top_rank = max(cards.rank(c) for c in cat)
                    contribs.append(("D1", {c: p.w_disposal for c in cat
                                            if cards.rank(c) == top_rank}))
                    break
        # D2: useful new void, not the sole snapshot exit
        if p.w_D2 > 0:
            contribs.append(("D2", {
                c: boosted(p.w_D2) for c in cands
                if f.useful_void(c) and f.snap != cards.bit(c)}))

    cands = _apply("3D", cands, contribs, explain)
    if len(cands) == 1:
        return list(cands), explain

    # ---------------- stage 4: total fallback
    ties = _fallback(f, cands)
    explain["stages"].append(("4", list(ties), []))
    explain["returned_at"] = "4" if len(ties) == 1 else "tie"
    return sorted(ties, key=fixed_order_key), explain


def _fallback(f: Facts, cands):
    """Catalogue §3 'Total fallback over the surviving candidates'.  Returns
    the tie set (usually one card)."""
    if f.kind == "follow":
        ls = [c for c in cands if f.is_loser(c)]
        if ls:
            return [_highest(ls)]
        nq = [c for c in cands if c != QS]
        if nq:
            return [_lowest(nq)]
        return [QS]
    if f.kind == "discard":
        for cat in (
            [c for c in cands if c == QS],
            [c for c in cands if f.is_exposed_high_spade(c)],
            [c for c in cands if cards.suit(c) == HEARTS],
            [c for c in cands if c in (KS, AS) and f.queen_unseen],
            list(cands),
        ):
            if cat:
                top = max(cards.rank(c) for c in cat)
                return [c for c in cat if cards.rank(c) == top]
        return list(cands)
    # lead
    exits = [c for c in cands
             if is_exit(c, f.hand, f.played, f.hearts_broken, f.t)]
    if exits:
        cands = exits
    if f.queen_live:
        safe = [c for c in cands if c not in (QS, KS, AS)]
        if safe:
            cands = safe
    by_suit = {}
    for c in cands:
        by_suit.setdefault(cards.suit(c), []).append(c)
    shortest = min(_suit_len(f.hand, s) for s in by_suit)
    lows = [_lowest(cs) for s, cs in by_suit.items()
            if _suit_len(f.hand, s) == shortest]
    lo = min(cards.rank(c) for c in lows)
    return [c for c in lows if cards.rank(c) == lo]


# ---------------------------------------------------------------- public API

def action_distribution(view: PlayerView, params: ExpertParams):
    """Side-effect-free legal action distribution: (cards, probs).
    One-hot under tie_mode='fixed'; uniform over the final tie set under
    'seeded_uniform'.  Support is a subset of the legal moves and sums to 1."""
    ties, _ = _decide(view, params)
    if len(ties) == 1 or params.tie_mode == "fixed":
        return [min(ties, key=fixed_order_key)], [1.0]
    n = len(ties)
    return list(ties), [1.0 / n] * n


def greedy_pick(view: PlayerView, params: ExpertParams) -> int:
    """The tie_mode='fixed' choice for any record (divergence gate)."""
    ties, _ = _decide(view, params)
    return min(ties, key=fixed_order_key)


def explain(view: PlayerView, params: ExpertParams) -> dict:
    """Diagnostics only: survivors per stage, rules that contributed, and
    where the decision was settled.  Never consulted by `choose`."""
    ties, ex = _decide(view, params)
    ex["ties"] = ties
    ex["fired"] = sorted({rule for _, _, fired in ex["stages"]
                          for rule, _ in fired})
    return ex


def action_changing_rules(view: PlayerView, params: ExpertParams):
    """Which optional weights, if zeroed one at a time, change the greedy
    choice at this view (the 'effective width' instrument)."""
    from openhearts.players.expert_population import OPTIONAL_WEIGHTS
    base = greedy_pick(view, params)
    out = []
    for w in OPTIONAL_WEIGHTS:
        if getattr(params, w) > 0:
            if greedy_pick(view, replace(params, **{w: 0.0})) != base:
                out.append(w)
    return out


class ExpertPlayer:
    """Rule-following expert.  Sees only PlayerView."""

    def __init__(self, rng, params: ExpertParams = REFERENCE_PARAMS):
        self.rng = rng
        self.params = params

    def choose(self, view: PlayerView) -> int:
        ties, _ = _decide(view, self.params)
        if len(ties) == 1 or self.params.tie_mode == "fixed":
            return min(ties, key=fixed_order_key)
        return ties[int(self.rng.integers(len(ties)))]

    def greedy_choice(self, view: PlayerView) -> int:
        return greedy_pick(view, self.params)

    def action_distribution(self, view: PlayerView):
        return action_distribution(view, self.params)
