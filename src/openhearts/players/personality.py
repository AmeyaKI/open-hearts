"""Parameterized "unpredictable" opponents — the Phase-5 population.

WHY THIS EXISTS.  Phase 4's league showed that honest-CHOICE's crown is
heuristic-specific: its choice posterior assumes every opponent runs the house
heuristic script, and collapses when they don't.  Phase 5 trains a learned
model of how *human-like* players choose cards.  That needs a population of
opponents that (a) is not the heuristic, (b) is not uniform noise, and (c) is
INTERNALLY CONSISTENT — each personality has stable tendencies you could learn
from watching it.  That is what a `PersonalityParams` draw is.

DESIGN SHAPE (fixed by the plan, not a free choice).  A personality is NOT a
`HeuristicPlayer` wrapped in noise — that family already exists as
`RandomizedHeuristic` and serves as an anchor.  Instead every legal card is
scored by hand-coded features whose WEIGHTS are the personality's parameters,
and the card is drawn from a softmax over those scores (plus an epsilon roll
that plays a uniformly random legal card outright).  Two personalities with
opposite `hoard` signs genuinely shed opposite cards; they are not the same
policy at different noise levels.

INFORMATION BOUNDARY.  `choose` and `greedy_choice` take a `PlayerView` and
nothing else.  Everything the scorer knows is derived from the view: own hand,
completed history, the current trick, hearts-broken, trick number, scores.

DETERMINISM.  Given (rng state, params) the choice sequence is fixed.  All
randomness flows through the player's own `rng`, in a fixed order: the epsilon
roll first, then (if it did not fire) the softmax draw.

THE PARAMETER-DRAW ORDER IS A FROZEN CONTRACT.  Task 2 writes ~5M decision
rows keyed by personality id; Tasks 3 and 5 re-derive params from those ids.
Reordering or inserting a draw inside `sample_personality` would silently
change what every previously generated id means.  New axes may only be
APPENDED at the end of the draw sequence.
"""
from dataclasses import dataclass, fields

import numpy as np

from openhearts.engine import cards
from openhearts.engine.state import PlayerView

# Anchor players live on the TRAIN side of the population only, never
# held-out (they are the families we already have; a learned model must not be
# credited for "unseen" opponents it was trained on).  Task 2 (the data
# generator) instantiates them — this module only owns the id convention, so
# the contract is imported rather than re-derived.  Personality ids are always
# positive; anchor ids are negative and can never collide.
ANCHOR_IDS = {
    "heuristic": -1,            # HeuristicPlayer()
    "randomized_heuristic": -2,  # RandomizedHeuristic(rng, epsilon=0.1)
    "random": -3,               # RandomPlayer(rng)
}

Q_POSTURES = ("hunt", "dump_early", "avoid")


@dataclass(frozen=True)
class PersonalityParams:
    """One player's stable tendencies.  All values are drawn by
    `sample_personality`; the field order here mirrors the draw order.

    Axes (the plan's minimum, plus three additions flagged NEW):

    duck        duck-vs-take appetite.  >0 = avoid winning tricks (duck under
                the current winner); <0 = a "taker" who happily wins cheap
                tricks to keep the lead.
    hoard       high-card hoarding vs dumping.  >0 = hold big cards back and
                shed low ones; <0 = get the big cards out of hand early.
    q_posture   Q-spades posture: "hunt" (chase the queen onto someone),
                "dump_early" (shed it at the first opportunity),
                "avoid" (stay out of spades entirely while holding it).
    q_strength  NEW — how strongly the posture is expressed.  Without it every
                "hunt" personality hunts equally hard and the posture axis is
                three points instead of a continuum.
    lead_short  lead style: preference for leading a safe SHORT suit.
    lead_long   lead style: preference for probing a LONG suit.
    lead_heart  lead style: heart aggression once hearts are broken.
    heart_dump  heart-dumping aggression on discards (void in the led suit).
    suit_quirk  per-suit quirk weights (clubs, diamonds, spades, hearts): a
                small idiosyncratic like/dislike added to every card of a
                suit.  This is what makes two otherwise-similar personalities
                distinguishable to a profiler.
    danger      NEW — sensitivity to points already sitting in the trick.
                Separates "cautious once the pot is fat" from "indifferent",
                a distinction real players show and `duck` alone cannot.
    temperature NEW (explicit, not hard-coded) — softmax sharpness.  Low =
                near-deterministic conviction, high = loose.  Kept away from
                near-uniform: a player who plays uniformly is not internally
                consistent and would be unlearnable.
    epsilon     probability of a uniformly random legal deviation (documented
                range 0.03-0.25 per the plan).  CONVENTION, which Task 2
                records per row and Task 3 buckets by: the epsilon branch
                draws uniformly over ALL n legal cards, not over the n-1
                "other" cards the way `RandomizedHeuristic` does.  So the
                probability of actually departing from the scored draw is
                eps*(1 - 1/n), and the mixture density of any legal card c is
                eps/n + (1-eps)*softmax(c) -- which is also why no legal card
                ever gets probability exactly zero.
    """

    duck: float
    hoard: float
    q_posture: str
    q_strength: float
    lead_short: float
    lead_long: float
    lead_heart: float
    heart_dump: float
    suit_quirk: tuple
    danger: float
    temperature: float
    epsilon: float


# ---------------------------------------------------------------------------
# Sampling distributions.  Ranges are deliberately wide and SIGN-INVERTING on
# duck / hoard / danger: a population where every draw only differs in
# magnitude collapses toward one policy, which is exactly what gate (c)
# forbids.  Numbers are score weights in "logit units" — a weight of 1.0 moves
# a feature by one softmax logit at temperature 1.0.
#
#   duck        Normal(0.8, 1.1)      mostly duckers, a real minority of takers
#   hoard       Uniform(-1.5, 1.5)    symmetric: hoarders and dumpers alike
#   q_posture   {hunt .25, dump_early .40, avoid .35}
#   q_strength  Uniform(0.5, 3.0)
#   lead_short  Uniform(-0.5, 2.0)
#   lead_long   Uniform(-0.5, 2.0)
#   lead_heart  Uniform(-1.0, 2.0)
#   heart_dump  Uniform(-1.0, 2.5)
#   suit_quirk  Normal(0, 0.35), 4 values
#   danger      Normal(0.7, 1.0)
#   temperature Uniform(0.35, 1.30)
#   epsilon     Uniform(0.03, 0.25)
# ---------------------------------------------------------------------------

def sample_personality(personality_seed: int) -> PersonalityParams:
    """Draw one personality.  DRAW ORDER IS FROZEN — append only."""
    rng = np.random.default_rng(int(personality_seed))
    duck = float(rng.normal(0.8, 1.1))
    hoard = float(rng.uniform(-1.5, 1.5))
    # NOTE: categorical draws go through random() + searchsorted, never
    # rng.choice.  numpy guarantees stream compatibility for the low-level
    # distributions (random/integers/normal) but NOT for choice, whose
    # randomness consumption is an implementation detail -- and this draw
    # order is a frozen contract that later tasks re-derive from ids.
    q_posture = Q_POSTURES[
        int(np.searchsorted(np.array([0.25, 0.65, 1.0]), rng.random()))]
    q_strength = float(rng.uniform(0.5, 3.0))
    lead_short = float(rng.uniform(-0.5, 2.0))
    lead_long = float(rng.uniform(-0.5, 2.0))
    lead_heart = float(rng.uniform(-1.0, 2.0))
    heart_dump = float(rng.uniform(-1.0, 2.5))
    suit_quirk = tuple(float(x) for x in rng.normal(0.0, 0.35, size=4))
    danger = float(rng.normal(0.7, 1.0))
    temperature = float(rng.uniform(0.35, 1.30))
    epsilon = float(rng.uniform(0.03, 0.25))
    return PersonalityParams(
        duck=duck, hoard=hoard, q_posture=q_posture, q_strength=q_strength,
        lead_short=lead_short, lead_long=lead_long, lead_heart=lead_heart,
        heart_dump=heart_dump, suit_quirk=suit_quirk, danger=danger,
        temperature=temperature, epsilon=epsilon,
    )


PARAM_FIELDS = tuple(f.name for f in fields(PersonalityParams))


def make_population(n_train: int, n_heldout: int, master_seed: int):
    """Return (train_ids, heldout_ids): personality seeds split AT CREATION.

    The held-out wall (plan, Global Constraints): held-out ids must never
    appear in any training shard, mixture pool, or tuning step.  Splitting
    here — before a single game exists — is what makes that enforceable.

    Ids are distinct positive integers drawn from the master seed; anchors
    (`ANCHOR_IDS`) are appended to the TRAIN side by consumers and are never
    held out.
    """
    assert n_train > 0 and n_heldout > 0
    rng = np.random.default_rng(int(master_seed))
    ids = rng.choice(10_000_000, size=n_train + n_heldout, replace=False) + 1
    ids = [int(x) for x in ids]
    train, held = ids[:n_train], ids[n_train:]
    assert set(train).isdisjoint(held)
    assert all(i > 0 for i in ids)
    return train, held


class PersonalityPlayer:
    """Softmax-over-scored-legal-moves player.  Sees only PlayerView."""

    def __init__(self, rng, params: PersonalityParams):
        self.rng = rng
        self.params = params

    # -- public API ----------------------------------------------------
    def choose(self, view: PlayerView) -> int:
        legal = cards.cards_in(view.legal_moves)
        if len(legal) == 1:
            return legal[0]
        # rng order is part of the determinism contract: epsilon roll first.
        if self.rng.random() < self.params.epsilon:
            return legal[self.rng.integers(len(legal))]
        scores = self._scores(view, legal)
        w = np.exp((scores - scores.max()) / self.params.temperature)
        cdf = np.cumsum(w)
        # inverse-CDF draw rather than rng.choice(p=...): stream-stable across
        # numpy versions (choice is not) and much faster on the hot path.
        return legal[int(np.searchsorted(cdf, self.rng.random() * cdf[-1]))]

    def greedy_choice(self, view: PlayerView) -> int:
        """The personality's conviction with noise switched off: argmax score,
        ties broken by lowest card.  Used by the divergence gate so that rng
        luck cannot masquerade as (or hide) a difference in tendencies."""
        legal = cards.cards_in(view.legal_moves)
        if len(legal) == 1:
            return legal[0]
        scores = self._scores(view, legal)
        return legal[int(np.argmax(scores))]

    # -- scoring -------------------------------------------------------
    def _scores(self, view, legal) -> np.ndarray:
        if not view.current_trick:
            raw = self._lead_scores(view, legal)
        elif cards.suit(legal[0]) == cards.suit(view.current_trick[0][1]):
            raw = self._follow_scores(view, legal)
        else:
            raw = self._discard_scores(view, legal)
        q = self.params.suit_quirk
        return np.array(
            [s + q[cards.suit(c)] for s, c in zip(raw, legal)], dtype=float)

    # -- context helpers ------------------------------------------------
    @staticmethod
    def _seen(view) -> int:
        """Every card this player can account for: own hand + all plays,
        INCLUDING the trick in progress."""
        seen = view.hand
        for _, c in view.history:
            seen |= cards.bit(c)
        for _, c in view.current_trick:
            seen |= cards.bit(c)
        return seen

    @staticmethod
    def _hand_by_suit(view) -> dict:
        out = {}
        for c in cards.cards_in(view.hand):
            out.setdefault(cards.suit(c), []).append(c)
        return out

    def _q_scores(self, card, context) -> float:
        """Q-spades posture, expressed with strength `q_strength`.

        hunt        wants spades led/played so the queen lands on someone
                    else; happy to touch spades, and dumps the queen only
                    when it will not cost it the trick.
        dump_early  wants the queen out of hand at the earliest chance.
        avoid       stays out of spades while it holds the queen.
        """
        p = self.params
        if cards.suit(card) != cards.SPADES:
            return 0.0
        s = 0.0
        is_q = card == cards.QUEEN_SPADES
        if p.q_posture == "dump_early":
            s += p.q_strength * (2.0 if is_q else 0.3)
        elif p.q_posture == "hunt":
            # lead low spades to flush the queen out; hold the queen itself
            s += p.q_strength * (-1.0 if is_q else 0.6)
            if context == "lead" and not is_q and cards.rank(card) <= 8:
                s += 0.5 * p.q_strength
        else:  # avoid
            s -= p.q_strength * (1.2 if is_q else 0.5)
        return s

    # -- leading --------------------------------------------------------
    def _lead_scores(self, view, legal):
        p = self.params
        by_suit = self._hand_by_suit(view)
        lens = {s: len(v) for s, v in by_suit.items()}
        n_hand = max(1, sum(lens.values()))
        out = []
        for c in legal:
            s = cards.suit(c)
            ln = lens.get(s, 1)
            short = 1.0 - (ln - 1) / max(1.0, n_hand - 1)  # 1 = singleton
            score = p.lead_short * short + p.lead_long * (1.0 - short)
            if s == cards.HEARTS:
                score += p.lead_heart * (1.0 if view.hearts_broken else 0.0)
            # leading a high card risks winning the trick you led
            score -= p.duck * 0.6 * (cards.rank(c) / 12.0)
            # hoarders lead their small cards, dumpers unload the big ones
            score -= p.hoard * (cards.rank(c) / 12.0)
            score += self._q_scores(c, "lead")
            out.append(score)
        return out

    # -- following in suit ----------------------------------------------
    def _follow_scores(self, view, legal):
        p = self.params
        led = cards.suit(view.current_trick[0][1])
        win_rank = max(cards.rank(c) for _, c in view.current_trick
                       if cards.suit(c) == led)
        pot = 0.0
        for _, c in view.current_trick:
            if cards.suit(c) == cards.HEARTS:
                pot += 1.0
            elif c == cards.QUEEN_SPADES:
                pot += 13.0
        last = len(view.current_trick) == 3
        out = []
        for c in legal:
            beats = cards.rank(c) > win_rank
            score = 0.0
            # ducking appetite, scaled by how dangerous this trick already is
            danger = 1.0 + p.danger * min(pot, 13.0) / 6.0
            if beats:
                score -= p.duck * danger
                # a taker who is last to play knows exactly what winning costs
                if last and pot == 0.0:
                    score += 0.4 * max(0.0, -p.duck)
                # WHICH winner: a dumper overpays with the ace, a hoarder wins
                # as cheaply as it can.  Without this term every card that
                # beats the trick scores identically and the personality
                # coin-flips between its K and its A -- inconsistent play, and
                # irreducible entropy for the Task-3 profiler to choke on.
                score -= p.hoard * (cards.rank(c) / 12.0)
            else:
                score += p.duck * 0.5
                # among losers, hoarders keep the high ones back
                score -= p.hoard * (cards.rank(c) / 12.0)
            score += self._q_scores(c, "follow")
            out.append(score)
        return out

    # -- discarding (void in the led suit) --------------------------------
    def _discard_scores(self, view, legal):
        p = self.params
        by_suit = self._hand_by_suit(view)
        lens = {s: len(v) for s, v in by_suit.items()}
        out = []
        for c in legal:
            s = cards.suit(c)
            score = p.hoard * (-cards.rank(c) / 12.0) * 1.2
            if s == cards.HEARTS:
                score += p.heart_dump * (0.4 + 0.6 * cards.rank(c) / 12.0)
            else:
                # shedding from a long suit does less to create a void
                score -= 0.25 * (lens.get(s, 1) - 1) / 12.0
            score += self._q_scores(c, "discard")
            out.append(score)
        return out
