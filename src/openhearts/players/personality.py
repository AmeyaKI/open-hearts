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

PHASE 6 (Task A1) — THE HABIT DIAL AND THE CONTEXTUAL AXES
----------------------------------------------------------
Phase 6A asks a single scientific question: how much is reading opponents
worth AS A FUNCTION of how predictable they are?  That needs a population
whose predictability moves while its playing STYLE stays fixed.  Two additions
serve it, both strictly additive:

1. `sample_personality_v2(pid, habit)` — the same draw stream as v1, with
   three NEW axes appended (see `NEW_AXES_V2`), followed by a POST-DRAW
   transform that rewrites ONLY `epsilon` and `temperature` into the band of
   the requested dial setting.  Nothing else is touched, so H0-Priya and
   H2-Priya are literally the same style at two discipline levels.
   The transform is QUANTILE-PRESERVING: a personality's position within the
   v1 range (u = (x - lo) / (hi - lo)) is carried into the new band, so a
   relatively loose personality stays relatively loose at every setting.  At
   H2 the band IS the v1 range, so the transform is the identity — which is
   what makes H2 the exact Phase-5 predictability regime (the continuity
   anchor A2 compares against).

2. `sample_personality` is UNCHANGED, bit for bit.  It leaves the three new
   axes at exactly 0.0, and every new scoring term is strictly multiplicative
   on its new weight (`score += w_new * feature`, no additive constants, no
   reordering of existing terms, no change to `choose`'s rng consumption
   order).  Therefore every Phase-5 personality plays exactly as it did, and
   `tests/test_personality.py` plus the profiler/playout/fused corpora are the
   standing check.

WHY A SEPARATE ENTRY POINT rather than a flag on `sample_personality`: ~5M
Phase-5 decision rows are keyed by personality id.  A kwarg would make one id
mean two different players depending on the call site — precisely the failure
the frozen contract exists to prevent.
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

    APPENDED IN PHASE 6 (Task A1) — contextual strategy axes.  These express
    behaviours the twelve score-modulating dials structurally cannot: they are
    conditioned on the SITUATION (is this trick safe? how early is it? who is
    winning it?), not on the card alone.  All three default to 0.0, which is
    exactly "the Phase-5 player", so v1 personalities are unaffected.  Each is
    drawn spanning never -> always with real mass at both ends (SIGN-INVERTING
    like duck/hoard), so the population SPANS the behaviour instead of being
    SHIFTED toward it — a shared-sign axis would move every personality the
    same way and compress the vs-heuristic divergence floor.

    safe_dump   NEW (P6) — safe-trick high-dumping, the "Priya move".  When
                following to a trick that carries no points and is very
                unlikely to be poisoned, deliberately WIN it with a big
                dangerous card, shedding it for free.
                CARD EXAMPLE: fourth to a club trick of 2c/4c/6c holding Qc
                and 8c.  Both cards win.  A high `safe_dump` plays the Qc —
                the queen of clubs is a liability later, and this trick costs
                nothing.  A negative `safe_dump` keeps the Qc and wins with
                the 8c (or ducks where ducking is possible).
                >0 = dump high on safe tricks; <0 = never waste a big card.
                Rank-scaled on purpose so it is NOT collinear with the
                existing card-independent "taker is last on an empty pot"
                bonus.

    void_engineer NEW (P6) — void engineering.  Early in the hand, play off a
                SHORT suit so it empties and later discards become free (which
                is what makes hearts/Qc dumping possible at all).
                CARD EXAMPLE: trick 2, holding 3d 4d and five clubs.  A high
                `void_engineer` leads the 3d: two rounds and diamonds are gone,
                after which every diamond lead is a free discard.  A negative
                one leads clubs and keeps the diamond guard.
                Explicitly TIME-DEPENDENT (fades out by trick ~6), which is
                what separates it from the static `lead_short` preference.
                >0 = engineer voids; <0 = keep every suit guarded.

    feed_leader NEW (P6) — leader-feeding.  Dump point cards onto the player
                who is already winning the trick AND already carrying a big
                score (in a no-moon house game, piling on the leader is the
                cheapest way to keep them the leader — and real players do it
                out of spite as much as strategy).
                CARD EXAMPLE: void in clubs, seat 1 has taken the trick with
                the Ac and sits on 18 points.  A high `feed_leader` discards a
                heart onto them; a negative one sheds a safe diamond and saves
                its hearts for a lighter target.
                Scaled by the winner's running score, so it is a genuinely
                contextual feature and not a restatement of `heart_dump`.
                >0 = pile onto the leader; <0 = spare them.
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
    # --- appended Phase 6 (Task A1); default 0.0 == the Phase-5 player ---
    safe_dump: float = 0.0
    void_engineer: float = 0.0
    feed_leader: float = 0.0


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

def _draw_v1(rng) -> dict:
    """The twelve Phase-5 draws, in the FROZEN order, off an open rng stream.

    Factored out (Phase 6) so `sample_personality_v2` can continue the SAME
    stream and append to it.  Not one call was reordered: this body is the
    original `sample_personality` verbatim, and `sample_personality` below is
    still the only v1 entry point.
    """
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
    return dict(
        duck=duck, hoard=hoard, q_posture=q_posture, q_strength=q_strength,
        lead_short=lead_short, lead_long=lead_long, lead_heart=lead_heart,
        heart_dump=heart_dump, suit_quirk=suit_quirk, danger=danger,
        temperature=temperature, epsilon=epsilon,
    )


def sample_personality(personality_seed: int) -> PersonalityParams:
    """Draw one Phase-5 personality.  DRAW ORDER IS FROZEN — append only.

    Unchanged by Phase 6: the three appended axes stay at 0.0, so this
    player's scores, choices and rng consumption are bit-identical to Phase 5.
    """
    return PersonalityParams(
        **_draw_v1(np.random.default_rng(int(personality_seed))))


PARAM_FIELDS = tuple(f.name for f in fields(PersonalityParams))

# ---------------------------------------------------------------------------
# PHASE 6 (Task A1): the contextual axes and the habit dial.
# ---------------------------------------------------------------------------
# Appended draws, in order, AFTER `epsilon`.  Distributions (chosen to SPAN,
# per Ruling 1 — the population's job is stress-test coverage of strategy
# space, not human fidelity):
#
#   safe_dump      Uniform(-1.2, 2.6)   ~31% never-dumpers, ~69% dumpers of
#                                       varying conviction; the Priya end is
#                                       strongly represented because that is
#                                       the behaviour Ruling 1 says our script
#                                       could never predict.
#   void_engineer  Normal(0.4, 1.3)     centred slightly positive (most real
#                                       players do shorten suits) but ~38%
#                                       negative, so the axis spans.
#   feed_leader    Uniform(-1.5, 1.5)   symmetric: piling on and sparing are
#                                       equally represented.
#
NEW_AXES_V2 = ("safe_dump", "void_engineer", "feed_leader")

# The v1 (Phase-5) ranges the dial remaps FROM.  Kept beside the dial so the
# identity property at H2 is checkable by eye.
_V1_EPSILON_RANGE = (0.03, 0.25)
_V1_TEMPERATURE_RANGE = (0.35, 1.30)

# PRE-REGISTERED dial settings (PHASE6_PLAN Task A1).  Only these two axes
# move; everything else is invariant across settings by construction.
#
#   H0  near-deterministic habits: eps 0.005-0.02 (one whim in ~70 decisions
#       at the loose end), temperature 0.10-0.30 (strong convictions — the
#       softmax is nearly an argmax).  "Priya always leads her lowest club."
#   H1  mild: an audible habit with visible exceptions.
#   H2  IDENTITY: exactly the Phase-5 ranges.  The continuity anchor — any
#       H2-vs-Phase-5 difference is attributable to the three new axes alone.
#   H3  noisier than Phase 5: the far anchor of the curve.  DELIBERATE
#       DEVIATION from the v1 docstring's caution that temperature is "kept
#       away from near-uniform" — H3's job is precisely to be worse than the
#       Phase-5 hurricane, and a curve needs a far end.  Measured: ~1.25 nats
#       mean per-decision entropy, ~83% of uniform-over-legal.  H3 is an
#       anchor for the curve, NOT a population anything should be trained on.
HABIT_SETTINGS = {
    "H0": {"epsilon": (0.005, 0.020), "temperature": (0.10, 0.30)},
    "H1": {"epsilon": (0.015, 0.080), "temperature": (0.20, 0.60)},
    "H2": {"epsilon": _V1_EPSILON_RANGE, "temperature": _V1_TEMPERATURE_RANGE},
    "H3": {"epsilon": (0.100, 0.450), "temperature": (0.80, 2.50)},
}
HABIT_ORDER = ("H0", "H1", "H2", "H3")


def _remap(x: float, src: tuple, dst: tuple) -> float:
    """Quantile-preserving remap of `x` from range `src` into range `dst`.

    Both ranges are the supports of Uniform draws, so `u` is the draw's own
    uniform quantile: carrying `u` across preserves each personality's RANK
    within the population.  Identity when src == dst.
    """
    if src == dst:
        # EXACT identity, short-circuited.  The round trip through u is only
        # identity to within a float ulp, and H2 must be BIT-identical to
        # Phase 5 for the continuity anchor to mean what it says.
        return float(x)
    lo, hi = src
    u = (float(x) - lo) / (hi - lo)
    u = min(1.0, max(0.0, u))
    return dst[0] + u * (dst[1] - dst[0])


def sample_personality_v2(personality_seed: int,
                          habit: str = "H2") -> PersonalityParams:
    """Phase-6 personality: v1's draws, three appended axes, one dial setting.

    The rng stream is v1's, continued — so every v1 field is bit-identical to
    `sample_personality(personality_seed)` (the frozen contract, honoured by
    APPENDING rather than inserting).  `habit` then rewrites epsilon and
    temperature ONLY, via the quantile-preserving `_remap`.
    """
    if habit not in HABIT_SETTINGS:
        raise ValueError(f"unknown habit setting {habit!r}")
    rng = np.random.default_rng(int(personality_seed))
    base = _draw_v1(rng)
    # --- APPENDED DRAWS (never insert above this line) ---
    safe_dump = float(rng.uniform(-1.2, 2.6))
    void_engineer = float(rng.normal(0.4, 1.3))
    feed_leader = float(rng.uniform(-1.5, 1.5))
    band = HABIT_SETTINGS[habit]
    base["epsilon"] = _remap(base["epsilon"], _V1_EPSILON_RANGE,
                             band["epsilon"])
    base["temperature"] = _remap(base["temperature"], _V1_TEMPERATURE_RANGE,
                                 band["temperature"])
    return PersonalityParams(safe_dump=safe_dump, void_engineer=void_engineer,
                             feed_leader=feed_leader, **base)


# --- population constructors for the 6A curve experiment -------------------
# FRESH seed ranges, documented (PHASE6_PLAN Task A1 deliverable 3).  Phase 5
# used master_seed 314159 with 200/50; Phase 6's v2 population uses its own
# master seed so that no Phase-5 held-out id is silently reused as a Phase-6
# TRAIN id (which would breach the held-out wall retroactively).  Overlap with
# the Phase-5 pools is asserted away by `make_population_v2`.
#
# ONE id set is shared by ALL FOUR dial settings, deliberately: the dial is a
# transform, not a resample, so H0-Priya and H2-Priya are the same person.
# That also makes A2's curve a PAIRED comparison across settings — the same
# 250 personalities at four discipline levels — which is house discipline.
MASTER_SEED_V2 = 606060
N_TRAIN_V2 = 200
N_HELDOUT_V2 = 50


def make_population_v2():
    """(train_ids, heldout_ids) for Phase 6, disjoint from Phase 5's pools."""
    train, held = make_population(N_TRAIN_V2, N_HELDOUT_V2, MASTER_SEED_V2)
    p5_train, p5_held = make_population(200, 50, 314159)
    p5 = set(p5_train) | set(p5_held)
    assert not (set(train) | set(held)) & p5, (
        "Phase-6 population collides with Phase-5 ids; pick a new master seed")
    return train, held


def population_params_v2(habit: str = "H2"):
    """(train_params, heldout_params) at one dial setting, in id order.

    Each side is an ordered {pid: PersonalityParams} dict.  Same ids at every
    setting (see MASTER_SEED_V2's note); only the dial differs, so a curve
    drawn across settings is paired by personality.
    """
    train, held = make_population_v2()
    return ({p: sample_personality_v2(p, habit) for p in train},
            {p: sample_personality_v2(p, habit) for p in held})


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


# Void-engineering weight by suit length: a singleton is one play from gone,
# a doubleton two; a tripleton is marginal and anything longer is not a void
# plan at all.  A lookup rather than a formula so the intent is legible.
_VOID_SHORT = {1: 1.0, 2: 0.8, 3: 0.3}


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

    # -- Phase-6 contextual helpers ---------------------------------------
    # All three read ONLY the PlayerView (own hand, history, current trick,
    # trick number, scores) — the information boundary is untouched.

    @staticmethod
    def _early(view) -> float:
        """1.0 on trick 0, fading linearly to 0.0 by trick 6.

        Void engineering is a FIRST-HALF plan: shortening a suit on trick 10
        buys nothing, and making the weight time-dependent is exactly what
        keeps this axis from collapsing into the static `lead_short` taste.
        """
        return max(0.0, 1.0 - view.trick_number / 6.0)

    def _trick_is_safe(self, view) -> bool:
        """Approximate 'this trick carries no points and cannot be poisoned'.

        Certain only when we are LAST to act (nobody can add anything).
        Otherwise we accept the approximation: no points in the pot, the led
        suit is not hearts, and the Q(spades) is accounted for (played, or in
        our own hand) so nobody can drop it on us.  Someone void in the led
        suit could still discard a heart — a personality heuristic is allowed
        to be approximate, and pretending it is exact would be worse.
        """
        for _, c in view.current_trick:
            if cards.suit(c) == cards.HEARTS or c == cards.QUEEN_SPADES:
                return False
        if len(view.current_trick) == 3:
            return True
        led = cards.suit(view.current_trick[0][1])
        q_accounted = bool(self._seen(view) & cards.bit(cards.QUEEN_SPADES))
        return led != cards.HEARTS and q_accounted

    @staticmethod
    def _leader_burden(view) -> float:
        """How much the CURRENT trick-winner already carries, on ~[0, 1].

        `view.scores` is the running per-seat total; 13 (a queen's worth) is
        the natural full-scale.  Returns 0 when nobody is winning yet.
        """
        if not view.current_trick:
            return 0.0
        led = cards.suit(view.current_trick[0][1])
        best_seat, best = None, -1
        for seat, c in view.current_trick:
            if cards.suit(c) == led and cards.rank(c) > best:
                best_seat, best = seat, cards.rank(c)
        if best_seat is None or best_seat == view.seat:
            return 0.0
        return min(1.0, view.scores[best_seat] / 13.0)

    @staticmethod
    def _feed_weight(card) -> float:
        """How much 'feeding the leader' this card is worth: 0 for a safe
        card, 0.7 for a heart, 1.6 for the queen.  Not the raw 1/13 point
        ratio — a 13x logit swing would make the queen term drown every other
        axis; the queen is *more* than a heart here, not thirteen times."""
        if card == cards.QUEEN_SPADES:
            return 1.6
        return 0.7 if cards.suit(card) == cards.HEARTS else 0.0

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
        early = self._early(view)
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
            # VOID ENGINEERING (P6): leading from a doubleton/singleton early
            # is the cheapest way to empty a suit.  Zero for v1 personalities.
            if p.void_engineer:
                score += p.void_engineer * early * _VOID_SHORT.get(ln, 0.0)
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
        # Phase-6 contextual terms (both zero-weighted for v1 personalities).
        safe = self._trick_is_safe(view) if p.safe_dump else False
        burden = self._leader_burden(view) if p.feed_leader else 0.0
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
            # SAFE-TRICK HIGH-DUMPING (P6, the Priya move): on a pointless,
            # unpoisonable trick, WINNING with a big card sheds it for free.
            # Rank-scaled so it is a different feature from the existing
            # card-independent "taker is last on an empty pot" bonus.
            if safe and beats:
                score += p.safe_dump * 1.2 * (cards.rank(c) / 12.0)
            # LEADER-FEEDING (P6): if we are not taking this trick, a point
            # card lands on whoever is winning it.  Scaled by their burden.
            if burden and not beats:
                score += p.feed_leader * burden * self._feed_weight(c)
            out.append(score)
        return out

    # -- discarding (void in the led suit) --------------------------------
    def _discard_scores(self, view, legal):
        p = self.params
        by_suit = self._hand_by_suit(view)
        lens = {s: len(v) for s, v in by_suit.items()}
        early = self._early(view)
        # Discarding never wins the trick (we are void in the led suit), so
        # whoever is currently winning it is certain to eat whatever we throw.
        burden = self._leader_burden(view) if p.feed_leader else 0.0
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
            # VOID ENGINEERING (P6): a discard from an already-short suit
            # finishes the void off.  Same early-only weighting as leading.
            if p.void_engineer:
                score += p.void_engineer * early * 0.7 * _VOID_SHORT.get(
                    lens.get(s, 1), 0.0)
            # LEADER-FEEDING (P6): this is the classic spot for it.
            if burden:
                score += p.feed_leader * burden * self._feed_weight(c)
            out.append(score)
        return out
