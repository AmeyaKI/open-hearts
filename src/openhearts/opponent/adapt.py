"""Phase 5 Task 5: per-seat Bayesian mixture adaptation ("getting to know you").

Task 3 measured the prize: a GENERIC profiler scores -1.2616 nats on
personalities it never saw, while the CONDITIONED profiler fed the TRUE
personality parameters scores -1.1480 -- an **adaptation headroom of +0.1136
nats / +8.17 top-1 points** that lives entirely in knowing WHO you face.
Deployment cannot know that.  What it CAN do is watch the seat play and infer
which of a small pool of reference personalities it resembles.

THE MODEL
---------
Fix a pool of K reference personalities (ids drawn deterministically from the
TRAIN side -- see `pool_ids`).  Each member k is the CONDITIONED profiler
evaluated with member k's parameter vector, giving a full choice distribution
`P_k(card | observer view)`.  Treat the seat as drawn from that pool with a
prior (uniform by default) and do exact Bayes on observed choice events:

    log w_k  <-  log w_k + log P_k(chosen | view)      per observed event
    w        <-  normalize(exp(log w - max log w))
    w        <-  (w + EPS_W) / (1 + K*EPS_W)           the floor, below

The adapted likelihood handed to the socket is the posterior predictive

    P_adapted(card | view) = (1-b) * sum_k w_k P_k(card | view)
                             + b * P_generic(card | view)

LOG SPACE IS NOT OPTIONAL.  A hand contributes ~10 events per seat at ~-1.3
nats each; six hands of a match put the best member's raw product near e^-80
and the worst members at exact 0.0 in float64, so a naive multiply-and-
renormalize divides 0 by 0.  Weights are therefore carried as LOGS and
exponentiated relative to their max, the same contract `search/profiled.py`
records for world weights.

THE WEIGHT FLOOR, and why it exists.  After renormalizing, every member is
lifted by `EPS_W = 1e-6` and the vector renormalized again, so the smallest
achievable weight is EPS_W/(1 + K*EPS_W) -- not exactly EPS_W; the honest
invariant is stated that way in `SeatMixture.weights`'s assertion.  Reason: a
held-out personality is NEVER in the pool (that is the whole point of the
held-out wall), so the truth is always OUTSIDE the simplex's corners and the
best description of it is usually a BLEND of several members.  A member driven
to exact zero by one surprising card can never come back, and the mixture
would collapse onto whichever member happened to survive the first few tricks.
The floor keeps every direction of the simplex reachable forever.  It is also
what makes truth-safety trivial: a convex combination with all-positive
weights of strictly-positive masked softmaxes is strictly positive on every
legal card.

WHEN THE UPDATE HAPPENS, and why it does not peek.  Updates are fed COMPLETED
hands (`events_from_history`).  Reconstructing the events needs each seat's
hand at each ply, which is a deterministic function of the completed public
history -- every card a seat held is a card it played.  Nothing hidden at the
time of play is used, and `obsfeat.observer_features` structurally zeroes the
other seats' hand blocks anyway (Task 2's proven contract), so a leak is
impossible by construction rather than by care.  Accumulation is ACROSS hands
within a match; `reset()` starts a new opponent identity.

THE BLEND SWEEP, PRE-REGISTERED BEFORE ANY HELD-OUT NUMBER WAS COMPUTED
-----------------------------------------------------------------------
Grid: b in {0.0, 0.1, 0.2, 0.3, 0.5}.  Selection rule: highest mean
out-of-sample log-likelihood on hand 4's events with weights fitted on hands
1-3, measured on 40 TRAIN personalities that are NOT in the pool (the
deployment regime -- tuning on pool members would measure identification, not
generalization), fresh game seeds from `SWEEP_SEED_BASE = 760000`.  Ties
broken toward the SMALLER b (less generic mass = more adaptation, the thing
being tested).  Result of that sweep is recorded in `BLEND_B` below.

DISCLOSED FLAW IN THE SWEEP, not corrected because correcting it after seeing
the result is exactly what pre-registration forbids: `SWEEP_SEED_BASE` sits
inside Task 2's training deal range (700000..899999), so the CONDITIONED net
has seen those exact deals.  The sweep's LEVELS are therefore optimistic.  Its
DECISION is not sensitive to that: the curve is flat to the third decimal
across every b <= 0.3 (-1.2479 / -1.2466 / -1.2463 / -1.2470), so any of them
would have been a defensible pick.  All HELD-OUT numbers use bases 965000+ /
980000+ / 990000+, clear of the training range.

COST (measured, `experiments/measure_adaptation.py`): 396 us per adapted ply
at K=32 against 12 us for a GENERIC ply -- 33x, exactly K forward passes, no
hidden win.  That is the number Task 6 must budget the `profiled-RIA` row
against (a 100-world audit of ~29 opponent plies costs ~1.1 s per posterior);
`member_probs` is deliberately the single choke point so a top-M prune is a
one-line change IF a profile ever demands one (no speculative optimization).
"""
import numpy as np

from ..engine.features import NF
from ..players.personality import make_population
from ..search.profiled import ProfilerLikelihood
from .infer import profiler_probs_batch
from .params import PARAM_DIM, param_vector

# Task 1/2's frozen population contract, re-derived here rather than imported
# from `experiments/gen_population_data.py` (src must not import experiments).
# These three constants ARE that file's constants; the assertions in
# `pool_ids` fail loudly if they ever drift.
MASTER_SEED = 314159
N_TRAIN_PERSONALITIES = 200
N_HELDOUT_PERSONALITIES = 50

POOL_SALT = 5150        # Task 5's own salt off MASTER_SEED, so the pool draw
                        # is independent of the table draw (salt 777) and the
                        # deal streams.
DEFAULT_K = 32          # the plan's suggestion, kept. MEASURED cost: 396 us
                        # per adapted ply vs 12 us GENERIC -- 33x, i.e. K
                        # forward passes and no hidden win. Task 6 must budget
                        # the RIA row against that number, not against a hope.
EPS_W = 1e-6            # weight floor; see module docstring.

BLEND_GRID = (0.0, 0.1, 0.2, 0.3, 0.5)
SWEEP_SEED_BASE = 760_000
# RESULT of the pre-registered sweep (results/adaptation_sweep.txt, 40
# TRAIN-not-in-pool personalities, fit hands 1-3 / score hand 4):
#   b=0.0 -1.2479   b=0.1 -1.2466   b=0.2 -1.2463   b=0.3 -1.2470   b=0.5 -1.2511
# The curve is flat to the 3rd decimal across b<=0.3 -- the honest reading is
# that a little GENERIC mass is harmless insurance, not a lever.
BLEND_B = 0.2


def train_heldout_ids():
    """`(train_ids, heldout_ids)` for the frozen Phase-5 population split."""
    return make_population(N_TRAIN_PERSONALITIES, N_HELDOUT_PERSONALITIES,
                           MASTER_SEED)


def pool_ids(k=DEFAULT_K, salt=POOL_SALT):
    """K TRAIN personality ids, deterministic, NEVER held-out (asserted).

    Anchors are deliberately excluded: the opponents this phase measures
    against are personalities, so spending pool mass on the plain heuristic /
    random anchors would buy coverage of a region no held-out seat occupies.
    """
    train, held = train_heldout_ids()
    rng = np.random.default_rng([int(MASTER_SEED), int(salt)])
    idx = rng.choice(len(train), size=int(k), replace=False)
    ids = [int(train[int(i)]) for i in sorted(idx)]
    assert len(set(ids)) == int(k)
    assert set(ids).isdisjoint(set(held)), (
        "mixture pool contains HELD-OUT personalities -- the held-out wall "
        "(PHASE5_PLAN Global Constraints) forbids it")
    return ids


def pool_param_matrix(ids):
    """float64[K, PARAM_DIM] of the pool's personality parameter vectors."""
    return np.ascontiguousarray(
        np.stack([param_vector(int(p)) for p in ids]), dtype=np.float64)


# ------------------------------------------------------------- observation
def events_from_history(history):
    """A completed hand's `[(seat, card), ...]` -> per-ply decision events.

    Returns `(seats, feats, masks, chosen)` for the MULTI-LEGAL plies only --
    exactly the row filter `gen_population_data.play_and_record` used to build
    the profiler's training diet, so the update is on-distribution (the net
    never saw forced plies).  `feats` is float64[N, NF].

    Imports are local: this is the only place `adapt` touches the engine's
    game objects, and keeping them here keeps the module's import cost at the
    inference path's level.
    """
    from ..engine import cards
    from ..engine.game import legal_moves
    from ..engine.state import GameState
    from .obsfeat import observer_features

    assert len(history) == 52, f"expected a completed hand, got {len(history)}"
    hands = np.zeros(4, dtype=np.int64)
    for seat, card in history:
        hands[seat] |= cards.bit(card)
    state = GameState(hands=hands)
    state.to_play = history[0][0]

    seats, feats, masks, chosen = [], [], [], []
    for seat, card in history:
        assert seat == state.to_play, "history replay desync"
        legal = legal_moves(state.hands[seat], tuple(state.current_trick),
                            state.hearts_broken, state.trick_number)
        assert legal & cards.bit(card), "history contains an illegal play"
        if bin(int(legal)).count("1") > 1:
            seats.append(int(seat))
            feats.append(observer_features(state, seat))
            masks.append(int(legal))
            chosen.append(int(card))
        state.play(card)
    if not feats:
        return ([], np.zeros((0, NF)), np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64))
    return (seats, np.ascontiguousarray(np.stack(feats), dtype=np.float64),
            np.asarray(masks, dtype=np.int64),
            np.asarray(chosen, dtype=np.int64))


# --------------------------------------------------------------- the pool
class PoolProfiler:
    """The CONDITIONED net evaluated at K fixed parameter vectors.

    One object shared by every seat's `SeatMixture` and by
    `AdaptedLikelihood`: the weights and the K x PARAM_DIM parameter block are
    read-only, so sharing is safe and avoids K copies of the network.
    """

    def __init__(self, weights, ids=None, k=DEFAULT_K):
        self.weights = tuple(weights)
        self.ids = list(ids) if ids is not None else pool_ids(k)
        self.params = pool_param_matrix(self.ids)
        self.k = len(self.ids)
        n_in = int(self.weights[0].shape[0])
        assert n_in == NF + PARAM_DIM, (
            f"adaptation needs the CONDITIONED profiler (n_in = NF + "
            f"{PARAM_DIM} = {NF + PARAM_DIM}), got {n_in}")

    def member_probs(self, feats, masks):
        """float64[N, K, 52]: every pool member's distribution for N plies.

        ONE `profiler_probs_batch` call over N*K rows -- the single choke
        point for adaptation's cost (module docstring).
        """
        feats = np.asarray(feats, dtype=np.float64)
        n = feats.shape[0]
        if n == 0:
            return np.zeros((0, self.k, 52), dtype=np.float64)
        big = np.empty((n * self.k, NF + PARAM_DIM), dtype=np.float64)
        big[:, :NF] = np.repeat(feats, self.k, axis=0)
        big[:, NF:] = np.tile(self.params, (n, 1))
        out = np.zeros((n * self.k, 52), dtype=np.float64)
        profiler_probs_batch(*self.weights, big,
                             np.repeat(np.asarray(masks, dtype=np.int64),
                                       self.k), out)
        return out.reshape(n, self.k, 52)


class SeatMixture:
    """Per-seat Bayesian weights over the K reference personalities.

    `pool_params` is a `PoolProfiler` (or the ids/weights to build one).
    `prior` defaults to uniform; any strictly-positive length-K vector is
    accepted and normalized.
    """

    def __init__(self, pool_params, prior=None):
        assert isinstance(pool_params, PoolProfiler), (
            "pass a PoolProfiler (built once, shared across seats)")
        self.pool = pool_params
        self.k = pool_params.k
        if prior is None:
            prior = np.full(self.k, 1.0 / self.k)
        prior = np.asarray(prior, dtype=np.float64)
        assert prior.shape == (self.k,) and (prior > 0).all()
        self._log_prior = np.log(prior / prior.sum())
        self.n_events = 0
        self.n_hands = 0
        self.reset()

    def reset(self):
        """New opponent identity: forget everything observed."""
        self._logw = self._log_prior.copy()
        self.n_events = 0
        self.n_hands = 0

    @property
    def weights(self):
        """float64[K] posterior weights, floored and normalized."""
        w = np.exp(self._logw - self._logw.max())
        w /= w.sum()
        w = (w + EPS_W) / (1.0 + self.k * EPS_W)
        floor = EPS_W / (1.0 + self.k * EPS_W)
        assert w.min() >= floor * (1 - 1e-12), "weight floor violated"
        assert abs(w.sum() - 1.0) < 1e-9
        return w

    def observe(self, feats, masks, chosen):
        """Bayes update from this seat's decision events (one hand's worth).

        Accumulates in log space; the floor is applied when `weights` is read,
        so repeated reads cannot compound it.
        """
        feats = np.asarray(feats, dtype=np.float64)
        if feats.shape[0] == 0:
            return self
        pm = self.pool.member_probs(feats, masks)
        p = pm[np.arange(feats.shape[0]), :, np.asarray(chosen, dtype=int)]
        assert (p > 0.0).all(), (
            "a pool member assigned exactly 0 to an observed LEGAL card; the "
            "masked softmax must never do that")
        self._logw = self._logw + np.log(p).sum(axis=0)
        self._logw -= self._logw.max()   # keep the accumulator bounded
        self.n_events += int(feats.shape[0])
        return self

    def observe_hand(self, history, seat):
        """Convenience: update from ONE seat's plies of a completed hand."""
        seats, feats, masks, chosen = events_from_history(history)
        sel = [i for i, s in enumerate(seats) if s == seat]
        if sel:
            self.observe(feats[sel], masks[sel],
                         np.asarray(chosen)[sel])
        self.n_hands += 1
        return self

    def top(self, n=1):
        """[(pool id, weight), ...] for the n heaviest members."""
        w = self.weights
        order = np.argsort(-w)[:n]
        return [(self.pool.ids[int(i)], float(w[int(i)])) for i in order]


# -------------------------------------------------- the socket-side object
class AdaptedLikelihood(ProfilerLikelihood):
    """Drop-in `ProfilerLikelihood` for `search/profiled.py`'s factory.

    Constructed additively (Task 6 builds `profiled-RIA` with it and
    `profiled.py`'s semantics are untouched): it subclasses
    `ProfilerLikelihood` and overrides exactly one method, `batch_probs`, the
    seam Task 5 added there -- the rows->probabilities step, which is the only
    place a MIXTURE differs from a single network evaluation.

        lik = AdaptedLikelihood(pool, mixtures, generic=generic_lik)
        factory = profiler_posterior_factory(lik, level=..., n_worlds=...)

    `mixtures` maps ABSOLUTE seat -> `SeatMixture`; a seat with no mixture
    (e.g. our own) falls back to the pool's uniform weights.  `generic` is an
    optional GENERIC `ProfilerLikelihood` blended in with weight `blend`.

    RESET ON RESEAT -- READ THIS BEFORE WIRING TASK 6.  The mapping is keyed
    by SEAT, but the belief being accumulated is about a PERSONALITY.  Task
    6's ablation rotates held-out trios across deals, so the person behind
    seat 1 changes between deals; whoever drives the match must call
    `SeatMixture.reset()` on every seat whose occupant changed, and must NOT
    reset between hands of the same match (accumulating across hands IS the
    "getting to know you" effect being measured).  Forgetting the reset does
    not raise -- it silently reads the previous opponent, which would corrupt
    the RIA row rather than crash it.
    """

    def __init__(self, pool: PoolProfiler, mixtures=None, generic=None,
                 blend=BLEND_B):
        assert isinstance(pool, PoolProfiler)
        # Satisfy ProfilerLikelihood's CONDITIONED-width contract with a
        # placeholder param block; `row()` is never reached because
        # `batch_probs` is fully overridden.
        zeros = np.zeros(PARAM_DIM, dtype=np.float64)
        super().__init__(pool.weights, {s: zeros for s in range(4)})
        self.pool = pool
        self.mixtures = dict(mixtures or {})
        self.generic = generic
        self.blend = float(blend)
        assert 0.0 <= self.blend <= 1.0
        if self.blend > 0.0:
            assert generic is not None, "blend > 0 needs a GENERIC likelihood"

    def mixture_for(self, seat):
        m = self.mixtures.get(int(seat))
        if m is None:
            return np.full(self.pool.k, 1.0 / self.pool.k)
        return m.weights

    def batch_probs(self, feat_rows, seats, masks):
        feats = np.ascontiguousarray(np.stack(feat_rows), dtype=np.float64)
        pm = self.pool.member_probs(feats, masks)          # [N, K, 52]
        w = np.stack([self.mixture_for(s) for s in seats])  # [N, K]
        out = np.einsum("nk,nkc->nc", w, pm)
        if self.blend > 0.0:
            g = self.generic.batch_probs(feat_rows, seats, masks)
            out = (1.0 - self.blend) * out + self.blend * g
        return out
