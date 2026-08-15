"""Phase 5 Task 4: the LEARNED-PROFILER likelihood in the Phase-3 socket.

`belief/weighted.py` reweights candidate worlds by how well each explains the
opponents' actual plays, using `play_likelihood` -- (1-eps)*[heuristic would
have played this card] + eps/n_legal. Phase 4's league showed that likelihood
is heuristic-specific: against anyone who is not the script it stakes ~0.9 on
a card played ~36% of the time (Task 3 measured heuristic+eps at -2.40 nats
held-out, WORSE than uniform's -1.30). This module swaps that factor for the
Task-3 profiler's `P_model(observed card | replayed observer-legal view)`,
which scored -1.26 nats on personalities it never saw.

WHAT IS AND IS NOT TOUCHED
--------------------------
`weighted.py` and `honest.py` are BYTE-UNCHANGED -- no dispatch hook, not even
an additive one. This module owns its own audit (`profiler_world_logweight`)
and its own draw loop (`profiler_posterior`), and hands the result back as a
real `WeightedPosterior` built through its public `__init__`, so every
downstream consumer (`eval.guessing.metrics_for`, `search/honest.py`'s world
source) works unchanged. The only shared code is `weighted`'s reconstruction
helper and its `PosteriorCollapse` type, both imported read-only.

DEVIATION FROM THE PLAN TEXT, stated plainly (Task 4 asks for it named):
the plan says "the JIT audit kernel (Phase 3.5) needs a profiler-likelihood
variant: new kernel entry beside the old one". What is implemented is a
PYTHON replay that BATCHES the network call: one `profiler_probs_batch` call
per candidate world covering all of that world's opponent decision plies. The
net itself is the existing numba kernel (`opponent/infer.py`), so the flops
are compiled; only the replay loop is Python. Reason: an njit replay would
have to re-port `featurize` + the audit loop into one compiled function, and
the measured cost of the Python version already projects comfortably inside
the compute budget (see the per-posterior timings printed by
`experiments/run_guessing5.py --smoke`). If Task 6's playouts need more, that
is the moment to compile the loop, with a profile in hand.

THREE CONTRACTS WORTH READING BEFORE EDITING
--------------------------------------------
1. SINGLE-LEGAL PLIES ARE SKIPPED, AND THAT IS EXACT. When only one card is
   legal, the masked softmax returns exactly 1.0 for it, so the factor is 1.0
   and multiplying it in is a no-op. Skipping also keeps us on distribution:
   `gen_population_data` only ever emitted `n_legal > 1` rows, so the profiler
   was never trained on forced plies. `tests/test_profiled_search.py` pins the
   skip against a reference that includes those plies as explicit 1.0 factors.
2. LOG SPACE. A trick-13 audit multiplies ~39 factors, some ~0.01; the product
   underflows float64 in the tail. Weights are accumulated as logs and
   exponentiated after subtracting the maximum over surviving worlds. Both
   reported quantities are scale-invariant (`probs /= total_w`, and
   `n_effective = (sum w)^2 / sum w^2`), so this changes nothing observable --
   EXCEPT that `total_weight` is now relative to that per-call maximum and is
   NOT comparable across calls or curves. Callers are told so in the output
   headers.
3. TRUTH-SAFETY. The masked softmax is strictly positive on every legal card
   (it exponentiates finite logits), so a world dies only if the observed card
   is ILLEGAL in it -- which can never happen to the TRUE world. The
   heuristic-match likelihood's eps=0 collapse mode does not exist here.
   Asserted directly in the tests.

THE ORACLE VARIANT IS SIMULATION-ONLY. `seat_params` attaches each seat's TRUE
personality parameter vector to that seat's features, feeding the CONDITIONED
profiler (n_in = NF + PARAM_DIM). No deployed bot can know those; it exists to
measure the ceiling perfect reading would buy (PHASE5_PLAN redesign
2026-08-13). Nothing in `src/` may construct it from a GameState -- only an
experiment that already knows the table may pass it in.
"""
import numpy as np

from ..belief.table import BeliefTable
from ..belief.weighted import (PosteriorCollapse, WeightedPosterior,
                               _reconstruct_original_hands)
from ..engine import cards, kernel
from ..engine.game import legal_moves
from ..engine.state import GameState
from ..opponent.infer import load_profiler, profiler_probs_batch
from ..opponent.obsfeat import observer_features
from ..sampler.sampler import sample_arrangement

NEG_INF = -np.inf


class ProfilerLikelihood:
    """Weights + optional per-seat params: everything one audit needs.

    `weights` is the 6-tuple `(W1, b1, W2, b2, W3, b3)` from
    `opponent.infer.load_profiler` (W1 already transposed for the sparse first
    layer -- that private contract is honoured by using `profiler_probs_batch`
    and nothing else).

    `seat_params`: None for the GENERIC model, or a mapping {absolute seat ->
    float64[PARAM_DIM]} for the CONDITIONED/ORACLE variant, in which case each
    ply's feature row is `concat(observer_features, seat_params[seat])`.
    """

    def __init__(self, weights, seat_params=None, meta=None):
        self.weights = tuple(weights)
        self.n_in = int(self.weights[0].shape[0])
        self.seat_params = seat_params
        self.meta = meta or {}
        if seat_params is None:
            assert self.n_in == self._nf(), (
                f"GENERIC profiler expects n_in == NF ({self._nf()}), got "
                f"{self.n_in}; pass seat_params for a CONDITIONED model")
        else:
            pd = {len(v) for v in seat_params.values()}
            assert len(pd) == 1, "seat_params rows differ in length"
            assert self.n_in == self._nf() + pd.pop(), (
                "CONDITIONED profiler input width does not match "
                "NF + len(param vector)")

    @staticmethod
    def _nf():
        from ..engine.features import NF
        return NF

    def row(self, feats, seat):
        if self.seat_params is None:
            return feats
        return np.concatenate([feats, self.seat_params[seat]])

    def batch_probs(self, feat_rows, seats, masks):
        """[N,52] masked choice distributions for N replayed opponent plies.

        ADDITIVE SEAM (Phase 5 Task 5). `profiler_world_logweight` used to
        inline exactly this body; it was lifted to a method so
        `opponent.adapt.AdaptedLikelihood` can override the rows->probs step
        with a per-seat MIXTURE (which needs the seat of each row, a thing
        `row()` alone cannot express since it is called per row and the seat
        is gone by the time the batch is assembled). The math here is
        byte-identical to the previous inline version -- pinned by
        `tests/test_profiled_search.py::_reference_logweight`, an independent
        dense implementation.
        """
        rows = [self.row(f, s) for f, s in zip(feat_rows, seats)]
        feats = np.ascontiguousarray(np.stack(rows), dtype=np.float64)
        out = np.zeros((feats.shape[0], 52), dtype=np.float64)
        profiler_probs_batch(*self.weights, feats,
                             np.asarray(masks, dtype=np.int64), out)
        return out


def load_profiler_likelihood(path, seat_params=None) -> ProfilerLikelihood:
    """Load an `.npz` profiler into a `ProfilerLikelihood`."""
    weights, meta = load_profiler(path)
    return ProfilerLikelihood(weights, seat_params, meta)


def profiler_world_logweight(view, world_hands, lik: ProfilerLikelihood,
                             include_forced: bool = False) -> float:
    """log P(observed opponent plays | this world) under the profiler.

    `world_hands`: the three opponents' CURRENT hands in
    `BeliefTable.opponent_seats` order -- the same convention
    `belief.weighted.world_weight` takes, so the two are drop-in comparable.

    Returns `-inf` when the world cannot have produced the observed play (the
    observed card is illegal in it). Otherwise a finite log-likelihood: the
    sum over OPPONENT decision plies with 2+ legal cards of
    `log P_model(observed card | that seat's observer-legal view)`. The
    observer's own plays contribute 0.0 (we know why we played them).

    `include_forced=True` also queries the net at single-legal plies. It is a
    TEST HOOK only: the factor there is exactly 1.0 either way (a masked
    softmax over one card), so it must not change the result -- which is
    precisely what the test asserts.
    """
    hands, all_plays = _reconstruct_original_hands(view, world_hands)
    if not all_plays:
        return 0.0  # no evidence yet: every world is equally consistent
    state = GameState(hands=hands)
    state.to_play = all_plays[0][0]
    observer = view.seat

    rows, seats, masks, obs_cards = [], [], [], []
    for seat, card in all_plays:
        assert seat == state.to_play, (
            f"replay desync: observed seat {seat} != {state.to_play}")
        legal = legal_moves(state.hands[seat], tuple(state.current_trick),
                            state.hearts_broken, state.trick_number)
        if not (legal & cards.bit(card)):
            return NEG_INF  # this world could not have produced the play
        if seat != observer:
            n_legal = bin(int(legal)).count("1")
            if n_legal > 1 or include_forced:
                rows.append(observer_features(state, seat))
                seats.append(seat)
                masks.append(int(legal))
                obs_cards.append(card)
        state.play(card)

    if not rows:
        return 0.0
    out = lik.batch_probs(rows, seats, masks)
    total = 0.0
    for i, card in enumerate(obs_cards):
        p = float(out[i, card])
        # Truth-safety, asserted rather than assumed: a masked softmax over a
        # mask that PROVABLY contains `card` (checked above) is strictly
        # positive there. If this ever fires, the likelihood -- not the world
        # -- is broken.
        assert p > 0.0, (
            f"profiler assigned probability {p} to a LEGAL observed card "
            f"{card}; masked softmax must never be exactly zero on the mask")
        total += np.log(p)
    assert np.isfinite(total), f"bad log weight {total}"
    return float(total)


def profiler_posterior(view, level, lik: ProfilerLikelihood, n_worlds: int,
                       rng, max_draws: int,
                       keep_worlds: bool = False) -> WeightedPosterior:
    """`WeightedPosterior` whose likelihood is the profiler's, not a policy's.

    Draw loop, proposal, chunking and `draws_used` semantics mirror
    `WeightedPosterior.from_view` exactly (batched `kernel.sample_arrangements`
    when the JIT is on, one-at-a-time `sample_arrangement` otherwise), so a
    PROFILER curve and a CHOICE curve differ in the LIKELIHOOD and nothing
    else. The accumulation differs in one documented way: log weights are
    collected first and exponentiated relative to their maximum (module
    docstring, contract 2).
    """
    table = BeliefTable.from_view(view, level)
    unseen = cards.cards_in(table.unseen_mask)
    probs = np.zeros((3, 52))
    if not unseen:
        return WeightedPosterior(probs, list(table.opponent_seats),
                                 table.unseen_mask, list(table.hand_sizes),
                                 0.0, 0, 0, 0.0)

    draws = 0
    kept_worlds, kept_logs = [], []
    batched = kernel.jit_enabled()

    while len(kept_worlds) < n_worlds and draws < max_draws:
        if batched:
            chunk = min(max_draws - draws, n_worlds - len(kept_worlds))
            batch, _n_failed = kernel.sample_arrangements(table, rng, chunk)
            draws += chunk
            candidates = batch
        else:
            draws += 1
            drawn = sample_arrangement(table, rng)
            if drawn is None:
                continue
            candidates = [drawn[0]]
        for world_hands in candidates:
            lw = profiler_world_logweight(view, world_hands, lik)
            if lw == NEG_INF:
                continue
            kept_worlds.append([int(world_hands[i]) for i in range(3)])
            kept_logs.append(lw)

    if not kept_worlds:
        raise PosteriorCollapse(
            f"no candidate world survived profiler filtering after {draws} "
            f"draws (level={level}); with a strictly-positive masked softmax "
            f"this can only mean every drawn world replayed ILLEGALLY")

    logs = np.asarray(kept_logs, dtype=np.float64)
    w = np.exp(logs - logs.max())
    total_w = float(w.sum())
    total_w2 = float((w * w).sum())
    assert total_w > 0.0, "log-space rescaling produced a zero total weight"
    for j, world in enumerate(kept_worlds):
        wj = float(w[j])
        if wj <= 0.0:
            continue  # underflowed relative to the best world; contributes 0
        for i in range(3):
            for c in cards.cards_in(world[i]):
                probs[i, c] += wj
    probs /= total_w
    assert np.isfinite(probs).all() and (probs >= 0.0).all()
    n_effective = (total_w * total_w) / total_w2
    return WeightedPosterior(
        probs, list(table.opponent_seats), table.unseen_mask,
        list(table.hand_sizes), float(n_effective), draws, len(kept_worlds),
        float(total_w),
        kept_worlds if keep_worlds else None,
        [float(x) for x in w] if keep_worlds else None)


def profiler_posterior_factory(weights, seat_params=None, meta=None,
                               level=None, n_worlds=100, max_draws=50000):
    """-> `f(view, rng, **overrides) -> WeightedPosterior`.

    The factory the plan names. `weights` is either a loaded 6-tuple, a
    `ProfilerLikelihood`, or a path to a profiler `.npz`. `level`, `n_worlds`
    and `max_draws` are defaults the caller may override per call, so a search
    can hold one factory and vary depth.
    """
    if isinstance(weights, ProfilerLikelihood):
        lik = weights
    elif isinstance(weights, str):
        lik = load_profiler_likelihood(weights, seat_params)
    else:
        lik = ProfilerLikelihood(weights, seat_params, meta)

    def make(view, rng, level=level, n_worlds=n_worlds, max_draws=max_draws,
             keep_worlds=False):
        assert level is not None, "no belief level: pass level= to the factory"
        return profiler_posterior(view, level, lik, n_worlds, rng, max_draws,
                                  keep_worlds=keep_worlds)

    make.likelihood = lik
    return make
