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
from ..engine import kernel_audit_profiled, kernel_profiled
from ..engine.game import legal_moves
from ..engine.state import GameState
from ..opponent.infer import load_profiler, profiler_probs_batch
from ..opponent.obsfeat import observer_features
from ..sampler.sampler import sample_arrangement
from .honest import HonestSearchPlayer

NEG_INF = -np.inf

#: Phase 5.5b. When True (and the JIT is on, and the likelihood is a plain
#: GENERIC/CONDITIONED `ProfilerLikelihood`), `profiler_world_logweight` runs
#: as ONE fused numba call -- replay + featurize + net + log accumulation --
#: instead of the Python replay below. Set to False to force the Python path
#: for testing WITHOUT touching `kernel.jit_enabled()`, which also selects the
#: world SAMPLER in `profiler_posterior` and would therefore change the drawn
#: worlds rather than just the arithmetic that scores them.
FUSED_AUDIT = True


def _fused_params(lik):
    """The `[4, pd]` per-seat parameter block for the fused kernel, or None.

    `None` means "not fusable, use the Python path". Two reasons to refuse:

    * a SUBCLASS of `ProfilerLikelihood` -- notably
      `opponent.adapt.AdaptedLikelihood`, which overrides `batch_probs` with a
      per-seat MIXTURE over a pool. That is a different computation, not a
      slower spelling of this one, and it stays Python this phase (5.5c's
      problem). `type(...) is` rather than `isinstance` is deliberate.
    A CONDITIONED likelihood is fusable, but `seat_params` need not cover all
    four seats: `row()` is only ever called for OPPONENTS, so a mapping over
    the three opponent seats is legitimate and the observer's slot is never
    read. The kernel wants a dense `[4, pd]` array anyway, and zero-filling a
    row we merely ASSUME is never read is the kind of silent divergence this
    task exists to rule out -- so absent seats are filled with NaN, which
    poisons the forward pass into the kernel's non-positive-probability status
    and a loud raise. Fail-loud beats a plausible wrong number.
    """
    if type(lik) is not ProfilerLikelihood:
        return None
    if lik.seat_params is None:
        return np.zeros((4, 0), dtype=np.float64)
    pd = len(next(iter(lik.seat_params.values())))
    params = np.full((4, pd), np.nan, dtype=np.float64)
    for seat, vec in lik.seat_params.items():
        params[int(seat)] = np.asarray(vec, dtype=np.float64)
    return params


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

    # Phase 5.5b: one fused numba call for the whole replay. Every structural
    # assert above stays on this side of the boundary; the loop below remains
    # the reference and the NO_JIT path, byte-unchanged, and the two are pinned
    # bitwise by `tests/test_fused_audit.py`.
    if FUSED_AUDIT and kernel.jit_enabled():
        params = _fused_params(lik)
        if params is not None:
            return kernel_audit_profiled.audit_world_logweight(
                hands, all_plays, view.seat, lik.weights, lik.n_in,
                params if params.shape[1] else None, include_forced)

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
                               level=None, n_worlds=100, max_draws=50000,
                               keep_worlds=False):
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
             keep_worlds=keep_worlds):
        assert level is not None, "no belief level: pass level= to the factory"
        return profiler_posterior(view, level, lik, n_worlds, rng, max_draws,
                                  keep_worlds=keep_worlds)

    make.likelihood = lik
    return make


# ==========================================================================
# Phase 5 Task 6: Organ 2 -- model-driven playouts
# ==========================================================================
class ProfiledSearchPlayer(HonestSearchPlayer):
    """Honest search with a profiler READER and (optionally) profiler ACTORS.

    Three configurations, all built from this one class -- the ablation rows
    the plan names:

    =========  ================================  ==========================
    row        posterior_factory                 playout_weights
    =========  ================================  ==========================
    ``R``      profiler (GENERIC likelihood)     ``None``  -> heuristic
    ``RI``     profiler (GENERIC likelihood)     GENERIC 6-tuple
    ``RIA``    profiler w/ `AdaptedLikelihood`   GENERIC 6-tuple
    =========  ================================  ==========================

    `playout_weights` is a GENERIC profiler weight 6-tuple (as returned by
    `opponent.infer.load_profiler`); `None` keeps `HonestSearchPlayer`'s
    heuristic playouts EXACTLY (delegated to `super()._playout`, so the R row
    is the Task-4 player plus nothing).

    WHICH PLAYOUTS ARE PROFILED (decide-and-state, not silent). The OUTER
    playout -- both segments, run-to-our-decision and finish -- uses the
    profiled kernel. The INNER re-determinization (`self._inner`, a Phase-1
    `SearchPlayer`) keeps its plain heuristic playouts. Rationale: the inner
    search runs once per (outer world x outer candidate), so profiling it too
    would multiply net calls by roughly `n_inner x n_candidates`; the measured
    cost of that variant is reported to the lead rather than adopted. `RI`
    therefore means "model-driven OUTER playouts". Anyone changing this must
    change the row's meaning in the plan too.

    RNG. `choose` draws ONE integer from `self.rng` per decision and seeds the
    kernel sampler with it (`kernel_profiled.seed_playouts`). Deterministic
    given the player's seed; NOT the same stream as `HonestSearchPlayer`'s,
    and not the same between JIT and NO_JIT -- the Phase-2.6 precedent,
    documented in `kernel_profiled`'s docstring.

    ADAPTATION HOOKS (`RIA`), contract for the harness -- READ BEFORE WIRING.
    `SeatMixture` is keyed by ABSOLUTE SEAT while the belief it accumulates is
    about a PERSONALITY (Task 5's recorded trap). The harness therefore MUST:

    * call `player.observe_hand(history)` after EVERY completed hand of a
      match, passing the full 52-ply `[(seat, card), ...]` history -- this
      updates every opponent seat's mixture from that hand's decision events
      (`events_from_history` is called ONCE and its rows dispatched per seat,
      not once per seat);
    * call `player.reset_mixtures()` whenever the occupants of the seats
      change (a new deal with rotated/redrawn personalities), and
      `player.reset_seat(seat)` for a single reseat;
    * NEVER reset between hands of the same match -- accumulating across hands
      IS the effect being measured.

    Forgetting the reset does not raise; it silently reads the previous
    opponent. `observe_hand` is a no-op for R/RI (no `AdaptedLikelihood`), so
    a harness may call it unconditionally.

    OPEN DESIGN QUESTION FOR THE ABLATION SCRIPT (raised by Task 6's wiring,
    for the lead to decide -- NOT decided here). The plan runs "trios rotated
    across deals". Combined with the reset-on-reseat rule above, that makes
    every deal a fresh identity: the mixture sits at its uniform prior for the
    whole hand (`observe_hand` can only fire once the hand is over) and is then
    reset. Under that deal structure `profiled-RIA` is `profiled-RI` plus the
    adapted audit's cost and ZERO adaptation -- structurally, not empirically.
    Task 5 measured N=9 hands to concentrate weight and a benefit "from hand 3
    on", so one discarded hand of evidence is nothing. Making the RIA row mean
    what the plan intends requires deals BLOCKED INTO MATCHES: one trio held
    fixed for a run of consecutive deals, `reset_mixtures()` only at block
    boundaries, with the block assignment derived from the deal index so it is
    identical across all eight rows and the paired per-deal bootstrap is
    unaffected.
    """

    def __init__(self, level, n_outer, n_inner, rng,
                 sampler_respects_voids=True, jit_sampler=True,
                 posterior_factory=None, playout_weights=None):
        super().__init__(level, n_outer, n_inner, rng, sampler_respects_voids,
                         jit_sampler, posterior_factory)
        if playout_weights is not None:
            playout_weights = tuple(playout_weights)
            assert len(playout_weights) == 6, (
                "playout_weights must be the 6-tuple from load_profiler")
            n_in = int(playout_weights[0].shape[0])
            from ..engine.features import NF as _NF
            assert n_in == _NF, (
                f"playout profiler must be GENERIC (n_in == NF == {_NF}), got "
                f"{n_in}: the in-kernel featurizer builds observer features "
                f"only, with no personality block to append")
        self.playout_weights = playout_weights
        self.playout_mode = kernel_profiled.MODE_PROFILER

    # ------------------------------------------------------------ playouts
    def choose(self, view):
        if self.playout_weights is not None:
            kernel_profiled.seed_playouts(int(self.rng.integers(2 ** 63)))
        return super().choose(view)

    def _playout(self, state, our_seat: int) -> None:
        if self.playout_weights is None:
            super()._playout(state, our_seat)
            return
        w = self.playout_weights
        if self.n_inner > 0:
            if kernel_profiled.run_playout_until_decision_profiled(
                    state, our_seat, our_seat, w, self.playout_mode):
                state.play(self._inner.choose(state.view_for(our_seat)))
        kernel_profiled.run_playout_profiled(state, our_seat, w,
                                             self.playout_mode)

    # -------------------------------------------------------- adaptation
    @property
    def _adapted(self):
        """The `AdaptedLikelihood` behind the posterior factory, or None."""
        lik = getattr(self.posterior_factory, "likelihood", None)
        return lik if hasattr(lik, "mixtures") else None

    def observe_hand(self, history) -> int:
        """Feed one COMPLETED hand to every opponent seat's mixture.

        Returns the number of seats updated (0 for R/RI). `events_from_history`
        runs ONCE for the whole hand and its rows are sliced per seat.
        """
        lik = self._adapted
        if lik is None or not lik.mixtures:
            return 0
        from ..opponent.adapt import events_from_history

        seats, feats, masks, chosen = events_from_history(history)
        seats = np.asarray(seats, dtype=np.int64)
        n = 0
        for seat, mix in lik.mixtures.items():
            sel = np.flatnonzero(seats == int(seat))
            if sel.size:
                mix.observe(feats[sel], masks[sel], chosen[sel])
            mix.n_hands += 1
            n += 1
        return n

    def reset_seat(self, seat) -> None:
        """Forget everything learned about `seat` (its occupant changed)."""
        lik = self._adapted
        if lik is not None and int(seat) in lik.mixtures:
            lik.mixtures[int(seat)].reset()

    def reset_mixtures(self) -> None:
        """Forget every seat -- call on ANY reseat of the table."""
        lik = self._adapted
        if lik is not None:
            for mix in lik.mixtures.values():
                mix.reset()
