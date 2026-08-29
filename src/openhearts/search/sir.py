"""Phase 7 Task A1: SAMPLING-IMPORTANCE-RESAMPLING for the profiled posterior.

WHY THIS EXISTS, IN PLAIN WORDS
-------------------------------
Every reading result since Phase 5 works like this: imagine 50 possible
arrangements of the hidden cards, then score each one by "how likely is it
that the opponents would have played the cards they actually played, if the
hidden cards were arranged like THIS?".  Arrangements that explain the play
well get a big number, ones that explain it badly get a tiny one, and the bot
averages its imagined futures with those numbers as weights.

The measured problem (Phase 5 limitation 9, promoted to root-cause suspect by
AUDIT_PHASE6.md Finding 1): those numbers are the product of many small
factors, so a handful of arrangements end up carrying essentially all of the
weight.  The published figure is an effective sample size (ESS) of roughly
1-6 out of 100.  The bot says "I averaged over 50 worlds"; it actually
averaged three worlds photocopied.  Whatever card looks good in those three
wins, and the decision is noise.

This module is the standard remedy from the particle-filtering literature,
and NOTHING ELSE.  It draws MANY MORE candidate arrangements than the search
needs (M >> N), weights them with the EXACT SAME likelihood the archived
profiled-ORACLE row uses, and then resamples N of them in proportion to
weight -- handing the search N worlds that are all equally weighted again and,
crucially, mostly DIFFERENT from each other.

THE TRAP, WRITTEN LARGE BECAUSE FALLING IN IT WOULD FAKE A RESULT
-----------------------------------------------------------------
Resampling the SAME 50 worlds the archived row already drew does NOT fix
anything.  If three of those 50 carry all the weight, then drawing 50 with
replacement from them just makes ~47 photocopies of those same three.  The
world set gets no more diverse; it gets LESS diverse.  An experiment built
that way would come back "no change" and be read as confirming that opponent
information is worthless -- when all it actually showed is that photocopying
a degenerate sample keeps it degenerate.

OVERSAMPLING FIRST IS THE ENTIRE POINT.  The pool must contain worlds the
plain draw never had, so that the resampled set contains worlds the plain
draw never had.  `tests/test_sir.py::test_trap_oversample_first_is_the_point`
pins this with a synthetic degenerate-weight scenario and asserts BOTH arms:
the SIR set is diverse AND the resample-the-original-50 set is not.

WHAT IS AND IS NOT CHANGED VS THE ARCHIVED ROW
----------------------------------------------
The ONLY manipulated variable is world-set construction:

  archived profiled-ORACLE   draw 50 -> weight -> (honest.py) resample 50
                             WITH REPLACEMENT from those 50, multinomial
  ORACLE-SIR (this module)   draw M (50, doubling to 5000) -> weight with the
                             IDENTICAL likelihood -> resample N=50 by
                             SYSTEMATIC resampling -> hand over as UNWEIGHTED

Same constraint sampler, same proposal (`BeliefTable.from_view(view, level)`),
same conditioned profiler, same true-parameter vectors, same epsilon
conventions, same log-space accumulation, same `PosteriorCollapse` contract.
The likelihood is literally `profiled.profiler_world_logweight`, called here;
it is not reimplemented.  Identical likelihoods are the design's control.

Two second-order differences, named rather than buried:
  * the resampler is SYSTEMATIC, where `honest.py` uses MULTINOMIAL.  This is
    forced by the frozen gate "SIR bitwise-reduces to the plain sampler when
    weights are uniform" -- only systematic resampling has that property (see
    `systematic_resample_indices`).  `resampler="multinomial"` is available
    for a sensitivity check.
  * systematic resampling consumes ONE uniform draw from the caller's rng
    where the plain path consumes none.  The gate is therefore on the
    returned WORLD LIST, not on the rng end state.  Special-casing "skip the
    resample when the weights happen to be uniform" would recover rng
    identity by making the code lie about what it does, and is not done.

HOW IT PLUGS IN WITHOUT TOUCHING THE SHIPPED BOT
------------------------------------------------
`sir_posterior` returns a real `WeightedPosterior` whose `weights` are all
exactly 1.0.  `search/honest.py::_posterior_worlds` already tests
`any(w != weights[0] ...)` and, finding uniform weights, uses the worlds
AS-IS.  So the resampled set reaches the search as an unweighted world set
through the existing socket: `honest.py`, `profiled.py` and `weighted.py` are
byte-unchanged, and the shipped bot's default behaviour is untouched because
nothing constructs this unless an experiment asks for it.

One counter reads differently as a result, said here so it is never misread:
`HonestSearchPlayer.posterior_worlds` counts what the posterior HANDED OVER,
which for SIR is N (the resampled set), not M (the pool).  M lives only in
this module's recorder.

WHY THIS FILE IS IN search/ AND NOT belief/
-------------------------------------------
It wraps `search/profiled.py`'s likelihood and feeds `search/honest.py`'s
world socket.  `belief/` is imported BY `search/`, never the other way round;
putting it there would invert the layering for no gain.
"""
import numpy as np

from ..belief.table import BeliefTable
from ..belief.weighted import PosteriorCollapse, WeightedPosterior
from ..engine import cards, kernel
from ..sampler.sampler import sample_arrangement
from .profiled import NEG_INF, ProfilerLikelihood, profiler_world_logweight

#: Pool ceiling.  JUSTIFICATION, WRITTEN BEFORE ANY MEASUREMENT: the brief's
#: default, kept unchanged -- 5000 is 100x the archived row's 50-world draw,
#: which is the order the published 1-6%-efficiency figure says is needed to
#: lift ESS to N, and it is the largest pool whose per-decision audit cost the
#: lead can still fit in the 2h single-run budget at 8 workers.
M_CAP = 5000

#: Draw ceiling, mirroring `profiler_posterior`'s `max_draws` semantics
#: (candidates DRAWN, including ones killed as illegal).  Must be >= M_CAP or
#: the pool could never reach the cap; asserted at call time.
MAX_DRAWS = 50_000


def kish_ess(weights) -> float:
    """Kish's effective sample size: (sum w)^2 / sum(w^2).

    Reads as "how many equally-weighted worlds is this weighted set really
    worth".  N equal weights give exactly N; one world carrying everything
    gives exactly 1.  This is the same formula `profiler_posterior` already
    reports as `n_effective`, spelled out here so the SIR path and the
    archived path are provably measuring the same quantity.
    """
    w = np.asarray(weights, dtype=np.float64)
    assert w.ndim == 1 and w.size > 0, "ESS needs a non-empty 1-D weight vector"
    assert np.isfinite(w).all() and (w >= 0.0).all(), (
        f"ESS on non-finite or negative weights: {w[:8]}...")
    s1 = float(w.sum())
    s2 = float((w * w).sum())
    assert s1 > 0.0 and s2 > 0.0, "ESS of an all-zero weight vector"
    return (s1 * s1) / s2


def systematic_resample_indices(weights, n: int, rng):
    """Systematic (a.k.a. stratified-with-one-draw) resampling indices.

    Lay the weights end to end on [0, 1], then take n equally spaced ticks
    starting from ONE uniform draw in [0, 1/n).  A world whose weight spans
    k ticks is selected k times.  Compared with multinomial resampling it has
    strictly lower variance for the same weights, and one property this task
    needs:

    THE UNIFORM IDENTITY.  With M == n and all weights equal, tick i lands in
    [i/n, (i+1)/n) and therefore selects index i -- the identity permutation,
    exactly.  That is what makes the frozen gate "SIR reduces to the plain
    sampler when weights are uniform" achievable on the world list rather
    than only in distribution.  (Corner: if the uniform draw rounds to
    within 2^-53 of 1.0 the last tick can spill one index; the returned
    indices are clipped to the pool, so the worst case is one duplicated
    world with probability ~1e-16.  Recorded, not defended against further.)

    Deterministic given `rng`, which is the player's own Generator.
    """
    w = np.asarray(weights, dtype=np.float64)
    total = float(w.sum())
    assert total > 0.0, "systematic resampling from an all-zero weight vector"
    c = np.cumsum(w) / total
    c[-1] = 1.0                       # guard the float tail
    u = (float(rng.random()) + np.arange(n, dtype=np.float64)) / n
    idx = np.searchsorted(c, u, side="right")
    return np.minimum(idx, w.size - 1).astype(np.int64)


def multinomial_resample_indices(weights, n: int, rng):
    """Independent draws with replacement -- what `honest.py` already does.

    Kept reachable so the lead can measure whether ANY of the SIR effect is
    the resampler rather than the pool.  Uses `rng.choice(..., p=...)`, the
    identical call `HonestSearchPlayer._posterior_worlds` makes.
    """
    p = np.asarray(weights, dtype=np.float64)
    total = float(p.sum())
    assert total > 0.0, "multinomial resampling from an all-zero weight vector"
    return np.asarray(rng.choice(p.size, size=n, p=p / total), dtype=np.int64)


RESAMPLERS = {"systematic": systematic_resample_indices,
              "multinomial": multinomial_resample_indices}


class SIRRecorder:
    """Per-decision SIR diagnostics, accumulated per TRICK.

    Lives in the worker process and is handed back as a JSON payload, so the
    experiment's partial files stay small: one row per trick, not one per
    decision.  Every alarm counts its own firings (house rule): `cap_hits`
    counts decisions that hit the pool ceiling without meeting the ESS
    target, and `n_decisions` counts the opportunities, so a cap-hit rate of
    0 can never be confused with "the counter was never called".
    """

    KEYS = ("n", "m", "ess_plain", "ess_pool", "distinct_pool",
            "distinct_resampled", "cap_hits", "n_illegal", "n_underflow",
            "draws", "stages")

    def __init__(self):
        self.per_trick = {}
        self.n_decisions = 0
        self.cap_hits = 0

    def _row(self, trick):
        return self.per_trick.setdefault(int(trick),
                                         {k: 0.0 for k in self.KEYS})

    def observe(self, trick, m, ess_plain, ess_pool, distinct_pool,
                distinct_resampled, cap_hit, n_illegal, n_underflow, draws,
                stages):
        r = self._row(trick)
        r["n"] += 1.0
        r["m"] += float(m)
        r["ess_plain"] += float(ess_plain)
        r["ess_pool"] += float(ess_pool)
        r["distinct_pool"] += float(distinct_pool)
        r["distinct_resampled"] += float(distinct_resampled)
        r["cap_hits"] += 1.0 if cap_hit else 0.0
        r["n_illegal"] += float(n_illegal)
        r["n_underflow"] += float(n_underflow)
        r["draws"] += float(draws)
        r["stages"] += float(stages)
        self.n_decisions += 1
        self.cap_hits += 1 if cap_hit else 0

    def payload(self):
        """JSON-serializable sums (never means -- means are taken after the
        blocks are merged, so block sizes cannot bias the average)."""
        return {"per_trick": {str(t): dict(r)
                              for t, r in sorted(self.per_trick.items())},
                "n_decisions": int(self.n_decisions),
                "cap_hits": int(self.cap_hits)}


def merge_payloads(payloads):
    """Sum a list of `SIRRecorder.payload()` dicts into one."""
    out = {"per_trick": {}, "n_decisions": 0, "cap_hits": 0}
    for p in payloads:
        out["n_decisions"] += int(p.get("n_decisions", 0))
        out["cap_hits"] += int(p.get("cap_hits", 0))
        for t, row in p.get("per_trick", {}).items():
            dst = out["per_trick"].setdefault(
                str(t), {k: 0.0 for k in SIRRecorder.KEYS})
            for k, v in row.items():
                dst[k] = dst.get(k, 0.0) + float(v)
    return out


def _draw_candidates(table, rng, want, batched):
    """`want` more raw candidate worlds from the CONSTRAINT sampler.

    Draw loop, proposal and chunking mirror `profiler_posterior` exactly (the
    batched compiled sampler when the JIT is on, one-at-a-time otherwise), so
    the SIR pool's first `n_worlds` entries are drawn from the same
    distribution -- and, on the same rng state, are the same worlds -- as the
    archived row's plain draw.  Returns (candidates, draws_consumed).
    """
    if batched:
        batch, _n_failed = kernel.sample_arrangements(table, rng, want)
        return batch, want
    out = []
    for _ in range(want):
        drawn = sample_arrangement(table, rng)
        if drawn is not None:
            out.append(drawn[0])
    return out, want


def _probs_from_worlds(worlds, weights, n_op=3):
    """(3, 52) marginals from a weighted world list, normalised."""
    probs = np.zeros((n_op, 52))
    total = float(np.asarray(weights, dtype=np.float64).sum())
    assert total > 0.0, "cannot build marginals from zero total weight"
    for w, world in zip(weights, worlds):
        wj = float(w)
        if wj <= 0.0:
            continue
        for i in range(n_op):
            for c in cards.cards_in(world[i]):
                probs[i, c] += wj
    probs /= total
    assert np.isfinite(probs).all() and (probs >= 0.0).all()
    return probs


def sir_posterior(view, level, lik: ProfilerLikelihood, n_worlds: int, rng,
                  max_draws: int = MAX_DRAWS, ess_target=None, m_start=None,
                  m_cap: int = M_CAP, resampler: str = "systematic",
                  keep_worlds: bool = True, recorder: SIRRecorder = None,
                  logweight_fn=None) -> WeightedPosterior:
    """Oversample, weight with the ORACLE likelihood, resample N unweighted.

    `n_worlds` (N) is what the SEARCH consumes and what comes back in
    `.worlds`, all at weight exactly 1.0.

    ESS ESCALATION.  The pool starts at `m_start` (default N -- so a decision
    whose weights were never degenerate costs exactly what the archived row
    cost, and only a degenerate one pays for more worlds), and DOUBLES until
    the pool's Kish ESS reaches `ess_target` (default N) or the pool reaches
    `m_cap`.  Growth is INCREMENTAL: each stage draws only the shortfall and
    keeps every world already audited, so the total audit cost is the FINAL M
    and not the sum of the stages.

    CAP HITS ARE NOT ALWAYS DEGENERACY, and the distinction is pre-registered
    here rather than argued after the run.  Late in a hand the number of
    DISTINCT worlds consistent with the constraints can itself be below N; ESS
    is bounded above by that count, so no amount of oversampling can reach the
    target and the cap fires structurally.  `distinct_pool` is recorded
    alongside every cap hit precisely so the two cases are separable.

    TRUTH-SAFETY (unchanged from the archived path, and live here).  A world
    leaves the pool only when the observed play is ILLEGAL in it, which can
    never be true of the world that actually happened; the masked softmax is
    strictly positive on every legal card, asserted inside
    `profiler_world_logweight`.  Weights that UNDERFLOW to 0.0 relative to
    the best world in the pool are counted (`n_underflow`) rather than
    ignored: that is degeneracy being measured, and it is exactly as present
    in the archived row.

    `logweight_fn` is a TEST SEAM ONLY (same spirit as
    `profiler_world_logweight`'s `include_forced`): it replaces the
    likelihood with `f(view, world_hands, lik) -> float` so a test can build a
    synthetic degenerate-weight scenario without a trained net.  Every
    experiment leaves it None, which means `profiler_world_logweight` -- the
    archived row's own function, byte-identical.
    """
    n_worlds = int(n_worlds)
    m_start = n_worlds if m_start is None else int(m_start)
    ess_target = float(n_worlds if ess_target is None else ess_target)
    m_cap = int(m_cap)
    assert n_worlds > 0 and m_start > 0, "N and m_start must be positive"
    assert m_cap >= m_start, f"m_cap {m_cap} below m_start {m_start}"
    assert max_draws >= m_cap, (
        f"max_draws {max_draws} < m_cap {m_cap}: the pool could never reach "
        f"the ceiling and the cap-hit counter would be meaningless")
    assert resampler in RESAMPLERS, f"unknown resampler {resampler!r}"
    weigh = profiler_world_logweight if logweight_fn is None else logweight_fn

    table = BeliefTable.from_view(view, level)
    if not cards.cards_in(table.unseen_mask):
        # Degenerate but well-defined, and byte-for-byte what
        # `profiler_posterior` returns here.
        return WeightedPosterior(np.zeros((3, 52)), list(table.opponent_seats),
                                 table.unseen_mask, list(table.hand_sizes),
                                 0.0, 0, 0, 0.0)

    batched = kernel.jit_enabled()
    pool, logs = [], []
    draws = n_illegal = stages = 0
    target_m = m_start
    ess_plain = float("nan")
    cap_hit = False

    while True:
        stages += 1
        while len(pool) < target_m and draws < max_draws:
            want = min(max_draws - draws, target_m - len(pool))
            candidates, used = _draw_candidates(table, rng, want, batched)
            draws += used
            for world_hands in candidates:
                lw = weigh(view, world_hands, lik)
                if lw == NEG_INF:
                    n_illegal += 1
                    continue
                pool.append([int(world_hands[i]) for i in range(3)])
                logs.append(float(lw))

        if not pool:
            raise PosteriorCollapse(
                f"no candidate world survived profiler filtering after "
                f"{draws} draws (level={level}); with a strictly-positive "
                f"masked softmax this can only mean every drawn world "
                f"replayed ILLEGALLY")

        lg = np.asarray(logs, dtype=np.float64)
        w = np.exp(lg - lg.max())
        if np.isnan(ess_plain):
            # What the ARCHIVED row's plain N-world draw suffers, measured on
            # the first N pool members -- which ARE a plain draw, because the
            # pool starts at N and grows only by appending.
            head = lg[:n_worlds]
            ess_plain = kish_ess(np.exp(head - head.max()))
        ess_pool = kish_ess(w)

        if ess_pool >= ess_target or len(pool) >= m_cap or draws >= max_draws:
            cap_hit = ess_pool < ess_target
            break
        target_m = min(m_cap, target_m * 2)

    idx = RESAMPLERS[resampler](w, n_worlds, rng)
    resampled = [pool[int(i)] for i in idx]
    ones = [1.0] * n_worlds

    if recorder is not None:
        recorder.observe(
            trick=view.trick_number, m=len(pool), ess_plain=ess_plain,
            ess_pool=ess_pool,
            distinct_pool=len({tuple(x) for x in pool}),
            distinct_resampled=len({tuple(x) for x in resampled}),
            cap_hit=cap_hit, n_illegal=n_illegal,
            n_underflow=int((w == 0.0).sum()), draws=draws, stages=stages)

    post = WeightedPosterior(
        _probs_from_worlds(resampled, ones), list(table.opponent_seats),
        table.unseen_mask, list(table.hand_sizes), float(ess_pool), draws,
        len(pool), float(w.sum()),
        resampled if keep_worlds else None, ones if keep_worlds else None)
    # Diagnostics the search never reads.  `sir_pool_probs` is the
    # lower-variance estimate of the SAME posterior from the whole weighted
    # pool: it is the apples-to-apples partner for the archived row's
    # `probs`, which is also a pool-weighted estimate.  `.probs` above is the
    # resampled set's marginals -- what the bot's world set actually
    # represents.
    post.sir_pool_probs = _probs_from_worlds(pool, w)
    post.sir_ess_plain = float(ess_plain)
    post.sir_ess_pool = float(ess_pool)
    post.sir_m = len(pool)
    post.sir_cap_hit = bool(cap_hit)
    return post


def sir_posterior_factory(lik: ProfilerLikelihood, level=None, n_worlds=50,
                          max_draws=MAX_DRAWS, ess_target=None, m_start=None,
                          m_cap=M_CAP, resampler="systematic",
                          keep_worlds=True, recorder=None, logweight_fn=None):
    """-> `f(view, rng) -> WeightedPosterior`, the shape `honest.py` wants.

    Deliberately mirrors `profiled.profiler_posterior_factory`'s signature and
    exposes `.likelihood`, so an experiment swaps one factory for the other
    and changes nothing else about the row.
    """
    assert isinstance(lik, ProfilerLikelihood), (
        "SIR must wrap the SAME ProfilerLikelihood object the archived row "
        "uses -- identical likelihoods are this experiment's control")

    def make(view, rng, level=level, n_worlds=n_worlds, max_draws=max_draws,
             keep_worlds=keep_worlds):
        assert level is not None, "no belief level: pass level= to the factory"
        return sir_posterior(view, level, lik, n_worlds, rng, max_draws,
                             ess_target=ess_target, m_start=m_start,
                             m_cap=m_cap, resampler=resampler,
                             keep_worlds=keep_worlds, recorder=recorder,
                             logweight_fn=logweight_fn)

    make.likelihood = lik
    make.recorder = recorder
    return make


class RecordingProfilerFactory:
    """The ARCHIVED profiled-ORACLE factory, plus an ESS tape.

    The archived row already computes exactly the number this experiment
    needs -- `WeightedPosterior.n_effective`, the Kish ESS over its 50
    weighted worlds -- and then throws it away.  This wrapper calls
    `profiler_posterior` unchanged and writes that number down.

    IT DRAWS NO RANDOMNESS AND CHANGES NO ARITHMETIC, so a row built with it
    is bitwise the archived row.  That is what lets the experiment reproduce
    a banked ORACLE block exactly while measuring the per-trick ESS the
    archived row actually suffered, on the archived row's OWN decision path.
    """

    def __init__(self, inner, recorder: SIRRecorder):
        self.inner = inner
        self.recorder = recorder
        self.likelihood = getattr(inner, "likelihood", None)

    def __call__(self, view, rng, **kw):
        post = self.inner(view, rng, **kw)
        n = len(post.worlds)
        self.recorder.observe(
            trick=view.trick_number, m=n, ess_plain=post.n_effective,
            ess_pool=post.n_effective,
            distinct_pool=len({tuple(x) for x in post.worlds}),
            # The archived row's world set as the SEARCH sees it: honest.py
            # multinomially resamples n_outer of these AFTER we return, so
            # the distinct count of what search consumes is not observable
            # from here.  Recorded as the pool's own distinct count and
            # labelled that way in the report.
            distinct_resampled=len({tuple(x) for x in post.worlds}),
            cap_hit=False, n_illegal=0,
            n_underflow=int(sum(1 for x in post.weights if float(x) == 0.0)),
            draws=post.draws_used, stages=1)
        return post
