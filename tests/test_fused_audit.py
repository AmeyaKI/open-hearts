"""Phase 5.5b: the fused profiler audit, pinned BITWISE against the Python path.

WHY BITWISE AND NOT A TOLERANCE. This kernel was written to be enabled mid-way
through a PAUSED ablation run, for the remainder of its checkpoints. A
near-equal likelihood would make the run's two halves incomparable in a way no
downstream number could reveal. Exactness is the enabling condition, so the
assertion is `np.float64` byte equality (Phase 2.7 precedent), never `allclose`.

WHAT MAKES THAT ACHIEVABLE (the design, restated so a reader can audit the
claim rather than trust the test): every float operation is SHARED SOURCE with
the Python path -- `kernel_profiled._ply_features` for the features (already
pinned exact against `obsfeat.observer_features`), `infer._profiler_probs_njit`
for the net (literally the function `ProfilerLikelihood.batch_probs` reaches
through `profiler_probs_batch`), and a plain sequential `total += np.log(p)`
loop for the reduction -- `search/profiled.py`'s summation is a simple loop,
NOT `np.sum`, so there is no pairwise reduction order to emulate and that file's
arithmetic is BYTE-UNCHANGED by this task.

The one operation that is not shared source is the scalar `np.log`: numpy's and
numba's are different implementations. `test_log_reduction_variants_agree`
pins them over the real audit corpus, and `kernel_audit_profiled.LOG_IN_KERNEL`
exists so a platform where they ever disagree can fall back to numpy's log over
the kernel's (bitwise-identical) gathered probabilities without giving up the
fusion.

These tests are JIT-only: under `OPENHEARTS_NO_JIT=1` there is no fused path to
compare against (the dispatch is guarded by `kernel.jit_enabled()`), so the
Python path is the only path and the existing suite already covers it.
"""
import numpy as np
import pytest

from openhearts.belief.table import BeliefTable, Level
from openhearts.engine import cards, kernel
from openhearts.engine import kernel_audit_profiled as KA
from openhearts.engine.features import NF
from openhearts.engine.game import deal
from openhearts.engine.state import GameState
from openhearts.opponent.infer import load_profiler
from openhearts.opponent.npz_io import N_CARDS, save_npz
from openhearts.opponent.params import PARAM_DIM, param_vector
from openhearts.players.heuristic import HeuristicPlayer
from openhearts.players.personality import (PersonalityPlayer, make_population,
                                            sample_personality)
import openhearts.search.profiled as P
from openhearts.search.profiled import (ProfilerLikelihood,
                                        profiler_posterior,
                                        profiler_world_logweight)

pytestmark = pytest.mark.skipif(
    not kernel.jit_enabled(),
    reason="the fused audit only exists under the JIT; NO_JIT runs the "
           "Python path the fused one is pinned against")

H1, H2 = 24, 12


def _bits(x):
    """Byte image of a float64 -- the equality this module actually asserts."""
    return np.float64(x).tobytes()


def _npz(tmp_path, name, n_param_in, seed):
    r = np.random.default_rng(seed)
    n_in = NF + n_param_in
    w = {"W1": r.normal(0, 0.1, (H1, n_in)), "b1": r.normal(0, 0.1, H1),
         "W2": r.normal(0, 0.2, (H2, H1)), "b2": r.normal(0, 0.1, H2),
         "W3": r.normal(0, 0.3, (N_CARDS, H2)),
         "b3": r.normal(0, 0.1, N_CARDS)}
    path = str(tmp_path / name)
    save_npz(path, w, [n_in, H1, H2, N_CARDS], n_param_in)
    return path


@pytest.fixture(scope="module")
def liks(tmp_path_factory):
    """A GENERIC and a CONDITIONED likelihood.

    Weights are synthesized for the same reason `tests/test_profiled_search.py`
    synthesizes them (the trained npz is the lead's artifact, and every
    property here is weight-independent). The CONDITIONED one is not optional
    decoration: it exercises the kernel's params-appending branch, which the
    GENERIC path never touches.
    """
    d = tmp_path_factory.mktemp("p55b")
    gw, gm = load_profiler(_npz(d, "gen.npz", 0, 5))
    cw, cm = load_profiler(_npz(d, "cond.npz", PARAM_DIM, 11))
    sp = {s: param_vector(1 + s) for s in range(4)}
    return (ProfilerLikelihood(gw, None, gm),
            ProfilerLikelihood(cw, sp, cm))


def _play(seed, pids, heuristic=False):
    if heuristic:
        players = [HeuristicPlayer() for _ in range(4)]
    else:
        players = [PersonalityPlayer(np.random.default_rng([seed, s, 0xA1CE]),
                                     sample_personality(p))
                   for s, p in enumerate(pids)]
    state = deal(np.random.default_rng(seed))
    orig, plays = list(state.hands), []
    while not state.is_over():
        s = state.to_play
        c = players[s].choose(state.view_for(s))
        plays.append((s, c))
        state.play(c)
    return orig, plays


@pytest.fixture(scope="module")
def corpus():
    """(view, world, kind) pairs spanning every trick and three world kinds.

    * `true`     -- the real world; never dies, always finite.
    * `sampled`  -- what actually flows through `profiler_posterior`.
    * `permuted` -- opponent hands swapped, so the observed play is usually
                    ILLEGAL: this is how `-inf` gets into the corpus. Sampled
                    worlds respect the belief table's void deductions and in
                    practice never die, so without this the dead-world branch
                    would be pinned by nothing.

    Games are both personality (the deployment population) and heuristic (the
    Phase 1-4 script), and positions include mid-trick ones so the partial-trick
    featurization branches are covered.
    """
    _train, heldout = make_population(200, 50, 314159)
    rng = np.random.default_rng(20250815)
    pairs = []
    for g in range(6):
        pids = [int(x) for x in rng.choice(heldout, 4, replace=False)]
        orig, plays = _play(90000 + g, pids, heuristic=(g % 3 == 0))
        for trick in range(1, 14):
            for extra in (0, 2):
                n = 4 * (trick - 1) + extra
                if n > len(plays):
                    continue
                st = GameState(hands=list(orig))
                st.to_play = plays[0][0]
                for s, c in plays[:n]:
                    st.play(c)
                if st.is_over():
                    continue
                obs = st.to_play
                view = st.view_for(obs)
                opp = [(obs + 1 + i) % 4 for i in range(3)]
                pairs.append((view, [int(st.hands[s]) for s in opp], "true"))
                table = BeliefTable.from_view(view, Level.FULL)
                if not cards.cards_in(table.unseen_mask):
                    continue
                batch, _nf = kernel.sample_arrangements(
                    table, np.random.default_rng([g, trick, extra]), 3)
                for wh in batch:
                    pairs.append((view, [int(wh[i]) for i in range(3)],
                                  "sampled"))
                wh = batch[0]
                sz = [bin(int(wh[i])).count("1") for i in range(3)]
                if sz[0] == sz[1]:
                    pairs.append((view, [int(wh[1]), int(wh[0]), int(wh[2])],
                                  "permuted"))
    return pairs


def _both(view, world, lik, include_forced=False):
    """(fused, python) log-weights for one pair, restoring the flag."""
    try:
        P.FUSED_AUDIT = True
        got = profiler_world_logweight(view, world, lik, include_forced)
        P.FUSED_AUDIT = False
        want = profiler_world_logweight(view, world, lik, include_forced)
    finally:
        P.FUSED_AUDIT = True
    return got, want


def test_corpus_covers_the_branches_that_matter(corpus, liks):
    """The gate is only worth its name if the corpus reaches the hard cases."""
    generic, _cond = liks
    kinds = {k for _v, _w, k in corpus}
    assert {"true", "sampled", "permuted"} <= kinds
    n_inf = n_zero = 0
    for view, world, _k in corpus:
        lw, _ = _both(view, world, generic)
        n_inf += lw == -np.inf
        n_zero += lw == 0.0
    assert len(corpus) >= 500, len(corpus)
    assert n_inf > 0, "no DEAD world in the corpus: -inf branch unpinned"
    assert n_zero > 0, "no zero-evidence position: early-return branch unpinned"


@pytest.mark.parametrize("which", [0, 1], ids=["generic", "conditioned"])
def test_fused_logweight_is_bitwise_identical(corpus, liks, which):
    """THE GATE. Not a tolerance -- byte equality of the float64."""
    lik = liks[which]
    n = 0
    for view, world, kind in corpus:
        got, want = _both(view, world, lik)
        if got == -np.inf or want == -np.inf:
            assert got == want, f"{kind}: dead-world disagreement"
        else:
            assert _bits(got) == _bits(want), (
                f"{kind}: fused {got!r} != python {want!r} "
                f"(delta {got - want!r})")
        n += 1
    assert n >= 500


@pytest.mark.parametrize("which", [0, 1], ids=["generic", "conditioned"])
def test_include_forced_hook_is_bitwise_identical(corpus, liks, which):
    """The `include_forced` test hook must survive the fusion.

    `tests/test_profiled_search.py` uses it to prove the forced-ply skip is a
    no-op; if the kernel ignored the flag that proof would silently become
    vacuous rather than fail. Parametrized over both likelihoods so the
    forced-ply and params-appending branches are covered together rather than
    only one at a time.
    """
    lik = liks[which]
    for view, world, _k in corpus[:400]:
        got, want = _both(view, world, lik, include_forced=True)
        if got == -np.inf or want == -np.inf:
            assert got == want
        else:
            assert _bits(got) == _bits(want)


def test_log_reduction_variants_agree(corpus, liks):
    """In-kernel `np.log` vs numpy's, over bitwise-identical probabilities.

    The kernel returns both its own log-sum and the gathered per-ply
    probabilities; this pins the two reductions against each other. A failure
    here is NOT a bug in the fusion -- it means this platform's numba and numpy
    logs differ, and `LOG_IN_KERNEL = False` is the supported answer.
    """
    generic, _c = liks
    for view, world, _k in corpus[:400]:
        try:
            KA.LOG_IN_KERNEL = True
            a = profiler_world_logweight(view, world, generic)
            KA.LOG_IN_KERNEL = False
            b = profiler_world_logweight(view, world, generic)
        finally:
            KA.LOG_IN_KERNEL = True
        assert _bits(a) == _bits(b), f"{a!r} != {b!r}"


def test_kernel_probabilities_match_the_python_batch(corpus, liks):
    """The gathered p(observed card) equal `batch_probs`', ply for ply.

    Pins the fusion one level below the sum: if features or the net forward
    drifted, this fails on the probability rather than on a cancelled-out
    total.
    """
    from openhearts.belief.weighted import _reconstruct_original_hands
    from openhearts.engine.game import legal_moves
    from openhearts.opponent.obsfeat import observer_features

    generic, _c = liks
    checked = 0
    for view, world, kind in corpus[:200]:
        if kind == "permuted":
            continue
        hands, all_plays = _reconstruct_original_hands(view, world)
        if not all_plays:
            continue
        got = KA.audit_world_probs(hands, all_plays, view.seat,
                                   generic.weights, generic.n_in)
        if got is None:
            continue
        # the Python path's own rows -> batch_probs -> gather
        st = GameState(hands=list(hands))
        st.to_play = all_plays[0][0]
        rows, seats, masks, obs = [], [], [], []
        for seat, card in all_plays:
            legal = legal_moves(st.hands[seat], tuple(st.current_trick),
                                st.hearts_broken, st.trick_number)
            if seat != view.seat and len(cards.cards_in(int(legal))) > 1:
                rows.append(observer_features(st, seat))
                seats.append(seat)
                masks.append(int(legal))
                obs.append(card)
            st.play(card)
        if not rows:
            assert got.size == 0
            continue
        out = generic.batch_probs(rows, seats, masks)
        want = np.array([out[i, c] for i, c in enumerate(obs)])
        assert got.shape == want.shape
        assert got.tobytes() == want.tobytes(), "per-ply probabilities differ"
        checked += 1
    assert checked >= 20


def test_posterior_is_identical_through_the_factory(liks, corpus):
    """Gate 2: the consumer's outputs, not just the likelihood's.

    NOTE the flag being toggled is `FUSED_AUDIT`, not `kernel.jit_enabled()`.
    `profiler_posterior` reads `jit_enabled()` to choose the world SAMPLER, so
    forcing the Python likelihood that way would change which worlds are drawn
    and make this comparison meaningless. The rng is re-seeded per run for the
    same reason.
    """
    generic, _c = liks
    views, seen = [], set()
    for view, _w, _k in corpus:
        key = (view.seat, view.hand, len(view.history),
               len(view.current_trick))
        if key not in seen:
            seen.add(key)
            views.append(view)
    views = views[:25]
    for i, view in enumerate(views):
        outs = []
        for fused in (True, False):
            try:
                P.FUSED_AUDIT = fused
                outs.append(profiler_posterior(
                    view, Level.FULL, generic, 40,
                    np.random.default_rng([424242, i]), 5000,
                    keep_worlds=True))
            finally:
                P.FUSED_AUDIT = True
        a, b = outs
        assert a.probs.tobytes() == b.probs.tobytes()
        assert _bits(a.n_effective) == _bits(b.n_effective)
        assert a.worlds == b.worlds
        assert a.weights == b.weights
        assert (a.draws_used, a.n_worlds_used) == (b.draws_used,
                                                   b.n_worlds_used)


def test_oracle_shaped_seat_params_are_bitwise_identical(corpus, liks):
    """`profiled-ORACLE`'s exact seat_params shape: THREE seats, not four.

    `experiments/run_ablation5.py`'s ORACLE hook rebuilds `lik.seat_params` per
    deal as {opponent seat -> true param vector}, so the OBSERVER's seat is
    ABSENT from the mapping. `_fused_params` fills that row with NaN precisely
    because it is never read (`row()` is only called for opponents) -- this
    pins that reasoning instead of trusting it: if the kernel ever read the
    observer's row, the NaN would poison the softmax and the log-weight would
    come back NaN rather than matching.

    ORACLE is the one row of the paused ablation's remainder that the fusion
    actually serves, so this case is pinned by name.
    """
    _generic, cond = liks
    n = 0
    for view, world, _k in corpus[:300]:
        opp = [(view.seat + 1 + i) % 4 for i in range(3)]
        cond.seat_params = {s: param_vector(1 + s) for s in opp}
        params = P._fused_params(cond)
        assert np.isnan(params[view.seat]).all(), (
            "the observer's absent row should be NaN-filled, not zero-filled")
        got, want = _both(view, world, cond)
        if got == -np.inf or want == -np.inf:
            assert got == want
        else:
            assert not np.isnan(got), "observer's NaN row leaked into the net"
            assert _bits(got) == _bits(want)
        n += 1
    cond.seat_params = {s: param_vector(1 + s) for s in range(4)}
    assert n >= 100


def test_adapted_likelihood_falls_back_to_python(liks):
    """A `ProfilerLikelihood` SUBCLASS must not be fused (5.5b scope).

    `opponent.adapt.AdaptedLikelihood` overrides `batch_probs` with a per-seat
    mixture over a pool -- a different computation the kernel does not
    implement. The dispatch refuses anything that is not exactly
    `ProfilerLikelihood`, so the fallback is automatic rather than something a
    caller must remember.
    """
    generic, cond = liks

    class Sneaky(ProfilerLikelihood):
        pass

    sub = Sneaky(generic.weights, None, generic.meta)
    assert P._fused_params(sub) is None, "a subclass must not be fused"
    assert P._fused_params(generic) is not None
    assert P._fused_params(cond) is not None
    assert P._fused_params(cond).shape == (4, PARAM_DIM)
