"""Gate tests for the JIT'd world audit (Phase 3.5).

`kernel.audit_world` replays an observed play sequence inside a candidate
world and returns the same play-likelihood product that
`belief.weighted.world_weight` computes in pure Python. The Python function
is the reference; these tests pin the compiled one against it over a large
corpus of (view, candidate world) pairs spanning every game phase, true and
perturbed worlds, and epsilon in {0, 0.1, 0.3}.

Bitwise equality is the target (same operation order, float64, no fastmath);
the tests assert bitwise and would need to be consciously relaxed -- never
silently -- if that ever stopped holding.
"""
import os

import numpy as np
import pytest

from openhearts.belief import weighted
from openhearts.belief.table import Level
from openhearts.belief.weighted import WeightedPosterior, world_weight
from openhearts.engine import cards, kernel
from openhearts.engine.game import deal
from openhearts.players.heuristic import HeuristicPlayer
from openhearts.players.randomized import RandomizedHeuristic

pytestmark = pytest.mark.skipif(
    not kernel.HAVE_NUMBA or os.environ.get("OPENHEARTS_NO_JIT") == "1",
    reason="JIT path disabled (no numba, or OPENHEARTS_NO_JIT=1)")

EPSILONS = (0.0, 0.1, 0.3)


# ---------------------------------------------------------------- corpus
def _phase(ply):
    """Game phase bucket for a ply index (0..51)."""
    if ply <= 12:
        return "early"
    if ply <= 38:
        return "mid"
    return "late"


def _swap_perturbation(world, rng):
    """Swap two same-suit cards between two different opponents.

    Same-suit keeps the perturbation inside the interesting part of the
    space (legality of the observed plays can still change, but hand sizes
    and the deck partition are preserved, so the structural asserts in
    `_reconstruct_original_hands` are never the thing being tested).
    Returns None when no such pair exists.
    """
    pairs = []
    for i in range(3):
        for j in range(i + 1, 3):
            for s in range(4):
                a = cards.cards_in(world[i] & (((1 << 13) - 1) << (13 * s)))
                b = cards.cards_in(world[j] & (((1 << 13) - 1) << (13 * s)))
                if a and b:
                    pairs.append((i, j, a, b))
    if not pairs:
        return None
    i, j, a, b = pairs[rng.integers(len(pairs))]
    ca = a[rng.integers(len(a))]
    cb = b[rng.integers(len(b))]
    out = list(world)
    out[i] = (out[i] & ~cards.bit(ca)) | cards.bit(cb)
    out[j] = (out[j] & ~cards.bit(cb)) | cards.bit(ca)
    return out


def _cross_suit_perturbation(world, rng):
    """Swap two DIFFERENT-suit cards between two different opponents.

    Unlike the same-suit swap, this changes each seat's per-suit counts, so
    it can make a seat void in a suit it was observed to follow -- i.e. it
    is the perturbation class that produces genuinely ILLEGAL replays, and
    therefore zero weights at every epsilon (not just at eps=0, where a mere
    policy mismatch already suffices). Hand sizes and the deck partition are
    still preserved, so the structural asserts stay out of it.
    """
    pairs = []
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            for s in range(4):
                for t in range(4):
                    if s == t:
                        continue
                    a = cards.cards_in(world[i] & (((1 << 13) - 1) << (13 * s)))
                    b = cards.cards_in(world[j] & (((1 << 13) - 1) << (13 * t)))
                    if a and b:
                        pairs.append((i, j, a, b))
    if not pairs:
        return None
    i, j, a, b = pairs[rng.integers(len(pairs))]
    ca = a[rng.integers(len(a))]
    cb = b[rng.integers(len(b))]
    out = list(world)
    out[i] = (out[i] & ~cards.bit(ca)) | cards.bit(cb)
    out[j] = (out[j] & ~cards.bit(cb)) | cards.bit(ca)
    return out


def _corpus(n_pairs, seed0=7000):
    """(view, world, is_truth, phase) pairs from replayed games.

    Games are played by the plain heuristic and by RandomizedHeuristic (so
    that perturbed *and* truth worlds can both fail the strict audit), and a
    view is taken at every ply for a rotating observer seat -- mid-trick
    views included, since `trick_len > 0` is a distinct kernel path.
    """
    out = []
    rng = np.random.default_rng(seed0)
    g = 0
    while len(out) < n_pairs:
        deviant = (g % 3 == 2)
        state = deal(np.random.default_rng(seed0 + g))
        if deviant:
            players = [RandomizedHeuristic(np.random.default_rng(seed0 + g + s),
                                           0.15) for s in range(4)]
        else:
            players = [HeuristicPlayer() for _ in range(4)]
        g += 1
        for ply in range(52):
            if state.is_over():
                break
            observer = (ply + g) % 4
            view = state.view_for(observer)
            truth = [state.hands[(observer + 1 + i) % 4] for i in range(3)]
            out.append((view, truth, True, _phase(ply)))
            for make in (_swap_perturbation, _cross_suit_perturbation):
                pert = make(truth, rng)
                if pert is not None:
                    out.append((view, pert, False, _phase(ply)))
            seat = state.to_play
            state.play(players[seat].choose(state.view_for(seat)))
    return out[:n_pairs]


_CORPUS = None


def corpus():
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = _corpus(3200)
    return _CORPUS


def _kernel_weight(view, world, epsilon, early_exit=True):
    hands, all_plays = weighted._reconstruct_original_hands(view, world)
    return kernel.audit_world_weight(hands, all_plays, view.seat, epsilon,
                                     early_exit=early_exit)


# ------------------------------------------------- gate 1: kernel == python
def test_corpus_covers_phases_and_world_kinds():
    c = corpus()
    assert len(c) >= 3000
    phases = {p: 0 for p in ("early", "mid", "late")}
    truths = 0
    for _v, _w, is_truth, ph in c:
        phases[ph] += 1
        truths += is_truth
    for ph, n in phases.items():
        assert n >= 200, f"phase {ph} underrepresented: {n}"
    assert truths >= 500 and (len(c) - truths) >= 1000


def test_kernel_matches_python_weight():
    policy = HeuristicPlayer()
    c = corpus()
    max_rel = 0.0
    n_zero = {e: 0 for e in EPSILONS}
    n_pos = 0
    for eps in EPSILONS:
        for view, world, _is_truth, _ph in c:
            py = weighted.world_weight_python(view, world, policy, eps)
            jt = _kernel_weight(view, world, eps)
            # support equality first: rejection sampling only cares that the
            # zero sets agree, and a relative error is undefined at 0.
            assert (py == 0.0) == (jt == 0.0), (
                f"support mismatch eps={eps}: python={py} kernel={jt}"
            )
            if py == 0.0:
                n_zero[eps] += 1
                continue
            n_pos += 1
            if py != jt:
                max_rel = max(max_rel, abs(py - jt) / max(abs(py), abs(jt)))
            assert py == jt, (
                f"weight mismatch eps={eps}: python={py!r} kernel={jt!r}"
            )
    assert max_rel == 0.0, f"not bitwise: max relative error {max_rel}"
    assert n_pos > 0
    # zeros must appear at EVERY epsilon, not only at eps=0: at eps>0 the
    # only source of a zero is an ILLEGAL observed play, so this is what
    # actually exercises the kernel's legality check.
    for eps in EPSILONS:
        assert n_zero[eps] >= 50, f"only {n_zero[eps]} zero weights at {eps}"


def test_perturbations_produce_both_outcomes():
    """The corpus must actually exercise pass AND fail audits at eps=0."""
    policy = HeuristicPlayer()
    passing = failing = 0
    for view, world, is_truth, _ph in corpus():
        if is_truth:
            continue
        w = weighted.world_weight_python(view, world, policy, 0.0)
        if w > 0.0:
            passing += 1
        else:
            failing += 1
    assert passing >= 50, f"only {passing} audit-passing perturbations"
    assert failing >= 500, f"only {failing} audit-failing perturbations"


def test_early_exit_agrees_with_full_replay():
    """Rung-1 early exit changes speed, never the accept/reject decision.

    "Full replay" here means: do not short-circuit on a zero POLICY factor.
    An illegal observed play still terminates the replay in both variants --
    the world cannot produce the rest of the sequence at all.
    """
    checked = 0
    for eps in EPSILONS:
        for view, world, _is_truth, _ph in corpus():
            fast = _kernel_weight(view, world, eps, early_exit=True)
            full = _kernel_weight(view, world, eps, early_exit=False)
            assert (fast == 0.0) == (full == 0.0)
            assert fast == full, f"eps={eps}: {fast!r} != {full!r}"
            checked += 1
    assert checked >= 9000


def test_truth_world_always_survives_heuristic_games():
    policy = HeuristicPlayer()
    n = 0
    for g in range(6):
        state = deal(np.random.default_rng(9100 + g))
        players = [HeuristicPlayer() for _ in range(4)]
        for ply in range(52):
            if state.is_over():
                break
            for observer in range(4):
                view = state.view_for(observer)
                truth = [state.hands[(observer + 1 + i) % 4] for i in range(3)]
                hands, all_plays = weighted._reconstruct_original_hands(
                    view, truth)
                assert kernel.audit_world_weight(
                    hands, all_plays, observer, 0.0) == 1.0
                n += 1
            seat = state.to_play
            state.play(players[seat].choose(state.view_for(seat)))
    assert n >= 800


def test_no_evidence_world_weighs_one():
    """Before any card is played there is nothing to audit."""
    state = deal(np.random.default_rng(4242))
    observer = state.to_play
    view = state.view_for(observer)
    truth = [state.hands[(observer + 1 + i) % 4] for i in range(3)]
    for eps in EPSILONS:
        assert world_weight(view, truth, HeuristicPlayer(), eps) == 1.0
        assert weighted.world_weight_python(
            view, truth, HeuristicPlayer(), eps) == 1.0


def test_dispatch_uses_kernel_only_for_plain_heuristic():
    """A non-heuristic policy must never be answered by the hardcoded kernel."""
    state = deal(np.random.default_rng(555))
    players = [HeuristicPlayer() for _ in range(4)]
    observer = state.to_play
    for _ in range(20):
        seat = state.to_play
        state.play(players[seat].choose(state.view_for(seat)))
    view = state.view_for(observer)
    truth = [state.hands[(observer + 1 + i) % 4] for i in range(3)]

    class Contrary(HeuristicPlayer):
        def choose(self, view):
            legal = cards.cards_in(view.legal_moves)
            return legal[-1]

    p = Contrary()
    # the truth world weighs 1.0 under the heuristic, so a wrongly dispatched
    # kernel would answer 1.0 here; Contrary explains almost nothing.
    assert world_weight(view, truth, p, 0.0) == 0.0
    assert world_weight(view, truth, p, 0.0) == \
        weighted.world_weight_python(view, truth, p, 0.0)
    assert _kernel_weight(view, truth, 0.0) == 1.0


# ------------------------------------------- gate 2: posterior JIT vs NO_JIT
def _posterior(view, seed):
    return WeightedPosterior.from_view(
        view, Level.FULL, HeuristicPlayer(), epsilon=0.0, n_worlds=60,
        rng=np.random.default_rng(seed), max_draws=20000)


def _mid_view(seed=31, plies=24):
    state = deal(np.random.default_rng(seed))
    players = [HeuristicPlayer() for _ in range(4)]
    observer = state.to_play
    for _ in range(plies):
        seat = state.to_play
        state.play(players[seat].choose(state.view_for(seat)))
    return state, state.view_for(observer)


def test_posterior_jit_vs_nojit_consistent(monkeypatch):
    # A late-trick boundary: strict filtering is survivable there, so both
    # modes fill their world quota and the comparison is not confounded by
    # max_draws exhaustion. (Mid-game boundaries around trick 3-6 exhaust
    # 20k draws on BOTH paths -- a pre-existing property of epsilon=0
    # rejection sampling, measured in results/survival_smoke_pre35.txt, not
    # something this phase changes.)
    state, view = _mid_view(seed=31, plies=44)
    assert kernel.jit_enabled()
    jit_posts = [_posterior(view, s) for s in (1, 2, 3)]
    monkeypatch.setenv("OPENHEARTS_NO_JIT", "1")
    kernel.reset_jit_enabled()
    try:
        assert not kernel.jit_enabled()
        py_posts = [_posterior(view, s) for s in (1, 2, 3)]
    finally:
        monkeypatch.delenv("OPENHEARTS_NO_JIT", raising=False)
        kernel.reset_jit_enabled()
    assert kernel.jit_enabled()

    for p in jit_posts + py_posts:
        # epsilon=0 => every surviving weight is exactly 1.0
        assert p.n_worlds_used == 60
        assert p.n_effective == 60.0
        assert p.total_weight == 60.0
        assert p.draws_used >= p.n_worlds_used

    # strictness consistency: the two paths must accept worlds at a similar
    # RATE. A too-lenient audit on either side would show up here.
    jit_draws = np.mean([p.draws_used for p in jit_posts])
    py_draws = np.mean([p.draws_used for p in py_posts])
    assert 0.5 < jit_draws / py_draws < 2.0, (jit_draws, py_draws)

    jit_probs = np.mean([p.probs for p in jit_posts], axis=0)
    py_probs = np.mean([p.probs for p in py_posts], axis=0)
    # both estimate the same posterior from 180 worlds each; Monte Carlo
    # noise at this sample size is a few hundredths per card.
    assert np.abs(jit_probs - py_probs).max() < 0.25
    assert np.abs(jit_probs.sum(axis=1) - py_probs.sum(axis=1)).max() < 1e-9
    # support: a card impossible under one path must be impossible under the
    # other only up to sampling, but the ZEROS from constraints are shared.
    unseen = jit_posts[0].unseen_mask
    for i in range(3):
        for c in range(52):
            if not (unseen & cards.bit(c)):
                assert jit_probs[i, c] == 0.0 and py_probs[i, c] == 0.0


def test_nojit_python_path_is_bitwise_unchanged(monkeypatch):
    """With NO_JIT the audit must be the untouched Python replay."""
    state, view = _mid_view(seed=77, plies=30)
    truth = [state.hands[(view.seat + 1 + i) % 4] for i in range(3)]
    monkeypatch.setenv("OPENHEARTS_NO_JIT", "1")
    kernel.reset_jit_enabled()
    try:
        for eps in EPSILONS:
            assert world_weight(view, truth, HeuristicPlayer(), eps) == \
                weighted.world_weight_python(view, truth, HeuristicPlayer(),
                                             eps)
    finally:
        monkeypatch.delenv("OPENHEARTS_NO_JIT", raising=False)
        kernel.reset_jit_enabled()
    assert os.environ.get("OPENHEARTS_NO_JIT") is None
