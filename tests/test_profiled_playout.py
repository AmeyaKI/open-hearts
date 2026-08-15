"""Phase 5 Task 6: model-driven playouts (Organ 2) -- the playout wiring.

Weights are SYNTHESIZED (seeded random export through `npz_io.save_npz`), for
the reason `tests/test_profiled_search.py` records: `models/profiler_v1.npz` is
the LEAD's artifact and a suite that requires it is red for anyone who has not
trained. Every property here is weight-independent (a reduction identity, an
exact featurization equality, legality/termination, and sampling frequencies
compared against THAT net's own distribution). The GAMES and the states are
real -- dealt and played by the engine and by Task 1's `PersonalityPlayer`s.

The five links of the chain, and which test pins each (no test pretends to
cover more than its own link):

  features       -> `test_inkernel_features_match_obsfeat`   (EXACT)
  distribution   -> `test_inkernel_probs_match_profiler`     (EXACT)
  loop/bookkeep. -> `test_reduction_*`                       (bitwise vs kernel)
  sampler        -> `test_sampling_frequencies_*`            (3 sigma)
  both modes     -> `test_jit_vs_nojit_same_distribution`    (3 sigma, NOT
                                                              bitwise -- two
                                                              different RNGs)
"""
import copy
import os

import numpy as np
import pytest

from openhearts.engine import cards, kernel
from openhearts.engine import kernel_profiled as kp
from openhearts.engine.features import NF
from openhearts.engine.game import deal, legal_moves, play_game
from openhearts.engine.kernel import jit_enabled, reset_jit_enabled
from openhearts.engine.state import GameState
from openhearts.opponent.infer import load_profiler, profiler_probs
from openhearts.opponent.npz_io import N_CARDS, save_npz
from openhearts.opponent.obsfeat import observer_features
from openhearts.opponent.params import PARAM_DIM
from openhearts.players.heuristic import HeuristicPlayer
from openhearts.players.personality import (PersonalityPlayer,
                                            make_population,
                                            sample_personality)
from openhearts.search.profiled import (ProfiledSearchPlayer,
                                        ProfilerLikelihood,
                                        profiler_posterior_factory)
from openhearts.belief.table import Level

H1, H2 = 24, 12


def _make_npz(tmp_path, name, n_param_in=0, seed=11):
    rng = np.random.default_rng(seed)
    n_in = NF + n_param_in
    w = {"W1": rng.normal(0, 0.1, (H1, n_in)), "b1": rng.normal(0, 0.1, H1),
         "W2": rng.normal(0, 0.2, (H2, H1)), "b2": rng.normal(0, 0.1, H2),
         "W3": rng.normal(0, 0.3, (N_CARDS, H2)),
         "b3": rng.normal(0, 0.1, N_CARDS)}
    path = str(tmp_path / name)
    save_npz(path, w, [n_in, H1, H2, N_CARDS], n_param_in)
    return path


@pytest.fixture(scope="module")
def generic(tmp_path_factory):
    path = _make_npz(tmp_path_factory.mktemp("p5t6"), "generic.npz", 0, 11)
    return load_profiler(path)[0]


@pytest.fixture(scope="module")
def conditioned(tmp_path_factory):
    path = _make_npz(tmp_path_factory.mktemp("p5t6c"), "cond.npz", PARAM_DIM,
                     12)
    return load_profiler(path)[0]


# --------------------------------------------------------------------------
# state generators
# --------------------------------------------------------------------------
def _states_along_a_game(seed, every=1):
    """Snapshots of one heuristic game, one per ply (all game phases)."""
    state = deal(np.random.default_rng(seed))
    h = HeuristicPlayer()
    out = []
    i = 0
    while not state.is_over():
        if i % every == 0:
            out.append(state.copy())
        state.play(h.choose(state.view_for(state.to_play)))
        i += 1
    return out


def _personality_states(seed, pids, stop_ply):
    """A mid-game state from a 4-personality game, stopped at `stop_ply`."""
    players = [PersonalityPlayer(np.random.default_rng([seed, s, 0xA1CE]),
                                 sample_personality(p))
               for s, p in enumerate(pids)]
    state = deal(np.random.default_rng(seed))
    for _ in range(stop_ply):
        if state.is_over():
            break
        seat = state.to_play
        state.play(players[seat].choose(state.view_for(seat)))
    return state


def _assert_legal_continuation(before: GameState, after: GameState):
    """Replay everything `after` played beyond `before` through GameState.play.

    `GameState.play` asserts legality, so a single illegal card anywhere in the
    playout raises here. It also proves the kernel's write-back left a
    consistent state: the replay is driven by the engine's own turn order.
    """
    n0 = len(before.history) + len(before.current_trick)
    plays = list(after.history) + list(after.current_trick)
    replay = before.copy()
    for seat, card in plays[n0:]:
        assert seat == replay.to_play, "kernel play log desyncs from the engine"
        replay.play(card)
    assert replay.hands == after.hands
    assert replay.scores == after.scores
    assert replay.hearts_broken == after.hearts_broken
    assert replay.trick_number == after.trick_number


# --------------------------------------------------------------------------
# (a) REDUCTION: one-hot heuristic-match distribution == heuristic playout
# --------------------------------------------------------------------------
def test_reduction_playout_to_end(generic):
    """Pre-registered reduction test (PHASE5_PLAN Task 6, playout wiring)."""
    n = 0
    for seed in range(30):
        for st in _states_along_a_game(seed, every=4):
            for our_seat in range(4):
                a, b = st.copy(), st.copy()
                kernel.run_playout(a)
                kp.seed_playouts(1234 + n)
                kp.run_playout_profiled(b, our_seat, generic,
                                        kp.MODE_HEURISTIC_ONEHOT)
                assert a.scores == b.scores
                assert a.history == b.history
                assert a.hands == b.hands
                assert a.hearts_broken == b.hearts_broken
                assert a.trick_number == b.trick_number
                n += 1
    assert n >= 500, n


def test_reduction_playout_until_decision(generic):
    """Same reduction for the stop-at-our-decision entry, status included."""
    n = 0
    for seed in range(20):
        for st in _states_along_a_game(100 + seed, every=5):
            for stop_seat in range(4):
                a, b = st.copy(), st.copy()
                sa = kernel.run_playout_until_decision(a, stop_seat)
                kp.seed_playouts(99 + n)
                sb = kp.run_playout_until_decision_profiled(
                    b, stop_seat, stop_seat, generic,
                    kp.MODE_HEURISTIC_ONEHOT)
                assert sa == sb
                assert a.scores == b.scores and a.history == b.history
                assert a.hands == b.hands and a.to_play == b.to_play
                n += 1
    assert n >= 300, n


# --------------------------------------------------------------------------
# (b) in-kernel featurization / distribution pinned against the Python helpers
# --------------------------------------------------------------------------
def _feature_pairs(limit):
    """(state, seat) pairs spanning every phase of several games."""
    pairs = []
    for seed in range(40):
        for st in _states_along_a_game(200 + seed, every=2):
            for seat in range(4):
                pairs.append((st, seat))
                if len(pairs) >= limit:
                    return pairs
    return pairs


def test_inkernel_features_match_obsfeat(generic):
    pairs = _feature_pairs(1200)
    assert len(pairs) >= 1000, len(pairs)
    for st, seat in pairs:
        legal = legal_moves(st.hands[seat], tuple(st.current_trick),
                            st.hearts_broken, st.trick_number)
        feats, _p = kp.ply_probs_for_state(st, seat, legal, generic)
        ref = observer_features(st, seat)
        assert feats.shape == (NF,)
        # EXACT: both go through the same featurizer with the same array, so
        # anything but bitwise equality means the in-kernel row is built from
        # different inputs (the scratch-buffer leak this guards against).
        assert np.array_equal(feats, ref), (seat, st.trick_number)


def test_inkernel_probs_match_profiler(generic):
    """features -> distribution link, EXACT against `infer.profiler_probs`."""
    pairs = _feature_pairs(1000)
    assert len(pairs) >= 1000, len(pairs)
    for st, seat in pairs:
        legal = int(legal_moves(st.hands[seat], tuple(st.current_trick),
                                st.hearts_broken, st.trick_number))
        if legal == 0:
            continue
        _f, probs = kp.ply_probs_for_state(st, seat, legal, generic)
        ref = profiler_probs(*generic, observer_features(st, seat), legal)
        assert np.array_equal(probs, ref)
        # truth-safety of the playout policy: strictly positive on the mask,
        # exactly zero off it.
        for c in range(52):
            if (legal >> c) & 1:
                assert probs[c] > 0.0
            else:
                assert probs[c] == 0.0


# --------------------------------------------------------------------------
# (c) legality + termination fuzz
# --------------------------------------------------------------------------
def test_profiled_playouts_are_legal_and_terminate(generic):
    train, _held = make_population(200, 50, 314159)
    rng = np.random.default_rng(4242)
    n = 0
    i = 0
    # Loops until 500 VALID mid-game states have been played out; the plan's
    # ">=500 diverse mid-game states" is a count of states exercised, not of
    # states attempted.
    while n < 500:
        i += 1
        pids = [int(x) for x in rng.choice(train, size=4, replace=False)]
        stop = int(rng.integers(1, 45))
        st = _personality_states(500000 + i, pids, stop)
        if st.is_over():
            continue
        before = st.copy()
        our_seat = int(rng.integers(0, 4))
        kp.seed_playouts(int(rng.integers(2 ** 31)))
        kp.run_playout_profiled(st, our_seat, generic)
        assert st.is_over()
        assert sum(st.scores) == 26, st.scores
        played = 0
        for _s, c in st.history:
            assert not (played >> c) & 1
            played |= 1 << c
        assert played == cards.FULL_DECK
        _assert_legal_continuation(before, st)
        n += 1
    assert n == 500


# --------------------------------------------------------------------------
# (d) sampling statistics (statistical pinning, Phase 2.6 precedent)
# --------------------------------------------------------------------------
def _first_play_frequencies(state, our_seat, weights, n_draws, seed0):
    counts = np.zeros(52, dtype=np.int64)
    for i in range(n_draws):
        st = state.copy()
        kp.seed_playouts(seed0 + i)
        kp.run_playout_profiled(st, our_seat, weights)
        plays = list(st.history) + list(st.current_trick)
        n0 = len(state.history) + len(state.current_trick)
        counts[plays[n0][1]] += 1
    return counts


def _fixed_sampling_state():
    """A mid-game state whose seat to act is an OPPONENT with >1 legal move."""
    train, _h = make_population(200, 50, 314159)
    st = _personality_states(500777, [int(x) for x in train[:4]], 17)
    seat = st.to_play
    legal = int(legal_moves(st.hands[seat], tuple(st.current_trick),
                            st.hearts_broken, st.trick_number))
    assert bin(legal).count("1") > 1
    our_seat = (seat + 1) % 4        # the acting seat is NOT ours
    return st, seat, legal, our_seat


def test_sampling_frequencies_match_profiler(generic):
    st, seat, legal, our_seat = _fixed_sampling_state()
    ref = profiler_probs(*generic, observer_features(st, seat), legal)
    n_draws = 6000
    counts = _first_play_frequencies(st, our_seat, generic, n_draws, 90000)
    assert counts.sum() == n_draws
    for c in cards.cards_in(legal):
        p = ref[c]
        sigma = np.sqrt(n_draws * p * (1 - p))
        assert abs(counts[c] - n_draws * p) <= 3.0 * sigma, (
            f"card {c}: {counts[c]} vs expected {n_draws * p:.1f} "
            f"(3 sigma = {3 * sigma:.1f})")
    assert counts[~np.isin(np.arange(52), cards.cards_in(legal))].sum() == 0


# --------------------------------------------------------------------------
# (e) JIT vs NO_JIT: same distribution FAMILY, not bitwise
# --------------------------------------------------------------------------
def test_jit_vs_nojit_same_distribution(generic):
    """Two different generators (numba's vs numpy's global) draw from the SAME
    masked softmax, so only the DISTRIBUTION can be pinned across the modes --
    documented in `kernel_profiled`'s docstring, not a weakened assertion."""
    st, seat, legal, our_seat = _fixed_sampling_state()
    ref = profiler_probs(*generic, observer_features(st, seat), legal)
    n_draws = 4000
    before = os.environ.get("OPENHEARTS_NO_JIT")
    try:
        os.environ["OPENHEARTS_NO_JIT"] = "1"
        reset_jit_enabled()
        assert not jit_enabled()
        c_off = _first_play_frequencies(st, our_seat, generic, n_draws, 700)
        os.environ.pop("OPENHEARTS_NO_JIT")
        reset_jit_enabled()
        c_on = _first_play_frequencies(st, our_seat, generic, n_draws, 700) \
            if jit_enabled() else c_off
    finally:
        if before is None:
            os.environ.pop("OPENHEARTS_NO_JIT", None)
        else:
            os.environ["OPENHEARTS_NO_JIT"] = before
        reset_jit_enabled()
    for c in cards.cards_in(legal):
        p = ref[c]
        sigma = np.sqrt(n_draws * p * (1 - p))
        for counts, tag in ((c_off, "NO_JIT"), (c_on, "JIT")):
            assert abs(counts[c] - n_draws * p) <= 3.0 * sigma, (
                f"{tag} card {c}: {counts[c]} vs {n_draws * p:.1f}")


def test_nojit_playouts_do_not_disturb_global_numpy_rng(generic):
    """The Python fallback's stream is private (module `_enter`/`_exit`)."""
    before = os.environ.get("OPENHEARTS_NO_JIT")
    try:
        os.environ["OPENHEARTS_NO_JIT"] = "1"
        reset_jit_enabled()
        np.random.seed(3)
        expected = [np.random.random() for _ in range(3)]
        np.random.seed(3)
        got = [np.random.random()]
        st = _personality_states(500123, [0, 1, 2, 3], 12)
        kp.seed_playouts(5)
        kp.run_playout_profiled(st.copy(), 0, generic)
        got += [np.random.random(), np.random.random()]
    finally:
        if before is None:
            os.environ.pop("OPENHEARTS_NO_JIT", None)
        else:
            os.environ["OPENHEARTS_NO_JIT"] = before
        reset_jit_enabled()
    assert got == expected


# --------------------------------------------------------------------------
# (f) end-to-end: R / RI / RIA play legal games
# --------------------------------------------------------------------------
def _make_player(kind, generic_w, conditioned_w, rng):
    lik = ProfilerLikelihood(generic_w, None)
    mixtures = None
    if kind == "RIA":
        from openhearts.opponent.adapt import (AdaptedLikelihood, PoolProfiler,
                                               SeatMixture)
        pool = PoolProfiler(conditioned_w, k=4)
        mixtures = {s: SeatMixture(pool) for s in (1, 2, 3)}
        lik = AdaptedLikelihood(pool, mixtures, generic=lik, blend=0.2)
    factory = profiler_posterior_factory(lik, level=Level.FULL, n_worlds=8,
                                         max_draws=400, keep_worlds=True)
    weights = None if kind == "R" else generic_w
    return ProfiledSearchPlayer(Level.FULL, 8, 2, rng,
                                posterior_factory=factory,
                                playout_weights=weights)


@pytest.mark.parametrize("kind", ["R", "RI", "RIA"])
def test_end_to_end_game(kind, generic, conditioned):
    rng = np.random.default_rng(7)
    bot = _make_player(kind, generic, conditioned, rng)
    train, _h = make_population(200, 50, 314159)
    opps = [PersonalityPlayer(np.random.default_rng([9, s, 0xA1CE]),
                              sample_personality(int(train[s])))
            for s in range(1, 4)]
    state = play_game(deal(np.random.default_rng(9)), [bot] + opps)
    assert state.is_over()
    assert sum(state.scores) == 26, state.scores
    assert len(state.history) == 52
    n = bot.observe_hand(state.history)
    assert n == (3 if kind == "RIA" else 0)
    bot.reset_mixtures()


def test_r_row_is_honest_playouts(generic):
    """`playout_weights=None` must delegate to HonestSearchPlayer EXACTLY."""
    from openhearts.search.honest import HonestSearchPlayer

    st = _personality_states(500321, [0, 1, 2, 3], 10)
    a, b = st.copy(), st.copy()
    p = ProfiledSearchPlayer(Level.FULL, 4, 0, np.random.default_rng(1),
                             playout_weights=None)
    h = HonestSearchPlayer(Level.FULL, 4, 0, np.random.default_rng(1))
    p._playout(a, 0)
    h._playout(b, 0)
    assert a.history == b.history and a.scores == b.scores


def test_adapted_mixture_weights_move_after_a_hand(generic, conditioned):
    """`observe_hand` actually updates (and `reset_mixtures` undoes it)."""
    from openhearts.opponent.adapt import PoolProfiler, SeatMixture

    rng = np.random.default_rng(3)
    bot = _make_player("RIA", generic, conditioned, rng)
    lik = bot._adapted
    assert lik is not None
    w0 = lik.mixtures[1].weights.copy()
    state = play_game(deal(np.random.default_rng(21)),
                      [HeuristicPlayer() for _ in range(4)])
    bot.observe_hand(state.history)
    w1 = lik.mixtures[1].weights
    assert not np.allclose(w0, w1)
    assert lik.mixtures[1].n_hands == 1
    bot.reset_mixtures()
    assert np.allclose(lik.mixtures[1].weights, w0)
    assert isinstance(lik.mixtures[1], SeatMixture)
    assert isinstance(lik.pool, PoolProfiler)
