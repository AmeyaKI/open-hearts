"""Phase 5 Task 4: the profiler likelihood in the belief socket.

Scope: the LIKELIHOOD wiring only. Task 6 adds model-driven playouts and its
tests land in this file beside these.

Weights are SYNTHESIZED (seeded random export through `npz_io.save_npz`), for
the same reason `tests/test_profiler_infer.py` synthesizes them: the trained
`models/profiler_v1.npz` is the LEAD's artifact and a suite that needs it is
red for anyone who has not trained. Every property under test here -- the
product-of-probabilities identity, the forced-ply skip being a no-op,
truth-safety, JIT/NO_JIT agreement -- is weight-independent. The GAMES,
however, are real: played by real `PersonalityPlayer`s from Task 1, so the
replayed views, legal masks and observed cards are the genuine article.

The reference the audit is pinned against is a pure-Python replay using
`npz_io.forward_numpy` + `masked_probs_numpy` (dense, unmasked-then-masked,
untransposed W1) -- it shares neither of `infer.py`'s two optimizations, so
agreement is evidence and not a tautology.
"""
import os

import numpy as np
import pytest

from openhearts.belief.table import Level
from openhearts.belief.weighted import (WeightedPosterior,
                                        _reconstruct_original_hands)
from openhearts.engine import cards
from openhearts.engine.features import NF
from openhearts.engine.game import deal, legal_moves
from openhearts.engine.kernel import jit_enabled
from openhearts.engine.state import GameState
from openhearts.opponent.infer import load_profiler
from openhearts.opponent.npz_io import (N_CARDS, forward_numpy, load_npz,
                                        masked_probs_numpy, save_npz)
from openhearts.opponent.obsfeat import observer_features
from openhearts.opponent.params import PARAM_DIM, param_vector
from openhearts.players.heuristic import HeuristicPlayer
from openhearts.players.personality import (PersonalityPlayer,
                                            make_population,
                                            sample_personality)
from openhearts.search.profiled import (ProfilerLikelihood,
                                        profiler_posterior,
                                        profiler_posterior_factory,
                                        profiler_world_logweight)

H1, H2 = 24, 12


def _make_npz(tmp_path, name, n_param_in=0, seed=5):
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
    path = _make_npz(tmp_path_factory.mktemp("p5t4"), "generic.npz", 0, 5)
    weights, meta = load_profiler(path)
    return ProfilerLikelihood(weights, None, meta), path


@pytest.fixture(scope="module")
def population():
    train, heldout = make_population(200, 50, 314159)
    return train, heldout


def _play_game(seed, pids):
    """Play one 4-personality game; return (original hands, plays)."""
    players = [PersonalityPlayer(np.random.default_rng([seed, s, 0xA1CE]),
                                 sample_personality(p))
               for s, p in enumerate(pids)]
    state = deal(np.random.default_rng(seed))
    orig = list(state.hands)
    plays = []
    while not state.is_over():
        seat = state.to_play
        card = players[seat].choose(state.view_for(seat))
        plays.append((seat, card))
        state.play(card)
    assert sum(state.scores) == 26
    return orig, plays


def _state_at(orig, plays, n_plies):
    state = GameState(hands=list(orig))
    state.to_play = plays[0][0]
    for seat, card in plays[:n_plies]:
        assert seat == state.to_play
        state.play(card)
    return state


def _boundary(seed, pids, trick):
    """(state, view, true world hands in opponent_seats order) at a boundary."""
    orig, plays = _play_game(seed, pids)
    state = _state_at(orig, plays, 4 * (trick - 1))
    return orig, plays, state


# ------------------------------------------------------------- (a) reference
def _reference_logweight(view, world_hands, path, seat_params=None,
                         skip_forced=True):
    """Independent pure-Python audit: sum of log P over opponent decisions.

    Uses the DENSE untransposed reference forward pass. When
    `skip_forced=False` the single-legal plies are included as explicit
    factors, which is how the "skipping them is exact" claim is tested rather
    than asserted.
    """
    w, _meta = load_npz(path)
    hands, all_plays = _reconstruct_original_hands(view, world_hands)
    if not all_plays:
        return 0.0
    state = GameState(hands=hands)
    state.to_play = all_plays[0][0]
    total = 0.0
    for seat, card in all_plays:
        legal = legal_moves(state.hands[seat], tuple(state.current_trick),
                            state.hearts_broken, state.trick_number)
        if not (legal & cards.bit(card)):
            return -np.inf
        if seat != view.seat:
            n_legal = len(cards.cards_in(int(legal)))
            if n_legal > 1 or not skip_forced:
                x = observer_features(state, seat)
                if seat_params is not None:
                    x = np.concatenate([x, seat_params[seat]])
                logits = forward_numpy(w, x[None, :])
                p = masked_probs_numpy(logits,
                                       np.array([int(legal)], np.int64))
                total += float(np.log(p[0, card]))
        state.play(card)
    return total


@pytest.mark.parametrize("trick", [3, 7, 12])
def test_logweight_equals_product_of_profiler_probs(generic, population,
                                                    trick):
    """(a) hand-check: the world's weight IS the product of model probs."""
    lik, path = generic
    _train, heldout = population
    pids = list(heldout[:4])
    orig, plays, state = _boundary(96001, pids, trick)
    for observer in range(4):
        view = state.view_for(observer)
        opp = [(observer + 1 + i) % 4 for i in range(3)]
        world = [int(state.hands[s]) for s in opp]
        got = profiler_world_logweight(view, world, lik)
        want = _reference_logweight(view, world, path)
        assert np.isfinite(got) and np.isfinite(want)
        assert abs(got - want) <= 1e-9 * max(1.0, abs(want)), (
            f"observer {observer} trick {trick}: {got} != {want}")


def test_forced_ply_skip_is_exactly_a_no_op(generic, population):
    """Contract 1: single-legal plies contribute exactly 1.0 (log 0)."""
    lik, path = generic
    _train, heldout = population
    pids = list(heldout[4:8])
    _orig, _plays, state = _boundary(96002, pids, 11)
    view = state.view_for(0)
    opp = [(0 + 1 + i) % 4 for i in range(3)]
    world = [int(state.hands[s]) for s in opp]
    skipped = profiler_world_logweight(view, world, lik)
    included = profiler_world_logweight(view, world, lik, include_forced=True)
    ref_incl = _reference_logweight(view, world, path, skip_forced=False)
    assert abs(skipped - included) <= 1e-12
    assert abs(included - ref_incl) <= 1e-9 * max(1.0, abs(ref_incl))


def test_impossible_world_is_minus_inf(generic, population):
    lik, _path = generic
    _train, heldout = population
    pids = list(heldout[8:12])
    _orig, _plays, state = _boundary(96003, pids, 6)
    view = state.view_for(1)
    opp = [(1 + 1 + i) % 4 for i in range(3)]
    world = [int(state.hands[s]) for s in opp]
    # swap two cards between two opponents -> a DIFFERENT world; most such
    # worlds still replay legally, so search for one that does not.
    found = False
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b = 0, 1
        ca = rng.choice(cards.cards_in(world[a]))
        cb = rng.choice(cards.cards_in(world[b]))
        alt = list(world)
        alt[a] = (alt[a] & ~cards.bit(int(ca))) | cards.bit(int(cb))
        alt[b] = (alt[b] & ~cards.bit(int(cb))) | cards.bit(int(ca))
        if profiler_world_logweight(view, alt, lik) == -np.inf:
            found = True
            break
    assert found, ("no illegal-replay world found in 200 swaps -- the -inf "
                   "branch is then untested, which is itself a finding")


# ---------------------------------------------------------- (b) truth-safety
@pytest.mark.parametrize("trick", [2, 5, 9, 13])
def test_true_world_always_has_positive_weight(generic, population, trick):
    """(b) The truth is never excluded: finite log weight, positive prob."""
    lik, _path = generic
    _train, heldout = population
    for k, seed in enumerate((96010, 96011, 96012)):
        pids = list(heldout[k:k + 4])
        _orig, _plays, state = _boundary(seed, pids, trick)
        for observer in range(4):
            view = state.view_for(observer)
            opp = [(observer + 1 + i) % 4 for i in range(3)]
            world = [int(state.hands[s]) for s in opp]
            lw = profiler_world_logweight(view, world, lik)
            assert np.isfinite(lw), (
                f"TRUE world got log weight {lw} (seed {seed}, trick {trick}, "
                f"observer {observer}): truth-safety violated")
            assert np.exp(lw - lw) == 1.0  # relative weight of the truth > 0


def test_posterior_gives_truth_cards_positive_probability(generic,
                                                          population):
    """Truth-safety at the POSTERIOR level, not just the world level.

    Weaker claim than the world-level one on purpose: with 100 sampled worlds
    a particular true card can legitimately be absent from every world, so
    this asserts the posterior is well-formed, sums right, and puts positive
    mass on a healthy majority of truth cards -- and that it never RAISES.
    """
    lik, _path = generic
    _train, heldout = population
    _orig, _plays, state = _boundary(96020, list(heldout[:4]), 8)
    n_pos, n_tot = 0, 0
    for observer in range(4):
        view = state.view_for(observer)
        rng = np.random.default_rng([96020, 8, observer])
        post = profiler_posterior(view, Level.FULL, lik, 60, rng, 20000)
        assert post.n_worlds_used > 0 and post.n_effective > 0.0
        assert np.isfinite(post.probs).all() and (post.probs >= 0.0).all()
        for i, s in enumerate(post.opponent_seats):
            for c in cards.cards_in(int(state.hands[s])):
                n_tot += 1
                n_pos += post.probs[i, c] > 0.0
    assert n_pos / n_tot >= 0.5, f"only {n_pos}/{n_tot} truth cards had p>0"


def test_posterior_marginals_are_a_distribution_over_holders(generic,
                                                             population):
    lik, _path = generic
    _train, heldout = population
    _orig, _plays, state = _boundary(96021, list(heldout[3:7]), 5)
    view = state.view_for(2)
    rng = np.random.default_rng(7)
    post = profiler_posterior(view, Level.FULL, lik, 50, rng, 20000)
    unseen = cards.cards_in(post.unseen_mask)
    for c in unseen:
        assert abs(float(post.probs[:, c].sum()) - 1.0) <= 1e-9
    seen = [c for c in range(52) if c not in set(unseen)]
    assert np.all(post.probs[:, seen] == 0.0)
    assert isinstance(post, WeightedPosterior)


# --------------------------------------------------------------- ORACLE path
def test_conditioned_variant_runs_and_differs(tmp_path, population):
    """The ORACLE wiring: NF+PARAM_DIM input, true params per seat.

    Also pins that params actually REACH the network -- two different param
    assignments must give different log weights, otherwise the concatenation
    could be silently dropped.
    """
    path = _make_npz(tmp_path, "cond.npz", PARAM_DIM, seed=9)
    weights, meta = load_profiler(path)
    _train, heldout = population
    pids = list(heldout[:4])
    _orig, _plays, state = _boundary(96030, pids, 7)
    view = state.view_for(0)
    opp = [(1 + i) % 4 for i in range(3)]
    world = [int(state.hands[s]) for s in opp]

    sp = {s: param_vector(p) for s, p in enumerate(pids)}
    lik = ProfilerLikelihood(weights, sp, meta)
    got = profiler_world_logweight(view, world, lik)
    want = _reference_logweight(view, world, path, seat_params=sp)
    assert np.isfinite(got)
    assert abs(got - want) <= 1e-9 * max(1.0, abs(want))

    other = {s: param_vector(p) for s, p in enumerate(heldout[10:14])}
    lik2 = ProfilerLikelihood(weights, other, meta)
    assert profiler_world_logweight(view, world, lik2) != got

    with pytest.raises(AssertionError):
        ProfilerLikelihood(weights, None, meta)  # width mismatch must shout


def test_factory_returns_posteriors(generic, population):
    lik, _path = generic
    _train, heldout = population
    _orig, _plays, state = _boundary(96040, list(heldout[:4]), 4)
    make = profiler_posterior_factory(lik, level=Level.FULL, n_worlds=40,
                                      max_draws=20000)
    post = make(state.view_for(3), np.random.default_rng(3))
    assert isinstance(post, WeightedPosterior)
    assert post.n_worlds_used > 0
    assert make.likelihood is lik


# ------------------------------------------------- (d) JIT / NO_JIT identity
def test_jit_and_nojit_logweights_agree(generic, population):
    """(d) The audit's value must not depend on `OPENHEARTS_NO_JIT`.

    Run in-process against the reference, which is numpy-only and therefore
    identical in both modes; the suite is run under both env settings, so this
    test passing in both is exactly the two-mode pin. `jit_enabled()` is
    recorded in the failure message so a one-mode failure is unambiguous.
    """
    lik, path = generic
    _train, heldout = population
    _orig, _plays, state = _boundary(96050, list(heldout[2:6]), 10)
    for observer in range(4):
        view = state.view_for(observer)
        opp = [(observer + 1 + i) % 4 for i in range(3)]
        world = [int(state.hands[s]) for s in opp]
        got = profiler_world_logweight(view, world, lik)
        want = _reference_logweight(view, world, path)
        assert abs(got - want) <= 1e-9 * max(1.0, abs(want)), (
            f"jit_enabled={jit_enabled()} observer={observer}")


def test_heuristic_policy_untouched_by_this_module():
    """(c) guard: importing this module must not perturb the old socket.

    The real default-path proof is `tests/test_weighted_belief.py` and
    friends running green (they do, both modes); this is the cheap in-file
    canary that `weighted.world_weight`'s heuristic dispatch still behaves
    after `profiled` has been imported.
    """
    from openhearts.belief import weighted
    state = deal(np.random.default_rng(96060))
    pol = HeuristicPlayer()
    hands = list(state.hands)
    st = GameState(hands=hands)
    st.to_play = next(s for s in range(4)
                      if hands[s] & cards.bit(cards.TWO_CLUBS))
    for _ in range(4):
        st.play(pol.choose(st.view_for(st.to_play)))
    view = st.view_for(0)
    opp = [1, 2, 3]
    world = [int(st.hands[s]) for s in opp]
    assert weighted.world_weight(view, world, pol, 0.0) == 1.0


def test_oracle_params_are_keyed_to_the_right_seat(population):
    """The ORACLE seat mapping is EVIDENCE, not inspection.

    A systematic off-by-one in `seat_params` would pass
    `test_conditioned_variant_runs_and_differs` (params still reach the net,
    still change the answer) while silently UNDERSTATING the reading ceiling
    -- the number Phase 6's two-sided gate turns on. So: on TRUE worlds, the
    correctly-keyed mapping must explain the observed plays better, on
    average, than the same params ROTATED by one seat.

    This is the ONE test that needs the REAL trained CONDITIONED net (skipped
    if training has not been run): with random weights the claim is
    meaningless -- there is nothing for the params to mean -- so a passing
    random-weight version would be luck, not evidence.
    """
    cond = os.path.join("results", "profiler_train",
                        "profiler_conditioned.npz")
    if not os.path.exists(cond):
        pytest.skip(f"{cond} not present (Task 3 training has not been run)")
    weights, meta = load_profiler(cond)
    _train, heldout = population
    wins, total = 0, 0
    for k, seed in enumerate((96070, 96071, 96072, 96073)):
        pids = list(heldout[k:k + 4])
        _orig, _plays, state = _boundary(seed, pids, 10)
        right = {s: param_vector(p) for s, p in enumerate(pids)}
        rot = {s: param_vector(pids[(s + 1) % 4]) for s in range(4)}
        for observer in range(4):
            view = state.view_for(observer)
            opp = [(observer + 1 + i) % 4 for i in range(3)]
            world = [int(state.hands[s]) for s in opp]
            a = profiler_world_logweight(
                view, world, ProfilerLikelihood(weights, right, meta))
            b = profiler_world_logweight(
                view, world, ProfilerLikelihood(weights, rot, meta))
            assert a != b, "rotating the seat keying changed nothing"
            wins += a > b
            total += 1
    assert wins > total / 2, (
        f"the TRUE seat keying explained the observed plays better in only "
        f"{wins}/{total} boundary-observers -- params may not be following "
        f"their seat")
