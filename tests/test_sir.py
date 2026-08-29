"""Phase 7 Task A1 gates: the SIR posterior (`search/sir.py`).

Scope: the world-set CONSTRUCTION only. The likelihood itself is the archived
profiled-ORACLE one and is already pinned by `tests/test_profiled_search.py`
and `tests/test_fused_audit.py`; nothing here re-tests it.

Weights are SYNTHESIZED (seeded random export, exactly as
`test_profiled_search.py` does and for the same reason: the trained nets are
the lead's artifacts and a suite that needs them is red for anyone who has not
trained). The GAMES are real, played by real `PersonalityPlayer`s, so the
views, legal masks and observed cards are the genuine article.

The four gates the plan names, and where they live:
  (i)   uniform-weights reduction -> `test_uniform_reduces_to_plain_bitwise`
        (real likelihood, opening view) and
        `test_uniform_reduces_to_plain_bitwise_midgame` (synthetic flat
        weights, mid-game view -- the harder version)
  (ii)  truth-safety              -> `test_truth_safety_*`
  (iii) THE TRAP                  -> `test_trap_oversample_first_is_the_point`
  (iv)  ESS math + counting alarm -> `test_kish_ess_known_values`,
        `test_cap_hit_counter_counts_its_own_firings`
"""
import numpy as np
import pytest

from openhearts.belief.table import BeliefTable, Level
from openhearts.belief.weighted import PosteriorCollapse
from openhearts.engine import cards
from openhearts.engine.game import deal
from openhearts.engine.state import GameState
from openhearts.opponent.infer import load_profiler
from openhearts.opponent.npz_io import N_CARDS, save_npz
from openhearts.engine.features import NF
from openhearts.players.personality import (PersonalityPlayer,
                                            make_population,
                                            sample_personality)
from openhearts.search.honest import HonestSearchPlayer
from openhearts.search.profiled import (NEG_INF, ProfilerLikelihood,
                                        profiler_posterior,
                                        profiler_world_logweight)
from openhearts.search.sir import (M_CAP, SIRRecorder, kish_ess,
                                   merge_payloads, sir_posterior,
                                   sir_posterior_factory,
                                   systematic_resample_indices)

H1, H2 = 24, 12
N = 50            # the world count the archived ORACLE row consumes


@pytest.fixture(scope="module")
def lik(tmp_path_factory):
    rng = np.random.default_rng(7)
    w = {"W1": rng.normal(0, 0.1, (H1, NF)), "b1": rng.normal(0, 0.1, H1),
         "W2": rng.normal(0, 0.2, (H2, H1)), "b2": rng.normal(0, 0.1, H2),
         "W3": rng.normal(0, 0.3, (N_CARDS, H2)),
         "b3": rng.normal(0, 0.1, N_CARDS)}
    path = str(tmp_path_factory.mktemp("p7a1") / "generic.npz")
    save_npz(path, w, [NF, H1, H2, N_CARDS], 0)
    weights, meta = load_profiler(path)
    return ProfilerLikelihood(weights, None, meta)


@pytest.fixture(scope="module")
def pids():
    train, _heldout = make_population(200, 50, 314159)
    return [int(train[i]) for i in range(4)]


def _play(seed, pids):
    players = [PersonalityPlayer(np.random.default_rng([seed, s, 0xA1CE]),
                                 sample_personality(p))
               for s, p in enumerate(pids)]
    state = deal(np.random.default_rng(seed))
    orig, plays = list(state.hands), []
    while not state.is_over():
        seat = state.to_play
        card = players[seat].choose(state.view_for(seat))
        plays.append((seat, card))
        state.play(card)
    return orig, plays


def _state_at(orig, plays, n_plies):
    state = GameState(hands=list(orig))
    state.to_play = plays[0][0]
    for seat, card in plays[:n_plies]:
        state.play(card)
    return state


def _view_at(seed, pids, trick, seat=None):
    """(view, true opponent world in opponent_seats order) at a boundary."""
    orig, plays = _play(seed, pids)
    state = _state_at(orig, plays, 4 * (trick - 1))
    seat = state.to_play if seat is None else seat
    view = state.view_for(seat)
    table = BeliefTable.from_view(view, Level.FULL)
    true_world = [int(state.hands[s]) for s in table.opponent_seats]
    return view, true_world, state, table


# --------------------------------------------------------------- (iv) ESS
def test_kish_ess_known_values():
    assert kish_ess([1.0, 1.0, 1.0, 1.0]) == pytest.approx(4.0)
    assert kish_ess([1.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert kish_ess([3.0, 1.0]) == pytest.approx(16.0 / 10.0)
    assert kish_ess([2.0, 2.0, 2.0]) == pytest.approx(3.0)   # scale-invariant
    assert kish_ess(np.full(97, 0.25)) == pytest.approx(97.0)
    # one world carrying essentially everything is the degenerate case the
    # whole task is about
    w = np.concatenate([[1.0], np.full(99, 1e-9)])
    assert kish_ess(w) == pytest.approx(1.0, abs=1e-6)


def test_kish_ess_matches_the_archived_rows_n_effective(lik, pids):
    """The SIR formula and the archived row's `n_effective` are one number."""
    view, _tw, _st, _tb = _view_at(4242, pids, 6)
    post = profiler_posterior(view, Level.FULL, lik, N,
                              np.random.default_rng(11), 50_000,
                              keep_worlds=True)
    assert kish_ess(post.weights) == pytest.approx(post.n_effective, rel=1e-12)


def test_systematic_resample_is_the_identity_under_uniform_weights():
    """The property the frozen bitwise gate rests on, tested on its own."""
    for seed in range(25):
        rng = np.random.default_rng(seed)
        idx = systematic_resample_indices(np.ones(N), N, rng)
        assert list(idx) == list(range(N)), f"seed {seed} broke the identity"


def test_systematic_resample_respects_weights():
    idx = systematic_resample_indices(np.array([0.0, 1.0, 0.0]), 10,
                                      np.random.default_rng(0))
    assert set(idx.tolist()) == {1}
    idx = systematic_resample_indices(np.array([1.0, 1.0]), 10,
                                      np.random.default_rng(0))
    assert sorted(idx.tolist()) == [0] * 5 + [1] * 5


# ---------------------------------------------- (i) the uniform reduction
def test_uniform_reduces_to_plain_bitwise(lik, pids):
    """GATE (i), real likelihood.

    At the opening lead nothing has been played, so `profiler_world_logweight`
    returns 0.0 for EVERY world -- genuinely uniform weights, produced by the
    real audit rather than by a stub. On the same rng state the SIR path must
    then return the plain draw itself, world for world, in order.
    """
    orig, plays = _play(909, pids)
    state = _state_at(orig, plays, 0)
    view = state.view_for(state.to_play)
    assert not view.history and not view.current_trick

    plain = profiler_posterior(view, Level.FULL, lik, N,
                               np.random.default_rng(3), 50_000,
                               keep_worlds=True)
    assert all(w == plain.weights[0] for w in plain.weights), (
        "the opening view was expected to give uniform weights")
    sir = sir_posterior(view, Level.FULL, lik, N, np.random.default_rng(3))
    assert [list(w) for w in sir.worlds] == [list(w) for w in plain.worlds]
    assert sir.weights == [1.0] * N
    assert sir.sir_m == N, "uniform weights must not trigger any oversampling"


def test_uniform_reduces_to_plain_bitwise_midgame(lik, pids):
    """GATE (i), the harder version: a mid-game view with flat weights.

    The synthetic likelihood is the ONLY way to hold the weights flat deep in
    a hand; every other ingredient (proposal table, sampler, rng stream,
    chunking) is the production one. Bitwise on the world list.
    """
    view, _tw, _st, _tb = _view_at(515, pids, 7)
    plain = profiler_posterior(view, Level.FULL, lik, N,
                               np.random.default_rng(5), 50_000,
                               keep_worlds=True)
    sir = sir_posterior(view, Level.FULL, lik, N, np.random.default_rng(5),
                        logweight_fn=lambda *_a: 0.0)
    assert [list(w) for w in sir.worlds] == [list(w) for w in plain.worlds]
    assert sir.sir_m == N


# ------------------------------------------------------------- (iii) TRAP
def _degenerate_logweight(alpha=33.0):
    """A deterministic, world-keyed heavy-tailed log weight.

    `w = exp(-alpha*u)` with u ~ U(0,1) has ESS/M ~ 2/alpha, so alpha = 33
    puts the plain-50 ESS at ~3 -- the published degeneracy (1-6 of 100),
    reproduced synthetically so the trap test reflects the real failure mode
    rather than a made-up one.
    """
    def f(_view, world_hands, _lik):
        h = 0
        for k, mul in enumerate((0x9E3779B1, 0x85EBCA77, 0xC2B2AE3D)):
            h ^= (int(world_hands[k]) * mul) & 0xFFFFFFFF
        return -alpha * ((h * 2654435761) % (2 ** 32)) / float(2 ** 32)
    return f


def test_trap_oversample_first_is_the_point(lik, pids):
    """GATE (iii). BOTH arms are asserted, so the test cannot pass by the
    synthetic scenario failing to be degenerate in the first place.

    Arm A (the trap): resample 50 worlds from the SAME 50 the archived row
    drew. Arm B (SIR): oversample first, then resample 50. If A and B come out
    equally diverse, the experiment measures nothing.
    """
    view, _tw, _st, _tb = _view_at(2024, pids, 5)
    lw = _degenerate_logweight()
    rec = SIRRecorder()

    # Arm A -- the trap, spelled out rather than described: the archived
    # row's own world set, resampled the archived row's own way.
    plain = profiler_posterior(view, Level.FULL, lik, N,
                               np.random.default_rng(8), 50_000,
                               keep_worlds=True)
    logs = np.array([lw(view, w, lik) for w in plain.worlds])
    w_plain = np.exp(logs - logs.max())
    ess_plain = kish_ess(w_plain)
    idx = np.random.default_rng(8).choice(N, size=N, p=w_plain / w_plain.sum())
    trap_distinct = len({tuple(plain.worlds[int(i)]) for i in idx})

    # The scenario must actually BE degenerate, or arm B proves nothing.
    assert 1.5 <= ess_plain <= 8.0, (
        f"synthetic scenario is not degenerate (plain-50 ESS {ess_plain:.2f}); "
        f"recalibrate alpha before reading this test as a pass")
    assert trap_distinct <= 10, (
        f"the trap arm produced {trap_distinct} distinct worlds -- it was "
        f"supposed to photocopy the dominant few")

    # Arm B -- oversample FIRST.
    sir = sir_posterior(view, Level.FULL, lik, N, np.random.default_rng(8),
                        recorder=rec, logweight_fn=lw)
    sir_distinct = len({tuple(w) for w in sir.worlds})

    assert sir.sir_m > N, "SIR did not oversample at all"
    assert sir.sir_ess_pool > ess_plain * 3, (
        f"pool ESS {sir.sir_ess_pool:.1f} barely beat the plain-50 ESS "
        f"{ess_plain:.1f}")
    assert sir_distinct >= 25, (
        f"SIR resampled only {sir_distinct} distinct worlds of {N}")
    assert sir_distinct >= 3 * trap_distinct, (
        f"SIR {sir_distinct} vs trap {trap_distinct}: oversampling first must "
        f"MATERIALLY beat resampling the original 50, or the experiment "
        f"cannot distinguish the two hypotheses")
    assert rec.n_decisions == 1 and rec.cap_hits in (0, 1)


# ------------------------------------------------------- (ii) truth safety
def test_truth_safety_true_world_never_gets_minus_inf(lik, pids):
    """The true world is never killed: its replay is legal by construction and
    the masked softmax is strictly positive on every legal card."""
    checks = 0
    for seed in (11, 12, 13):
        for trick in (2, 6, 10, 13):
            view, true_world, _st, _tb = _view_at(seed, pids, trick)
            lwt = profiler_world_logweight(view, true_world, lik)
            assert lwt != NEG_INF and np.isfinite(lwt), (
                f"the TRUE world was killed at seed {seed} trick {trick}")
            checks += 1
    assert checks == 12, "truth-safety check did not run as often as claimed"


def test_truth_safety_sir_pool_keeps_the_truth_positive(lik, pids):
    """End-to-end: every card's TRUE holder keeps positive probability in the
    SIR pool's marginals, at a depth where the sampler reaches the truth."""
    checks = 0
    for seed in (21, 22, 23):
        view, _tw, state, table = _view_at(seed, pids, 12)
        post = sir_posterior(view, Level.FULL, lik, N,
                             np.random.default_rng(seed))
        for c in cards.cards_in(table.unseen_mask):
            holder = next(s for s in range(4) if state.hands[s] & cards.bit(c))
            i = table.opponent_seats.index(holder)
            assert post.sir_pool_probs[i, c] > 0.0, (
                f"seed {seed}: true holder zeroed for card {c}")
            checks += 1
    assert checks > 0, "truth-safety check never ran"


def test_collapse_is_loud_not_silent(lik, pids):
    view, _tw, _st, _tb = _view_at(31, pids, 4)
    with pytest.raises(PosteriorCollapse):
        sir_posterior(view, Level.FULL, lik, N, np.random.default_rng(1),
                      logweight_fn=lambda *_a: NEG_INF)


# ---------------------------------------------------- (iv) counting alarms
def test_cap_hit_counter_counts_its_own_firings(lik, pids):
    """An alarm that can never fire is not an alarm (Session Lesson 8).

    Two runs over the SAME decisions: one whose ESS target is unreachable
    (every decision must be a cap hit) and one whose target is trivially met
    (no decision may be). Both must report the same decision COUNT.
    """
    views = [_view_at(s, pids, t)[0] for s in (41, 42) for t in (3, 8)]

    hot = SIRRecorder()
    for i, v in enumerate(views):
        sir_posterior(v, Level.FULL, lik, N, np.random.default_rng(i),
                      ess_target=1e9, m_cap=200, recorder=hot,
                      logweight_fn=_degenerate_logweight())
    assert hot.n_decisions == len(views)
    assert hot.cap_hits == len(views), "the cap alarm failed to fire"

    cold = SIRRecorder()
    for i, v in enumerate(views):
        sir_posterior(v, Level.FULL, lik, N, np.random.default_rng(i),
                      ess_target=1.0, recorder=cold)
    assert cold.n_decisions == len(views)
    assert cold.cap_hits == 0, "the cap alarm fired when it should not have"

    merged = merge_payloads([hot.payload(), cold.payload()])
    assert merged["n_decisions"] == 2 * len(views)
    assert merged["cap_hits"] == len(views)
    assert sum(r["n"] for r in merged["per_trick"].values()) == 2 * len(views)


def test_escalation_is_incremental_and_capped(lik, pids):
    """The pool doubles from N and never exceeds the cap; the audit bill is
    the FINAL M (worlds already weighted are kept, not redrawn)."""
    view, _tw, _st, _tb = _view_at(77, pids, 6)
    rec = SIRRecorder()
    post = sir_posterior(view, Level.FULL, lik, N, np.random.default_rng(2),
                         ess_target=1e9, m_cap=400, recorder=rec,
                         logweight_fn=_degenerate_logweight())
    assert post.sir_m == 400 <= M_CAP
    row = rec.per_trick[view.trick_number]
    assert row["stages"] == 4.0, "50 -> 100 -> 200 -> 400 is four stages"
    assert row["draws"] == 400.0, "growth must be incremental, not restarted"


# ------------------------------------------------- the socket into search
def test_sir_worlds_reach_search_unweighted(lik, pids):
    """The resampled set arrives at `honest.py` as an UNWEIGHTED world set --
    no second resampling, no code change in the shipped search."""
    view, _tw, _st, _tb = _view_at(55, pids, 5)
    rec = SIRRecorder()
    pf = sir_posterior_factory(lik, level=Level.FULL, n_worlds=N, recorder=rec)
    bot = HonestSearchPlayer(Level.FULL, N, 20, np.random.default_rng(4),
                             sampler_respects_voids=True, posterior_factory=pf)
    worlds = bot._posterior_worlds(view)
    assert len(worlds) == N
    assert bot.posterior_decisions == 1
    assert bot.posterior_worlds == N          # N handed over, NOT the pool M
    assert rec.n_decisions == 1
    assert rec.per_trick[view.trick_number]["m"] >= N


def test_shipped_default_bot_is_untouched(pids):
    """Gate (vi), the structural half: with no posterior factory the player
    takes the plain sampler path and never enters any SIR code."""
    view, _tw, _st, _tb = _view_at(66, pids, 4)
    bot = HonestSearchPlayer(Level.FULL, 8, 0, np.random.default_rng(1))
    assert bot.posterior_factory is None
    card = bot.choose(view)
    assert cards.bit(card) & view.legal_moves
    assert bot.posterior_decisions == 0 and bot.posterior_worlds == 0
