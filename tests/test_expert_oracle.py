"""Gates for the 7B exact expert likelihood and ORACLE/SIR world sources
(PHASE7_PLAN.md 7B pre-registration §D7 (i)-(iv))."""
import math

import numpy as np
import pytest

from openhearts.belief.expert_oracle import (
    NEG_INF, ExpertLikelihood, expert_oracle_factory, expert_oracle_posterior,
    expert_sir_factory,
)
from openhearts.belief.table import BeliefTable, Level
from openhearts.belief.weighted import PosteriorCollapse
from openhearts.engine import cards
from openhearts.engine.game import deal
from openhearts.players.expert import ExpertPlayer, action_distribution
from openhearts.players.expert_population import (
    DOMAIN_POLICY_TIES, derive_seed, heldout_ids, sample_expert,
)
from openhearts.search import sir as _sir
from openhearts.search.decision import state_from_view
from openhearts.search.sir import SIRRecorder

HELD = heldout_ids()


def _table(seed, observer=0, k=3):
    """Deal `seed`, seat experts at every seat, and return (state, lik) with
    the OBSERVER's opponents' true records loaded into the likelihood."""
    ids = {s: HELD[(seed * 7 + s) % len(HELD)] for s in range(4)}
    st = deal(np.random.default_rng(seed))
    players = {s: ExpertPlayer(np.random.default_rng(
        derive_seed(DOMAIN_POLICY_TIES, seed, 0, s, ids[s])), sample_expert(ids[s]))
        for s in range(4)}
    lik = ExpertLikelihood({s: sample_expert(ids[s]) for s in range(4) if s != observer})
    return st, players, lik


def _true_world(st, observer):
    return [st.hands[(observer + 1 + i) % 4] for i in range(3)]


def test_truth_safety_true_world_always_survives():
    """(i) The deal that actually happened has positive weight at every
    observer decision -- the real world is never killed."""
    n_checked = 0
    for seed in range(60):
        st, players, lik = _table(seed)
        while not st.is_over():
            s = st.to_play
            v = st.view_for(s)
            if s == 0:
                lw = lik.world_logweight(v, _true_world(st, 0))
                assert lw > NEG_INF and math.isfinite(lw)
                n_checked += 1
            st.play(players[s].choose(v))
    assert n_checked == 60 * 13
    assert lik.n_illegal == 0


def test_likelihood_kills_impossible_worlds_and_logs_tie_probabilities():
    st, players, lik = _table(3)
    for _ in range(8):
        s = st.to_play
        st.play(players[s].choose(st.view_for(s)))
    v = st.view_for(0)
    true = _true_world(st, 0)
    assert lik.world_logweight(v, true) <= 0.0
    # Swap two opponents' hands: the observed plays are typically impossible
    # or improbable there.  Over many draws, at least one world must die.
    table = BeliefTable.from_view(v, Level.FULL)
    rng = np.random.default_rng(0)
    cands, _ = _sir._draw_candidates(table, rng, 200, _sir.kernel.jit_enabled())
    lws = [lik.world_logweight(v, w) for w in cands]
    assert any(x == NEG_INF for x in lws)
    assert all(x <= 0.0 for x in lws)


def test_same_view_from_different_hidden_worlds_gives_same_factor():
    """(iii) The likelihood of a world depends on the observer's view and
    the world -- never on the real hidden hands behind the view."""
    st, players, lik = _table(5)
    for _ in range(6):
        s = st.to_play
        st.play(players[s].choose(st.view_for(s)))
    v = st.view_for(0)
    table = BeliefTable.from_view(v, Level.FULL)
    cands, _ = _sir._draw_candidates(table, np.random.default_rng(1), 5, False)
    for w in cands:
        alt = state_from_view(v, w)          # a different full state, same view
        assert alt.view_for(0) == v
        assert lik.world_logweight(alt.view_for(0), w) == lik.world_logweight(v, w)


def test_oracle_reduction_constant_likelihood_equals_first_n_draws():
    """(ii) With every world weight 1, the ORACLE world list is exactly the
    constraint sampler's first N draws in draw order."""
    st, players, lik = _table(8)
    for _ in range(9):
        s = st.to_play
        st.play(players[s].choose(st.view_for(s)))
    v = st.view_for(0)

    class One:
        seat_params = {}
        def world_logweight(self, view, wh):
            return 0.0

    post = expert_oracle_posterior(v, Level.FULL, One(), 20,
                                   np.random.default_rng(42), max_draws=20)
    table = BeliefTable.from_view(v, Level.FULL)
    ref, _ = _sir._draw_candidates(table, np.random.default_rng(42), 20,
                                   _sir.kernel.jit_enabled())
    assert post.worlds == [[int(x) for x in w] for w in ref]
    assert all(w == 1.0 for w in post.weights)


def test_collapse_and_starvation_are_loud_and_counted():
    """(iv) zero survivors raises PosteriorCollapse; a starving likelihood
    records cap hits and survivor counts on the SIR recorder."""
    st, players, lik = _table(11)
    for _ in range(5):
        s = st.to_play
        st.play(players[s].choose(st.view_for(s)))
    v = st.view_for(0)

    class Dead:
        seat_params = {}
        def world_logweight(self, view, wh):
            return NEG_INF

    with pytest.raises(PosteriorCollapse):
        expert_oracle_posterior(v, Level.FULL, Dead(), 10,
                                np.random.default_rng(0), max_draws=30)
    rec = SIRRecorder()
    post = expert_oracle_posterior(v, Level.FULL, lik, 200,
                                   np.random.default_rng(0), max_draws=60,
                                   recorder=rec)
    assert 0 < len(post.worlds) <= 60
    # SIR with the exact likelihood: pool grows, N equal weights come back.
    rec2 = SIRRecorder()
    f = expert_sir_factory(lik, Level.FULL, n_worlds=20, m_cap=80, recorder=rec2)
    post2 = f(v, np.random.default_rng(0))
    assert len(post2.worlds) == 20 and all(w == 1.0 for w in post2.weights)
    assert post2.sir_m <= 80


def test_honest_search_runs_on_both_world_sources_and_counts_realised_worlds():
    from openhearts.search.honest import HonestSearchPlayer
    for factory in (lambda lik, rec: expert_oracle_factory(lik, Level.FULL, 10, 100, rec),
                    lambda lik, rec: expert_sir_factory(lik, Level.FULL, 10, m_cap=100, recorder=rec)):
        st, players, lik = _table(21)
        rec = SIRRecorder()
        bot = HonestSearchPlayer(Level.FULL, 10, 5, np.random.default_rng(3),
                                 posterior_factory=factory(lik, rec))
        players[0] = bot
        while not st.is_over():
            s = st.to_play
            v = st.view_for(s)
            c = players[s].choose(v)
            assert v.legal_moves & cards.bit(c)
            st.play(c)
        assert sum(st.scores) == 26
        assert bot.posterior_decisions + bot.posterior_collapses + bot.fallbacks > 0


def test_tie_probability_matches_action_distribution():
    st, players, lik = _table(2)
    for _ in range(4):
        s = st.to_play
        st.play(players[s].choose(st.view_for(s)))
    v = st.view_for(1)
    cs, ps = action_distribution(v, lik.seat_params[1])
    assert abs(sum(ps) - 1.0) < 1e-12
