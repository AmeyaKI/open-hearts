import numpy as np
from openhearts.belief.table import Level
from openhearts.belief.weighted import WeightedPosterior, world_weight
from openhearts.engine import cards
from openhearts.engine.game import deal
from openhearts.players.heuristic import HeuristicPlayer


def _play_tricks(state, players, n_plies):
    for _ in range(n_plies):
        seat = state.to_play
        state.play(players[seat].choose(state.view_for(seat)))


def test_true_world_has_full_weight_when_deterministic():
    # The actual hidden hands must replay the observed game exactly,
    # so with epsilon=0 the true world's weight is exactly 1.
    rng = np.random.default_rng(11)
    state = deal(rng)
    players = [HeuristicPlayer() for _ in range(4)]
    observer = state.to_play
    _play_tricks(state, players, 20)
    view = state.view_for(observer)
    # DEVIATION FROM PLAN TEXT (deliberate, assertion unchanged): the plan
    # wrote `truth` in ascending seat order, but world_weight takes hands in
    # BeliefTable.opponent_seats order ((observer+1+i) % 4). Those coincide
    # only for observer 0 or 3; deal(rng(11)) has observer 1, so the plan's
    # version passes a permuted world and correctly gets weight 0.0.
    truth = [state.hands[(observer + 1 + i) % 4] for i in range(3)]
    w = world_weight(view, truth, HeuristicPlayer(), epsilon=0.0)
    assert w == 1.0


def test_posterior_sharper_than_constraints_alone():
    # After several tricks of heuristic play, choice evidence must place
    # strictly more probability on the truth than the FULL table does
    # (averaged over unseen cards).
    from openhearts.belief.table import BeliefTable
    rng = np.random.default_rng(12)
    state = deal(rng)
    players = [HeuristicPlayer() for _ in range(4)]
    observer = state.to_play
    _play_tricks(state, players, 32)
    view = state.view_for(observer)
    post = WeightedPosterior.from_view(view, Level.FULL, HeuristicPlayer(),
                                       epsilon=0.0, n_worlds=100,
                                       rng=np.random.default_rng(1),
                                       max_draws=20000)
    table = BeliefTable.from_view(view, Level.FULL)
    def mean_p_truth(probs):
        ps = []
        for c in cards.cards_in(table.unseen_mask):
            holder = next(s for s in range(4) if state.hands[s] & cards.bit(c))
            ps.append(probs[table.opponent_seats.index(holder), c])
        return float(np.mean(ps))
    assert mean_p_truth(post.probs) > mean_p_truth(table.probs) + 0.05


def test_epsilon_keeps_truth_alive_against_deviants():
    # Against a RANDOMIZED opponent, strict filtering may kill the true
    # world; epsilon>0 must always leave it positive weight.
    from openhearts.players.randomized import RandomizedHeuristic
    rng = np.random.default_rng(13)
    state = deal(rng)
    players = [RandomizedHeuristic(np.random.default_rng(s), 0.3)
               for s in range(4)]
    observer = state.to_play
    _play_tricks(state, players, 24)
    view = state.view_for(observer)
    # same opponent_seats ordering fix as above (here observer == 3, so the
    # plan's ascending order happened to coincide; made explicit anyway)
    truth = [state.hands[(observer + 1 + i) % 4] for i in range(3)]
    w = world_weight(view, truth, HeuristicPlayer(), epsilon=0.1)
    assert w > 0.0
