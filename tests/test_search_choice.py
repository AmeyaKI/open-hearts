"""Honest search driven by a choice-aware posterior (Task 9, item 1).

The pipeline under test: WeightedPosterior (choice evidence) -> honest
search's OUTER worlds -> points. Three properties are pinned here:

1. it plays complete legal games;
2. `posterior_factory=None` changes nothing (the Task-3 reduction test's
    seed comparison still holds bitwise);
3. when the posterior collapses (no candidate world survives choice
    filtering) the player counts it and falls back to the plain table
    sampler for that one decision, rather than dying mid-game.
"""
import numpy as np
from openhearts.belief.table import Level
from openhearts.belief.weighted import PosteriorCollapse, WeightedPosterior
from openhearts.engine.game import deal, play_game
from openhearts.players.heuristic import HeuristicPlayer
from openhearts.search.decision import SearchPlayer
from openhearts.search.honest import HonestSearchPlayer


def _factory(n_worlds=12, max_draws=20000, epsilon=0.0):
    def make(view, rng):
        return WeightedPosterior.from_view(
            view, Level.FULL, HeuristicPlayer(), epsilon,
            n_worlds, rng=rng, max_draws=max_draws, keep_worlds=True)
    return make


def test_choice_player_completes_games_legally():
    rng = np.random.default_rng(3)
    bot = HonestSearchPlayer(Level.FULL, n_outer=8, n_inner=3, rng=rng,
                             posterior_factory=_factory(n_worlds=8))
    state = play_game(deal(np.random.default_rng(11)),
                      [bot] + [HeuristicPlayer() for _ in range(3)])
    assert sum(state.scores) == 26
    assert bot.posterior_worlds > 0


def test_no_factory_is_bitwise_unchanged():
    # Same comparison as tests/test_honest_search.py's reduction test: with
    # posterior_factory left at its default the player must still draw the
    # Phase-1 rng stream exactly.
    state = deal(np.random.default_rng(9))
    for _ in range(8):
        state.play(HeuristicPlayer().choose(state.view_for(state.to_play)))
    view = state.view_for(state.to_play)
    a = HonestSearchPlayer(Level.FULL, n_outer=30, n_inner=0,
                           rng=np.random.default_rng(7), jit_sampler=False,
                           posterior_factory=None).choose(view)
    b = SearchPlayer(Level.FULL, 30, np.random.default_rng(7)).choose(view)
    assert a == b


def test_collapse_falls_back_and_counts():
    # max_draws=1 at epsilon=0: past the opening plies (where an empty play
    # history makes every world weight 1.0) a single draw almost never
    # survives choice filtering, so the collapse path is exercised many
    # times in one game.
    rng = np.random.default_rng(5)
    bot = HonestSearchPlayer(Level.FULL, n_outer=8, n_inner=3, rng=rng,
                             posterior_factory=_factory(n_worlds=8,
                                                        max_draws=1))
    state = play_game(deal(np.random.default_rng(12)),
                      [bot] + [HeuristicPlayer() for _ in range(3)])
    assert sum(state.scores) == 26
    assert bot.posterior_collapses > 0


def test_collapse_fallback_uses_the_table_sampler():
    # A stub factory that always collapses must leave the player behaving
    # exactly like plain honest search on the same rng stream.
    def collapsing(view, rng):
        raise PosteriorCollapse("forced collapse")

    state = deal(np.random.default_rng(9))
    for _ in range(8):
        state.play(HeuristicPlayer().choose(state.view_for(state.to_play)))
    view = state.view_for(state.to_play)
    a = HonestSearchPlayer(Level.FULL, n_outer=30, n_inner=0,
                           rng=np.random.default_rng(7), jit_sampler=False,
                           posterior_factory=collapsing)
    b = HonestSearchPlayer(Level.FULL, n_outer=30, n_inner=0,
                           rng=np.random.default_rng(7), jit_sampler=False)
    assert a.choose(view) == b.choose(view)
    assert a.posterior_collapses == 1 and b.posterior_collapses == 0
