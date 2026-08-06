"""Rotated-seat matches: the fair way to compare two players at Hearts.

Hearts is luck-heavy. Three defenses live here and in `stats.py`:

1. Each deal is played 4 times with the bot rotated through every seat, so a
   deal that hands seat 2 a monstrous hand hurts the bot in exactly one of the
   four games and helps it in the others. Averaging the 4 rotations cancels
   most of the deal's luck.
2. The deal is regenerated with `np.random.default_rng(seed)` for every single
   game, so every rotation -- and every configuration compared later -- sees
   the *identical* 52 cards in the identical seats.
3. The returned array is one value per deal, which is what
   `stats.bootstrap_ci` must resample. The 4 rotations of one deal share that
   deal's luck, so they are one independent observation; resampling all
   4 x n_deals games as if independent would fake extra certainty.

Iteration order is part of the contract: seeds outer, rotations 0..3 inner,
and `bot_factory()` is called exactly once per game in that order (before the
opponent factories). Callers that need per-game determinism -- e.g. giving a
`SearchPlayer` an rng derived from (deal seed, rotation, config) -- rely on
that order to know which game they are constructing a player for.

Factories take no arguments and are called fresh for every game, so a stateful
player cannot leak information (or an advancing rng stream) between games. If
a player needs an rng, the factory closure owns it; the harness stays
agnostic about how players are built.
"""
import numpy as np

from openhearts.engine.game import deal, play_game


def rotated_match(deal_seeds, bot_factory, opp_factory, on_deal=None) -> np.ndarray:
    """Play every deal 4 times, rotating the bot through all seats.

    Returns an array of shape (n_deals,): the bot's points averaged over its
    4 rotations for that deal.

    `on_deal(deals_done, n_deals)`, if given, is called after each deal --
    used for progress reporting only; it must not affect play.
    """
    seeds = list(deal_seeds)
    out = np.zeros(len(seeds), dtype=float)
    for d, seed in enumerate(seeds):
        total = 0.0
        for rotation in range(4):
            state = deal(np.random.default_rng(seed))  # identical cards each rotation
            bot = bot_factory()
            players = [None, None, None, None]
            players[rotation] = bot
            for s in range(4):
                if players[s] is None:
                    players[s] = opp_factory()
            final = play_game(state, players)
            assert sum(final.scores) == 26, "engine invariant broken during match"
            total += final.scores[rotation]
        out[d] = total / 4.0
        if on_deal is not None:
            on_deal(d + 1, len(seeds))
    return out
