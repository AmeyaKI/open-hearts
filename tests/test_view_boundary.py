import dataclasses
import numpy as np
from openhearts.engine.game import deal
from openhearts.engine.state import PlayerView


ALLOWED_FIELDS = {
    "seat", "hand", "history", "current_trick",
    "hearts_broken", "trick_number", "scores", "legal_moves",
}


def test_view_has_no_extra_fields():
    # The information boundary: the view type simply cannot carry hidden hands.
    names = {f.name for f in dataclasses.fields(PlayerView)}
    assert names == ALLOWED_FIELDS


def test_view_is_frozen_and_shows_only_own_cards():
    state = deal(np.random.default_rng(0))
    for seat in range(4):
        v = state.view_for(seat)
        assert v.hand == state.hands[seat]
        assert v.seat == seat
        try:
            v.hand = 0
            assert False, "view must be immutable"
        except dataclasses.FrozenInstanceError:
            pass
