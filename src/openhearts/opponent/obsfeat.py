"""PROFILER_FEATURES_V=1 featurization of ONE seat's observer-legal view.

Extracted VERBATIM from `experiments/gen_population_data.py` (Phase 5 Task 2)
by Task 4 so the training rows and the likelihood-socket replay are produced
by ONE function, not two copies that can drift. The generator now delegates to
this helper; its `--smoke` output is byte-identical across the refactor (that
is the evidence, not the argument).

The contract, restated from the generator's docstring: this reuses
`engine.features` UNCHANGED (FEATURES_V=1, NF=333) and passes a `hands` array
in which every seat except the acting one is ZERO. Because `OFF_HANDS` block
r=0 is always the calling seat's own hand (rotation puts it there), blocks
r=1..3 are STRUCTURALLY zero in every row -- the hidden information is never
in the array, so it cannot leak. `verify_hidden_hand_independence` in the
generator proves this by byte-identity under two differing hidden completions.

`observer_features(state, seat)` reads only fields a PlayerView also exposes
(own hand, completed history, current trick, hearts_broken, trick_number,
scores) plus `state.hands[seat]`, which IS that seat's own hand. It takes a
GameState because both callers already hold one (the generator plays forward;
the Task-4 audit replays a candidate world) -- no information beyond the
acting seat's own view is read out of it.
"""
import numpy as np

from ..engine import features


def observer_features(state, seat: int) -> np.ndarray:
    """float64[NF] observer-legal features for `seat` acting in `state`."""
    hands = np.zeros(4, dtype=np.int64)
    hands[seat] = state.hands[seat]  # only the acting seat's hand
    pm = 0
    for _s, c in state.history:
        pm |= 1 << c
    trick_cards = np.zeros(4, dtype=np.int64)
    trick_seats = np.zeros(4, dtype=np.int64)
    for i, (s, c) in enumerate(state.current_trick):
        trick_cards[i] = c
        trick_seats[i] = s
        pm |= 1 << c
    tl = len(state.current_trick)
    led_suit, win_seat = -1, -1
    if tl:
        led = int(trick_cards[0]) // 13
        wr = int(trick_cards[0]) % 13
        ws = int(trick_seats[0])
        for i in range(1, tl):
            c = int(trick_cards[i])
            if c // 13 == led and c % 13 > wr:
                wr, ws = c % 13, int(trick_seats[i])
        led_suit, win_seat = led, ws
    return features.featurize(
        hands, pm, trick_cards, trick_seats, tl, led_suit, win_seat,
        state.hearts_broken, state.trick_number,
        np.asarray(state.scores, dtype=np.int64), seat)
