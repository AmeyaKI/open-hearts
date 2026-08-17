"""Phase 6 Task A2: the PROFILER_V2 parameter vector.

Two things are being protected here.

1. **Phase-5 reproducibility.**  `models/profiler_v1.npz` was trained with an
   input width of NF + 20 and a specific normalization per slot.  Adding a v2
   path must leave `param_vector` byte-identical and `PARAM_DIM` at 20.
2. **The v2 contract itself.**  The vector must carry the HABIT-TRANSFORMED
   epsilon/temperature (so the ORACLE row knows how predictable its opponent
   is) and the three appended axes, at a width that DIFFERS from v1's — the
   shape mismatch is the tripwire against feeding v2 vectors to the v1 net.
"""
import numpy as np
import pytest

from openhearts.opponent import params as P
from openhearts.players.personality import (HABIT_ORDER, NEW_AXES_V2,
                                            sample_personality,
                                            sample_personality_v2)

PIDS = [11, 12, 4242, 999_983]


def test_v1_vector_is_untouched():
    """The pinned v1 contract: width 20 and the exact v1 normalization."""
    assert P.PARAM_DIM == 20
    p = sample_personality(11)
    v = P.param_vector(11)
    assert v.shape == (20,)
    # spot-check two slots against the documented v1 normalization.
    assert v[0] == pytest.approx((p.duck - 0.8) / 3.3)
    assert v[16] == pytest.approx((p.epsilon - 0.03) / 0.22)
    # anchors still one-hot in the last three slots, personality slots zero.
    a = P.param_vector(-2)
    assert a[17 + 1] == 1.0 and a[:17].tolist() == [0.0] * 17


def test_v2_width_differs_from_v1():
    """A v1 CONDITIONED net must FAIL on shape, not silently accept v2."""
    assert P.PARAM_DIM_V2 == P.PARAM_DIM + len(NEW_AXES_V2) == 23
    assert P.PARAM_DIM_V2 != P.PARAM_DIM


@pytest.mark.parametrize("habit", HABIT_ORDER)
@pytest.mark.parametrize("pid", PIDS)
def test_v2_vector_shape_range_and_anchor_block(pid, habit):
    v = P.param_vector_v2(pid, habit)
    assert v.shape == (P.PARAM_DIM_V2,)
    assert np.isfinite(v).all()
    # the anchor one-hot block is structurally zero (v2 has no anchors).
    assert v[P._N_PERSONALITY:P.PARAM_DIM].tolist() == [0.0, 0.0, 0.0]
    # every normalized axis lands in a sane band: bounded draws on [0,1],
    # Gaussians on roughly [-0.5, 0.5]; nothing may dominate the first layer.
    assert v.min() > -1.5 and v.max() < 1.5


@pytest.mark.parametrize("pid", PIDS)
def test_v2_carries_the_habit_transformed_dials(pid):
    """THE point of the v2 vector: eps/temperature as ACTUALLY IN EFFECT."""
    i_temp = len(P._SCALAR_NORM) - 2 + 0  # position check done below instead
    # locate the slots by replaying the documented layout order.
    idx, i = {}, 0
    for name in P._V1_FIELDS:
        if name == "q_posture":
            i += 3
        elif name == "suit_quirk":
            i += 4
        else:
            idx[name] = i
            i += 1
    del i_temp
    seen_temp, seen_eps = [], []
    for habit in HABIT_ORDER:
        p = sample_personality_v2(pid, habit)
        v = P.param_vector_v2(pid, habit)
        off, sc = P._SCALAR_NORM_V2["temperature"]
        assert v[idx["temperature"]] == pytest.approx((p.temperature - off) / sc)
        off, sc = P._SCALAR_NORM_V2["epsilon"]
        assert v[idx["epsilon"]] == pytest.approx((p.epsilon - off) / sc)
        seen_temp.append(v[idx["temperature"]])
        seen_eps.append(v[idx["epsilon"]])
    # H0 -> H3 is strictly increasing in both dials, so the net can read
    # "how habitual is this opponent" straight off these two slots.
    assert seen_temp == sorted(seen_temp) and seen_temp[0] < seen_temp[-1]
    assert seen_eps == sorted(seen_eps) and seen_eps[0] < seen_eps[-1]


@pytest.mark.parametrize("pid", PIDS)
def test_v2_style_slots_are_invariant_across_the_dial(pid):
    """The dial moves ONLY predictability — the curve's whole premise.

    Every slot except epsilon/temperature must be bit-identical across the
    four settings; if a style slot moved, the curve would be confounding
    predictability with a change of playing style.
    """
    idx, i = {}, 0
    for name in P._V1_FIELDS:
        n = 3 if name == "q_posture" else (4 if name == "suit_quirk" else 1)
        idx[name] = list(range(i, i + n))
        i += n
    moving = set(idx["temperature"] + idx["epsilon"])
    static = [j for j in range(P.PARAM_DIM_V2) if j not in moving]
    ref = P.param_vector_v2(pid, "H2")
    for habit in HABIT_ORDER:
        v = P.param_vector_v2(pid, habit)
        assert np.array_equal(v[static], ref[static]), habit


@pytest.mark.parametrize("pid", PIDS)
def test_v2_new_axes_are_present_and_nonzero_somewhere(pid):
    v = P.param_vector_v2(pid, "H2")
    p = sample_personality_v2(pid, "H2")
    for k, name in enumerate(NEW_AXES_V2):
        off, sc = P._NEW_AXIS_NORM[name]
        assert v[P.PARAM_DIM + k] == pytest.approx(
            (getattr(p, name) - off) / sc)
    assert np.any(v[P.PARAM_DIM:] != 0.0)


def test_v2_refuses_anchors_and_unknown_settings():
    with pytest.raises(AssertionError):
        P.param_vector_v2(-1, "H0")
    with pytest.raises(AssertionError):
        P.param_vector_v2(11, "H9")


def test_param_table_v2():
    t = P.param_table_v2([11, 12], "H1")
    assert set(t) == {11, 12}
    assert all(v.shape == (P.PARAM_DIM_V2,) for v in t.values())
