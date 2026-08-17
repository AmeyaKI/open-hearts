"""Personality-parameter VECTORIZATION for the CONDITIONED profiler variant.

Torch-free (it is imported by the inference path and by tests). Lives beside
`npz_io.py` for the same reason: `model.py` imports torch, and nothing on the
play path may.

THE CONTRACT (PROFILER_V=1, frozen alongside it)
------------------------------------------------
`param_vector(pid)` maps a personality/anchor id to a `float64[PARAM_DIM]`
appended AFTER the 333 PROFILER_FEATURES_V=1 features, so the CONDITIONED
network's input width is `NF + PARAM_DIM`.

Layout, in `PersonalityParams`'s FROZEN DRAW ORDER (`personality.py` states
that order is append-only, and `PARAM_FIELDS` is derived from the dataclass,
so this vector inherits the same guarantee):

    idx  field         source range              normalization
    ---  -----------   -----------------------   ----------------------------
      0  duck          Normal(0.8, 1.1)          (x - 0.8) / 3.3   (~3 sigma)
      1  hoard         Uniform(-1.5, 1.5)        x / 1.5
    2-4  q_posture     {hunt, dump_early, avoid} ONE-HOT, in Q_POSTURES order
      5  q_strength    Uniform(0.5, 3.0)         (x - 0.5) / 2.5
      6  lead_short    Uniform(-0.5, 2.0)        (x + 0.5) / 2.5
      7  lead_long     Uniform(-0.5, 2.0)        (x + 0.5) / 2.5
      8  lead_heart    Uniform(-1.0, 2.0)        (x + 1.0) / 3.0
      9  heart_dump    Uniform(-1.0, 2.5)        (x + 1.0) / 3.5
  10-13  suit_quirk    Normal(0, 0.35) x4        x / 1.05          (~3 sigma)
     14  danger        Normal(0.7, 1.0)          (x - 0.7) / 3.0   (~3 sigma)
     15  temperature   Uniform(0.35, 1.30)       (x - 0.35) / 0.95
     16  epsilon       Uniform(0.03, 0.25)       (x - 0.03) / 0.22
  17-19  anchor flag   ANCHOR one-hot            see below

    PARAM_DIM = 20

Every axis is put on roughly [0, 1] (unbounded Gaussians on roughly
[-0.5, 0.5]) so no single parameter dominates the first layer's scale, which
matters because these dims sit next to 333 binary/0-1 features.

WHY NORMALIZE AT ALL, given an MLP could rescale?  It could, but the first
layer here is trained with a single shared learning rate and the feature half
of the input is 0-1 bounded; leaving `q_strength` on 0.5..3.0 and
`suit_quirk` on ~+-1 would give the two halves order-of-magnitude different
gradient scales for no reason.

ANCHORS (`ANCHOR_IDS`, negative ids: HeuristicPlayer, RandomizedHeuristic
eps=0.1, RandomPlayer).  Anchors have NO `PersonalityParams` at all, so idx
0..16 are set to ZERO and the identity is carried entirely by the 3-dim
one-hot at idx 17..19 (order: heuristic, randomized_heuristic, random).  A
personality's one-hot is all zeros, so "is an anchor" is linearly readable.

Why pseudo-params rather than dropping anchor rows from CONDITIONED training:
the headline number this task reports is CONDITIONED minus GENERIC, and that
difference only means "knowing the personality helps" if BOTH variants see
exactly the same rows.  Excluding anchors would confound it with a
training-set change.  Anchors are ~1.5% of acting-seat rows (3 of 203 pool
members), so the encoding barely matters -- identical row sets do.

The anchors' CONVENTION epsilons (0.0 / 0.1 / 1.0, from the generator) are
deliberately NOT written into idx 16: that slot's normalization is calibrated
to the personality range 0.03..0.25, and 1.0 would map to +3.4, an outlier in
a column whose other entries live in [0, 1].  The one-hot already identifies
each anchor uniquely, epsilon included.

PHASE 6 (Task A2) — THE PROFILER_V2 PARAMETER VECTOR
----------------------------------------------------
`param_vector_v2(pid, habit)` is a SECOND, INDEPENDENT contract used only by
the Phase-6 v2 profiler (`results/profiler_train_v2/`). It exists because the
v2 population differs from v1 in two ways the v1 vector structurally cannot
express:

  * three appended personality axes (`safe_dump`, `void_engineer`,
    `feed_leader`), and
  * epsilon/temperature that have been HABIT-TRANSFORMED, so the same
    personality id means four different levels of predictability.  The
    CONDITIONED v2 net is told the values ACTUALLY IN EFFECT (post-transform),
    which is the whole point of the curve: the ORACLE row's advantage at H0
    must include "this opponent is nearly deterministic".

    PARAM_DIM_V2 = PARAM_DIM + 3 = 23

The v1 20-slot layout is kept as an exact structural PREFIX (same slots in the
same order, anchor one-hot included and always zero — v2 tables are
personalities only, never anchors), with the three new axes appended.  The
DIMENSION DELIBERATELY DIFFERS from v1's: a v1 CONDITIONED net fed v2 vectors
would then fail on shape rather than silently consuming a wrong-meaning input.
That shape mismatch IS the tripwire; do not "fix" it by padding to 20.

Two normalizations differ from v1's, because the ranges differ:

    temperature  union of all four habit bands, 0.10 .. 2.50  -> (x-0.10)/2.40
    epsilon      union of all four habit bands, 0.005 .. 0.45 -> (x-0.005)/0.445

`_SCALAR_NORM` itself is NOT mutated: `param_vector` still feeds the frozen
`models/profiler_v1.npz`, and changing its scaling would silently invalidate
every Phase-5 number.
"""
import numpy as np

from ..players.personality import (ANCHOR_IDS, HABIT_SETTINGS, NEW_AXES_V2,
                                   PARAM_FIELDS, Q_POSTURES,
                                   sample_personality, sample_personality_v2)

N_ANCHOR = len(ANCHOR_IDS)
# scalar axes, in dataclass/draw order, with their (offset, scale):
# normalized = (x - offset) / scale
_SCALAR_NORM = {
    "duck": (0.8, 3.3),
    "hoard": (0.0, 1.5),
    "q_strength": (0.5, 2.5),
    "lead_short": (-0.5, 2.5),
    "lead_long": (-0.5, 2.5),
    "lead_heart": (-1.0, 3.0),
    "heart_dump": (-1.0, 3.5),
    "danger": (0.7, 3.0),
    "temperature": (0.35, 0.95),
    "epsilon": (0.03, 0.22),
}
_QUIRK_SCALE = 1.05
# Derived, not hand-counted, so a future APPENDED axis cannot silently desync
# the documented layout from the dataclass: one slot per scalar field, 3 for
# the q_posture one-hot, 4 for suit_quirk, plus the anchor one-hot.
_N_PERSONALITY = (len(_SCALAR_NORM) + len(Q_POSTURES) + 4)
PARAM_DIM = _N_PERSONALITY + N_ANCHOR
# fixed anchor order for the one-hot (sorted by id: -3, -2, -1 would be
# arbitrary; use the declaration order of ANCHOR_IDS, which is frozen there).
ANCHOR_ORDER = tuple(ANCHOR_IDS.keys())
_ANCHOR_SLOT = {ANCHOR_IDS[name]: i for i, name in enumerate(ANCHOR_ORDER)}

# The PROFILER_V=1 field set, PINNED.  `models/profiler_v1.npz` was trained
# with an input width of NF + PARAM_DIM = NF + 20; appending a personality axis
# (Phase 6's Task A1 appends three) must NOT silently change that width or the
# trained weights stop matching their own input.  So the tripwire is a PREFIX
# check, not an equality: v1's fields must remain the first fields of
# `PersonalityParams`, in order, and anything appended after them is invisible
# to PROFILER_V=1 by design.  A future PROFILER_V=2 that wants the new axes
# must extend this list deliberately and retrain.
_V1_FIELDS = ("duck", "hoard", "q_posture", "q_strength", "lead_short",
              "lead_long", "lead_heart", "heart_dump", "suit_quirk", "danger",
              "temperature", "epsilon")
assert PARAM_FIELDS[:len(_V1_FIELDS)] == _V1_FIELDS, (
    "PersonalityParams' frozen draw order changed: PROFILER_V=1's parameter "
    "vector is no longer a prefix of it")
assert set(_SCALAR_NORM) | {"q_posture", "suit_quirk"} == set(_V1_FIELDS), (
    "params.py's layout is out of sync with the pinned PROFILER_V=1 fields")


def param_vector(pid: int) -> np.ndarray:
    """float64[PARAM_DIM] for personality/anchor id `pid`. See module docs."""
    v = np.zeros(PARAM_DIM, dtype=np.float64)
    pid = int(pid)
    if pid in _ANCHOR_SLOT:
        v[_N_PERSONALITY + _ANCHOR_SLOT[pid]] = 1.0
        return v
    p = sample_personality(pid)
    i = 0
    for name in _V1_FIELDS:
        if name == "q_posture":
            v[i + Q_POSTURES.index(p.q_posture)] = 1.0
            i += len(Q_POSTURES)
        elif name == "suit_quirk":
            for k, x in enumerate(p.suit_quirk):
                v[i + k] = float(x) / _QUIRK_SCALE
            i += 4
        else:
            off, scale = _SCALAR_NORM[name]
            v[i] = (float(getattr(p, name)) - off) / scale
            i += 1
    assert i == _N_PERSONALITY
    return v


def param_table(pids) -> dict:
    """{pid: float64[PARAM_DIM]} built ONCE for a pool of ids.

    Millions of shard rows carry only ~200 distinct ids; re-deriving a
    personality from its seed per row would dominate featurization cost.
    """
    return {int(p): param_vector(p) for p in pids}


# ===========================================================================
# PROFILER_V2 (Phase 6 Task A2).  See the module docstring's PHASE 6 section.
# ===========================================================================
PARAM_DIM_V2 = PARAM_DIM + len(NEW_AXES_V2)

# The union of every habit band, so one normalization covers all four dial
# settings and an H0 epsilon is not an outlier in a column calibrated to H2.
_V2_TEMP_RANGE = (min(b["temperature"][0] for b in HABIT_SETTINGS.values()),
                  max(b["temperature"][1] for b in HABIT_SETTINGS.values()))
_V2_EPS_RANGE = (min(b["epsilon"][0] for b in HABIT_SETTINGS.values()),
                 max(b["epsilon"][1] for b in HABIT_SETTINGS.values()))
_SCALAR_NORM_V2 = dict(
    _SCALAR_NORM,
    temperature=(_V2_TEMP_RANGE[0], _V2_TEMP_RANGE[1] - _V2_TEMP_RANGE[0]),
    epsilon=(_V2_EPS_RANGE[0], _V2_EPS_RANGE[1] - _V2_EPS_RANGE[0]),
)
# Appended axes, in `NEW_AXES_V2` order, with their (offset, scale) — the
# same "~[0,1], Gaussians on ~[-0.5,0.5]" rule the v1 layout uses.
#   safe_dump      Uniform(-1.2, 2.6)   -> (x + 1.2) / 3.8
#   void_engineer  Normal(0.4, 1.3)     -> (x - 0.4) / 3.9   (~3 sigma)
#   feed_leader    Uniform(-1.5, 1.5)   -> x / 3.0
_NEW_AXIS_NORM = {"safe_dump": (-1.2, 3.8), "void_engineer": (0.4, 3.9),
                  "feed_leader": (0.0, 3.0)}
assert tuple(_NEW_AXIS_NORM) == NEW_AXES_V2, (
    "params.py's v2 axis order is out of sync with personality.NEW_AXES_V2")
assert PARAM_FIELDS[len(_V1_FIELDS):] == NEW_AXES_V2, (
    "PersonalityParams gained an axis PROFILER_V2 does not vectorize")


def param_vector_v2(pid: int, habit: str) -> np.ndarray:
    """float64[PARAM_DIM_V2] for a v2 personality at one habit setting.

    Slots 0..PARAM_DIM-1 mirror v1's layout exactly (the 3 anchor slots stay
    zero: v2 populations contain no anchors), with v2's epsilon/temperature
    normalization; slots PARAM_DIM.. carry the three appended axes.

    `habit` is REQUIRED and part of the key: the same id is a different player
    at each dial setting, and the vector reports the epsilon/temperature
    ACTUALLY IN EFFECT after the habit transform.
    """
    assert habit in HABIT_SETTINGS, f"unknown habit setting {habit!r}"
    assert int(pid) > 0, (
        f"anchor id {pid} has no v2 personality; v2 populations are "
        "personalities only")
    p = sample_personality_v2(int(pid), habit)
    v = np.zeros(PARAM_DIM_V2, dtype=np.float64)
    i = 0
    for name in _V1_FIELDS:
        if name == "q_posture":
            v[i + Q_POSTURES.index(p.q_posture)] = 1.0
            i += len(Q_POSTURES)
        elif name == "suit_quirk":
            for k, x in enumerate(p.suit_quirk):
                v[i + k] = float(x) / _QUIRK_SCALE
            i += 4
        else:
            off, scale = _SCALAR_NORM_V2[name]
            v[i] = (float(getattr(p, name)) - off) / scale
            i += 1
    assert i == _N_PERSONALITY
    i = PARAM_DIM                      # skip the (always-zero) anchor block
    for name in NEW_AXES_V2:
        off, scale = _NEW_AXIS_NORM[name]
        v[i] = (float(getattr(p, name)) - off) / scale
        i += 1
    assert i == PARAM_DIM_V2
    return v


def param_table_v2(pids, habit: str) -> dict:
    """{pid: float64[PARAM_DIM_V2]} for one pool at one habit setting."""
    return {int(p): param_vector_v2(p, habit) for p in pids}
