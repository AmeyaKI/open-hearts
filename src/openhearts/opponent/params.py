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
"""
import numpy as np

from ..players.personality import (ANCHOR_IDS, PARAM_FIELDS, Q_POSTURES,
                                   sample_personality)

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
