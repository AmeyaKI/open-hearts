"""Expert population — parameter schema v1, sampling v1, seeds, split, wall.

WHY THIS MODULE EXISTS.  Phase 7A builds a gym of 250 rule-following
"experts" (EXPERT_RULEBOOK_DRAFT.md) whose STYLE varies while their shared
safety reflexes do not.  Everything about how an expert's record is produced
lives here so that the frozen contract is one file: the field order, the draw
order, the seed derivation, and the train/held-out split.  The policy itself
(what an expert does with its record) is `players/expert.py`.

FROZEN CONTRACTS (PHASE7_PLAN.md, 7A-1 spec §3, owner-approved 2026-09-05):

* `ExpertParams` field order == draw order.  New fields APPEND only; a new
  draw stream is a new schema version under a NEW master seed (Phase 6's v2
  precedent), never an edit of this one.
* `sample_expert(expert_id)` takes every draw on every path, so a branch that
  zeroes a weight never shifts the stream for later fields.
* `derive_seed` is SplitMix64 mixing over documented constants.  No Python
  `hash()` anywhere: string hashing is salted per process and integer-tuple
  hashing is an implementation detail, neither is a contract.
* ids 1..200 are TRAIN, 201..250 are HELD-OUT, split by index.  The record is
  a hash of the id, so the id carries no information about the record and an
  index split is as blind as a random one.

THE WALL.  The owner edits the rule catalogue and must never see a held-out
record.  Enforcement here is one assertion plus discipline: `dump_params`
refuses held-out ids, and `repr()` of a held-out record prints the id only.
Anyone can recompute a record from the master seed; the discipline is that
nobody prints one.  Held-out experts are still INSTANTIATED in-process by the
7B/7C harnesses -- seating them is the whole point -- they just never get
written or printed.
"""
from dataclasses import dataclass, fields, asdict

import numpy as np

# --------------------------------------------------------------- seeds

_MASK64 = (1 << 64) - 1
_SM_GAMMA = 0x9E3779B97F4A7C15
_SM_M1 = 0xBF58476D1CE4E5B9
_SM_M2 = 0x94D049BB133111EB

# Separate seed domains so a deal seed can never coincide with a population
# seed or a tie seed by accident (rulebook §3 "separate domains").
DOMAIN_POPULATION = 1
DOMAIN_DEALS = 2
DOMAIN_SEAT_ROTATION = 3
DOMAIN_POLICY_TIES = 4
DOMAIN_SEARCH = 5

MASTER_SEED_EXPERT_V1 = 7_070_707   # distinct from personality masters 314159 / 606060

N_TRAIN = 200
N_HELDOUT = 50


def derive_seed(domain: int, *ints: int) -> int:
    """Stable 64-bit seed from (domain, *ints): SplitMix64 folded over each
    integer in order.  Pure integer arithmetic, identical on every platform
    and Python version; pinned by tests/test_expert_population.py."""
    z = 0
    for x in (domain, *ints):
        z = (z + (int(x) & _MASK64) + _SM_GAMMA) & _MASK64
        z ^= z >> 30
        z = (z * _SM_M1) & _MASK64
        z ^= z >> 27
        z = (z * _SM_M2) & _MASK64
        z ^= z >> 31
    return z


# --------------------------------------------------------------- schema

TIE_MODES = ("fixed", "seeded_uniform")

OPTIONAL_WEIGHTS = (
    "w_Q1", "w_guard", "w_F4", "w_X1", "w_F5", "w_L2", "w_D2", "w_exit",
    "w_L4", "w_L5", "w_exposed", "w_disposal", "w_F1", "w_F1_spades",
    "w_F1_hearts", "w_F2", "w_H4", "w_H5", "w_L1",
)


@dataclass(frozen=True)
class ExpertParams:
    """One expert's record.  Field order below `expert_id` IS the draw order.

    Weights lie in [0, 1]; 0 disables that contribution ("enabled" means a
    positive weight, not a probability of obedience).  Thresholds are the
    rulebook's knobs with their allowed ranges.  See EXPERT_RULEBOOK_DRAFT.md
    §3 for what each one does.
    """
    expert_id: int
    tie_mode: str
    # queen posture (Q1)
    w_Q1: float
    queen_hunt_until: int      # 1..12
    hunt_min_spades: int       # 1..13
    # guard feature (Q2/Q4, one shared weight)
    w_guard: float
    guard_target: int          # 1..3
    # control posture (F4 / X1 / F5)
    w_F4: float
    w_X1: float
    w_F5: float
    # development vs retention (L2/X2, D2, D3/E3 shared)
    w_L2: float
    w_D2: float
    void_window: int           # 2..13
    w_exit: float
    exit_count: int            # 1..3
    # lead caution (L4, L5)
    w_L4: float
    w_L5: float
    void_lead_threshold: int   # 1..3
    # disposal (D1)
    guarded_spade_first: bool
    w_exposed: float
    w_disposal: float
    # following (F1 with suit-specific Q6/H3 weights, F2)
    w_F1: float
    w_F1_spades: float
    w_F1_hearts: float
    w_F2: float
    # hearts (H4, H5)
    w_H4: float
    ace_release_cost: int      # 1..4
    w_H5: float
    # opening (L1)
    w_L1: float
    # rank alarm (E4)
    escape_boost: float
    high_hand_mean: float      # 0..12
    small_rank: int            # 0..11

    def __repr__(self) -> str:
        if is_heldout(self.expert_id):
            return (f"ExpertParams(expert_id={self.expert_id}, "
                    "held-out: record withheld)")
        body = ", ".join(f"{f.name}={getattr(self, f.name)!r}"
                         for f in fields(self))
        return f"ExpertParams({body})"

    __str__ = __repr__


REFERENCE_PARAMS = ExpertParams(
    expert_id=0, tie_mode="fixed",
    w_Q1=1.0, queen_hunt_until=5, hunt_min_spades=3,
    w_guard=1.0, guard_target=1,
    w_F4=1.0, w_X1=1.0, w_F5=1.0,
    w_L2=1.0, w_D2=1.0, void_window=8, w_exit=1.0, exit_count=1,
    w_L4=1.0, w_L5=1.0, void_lead_threshold=2,
    guarded_spade_first=False, w_exposed=1.0, w_disposal=1.0,
    w_F1=1.0, w_F1_spades=1.0, w_F1_hearts=1.0, w_F2=1.0,
    w_H4=1.0, ace_release_cost=4, w_H5=1.0,
    w_L1=1.0,
    escape_boost=0.5, high_hand_mean=8.0, small_rank=4,
)
"""The rulebook's deterministic reference configuration: a TEST BASELINE and
the corpus-pin anchor.  Never a population member, never seated in 7B/7C."""


def all_optional_off(expert_id: int = -1) -> ExpertParams:
    """Every optional weight zero: the pure mandatory-stages + fallback
    player.  Used by the coverage gate; forbidden in the population."""
    kw = asdict(REFERENCE_PARAMS)
    for w in OPTIONAL_WEIGHTS:
        kw[w] = 0.0
    kw["escape_boost"] = 0.0
    kw["expert_id"] = expert_id
    return ExpertParams(**kw)


# --------------------------------------------------------------- split

def train_ids():
    return list(range(1, N_TRAIN + 1))


def heldout_ids():
    return list(range(N_TRAIN + 1, N_TRAIN + N_HELDOUT + 1))


def is_heldout(expert_id: int) -> bool:
    return N_TRAIN < int(expert_id) <= N_TRAIN + N_HELDOUT


def make_expert_population():
    """(train_ids, heldout_ids) -- the sealed 200/50 split, by index."""
    tr, he = train_ids(), heldout_ids()
    assert set(tr).isdisjoint(he) and len(tr) == N_TRAIN and len(he) == N_HELDOUT
    return tr, he


# --------------------------------------------------------------- sampling v1

def _u(rng, lo, hi):
    return float(lo + (hi - lo) * rng.random())


def _i(rng, lo, hi):
    """Integer in [lo, hi] inclusive."""
    return int(rng.integers(lo, hi + 1))


def sample_expert(expert_id: int) -> ExpertParams:
    """Sampling distribution v1 (frozen 2026-09-05).  Coherent axes: a
    'hunter' really hunts, a 'ducker' really ducks -- see the plan's §3.
    EVERY draw is taken on every path; branches only decide whether a drawn
    value is kept or zeroed, so the stream never shifts."""
    expert_id = int(expert_id)
    rng = np.random.default_rng(
        derive_seed(DOMAIN_POPULATION, MASTER_SEED_EXPERT_V1, expert_id))

    # 1. tie mode
    tie_mode = "seeded_uniform" if rng.random() < 0.5 else "fixed"
    # 2. queen posture
    u = rng.random()
    wq = _u(rng, 0.5, 1.0)
    w_Q1 = wq if u < 0.35 else 0.0
    queen_hunt_until = _i(rng, 4, 9)
    hunt_min_spades = _i(rng, 2, 4)
    # 3. guard
    u = rng.random()
    wg = _u(rng, 0.3, 1.0)
    w_guard = 0.0 if u < 0.3 else wg
    guard_target = _i(rng, 1, 3)
    # 4. control posture
    u = rng.random()
    a, b, c = rng.random(), rng.random(), rng.random()
    if u < 0.35:            # controller
        w_F4, w_X1, w_F5 = 0.5 + 0.5 * a, 0.5 + 0.5 * b, 0.4 * c
    elif u < 0.70:          # ducker
        w_F4, w_X1, w_F5 = 0.3 * a, 0.5 * b, 0.6 + 0.4 * c
    else:                   # neutral
        w_F4, w_X1, w_F5 = a, b, c
    # 5. development vs retention
    u = rng.random()
    a, b, c = rng.random(), rng.random(), rng.random()
    vw = rng.random()
    if u < 0.35:            # developer
        w_L2, w_D2, w_exit = 0.5 + 0.5 * a, 0.5 + 0.5 * b, 0.5 * c
        void_window = 6 + int(vw * 8)          # 6..13
    elif u < 0.70:          # retainer
        w_L2, w_D2, w_exit = 0.5 * a, 0.5 * b, 0.6 + 0.4 * c
        void_window = 2 + int(vw * 6)          # 2..7
    else:                   # neutral
        w_L2, w_D2, w_exit = a, b, c
        void_window = 2 + int(vw * 12)         # 2..13
    exit_count = _i(rng, 1, 3)
    # 6. lead caution
    u4, w4 = rng.random(), _u(rng, 0.2, 1.0)
    u5, w5 = rng.random(), _u(rng, 0.2, 1.0)
    w_L4 = 0.0 if u4 < 0.25 else w4
    w_L5 = 0.0 if u5 < 0.25 else w5
    void_lead_threshold = _i(rng, 1, 3)
    # 7. disposal
    guarded_spade_first = bool(rng.random() < 0.5)
    ue, we = rng.random(), _u(rng, 0.3, 1.0)
    w_exposed = 0.0 if ue < 0.2 else we
    w_disposal = _u(rng, 0.2, 1.0)
    # 8. following
    w_F1 = _u(rng, 0.5, 1.0)
    w_F1_spades = _u(rng, 0.3, 1.0)
    w_F1_hearts = _u(rng, 0.3, 1.0)
    w_F2 = _u(rng, 0.3, 1.0)
    # 9. hearts
    uh, wh = rng.random(), _u(rng, 0.3, 1.0)
    w_H4 = 0.0 if uh < 0.4 else wh
    ace_release_cost = _i(rng, 1, 4)
    u5h, w5h = rng.random(), _u(rng, 0.2, 1.0)
    w_H5 = 0.0 if u5h < 0.3 else w5h
    # 10. opening
    u1, w1 = rng.random(), _u(rng, 0.3, 1.0)
    w_L1 = 0.0 if u1 < 0.2 else w1
    # 11. rank alarm
    ub, wb = rng.random(), _u(rng, 0.2, 1.0)
    escape_boost = 0.0 if ub < 0.5 else wb
    high_hand_mean = _u(rng, 6.5, 9.5)
    small_rank = _i(rng, 2, 5)

    p = ExpertParams(
        expert_id=expert_id, tie_mode=tie_mode,
        w_Q1=w_Q1, queen_hunt_until=queen_hunt_until,
        hunt_min_spades=hunt_min_spades,
        w_guard=w_guard, guard_target=guard_target,
        w_F4=w_F4, w_X1=w_X1, w_F5=w_F5,
        w_L2=w_L2, w_D2=w_D2, void_window=void_window,
        w_exit=w_exit, exit_count=exit_count,
        w_L4=w_L4, w_L5=w_L5, void_lead_threshold=void_lead_threshold,
        guarded_spade_first=guarded_spade_first,
        w_exposed=w_exposed, w_disposal=w_disposal,
        w_F1=w_F1, w_F1_spades=w_F1_spades, w_F1_hearts=w_F1_hearts,
        w_F2=w_F2,
        w_H4=w_H4, ace_release_cost=ace_release_cost, w_H5=w_H5,
        w_L1=w_L1,
        escape_boost=escape_boost, high_hand_mean=high_hand_mean,
        small_rank=small_rank,
    )
    check_coherence(p)
    return p


def check_coherence(p: ExpertParams) -> None:
    """Generation-time checks (train and held-out alike).  They forbid
    degenerate records; they never edit one.  A failure is a generation bug
    to fix as a versioned change."""
    n_on = sum(1 for w in OPTIONAL_WEIGHTS if getattr(p, w) > 0)
    assert n_on >= 5, f"expert {p.expert_id}: only {n_on} optional weights on"
    assert p.tie_mode in TIE_MODES
    assert 1 <= p.queen_hunt_until <= 12 and 1 <= p.hunt_min_spades <= 13
    assert 1 <= p.guard_target <= 3 and 1 <= p.void_lead_threshold <= 3
    assert 2 <= p.void_window <= 13 and 1 <= p.exit_count <= 3
    assert 1 <= p.ace_release_cost <= 4 and 0 <= p.small_rank <= 11
    assert 0.0 <= p.high_hand_mean <= 12.0 and 0.0 <= p.escape_boost <= 1.0
    for w in OPTIONAL_WEIGHTS:
        assert 0.0 <= getattr(p, w) <= 1.0, w
    ref = asdict(REFERENCE_PARAMS); ref.pop("expert_id")
    mine = asdict(p); mine.pop("expert_id")
    assert mine != ref, "a population record equals the reference config"


def record_key(p: ExpertParams) -> tuple:
    """Hashable record without the id, for duplicate detection."""
    d = asdict(p)
    d.pop("expert_id")
    return tuple(sorted(d.items()))


def dump_params(ids, allow_heldout_dump: bool = False):
    """Parameter records as dicts, TRAIN ONLY.  The single assertion behind
    the held-out wall: no committed code passes `allow_heldout_dump=True`."""
    out = []
    for i in ids:
        if is_heldout(i) and not allow_heldout_dump:
            raise PermissionError(
                f"expert {i} is HELD-OUT: its record must never be dumped, "
                "printed, or written (rulebook §5, plan 7A-1 §3)")
        out.append(asdict(sample_expert(i)))
    return out
