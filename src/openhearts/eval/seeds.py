"""B0 (ROADMAP post-Phase-7 program, item 1): stable benchmark seat seeds.

WHY. Until 2026-09-26 the C-bench and xinxin harnesses seeded our bot's per-seat
rng from Python's built-in ``hash((config_id, deal_seed, seat))``. That hash is
salted per process (PEP 456), so two reruns of the same match draw different
dice unless PYTHONHASHSEED is pinned -- the banked rows are honest samples but
not replayable game for game (audit finding, 2026-08-30). This module replaces
it with the SplitMix64 mixer the expert population already uses
(``derive_seed``: documented constants, unit-pinned outputs, no ``hash()``).

CONTRACT. ``bench_seat_seed(harness, *cfg_ints, deal_seed, seat)`` is a pure
function of small integers and returns a 32-bit seed. Harness ids are fixed
here so two harnesses can never share a stream by accident:

    HARNESS_CBENCH = 1   experiments/run_cbench.py   (our seats AND OpenSpiel's
                         ISMCTS bot -- a C-bench game is fully replayable)
    HARNESS_XINXIN = 2   experiments/run_xinxin_match.py (our seats only;
                         xinxin's own generators are not controllable through
                         OpenSpiel's wrapper, so games there are NOT replayable
                         -- recorded in docs/BENCHMARK_REGISTRY.md)
    HARNESS_PERILUNE = 3 experiments/run_perilune_match.py and
                         run_perilune_belief.py (our seats; Perilune's raw net
                         is a deterministic argmax, so these games ARE fully
                         replayable)

Rows written under this derivation carry the header token ``seeds=stable-v1``;
the harnesses refuse to resume a partial without it, so no pre-B0 banked file
can ever be appended to by a stable-seed run.
"""
from openhearts.players.expert_population import DOMAIN_BENCH_SEAT, derive_seed

HARNESS_CBENCH = 1
HARNESS_XINXIN = 2
HARNESS_PERILUNE = 3
SEEDS_TOKEN = "seeds=stable-v1"


def bench_seat_seed(harness: int, *ints: int) -> int:
    """32-bit seat seed; ``ints`` = (config ints..., deal_seed, seat)."""
    return derive_seed(DOMAIN_BENCH_SEAT, int(harness), *(int(x) for x in ints)) & 0xFFFFFFFF
