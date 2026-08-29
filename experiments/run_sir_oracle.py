"""Phase 7 Task A1: THE ESS / RESAMPLING TEST.

THE QUESTION, IN ONE SENTENCE.  Phase 6A retired opponent reading because the
profiled-ORACLE row -- which guesses the hidden cards measurably better than
anything else we own -- LOST to plain honest search by 0.202 pts/hand at H0
and 0.340 at H2.  This script asks whether that loss was caused by the
information being worthless, or by the way we delivered it.

WHY THERE IS A SECOND CANDIDATE EXPLANATION.  The ORACLE row imagines 50
arrangements of the hidden cards and weights each one by how well it explains
the opponents' actual plays.  Phase 5 limitation 9 measured what those weights
look like: as few as 1-6 of 100 worlds carry essentially all the weight.  So
"averaged over 50 worlds" was really "averaged over three worlds, photocopied"
-- and whichever card happens to look good in those three wins.  In the
particle-filtering literature that is *weight degeneracy*, the best-known
failure mode of importance sampling, and it has a standard remedy:
oversample, then resample to restore an equally-weighted, diverse set.
Phase 3 -- the ONLY time reading ever won (+0.385) -- is also the only
configuration whose surviving worlds stayed equally weighted and diverse.

WHAT THIS SCRIPT RUNS.  One new row, ORACLE-SIR: the archived ORACLE
machinery with EXACTLY ONE THING CHANGED -- how the world set is built.  Same
deals, same held-out personality trios, same rotations, same opponent
streams, same conditioned profiler, same true parameter vectors, same
likelihood function.  See `search/sir.py` for the construction and for THE
TRAP (resampling the original 50 worlds photocopies the dominant few and would
falsely "confirm" the retirement; oversampling FIRST is the entire point).

WHAT IT COMPARES AGAINST.  The banked 6A per-deal values for honest-FULL and
profiled-ORACLE, read out of `results/entropy_curve_{H0,H2}_partial.txt`.
Those files hold one JSON line per 25-deal match block with a `values` array
of per-deal scores, so the comparison is PAIRED DEAL BY DEAL -- the same card
luck, the same people, the same seats.  `--gate` proves the pairing is real
rather than assumed: it re-runs a banked ORACLE block from scratch and checks
the numbers come back bit for bit.

READ THE SIGNS CAREFULLY.  Points: LOWER IS BETTER (6.5 is symmetric
break-even).  "Reading value", the convention every 6A output uses, is
`honest-FULL - row`, so POSITIVE means the row BEATS honest-FULL.  Both
conventions are printed side by side because a sign error here would invert a
phase-defining verdict.

MODES
-----
  --gate                 offline gates; no long run.  Includes the banked
                         block reproduction check (--repro-block N).
  --smoke                plumbing only, own `_smoke` paths, tiny settings.
  --probe --habit H0     2 deals; prints s/game AND the achieved pool sizes
                         per trick (cost is driven by M, not by deals), then
                         projects the full run.
  --habit H0             THE RUN (lead only).
  --guessing --habit H0  optional: expectation (2)'s diagnostic panel.
  --report               merge + paired bootstraps + the pre-registered
                         decision rule, verbatim.

Usage:
  .venv/bin/python experiments/run_sir_oracle.py --gate
  .venv/bin/python experiments/run_sir_oracle.py --probe --habit H0
  .venv/bin/python experiments/run_sir_oracle.py --habit H0 --workers 8
  .venv/bin/python experiments/run_sir_oracle.py --report
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The 6A machinery.  EVERY seed derivation, trio draw and opponent stream is
# IMPORTED from here -- none of it is re-spelled in this file, because a
# re-spelling that drifted by one integer would silently unpair the
# comparison and nothing downstream would notice.
import run_entropy_curve as ec                                  # noqa: E402

from openhearts.belief.table import Level                        # noqa: E402
from openhearts.engine.game import deal, play_game               # noqa: E402
from openhearts.eval import blockdriver                          # noqa: E402
from openhearts.eval.stats import bootstrap_ci                   # noqa: E402
from openhearts.opponent.params import (PARAM_DIM_V2,            # noqa: E402
                                        param_vector_v2)
from openhearts.search.profiled import (ProfiledSearchPlayer,    # noqa: E402
                                        ProfilerLikelihood)
from openhearts.search.sir import (M_CAP, MAX_DRAWS,             # noqa: E402
                                   RecordingProfilerFactory,
                                   SIRRecorder, merge_payloads,
                                   sir_posterior_factory)

ROOT = ec.ROOT
RESULTS = ec.RESULTS

# ---------------------------------------------------------------- settings
SETTINGS = ("H0", "H2")          # the two the pre-registration names
N_DEALS = 250
SIR_BLOCK = blockdriver.DEFAULT_BLOCK_SIZE       # 5 -- F0's checkpoint size
ORACLE_BLOCK = ec.BLOCK_SIZE                     # 25 -- pinned for BITWISE
                                                 # reproduction of the banked
                                                 # ORACLE blocks
WORKERS = 8
BUDGET_SECONDS = 2 * 3600

ROW_SIR = "ORACLE-SIR"
ROW_ORACLE_INSTR = "ORACLE-instr"
BASELINE = "honest-FULL"
ARCHIVED_ORACLE = "profiled-ORACLE"

#: SIR knobs.  JUSTIFICATIONS, WRITTEN BEFORE ANY MEASUREMENT (the brief
#: allows changing the brief's constants on exactly that condition):
#:
#: N_WORLDS = 50, not the brief's 100.  The archived profiled-ORACLE row runs
#: `n_outer=50 n_worlds=50` (results/entropy_curve_H0.txt line 7) -- the "100"
#: in the plan text describes the GUESSING panel, not the points row.  Handing
#: search 100 worlds would move TWO variables at once (diversity AND world
#: count); world count is Task A3's manipulated variable and Phase 1 measured
#: 100->500 worlds buying 0.12 pts, the same order as half the gap this
#: experiment must resolve.  N is defined by the pre-registration as "the N the
#: search consumes", and that is 50.
N_WORLDS = 50
#: ESS_TARGET = N.  The frozen text sets the floor at the number of worlds the
#: search consumes; at the published 1-6%-of-100 efficiency a target of 100
#: would need M ~ 1700-10000 and would turn expectation (1) into a
#: cap-hit report instead of a measurement.  Per-decision pool ESS is logged,
#: so a stricter floor can be imposed post hoc as a sensitivity analysis
#: without rerunning anything.
ESS_TARGET = float(N_WORLDS)
#: M_START = N.  A decision whose weights were never degenerate then costs
#: EXACTLY what the archived row cost, and only a degenerate decision pays for
#: more worlds.  It is also what makes the uniform-weight reduction bitwise.
M_START = N_WORLDS
RESAMPLER = "systematic"         # see search/sir.py: forced by the frozen
                                 # bitwise-reduction gate; `multinomial` (what
                                 # honest.py does) stays available as a
                                 # sensitivity row via --resampler.

SMOKE_DEALS = 4
SMOKE_BLOCK = 2
SMOKE_M_CAP = 200

PRE_REGISTERED = [
    "ORACLE-SIR's ESS meets target by construction",
    "ORACLE-SIR's guessing quality is approximately ORACLE's (it carries the "
    "same information)",
    "DECISION RULE: if ORACLE-SIR - honest-FULL improves materially on "
    "ORACLE - honest-FULL (recovers >= half the gap, or its CI includes 0 "
    "from below) at either setting -> the degeneracy mechanism is CONFIRMED; "
    "if ORACLE-SIR stays <= -0.2 with healthy measured ESS -> the post-hoc "
    "hypothesis is FALSIFIED and retirement stands on stronger grounds; "
    "partial recovery -> both statements, scoped",
]

HALF_GAP_RULE = 0.5
FALSIFIED_AT = -0.2


# ------------------------------------------------------------------ paths
def partial_path(habit, row, smoke=False):
    tag = f"{habit}_smoke" if smoke else habit
    return os.path.join(RESULTS, f"sir_oracle_{tag}_{row}_partial.txt")


def json_path(habit, smoke=False):
    tag = f"{habit}_smoke" if smoke else habit
    return os.path.join(RESULTS, f"sir_oracle_{tag}.json")


def txt_path(smoke=False):
    return os.path.join(RESULTS,
                        "sir_oracle_smoke.txt" if smoke else "sir_oracle.txt")


def guess_path(habit, smoke=False):
    tag = f"{habit}_smoke" if smoke else habit
    return os.path.join(RESULTS, f"sir_oracle_guessing_{tag}.json")


# ----------------------------------------------------------------- guards
def match_trios(habit):
    """The 6A match trios: 10 blocks of 25 deals, deal d -> trios[d // 25].

    THE TRAP THIS FUNCTION EXISTS TO BLOCK: `ec.block_trios(n)` draws n trios
    from one seeded stream, so asking for 50 blocks (250 deals / 5-deal
    checkpoints) would return a DIFFERENT trio sequence and silently unpair
    every comparison in this file.  The checkpoint size and the MATCH size are
    different things; only the match size may reach `block_trios`.
    """
    trios = ec.block_trios(N_DEALS // ec.BLOCK_SIZE, habit)
    assert len(trios) == 10, f"expected 10 match trios, got {len(trios)}"
    return trios


def guards(habit=None):
    """Fail loudly, before anything expensive, if the pairing cannot hold."""
    assert ec.BLOCK_SIZE == 25, (
        f"run_entropy_curve.BLOCK_SIZE is {ec.BLOCK_SIZE}, not 25: the banked "
        f"rows were cut into 25-deal matches and the trio mapping depends on "
        f"it.  OPENHEARTS_CURVE_BLOCK is probably set in this shell.")
    assert not os.environ.get("OPENHEARTS_CURVE_SMOKE"), (
        "OPENHEARTS_CURVE_SMOKE is set: run_entropy_curve would load the "
        "2-epoch SMOKE profiler nets and this experiment would score a "
        "phase-defining number against a throwaway model.  Unset it.")
    assert ec.DEAL_SEED_BASE == 1_600_000, "6A deal seed base moved"
    assert ec.N_OUTER == N_WORLDS and ec.N_WORLDS == N_WORLDS, (
        f"6A ran n_outer={ec.N_OUTER} n_worlds={ec.N_WORLDS}; this row is "
        f"built for {N_WORLDS} and would otherwise move world COUNT as well "
        f"as world diversity")
    assert ORACLE_BLOCK % SIR_BLOCK == 0, (
        "SIR checkpoint blocks must nest inside 25-deal match blocks")
    if habit is None:
        return
    # Pair against the ARTIFACT, not against a re-derivation of it.
    jp = os.path.join(RESULTS, f"entropy_curve_{habit}.json")
    assert os.path.exists(jp), f"missing banked 6A result {jp}"
    with open(jp) as f:
        banked = json.load(f)
    assert int(banked["deal_seed_base"]) == ec.DEAL_SEED_BASE
    assert int(banked["n_deals"]) == N_DEALS
    derived = [list(t) for t in match_trios(habit)]
    assert banked["trios"] == derived, (
        f"derived trios differ from the banked ones in {jp}: the rows would "
        f"face different people and the pairing would be a fiction.\n"
        f"banked  {banked['trios'][:2]}...\nderived {derived[:2]}...")


def banked_reference(habit):
    """{row: per-deal values [250]} for honest-FULL and profiled-ORACLE.

    Read from the 6A partial file, which banks the per-deal values themselves
    (not just block means), so every comparison in this script is paired on
    the deal index.  Missing or short blocks are a hard error: an unpaired
    bootstrap would be worse than no bootstrap.
    """
    path = ec.partial_path(habit, smoke=False)
    assert os.path.exists(path), f"missing banked partial {path}"
    banked = ec._load_partial(path)
    out = {}
    for row in (BASELINE, ARCHIVED_ORACLE):
        vals = np.full(N_DEALS, np.nan)
        for b in range(N_DEALS // ec.BLOCK_SIZE):
            key = (row, b)
            assert key in banked, f"{path}: {row} block {b} is not banked"
            v = banked[key]["values"]
            assert len(v) == ec.BLOCK_SIZE, (
                f"{path}: {row} block {b} has {len(v)} values, expected "
                f"{ec.BLOCK_SIZE}")
            vals[b * ec.BLOCK_SIZE:(b + 1) * ec.BLOCK_SIZE] = v
        assert np.isfinite(vals).all()
        out[row] = vals
    return out


# --------------------------------------------------------------- bot build
def _oracle_likelihood(trio_ids, habit):
    """The archived ORACLE likelihood, built exactly as `ec.build_bot` does.

    Returns (likelihood, hook) where `hook(seat_to_pos)` attaches the TRUE v2
    parameter vectors to the seats the trio occupies this rotation.
    """
    w = ec.nets()
    zeros = np.zeros(PARAM_DIM_V2, dtype=np.float64)
    lik = ProfilerLikelihood(w["conditioned"],
                             seat_params={s: zeros for s in range(4)})
    identity = {p: param_vector_v2(trio_ids[p], habit) for p in range(3)}

    def hook(seat_to_pos):
        lik.seat_params = {s: identity[p] for s, p in seat_to_pos.items()}

    return lik, hook


def build_sir_bot(trio_ids, habit, rng, cfg, recorder):
    """The ORACLE-SIR row: `ec.build_bot('profiled-ORACLE', ...)` with the
    world-set construction swapped and NOTHING else.

    Heuristic playouts (`playout_weights=None`) exactly as the 6A ORACLE row
    used, so this stays a reading comparison and never a playout one.
    """
    lik, hook = _oracle_likelihood(trio_ids, habit)
    pf = sir_posterior_factory(
        lik, level=Level.FULL, n_worlds=N_WORLDS, max_draws=MAX_DRAWS,
        ess_target=cfg["ess_target"], m_start=cfg["m_start"],
        m_cap=cfg["m_cap"], resampler=cfg["resampler"], keep_worlds=True,
        recorder=recorder)
    bot = ProfiledSearchPlayer(Level.FULL, ec.N_OUTER, ec.N_INNER, rng,
                               sampler_respects_voids=True,
                               posterior_factory=pf, playout_weights=None)
    return bot, hook


def build_oracle_instr_bot(trio_ids, habit, rng, recorder):
    """The ARCHIVED row plus an ESS tape.

    Built through `ec.build_bot` itself (not a copy), then its posterior
    factory is wrapped by `RecordingProfilerFactory`, which draws no
    randomness and changes no arithmetic.  The row therefore stays BITWISE the
    banked one -- which is what makes `--repro-block` a real check.
    """
    bot, hook = ec.build_bot(ARCHIVED_ORACLE, trio_ids, habit, rng)
    bot.posterior_factory = RecordingProfilerFactory(bot.posterior_factory,
                                                     recorder)
    return bot, hook


# ------------------------------------------------------------- the players
def play_deals(bot, hook, trio_ids, habit, deal_indices):
    """Play `deal_indices` x 4 rotations against one fixed trio.

    Structurally identical to `ec.run_block`'s loop, with every SEED taken
    from `run_entropy_curve` by import (`DEAL_SEED_BASE`, `opp_seat_rng`,
    `personality_player`, `OPP_CONFIG_ID`).  `--gate` proves the equivalence
    empirically rather than by inspection: it runs this function and
    `ec.run_block` on the same deals with the same bot and compares bitwise.

    Returns (per_deal_means[n], rot_scores[n, 4]).
    """
    n = len(deal_indices)
    rot_scores = np.zeros((n, 4))
    for i, idx in enumerate(deal_indices):
        seed = ec.DEAL_SEED_BASE + int(idx)
        for rotation in range(4):
            state = deal(np.random.default_rng(seed))
            other_seats = sorted(s for s in range(4) if s != rotation)
            seat_to_pos = {s: p for p, s in enumerate(other_seats)}
            hook(seat_to_pos)
            players = [None, None, None, None]
            players[rotation] = bot
            for s, p in seat_to_pos.items():
                players[s] = ec.personality_player(
                    trio_ids[p], habit,
                    ec.opp_seat_rng(ec.OPP_CONFIG_ID, seed, rotation, s))
            final = play_game(state, players)
            assert sum(final.scores) == 26, "engine invariant broken"
            rot_scores[i, rotation] = final.scores[rotation]
    return rot_scores.mean(axis=1), rot_scores


def _trio_for(deal_indices, habit):
    """The match trio these deals belong to; asserts they share one match."""
    trios = match_trios(habit)
    matches = {int(i) // ec.BLOCK_SIZE for i in deal_indices}
    assert len(matches) == 1, (
        f"checkpoint block {sorted(deal_indices)} straddles match blocks "
        f"{sorted(matches)} -- the trio would change mid-block")
    return trios[matches.pop()]


def worker(cfg, name, block_idx, item_indices):
    """One checkpoint block, in a worker process (bound via functools.partial
    so the config travels with the callable -- spawned children do not inherit
    `main`'s globals)."""
    habit = cfg["habit"]
    trio = _trio_for(item_indices, habit)
    rec = SIRRecorder()
    if name == ROW_SIR:
        rng = np.random.default_rng([cfg["sir_config_id"], block_idx])
        bot, hook = build_sir_bot(trio, habit, rng, cfg, rec)
    elif name == ROW_ORACLE_INSTR:
        # The banked row's own rng seed -- this row must reproduce it.
        rng = ec.block_rng(ARCHIVED_ORACLE, block_idx)
        bot, hook = build_oracle_instr_bot(trio, habit, rng, rec)
    else:
        raise ValueError(name)
    per_deal, rot = play_deals(bot, hook, trio, habit, item_indices)
    return {"values": [float(v) for v in per_deal],
            "rot": [[float(x) for x in r] for r in rot],
            "cnt": {"fallbacks": bot.fallbacks,
                    "posterior_collapses": bot.posterior_collapses,
                    "posterior_decisions": bot.posterior_decisions,
                    "posterior_worlds": bot.posterior_worlds},
            "sir": rec.payload()}


def run_row(name, habit, cfg, n_deals, block_size, workers, smoke):
    """One row, checkpointed through the F0 block driver."""
    import functools
    path = partial_path(habit, name, smoke)
    values = np.full(n_deals, np.nan)
    rot = np.full((n_deals, 4), np.nan)
    payloads, counters = [], []

    def on_done(_b, items, payload):
        values[items] = payload["values"]
        rot[items] = payload["rot"]
        payloads.append(payload["sir"])
        counters.append(payload["cnt"])

    used = blockdriver.run_blocks(
        name, n_deals, block_size, functools.partial(worker, cfg), workers,
        path, on_block_done=on_done, explicit_block=True)
    assert np.isfinite(values).all(), f"{name}: some deals never completed"
    cnt = {k: int(sum(c[k] for c in counters)) for k in counters[0]}
    return {"values": values, "rot": rot, "sir": merge_payloads(payloads),
            "cnt": cnt, "block_size": used}


# ------------------------------------------------------------------ gates
def gate_playloop_equivalence(habit="H0", n_deals=2):
    """`play_deals` == `ec.run_block`, bitwise, on the same bot and deals.

    This is what licenses re-implementing the game loop here instead of
    importing it: `ec.run_block` builds its bot internally, so it cannot be
    handed an instrumented one -- but any drift between the two loops would
    unpair every number in this file, so the equivalence is measured.
    """
    trio = match_trios(habit)[0]
    seeds = [ec.DEAL_SEED_BASE + d for d in range(n_deals)]
    ref_per_deal, ref_rot, _o, _e, _c = ec.run_block(
        ARCHIVED_ORACLE, 0, trio, habit, seeds)
    bot, hook = ec.build_bot(ARCHIVED_ORACLE, trio, habit,
                             ec.block_rng(ARCHIVED_ORACLE, 0))
    per_deal, rot = play_deals(bot, hook, trio, habit, list(range(n_deals)))
    ok = (np.asarray(ref_per_deal).tobytes() == np.asarray(per_deal).tobytes()
          and np.asarray(ref_rot).tobytes() == np.asarray(rot).tobytes())
    print(f"[gate] play_deals == ec.run_block on {n_deals} deals: "
          f"{'BITWISE EQUAL' if ok else 'DIFFERENT'}")
    if not ok:
        print(f"        ec.run_block   {list(ref_per_deal)}")
        print(f"        play_deals     {list(per_deal)}")
    return ok


def gate_repro_banked_block(habit, block_idx):
    """Reproduce a BANKED profiled-ORACLE block from scratch, bitwise.

    If this fails, the per-deal pairing against the banked rows is a
    comparison against a different numerical environment, and the paired
    bootstrap in `--report` is not valid.  It is a hard blocker, not a
    warning.  ~25 deals x 4 rotations on one core.
    """
    path = ec.partial_path(habit, smoke=False)
    banked = ec._load_partial(path)[(ARCHIVED_ORACLE, block_idx)]
    trio = match_trios(habit)[block_idx]
    rec = SIRRecorder()
    bot, hook = build_oracle_instr_bot(
        trio, habit, ec.block_rng(ARCHIVED_ORACLE, block_idx), rec)
    items = list(range(block_idx * ec.BLOCK_SIZE,
                       (block_idx + 1) * ec.BLOCK_SIZE))
    t0 = time.time()
    per_deal, rot = play_deals(bot, hook, trio, habit, items)
    want = np.asarray(banked["values"], dtype=float)
    got = np.asarray(per_deal, dtype=float)
    bitwise = want.tobytes() == got.tobytes()
    dmax = float(np.max(np.abs(want - got)))
    print(f"[gate] banked {habit} {ARCHIVED_ORACLE} block {block_idx} "
          f"reproduction: {'BITWISE EQUAL' if bitwise else 'DIFFERENT'} "
          f"(max |delta| = {dmax:.17g}) in {time.time() - t0:.0f}s")
    if not bitwise:
        bad = np.flatnonzero(want != got)[:5]
        for i in bad:
            print(f"        deal {items[i]}: banked {want[i]!r} vs "
                  f"reproduced {got[i]!r}")
    print(f"[gate] instrumented ORACLE recorded {rec.n_decisions} decisions "
          f"(ESS tape live: {rec.n_decisions > 0})")
    return bitwise


def run_gates(args):
    guards()
    for h in SETTINGS:
        guards(h)
        ref = banked_reference(h)
        print(f"[gate] {h}: banked per-deal values complete for "
              f"{sorted(ref)} ({N_DEALS} deals each); "
              f"FULL mean {ref[BASELINE].mean():.3f}, ORACLE mean "
              f"{ref[ARCHIVED_ORACLE].mean():.3f}, reading value "
              f"{(ref[BASELINE] - ref[ARCHIVED_ORACLE]).mean():+.3f}")
    ok = gate_playloop_equivalence()
    if args.repro_block is not None:
        ok = gate_repro_banked_block(args.habit or "H0",
                                     args.repro_block) and ok
    else:
        print("[gate] banked-block reproduction SKIPPED (pass --repro-block 0 "
              "-- it is a hard blocker for the paired bootstrap)")
    print(f"[gate] overall: {'PASS' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------------ probe
def timing_probe(habit, cfg, n_deals=2, workers=WORKERS):
    """Measure BOTH rows, and print the achieved pool sizes -- cost is driven
    by M (how far the ESS escalation has to climb), not by deals."""
    guards(habit)
    trio = match_trios(habit)[0]
    items = list(range(n_deals))
    out = {}

    for name, builder in ((ROW_ORACLE_INSTR,
                           lambda r: build_oracle_instr_bot(
                               trio, habit, ec.block_rng(ARCHIVED_ORACLE, 0),
                               r)),
                          (ROW_SIR,
                           lambda r: build_sir_bot(
                               trio, habit,
                               np.random.default_rng([cfg["sir_config_id"], 0]),
                               cfg, r))):
        rec = SIRRecorder()
        bot, hook = builder(rec)
        t0 = time.time()
        play_deals(bot, hook, trio, habit, items)
        spg = (time.time() - t0) / (4 * n_deals)
        out[name] = (spg, rec.payload())
        print(f"[probe {habit}] {name}: {spg:.3f}s/game "
              f"({rec.n_decisions} posterior decisions, "
              f"{rec.cap_hits} cap hits)")

    print(f"\n[probe {habit}] per-trick pool behaviour "
          f"(the cost driver; ESS_plain is what the ARCHIVED weighting "
          f"suffers):")
    print(f"  {'trick':>6}{'n':>6}{'ESS_plain':>11}{'ESS_pool':>10}"
          f"{'mean M':>9}{'dist(pool)':>12}{'dist(out)':>11}{'cap%':>7}")
    for row_name in (ROW_ORACLE_INSTR, ROW_SIR):
        print(f"  -- {row_name}")
        pt = out[row_name][1]["per_trick"]
        for t in sorted(pt, key=int):
            r = pt[t]
            n = max(1.0, r["n"])
            print(f"  {t:>6}{int(r['n']):>6}{r['ess_plain']/n:>11.1f}"
                  f"{r['ess_pool']/n:>10.1f}{r['m']/n:>9.0f}"
                  f"{r['distinct_pool']/n:>12.0f}"
                  f"{r['distinct_resampled']/n:>11.0f}"
                  f"{100*r['cap_hits']/n:>7.1f}")

    games = 4 * N_DEALS
    sir_total = out[ROW_SIR][0] * games
    orc_total = out[ROW_ORACLE_INSTR][0] * games
    print("=" * 72)
    for label, tot in (("ORACLE-SIR row", sir_total),
                       ("ORACLE-instr row", orc_total),
                       ("both rows", sir_total + orc_total)):
        print(f"[BUDGET] {habit} {label}: {tot:.0f} worker-seconds "
              f"({tot/3600:.2f}h serial) -> {tot/workers/3600:.2f}h wall at "
              f"{workers} workers")
    both = (sir_total + orc_total) / workers
    print(f"[BUDGET] one setting, both rows: {both/3600:.2f}h wall vs the "
          f"{BUDGET_SECONDS/3600:.0f}h single-run cap -> "
          f"{'WITHIN' if both <= BUDGET_SECONDS else 'OVER'}")
    print(f"[BUDGET] both settings ({', '.join(SETTINGS)}), run separately: "
          f"{2*both/3600:.2f}h total wall")
    print("=" * 72)
    print("[probe] done (no full run performed)")
    return out


# ---------------------------------------------------------------- one run
def run_setting(habit, args, smoke=False):
    guards(habit)
    cfg = dict(habit=habit, ess_target=args.ess_target, m_start=M_START,
               m_cap=args.m_cap, resampler=args.resampler,
               sir_config_id=7101)
    n_deals = SMOKE_DEALS if smoke else N_DEALS
    sir_block = SMOKE_BLOCK if smoke else SIR_BLOCK
    orc_block = SMOKE_BLOCK if smoke else ORACLE_BLOCK
    t0 = time.time()

    rows = {}
    rows[ROW_SIR] = run_row(ROW_SIR, habit, cfg, n_deals, sir_block,
                            args.workers, smoke)
    if args.rerun_oracle:
        rows[ROW_ORACLE_INSTR] = run_row(ROW_ORACLE_INSTR, habit, cfg,
                                         n_deals, orc_block, args.workers,
                                         smoke)

    ref = banked_reference(habit)
    out = {"habit": habit, "n_deals": int(n_deals), "smoke": bool(smoke),
           "wall_time_s": time.time() - t0, "workers": args.workers,
           "cfg": {k: v for k, v in cfg.items()},
           "n_worlds": N_WORLDS, "entropy_nats": None,
           "banked": {k: [float(x) for x in v[:n_deals]]
                      for k, v in ref.items()},
           "rows": {}}
    jp = os.path.join(RESULTS, f"entropy_curve_{habit}.json")
    with open(jp) as f:
        out["entropy_nats"] = json.load(f)["entropy_nats"]
    for name, r in rows.items():
        out["rows"][name] = {"values": [float(x) for x in r["values"]],
                             "cnt": r["cnt"], "sir": r["sir"],
                             "block_size": r["block_size"]}
    with open(json_path(habit, smoke), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] {habit} in {(time.time()-t0)/60:.1f} min; wrote "
          f"{json_path(habit, smoke)}", flush=True)
    return out


# ----------------------------------------------------------------- report
def _paired(a, b):
    """(mean, lo, hi) of the paired per-deal difference a - b."""
    m, lo, hi = bootstrap_ci(np.asarray(a, float) - np.asarray(b, float))
    return {"mean": float(m), "lo": float(lo), "hi": float(hi)}


def _ess_table(A, sir_payload, oracle_payload):
    """Per-trick ESS for BOTH configurations, clearly labelled."""
    A("PER-TRICK EFFECTIVE SAMPLE SIZE -- the causal variable.")
    A("  ESS (Kish) = (sum w)^2 / sum(w^2): 'how many of the imagined worlds "
      "is this weighted set really worth'.  50 means all 50 count; 3 means "
      "the bot averaged three worlds photocopied.")
    A("  CONFIG A = the ARCHIVED profiled-ORACLE weighting (draw 50, weight, "
      "use).  CONFIG B = ORACLE-SIR's pool after escalation.")
    A("  A CAP HIT IS NOT ALWAYS DEGENERACY, stated before the run: late in a "
      "hand the number of DISTINCT worlds consistent with the constraints "
      "can itself fall below 50, so no amount of oversampling can add "
      "diversity that does not exist.  Read cap% next to 'distinct(pool)' -- "
      "a cap hit with distinct < 50 is a support limit, a cap hit with "
      "distinct >> 50 is degeneracy.")
    A("  AND READ ESS NEXT TO distinct(pool) FOR A SECOND REASON: the pool "
      "can contain the SAME world drawn twice, and Kish ESS counts pool "
      "ENTRIES, so late in a hand a high ESS over a small support is real "
      "arithmetic but overstates diversity.  `distinct(out)` -- how many "
      "different worlds the search actually received -- is the number that "
      "cannot be inflated that way, and it is the honest headline for "
      "expectation (1).")
    A("")
    # The CONFIG-A proxy is ESS-ONLY on purpose: it is read off the first 50
    # members of the SIR pool, so the pool's M / distinct / cap columns belong
    # to CONFIG B and printing them beside it would attribute SIR's pool to
    # the archived weighting.
    A("  CONFIG A: profiled-ORACLE weighting, measured on ORACLE-SIR's own "
      "decision path (the first 50 of the SIR pool ARE a plain draw, so this "
      "is what the archived weighting WOULD have suffered at these views).")
    A(f"    {'trick':>6}{'n':>7}{'ESS':>8}")
    pt = sir_payload["per_trick"]
    for t in sorted(pt, key=int):
        r = pt[t]
        A(f"    {t:>6}{int(r['n']):>7}{r['ess_plain']/max(1.0, r['n']):>8.1f}")
    A(f"    ALL   {sir_payload['n_decisions']:>7}"
      f"{sum(r['ess_plain'] for r in pt.values())/max(1, sir_payload['n_decisions']):>8.1f}")
    A("")

    for label, payload, note, foot in (
            ("CONFIG A: profiled-ORACLE weighting, measured on the ARCHIVED "
             "row's own decision path (ORACLE-instr; bitwise the banked row)",
             oracle_payload, "ess_plain", True),
            ("CONFIG B: ORACLE-SIR pool after ESS escalation",
             sir_payload, "ess_pool", False)):
        if payload is None:
            A(f"  {label}: NOT RUN (pass --rerun-oracle)")
            A("")
            continue
        A(f"  {label}")
        A(f"    {'trick':>6}{'n':>7}{'ESS':>8}{'mean M':>9}"
          f"{'distinct(pool)':>15}{'distinct(out)':>14}{'cap%':>7}"
          f"{'underflow':>11}")
        pt = payload["per_trick"]
        nd = max(1, payload["n_decisions"])
        for t in sorted(pt, key=int):
            r = pt[t]
            n = max(1.0, r["n"])
            A(f"    {t:>6}{int(r['n']):>7}{r[note]/n:>8.1f}{r['m']/n:>9.0f}"
              f"{r['distinct_pool']/n:>15.0f}{r['distinct_resampled']/n:>14.0f}"
              f"{100*r['cap_hits']/n:>7.1f}{r['n_underflow']/n:>11.1f}")
        A(f"    ALL   {payload['n_decisions']:>7}"
          f"{sum(r[note] for r in pt.values())/nd:>8.1f}"
          f"{sum(r['m'] for r in pt.values())/nd:>9.0f}"
          f"{sum(r['distinct_pool'] for r in pt.values())/nd:>15.0f}"
          f"{sum(r['distinct_resampled'] for r in pt.values())/nd:>14.0f}"
          f"{100*payload['cap_hits']/nd:>7.1f}")
        if foot:
            A("    (distinct(out) here is the pool's OWN distinct count: "
              "honest.py multinomially resamples 50 of those AFTER the "
              "posterior returns, so what the archived search finally "
              "consumed is not observable from inside the posterior.  Its "
              "true diversity is therefore AT MOST the number shown and, by "
              "the ESS beside it, far less.)")
        A("")


def build_report(smoke=False):
    data = {}
    for h in SETTINGS:
        p = json_path(h, smoke)
        if os.path.exists(p):
            with open(p) as f:
                data[h] = json.load(f)
    if not data:
        print("no per-setting JSONs found; run a setting first")
        return
    L = []
    A = L.append
    A("open-hearts Phase 7 Task A1: THE ESS / RESAMPLING TEST")
    A("Does reading fail because the information is worthless, or because our "
      "estimator is degenerate?")
    A("")
    if smoke or any(data[h]["smoke"] for h in data):
        A("*** SMOKE INPUTS -- plumbing only, not a result. ***")
        A("")
    A("Points: LOWER IS BETTER (6.5 = symmetric break-even).")
    A("READING VALUE (the 6A convention, used for the decision rule) = "
      "honest-FULL - row, paired per deal.  POSITIVE means the row BEATS "
      "honest-FULL.  The raw difference (row - honest-FULL) is printed beside "
      "it so the sign can never be misread.")
    A("")
    A(f"world set: N={N_WORLDS} worlds handed to search (the archived ORACLE "
      f"row's own n_outer/n_worlds), ESS floor {ESS_TARGET:.0f}, pool "
      f"{M_START} doubling to at most {M_CAP}, {RESAMPLER} resampling.")
    A("NOTE ON A COUNTER: `posterior_worlds` counts what the posterior HANDED "
      "OVER, which for SIR is N (the resampled set) and never M (the pool).  "
      "M is reported in the ESS tables below.")
    A(f"deals: {N_DEALS} x 4 rotations, seeds {ec.DEAL_SEED_BASE}+, in 10 "
      f"match blocks of {ec.BLOCK_SIZE} vs one fixed HELD-OUT v2 trio -- the "
      f"IDENTICAL deals, trios, rotations and opponent streams the banked 6A "
      f"rows used (asserted against results/entropy_curve_<H>.json at run "
      f"time, and against a bitwise reproduction of a banked ORACLE block "
      f"under --gate --repro-block).")
    A("")

    for h in sorted(data):
        d = data[h]
        base = np.asarray(d["banked"][BASELINE], float)
        orc = np.asarray(d["banked"][ARCHIVED_ORACLE], float)
        sir = np.asarray(d["rows"][ROW_SIR]["values"], float)
        n = len(sir)
        base, orc = base[:n], orc[:n]
        A("=" * 74)
        A(f"SETTING {h}  (measured opponent entropy "
          f"{d['entropy_nats']:.3f} nats, {n} deals, "
          f"wall {d['wall_time_s']:.0f}s on {d['workers']} workers)")
        A("")
        A(f"{'row':<26}{'mean':>8}{'lo95':>9}{'hi95':>9}   {'source':<28}")
        for name, vals, src in (
                (BASELINE, base, "banked 6A"),
                (ARCHIVED_ORACLE, orc, "banked 6A"),
                (ROW_SIR, sir, "this run")):
            m, lo, hi = bootstrap_ci(vals)
            A(f"{name:<26}{m:>8.3f}{lo:>9.3f}{hi:>9.3f}   {src:<28}")
        if ROW_ORACLE_INSTR in d["rows"]:
            v = np.asarray(d["rows"][ROW_ORACLE_INSTR]["values"], float)[:n]
            m, lo, hi = bootstrap_ci(v)
            same = "BITWISE EQUAL to banked" if v.tobytes() == orc.tobytes() \
                else f"DIFFERS from banked (max |d| {np.abs(v-orc).max():.6g})"
            A(f"{ROW_ORACLE_INSTR:<26}{m:>8.3f}{lo:>9.3f}{hi:>9.3f}   {same}")
        A("")
        v_orc = _paired(base, orc)
        v_sir = _paired(base, sir)
        d_sir_orc = _paired(orc, sir)
        A("PAIRED PER-DEAL COMPARISONS (10,000-resample bootstrap over deals)")
        A(f"  reading value, ORACLE     (FULL - ORACLE)     "
          f"{v_orc['mean']:+.3f}  CI ({v_orc['lo']:+.3f}, {v_orc['hi']:+.3f})")
        A(f"  reading value, ORACLE-SIR (FULL - SIR)        "
          f"{v_sir['mean']:+.3f}  CI ({v_sir['lo']:+.3f}, {v_sir['hi']:+.3f})")
        A(f"  SIR vs ORACLE             (ORACLE - SIR)      "
          f"{d_sir_orc['mean']:+.3f}  CI ({d_sir_orc['lo']:+.3f}, "
          f"{d_sir_orc['hi']:+.3f})   [+ = SIR beats ORACLE]")
        A(f"  raw differences: SIR - FULL = {-v_sir['mean']:+.3f}, "
          f"SIR - ORACLE = {-d_sir_orc['mean']:+.3f} (lower is better)")
        A("")
        _ess_table(A, d["rows"][ROW_SIR]["sir"],
                   d["rows"].get(ROW_ORACLE_INSTR, {}).get("sir"))
        c = d["rows"][ROW_SIR]["cnt"]
        A(f"counters (ORACLE-SIR): fallbacks {c['fallbacks']}, posterior "
          f"collapses {c['posterior_collapses']}, posterior decisions "
          f"{c['posterior_decisions']}, worlds handed to search "
          f"{c['posterior_worlds']}")
        A("")

    A("#" * 74)
    A("## PRE-REGISTERED EXPECTATIONS (PHASE7_PLAN.md Task A1), VERBATIM")
    for text, ok, note in _judge(data):
        A(f"##  [{'PASS' if ok else 'FAIL' if ok is False else 'N/A '}] {text}")
        A(f"##        {note}")
    A("#" * 74)
    txt = "\n".join(L) + "\n"
    with open(txt_path(smoke), "w") as f:
        f.write(txt)
    print(txt, flush=True)
    print(f"wrote {txt_path(smoke)}", flush=True)


def _judge(data):
    out = []
    # (1) ESS meets target by construction.
    lines = []
    ok1 = True
    for h in sorted(data):
        p = data[h]["rows"][ROW_SIR]["sir"]
        nd = max(1, p["n_decisions"])
        pool = sum(r["ess_pool"] for r in p["per_trick"].values()) / nd
        plain = sum(r["ess_plain"] for r in p["per_trick"].values()) / nd
        dout = sum(r["distinct_resampled"] for r in p["per_trick"].values()) / nd
        cap = 100.0 * p["cap_hits"] / nd
        lines.append(f"{h}: mean pool ESS {pool:.1f} vs plain-{N_WORLDS} ESS "
                     f"{plain:.1f}; mean DISTINCT worlds actually handed to "
                     f"search {dout:.1f} of {N_WORLDS}; cap-hit {cap:.1f}% of "
                     f"{nd} decisions")
        ok1 = ok1 and pool >= ESS_TARGET * 0.8
    out.append((PRE_REGISTERED[0], ok1,
                "; ".join(lines) + f" (target {ESS_TARGET:.0f}; judged on the "
                f"MEAN pool ESS reaching 80% of target, with cap hits "
                f"reported beside it -- some late-trick cap hits are support "
                f"limits, not degeneracy)"))

    # (2) guessing -- measured by --guessing, not by this run.
    gl = []
    for h in sorted(data):
        gp = guess_path(h, data[h]["smoke"])
        if os.path.exists(gp):
            with open(gp) as f:
                g = json.load(f)
            gl.append(f"{h}: ORACLE meanP {g['ORACLE']['meanP']:.4f} vs "
                      f"SIR(pool) {g['SIR-pool']['meanP']:.4f} vs "
                      f"SIR(resampled) {g['SIR-resampled']['meanP']:.4f}")
    out.append((PRE_REGISTERED[1], None if not gl else True,
                "; ".join(gl) if gl else
                "not measured in this run -- run `--guessing --habit <H>` "
                "(the panel is priced separately because a SIR posterior at "
                "every boundary x seat costs far more than a points row)"))

    # (3) the decision rule.
    verdicts = []
    for h in sorted(data):
        d = data[h]
        n = len(d["rows"][ROW_SIR]["values"])
        base = np.asarray(d["banked"][BASELINE], float)[:n]
        orc = np.asarray(d["banked"][ARCHIVED_ORACLE], float)[:n]
        sir = np.asarray(d["rows"][ROW_SIR]["values"], float)
        v_orc = _paired(base, orc)
        v_sir = _paired(base, sir)
        gap = -v_orc["mean"]                     # how far ORACLE sat below 0
        recovered = (v_sir["mean"] - v_orc["mean"]) / gap if gap > 0 else \
            float("nan")
        confirms = (recovered >= HALF_GAP_RULE) or (v_sir["hi"] >= 0.0)
        falsifies = v_sir["mean"] <= FALSIFIED_AT
        verdicts.append(
            f"{h}: ORACLE {v_orc['mean']:+.3f} -> SIR {v_sir['mean']:+.3f} "
            f"(CI {v_sir['lo']:+.3f}, {v_sir['hi']:+.3f}); recovered "
            f"{100*recovered:.0f}% of the {gap:.3f} gap; "
            f"CONFIRMS={confirms} FALSIFIES={falsifies}")
    out.append((PRE_REGISTERED[2], None, "; ".join(verdicts) +
                " -- the owner reads the verdict against these numbers and "
                "against the GO-MCTS frontier stake; this line reports, it "
                "does not decide."))
    return out


# -------------------------------------------------------------- guessing
def run_guessing(habit, args, smoke=False):
    """Expectation (2): does SIR carry the same information as ORACLE?

    Priced separately from the points row on purpose -- a SIR posterior at
    every trick boundary x every seat is far more expensive than one played
    game, so the panel runs on a handful of games and says so.
    """
    guards(habit)
    from openhearts.eval import guessing as gmod
    from openhearts.search.profiled import profiler_posterior
    from openhearts.search.sir import sir_posterior
    from openhearts.engine.state import GameState

    curves = ("ORACLE", "SIR-pool", "SIR-resampled")
    acc = {c: [] for c in curves}
    for gi in range(args.guess_games):
        seed = ec.GUESS_SEED_BASE + gi
        hands, plays, pids = ec.guess_play(seed, habit)
        rec = ec._Rec(seed, hands, plays, pids)
        lik = ProfilerLikelihood(
            ec.nets()["conditioned"],
            {s: param_vector_v2(p, habit) for s, p in enumerate(pids)})

        def boundary(state, trick):
            for seat in range(4):
                view = state.view_for(seat)
                rng_seed = int(np.random.default_rng(
                    [seed, trick, seat]).integers(0, 2 ** 63 - 1))
                o = profiler_posterior(view, Level.FULL, lik, N_WORLDS,
                                       np.random.default_rng(rng_seed),
                                       ec.CHOICE_MAX_DRAWS, keep_worlds=True)
                s = sir_posterior(view, Level.FULL, lik, N_WORLDS,
                                  np.random.default_rng(rng_seed),
                                  ess_target=args.ess_target,
                                  m_start=M_START, m_cap=args.m_cap,
                                  resampler=args.resampler)
                truth = gmod._truth_for(seat, rec, o)
                for c, probs in (("ORACLE", o.probs),
                                 ("SIR-pool", s.sir_pool_probs),
                                 ("SIR-resampled", s.probs)):
                    mp, top1, lt01, ncards = ec._guess_metrics(probs, truth)
                    acc[c].append({"meanP": mp, "top1": top1, "n_lt01": lt01,
                                   "n_cards": ncards, "trick": trick})

        state = GameState(hands=list(hands))
        state.to_play = plays[0][0]
        boundary(state, 1)
        for seat, card in plays:
            state.play(card)
            if not state.current_trick and state.trick_number < ec.NUM_TRICKS:
                boundary(state, state.trick_number + 1)
        print(f"[guessing {habit}] game {gi + 1}/{args.guess_games}",
              flush=True)

    out = {}
    for c in curves:
        rows = acc[c]
        ncards = sum(r["n_cards"] for r in rows) or 1
        out[c] = {"meanP": float(np.mean([r["meanP"] for r in rows])),
                  "top1": float(np.mean([r["top1"] for r in rows])),
                  "truth_lt01_frac": sum(r["n_lt01"] for r in rows) / ncards,
                  "n": len(rows)}
    out["n_games"] = args.guess_games
    out["n_worlds"] = N_WORLDS
    with open(guess_path(habit, smoke), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nGUESSING PANEL {habit} ({args.guess_games} held-out v2 games, "
          f"n_worlds={N_WORLDS} -- NOT comparable to the banked 6A panel, "
          f"which ran 100 worlds):")
    print(f"  {'curve':<16}{'meanP':>9}{'top1':>9}{'truth<.01':>11}{'n':>8}")
    for c in curves:
        g = out[c]
        print(f"  {c:<16}{g['meanP']:>9.4f}{g['top1']:>9.4f}"
              f"{g['truth_lt01_frac']:>11.4f}{g['n']:>8}")
    print(f"wrote {guess_path(habit, smoke)}")
    return out


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--habit", choices=list(SETTINGS))
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--repro-block", type=int, default=None,
                    help="--gate: reproduce this banked ORACLE match block "
                         "(0-9) bitwise; the hard blocker for the pairing")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--guessing", action="store_true")
    ap.add_argument("--guess-games", type=int, default=6)
    ap.add_argument("--rerun-oracle", action="store_true", default=False,
                    help="also run the ARCHIVED ORACLE row with an ESS tape "
                         "(bitwise the banked row; gives CONFIG A's per-trick "
                         "ESS on its own decision path)")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--m-cap", type=int, default=M_CAP)
    ap.add_argument("--ess-target", type=float, default=ESS_TARGET)
    ap.add_argument("--resampler", choices=("systematic", "multinomial"),
                    default=RESAMPLER)
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    if args.report:
        build_report(smoke=args.smoke)
        return
    if args.gate:
        sys.exit(0 if run_gates(args) else 1)
    if args.smoke:
        args.m_cap = min(args.m_cap, SMOKE_M_CAP)
        args.rerun_oracle = True
        args.guess_games = min(args.guess_games, 2)
        for h in ([args.habit] if args.habit else list(SETTINGS)):
            run_setting(h, args, smoke=True)
        build_report(smoke=True)
        return
    if not args.habit:
        ap.error("pass --habit H0|H2, or --gate / --smoke / --probe / "
                 "--report")
    cfg = dict(habit=args.habit, ess_target=args.ess_target, m_start=M_START,
               m_cap=args.m_cap, resampler=args.resampler, sir_config_id=7101)
    if args.probe:
        timing_probe(args.habit, cfg, workers=args.workers)
        return
    if args.guessing:
        run_guessing(args.habit, args)
        return
    run_setting(args.habit, args, smoke=False)


if __name__ == "__main__":
    main()
