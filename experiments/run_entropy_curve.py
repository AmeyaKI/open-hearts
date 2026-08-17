"""Phase 6 Task A2: THE ENTROPY CURVE -- what is reading worth, as a function
of how predictable the opponents are?

THE ONE-SENTENCE QUESTION.  Phase 5 measured reading against a population that
was noisy by design (3-25% of every player's moves were pure whim) and found
it worth ~nothing in points.  Task A1 built a dial -- H0 (creature of habit)
through H3 (whim machine, worse than Phase 5) -- that moves ONLY predictability
and leaves playing style bit-identical.  This script re-runs the reading
experiments at each setting and draws the curve.  If reading's value climbs as
opponents become habitual, Phase 5 tested reading in a hurricane and the
machinery revives for humans; if it stays flat even at H0, reading is dead.

WHAT RUNS AT EACH SETTING
-------------------------
(a) GUESSING PANEL -- the prediction-currency diagnostic, vs HELD-OUT v2
    personalities at that setting: FULL, CHOICE-strict, PROFILER (v2 GENERIC),
    PROFILER-ORACLE (v2 CONDITIONED + true v2 params).
(b) SLIM PAIRED ABLATION -- 250 deals, match-blocked into 10 blocks of 25 vs
    one fixed held-out trio each, 4 rotations per deal, four rows:

      honest-FULL            the incumbent and THE BASELINE (Phase 5's winner)
      honest-CHOICE-strict   the script-assuming reader (expected to collapse)
      profiled-R             v2 GENERIC reading
      profiled-ORACLE        v2 CONDITIONED reading + the TRUE v2 params

    NO CHOICE-soft row: Phase 5 measured it at 3.443 vs FULL's 2.345 -- reading
    without a license is actively harmful, the question is settled, and a row
    that exists only to lose again is 25% of the budget.  NO RI/RIA rows:
    playouts (Organ 2 = +0.009 on FULL, decisively no stack) and adaptation
    (~0 in points) are settled/parked.  This curve isolates READING.

    BOTH profiled rows use HEURISTIC playouts (`playout_weights=None`).  This
    DEVIATES from run_ablation5.py, where profiled-ORACLE carried GENERIC
    playouts, and the deviation is deliberate: with profiled playouts on one
    row and not the other, ORACLE - FULL would confound reading value with
    playout value -- the exact quantity this curve exists to isolate.

(c) RARE-MOMENT CONDITIONALS (Global Constraints: "value that lives in rare
    moments dilutes to invisibility in overall averages").  See RARE MOMENTS.

WHY THE PROFILER IS RETRAINED FIRST (the lead's design addition).  Both
profiled rows load the v2 nets from `results/profiler_train_v2/`, trained by
`train_profiler.py --v2` on a 25/25/25/25 habit mix of v2 TRAIN personalities.
Scoring the Phase-5 net here would measure model STALENESS (it never saw the
three contextual axes, never saw an H0 epsilon) rather than reading value.
The npz's `population_v` is asserted on load; `models/profiler_v1.npz` is never
read, and no Phase-5 result file is written.

INTERPRETATION NOTE, WRITTEN BEFORE ANY RESULT EXISTS.  The v2 GENERIC net is
trained across all four settings and is never told which one it faces, so
`profiled-R` at H0 UNDERSTATES what a habit-aware reader could do.
`profiled-ORACLE` -- whose parameter vector carries the habit-transformed
epsilon/temperature actually in effect -- is the row that answers "is reading
worth anything against habitual opponents", and the pre-registered decision
rule keys on it.  A flat R row with a rising ORACLE row means "reading works,
generic reading does not"; a flat ORACLE row is the one that retires reading.

PAIRING, STATED LOUDLY BECAUSE IT IS A READING OF "FRESH SEED RANGES".  All
four settings use the SAME fresh deal-seed block and the SAME held-out
personality ids (the dial is a transform, not a resample -- H0-Priya and
H2-Priya are the same person).  The curve is therefore paired on both card
luck and personality; only predictability moves.  The seeds are fresh with
respect to every Phase-5 range, which is what the plan requires.

ONE PRE-REGISTERED EXPECTATION MAY BE STRUCTURALLY FALSE, SAID NOW.
Expectation (4) predicts CHOICE-strict's collapse rate FALLS as habit
strengthens ("habitual players ARE closer to scripts").  CHOICE-strict does
not collapse on unpredictability, though -- it collapses on INCONSISTENCY WITH
`HeuristicPlayer`.  An H0 personality is a near-argmax of its OWN scorer,
including the three contextual axes A1 added precisely because our beginner
script could never predict them, so it contradicts the script MORE reliably
than a noisy player who sometimes stumbles onto the heuristic's card by
chance.  Habitual is not the same as script-like.  If (4) fails, that is the
reason, recorded before the run rather than after it.

RARE MOMENTS (pre-registered slicing rule)
------------------------------------------
Both slices are pure functions of (deal, rotation) -- computable from the
dealt hands alone, BEFORE a card is played.  That is not a stylistic choice:
a slice that depended on how the hand went would select different subsets for
different rows and destroy the pairing that makes the diffs meaningful.

  QS-OPP     the Q(spades) sat in an OPPONENT's hand at our first Q-relevant
             decision -- operationalized as "not in our dealt hand".  RETENTION
             IS REPORTED because it is high by construction: the bot occupies
             one seat, so ~3 of every 4 rotations qualify.  This is the plan's
             literal pre-registration; read it as "the queen is not ours to
             control", not as a rare event.
  QS-EXPOSED a documented SHARPENING (not pre-registered): QS-OPP *and* we
             hold the A(spades) or K(spades) -- the spot where who holds the
             queen actually changes our play, because we are the one who can
             be made to eat it.

Per-decision divergence weighting (the other rider) was assessed and is NOT
feasible cheaply: `HonestSearchPlayer.choose` computes each candidate's mean
imagined value and discards it, so the spread is not observable without
editing the search hot path; and the spread is ROW-DEPENDENT (each row's own
beliefs produce it), so weighting by it would break the paired comparison
unless recomputed from one shared reference row.  Recorded as infeasible
within this task rather than approximated.

MECHANICS
---------
One background run per setting (the lead runs four).  Checkpointed exactly as
run_ablation5.py: append-only JSON-v2 lines, resume skips banked chunks,
NEVER truncates.  `--smoke` writes to `_smoke` paths and can never touch a
real checkpoint or a real output.  `--report` merges the four per-setting
JSONs into results/entropy_curve.txt + .png and prints the pre-registered
expectations verbatim with PASS/FAIL and the decision rule.

Usage:
  .venv/bin/python experiments/run_entropy_curve.py --smoke
  .venv/bin/python experiments/run_entropy_curve.py --probe --habit H3
  .venv/bin/python experiments/run_entropy_curve.py --habit H0     # LEAD ONLY
  .venv/bin/python experiments/run_entropy_curve.py --report
"""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openhearts.belief.table import BeliefTable, Level  # noqa: E402
from openhearts.belief.weighted import (PosteriorCollapse,  # noqa: E402
                                        WeightedPosterior)
from openhearts.engine import cards  # noqa: E402
from openhearts.engine.game import deal, play_game  # noqa: E402
from openhearts.engine.state import GameState  # noqa: E402
from openhearts.eval import guessing  # noqa: E402
from openhearts.eval.stats import bootstrap_ci  # noqa: E402
from openhearts.opponent.infer import load_profiler  # noqa: E402
from openhearts.opponent.npz_io import load_npz  # noqa: E402
from openhearts.opponent.params import (PARAM_DIM_V2,  # noqa: E402
                                        param_vector_v2)
from openhearts.players.heuristic import HeuristicPlayer  # noqa: E402
from openhearts.players.personality import (HABIT_ORDER,  # noqa: E402
                                            PersonalityPlayer,
                                            make_population_v2,
                                            sample_personality_v2)
from openhearts.search.honest import HonestSearchPlayer  # noqa: E402
from openhearts.search.profiled import (ProfiledSearchPlayer,  # noqa: E402
                                        ProfilerLikelihood,
                                        profiler_posterior,
                                        profiler_posterior_factory)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

# The v2 nets.  NOT models/profiler_v1.npz, and NOT
# results/profiler_train/profiler_conditioned.npz (Phase 5's nets) -- both are
# left untouched by this experiment.
# `--smoke` resolves to the SMOKE training directory instead, so a plumbing
# run needs no full retrain and can never be mistaken for a real one.  The
# switch travels through the ENVIRONMENT because worker processes are spawned
# (they re-import this module and would otherwise reset a module global).
def net_dir():
    suffix = "_smoke" if os.environ.get("OPENHEARTS_CURVE_SMOKE") else ""
    return os.path.join(RESULTS, "profiler_train_v2" + suffix)


def generic_npz():
    return os.path.join(net_dir(), "profiler_generic.npz")


def conditioned_npz():
    return os.path.join(net_dir(), "profiler_conditioned.npz")

# ---- deals / blocks (ablation half) --------------------------------------
N_DEALS_TARGET = 250
# A block is one MATCH: one trio held fixed for 25 consecutive deals.  The
# ENVIRONMENT override exists for `--smoke` only (3-deal blocks, so a plumbing
# run finishes in a minute); it travels through the environment because worker
# processes are spawned and re-import this module.  A real run never sets it,
# and the value is recorded in every output file.
BLOCK_SIZE = int(os.environ.get("OPENHEARTS_CURVE_BLOCK", 25))
SMOKE_BLOCK_SIZE = 3
DEAL_SEED_BASE = 1_600_000     # fresh; shared across all four settings
TRIO_SEED = [606060, 6161]
BUDGET_SECONDS = 2 * 3600
WORKERS = 8
MEM_LIMIT_GB = 100.0

N_OUTER = 50
N_INNER = 20
N_WORLDS = 50
CHOICE_MAX_DRAWS = 50_000
CHOICE_STRICT_EPS = 0.0

OPP_CONFIG_ID = 9100           # shared by every row: identical opponent
                               # behaviour streams, only the bot differs
ROW_NAMES = ["honest-FULL", "honest-CHOICE-strict", "profiled-R",
             "profiled-ORACLE"]
ROW_CONFIG_ID = {name: 6200 + i for i, name in enumerate(ROW_NAMES)}
BASELINE_ROW = "honest-FULL"
READING_ROWS = ["honest-CHOICE-strict", "profiled-R", "profiled-ORACLE"]

# ---- guessing panel -------------------------------------------------------
GUESS_SEED_BASE = 1_700_000    # fresh; shared across settings
GUESS_GAMES = 120
SMOKE_GUESS_GAMES = 2
GUESS_N_WORLDS = 100
GUESS_MAX_DRAWS = 50_000
CURVES = ("FULL", "CHOICE-strict", "PROFILER", "PROFILER-ORACLE")
NUM_TRICKS = guessing.NUM_TRICKS
TABLE_SALT = 777

BREAK_EVEN = 6.5
READING_LIVES_THRESHOLD = 0.3  # pts/hand; PHASE6_PLAN Task A2 decision rule

PRE_REGISTERED = [
    "reading value (ORACLE - honest-FULL, paired) increases monotonically "
    "from H3 to H0",
    "at H0 reading value is clearly positive: ORACLE at least 0.3 pts/hand "
    "better than honest-FULL",
    "at H2 reading value lands NEAR Phase 5's ~0 (approximate continuity; a "
    "material deviation is a finding about the v2 axes, reported as such)",
    "CHOICE-strict collapses at H2-H3, and its collapse rate FALLS as habit "
    "strengthens (habitual players ARE closer to scripts)",
    "the rare-moment slices show larger reading value than the overall means",
]

_HELDOUT_V2 = list(make_population_v2()[1])


# ------------------------------------------------------------------- paths
def partial_path(habit, smoke=False):
    tag = f"{habit}_smoke" if smoke else habit
    return os.path.join(RESULTS, f"entropy_curve_{tag}_partial.txt")


def json_path(habit, smoke=False):
    tag = f"{habit}_smoke" if smoke else habit
    return os.path.join(RESULTS, f"entropy_curve_{tag}.json")


def setting_txt_path(habit, smoke=False):
    tag = f"{habit}_smoke" if smoke else habit
    return os.path.join(RESULTS, f"entropy_curve_{tag}.txt")


# -------------------------------------------------------------- population
def block_trios(n_blocks, habit):
    """Distinct held-out v2 trios; identical id sets at every setting.

    The trio DRAW does not depend on `habit` (the dial is a transform, not a
    resample), so block b faces the same three people at every setting -- the
    curve is paired on personality.  `habit` is taken only to make that
    explicit at the call site and to key the seeded rng's documentation.
    """
    rng = np.random.default_rng(TRIO_SEED)
    trios, seen = [], set()
    while len(trios) < n_blocks:
        idx = rng.choice(len(_HELDOUT_V2), size=3, replace=False)
        trio = tuple(int(_HELDOUT_V2[int(i)]) for i in idx)
        key = frozenset(trio)
        if key in seen:
            continue
        seen.add(key)
        trios.append(trio)
    return trios


def personality_player(pid, habit, rng):
    return PersonalityPlayer(rng, sample_personality_v2(pid, habit))


def opp_seat_rng(config_id, deal_seed, rotation, seat):
    return np.random.default_rng([config_id, deal_seed, rotation, seat])


def block_rng(row_name, block_idx):
    return np.random.default_rng([ROW_CONFIG_ID[row_name], block_idx])


# ------------------------------------------------------------------- nets
_NETS = {}


def nets():
    """Per-process lazy load of the two v2 nets, with the version tripwire."""
    if not _NETS:
        for path, want_param in ((generic_npz(), 0),
                                 (conditioned_npz(), PARAM_DIM_V2)):
            assert os.path.exists(path), (
                f"missing v2 profiler {path}; run "
                f"`train_profiler.py --v2 --full` first")
            _w, meta = load_npz(path)
            pv = meta.get("population_v", 1)
            assert int(pv) == 2, (
                f"{path} was trained on population_v={pv}; the curve must use "
                f"the v2 nets or it measures model staleness, not reading")
            if want_param:
                assert int(meta.get("param_dim", -1)) == want_param, (
                    f"{path}: param_dim={meta.get('param_dim')} != "
                    f"PARAM_DIM_V2={want_param}")
        _NETS["generic"] = load_profiler(generic_npz())[0]
        _NETS["conditioned"] = load_profiler(conditioned_npz())[0]
    return _NETS


# --------------------------------------------------------------- bot build
def choice_posterior_factory(epsilon):
    def factory(view, rng):
        return WeightedPosterior.from_view(
            view, Level.FULL, HeuristicPlayer(), epsilon, N_WORLDS, rng=rng,
            max_draws=CHOICE_MAX_DRAWS, keep_worlds=True)
    return factory


def build_bot(row_name, trio_ids, habit, rng):
    """-> (bot, per_game_hook(seat_to_pos)).

    Both profiled rows keep HEURISTIC playouts (`playout_weights=None`) so
    that ORACLE - FULL is reading value and nothing else (module docstring).
    """
    if row_name == "honest-FULL":
        return HonestSearchPlayer(Level.FULL, N_OUTER, N_INNER, rng,
                                  sampler_respects_voids=True,
                                  posterior_factory=None), (lambda s2p: None)

    if row_name == "honest-CHOICE-strict":
        return HonestSearchPlayer(
            Level.FULL, N_OUTER, N_INNER, rng, sampler_respects_voids=True,
            posterior_factory=choice_posterior_factory(CHOICE_STRICT_EPS)), \
            (lambda s2p: None)

    w = nets()
    if row_name == "profiled-R":
        lik = ProfilerLikelihood(w["generic"])
        pf = profiler_posterior_factory(lik, level=Level.FULL,
                                        n_worlds=N_WORLDS,
                                        max_draws=CHOICE_MAX_DRAWS,
                                        keep_worlds=True)
        return ProfiledSearchPlayer(Level.FULL, N_OUTER, N_INNER, rng,
                                    sampler_respects_voids=True,
                                    posterior_factory=pf,
                                    playout_weights=None), (lambda s2p: None)

    if row_name == "profiled-ORACLE":
        zeros = np.zeros(PARAM_DIM_V2, dtype=np.float64)
        lik = ProfilerLikelihood(w["conditioned"],
                                 seat_params={s: zeros for s in range(4)})
        # TRUE v2 params AT THIS SETTING: the ORACLE knows both the style axes
        # and how habitual this opponent is.
        identity_params = {p: param_vector_v2(trio_ids[p], habit)
                           for p in range(3)}

        def hook(seat_to_pos):
            lik.seat_params = {s: identity_params[p]
                               for s, p in seat_to_pos.items()}

        pf = profiler_posterior_factory(lik, level=Level.FULL,
                                        n_worlds=N_WORLDS,
                                        max_draws=CHOICE_MAX_DRAWS,
                                        keep_worlds=True)
        return ProfiledSearchPlayer(Level.FULL, N_OUTER, N_INNER, rng,
                                    sampler_respects_voids=True,
                                    posterior_factory=pf,
                                    playout_weights=None), hook

    raise ValueError(row_name)


def _counters(bot):
    return dict(fallbacks=bot.fallbacks, failed_samples=bot.failed_samples,
                inner_fallbacks=bot.inner_fallbacks,
                inner_failed_samples=bot.inner_failed_samples,
                posterior=[bot.posterior_collapses, bot.posterior_decisions,
                           bot.posterior_worlds])


# ------------------------------------------------------------ rare moments
QS = cards.QUEEN_SPADES
_ACE_SPADES = cards.SPADES * 13 + 12
_KING_SPADES = cards.SPADES * 13 + 11


def rare_flags(state, our_seat):
    """(qs_opp, qs_exposed) for one (deal, rotation), from the DEALT hands.

    Row-independent by construction -- see the module docstring's RARE MOMENTS
    section for why that is required rather than merely convenient.
    """
    ours = state.hands[our_seat]
    qs_opp = not bool(ours & cards.bit(QS))
    exposed = qs_opp and bool(ours & (cards.bit(_ACE_SPADES)
                                      | cards.bit(_KING_SPADES)))
    return bool(qs_opp), bool(exposed)


def slice_means(rot_scores, flags):
    """Per-deal means restricted to the rotations where `flags` holds.

    `rot_scores`/`flags` are [n_deals, 4].  Returns (values, mask) where mask
    marks deals with at least one qualifying rotation; deals with none are
    dropped rather than imputed.  The mask depends only on the deal seeds, so
    every row is restricted to exactly the same deals -- the pairing survives.
    """
    rot_scores = np.asarray(rot_scores, dtype=float)
    flags = np.asarray(flags, dtype=bool)
    n = flags.sum(axis=1)
    keep = n > 0
    vals = np.zeros(rot_scores.shape[0])
    vals[keep] = (rot_scores * flags)[keep].sum(axis=1) / n[keep]
    return vals, keep


# --------------------------------------------------------------- one block
def run_block(row_name, block_idx, trio_ids, habit, deal_seeds):
    """One MATCH: one fixed trio at one dial setting, deals x 4 rotations."""
    bot, hook = build_bot(row_name, trio_ids, habit,
                          block_rng(row_name, block_idx))
    n = len(deal_seeds)
    rot_scores = np.zeros((n, 4))
    qs_opp = np.zeros((n, 4), dtype=bool)
    qs_exposed = np.zeros((n, 4), dtype=bool)
    for i, seed in enumerate(deal_seeds):
        for rotation in range(4):
            state = deal(np.random.default_rng(seed))
            qs_opp[i, rotation], qs_exposed[i, rotation] = rare_flags(
                state, rotation)
            other_seats = sorted(s for s in range(4) if s != rotation)
            seat_to_pos = {s: p for p, s in enumerate(other_seats)}
            hook(seat_to_pos)
            players = [None, None, None, None]
            players[rotation] = bot
            for s, p in seat_to_pos.items():
                players[s] = personality_player(
                    trio_ids[p], habit,
                    opp_seat_rng(OPP_CONFIG_ID, seed, rotation, s))
            final = play_game(state, players)
            assert sum(final.scores) == 26, "engine invariant broken"
            rot_scores[i, rotation] = final.scores[rotation]
    return (rot_scores.mean(axis=1), rot_scores, qs_opp, qs_exposed,
            _counters(bot))


# ----------------------------------------------------------- guessing half
def guess_table_for_seed(seed):
    """4 DISTINCT held-out v2 ids; sampled order IS the seat order."""
    rng = np.random.default_rng([seed, TABLE_SALT])
    idx = rng.choice(len(_HELDOUT_V2), size=4, replace=False)
    return [int(_HELDOUT_V2[int(i)]) for i in idx]


def guess_play(seed, habit):
    pids = guess_table_for_seed(seed)
    players = [PersonalityPlayer(np.random.default_rng([seed, s, 0xA1CE]),
                                 sample_personality_v2(p, habit))
               for s, p in enumerate(pids)]
    state = deal(np.random.default_rng(seed))
    hands = list(state.hands)
    plays = []
    while not state.is_over():
        seat = state.to_play
        card = players[seat].choose(state.view_for(seat))
        plays.append((seat, card))
        state.play(card)
    assert sum(state.scores) == 26
    return hands, plays, pids


class _Rec:
    def __init__(self, seed, hands, plays, pids):
        self.seed, self.hands, self.plays, self.pids = seed, hands, plays, pids


def _guess_posterior(curve, view, rng, lik_generic, lik_oracle, policy):
    if curve == "FULL":
        return BeliefTable.from_view(view, Level.FULL)
    if curve == "CHOICE-strict":
        return WeightedPosterior.from_view(
            view, Level.FULL, policy, epsilon=0.0, n_worlds=GUESS_N_WORLDS,
            rng=rng, max_draws=GUESS_MAX_DRAWS)
    lik = lik_generic if curve == "PROFILER" else lik_oracle
    return profiler_posterior(view, Level.FULL, lik, GUESS_N_WORLDS, rng,
                              GUESS_MAX_DRAWS)


def _guess_metrics(probs, truth):
    ps, hits = [], []
    for c, i in truth.items():
        p = float(probs[i, c])
        assert np.isfinite(p) and p >= 0.0
        ps.append(p)
        hits.append(1.0 if int(np.argmax(probs[:, c])) == i else 0.0)
    ps = np.asarray(ps, dtype=float)
    return float(ps.mean()), float(np.mean(hits)), int((ps < 0.01).sum()), \
        int(ps.size)


def guess_game(seed, habit):
    """{curve: {trick: [per-observer dicts]}} for one held-out game."""
    hands, plays, pids = guess_play(seed, habit)
    rec = _Rec(seed, hands, plays, pids)
    w = nets()
    lik_generic = ProfilerLikelihood(w["generic"])
    lik_oracle = ProfilerLikelihood(
        w["conditioned"],
        {s: param_vector_v2(p, habit) for s, p in enumerate(pids)})
    policy = HeuristicPlayer()
    acc = {c: {t: [] for t in range(1, NUM_TRICKS + 1)} for c in CURVES}

    def boundary(state, trick):
        for seat in range(4):
            view = state.view_for(seat)
            for curve in CURVES:
                rng = np.random.default_rng(
                    int(np.random.default_rng(
                        [seed, trick, seat]).integers(0, 2 ** 63 - 1)))
                try:
                    post = _guess_posterior(curve, view, rng, lik_generic,
                                            lik_oracle, policy)
                except PosteriorCollapse:
                    acc[curve][trick].append({"collapsed": True})
                    continue
                truth = guessing._truth_for(seat, rec, post)
                mp, top1, n_lt01, n_cards = _guess_metrics(post.probs, truth)
                acc[curve][trick].append({
                    "collapsed": False, "meanP": mp, "top1": top1,
                    "n_lt01": n_lt01, "n_cards": n_cards})

    state = GameState(hands=list(hands))
    state.to_play = plays[0][0]
    boundary(state, 1)
    for seat, card in plays:
        assert seat == state.to_play, "replay desync"
        state.play(card)
        if not state.current_trick and state.trick_number < NUM_TRICKS:
            boundary(state, state.trick_number + 1)
    assert state.is_over()
    return acc


def guess_worker(args):
    seeds, habit = args
    acc = {c: {t: [] for t in range(1, NUM_TRICKS + 1)} for c in CURVES}
    for s in seeds:
        a = guess_game(s, habit)
        for c in CURVES:
            for t in acc[c]:
                acc[c][t].extend(a[c][t])
    return acc


def summarize_guessing(acc):
    """-> {curve: {meanP, top1, truth_lt01_frac, collapse_frac, per_trick}}."""
    out = {}
    for curve in CURVES:
        rows, per_trick = [], {}
        n_calls = n_coll = 0
        for trick in range(1, NUM_TRICKS + 1):
            st = acc[curve][trick]
            if not st:
                continue
            ok = [s for s in st if not s["collapsed"]]
            n_calls += len(st)
            n_coll += len(st) - len(ok)
            per_trick[trick] = {
                "meanP": float(np.mean([s["meanP"] for s in ok])) if ok
                else float("nan"),
                "collapse_frac": 1.0 - len(ok) / len(st),
                "n": len(st)}
            rows.extend(ok)
        n_cards = sum(s["n_cards"] for s in rows) or 1
        out[curve] = {
            "meanP": float(np.mean([s["meanP"] for s in rows])) if rows
            else float("nan"),
            "top1": float(np.mean([s["top1"] for s in rows])) if rows
            else float("nan"),
            "truth_lt01_frac": sum(s["n_lt01"] for s in rows) / n_cards,
            "collapse_frac": n_coll / max(1, n_calls),
            "per_trick": per_trick}
    return out


# ------------------------------------------------------- measured entropy
def measured_entropy(habit, n_games=6, n_personalities=25):
    """Mean per-decision choice entropy (nats) of the HELD-OUT v2 population
    at this setting -- the curve's x-axis.

    Same closed form as Task A1's report (`tests/test_habit_dial.py`):
    H[eps/n + (1-eps)*softmax(scores/T)], averaged over real decision views.
    Recomputed here on the HELD-OUT ids (A1 measured a different id set), so
    the x-value names the population the curve was actually measured against.
    """
    # THE DRIVER IS PINNED AT H2 at every setting, matching Task A1's
    # measurement exactly.  If the driver played at `habit`, H0's entropy
    # would be measured over H0-generated positions and H3's over
    # H3-generated ones -- conflating the dial with the positions the dial
    # produces, and breaking comparability with A1's published
    # 0.648/0.898/1.119/1.246.  Only the SCORED personalities move.
    views = []
    driver = PersonalityPlayer(np.random.default_rng(0),
                               sample_personality_v2(_HELDOUT_V2[0], "H2"))
    for g in range(n_games):
        seed = GUESS_SEED_BASE + 900_000 + g   # disjoint from the panel seeds
        state = deal(np.random.default_rng(seed))
        while not state.is_over():
            seat = state.to_play
            view = state.view_for(seat)
            if len(cards.cards_in(view.legal_moves)) > 1:
                views.append(view)
            state.play(driver.choose(view))
    tot, n = 0.0, 0
    for pid in _HELDOUT_V2[:n_personalities]:
        p = sample_personality_v2(pid, habit)
        pl = PersonalityPlayer(np.random.default_rng(0), p)
        for v in views:
            legal = cards.cards_in(v.legal_moves)
            k = len(legal)
            s = pl._scores(v, legal)
            w = np.exp((s - s.max()) / p.temperature)
            q = p.epsilon / k + (1.0 - p.epsilon) * (w / w.sum())
            tot += float(-np.sum(np.where(q > 0.0, q * np.log(
                np.where(q > 0.0, q, 1.0)), 0.0)))
            n += 1
    return tot / max(1, n), len(views), min(n_personalities, len(_HELDOUT_V2))


# ------------------------------------------------------------ checkpointing
def _append_partial(path, name, block_idx, per_deal, rot, qs_opp, qs_exp, cnt):
    """Bank one completed chunk (JSON-v2, append-only -- never truncate)."""
    payload = {"values": [float(v) for v in per_deal],
               "rot": [[float(x) for x in r] for r in rot],
               "qs_opp": [[bool(x) for x in r] for r in qs_opp],
               "qs_exposed": [[bool(x) for x in r] for r in qs_exp],
               "cnt": cnt}
    with open(path, "a") as f:
        f.write(f"{name}@{block_idx} J{json.dumps(payload)}\n")
        f.flush()


def _load_partial(path):
    banked = {}
    if not os.path.exists(path):
        return banked
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            head, rest = line.split(" ", 1)
            name, b = head.rsplit("@", 1)
            if name not in ROW_NAMES or not rest.startswith("J{"):
                continue
            banked[(name, int(b))] = json.loads(rest[1:])
    return banked


def total_rss_gb():
    out = subprocess.run(["ps", "-o", "rss=", "-g", str(os.getpgrp())],
                         capture_output=True, text=True).stdout
    return sum(int(x) for x in out.split()) / (1024 ** 2)


def worker(args):
    row_name, block_idx, trio_ids, habit, deal_seeds = args
    per_deal, rot, qs_opp, qs_exp, cnt = run_block(
        row_name, block_idx, trio_ids, habit, deal_seeds)
    return row_name, block_idx, per_deal, rot, qs_opp, qs_exp, cnt


def run_ablation(habit, n_deals, workers, trios, path):
    n_blocks = n_deals // BLOCK_SIZE
    res = {n: {"values": np.zeros(n_deals),
               "rot": np.zeros((n_deals, 4))} for n in ROW_NAMES}
    qs_opp = np.zeros((n_deals, 4), dtype=bool)
    qs_exp = np.zeros((n_deals, 4), dtype=bool)
    counters = {n: [] for n in ROW_NAMES}

    def bank(name, b, values, rot, o, e, cnt):
        sl = slice(b * BLOCK_SIZE, (b + 1) * BLOCK_SIZE)
        res[name]["values"][sl] = values
        res[name]["rot"][sl] = rot
        qs_opp[sl] = o
        qs_exp[sl] = e
        if cnt is not None:
            counters[name].append(cnt)

    banked = _load_partial(path)
    for (name, b), p in banked.items():
        if b >= n_blocks or len(p["values"]) != BLOCK_SIZE:
            continue
        bank(name, b, p["values"], p["rot"], p["qs_opp"], p["qs_exposed"],
             p.get("cnt"))
    if banked:
        print(f"[resume] {len(banked)} banked chunks loaded from {path}; "
              f"skipping them", flush=True)

    jobs = []
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        for name in ROW_NAMES:
            for b in range(n_blocks):
                if (name, b) in banked:
                    continue
                seeds = [DEAL_SEED_BASE + b * BLOCK_SIZE + d
                         for d in range(BLOCK_SIZE)]
                jobs.append(pool.submit(
                    worker, (name, b, trios[b], habit, seeds)))
        done, t0 = 0, time.time()
        for fut in cf.as_completed(jobs):
            name, b, per_deal, rot, o, e, cnt = fut.result()
            bank(name, b, per_deal, rot, o, e, cnt)
            _append_partial(path, name, b, per_deal, rot, o, e, cnt)
            done += 1
            mem = total_rss_gb()
            print(f"[{done}/{len(jobs)}] {habit} {name} block {b} | "
                  f"mem={mem:.1f}GB | {time.time() - t0:.0f}s", flush=True)
            if mem > MEM_LIMIT_GB:
                for j in jobs:
                    j.cancel()
                raise MemoryError(f"memory {mem:.1f}GB over {MEM_LIMIT_GB}GB")
    return res, qs_opp, qs_exp, counters


# ------------------------------------------------------------------- probe
def timing_probe(habit, n_blocks=2, n_deals_per_block=3):
    trios = block_trios(n_blocks, habit)
    s_per_game = {}
    for name in ROW_NAMES:
        t0, n_games = time.time(), 0
        for b in range(n_blocks):
            seeds = [DEAL_SEED_BASE + b * BLOCK_SIZE + d
                     for d in range(n_deals_per_block)]
            run_block(name, b, trios[b], habit, seeds)
            n_games += 4 * len(seeds)
        spg = (time.time() - t0) / n_games
        s_per_game[name] = spg
        print(f"[probe {habit}] {name}: {spg:.3f}s/game", flush=True)
    t0 = time.time()
    guess_game(GUESS_SEED_BASE, habit)
    s_guess = time.time() - t0
    print(f"[probe {habit}] guessing: {s_guess:.2f}s/game", flush=True)
    return s_per_game, s_guess


def print_budget_projection(s_per_game, s_guess, n_deals, n_guess, workers):
    games = 4 * n_deals
    total = sum(spg * games for spg in s_per_game.values()) + s_guess * n_guess
    wall = total / workers
    print("=" * 72, flush=True)
    print(f"[BUDGET] one SETTING at {n_deals} deals + {n_guess} guessing "
          f"games = {total:.0f} worker-seconds ({total/3600:.2f}h serial)",
          flush=True)
    print(f"[BUDGET] DIVIDED BY {workers} workers -> wall clock "
          f"{wall:.0f}s ({wall/3600:.2f}h) vs {BUDGET_SECONDS/3600:.1f}h cap "
          f"-> {'WITHIN' if wall <= BUDGET_SECONDS else 'OVER'} budget",
          flush=True)
    print(f"[BUDGET] all four settings, run separately: "
          f"{4 * wall/3600:.2f}h total wall clock", flush=True)
    print("=" * 72, flush=True)
    return total, wall


# ------------------------------------------------------------- per setting
def analyse(habit, res, qs_opp, qs_exp, guess, counters, entropy, n_deals,
            trios, elapsed, workers, smoke):
    base = res[BASELINE_ROW]["values"]
    base_rot = res[BASELINE_ROW]["rot"]
    out = {"habit": habit, "n_deals": int(n_deals), "smoke": bool(smoke),
           "entropy_nats": entropy[0], "entropy_views": entropy[1],
           "entropy_personalities": entropy[2],
           "deal_seed_base": DEAL_SEED_BASE, "guess_seed_base": GUESS_SEED_BASE,
           "trios": [list(t) for t in trios], "rows": {}, "diffs": {},
           "slices": {}, "guessing": guess, "wall_time_s": elapsed,
           "workers": workers, "generic_npz": generic_npz(),
           "conditioned_npz": conditioned_npz()}

    for name in ROW_NAMES:
        m, lo, hi = bootstrap_ci(res[name]["values"])
        cl = counters[name]
        out["rows"][name] = {
            "mean": m, "lo": lo, "hi": hi,
            "fallbacks": int(sum(c["fallbacks"] for c in cl)) if cl else None,
            "posterior_collapses": (int(sum(c["posterior"][0] for c in cl))
                                    if cl else None),
            "posterior_decisions": (int(sum(c["posterior"][1] for c in cl))
                                    if cl else None)}
        if name != BASELINE_ROW:
            # POSITIVE = the row BEATS honest-FULL (FULL's points minus the
            # row's), i.e. "reading value in FULL's favour being beaten".
            d = base - res[name]["values"]
            dm, dlo, dhi = bootstrap_ci(d)
            out["diffs"][name] = {"mean": dm, "lo": dlo, "hi": dhi}

    for sname, flags in (("QS-OPP", qs_opp), ("QS-EXPOSED", qs_exp)):
        b, keep = slice_means(base_rot, flags)
        entry = {"n_deals_kept": int(keep.sum()),
                 "deal_retention": float(keep.mean()),
                 "rotation_retention": float(np.asarray(flags).mean()),
                 "rows": {}}
        for name in ROW_NAMES:
            v, _k = slice_means(res[name]["rot"], flags)
            entry["rows"][name] = float(np.mean(v[keep])) if keep.any() \
                else float("nan")
            if name != BASELINE_ROW and keep.any():
                dm, dlo, dhi = bootstrap_ci((b - v)[keep])
                entry.setdefault("diffs", {})[name] = {
                    "mean": dm, "lo": dlo, "hi": dhi}
        out["slices"][sname] = entry
    return out


def write_setting(out, path):
    h = out["habit"]
    L = []
    A = L.append
    A(f"open-hearts Phase 6 Task A2: THE ENTROPY CURVE -- setting {h}")
    A("Lower points per hand is better; 6.5 is the symmetric break-even. "
      "The BASELINE is honest-FULL (Phase 5's true incumbent, 2.345).")
    A("")
    if out["smoke"]:
        A("*** SMOKE RUN -- plumbing only. The numbers below are NOT a "
          "result and no pre-registered expectation is judged on them. ***")
        A("")
    A(f"measured mean per-decision choice entropy at {h}: "
      f"{out['entropy_nats']:.4f} nats "
      f"({out['entropy_personalities']} HELD-OUT v2 personalities x "
      f"{out['entropy_views']} real decision views generated by a driver "
      f"PINNED AT H2 at every setting, so only the scored dial moves; "
      f"closed-form mixture density, same formula as Task A1's report)")
    A(f"deals: {out['n_deals']} (seeds {DEAL_SEED_BASE}.."
      f"{DEAL_SEED_BASE + out['n_deals'] - 1}) in "
      f"{out['n_deals'] // BLOCK_SIZE} match-blocks of {BLOCK_SIZE} vs one "
      f"fixed HELD-OUT v2 trio; 4 rotations per deal; identical deals, blocks "
      f"and opponent streams across every row. THE SAME deal block and the "
      f"SAME trios are used at all four settings, so the curve is paired on "
      f"card luck AND on personality.")
    A(f"block size (deals per match): {BLOCK_SIZE}")
    A(f"search: n_outer={N_OUTER} n_inner={N_INNER} n_worlds={N_WORLDS} "
      f"max_draws={CHOICE_MAX_DRAWS}; CHOICE-strict eps={CHOICE_STRICT_EPS}")
    A(f"opponent config_id (shared by all rows): {OPP_CONFIG_ID}; bot rng "
      f"seeded per block from (row config id, block index)")
    A(f"v2 nets: GENERIC={os.path.relpath(out['generic_npz'], ROOT)} "
      f"CONDITIONED={os.path.relpath(out['conditioned_npz'], ROOT)} "
      f"(population_v=2 asserted on load)")
    A("BOTH profiled rows use HEURISTIC playouts, unlike run_ablation5's "
      "ORACLE row: mixing playout variants would confound reading value with "
      "Organ-2 playout value, which is exactly what this curve isolates.")
    A(f"CIs: 10,000-resample bootstrap over DEALS. wall={out['wall_time_s']:.0f}s "
      f"with {out['workers']} worker(s).")
    A("")
    A("trio table (block: held-out v2 personality ids):")
    for b, t in enumerate(out["trios"]):
        A(f"  block {b:2d}: {tuple(t)}")
    A("")
    A(f"{'row':<24}{'mean':>8}{'lo95':>9}{'hi95':>9}{'fallbk':>8}"
      f"{'pcollapse':>11}{'pdecisions':>12}")
    for name in ROW_NAMES:
        r = out["rows"][name]
        A(f"{name:<24}{r['mean']:>8.3f}{r['lo']:>9.3f}{r['hi']:>9.3f}"
          f"{str(r['fallbacks']):>8}{str(r['posterior_collapses']):>11}"
          f"{str(r['posterior_decisions']):>12}")
    A("")
    A("READING VALUE: paired per-deal (honest-FULL - row). POSITIVE means the "
      "row BEATS honest-FULL by that many points per hand.")
    A(f"{'row':<24}{'value':>9}{'lo95':>9}{'hi95':>9}")
    for name in READING_ROWS:
        d = out["diffs"][name]
        A(f"{name:<24}{d['mean']:>9.3f}{d['lo']:>9.3f}{d['hi']:>9.3f}")
    A("")
    A("RARE-MOMENT CONDITIONALS (row-independent slices of (deal, rotation); "
      "means over qualifying rotations only, deals with none dropped):")
    for sname, e in out["slices"].items():
        A(f"  {sname}: rotations retained {e['rotation_retention']*100:.1f}%, "
          f"deals kept {e['n_deals_kept']} of {out['n_deals']} "
          f"({e['deal_retention']*100:.1f}%)")
        A(f"    {'row':<24}{'mean':>8}{'value vs FULL':>15}{'lo95':>9}"
          f"{'hi95':>9}")
        for name in ROW_NAMES:
            v = e["rows"][name]
            d = None if name == BASELINE_ROW else e.get("diffs", {}).get(name)
            if name == BASELINE_ROW:
                A(f"    {name:<24}{v:>8.3f}{'(baseline)':>15}")
            elif d is None:
                A(f"    {name:<24}{v:>8.3f}{'n/a':>15}")
            else:
                A(f"    {name:<24}{v:>8.3f}{d['mean']:>15.3f}"
                  f"{d['lo']:>9.3f}{d['hi']:>9.3f}")
    A("  QS-OPP is the plan's literal pre-registration ('the Q(spades) sat in "
      "an opponent hand at our first Q-relevant decision'); its retention is "
      "high by construction (the bot holds the queen in exactly one rotation "
      "of four), which is why the retention line is printed beside it. "
      "QS-EXPOSED is a documented SHARPENING, not a pre-registration: the "
      "queen is an opponent's AND we hold the A/K of spades, the spot where "
      "knowing who holds it changes our play.")
    A("  Per-decision divergence weighting: assessed, NOT feasible cheaply -- "
      "the candidate values that would define divergence are computed and "
      "discarded inside the search hot path, and they are row-dependent, so "
      "weighting by them would break the paired comparison. Recorded as "
      "infeasible rather than approximated.")
    A("")
    A(f"GUESSING PANEL ({GUESS_GAMES if not out['smoke'] else '(smoke)'} "
      f"held-out v2 games, seeds {GUESS_SEED_BASE}+, n_worlds="
      f"{GUESS_N_WORLDS}): mean P(truth) over all truth cards, top-1, the "
      f"fraction of truth cards given p<0.01, and collapse rate.")
    A(f"  {'curve':<18}{'meanP':>9}{'top1':>9}{'truth<.01':>11}"
      f"{'collapse':>10}")
    for c in CURVES:
        g = out["guessing"][c]
        A(f"  {c:<18}{g['meanP']:>9.4f}{g['top1']:>9.4f}"
          f"{g['truth_lt01_frac']:>11.4f}{g['collapse_frac']:>10.4f}")
    A("  CHOICE-strict's meanP is SURVIVOR-BIASED (its true world dies the "
      "moment an opponent departs from the script, so only script-like "
      "boundaries survive to be scored) -- judge it by collapse_frac.")
    txt = "\n".join(L) + "\n"
    with open(path, "w") as f:
        f.write(txt)
    print(txt, flush=True)


# ----------------------------------------------------------------- report
def build_report(smoke=False):
    """Merge the four per-setting JSONs into THE CURVE + the verdict."""
    data = {}
    for h in HABIT_ORDER:
        p = json_path(h, smoke)
        if os.path.exists(p):
            with open(p) as f:
                data[h] = json.load(f)
    if not data:
        print("no per-setting JSONs found; run each setting first")
        return
    have = [h for h in HABIT_ORDER if h in data]
    L = []
    A = L.append
    A("open-hearts Phase 6 Task A2: THE ENTROPY CURVE")
    A("How many points is READING opponents worth, as a function of how "
      "predictable they are?")
    A("")
    if smoke or any(data[h]["smoke"] for h in have):
        A("*** SMOKE INPUTS -- plumbing only, not a result. ***")
        A("")
    A(f"settings present: {have} (of {list(HABIT_ORDER)})")
    A(f"deals per setting: {data[have[0]]['n_deals']} x 4 rotations, "
      f"seeds {DEAL_SEED_BASE}+; SAME deals, SAME held-out trios and SAME "
      f"opponent config at every setting -- the curve is paired on card luck "
      f"and on personality, so only predictability moves.")
    A(f"v2 profiler nets: "
      f"{os.path.relpath(data[have[0]]['generic_npz'], ROOT)} / "
      f"{os.path.relpath(data[have[0]]['conditioned_npz'], ROOT)}")
    A("Reading value = paired per-deal (honest-FULL - row); POSITIVE means "
      "the reading row BEATS honest-FULL.")
    A("")
    A(f"{'setting':<9}{'entropy':>9}{'FULL':>9}{'strict':>9}{'R':>9}"
      f"{'ORACLE':>9}   {'R-value':>9}{'ORACLE-value':>14}{'lo95':>9}"
      f"{'hi95':>9}")
    for h in have:
        d = data[h]
        r = d["rows"]
        A(f"{h:<9}{d['entropy_nats']:>9.3f}"
          f"{r['honest-FULL']['mean']:>9.3f}"
          f"{r['honest-CHOICE-strict']['mean']:>9.3f}"
          f"{r['profiled-R']['mean']:>9.3f}"
          f"{r['profiled-ORACLE']['mean']:>9.3f}   "
          f"{d['diffs']['profiled-R']['mean']:>9.3f}"
          f"{d['diffs']['profiled-ORACLE']['mean']:>14.3f}"
          f"{d['diffs']['profiled-ORACLE']['lo']:>9.3f}"
          f"{d['diffs']['profiled-ORACLE']['hi']:>9.3f}")
    A("")
    A("rare-moment reading value (ORACLE, paired, within slice):")
    A(f"{'setting':<9}{'overall':>9}{'QS-OPP':>9}{'QS-EXPOSED':>12}"
      f"{'kept deals (QS-EXPOSED)':>26}")
    for h in have:
        d = data[h]
        so = d["slices"]["QS-OPP"].get("diffs", {}).get("profiled-ORACLE")
        se = d["slices"]["QS-EXPOSED"].get("diffs", {}).get("profiled-ORACLE")
        A(f"{h:<9}{d['diffs']['profiled-ORACLE']['mean']:>9.3f}"
          f"{(so or {}).get('mean', float('nan')):>9.3f}"
          f"{(se or {}).get('mean', float('nan')):>12.3f}"
          f"{d['slices']['QS-EXPOSED']['n_deals_kept']:>26}")
    A("")
    A("guessing panel (mean P(truth) / CHOICE-strict collapse rate):")
    A(f"{'setting':<9}{'FULL':>9}{'strict':>9}{'PROFILER':>10}{'ORACLE':>9}"
      f"{'strict collapse':>17}")
    for h in have:
        g = data[h]["guessing"]
        A(f"{h:<9}{g['FULL']['meanP']:>9.4f}"
          f"{g['CHOICE-strict']['meanP']:>9.4f}"
          f"{g['PROFILER']['meanP']:>10.4f}"
          f"{g['PROFILER-ORACLE']['meanP']:>9.4f}"
          f"{g['CHOICE-strict']['collapse_frac']:>17.4f}")
    A("")
    A("#" * 74)
    A("## PRE-REGISTERED EXPECTATIONS (PHASE6_PLAN Task A2 + the lead's "
      "additions), stated VERBATIM")
    verdicts = _judge(data, have)
    for text, ok, note in verdicts:
        A(f"##  [{'PASS' if ok else 'FAIL' if ok is False else 'N/A '}] {text}")
        A(f"##        {note}")
    A("#" * 74)
    A("")
    A("## DECISION RULE (pre-registered, PHASE6_PLAN Task A2)")
    A("##  reading value at H0-H1 >= ~0.3 pts/hand -> reading machinery LIVES")
    A("##  reading value < 0.3 EVERYWHERE, rare-moment slices included -> "
      "reading machinery is RETIRED from the main line (archived, not "
      "deleted)")
    A(_verdict_line(data, have))
    txt = "\n".join(L) + "\n"
    out_txt = os.path.join(RESULTS,
                           "entropy_curve_smoke.txt" if smoke
                           else "entropy_curve.txt")
    with open(out_txt, "w") as f:
        f.write(txt)
    print(txt, flush=True)
    _plot(data, have, os.path.join(RESULTS, "entropy_curve_smoke.png" if smoke
                                   else "entropy_curve.png"))
    print(f"wrote {out_txt}", flush=True)


def _judge(data, have):
    out = []
    order = [h for h in ("H3", "H2", "H1", "H0") if h in data]
    vals = [data[h]["diffs"]["profiled-ORACLE"]["mean"] for h in order]
    if len(order) < 4:
        out.append((PRE_REGISTERED[0], None,
                    f"only {have} present; monotonicity needs all four"))
    else:
        ok = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
        out.append((PRE_REGISTERED[0], ok, "H3->H0 reading value: " +
                    " -> ".join(f"{h}:{v:+.3f}" for h, v in zip(order, vals))))

    if "H0" in data:
        d = data["H0"]["diffs"]["profiled-ORACLE"]
        out.append((PRE_REGISTERED[1], d["mean"] >= READING_LIVES_THRESHOLD,
                    f"H0 ORACLE value {d['mean']:+.3f} pts/hand, CI "
                    f"({d['lo']:+.3f}, {d['hi']:+.3f}); threshold "
                    f"+{READING_LIVES_THRESHOLD}"))
    else:
        out.append((PRE_REGISTERED[1], None, "H0 not present"))

    if "H2" in data:
        d = data["H2"]["diffs"]["profiled-ORACLE"]
        near = abs(d["mean"]) < READING_LIVES_THRESHOLD
        out.append((PRE_REGISTERED[2], near,
                    f"H2 ORACLE value {d['mean']:+.3f} (Phase 5's comparable "
                    f"number was ORACLE 2.393 vs FULL 2.345, i.e. -0.048 in "
                    f"this sign convention). Phase 5's ORACLE carried "
                    f"GENERIC playouts and this one carries heuristic "
                    f"playouts; the comparison is licensed by Phase 5's own "
                    f"follow-up row FULL-profiled-playouts = 2.354, +0.009, "
                    f"CI (-0.181, +0.196) -- playouts contribute ~0 on top of "
                    f"FULL. |value| < {READING_LIVES_THRESHOLD} read as "
                    f"'near 0'. A material deviation is a finding about the "
                    f"v2 contextual axes, not a failure of the harness."))
    else:
        out.append((PRE_REGISTERED[2], None, "H2 not present"))

    coll = {h: data[h]["guessing"]["CHOICE-strict"]["collapse_frac"]
            for h in have}
    if len(have) == 4:
        high = coll["H2"] > 0.5 and coll["H3"] > 0.5
        falls = all(coll[a] <= coll[b] for a, b in
                    (("H0", "H1"), ("H1", "H2"), ("H2", "H3")))
        out.append((PRE_REGISTERED[3], bool(high and falls),
                    "CHOICE-strict collapse fraction " +
                    " ".join(f"{h}:{coll[h]:.3f}" for h in HABIT_ORDER) +
                    f"; 'collapses at H2-H3' read as >0.5 ({high}), 'falls as "
                    f"habit strengthens' as monotone H0<=H1<=H2<=H3 "
                    f"({falls}). PRE-REGISTERED READING, written before the "
                    f"run: this expectation may be structurally FALSE, "
                    f"because CHOICE-strict collapses on inconsistency with "
                    f"HeuristicPlayer, not on unpredictability. An H0 "
                    f"personality is a near-argmax of ITS OWN scorer -- "
                    f"including the three contextual axes A1 added precisely "
                    f"because the beginner script cannot predict them -- so "
                    f"it contradicts the script MORE reliably than a noisy "
                    f"player, who occasionally stumbles onto the heuristic's "
                    f"card by chance. Habitual is not the same as "
                    f"script-like. A FAIL here is that fact, not a harness "
                    f"defect."))
    else:
        out.append((PRE_REGISTERED[3], None, f"needs all four; have {have}"))

    rows = []
    any_bigger = False
    for h in have:
        o = data[h]["diffs"]["profiled-ORACLE"]["mean"]
        s = data[h]["slices"]["QS-EXPOSED"].get("diffs", {}).get(
            "profiled-ORACLE", {}).get("mean", float("nan"))
        q = data[h]["slices"]["QS-OPP"].get("diffs", {}).get(
            "profiled-ORACLE", {}).get("mean", float("nan"))
        rows.append(f"{h}: overall {o:+.3f} | QS-OPP {q:+.3f} | "
                    f"QS-EXPOSED {s:+.3f}")
        if s == s and s > o:
            any_bigger = True
    out.append((PRE_REGISTERED[4], any_bigger,
                "; ".join(rows) + " (PASS = the sharpened slice exceeds the "
                "overall mean at any setting)"))
    return out


def _verdict_line(data, have):
    """The decision rule keys on H0/H1 -- overall OR either rare-moment
    slice, since the rule retires reading only if it is below threshold
    'everywhere, including rare-moment slices'."""
    best = float("-inf")
    for h in have:
        if h not in ("H0", "H1"):
            continue
        cands = [data[h]["diffs"]["profiled-ORACLE"]["mean"]]
        for s in ("QS-OPP", "QS-EXPOSED"):
            d = data[h]["slices"][s].get("diffs", {}).get("profiled-ORACLE")
            if d:
                cands.append(d["mean"])
        best = max(best, max(cands))
    if best == float("-inf"):
        return "##  VERDICT: neither H0 nor H1 is present -- no verdict."
    lives = best >= READING_LIVES_THRESHOLD
    return (f"##  VERDICT: best H0/H1 reading value (overall or rare-moment) "
            f"= {best:+.3f} pts/hand -> reading "
            f"{'LIVES' if lives else 'RETIRED (pending owner review)'}"
            + ("" if len(have) == 4 else
               "   [PROVISIONAL: not all four settings present]"))


def _plot(data, have, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    xs = [data[h]["entropy_nats"] for h in have]
    ax = axes[0]
    for row, style in (("profiled-ORACLE", "-o"), ("profiled-R", "-s"),
                       ("honest-CHOICE-strict", "-^")):
        ys = [data[h]["diffs"][row]["mean"] for h in have]
        lo = [data[h]["diffs"][row]["mean"] - data[h]["diffs"][row]["lo"]
              for h in have]
        hi = [data[h]["diffs"][row]["hi"] - data[h]["diffs"][row]["mean"]
              for h in have]
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=style, capsize=4, label=row)
    ax.axhline(0.0, color="grey", lw=1)
    ax.axhline(READING_LIVES_THRESHOLD, color="green", ls="--", lw=1,
               label=f"decision rule {READING_LIVES_THRESHOLD}")
    for h, x in zip(have, xs):
        ax.annotate(h, (x, 0.0), textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=8)
    ax.set_xlabel("measured mean per-decision choice entropy (nats) "
                  "— left = habitual, right = whimsical")
    ax.set_ylabel("reading value (pts/hand vs honest-FULL, + = better)")
    ax.set_title("THE CURVE: what reading is worth vs opponent predictability")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for c in CURVES:
        ax.plot(xs, [data[h]["guessing"][c]["meanP"] for h in have],
                "-o", label=c)
    ax.set_xlabel("entropy (nats)")
    ax.set_ylabel("mean P(truth)")
    ax.set_title("guessing panel vs held-out v2 personalities")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


# -------------------------------------------------------------------- main
def run_setting(habit, n_deals, workers, n_guess, smoke):
    if not smoke:
        # THE GUARD (added after review): `--smoke` switches the OUTPUT paths
        # via the CLI flag but the NETS via the environment, and the two are
        # otherwise unrelated.  A leftover OPENHEARTS_CURVE_SMOKE=1 in the
        # shell -- easy, since the probe command sets it -- would produce a
        # phase-defining number from a 2-epoch throwaway net and write it to
        # the REAL output file.  `population_v` cannot catch it: the smoke net
        # carries population_v=2 too.
        assert "_smoke" not in generic_npz(), (
            "OPENHEARTS_CURVE_SMOKE is set but this is a REAL run: it would "
            "score the smoke-trained nets and write them to the real output. "
            "Unset the variable and re-run.")
    path = partial_path(habit, smoke)
    trios = block_trios(n_deals // BLOCK_SIZE, habit)
    if os.path.exists(path) and _load_partial(path):
        with open(path, "a") as f:                       # never truncate
            f.write(f"# resume {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    else:
        with open(path, "w") as f:
            f.write(f"# {habit} run start "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    t0 = time.time()
    res, qs_opp, qs_exp, counters = run_ablation(habit, n_deals, workers,
                                                 trios, path)
    seeds = [GUESS_SEED_BASE + i for i in range(n_guess)]
    chunks = [(seeds[i:i + 5], habit) for i in range(0, len(seeds), 5)]
    acc = {c: {t: [] for t in range(1, NUM_TRICKS + 1)} for c in CURVES}
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        for a in pool.map(guess_worker, chunks):
            for c in CURVES:
                for t in acc[c]:
                    acc[c][t].extend(a[c][t])
    guess = summarize_guessing(acc)
    ent = measured_entropy(habit, n_games=2 if smoke else 6,
                           n_personalities=5 if smoke else 25)
    elapsed = time.time() - t0
    out = analyse(habit, res, qs_opp, qs_exp, guess, counters, ent, n_deals,
                  trios, elapsed, workers, smoke)
    with open(json_path(habit, smoke), "w") as f:
        json.dump(out, f, indent=2)
    write_setting(out, setting_txt_path(habit, smoke))
    print(f"[done] {habit} in {elapsed/60:.1f} min with {workers} workers; "
          f"wrote {json_path(habit, smoke)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--habit", choices=list(HABIT_ORDER))
    ap.add_argument("--smoke", action="store_true",
                    help="1 block x 3 deals + 2 guessing games per setting, "
                         "own output paths; touches no real checkpoint")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--deals", type=int, default=N_DEALS_TARGET)
    ap.add_argument("--guess-games", type=int, default=GUESS_GAMES)
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    if args.report:
        build_report(smoke=args.smoke)
        return

    if args.probe:
        habit = args.habit or "H3"   # H3 is the timing worst case: the most
                                     # collapse, so CHOICE-strict spends the
                                     # most draws hunting survivors
        s_per_game, s_guess = timing_probe(habit)
        print_budget_projection(s_per_game, s_guess, args.deals,
                                args.guess_games, args.workers)
        print("[probe] done (no full run performed)", flush=True)
        return

    if args.smoke:
        os.environ["OPENHEARTS_CURVE_SMOKE"] = "1"
        os.environ["OPENHEARTS_CURVE_BLOCK"] = str(SMOKE_BLOCK_SIZE)
        globals()["BLOCK_SIZE"] = SMOKE_BLOCK_SIZE
        for habit in ([args.habit] if args.habit else list(HABIT_ORDER)):
            run_setting(habit, SMOKE_BLOCK_SIZE, min(4, args.workers),
                        SMOKE_GUESS_GAMES, smoke=True)
        build_report(smoke=True)
        return

    if not args.habit:
        ap.error("pass --habit H0|H1|H2|H3 (one background run per setting), "
                 "or --smoke / --probe / --report")
    s_per_game, s_guess = timing_probe(args.habit)
    _t, wall = print_budget_projection(s_per_game, s_guess, args.deals,
                                       args.guess_games, args.workers)
    if wall > BUDGET_SECONDS:
        print("STOP: this setting is projected over the 2h budget. Consult "
              "the owner. Nothing has been run.", flush=True)
        return
    run_setting(args.habit, args.deals, args.workers, args.guess_games,
                smoke=False)


if __name__ == "__main__":
    main()
