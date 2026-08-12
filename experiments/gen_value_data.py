"""Phase 4 Task 2: self-play position/outcome shard generator.

Plays games with a pre-registered mix, deterministically assigned by deal
seed: seed % 5 < 3 (60%) -> four HeuristicPlayers; else (40%) -> four
RandomizedHeuristic(epsilon=0.1). Deal seeds are 500000+i -- a fresh range,
never overlapping the 100000+ ablation seeds (train/eval deal separation is a
hard rule, see PHASE4_PLAN.md Global Constraints).

At EVERY ply of every game (52 plies/game: one per card play), for EACH of
the 4 seats, this records:
  - features: `openhearts.engine.features.featurize`'s output for the
    position *before* that ply's card is played, rotated to that seat.
  - target: remaining points that seat takes from this ply to the end of the
    hand = final hand score[seat] - score[seat] at this ply. This is the
    actual outcome, not an estimate -- computed once the hand is over.
For a fixed (game, ply), the 4 seats' targets sum to 26 minus points already
taken at that ply (checked as a smoke assertion below).

Efficiency discipline (per PHASE4_PLAN.md): games are played with the
existing Python game loop (players are Python objects choosing via
PlayerView, same as every other experiment script) -- but featurization is
batched. Each game's 52 pre-play position snapshots are collected into flat
numpy arrays first, then `features.featurize_batch` is called ONCE per game,
producing all 52*4 rows' feature vectors in a single dispatch (one
Python/numba-boundary crossing per game, not one per ply per seat). See
`src/openhearts/engine/features.py`'s `featurize_batch`, added here per Task
1's implementation note ("an out-parameter variant... is acceptable").

Storage: float16 features (every value lives in [0, 1]; float16 has ~3
decimal digits of precision, ample) + float32 targets, in ~100k-row `.npz`
shards under `results/value_data/` (gitignored). Split BY GAME, deterministic
from the deal seed: bucket = seed % 100; <90 train, <95 val, else test. Split
is SHARD-per-split, not row-per-split: a `ShardWriter` is dedicated to one
split and only ever flushes rows from that split, so the split is fully
determined by the filename (`value_{split}_{idx}.npz`) and echoed in
`meta["split"]` -- it is deliberately NOT duplicated as a per-row column
(that would cost ~20 bytes/row for pure redundancy; the full run is
size-budget-tight against the ~6GB ceiling, see the manifest).

Each shard's .npz contains:
  features    float16[N, NF]
  targets     float32[N]
  game_ids    int32[N]   (the deal seed)
  seat_index  int8[N]    (absolute seat 0..3, i.e. whose points `targets` is)
  ply_index   int8[N]    (0..51, the ply within the game)
  meta        0-d object array: dict with seed range, player-mix rule,
              split rule, FEATURES_V, generator git hash, and this shard's
              own split label.

Task 6.5 (distillation retry): `--target playout`
--------------------------------------------------
`--target outcome` (the default) is the behaviour described above and is
byte-identical to the pre-6.5 script. `--target playout` changes ONLY the
labels: same seeds, same player mix, same split rule, same schema, same
features. Each ply's 4 labels become the result of ONE
`kernel.playout_to_end` call from that ply's position -- the playout's final
scores minus the scores at that ply, per ABSOLUTE seat, exactly the
convention the outcome labels use (rotation happens at load time in
`train_value.rotated_targets`). Output goes to `results/value_data_playout/`
and the manifest + shard meta carry `target=playout`.

Free correctness oracle: on the 60% pure-HeuristicPlayer games the kernel
playout IS the policy that played the game, so playout labels equal outcome
labels EXACTLY at every ply. `--smoke --target playout` asserts that, and
also asserts the two label sets DIFFER somewhere in the randomized-mix games.

Why the labels change anything (PHASE4_PLAN.md Task 6.5): outcome labels
carry irreducible future randomness; playout labels are the deterministic,
noise-free quantity the search actually consumes.

Usage:
  .venv/bin/python experiments/gen_value_data.py --smoke
  .venv/bin/python experiments/gen_value_data.py --games 40000
  .venv/bin/python experiments/gen_value_data.py --smoke --target playout
  .venv/bin/python experiments/gen_value_data.py --games 40000 --target playout
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from openhearts.engine import features, kernel  # noqa: E402
from openhearts.engine.game import deal  # noqa: E402
from openhearts.players.heuristic import HeuristicPlayer  # noqa: E402
from openhearts.players.randomized import RandomizedHeuristic  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_DIR = os.path.join(RESULTS_DIR, "value_data")
OUT_DIR_PLAYOUT = os.path.join(RESULTS_DIR, "value_data_playout")
TARGETS = ("outcome", "playout")

SEED_BASE = 500000
PLIES_PER_GAME = 52
TARGET_PLY_POSITIONS = 2_000_000
DEFAULT_GAMES = 40000  # ~2M ply-positions / 52 plies per game
SMOKE_GAMES = 100
SHARD_ROWS = 100_000


# --------------------------------------------------------------------- mix
def _mix_for_seed(seed):
    """'heuristic' (60%) or 'randomized' (40%), deterministic by deal seed."""
    return "heuristic" if seed % 5 < 3 else "randomized"


def _players_for(seed):
    mix = _mix_for_seed(seed)
    if mix == "heuristic":
        return [HeuristicPlayer() for _ in range(4)], mix
    return [RandomizedHeuristic(np.random.default_rng([seed, s]), epsilon=0.1)
            for s in range(4)], mix


def _split_for_seed(seed):
    bucket = seed % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "val"
    return "test"


# ------------------------------------------------------------- one game
def play_and_record(seed, return_raw=False, target="outcome", out_bufs=None):
    """Play one game; return (features[52,4,NF] f16, targets[52,4] f32).

    `features[p, s]` is the position before ply p's card is played, rotated
    to seat s. `targets[p, s]` is seat s's points from ply p to hand end
    (`target="outcome"`) or from ply p to the end of ONE heuristic
    `kernel.playout_to_end` started at ply p (`target="playout"`); both are
    indexed by ABSOLUTE seat s.

    `return_raw=True` also returns the per-ply raw arrays fed to
    `featurize_batch` (float64, pre-downcast), for cross-checking against
    single-row `featurize()` calls.
    """
    players, _mix = _players_for(seed)
    state = deal(np.random.default_rng(seed))

    hands = np.zeros((PLIES_PER_GAME, 4), dtype=np.int64)
    played_mask = np.zeros(PLIES_PER_GAME, dtype=np.int64)
    trick_cards = np.zeros((PLIES_PER_GAME, 4), dtype=np.int64)
    trick_seats = np.zeros((PLIES_PER_GAME, 4), dtype=np.int64)
    trick_len = np.zeros(PLIES_PER_GAME, dtype=np.int64)
    led_suit = np.full(PLIES_PER_GAME, -1, dtype=np.int64)
    win_seat = np.full(PLIES_PER_GAME, -1, dtype=np.int64)
    hearts_broken = np.zeros(PLIES_PER_GAME, dtype=np.bool_)
    trick_number = np.zeros(PLIES_PER_GAME, dtype=np.int64)
    scores = np.zeros((PLIES_PER_GAME, 4), dtype=np.int64)
    to_play = np.zeros(PLIES_PER_GAME, dtype=np.int64)

    p = 0
    while not state.is_over():
        assert p < PLIES_PER_GAME, "more than 52 plies in a hand"
        hands[p] = state.hands
        pm = 0
        for _s, c in state.history:
            pm |= 1 << c
        for i, (s, c) in enumerate(state.current_trick):
            trick_cards[p, i] = c
            trick_seats[p, i] = s
            pm |= 1 << c
        played_mask[p] = pm
        tl = len(state.current_trick)
        trick_len[p] = tl
        if tl:
            led = int(trick_cards[p, 0]) // 13
            wr = int(trick_cards[p, 0]) % 13
            ws = int(trick_seats[p, 0])
            for i in range(1, tl):
                c = int(trick_cards[p, i])
                if c // 13 == led and c % 13 > wr:
                    wr, ws = c % 13, int(trick_seats[p, i])
            led_suit[p] = led
            win_seat[p] = ws
        hearts_broken[p] = state.hearts_broken
        trick_number[p] = state.trick_number
        scores[p] = state.scores

        seat = state.to_play
        to_play[p] = seat
        card = players[seat].choose(state.view_for(seat))
        state.play(card)
        p += 1

    assert p == PLIES_PER_GAME, f"expected 52 plies, got {p}"
    assert state.is_over()
    final_scores = np.array(state.scores, dtype=np.int64)
    assert final_scores.sum() == 26

    out64 = np.zeros((PLIES_PER_GAME, 4, features.NF), dtype=np.float64)
    features.featurize_batch(hands, played_mask, trick_cards, trick_seats,
                             trick_len, led_suit, win_seat, hearts_broken,
                             trick_number, scores, out64)
    feats = out64.astype(np.float16)

    # targets[p, s] = points seat s takes from ply p onward (absolute seat s,
    # independent of the rotation baked into feats[p, s]).
    targets = (final_scores[None, :] - scores).astype(np.float32)

    if target == "playout":
        # ONE kernel playout per ply (52/game). playout_to_end MUTATES its
        # hands/scores/trick arrays in place, so every row it is handed is a
        # COPY -- the recorded position arrays above (already consumed by
        # featurize_batch) must survive untouched, and the outcome labels
        # computed just above must stay exactly what they were.
        if out_bufs is None:
            out_bufs = (np.zeros(PLIES_PER_GAME, dtype=np.int64),
                        np.zeros(PLIES_PER_GAME, dtype=np.int64))
        out_c, out_s = out_bufs
        po = np.zeros((PLIES_PER_GAME, 4), dtype=np.float32)
        for q in range(PLIES_PER_GAME):
            h = hands[q].copy()
            sc = scores[q].copy()
            tc = trick_cards[q].copy()
            ts = trick_seats[q].copy()
            kernel.playout_to_end(h, int(to_play[q]), int(played_mask[q]),
                                  tc, ts, int(trick_len[q]),
                                  bool(hearts_broken[q]), int(trick_number[q]),
                                  sc, out_c, out_s)
            po[q] = sc - scores[q]
        assert np.all(po >= 0.0)
        targets_out = po
    else:
        targets_out = targets

    if return_raw:
        raw = dict(hands=hands, played_mask=played_mask,
                  trick_cards=trick_cards, trick_seats=trick_seats,
                  trick_len=trick_len, led_suit=led_suit, win_seat=win_seat,
                  hearts_broken=hearts_broken, trick_number=trick_number,
                  scores=scores, out64=out64,
                  to_play=to_play, outcome_targets=targets)
        return feats, targets_out, raw
    return feats, targets_out


def verify_batch_matches_single(seeds, n_samples_per_game=25):
    """Cross-check `featurize_batch` against per-row `featurize` calls.

    Guards against the batch helper (Task 2's addition to features.py)
    silently disagreeing with the pinned single-row contract it wraps --
    every training row flows through the batch path, so this is not
    optional. Checked over both player-mix branches by construction (seeds
    span both).
    """
    rng = np.random.default_rng(12345)
    n_checked = 0
    for seed in seeds:
        _feats, _targets, raw = play_and_record(seed, return_raw=True)
        plies = rng.choice(PLIES_PER_GAME, size=n_samples_per_game,
                           replace=False)
        for p in plies:
            for seat in range(4):
                single = features.featurize(
                    raw["hands"][p], raw["played_mask"][p],
                    raw["trick_cards"][p], raw["trick_seats"][p],
                    raw["trick_len"][p], raw["led_suit"][p],
                    raw["win_seat"][p], raw["hearts_broken"][p],
                    raw["trick_number"][p], raw["scores"][p], seat)
                assert np.array_equal(single, raw["out64"][p, seat]), (
                    f"seed {seed} ply {p} seat {seat}: featurize_batch "
                    f"disagrees with featurize")
                n_checked += 1
    print(f"featurize_batch cross-check: {n_checked} (ply, seat) rows across "
          f"{len(seeds)} games, exact agreement with single-row featurize()")


def verify_playout_labels(seeds):
    """Task 6.5 smoke oracle for `--target playout`.

    (a) FREE CORRECTNESS ORACLE: in a pure-HeuristicPlayer game the kernel
        playout is a bitwise port of the very policy that played the game,
        so a playout started at ply p replays the recorded continuation
        card-for-card. Playout labels must therefore equal outcome labels
        EXACTLY at every ply of every heuristic-mix game. Any mismatch means
        the position handed to the kernel is wrong.
    (b) In randomized-mix games (eps=0.1) the two label sets must DIFFER
        somewhere -- checked per GAME, not per ply, because late plies
        (ply 51 always) leave nothing to diverge on.
    (c) Determinism: a second pass over the same seeds reproduces both label
        sets bit-for-bit.
    """
    n_heur = n_rand = n_rand_differing = 0
    n_diff_plies = 0
    for seed in seeds:
        _f, po, raw = play_and_record(seed, return_raw=True, target="playout")
        oc = raw["outcome_targets"]
        _f2, po2, _r2 = play_and_record(seed, return_raw=True,
                                        target="playout")
        assert np.array_equal(po, po2), f"seed {seed}: labels nondeterministic"
        if _mix_for_seed(seed) == "heuristic":
            n_heur += 1
            assert np.array_equal(po, oc), (
                f"seed {seed}: PURE-HEURISTIC game, playout labels differ "
                f"from outcome labels at plies "
                f"{sorted(set(np.nonzero(po != oc)[0].tolist()))} -- the "
                f"kernel playout IS this game's policy, so the position "
                f"marshalling must be wrong")
        else:
            n_rand += 1
            d = int(np.count_nonzero((po != oc).any(axis=1)))
            n_diff_plies += d
            if d:
                n_rand_differing += 1
    assert n_heur and n_rand, "seed range must span both mixes"
    assert n_rand_differing, (
        "no randomized-mix game had ANY ply where playout labels differ from "
        "outcome labels -- labels are suspiciously identical")
    print(f"playout-label verify: {n_heur} heuristic-mix games EXACTLY match "
          f"outcome labels at all 52 plies (correctness oracle); "
          f"{n_rand_differing}/{n_rand} randomized-mix games differ "
          f"({n_diff_plies} differing plies total); labels deterministic "
          f"across two runs")


# ----------------------------------------------------------------- shards
class ShardWriter:
    """Buffers rows for one split and flushes ~SHARD_ROWS-row .npz files."""

    def __init__(self, out_dir, split, meta):
        self.out_dir = out_dir
        self.split = split
        self.meta = meta
        self.shard_idx = 0
        self._reset()

    def _reset(self):
        self.features = []
        self.targets = []
        self.game_ids = []
        self.seat_index = []
        self.ply_index = []
        self.n = 0

    def add_game(self, seed, feats, targets):
        # feats: [52,4,NF] f16, targets: [52,4] f32 -> flatten to rows
        f = feats.reshape(-1, features.NF)
        t = targets.reshape(-1)
        n_rows = f.shape[0]
        plys = np.repeat(np.arange(PLIES_PER_GAME, dtype=np.int8), 4)
        seats = np.tile(np.arange(4, dtype=np.int8), PLIES_PER_GAME)
        self.features.append(f)
        self.targets.append(t)
        self.game_ids.append(np.full(n_rows, seed, dtype=np.int32))
        self.seat_index.append(seats)
        self.ply_index.append(plys)
        self.n += n_rows
        if self.n >= SHARD_ROWS:
            self.flush()

    def flush(self):
        if self.n == 0:
            return
        feats = np.concatenate(self.features, axis=0)
        targets = np.concatenate(self.targets, axis=0)
        game_ids = np.concatenate(self.game_ids, axis=0)
        seat_index = np.concatenate(self.seat_index, axis=0)
        ply_index = np.concatenate(self.ply_index, axis=0)
        # split is shard-per-split (this writer only ever holds one split's
        # rows): the filename and `meta["split"]` carry it, so it is NOT
        # duplicated per row (that would be ~20 bytes/row for nothing --
        # the full run is size-budget-tight, see manifest).
        path = os.path.join(
            self.out_dir, f"value_{self.split}_{self.shard_idx:05d}.npz")
        shard_meta = dict(self.meta, split=self.split)
        np.savez(path, features=feats, targets=targets, game_ids=game_ids,
                 seat_index=seat_index, ply_index=ply_index,
                 meta=np.array(shard_meta, dtype=object))
        self.shard_idx += 1
        self._reset()
        return path


# ------------------------------------------------------------------- main
def _git_hash():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def generate(n_games, out_dir, progress_every=1000, target="outcome"):
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "seed_base": SEED_BASE,
        "n_games": n_games,
        "seed_range": [SEED_BASE, SEED_BASE + n_games - 1],
        "player_mix_rule": "seed % 5 < 3 -> 4x HeuristicPlayer (60%); "
                           "else 4x RandomizedHeuristic(epsilon=0.1) (40%)",
        "split_rule": "seed % 100 < 90 -> train; < 95 -> val; else test",
        "features_v": features.FEATURES_V,
        "nf": features.NF,
        "git_hash": _git_hash(),
        "plies_per_game": PLIES_PER_GAME,
    }
    if target != "outcome":
        # Added ONLY off the default path: `np.savez` of this dict changes
        # bytes when a key appears, and outcome-mode shards must stay
        # byte-identical to the pre-6.5 run. Absent key == "outcome".
        meta["target"] = target
        meta["target_rule"] = (
            "labels = kernel.playout_to_end(position at this ply) final "
            "scores - scores at this ply, per absolute seat")
    writers = {s: ShardWriter(out_dir, s, meta)
              for s in ("train", "val", "test")}
    n_rows = 0
    out_bufs = (np.zeros(PLIES_PER_GAME, dtype=np.int64),
                np.zeros(PLIES_PER_GAME, dtype=np.int64))
    t0 = time.time()
    for i in range(n_games):
        seed = SEED_BASE + i
        feats, targets = play_and_record(seed, target=target,
                                         out_bufs=out_bufs)
        split = _split_for_seed(seed)
        writers[split].add_game(seed, feats, targets)
        n_rows += feats.shape[0] * 4
        if (i + 1) % progress_every == 0 or (i + 1) == n_games:
            elapsed = time.time() - t0
            print(f"[{i + 1}/{n_games}] games | {n_rows} rows | "
                  f"{elapsed:.1f}s elapsed | "
                  f"{(i + 1) / elapsed:.1f} games/s", flush=True)
    for w in writers.values():
        w.flush()
    elapsed = time.time() - t0
    return n_rows, elapsed, meta


def write_manifest(path, n_games, n_rows, elapsed, meta, smoke):
    game_counts = {"train": 0, "val": 0, "test": 0}
    for i in range(n_games):
        game_counts[_split_for_seed(meta["seed_base"] + i)] += 1
    row_counts = {s: c * PLIES_PER_GAME * 4 for s, c in game_counts.items()}
    with open(path, "w") as f:
        f.write("# Phase 4 Task 2: self-play position/outcome data\n")
        f.write(f"# smoke={smoke}\n")
        f.write(f"# n_games={n_games} seed_range={meta['seed_range']}\n")
        if "target" in meta:  # outcome mode stays byte-identical: no line
            f.write(f"# target={meta['target']}\n")
            f.write(f"# target_rule: {meta['target_rule']}\n")
        f.write(f"# player_mix_rule: {meta['player_mix_rule']}\n")
        f.write(f"# split_rule: {meta['split_rule']}\n")
        f.write(f"# games per split: train={game_counts['train']} "
                f"val={game_counts['val']} test={game_counts['test']}\n")
        f.write(f"# rows per split: train={row_counts['train']} "
                f"val={row_counts['val']} test={row_counts['test']}\n")
        f.write(f"# FEATURES_V={meta['features_v']} NF={meta['nf']} "
                f"git_hash={meta['git_hash']}\n")
        f.write(f"# n_rows={n_rows} plies_per_game={meta['plies_per_game']}\n")
        f.write(f"# wall_time_s={elapsed:.1f} rows_per_sec="
                f"{n_rows / elapsed:.1f} games_per_sec={n_games / elapsed:.2f}\n")
        if n_games:
            proj_s = DEFAULT_GAMES * elapsed / n_games
            f.write(f"# projected full-run ({DEFAULT_GAMES} games) time from "
                    f"this run's rate: {proj_s:.1f}s (~{proj_s / 60:.1f} min)\n")


def _load_and_verify_smoke(out_dir, target="outcome"):
    """Load every shard back; verify schema, bounds, and per-ply target sums.

    The target-sum check is a genuine cross-check against an independent
    quantity already in the shard: rotated scalar slot OFF_SCALARS+4 is each
    row's OWN (absolute) seat's score-so-far / 26 (rotation puts the
    evaluated seat at slot 0), so `round(f[OFF_SCALARS+4] * 26) + target`
    must equal that seat's FINAL score, constant across all 52 plies of a
    game -- and the 4 seats' final scores must sum to exactly 26.

    Under `target="playout"` the CONSTANT-across-plies part is not a property
    of the labels (each ply has its own playout, hence its own final scores),
    so that one check is skipped; everything else -- integral 4-seat sums,
    ply-0 sums of exactly 26 (a playout also ends with 26 points dealt out),
    schema, bounds -- still holds and is still asserted.
    """
    paths = sorted(p for p in os.listdir(out_dir) if p.startswith("value_"))
    assert paths, "no shards written"
    own_score_off = features.OFF_SCALARS + 4
    by_game = {}
    for name in paths:
        d = np.load(os.path.join(out_dir, name), allow_pickle=True)
        feats, targets = d["features"], d["targets"]
        game_ids, seat_index, ply_index = (d["game_ids"], d["seat_index"],
                                           d["ply_index"])
        assert feats.dtype == np.float16
        assert targets.dtype == np.float32
        assert feats.shape[1] == features.NF
        assert np.all(feats >= 0.0) and np.all(feats <= 1.0)
        assert np.all(targets >= 0.0)
        meta = d["meta"].item()
        assert meta["features_v"] == features.FEATURES_V
        assert meta.get("target", "outcome") == target, (
            f"shard {name} meta target={meta.get('target', 'outcome')}, "
            f"expected {target}")
        for row in range(feats.shape[0]):
            g, p, s = (int(game_ids[row]), int(ply_index[row]),
                      int(seat_index[row]))
            score_so_far = round(float(feats[row, own_score_off]) * 26.0)
            final_for_seat = score_so_far + float(targets[row])
            d_by_game = by_game.setdefault(g, {})
            d_by_game.setdefault("plies", {}).setdefault(p, {})[s] = \
                float(targets[row])
            d_by_game.setdefault("taken", {}).setdefault(p, {})[s] = \
                score_so_far
            d_by_game.setdefault("final_by_seat", {}).setdefault(s, set()) \
                .add(round(final_for_seat, 3))
    n_checked = 0
    for g, info in by_game.items():
        for p, seats in info["plies"].items():
            assert len(seats) == 4, f"game {g} ply {p}: missing seats"
            total = sum(seats.values())
            assert abs(total - round(total)) < 1e-3, (
                f"game {g} ply {p}: target sum {total} not integral")
            # Holds for BOTH label types: points already taken at this ply
            # plus points still to be taken (really, or in the playout) is
            # always the full 26 a hand distributes.
            taken = sum(info["taken"][p].values())
            assert abs(taken + total - 26.0) < 1e-3, (
                f"game {g} ply {p}: taken {taken} + targets {total} != 26")
            n_checked += 1
        # ply 0: nothing taken yet, so the 4 targets must sum to exactly 26.
        assert abs(sum(info["plies"][0].values()) - 26.0) < 1e-3, (
            f"game {g}: ply-0 target sum != 26")
        if target == "outcome":
            # Only outcome labels share ONE final score per game; each
            # playout label has its own hand ending (see docstring).
            for s, finals in info["final_by_seat"].items():
                assert len(finals) == 1, (
                    f"game {g} seat {s}: final score not constant across "
                    f"plies ({finals})")
            final_total = sum(next(iter(v))
                              for v in info["final_by_seat"].values())
            assert abs(final_total - 26.0) < 1e-3, (
                f"game {g}: 4 seats' final scores sum to {final_total}, "
                f"not 26")
    extra = ("per-seat (score_so_far + target) constant per game and sums "
             "to 26" if target == "outcome" else
             "per-ply (taken + targets) == 26 [the constant-final-score "
             "check does not apply to playout labels]")
    print(f"smoke verify (target={target}): {len(by_game)} games, "
          f"{n_checked} ply-slices; 4-seat target sums integral, ply-0 "
          f"sums == 26, {extra}")
    return by_game


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=DEFAULT_GAMES)
    ap.add_argument("--smoke", action="store_true",
                    help="100 games, separate output dir, schema verification")
    ap.add_argument("--target", choices=TARGETS, default="outcome",
                    help="'outcome' (default, Task 2 behaviour, byte-identical"
                         ") or 'playout' (Task 6.5 distillation labels)")
    args = ap.parse_args()

    base_dir = OUT_DIR if args.target == "outcome" else OUT_DIR_PLAYOUT
    base_name = os.path.basename(base_dir)
    if args.smoke:
        n_games = SMOKE_GAMES
        out_dir = os.path.join(base_dir, "smoke")
    else:
        n_games = args.games
        out_dir = base_dir

    print(f"generating {n_games} games (seeds {SEED_BASE}..."
          f"{SEED_BASE + n_games - 1}) target={args.target} -> {out_dir}",
          flush=True)
    n_rows, elapsed, meta = generate(n_games, out_dir,
                                     progress_every=10 if args.smoke else 1000,
                                     target=args.target)
    manifest_path = os.path.join(
        out_dir if args.smoke else RESULTS_DIR,
        "manifest_smoke.txt" if args.smoke else
        os.path.join(base_name, "manifest.txt"))
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    write_manifest(manifest_path, n_games, n_rows, elapsed, meta, args.smoke)
    print(f"wrote {manifest_path}")
    print(f"{n_rows} rows in {elapsed:.1f}s "
          f"({n_rows / elapsed:.1f} rows/s, {n_games / elapsed:.2f} games/s)")

    if args.smoke:
        _load_and_verify_smoke(out_dir, target=args.target)
        # a handful of seeds spanning both mixes (seed % 5 < 3 -> heuristic)
        cross_check_seeds = [SEED_BASE, SEED_BASE + 2, SEED_BASE + 3,
                             SEED_BASE + 4, SEED_BASE + 7]
        verify_batch_matches_single(cross_check_seeds)
        if args.target == "playout":
            verify_playout_labels(range(SEED_BASE, SEED_BASE + 40))
        full_games = DEFAULT_GAMES
        projected_s = full_games * elapsed / n_games
        print(f"projected full run ({full_games} games) from this smoke's "
              f"rate (includes JIT warmup): {projected_s:.0f}s "
              f"(~{projected_s / 60:.1f} min)")


if __name__ == "__main__":
    main()
