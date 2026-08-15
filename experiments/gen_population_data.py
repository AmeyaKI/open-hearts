"""Phase 5 Task 2: population choice-event generator.

WHY ONE ROW PER DECISION (not per-ply-per-seat like Phase 4).  Phase 4 trained
a VALUE function: every seat has a score-so-far and a final score at every
ply, so recording all 4 seats per ply is free signal.  Phase 5 trains a
CHOICE model: `P(card | view)`.  Only one seat actually chooses a card at each
ply (the seat `to_play`), and cards played by a single-legal decision (exactly
one card satisfies suit-following / forced-play rules) teach the model
nothing -- there was no choice to learn from.  So the unit here is a decision
EVENT: (view features, legal mask, chosen card, ...), one row per ply of every
game with 2+ legal cards, contributed only by the seat that actually acted.

POPULATION AND TABLES.  `make_population(200, 50, MASTER_SEED)` (Task 1) is
called once at import time to fix the TRAIN/HELD-OUT split; the anchors
(`ANCHOR_IDS`: HeuristicPlayer, RandomizedHeuristic(eps=0.1), RandomPlayer)
are appended to the TRAIN pool ONLY, per Task 1's contract -- they are never
held out because they are families we already have, not "unseen" opponents.
HELD-OUT ids are never instantiated by this script and never enter a shard;
`_load_and_verify_smoke` asserts no held-out id appears anywhere in written
rows, and `main` asserts the held-out set is disjoint from every table drawn.

Every game seed `g` (>= SEED_BASE = 700000, the plan's population-training
seed range) deterministically draws a 4-personality table from the combined
train pool (200 personalities + 3 anchors = 203 members) via
`np.random.default_rng([g, TABLE_SALT])` sampling 4 DISTINCT members without
replacement; the sampled order IS the seat order (seat 0..3), so table
composition and seating are both fixed functions of the game seed alone --
fully deterministic and reproducible from `g`.  Deal cards are dealt by the
existing `game.deal(np.random.default_rng(g))`, matching Phase 4's convention
of deriving both the deal and the table from the same seed.  Each seated
player's OWN internal randomness (epsilon rolls, softmax draws, anchor noise)
is driven by a separate rng stream keyed off `(g, seat)`, so re-seating the
same personality at a different table/seat never reuses another seat's random
stream.

PROFILER_FEATURES_V=1 -- the observer-legal feature subset
------------------------------------------------------------
The profiler models a player who can see only its OWN hand, never hidden
cards.  Rather than define a new, narrower layout, this reuses
`openhearts.engine.features` UNCHANGED (same FEATURES_V=1, same NF=333, same
offsets `OFF_HANDS/OFF_PLAYED/OFF_TRICK/OFF_WIN_SEAT/OFF_LED/OFF_SCALARS`) and
ZEROES every hand except the acting seat's own before calling `featurize`.
Concretely: for the ply where seat `s` acts, the `hands` array passed in has
`hands[t] = 0` for every seat `t != s` (`hands[s]` is the real hand), and
`featurize(..., seat=s)` is called as usual.  Because `OFF_HANDS` block r=0 is
always the CALLING seat's own hand (rotation puts it there by construction)
and blocks r=1..3 are OTHER seats' hands, zeroing every hand but the caller's
means blocks r=1..3 are STRUCTURALLY zero in every row this script emits --
not merely usually zero, provably zero, since the array handed to `featurize`
literally contains no other seat's cards.  This is "OPTION 1" from the task
spec (build the kernel arrays with other hands zeroed and call the existing
featurize) chosen over a hand-rolled narrower layout because it is provably
correct by inspection (the information is never in the array to leak) and
keeps ONE feature contract across the whole codebase instead of two.
PROFILER_FEATURES_V=1 is therefore: FEATURES_V=1's exact 333-dim layout, with
the invariant "OFF_HANDS blocks r=1,2,3 are always all-zero".  See
`verify_hidden_hand_independence` below for the smoke assertion that no
hidden-hand information leaks despite reusing the wider array shape: it
featurizes the SAME observer view (own hand, played cards, current trick,
scalars) under two DIFFERENT completions of the other three hidden hands and
asserts byte-identical output.

EPSILON CONVENTION (recorded per row, float32).  Personalities carry their
own `params.epsilon` (Task 1, range 0.03-0.25).  Anchors do not have an
`epsilon` field, so this script assigns the CONVENTION values the task spec
fixes: `heuristic`=0.0 (fully deterministic), `randomized_heuristic`=0.1 (its
actual deviation probability), `random`=1.0 (every decision is "noise" by
construction -- a uniform player's whole policy is the epsilon branch).

SHARDING / SPLIT.  Mirrors Phase 4 (`gen_value_data.py`) exactly: ~250k-row
`.npz` shards under `results/population_data/`, split BY GAME via
`seed % 100`: <90 train, <95 val, else test -- applied only within
TRAIN-personality games (by construction, since held-out personalities never
seat a game here, every game is a train-personality game and the split
applies uniformly).  dtypes are deliberately tight (see `ShardWriter`):
profiler_features float16[N,333] (~666B/row), legal_mask int64[N] (8B),
chosen_card int8 (1B), personality_id int32[N,4] (16B, all 4 seats' ids so
mixture-training work in later tasks can see who was at the table),
acting_seat int8 (1B), epsilon float32 (4B), game_seed int32 (4B), ply int8
(1B) -- about 700B/row, versus Phase 4's ~700B/row per (ply,seat) row but at
roughly 1/4 the ROW COUNT per game (one row per multi-legal ply, not per
seat), so the "~1/4 the width" framing in the task brief nets out to a
similarly small total footprint; see the printed projection for the actual
number.

Usage:
  .venv/bin/python experiments/gen_population_data.py --smoke
  .venv/bin/python experiments/gen_population_data.py --games 200000
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from openhearts.engine import cards, features  # noqa: E402
from openhearts.engine.game import deal  # noqa: E402
from openhearts.players.heuristic import HeuristicPlayer  # noqa: E402
from openhearts.players.randomized import RandomizedHeuristic  # noqa: E402
from openhearts.players.random_player import RandomPlayer  # noqa: E402
from openhearts.opponent.obsfeat import observer_features  # noqa: E402
from openhearts.players.personality import (  # noqa: E402
    ANCHOR_IDS, PersonalityPlayer, make_population, sample_personality)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_DIR = os.path.join(RESULTS_DIR, "population_data")

SEED_BASE = 700000            # plan: population-TRAINING seed range
MASTER_SEED = 314159          # Task 1's make_population seed, fixed here for
                               # reproducibility; documented per Global
                               # Constraints ("personality ids get their own
                               # seed derivation, documented in the
                               # generator").  Arbitrary but frozen: changing
                               # it reshuffles the whole train/held-out split.
N_TRAIN_PERSONALITIES = 200
N_HELDOUT_PERSONALITIES = 50
TABLE_SALT = 777              # salts table-draw rng away from the deal rng
                               # and away from per-seat player rngs, so none
                               # of the three streams ever collide.
PLIES_PER_GAME = 52
SHARD_ROWS = 250_000
SMOKE_GAMES = 200

# Anchor id -> convention epsilon, per the task spec ("anchors: heuristic=0.0,
# randomized=0.1, random=1.0 by convention -- document").
ANCHOR_EPSILON = {
    ANCHOR_IDS["heuristic"]: 0.0,
    ANCHOR_IDS["randomized_heuristic"]: 0.1,
    ANCHOR_IDS["random"]: 1.0,
}

TRAIN_IDS, HELDOUT_IDS = make_population(
    N_TRAIN_PERSONALITIES, N_HELDOUT_PERSONALITIES, MASTER_SEED)
HELDOUT_SET = frozenset(HELDOUT_IDS)
# Combined TRAIN pool: personalities + anchors (Task 1 contract: anchors are
# TRAIN-only). Anchor ids are negative, personality ids positive -- disjoint
# by construction (personality.py's docstring).
TRAIN_POOL = list(TRAIN_IDS) + list(ANCHOR_IDS.values())
assert HELDOUT_SET.isdisjoint(TRAIN_POOL), (
    "held-out ids leaked into the train pool")


# --------------------------------------------------------------- players
def _make_player(pid, seed, seat):
    """Instantiate the player for personality/anchor id `pid` at (seed, seat).

    Each seated player gets its OWN rng stream keyed off (seed, seat, a
    per-role salt) so re-seating the same id at a different table never
    reuses another seat's randomness.
    """
    rng = np.random.default_rng([int(seed), int(seat), 0xA1CE])
    if pid == ANCHOR_IDS["heuristic"]:
        return HeuristicPlayer()
    if pid == ANCHOR_IDS["randomized_heuristic"]:
        return RandomizedHeuristic(rng, epsilon=0.1)
    if pid == ANCHOR_IDS["random"]:
        return RandomPlayer(rng)
    params = sample_personality(pid)
    return PersonalityPlayer(rng, params)


def _epsilon_for(pid):
    if pid in ANCHOR_EPSILON:
        return ANCHOR_EPSILON[pid]
    return sample_personality(pid).epsilon


def _table_for_seed(seed, pool=None):
    """4 distinct member ids from `pool` (default TRAIN_POOL), from `seed`.

    The sampled order IS the seat order: table[s] plays seat s.

    `pool` was added by Phase 5 Task 3 (`experiments/train_profiler.py`) so
    the held-out evaluation can draw HELD-OUT tables through this exact draw
    rule instead of duplicating it. The default is TRAIN_POOL, so every
    existing caller -- and every already-generated shard -- is untouched.
    """
    members = TRAIN_POOL if pool is None else list(pool)
    rng = np.random.default_rng([int(seed), TABLE_SALT])
    idx = rng.choice(len(members), size=4, replace=False)
    return [members[int(i)] for i in idx]


def _split_for_seed(seed):
    bucket = seed % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "val"
    return "test"


# ------------------------------------------------------------- one game
def play_and_record(seed, return_raw=False, pool=None,
                    record_heuristic=False):
    """Play one game; return decision-event rows (multi-legal plies only).

    `pool` / `record_heuristic` were added by Phase 5 Task 3 so
    `experiments/train_profiler.py` can extract held-out-personality decision
    events through THIS function rather than a copy of it. Both default to the
    Task-2 behaviour exactly: `pool=None` means TRAIN_POOL, and
    `record_heuristic=False` emits no extra column. With `record_heuristic`,
    each row also carries the card a plain `HeuristicPlayer` would have played
    from the same view (baseline 2's "heuristic match"); `HeuristicPlayer()`
    is stateless and rng-free, so asking it cannot perturb the seated player's
    random stream and the game plays out identically either way.

    Returns a dict of equal-length arrays: profiler_features float16[K,NF],
    legal_mask int64[K], chosen_card int8[K], acting_seat int8[K],
    epsilon float32[K], ply int8[K] -- K = number of multi-legal decisions in
    this game (<=52).  `personality_ids` (the 4 table members, constant for
    the whole game) and `game_seed` are returned separately since they are
    per-GAME, not per-row.
    """
    table = _table_for_seed(seed, pool)
    players = [_make_player(pid, seed, s) for s, pid in enumerate(table)]
    eps_by_seat = [_epsilon_for(pid) for pid in table]
    href = HeuristicPlayer() if record_heuristic else None
    heur_rows = []

    state = deal(np.random.default_rng(seed))

    feats_rows = []
    legal_rows = []
    chosen_rows = []
    seat_rows = []
    eps_rows = []
    ply_rows = []

    p = 0
    while not state.is_over():
        assert p < PLIES_PER_GAME, "more than 52 plies in a hand"
        seat = state.to_play
        view = state.view_for(seat)
        legal_list = cards.cards_in(view.legal_moves)
        n_legal = len(legal_list)

        if n_legal > 1:
            # Phase 5 Task 4 refactor: this block moved VERBATIM into
            # `openhearts.opponent.obsfeat.observer_features` so the audit
            # replay in `search/profiled.py` featurizes exactly as training
            # did. Smoke output (rows, determinism, hidden-hand independence,
            # decisions/game) is unchanged across the move.
            f = observer_features(state, seat)
            # invariant check (cheap; hands array literally has no other
            # seat's cards, so this can never fail -- kept as a guard against
            # a future edit that changes the zeroing above).
            assert np.all(f[features.OFF_HANDS + 52:features.OFF_HANDS + 208]
                          == 0.0)

            feats_rows.append(f.astype(np.float16))
            legal_rows.append(view.legal_moves)
            seat_rows.append(seat)
            eps_rows.append(eps_by_seat[seat])
            ply_rows.append(p)
            if href is not None:
                heur_rows.append(href.choose(view))

        card = players[seat].choose(view)
        if n_legal > 1:
            assert (view.legal_moves >> card) & 1, (
                f"seed {seed} ply {p}: chosen card not in legal mask")
            chosen_rows.append(card)
        state.play(card)
        p += 1

    assert p == PLIES_PER_GAME, f"expected 52 plies, got {p}"
    assert state.is_over()
    assert sum(state.scores) == 26

    out = dict(
        profiler_features=(np.stack(feats_rows) if feats_rows
                           else np.zeros((0, features.NF), dtype=np.float16)),
        legal_mask=np.array(legal_rows, dtype=np.int64),
        chosen_card=np.array(chosen_rows, dtype=np.int8),
        acting_seat=np.array(seat_rows, dtype=np.int8),
        epsilon=np.array(eps_rows, dtype=np.float32),
        ply=np.array(ply_rows, dtype=np.int8),
    )
    if record_heuristic:
        out["heuristic_card"] = np.array(heur_rows, dtype=np.int8)
    if return_raw:
        out["table"] = table
    return out, table


def verify_hidden_hand_independence(seeds, n_checks_per_game=8):
    """No hidden-hand information ever enters a PROFILER_FEATURES_V=1 row.

    For a handful of decision plies, construct TWO different, both-valid
    completions of the hidden cards (the cards not in the acting seat's hand
    and not yet played) across the other three seats. Call the UNMODIFIED
    `features.featurize` -- with the REAL, non-zeroed hands -- once per
    completion; those two calls necessarily differ in `OFF_HANDS` blocks
    r=1..3 (different hidden deals), which is the point: it proves the two
    completions really are different inputs. Then assert that THIS
    GENERATOR's actual profiler row (own hand only, other hands zeroed) is
    identical to both completions' outputs with blocks r=1..3 masked off --
    i.e. whichever hidden reality happened to be true, the row this script
    would have written is exactly the same.
    """
    rng = np.random.default_rng(999)
    n_checked = 0
    hb0, hb1 = features.OFF_HANDS + 52, features.OFF_HANDS + 208
    for seed in seeds:
        table = _table_for_seed(seed)
        players = [_make_player(pid, seed, s) for s, pid in enumerate(table)]
        state = deal(np.random.default_rng(seed))
        checked_this_game = 0
        p = 0
        while not state.is_over() and checked_this_game < n_checks_per_game:
            seat = state.to_play
            view = state.view_for(seat)
            legal_list = cards.cards_in(view.legal_moves)
            if len(legal_list) > 1:
                own_hand = state.hands[seat]
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
                scores = np.asarray(state.scores, dtype=np.int64)

                # this generator's actual row: other hands zeroed.
                hands_zero = np.zeros(4, dtype=np.int64)
                hands_zero[seat] = own_hand
                row = features.featurize(
                    hands_zero, pm, trick_cards, trick_seats, tl, led_suit,
                    win_seat, state.hearts_broken, state.trick_number,
                    scores, seat)

                # two DIFFERENT, both-legal-shaped completions of the hidden
                # (non-own, unplayed) cards across the other three seats.
                other_seats = [s for s in range(4) if s != seat]
                hidden = sorted(
                    c for o in other_seats for c in cards.cards_in(state.hands[o]))
                perm_a = list(hidden)
                perm_b = list(hidden)
                rng.shuffle(perm_a)
                rng.shuffle(perm_b)
                # force perm_b to differ from perm_a whenever possible.
                if len(hidden) > 1 and perm_a == perm_b:
                    perm_b[0], perm_b[1] = perm_b[1], perm_b[0]

                def _completion(perm):
                    sizes = [cards.cards_in(state.hands[o]).__len__()
                            for o in other_seats]
                    hands = np.zeros(4, dtype=np.int64)
                    hands[seat] = own_hand
                    idx = 0
                    for o, sz in zip(other_seats, sizes):
                        h = 0
                        for c in perm[idx:idx + sz]:
                            h |= cards.bit(c)
                        hands[o] = h
                        idx += sz
                    return hands

                hands_a = _completion(perm_a)
                hands_b = _completion(perm_b)
                full_a = features.featurize(
                    hands_a, pm, trick_cards, trick_seats, tl, led_suit,
                    win_seat, state.hearts_broken, state.trick_number,
                    scores, seat)
                full_b = features.featurize(
                    hands_b, pm, trick_cards, trick_seats, tl, led_suit,
                    win_seat, state.hearts_broken, state.trick_number,
                    scores, seat)
                if len(hidden) > 3:
                    # the two completions must actually differ in the hidden
                    # hand blocks (otherwise this isn't testing anything).
                    assert not np.array_equal(full_a[hb0:hb1], full_b[hb0:hb1]), (
                        f"seed {seed} ply {p}: fabricated completions were "
                        f"not actually different -- test is vacuous")
                masked_a, masked_b = full_a.copy(), full_b.copy()
                masked_a[hb0:hb1] = 0.0
                masked_b[hb0:hb1] = 0.0
                assert np.array_equal(row, masked_a), (
                    f"seed {seed} ply {p}: profiler row differs from "
                    f"hidden-masked completion A")
                assert np.array_equal(row, masked_b), (
                    f"seed {seed} ply {p}: profiler row differs from "
                    f"hidden-masked completion B -- hidden-hand information "
                    f"leaked into the observer-legal row")
                n_checked += 1
                checked_this_game += 1
            card = players[seat].choose(view)
            state.play(card)
            p += 1
    assert n_checked > 0, "no multi-legal decisions found to check"
    print(f"hidden-hand independence: {n_checked} decision rows across "
          f"{len(seeds)} games -- row is byte-identical under two different, "
          f"actually-differing hidden completions of the other 3 hands")


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
        self.profiler_features = []
        self.legal_mask = []
        self.chosen_card = []
        self.personality_ids = []
        self.acting_seat = []
        self.epsilon = []
        self.game_seed = []
        self.ply = []
        self.n = 0

    def add_game(self, seed, table, rows):
        n_rows = rows["chosen_card"].shape[0]
        if n_rows == 0:
            return
        self.profiler_features.append(rows["profiler_features"])
        self.legal_mask.append(rows["legal_mask"])
        self.chosen_card.append(rows["chosen_card"])
        self.personality_ids.append(
            np.tile(np.array(table, dtype=np.int32), (n_rows, 1)))
        self.acting_seat.append(rows["acting_seat"])
        self.epsilon.append(rows["epsilon"])
        self.game_seed.append(np.full(n_rows, seed, dtype=np.int32))
        self.ply.append(rows["ply"])
        self.n += n_rows
        if self.n >= SHARD_ROWS:
            self.flush()

    def flush(self):
        if self.n == 0:
            return
        pf = np.concatenate(self.profiler_features, axis=0)
        lm = np.concatenate(self.legal_mask, axis=0)
        cc = np.concatenate(self.chosen_card, axis=0)
        pid = np.concatenate(self.personality_ids, axis=0)
        seat = np.concatenate(self.acting_seat, axis=0)
        eps = np.concatenate(self.epsilon, axis=0)
        gs = np.concatenate(self.game_seed, axis=0)
        ply = np.concatenate(self.ply, axis=0)
        path = os.path.join(
            self.out_dir, f"pop_{self.split}_{self.shard_idx:05d}.npz")
        shard_meta = dict(self.meta, split=self.split)
        np.savez(path, profiler_features=pf, legal_mask=lm, chosen_card=cc,
                 personality_ids=pid, acting_seat=seat, epsilon=eps,
                 game_seed=gs, ply=ply, meta=np.array(shard_meta, dtype=object))
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


def generate(n_games, out_dir, progress_every=1000):
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "seed_base": SEED_BASE,
        "n_games": n_games,
        "seed_range": [SEED_BASE, SEED_BASE + n_games - 1],
        "master_seed": MASTER_SEED,
        "n_train_personalities": N_TRAIN_PERSONALITIES,
        "n_heldout_personalities": N_HELDOUT_PERSONALITIES,
        "table_rule": "4 distinct ids drawn without replacement from "
                      "TRAIN_POOL (200 personalities + 3 anchors) via "
                      "np.random.default_rng([seed, TABLE_SALT]); sampled "
                      "order is seat order",
        "row_unit": "one row per multi-legal decision event (single-legal "
                    "plies excluded)",
        "epsilon_convention": "personality: params.epsilon; "
                              "anchor heuristic=0.0, randomized_heuristic="
                              "0.1, random=1.0",
        "split_rule": "seed % 100 < 90 -> train; < 95 -> val; else test",
        "profiler_features_v": 1,
        "features_v": features.FEATURES_V,
        "nf": features.NF,
        "git_hash": _git_hash(),
        "plies_per_game": PLIES_PER_GAME,
    }
    writers = {s: ShardWriter(out_dir, s, meta)
              for s in ("train", "val", "test")}
    n_rows = 0
    n_decisions_total = 0
    t0 = time.time()
    for i in range(n_games):
        seed = SEED_BASE + i
        rows, table = play_and_record(seed)
        split = _split_for_seed(seed)
        writers[split].add_game(seed, table, rows)
        n_rows += rows["chosen_card"].shape[0]
        n_decisions_total += rows["chosen_card"].shape[0]
        if (i + 1) % progress_every == 0 or (i + 1) == n_games:
            elapsed = time.time() - t0
            rate_g = (i + 1) / elapsed
            print(f"[{i + 1}/{n_games}] games | {n_rows} rows | "
                  f"{elapsed:.1f}s elapsed | {rate_g:.1f} games/s | "
                  f"{n_rows / elapsed:.1f} rows/s | "
                  f"{n_rows / (i + 1):.2f} decisions/game", flush=True)
    for w in writers.values():
        w.flush()
    elapsed = time.time() - t0
    return n_rows, elapsed, meta


def write_manifest(path, n_games, n_rows, elapsed, meta, smoke):
    game_counts = {"train": 0, "val": 0, "test": 0}
    for i in range(n_games):
        game_counts[_split_for_seed(meta["seed_base"] + i)] += 1
    with open(path, "w") as f:
        f.write("# Phase 5 Task 2: population choice-event data\n")
        f.write(f"# smoke={smoke}\n")
        f.write(f"# n_games={n_games} seed_range={meta['seed_range']}\n")
        f.write(f"# master_seed={meta['master_seed']} "
                f"n_train_personalities={meta['n_train_personalities']} "
                f"n_heldout_personalities={meta['n_heldout_personalities']}\n")
        f.write(f"# table_rule: {meta['table_rule']}\n")
        f.write(f"# row_unit: {meta['row_unit']}\n")
        f.write(f"# epsilon_convention: {meta['epsilon_convention']}\n")
        f.write(f"# split_rule: {meta['split_rule']}\n")
        f.write(f"# games per split: train={game_counts['train']} "
                f"val={game_counts['val']} test={game_counts['test']}\n")
        f.write(f"# PROFILER_FEATURES_V={meta['profiler_features_v']} "
                f"(FEATURES_V={meta['features_v']} layout with OFF_HANDS "
                f"blocks r=1,2,3 structurally zero) NF={meta['nf']} "
                f"git_hash={meta['git_hash']}\n")
        f.write(f"# n_rows={n_rows} plies_per_game={meta['plies_per_game']}\n")
        f.write(f"# decisions_per_game={n_rows / n_games:.3f}\n" if n_games
                else "")
        f.write(f"# wall_time_s={elapsed:.1f} rows_per_sec="
                f"{n_rows / elapsed:.1f} games_per_sec={n_games / elapsed:.2f}\n")


def _load_and_verify_smoke(out_dir):
    """Load every shard back; verify schema, legality, and the held-out wall."""
    paths = sorted(p for p in os.listdir(out_dir) if p.startswith("pop_"))
    assert paths, "no shards written"
    seen_pids = set()
    seat_seen = set()
    n_rows = 0
    for name in paths:
        d = np.load(os.path.join(out_dir, name), allow_pickle=True)
        pf, lm, cc = (d["profiler_features"], d["legal_mask"],
                     d["chosen_card"])
        pid, seat, eps, gs, ply = (d["personality_ids"], d["acting_seat"],
                                   d["epsilon"], d["game_seed"], d["ply"])
        assert pf.dtype == np.float16
        assert lm.dtype == np.int64
        assert cc.dtype == np.int8
        assert pid.dtype == np.int32 and pid.shape[1] == 4
        assert seat.dtype == np.int8
        assert eps.dtype == np.float32
        assert gs.dtype == np.int32
        assert ply.dtype == np.int8
        assert pf.shape[1] == features.NF
        assert np.all(pf >= 0.0) and np.all(pf <= 1.0)
        # OFF_HANDS blocks r=1..3 structurally zero (PROFILER_FEATURES_V=1).
        assert np.all(pf[:, features.OFF_HANDS + 52:features.OFF_HANDS + 208]
                      == 0.0)
        # chosen card always inside the legal mask.
        for row in range(cc.shape[0]):
            assert (int(lm[row]) >> int(cc[row])) & 1, (
                f"{name} row {row}: chosen card {cc[row]} not in legal mask "
                f"{lm[row]:052b}")
        n_rows += cc.shape[0]
        seen_pids.update(int(x) for x in pid.reshape(-1))
        seat_seen.update(int(x) for x in seat)
    # the held-out wall: no held-out id anywhere in any shard.
    leaked = seen_pids & HELDOUT_SET
    assert not leaked, f"held-out ids leaked into shards: {leaked}"
    assert seat_seen == {0, 1, 2, 3}, (
        f"not all 4 seats produced decision events: {seat_seen}")
    anchors_present = seen_pids & set(ANCHOR_IDS.values())
    assert anchors_present, "no anchor id appeared in any shard"
    print(f"smoke verify: {len(paths)} shards, {n_rows} rows; chosen card "
          f"always in legal mask; PROFILER_FEATURES_V=1 hand-zeroing intact; "
          f"held-out wall holds (0 of {len(HELDOUT_SET)} held-out ids seen); "
          f"all 4 seats present; anchors present ({sorted(anchors_present)}); "
          f"{len(seen_pids)} distinct personality/anchor ids seen")
    return n_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200_000,
                    help="games to generate (default targets >=5M rows "
                         "per the measured decisions/game rate)")
    ap.add_argument("--smoke", action="store_true",
                    help=f"{SMOKE_GAMES} games, separate output dir, full "
                         f"schema + determinism + hidden-hand verification")
    args = ap.parse_args()

    base_name = os.path.basename(OUT_DIR)
    if args.smoke:
        n_games = SMOKE_GAMES
        out_dir = os.path.join(OUT_DIR, "smoke")
    else:
        n_games = args.games
        out_dir = OUT_DIR

    print(f"generating {n_games} games (seeds {SEED_BASE}..."
          f"{SEED_BASE + n_games - 1}) -> {out_dir}", flush=True)
    n_rows, elapsed, meta = generate(
        n_games, out_dir, progress_every=20 if args.smoke else 1000)
    manifest_path = os.path.join(
        out_dir if args.smoke else RESULTS_DIR,
        "manifest_smoke.txt" if args.smoke else
        os.path.join(base_name, "manifest.txt"))
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    write_manifest(manifest_path, n_games, n_rows, elapsed, meta, args.smoke)
    print(f"wrote {manifest_path}")
    print(f"{n_rows} rows in {elapsed:.1f}s "
          f"({n_rows / elapsed:.1f} rows/s, {n_games / elapsed:.2f} games/s, "
          f"{n_rows / n_games:.2f} decisions/game)")

    if args.smoke:
        n_smoke_rows = _load_and_verify_smoke(out_dir)

        # determinism: rerun a handful of seeds, expect byte-identical rows.
        for seed in [SEED_BASE, SEED_BASE + 1, SEED_BASE + 5]:
            r1, t1 = play_and_record(seed)
            r2, t2 = play_and_record(seed)
            assert t1 == t2, f"seed {seed}: table not deterministic"
            for k in ("legal_mask", "chosen_card", "acting_seat", "epsilon",
                      "ply"):
                assert np.array_equal(r1[k], r2[k]), (
                    f"seed {seed}: {k} not deterministic")
            assert np.array_equal(r1["profiler_features"],
                                  r2["profiler_features"]), (
                f"seed {seed}: profiler_features not deterministic")
        print(f"determinism: 3 seeds reproduce identical tables + rows "
              f"across two independent runs")

        verify_hidden_hand_independence(
            range(SEED_BASE, SEED_BASE + 20))

        decisions_per_game = n_smoke_rows / n_games
        target_rows = 5_000_000
        games_needed = int(np.ceil(target_rows / decisions_per_game))
        proj_s = games_needed * elapsed / n_games
        print(f"decisions/game measured: {decisions_per_game:.3f}")
        print(f"games needed for >={target_rows} rows: {games_needed} "
              f"(projected {proj_s:.0f}s / {proj_s / 60:.1f} min at this "
              f"smoke's rate, includes JIT warmup so an underestimate of "
              f"true throughput)")


if __name__ == "__main__":
    main()
