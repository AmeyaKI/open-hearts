"""THE TABLE (Phase 7.5a V0) -- a local Hearts table in the browser: humans and bots in any mix.

Run:   .venv/bin/python apps/table/server.py [--port 8765] [--results-dir results/live]
Open:  http://127.0.0.1:8765/

Contract: apps/table/api_contract.py (the page and the tests are written against it).
Standard library only. ONE table per server. Localhost tool, no authentication.

DESIGN. The server holds the full GameState (all four hands) because in V0 every seat is at this
keyboard or is a bot; each seat only ever receives `state.view_for(seat)`, the same PlayerView
every benchmark player sees, so no bot or advisor is shown another seat's cards. (Transcribed
"human at a real table" seats -- where the server would NOT know a hand -- are the deferred
relay mode; the terminal relay in experiments/relay.py still covers that case.)

Bots are seeded from the hand's seed, so a hand replays exactly; undo restores the bots' random
streams along with the cards. Every finished hand is appended to <results_dir>/table_records.txt
in the project's GameRecord format; the table autosaves to <results_dir>/table_resume.json after
every play. The shipped bot is used read-only: nothing in engine/belief/search is modified.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from apps.table.api_contract import (BOT_NAME, DEFAULT_BOT, PARAM_KINDS, SEAT_KINDS,  # noqa: E402
                                     SEAT_LABELS)
from openhearts.belief.table import BeliefTable, Level  # noqa: E402
from openhearts.engine import cards  # noqa: E402
from openhearts.engine.game import deal, legal_moves, trick_points, trick_winner  # noqa: E402
from openhearts.engine.state import PlayerView  # noqa: E402
from openhearts.eval.records import record_from, to_line  # noqa: E402
from openhearts.players.expert import ExpertPlayer  # noqa: E402
from openhearts.players.expert_population import N_TRAIN, is_heldout, sample_expert  # noqa: E402
from openhearts.players.heuristic import HeuristicPlayer  # noqa: E402
from openhearts.players.personality import PersonalityPlayer, sample_personality  # noqa: E402
from openhearts.players.random_player import RandomPlayer  # noqa: E402
from openhearts.search.honest import HonestSearchPlayer  # noqa: E402

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
N_OUTER, N_INNER = 50, 20
LATENCY_BUDGET_S = 3.0
SUIT_NAMES = {0: "clubs", 1: "diamonds", 2: "spades", 3: "hearts"}
SUIT_CHARS = "cdsh"
BOT_SALT, ADVISOR_SALT = 7501, 7502


def card_str(c: int) -> str:
    return cards.card_name(c)


def parse_card(text: str) -> int:
    s = str(text).strip()
    if len(s) != 2 or s[0].upper() not in cards._RANKS or s[1].lower() not in cards._SUITS:
        raise ValueError(f"'{text}' is not a card like Qs or Th")
    return cards._SUITS.index(s[1].lower()) * 13 + cards._RANKS.index(s[0].upper())


def _perilune_available() -> bool:
    try:
        sys.path.insert(0, os.path.join(ROOT, "experiments"))
        from perilune import adapter
        return adapter.available()
    except Exception:
        return False


def available_kinds():
    """What this server can seat, in display order (GET /api/kinds)."""
    kinds = [{"kind": "human", "label": "Human (this keyboard)"},
             {"kind": "table", "label": "Human at the table"},
             {"kind": "search", "label": f"{BOT_NAME} (main bot)"},
             {"kind": "heuristic", "label": "Heuristic bot"},
             {"kind": "random", "label": "Random bot"},
             {"kind": "expert", "label": "Expert bot", "param": f"id 1-{N_TRAIN}"},
             {"kind": "personality", "label": "Personality bot", "param": "seed"}]
    if _perilune_available():
        kinds.append({"kind": "perilune", "label": "Perilune raw net", "param": "v5|v6.1"})
    return kinds


def parse_kind(kind: str):
    """'expert:17' -> ('expert', '17'); validates against the contract. Raises ValueError."""
    kind = str(kind)
    base, _, param = kind.partition(":")
    if base in SEAT_KINDS and not param:
        return base, None
    if base == "expert":
        try:
            i = int(param)
        except ValueError:
            raise ValueError(f"expert id must be a number, got {param!r}")
        if not 1 <= i <= N_TRAIN or is_heldout(i):
            raise ValueError(f"expert id must be 1-{N_TRAIN} (held-out experts are never seated)")
        return base, str(i)
    if base == "personality":
        try:
            n = int(param)
        except ValueError:
            raise ValueError(f"personality seed must be a number, got {param!r}")
        if n < 0:
            raise ValueError("personality seed must be non-negative")
        return base, str(n)
    if base == "perilune":
        if param not in ("v5", "v6.1"):
            raise ValueError("perilune net must be v5 or v6.1")
        if not _perilune_available():
            raise ValueError("Perilune weights are not installed (scripts/build_perilune.sh)")
        return base, param
    raise ValueError(f"unknown seat kind {kind!r}")


def _make_bot(kind: str, rng):
    base, param = parse_kind(kind)
    if base == "search":
        return HonestSearchPlayer(Level.FULL, N_OUTER, N_INNER, rng, fused=True)
    if base == "heuristic":
        return HeuristicPlayer()
    if base == "random":
        return RandomPlayer(rng)
    if base == "expert":
        return ExpertPlayer(rng, sample_expert(int(param)))
    if base == "personality":
        return PersonalityPlayer(rng, sample_personality(int(param)))
    if base == "perilune":
        from perilune import adapter
        return adapter.PeriluneRawPlayer(param)
    raise ValueError(kind)


class RealState:
    """A physical deal: the server knows only the typed-in hands of keyboard humans and bots.

    Same surface as engine.GameState as far as Table uses it (hands, history, current_trick,
    hearts_broken, trick_number, scores, to_play, is_over, copy, view_for, play) but `play`
    accepts a "table" seat's card from the WIDEST legal set (experiments/relay.py rule): every
    card not yet played and not in a known hand, with the lead rules applied when leading.
    Following suit is deliberately NOT enforced for a table seat -- a real void is common and
    undetectable from here, and a true play must never be rejected.
    """

    def __init__(self, known_hands: dict, table_seats: set, leader: int):
        self.hands = [int(known_hands.get(s, 0)) for s in range(4)]
        self.table_seats = set(table_seats)
        self.played_by = [0, 0, 0, 0]
        self.history, self.current_trick = [], []
        self.hearts_broken, self.trick_number = False, 0
        self.scores, self.to_play = [0, 0, 0, 0], leader

    def copy(self):
        c = RealState.__new__(RealState)
        c.__dict__.update({k: (list(v) if isinstance(v, list) else (set(v) if isinstance(v, set) else v))
                           for k, v in self.__dict__.items()})
        return c

    def is_over(self):
        return self.trick_number >= 13 and not self.current_trick

    def all_played(self):
        m = 0
        for _, c in self.history + self.current_trick:
            m |= cards.bit(c)
        return m

    def known_union(self):
        return self.hands[0] | self.hands[1] | self.hands[2] | self.hands[3]

    def candidates(self, seat):
        if seat not in self.table_seats:
            return legal_moves(self.hands[seat], tuple(self.current_trick), self.hearts_broken,
                               self.trick_number)
        pool = cards.FULL_DECK & ~self.all_played() & ~self.known_union()
        if not self.current_trick:
            return legal_moves(pool, (), self.hearts_broken, self.trick_number)
        return pool

    def view_for(self, seat):
        return PlayerView(seat=seat, hand=self.hands[seat], history=tuple(self.history),
                          current_trick=tuple(self.current_trick), hearts_broken=self.hearts_broken,
                          trick_number=self.trick_number, scores=tuple(self.scores),
                          legal_moves=self.candidates(seat))

    def play(self, card):
        seat = self.to_play
        assert self.candidates(seat) & cards.bit(card), f"illegal card {card} by seat {seat}"
        self.hands[seat] &= ~cards.bit(card)
        self.played_by[seat] |= cards.bit(card)
        self.current_trick.append((seat, card))
        if cards.suit(card) == cards.HEARTS:
            self.hearts_broken = True
        if len(self.current_trick) == 4:
            w = trick_winner(self.current_trick)
            self.scores[w] += trick_points(self.current_trick)
            self.history.extend(self.current_trick)
            self.current_trick = []
            self.trick_number += 1
            self.to_play = w
        else:
            self.to_play = (seat + 1) % 4


def illegal_reason(view, card: int) -> str:
    """Why `card` (in hand, not legal) cannot be played right now, in plain words."""
    if view.current_trick:
        led = cards.suit(view.current_trick[0][1])
        if cards.suit(card) != led and view.hand & cards.SUIT_MASK[led]:
            return f"must follow {SUIT_NAMES[led]}"
        if view.trick_number == 0 and cards.bit(card) & cards.POINTS_MASK:
            return "no points on the first trick"
        return "not legal here"
    if view.trick_number == 0:
        return "2♣ must lead the first trick"
    if cards.suit(card) == cards.HEARTS and not view.hearts_broken:
        return "hearts not broken yet"
    return "not legal here"


class Table:
    """One table. All mutations happen under `self.lock`."""

    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.lock = threading.Lock()
        self.phase = "setup"
        self.seats = [{"seat": i, "kind": "human" if i == 0 else DEFAULT_BOT,
                       "name": SEAT_LABELS[i], "is_human": i == 0} for i in range(4)]
        self.state = None
        self.initial_hands = None
        self.seed = 0
        self.hand_id = 0
        self.advisor_on = True
        self.revealed = True
        self.last_trick = None
        self.error = None
        self.bots = {}
        self.advisors = {}
        self.undo_stack = []
        self._advice_cache = None
        self._logged = False
        self.mode = "virtual"
        self.announce = None
        self.divergences = 0

    # ------------------------------------------------------------ setup
    def n_humans(self) -> int:
        return sum(1 for s in self.seats if s["is_human"])

    def n_keyboard_humans(self) -> int:
        return sum(1 for s in self.seats if s["kind"] == "human")

    def table_turn(self) -> bool:
        return self.phase == "playing" and self.seats[self.state.to_play]["kind"] == "table"

    def _rng(self, seat: int, salt: int):
        return np.random.default_rng([int(self.seed), int(seat), salt])

    def _parse_hands(self, hands, table_seats):
        """Real mode: {"seat": [13 card strings]} for every non-table seat; disjoint."""
        hands = hands or {}
        known, seen = {}, 0
        for i in range(4):
            if i in table_seats:
                continue
            lst = hands.get(str(i), hands.get(i))
            if not lst:
                raise ValueError(f"real-table hand needs the 13 cards of seat {i} ({self.seats[i]['name']})")
            mask = 0
            for text in lst:
                c = parse_card(text)
                if mask & cards.bit(c):
                    raise ValueError(f"{card_str(c)} listed twice for seat {i}")
                mask |= cards.bit(c)
            if bin(mask).count("1") != 13:
                raise ValueError(f"seat {i} needs exactly 13 cards, got {bin(mask).count('1')}")
            if mask & seen:
                dup = cards.cards_in(mask & seen)[0]
                raise ValueError(f"{card_str(dup)} is in two hands")
            seen |= mask
            known[i] = mask
        return known

    def new_hand(self, seats, seed, advisor_on, hands=None, leader=None):
        if len(seats) != 4:
            raise ValueError("need exactly four seats")
        cfg = []
        for i, s in enumerate(seats):
            kind = str(s.get("kind", DEFAULT_BOT))
            parse_kind(kind)                                    # raises on a bad kind
            name = str(s.get("name") or SEAT_LABELS[i])[:24]
            adv = s.get("advisor")
            if adv is None:
                adv = (i == 0)                              # coach seat 0 by default, nobody else
            cfg.append({"seat": i, "kind": kind, "name": name,
                        "is_human": kind in ("human", "table"), "advisor": bool(adv)})
        table_seats = {i for i, s in enumerate(cfg) if s["kind"] == "table"}
        self.seats = cfg
        self.seed = int(seed) if seed is not None else int(np.random.SeedSequence().entropy % (1 << 31))
        self.hand_id += 1
        self.advisor_on = bool(advisor_on)
        self.announce = None
        self.divergences = 0
        if table_seats:
            self.mode = "real"
            known = self._parse_hands(hands, table_seats)
            holder = [i for i, m in known.items() if m & cards.bit(cards.TWO_CLUBS)]
            if holder:
                lead = holder[0]
            elif len(table_seats) == 1:
                lead = next(iter(table_seats))
            elif leader is not None and int(leader) in table_seats:
                lead = int(leader)
            else:
                raise ValueError("several table seats and nobody known holds 2c: pass \"leader\"")
            self.state = RealState(known, table_seats, lead)
            self.initial_hands = None                           # reconstructed at the end
        else:
            self.mode = "virtual"
            self.state = deal(np.random.default_rng(self.seed))
            self.initial_hands = list(self.state.hands)
        self.bots = {i: _make_bot(s["kind"], self._rng(i, BOT_SALT))
                     for i, s in enumerate(self.seats) if not s["is_human"]}
        self.advisors = {i: HonestSearchPlayer(Level.FULL, N_OUTER, N_INNER,
                                               self._rng(i, ADVISOR_SALT), fused=True)
                         for i, s in enumerate(self.seats) if s["kind"] == "human"}
        self.last_trick = None
        self.error = None
        self.undo_stack = []
        self._advice_cache = None
        self._logged = False
        self.phase = "playing"
        self.revealed = self.n_keyboard_humans() <= 1
        self._advance_bots()
        self._autosave()

    # ------------------------------------------------------------ play
    def _apply(self, seat: int, card: int):
        st = self.state
        trick_before = list(st.current_trick) + [(seat, card)]
        st.play(card)
        if not st.current_trick:                                  # the trick just resolved
            self.last_trick = {"plays": [{"seat": s, "card": card_str(c)} for s, c in trick_before],
                               "winner": trick_winner(trick_before),
                               "points": trick_points(trick_before)}
        self._advice_cache = None
        if st.is_over():
            self.phase = "over"
            self._log_record()

    def _advance_bots(self):
        while self.phase == "playing" and not self.seats[self.state.to_play]["is_human"]:
            seat = self.state.to_play
            card = self.bots[seat].choose(self.state.view_for(seat))
            if self.mode == "real":
                self.announce = {"seat": seat, "card": card_str(card), "name": self.seats[seat]["name"]}
            self._apply(seat, card)
        if self.phase == "playing":
            self.revealed = self.n_keyboard_humans() <= 1

    def human_turn(self) -> bool:
        return self.phase == "playing" and self.seats[self.state.to_play]["is_human"]

    def play(self, card_text: str):
        if not self.human_turn():
            raise ValueError("it is not a human's turn")
        card = parse_card(card_text)
        seat = self.state.to_play
        view = self.state.view_for(seat)
        if self.table_turn():
            if not view.legal_moves & cards.bit(card):
                raise ValueError(f"{card_str(card)} cannot be {self.seats[seat]['name']}'s play: "
                                 "already played, held by a known seat, or an illegal lead")
        else:
            if not self.revealed:
                raise ValueError("reveal the hand before playing")
            if not view.hand & cards.bit(card):
                raise ValueError(f"{card_str(card)} is not in {self.seats[seat]['name']}'s hand")
            if not view.legal_moves & cards.bit(card):
                raise ValueError(f"{card_str(card)}: {illegal_reason(view, card)}")
            self._log_divergence(seat, card)
        self.undo_stack.append(self._snapshot())
        self._apply(seat, card)
        self._advance_bots()
        self._autosave()

    def _log_divergence(self, seat, card):
        """Keyboard human played against the advice that was on screen -> one log line."""
        adv = self._advice_cache
        if not self.coached(seat) or adv is None or adv["suggested"] == card_str(card):
            return
        means = {c["card"]: c["mean_points"] for c in adv["candidates"]}
        with open(self.divergence_path, "a") as f:
            f.write(f"{self.hand_id} {self.seed} {seat} {self.state.trick_number + 1} "
                    f"{adv['suggested']} {card_str(card)} {means.get(adv['suggested'], 'na')} "
                    f"{means.get(card_str(card), 'na')}\n")
        self.divergences += 1

    def reveal(self):
        if self.human_turn():
            self.revealed = True

    def undo(self):
        if not self.undo_stack:
            return
        self._restore(self.undo_stack.pop())
        self.revealed = True                   # the player who is undoing is at the keyboard
        self._advice_cache = None
        self._autosave()

    def set_advisor(self, on: bool, seat=None):
        if seat is None:
            self.advisor_on = bool(on)
        else:
            seat = int(seat)
            if not 0 <= seat < 4:
                raise ValueError("seat must be 0-3")
            self.seats[seat]["advisor"] = bool(on)
        self._advice_cache = None

    def coached(self, seat) -> bool:
        return self.advisor_on and bool(self.seats[seat].get("advisor", seat == 0))

    # ------------------------------------------------------------ snapshots / persistence
    def _rng_states(self):
        out = {}
        for i, b in self.bots.items():
            rng = getattr(b, "rng", None)
            if rng is not None:
                out[f"bot{i}"] = rng.bit_generator.state
        for i, a in self.advisors.items():
            out[f"adv{i}"] = a.rng.bit_generator.state
        return out

    def _set_rng_states(self, states):
        for key, st in states.items():
            idx = int(key[3:])
            obj = self.bots[idx] if key.startswith("bot") else self.advisors[idx]
            obj.rng.bit_generator.state = st

    def _snapshot(self):
        return {"state": self.state.copy(), "last_trick": copy.deepcopy(self.last_trick),
                "rng": copy.deepcopy(self._rng_states()), "phase": self.phase,
                "announce": copy.deepcopy(self.announce), "divergences": self.divergences}

    def _restore(self, snap):
        self.state = snap["state"].copy()
        self.last_trick = copy.deepcopy(snap["last_trick"])
        self._set_rng_states(snap["rng"])
        self.phase = snap["phase"]
        self.announce = snap.get("announce")
        self.divergences = snap.get("divergences", 0)

    def _plays(self):
        return list(self.state.history) + list(self.state.current_trick)

    def _autosave(self):
        if self.state is None:
            return
        doc = {"seats": self.seats, "seed": self.seed, "hand_id": self.hand_id,
               "advisor_on": self.advisor_on, "mode": self.mode,
               "initial_hands": [int(h) for h in self.initial_hands] if self.initial_hands else None,
               "known_hands": ({str(s): int(self.state.hands[s] | self.state.played_by[s])
                                for s in range(4) if s not in self.state.table_seats}
                               if self.mode == "real" else None),
               "table_seats": sorted(getattr(self.state, "table_seats", [])),
               "leader": (self._plays()[0][0] if self._plays() else self.state.to_play),
               "plays": [[int(s), int(c)] for s, c in self._plays()],
               "rng": self._rng_states(), "phase": self.phase, "divergences": self.divergences}
        tmp = self.resume_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(doc, f)
        os.replace(tmp, self.resume_path)

    @property
    def resume_path(self):
        return os.path.join(self.results_dir, "table_resume.json")

    @property
    def log_path(self):
        return os.path.join(self.results_dir, "table_records.txt")

    @property
    def divergence_path(self):
        return os.path.join(self.results_dir, "table_divergence.txt")

    def resume(self):
        if not os.path.exists(self.resume_path):
            raise ValueError("nothing to resume")
        with open(self.resume_path) as f:
            doc = json.load(f)
        self.seats = [dict(x, advisor=x.get("advisor", x["seat"] == 0)) for x in doc["seats"]]
        self.seed, self.hand_id, self.advisor_on = doc["seed"], doc["hand_id"], doc["advisor_on"]
        self.mode = doc.get("mode", "virtual")
        self.announce = None
        self.divergences = doc.get("divergences", 0)
        if self.mode == "real":
            known = {int(k): int(v) for k, v in doc["known_hands"].items()}
            self.state = RealState(known, set(doc["table_seats"]), int(doc["leader"]))
            self.initial_hands = None
        else:
            self.state = deal(np.random.default_rng(self.seed))
            assert [int(h) for h in self.state.hands] == doc["initial_hands"], "resume file does not match its seed"
            self.initial_hands = list(self.state.hands)
        self.bots = {i: _make_bot(s["kind"], self._rng(i, BOT_SALT))
                     for i, s in enumerate(self.seats) if not s["is_human"]}
        self.advisors = {i: HonestSearchPlayer(Level.FULL, N_OUTER, N_INNER,
                                               self._rng(i, ADVISOR_SALT), fused=True)
                         for i, s in enumerate(self.seats) if s["kind"] == "human"}
        self.last_trick = None
        self.phase = "playing"
        self._logged = doc["phase"] == "over"       # never log the same hand twice
        for s, c in doc["plays"]:
            assert s == self.state.to_play
            self._apply(s, c)
        self._set_rng_states(doc["rng"])
        self.undo_stack = []
        self.error = None
        self.revealed = self.n_keyboard_humans() <= 1
        self._advice_cache = None

    def _log_record(self):
        if self._logged:
            return
        hands = self.initial_hands if self.initial_hands is not None else list(self.state.played_by)
        rec = record_from(self.seed, hands, self.state)
        with open(self.log_path, "a") as f:
            f.write(to_line(rec) + "\n")
        self._logged = True

    # ------------------------------------------------------------ advice
    def _advice(self):
        if self._advice_cache is not None:
            return self._advice_cache
        seat = self.state.to_play
        view = self.state.view_for(seat)
        legal = cards.cards_in(view.legal_moves)
        adv = self.advisors[seat]
        t0 = time.perf_counter()
        suggested = adv.choose(view)
        latency = time.perf_counter() - t0
        cand, avgs = adv.last_candidates, adv.last_avgs
        if len(legal) == 1 or cand is None or avgs is None or len(cand) != len(avgs):
            cand, avgs = legal, [0.0] * len(legal)
            worlds = "forced" if len(legal) == 1 else "fallback"
        else:
            worlds = f"{N_OUTER}x{N_INNER}"
        order = sorted(range(len(cand)), key=lambda i: (float(avgs[i]), cand[i]))
        candidates = [{"card": card_str(cand[i]), "mean_points": round(float(avgs[i]), 3)}
                      for i in order]
        table = BeliefTable.from_view(view, Level.FULL)
        belief = {
            "opponents": [{"seat": s, "name": self.seats[s]["name"]} for s in table.opponent_seats],
            "cards": {card_str(c): [round(float(table.probs[i, c]), 4) for i in range(3)]
                      for c in cards.cards_in(table.unseen_mask)},
            "voids": [[SUIT_CHARS[s] for s in sorted(v)] for v in table.voids],
        }
        try:
            from apps.table.explain import sentences as _sentences
            names = [s["name"] for s in self.seats]
            text = list(_sentences(view, table, list(cand), [float(a) for a in avgs], names))
        except Exception as exc:  # the explain module is optional; never break play over a sentence
            text = [f"(reasoning unavailable: {type(exc).__name__})"]
        self._advice_cache = {"suggested": card_str(suggested), "candidates": candidates,
                              "worlds": worlds, "latency_s": round(latency, 4),
                              "belief": belief, "sentences": text}
        return self._advice_cache

    # ------------------------------------------------------------ state
    def snapshot_json(self):
        seats_json = [dict(x, is_human=(x["kind"] == "human")) for x in self.seats]   # "table" seats: false
        base = {"phase": self.phase, "seats": seats_json, "to_play": None, "human_turn": False,
                "revealed": self.revealed, "trick_number": 0, "hearts_broken": False,
                "scores": [0, 0, 0, 0], "current_trick": [], "last_trick": None, "history": [],
                "hand": None, "legal": [], "illegal_reasons": {}, "advisor_on": self.advisor_on,
                "advice": None, "hand_id": self.hand_id, "seed": self.seed,
                "log_path": os.path.relpath(self.log_path, ROOT),
                "latency_budget_s": LATENCY_BUDGET_S, "error": self.error,
                "mode": self.mode, "table_turn": False, "candidates": [], "announce": None,
                "divergences": self.divergences}
        if self.state is None:
            return base
        st = self.state
        base.update({
            "to_play": None if self.phase == "over" else st.to_play,
            "human_turn": self.human_turn() and not self.table_turn(),   # keyboard human to act
            "trick_number": st.trick_number, "hearts_broken": st.hearts_broken,
            "scores": list(st.scores),
            "current_trick": [{"seat": s, "card": card_str(c)} for s, c in st.current_trick],
            "last_trick": self.last_trick,
            "history": [{"seat": s, "card": card_str(c), "trick": i // 4}
                        for i, (s, c) in enumerate(st.history)],
        })
        base["announce"] = self.announce if self.mode == "real" else None
        if self.table_turn():
            base["table_turn"] = True
            base["candidates"] = [card_str(c) for c in cards.cards_in(st.candidates(st.to_play))]
            return base
        if self.human_turn() and self.revealed:
            view = st.view_for(st.to_play)
            hand = cards.cards_in(view.hand)
            legal = cards.cards_in(view.legal_moves)
            base["hand"] = [card_str(c) for c in hand]
            base["legal"] = [card_str(c) for c in legal]
            base["illegal_reasons"] = {card_str(c): illegal_reason(view, c)
                                       for c in hand if c not in legal}
            if self.coached(st.to_play):
                base["advice"] = self._advice()
        return base


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "OpenHeartsTable/0.1"

    def log_message(self, fmt, *args):      # quiet by default; set TABLE_VERBOSE=1 to see requests
        if os.environ.get("TABLE_VERBOSE"):
            super().log_message(fmt, *args)

    @property
    def table(self) -> Table:
        return self.server.table

    def _send_json(self, obj, status=HTTPStatus.OK):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, rel):
        path = os.path.normpath(os.path.join(STATIC_DIR, rel))
        if not path.startswith(STATIC_DIR) or not os.path.isfile(path):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        ctype = {".html": "text/html; charset=utf-8", ".css": "text/css", ".js": "application/javascript",
                 ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png"}.get(
            os.path.splitext(path)[1], "application/octet-stream")
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/state":
            with self.table.lock:
                self._send_json(self.table.snapshot_json())
        elif path == "/api/kinds":
            self._send_json({"kinds": available_kinds()})
        elif path == "/":
            self._send_file("index.html")
        elif path.startswith("/static/"):
            self._send_file(path[len("/static/"):])
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            body = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            self._send_json({"error": "body is not JSON"}, HTTPStatus.BAD_REQUEST)
            return
        t = self.table
        with t.lock:
            try:
                if path == "/api/new":
                    t.new_hand(body.get("seats") or [], body.get("seed"), body.get("advisor_on", True),
                               hands=body.get("hands"), leader=body.get("leader"))
                elif path == "/api/play":
                    t.play(body.get("card", ""))
                elif path == "/api/reveal":
                    t.reveal()
                elif path == "/api/undo":
                    t.undo()
                elif path == "/api/advisor":
                    t.set_advisor(body.get("on", True), body.get("seat"))
                elif path == "/api/resume":
                    t.resume()
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                t.error = None
                self._send_json(t.snapshot_json())
            except (ValueError, AssertionError) as exc:
                t.error = str(exc)
                self._send_json(t.snapshot_json(), HTTPStatus.BAD_REQUEST)


def make_server(host="127.0.0.1", port=8765, results_dir=None) -> ThreadingHTTPServer:
    results_dir = results_dir or os.path.join(ROOT, "results", "live")
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    httpd.table = Table(results_dir)
    return httpd


def main(argv=None):
    ap = argparse.ArgumentParser(description="The Table -- local Hearts dashboard")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args(argv)
    httpd = make_server(args.host, args.port, args.results_dir)
    host, port = httpd.server_address[:2]
    print(f"The Table: http://{host}:{port}/   (records -> {httpd.table.log_path}; Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
