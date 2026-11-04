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

from apps.table.api_contract import DEFAULT_BOT, SEAT_KINDS, SEAT_LABELS  # noqa: E402
from openhearts.belief.table import BeliefTable, Level  # noqa: E402
from openhearts.engine import cards  # noqa: E402
from openhearts.engine.game import deal, trick_points, trick_winner  # noqa: E402
from openhearts.eval.records import record_from, to_line  # noqa: E402
from openhearts.players.heuristic import HeuristicPlayer  # noqa: E402
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


def _make_bot(kind: str, rng):
    if kind == "search":
        return HonestSearchPlayer(Level.FULL, N_OUTER, N_INNER, rng, fused=True)
    if kind == "heuristic":
        return HeuristicPlayer()
    if kind == "random":
        return RandomPlayer(rng)
    raise ValueError(kind)


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

    # ------------------------------------------------------------ setup
    def n_humans(self) -> int:
        return sum(1 for s in self.seats if s["is_human"])

    def _rng(self, seat: int, salt: int):
        return np.random.default_rng([int(self.seed), int(seat), salt])

    def new_hand(self, seats, seed, advisor_on):
        if len(seats) != 4:
            raise ValueError("need exactly four seats")
        cfg = []
        for i, s in enumerate(seats):
            kind = str(s.get("kind", DEFAULT_BOT))
            if kind not in SEAT_KINDS:
                raise ValueError(f"unknown seat kind {kind!r}; choose from {list(SEAT_KINDS)}")
            name = str(s.get("name") or SEAT_LABELS[i])[:24]
            cfg.append({"seat": i, "kind": kind, "name": name, "is_human": kind == "human"})
        self.seats = cfg
        self.seed = int(seed) if seed is not None else int(np.random.SeedSequence().entropy % (1 << 31))
        self.hand_id += 1
        self.advisor_on = bool(advisor_on)
        self.state = deal(np.random.default_rng(self.seed))
        self.initial_hands = list(self.state.hands)
        self.bots = {i: _make_bot(s["kind"], self._rng(i, BOT_SALT))
                     for i, s in enumerate(self.seats) if not s["is_human"]}
        self.advisors = {i: HonestSearchPlayer(Level.FULL, N_OUTER, N_INNER,
                                               self._rng(i, ADVISOR_SALT), fused=True)
                         for i, s in enumerate(self.seats) if s["is_human"]}
        self.last_trick = None
        self.error = None
        self.undo_stack = []
        self._advice_cache = None
        self._logged = False
        self.phase = "playing"
        self.revealed = self.n_humans() <= 1
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
            self._apply(seat, card)
        if self.phase == "playing":
            self.revealed = self.n_humans() <= 1

    def human_turn(self) -> bool:
        return self.phase == "playing" and self.seats[self.state.to_play]["is_human"]

    def play(self, card_text: str):
        if not self.human_turn():
            raise ValueError("it is not a human's turn")
        if not self.revealed:
            raise ValueError("reveal the hand before playing")
        card = parse_card(card_text)
        seat = self.state.to_play
        view = self.state.view_for(seat)
        if not view.hand & cards.bit(card):
            raise ValueError(f"{card_str(card)} is not in {self.seats[seat]['name']}'s hand")
        if not view.legal_moves & cards.bit(card):
            raise ValueError(f"{card_str(card)}: {illegal_reason(view, card)}")
        self.undo_stack.append(self._snapshot())
        self._apply(seat, card)
        self._advance_bots()
        self._autosave()

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

    def set_advisor(self, on: bool):
        self.advisor_on = bool(on)
        self._advice_cache = None

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
                "rng": copy.deepcopy(self._rng_states()), "phase": self.phase}

    def _restore(self, snap):
        self.state = snap["state"].copy()
        self.last_trick = copy.deepcopy(snap["last_trick"])
        self._set_rng_states(snap["rng"])
        self.phase = snap["phase"]

    def _plays(self):
        return list(self.state.history) + list(self.state.current_trick)

    def _autosave(self):
        if self.state is None:
            return
        doc = {"seats": self.seats, "seed": self.seed, "hand_id": self.hand_id,
               "advisor_on": self.advisor_on, "initial_hands": [int(h) for h in self.initial_hands],
               "plays": [[int(s), int(c)] for s, c in self._plays()],
               "rng": self._rng_states(), "phase": self.phase}
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

    def resume(self):
        if not os.path.exists(self.resume_path):
            raise ValueError("nothing to resume")
        with open(self.resume_path) as f:
            doc = json.load(f)
        self.seats = doc["seats"]
        self.seed, self.hand_id, self.advisor_on = doc["seed"], doc["hand_id"], doc["advisor_on"]
        self.state = deal(np.random.default_rng(self.seed))
        assert [int(h) for h in self.state.hands] == doc["initial_hands"], "resume file does not match its seed"
        self.initial_hands = list(self.state.hands)
        self.bots = {i: _make_bot(s["kind"], self._rng(i, BOT_SALT))
                     for i, s in enumerate(self.seats) if not s["is_human"]}
        self.advisors = {i: HonestSearchPlayer(Level.FULL, N_OUTER, N_INNER,
                                               self._rng(i, ADVISOR_SALT), fused=True)
                         for i, s in enumerate(self.seats) if s["is_human"]}
        self.last_trick = None
        self.phase = "playing"
        self._logged = doc["phase"] == "over"       # never log the same hand twice
        for s, c in doc["plays"]:
            assert s == self.state.to_play
            self._apply(s, c)
        self._set_rng_states(doc["rng"])
        self.undo_stack = []
        self.error = None
        self.revealed = self.n_humans() <= 1
        self._advice_cache = None

    def _log_record(self):
        if self._logged:
            return
        rec = record_from(self.seed, self.initial_hands, self.state)
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
        base = {"phase": self.phase, "seats": self.seats, "to_play": None, "human_turn": False,
                "revealed": self.revealed, "trick_number": 0, "hearts_broken": False,
                "scores": [0, 0, 0, 0], "current_trick": [], "last_trick": None, "history": [],
                "hand": None, "legal": [], "illegal_reasons": {}, "advisor_on": self.advisor_on,
                "advice": None, "hand_id": self.hand_id, "seed": self.seed,
                "log_path": os.path.relpath(self.log_path, ROOT),
                "latency_budget_s": LATENCY_BUDGET_S, "error": self.error}
        if self.state is None:
            return base
        st = self.state
        base.update({
            "to_play": None if self.phase == "over" else st.to_play,
            "human_turn": self.human_turn(),
            "trick_number": st.trick_number, "hearts_broken": st.hearts_broken,
            "scores": list(st.scores),
            "current_trick": [{"seat": s, "card": card_str(c)} for s, c in st.current_trick],
            "last_trick": self.last_trick,
            "history": [{"seat": s, "card": card_str(c), "trick": i // 4}
                        for i, (s, c) in enumerate(st.history)],
        })
        if self.human_turn() and self.revealed:
            view = st.view_for(st.to_play)
            hand = cards.cards_in(view.hand)
            legal = cards.cards_in(view.legal_moves)
            base["hand"] = [card_str(c) for c in hand]
            base["legal"] = [card_str(c) for c in legal]
            base["illegal_reasons"] = {card_str(c): illegal_reason(view, c)
                                       for c in hand if c not in legal}
            if self.advisor_on:
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
                    t.new_hand(body.get("seats") or [], body.get("seed"), body.get("advisor_on", True))
                elif path == "/api/play":
                    t.play(body.get("card", ""))
                elif path == "/api/reveal":
                    t.reveal()
                elif path == "/api/undo":
                    t.undo()
                elif path == "/api/advisor":
                    t.set_advisor(body.get("on", True))
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
