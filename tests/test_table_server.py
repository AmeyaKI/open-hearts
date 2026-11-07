"""Tests for The Table dashboard server (apps/table/server.py), against apps/table/api_contract.py."""
import json
import os
import sys
import threading
import urllib.error
import urllib.request

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from apps.table import server  # noqa: E402
from openhearts.eval.records import from_line  # noqa: E402

RANKS = "23456789TJQKA"
SUITS = "cdsh"
ALL_CARDS = [r + s for s in SUITS for r in RANKS]
STATE_KEYS = [
    "phase", "seats", "to_play", "human_turn", "revealed", "trick_number", "hearts_broken",
    "scores", "current_trick", "last_trick", "history", "hand", "legal", "illegal_reasons",
    "advisor_on", "advice", "hand_id", "seed", "log_path", "latency_budget_s", "error",
]


def card_int(c):
    return 13 * SUITS.index(c[1]) + RANKS.index(c[0])


def card_points(c):
    return 13 if c == "Qs" else (1 if c[1] == "h" else 0)


class Client:
    def __init__(self, httpd):
        self.base = "http://%s:%d" % httpd.server_address[:2]

    def _req(self, path, obj=None):
        data = None if obj is None else json.dumps(obj).encode()
        req = urllib.request.Request(self.base + path, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    def get(self, path):
        try:
            return self._req(path)
        except urllib.error.HTTPError as e:
            raise AssertionError("GET %s -> %d: %s" % (path, e.code, e.read().decode()))

    def post(self, path, obj=None):
        try:
            return self._req(path, {} if obj is None else obj)
        except urllib.error.HTTPError as e:
            raise AssertionError("POST %s -> %d: %s" % (path, e.code, e.read().decode()))

    def post_expect_400(self, path, obj=None):
        try:
            self._req(path, {} if obj is None else obj)
        except urllib.error.HTTPError as e:
            assert e.code == 400, "expected 400, got %d: %s" % (e.code, e.read().decode())
            return
        raise AssertionError("POST %s expected 400 but succeeded" % path)


def _start(results_dir):
    httpd = server.make_server(host="127.0.0.1", port=0, results_dir=str(results_dir))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, t


def _stop(httpd, t):
    httpd.shutdown()
    httpd.server_close()
    t.join(timeout=5)


@pytest.fixture
def started(tmp_path):
    httpd, t = _start(tmp_path)
    yield Client(httpd), tmp_path
    _stop(httpd, t)


@pytest.fixture
def api(started):
    return started[0]


def seats(*kinds):
    names = ["Me", "West", "North", "East"]
    return [{"kind": k, "name": names[i] if k != "human" or i == 0 else "Human%d" % i}
            for i, k in enumerate(kinds)]


def new_hand(api, kinds, seed, advisor_on=False):
    return api.post("/api/new", {"seats": seats(*kinds), "seed": seed, "advisor_on": advisor_on})


def play_first_legal(api, st):
    if not st["revealed"]:
        st = api.post("/api/reveal", {})
    return api.post("/api/play", {"card": st["legal"][0]})


def play_until_over(api, st, limit=200):
    n = 0
    while st["phase"] != "over":
        assert st["phase"] == "playing" and st["human_turn"]
        st = play_first_legal(api, st)
        n += 1
        assert n < limit
    return st


def test_state_before_new_is_setup(api):
    st = api.get("/api/state")
    for k in STATE_KEYS:
        assert k in st, k
    assert st["phase"] == "setup"
    assert st["to_play"] is None
    assert st["human_turn"] is False
    assert st["hand"] is None
    assert st["advice"] is None


def test_new_hand_with_three_random_bots_reaches_first_human_turn(api):
    st = new_hand(api, ["human", "random", "random", "random"], 123)
    for k in STATE_KEYS:
        assert k in st, k
    assert st["phase"] == "playing"
    assert st["human_turn"] is True
    assert st["revealed"] is True
    assert st["seed"] == 123
    human = st["to_play"]
    assert st["seats"][human]["is_human"] and st["seats"][human]["kind"] == "human"
    played = len(st["history"]) + len(st["current_trick"])
    assert isinstance(st["hand"], list)
    mine = sum(1 for p in st["history"] if p["seat"] == human)
    assert len(st["hand"]) == 13 - mine
    assert st["legal"] and set(st["legal"]) <= set(st["hand"])
    for c in set(st["hand"]) - set(st["legal"]):
        assert c in st["illegal_reasons"]
    assert 0 <= len(st["current_trick"]) <= 3
    assert all(p["seat"] != human for p in st["current_trick"])
    assert len(st["scores"]) == 4
    assert sum(st["scores"]) == sum(card_points(p["card"]) for p in st["history"])
    assert played < 52


def test_illegal_card_is_rejected_with_400_and_error_message(api):
    st = new_hand(api, ["human", "random", "random", "random"], 123)
    bad = None
    for _ in range(13):
        bad = [c for c in st["hand"] if c not in st["legal"]]
        if bad:
            break
        st = api.post("/api/play", {"card": st["legal"][0]})
        if st["phase"] == "over":
            break
    if not bad:
        pytest.skip("no illegal card ever appeared in hand")
    api.post_expect_400("/api/play", {"card": bad[0]})
    assert api.get("/api/state")["error"]
    st = api.get("/api/state")
    st = api.post("/api/play", {"card": st["legal"][0]})
    assert st["error"] is None
    assert api.get("/api/state")["error"] is None


def test_full_hand_random_bots_completes_and_logs_record(started):
    api, results = started
    st = new_hand(api, ["human", "random", "random", "random"], 123)
    st = play_until_over(api, st)
    assert st["phase"] == "over"
    assert sum(st["scores"]) == 26
    assert st["trick_number"] == 13
    assert len(st["history"]) == 52
    assert all({"seat", "card", "trick"} <= set(h) for h in st["history"])
    for s in range(4):
        assert sum(1 for h in st["history"] if h["seat"] == s) == 13
    cards = [h["card"] for h in st["history"]]
    assert len(set(cards)) == 52
    path = os.path.join(str(results), "table_records.txt")
    assert os.path.exists(path)
    lines = [ln for ln in open(path).read().splitlines() if ln.strip()]
    assert len(lines) == 1
    rec = from_line(lines[0])
    assert list(rec.plays) == [(h["seat"], card_int(h["card"])) for h in st["history"]]
    assert list(rec.scores) == st["scores"]


def test_undo_restores_previous_human_decision(api):
    st0 = new_hand(api, ["human", "random", "random", "random"], 123)
    # undo before any human play is a no-op
    same = api.post("/api/undo", {})
    for k in ("hand", "history", "current_trick", "to_play", "scores", "trick_number"):
        assert same[k] == st0[k], k
    after = api.post("/api/play", {"card": st0["legal"][0]})
    assert after["hand"] != st0["hand"] or after["phase"] == "over"
    back = api.post("/api/undo", {})
    assert back["hand"] == st0["hand"]
    assert back["history"] == st0["history"]
    assert back["current_trick"] == st0["current_trick"]
    assert back["to_play"] == st0["to_play"]
    assert back["human_turn"] is True
    assert back["error"] is None


def test_resume_continues_same_hand(tmp_path):
    h1, t1 = _start(tmp_path)
    try:
        api = Client(h1)
        st = new_hand(api, ["human", "random", "random", "random"], 321)
        for _ in range(5):
            st = api.post("/api/play", {"card": st["legal"][0]})
        captured = api.get("/api/state")
    finally:
        _stop(h1, t1)
    h2, t2 = _start(tmp_path)
    try:
        api2 = Client(h2)
        res = api2.post("/api/resume", {})
        for k in ("hand", "history", "current_trick", "scores", "trick_number", "seed", "hand_id"):
            assert res[k] == captured[k], k
        assert res["phase"] == "playing"
    finally:
        _stop(h2, t2)


def test_hot_seat_hides_hand_until_reveal(api):
    st = new_hand(api, ["human", "human", "random", "random"], 5)
    assert st["phase"] == "playing" and st["human_turn"]
    assert st["revealed"] is False
    assert st["hand"] is None
    assert st["legal"] == []
    assert st["advice"] is None
    api.post_expect_400("/api/play", {"card": "2c"})
    st = api.post("/api/reveal", {})
    assert st["revealed"] is True
    assert st["hand"] is not None and st["legal"]
    st = api.post("/api/play", {"card": st["legal"][0]})
    assert st["phase"] == "playing" and st["human_turn"]
    assert st["revealed"] is False
    assert st["hand"] is None


def test_advisor_fields_when_on(api):
    st = new_hand(api, ["human", "heuristic", "heuristic", "heuristic"], 7, advisor_on=True)
    me = st["to_play"]
    adv = st["advice"]
    assert adv is not None
    assert adv["suggested"] in st["legal"]
    cands = adv["candidates"]
    assert sorted(c["card"] for c in cands) == sorted(st["legal"])
    means = [c["mean_points"] for c in cands]
    assert means == sorted(means)
    assert adv["suggested"] == cands[0]["card"]
    assert adv["worlds"] == "50x20"
    assert adv["latency_s"] >= 0
    bel = adv["belief"]
    assert [o["seat"] for o in bel["opponents"]] == [(me + 1) % 4, (me + 2) % 4, (me + 3) % 4]
    played = {h["card"] for h in st["history"]} | {p["card"] for p in st["current_trick"]}
    unseen = set(ALL_CARDS) - set(st["hand"]) - played
    assert set(bel["cards"].keys()) == unseen
    for c, probs in bel["cards"].items():
        assert len(probs) == 3
        assert abs(sum(probs) - 1.0) < 1e-6, c
    assert len(bel["voids"]) == 3
    for v in bel["voids"]:
        assert isinstance(v, list) and all(s in SUITS for s in v)
    assert isinstance(adv["sentences"], list)
    assert all(isinstance(s, str) and s for s in adv["sentences"])
    st = api.post("/api/advisor", {"on": False})
    assert st["advice"] is None
    st = api.post("/api/advisor", {"on": True})
    assert st["advice"] is not None


def test_search_bots_play_a_full_hand(api):
    st = new_hand(api, ["human", "search", "search", "search"], 11)
    st = play_until_over(api, st)
    assert sum(st["scores"]) == 26


def test_seed_determinism(tmp_path):
    out = []
    for i in range(2):
        d = tmp_path / ("r%d" % i)
        d.mkdir()
        httpd, t = _start(d)
        try:
            st = new_hand(Client(httpd), ["human", "heuristic", "heuristic", "heuristic"], 99)
            out.append((st["hand"], st["current_trick"]))
        finally:
            _stop(httpd, t)
    assert out[0] == out[1]


# ------------------------------------------------------------------ V1: kinds, real table, divergence
import numpy as np  # noqa: E402

from openhearts.engine import cards as _cards  # noqa: E402
from openhearts.engine.game import deal as _deal  # noqa: E402


def test_kinds_endpoint_lists_contract_kinds_and_never_heldout_experts(api):
    kinds = api.get("/api/kinds")["kinds"]
    names = [k["kind"] for k in kinds]
    for k in ("human", "table", "search", "heuristic", "random", "expert", "personality"):
        assert k in names
    expert = next(k for k in kinds if k["kind"] == "expert")
    assert expert["param"] == "id 1-200"
    assert any("TallyHearts" in k["label"] for k in kinds)
    for bad in ("expert:201", "expert:250", "expert:0", "personality:-3", "perilune:v9", "wizard"):
        api.post_expect_400("/api/new", {"seats": seats("human", bad, "random", "random"), "seed": 1})


def test_expert_and_personality_bots_play_a_full_hand(api):
    st = new_hand(api, ["human", "expert:17", "personality:4242", "heuristic"], 21)
    st = play_until_over(api, st)
    assert sum(st["scores"]) == 26 and st["mode"] == "virtual"


def test_divergence_log_counts_plays_against_the_advice(started):
    api, results_dir = started
    st = new_hand(api, ["human", "heuristic", "heuristic", "heuristic"], 7, advisor_on=True)
    n = 0
    while st["phase"] == "playing":
        adv = st["advice"]
        others = [c for c in st["legal"] if c != adv["suggested"]]
        card = others[0] if others else st["legal"][0]
        n += 1 if others else 0
        st = api.post("/api/play", {"card": card})
    assert st["divergences"] == n > 0
    lines = (results_dir / "table_divergence.txt").read_text().splitlines()
    assert len(lines) == n
    hand_id, seed, seat, trick, sugg, played, m1, m2 = lines[0].split()
    assert int(seed) == 7 and seat == "0" and sugg != played and 1 <= int(trick) <= 13


def _real_setup(table_seat=1, seed=3):
    d = _deal(np.random.default_rng(seed))
    hands = {str(s): [_cards.card_name(c) for c in _cards.cards_in(d.hands[s])]
             for s in range(4) if s != table_seat}
    truth = [_cards.card_name(c) for c in _cards.cards_in(d.hands[table_seat])]
    return hands, truth


def _table_play(api, st, truth):
    """Transcribe a truly legal card for the table seat (follow suit when it can)."""
    played = {h["card"] for h in st["history"]} | {p["card"] for p in st["current_trick"]}
    held = [c for c in truth if c not in played]
    led = st["current_trick"][0]["card"][1] if st["current_trick"] else None
    follow = [c for c in held if c[1] == led] if led else []
    pool = follow or [c for c in held if c in st["candidates"]]
    assert pool, "the true hand always has a candidate"
    return api.post("/api/play", {"card": pool[0]})


def test_real_table_seat_is_transcribed_and_the_record_reconstructs_every_hand(started):
    api, results_dir = started
    hands, truth = _real_setup()
    sts = seats("human", "table", "search", "heuristic")
    api.post_expect_400("/api/new", {"seats": sts, "seed": 3})                    # hands missing
    short = dict(hands); short["0"] = short["0"][:12]
    api.post_expect_400("/api/new", {"seats": sts, "seed": 3, "hands": short})    # 12 cards
    st = api.post("/api/new", {"seats": sts, "seed": 3, "hands": hands, "advisor_on": True})
    assert st["mode"] == "real" and st["phase"] == "playing"
    assert (not st["seats"][1]["is_human"]) and st["seats"][1]["kind"] == "table"
    saw_table_turn = saw_announce = False
    while st["phase"] == "playing":
        if st["table_turn"]:
            saw_table_turn = True
            assert st["hand"] is None and st["legal"] == [] and st["advice"] is None
            assert not st["human_turn"]
            assert all(c not in hands["0"] + hands["2"] + hands["3"] for c in st["candidates"])
            api.post_expect_400("/api/play", {"card": hands["0"][0]})     # a known seat's card
            st = _table_play(api, st, truth)
        else:
            assert st["human_turn"] and st["to_play"] == 0
            if st["announce"]:
                saw_announce = True
                assert st["announce"]["seat"] in (2, 3)
            st = play_first_legal(api, st)
    assert saw_table_turn and saw_announce and sum(st["scores"]) == 26
    rec = from_line((results_dir / "table_records.txt").read_text().splitlines()[-1])
    assert all(bin(h).count("1") == 13 for h in rec.hands)
    assert sorted(_cards.card_name(c) for c in _cards.cards_in(rec.hands[1])) == sorted(truth)


def test_real_table_resume_mid_hand(tmp_path):
    hands, truth = _real_setup()
    httpd, t = _start(tmp_path)
    api = Client(httpd)
    st = api.post("/api/new", {"seats": seats("human", "table", "search", "heuristic"),
                               "seed": 3, "hands": hands})
    for _ in range(6):
        st = _table_play(api, st, truth) if st["table_turn"] else play_first_legal(api, st)
    _stop(httpd, t)
    httpd2, t2 = _start(tmp_path)
    try:
        st2 = Client(httpd2).post("/api/resume", {})
        for k in ("mode", "history", "current_trick", "scores", "trick_number", "to_play",
                  "hand", "candidates", "table_turn", "seed", "hand_id"):
            assert st2[k] == st[k], k
    finally:
        _stop(httpd2, t2)
