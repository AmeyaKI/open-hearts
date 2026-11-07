"""THE TABLE (Phase 7.5a V0) -- the HTTP/JSON contract between apps/table/server.py and the page.

Server: Python standard library only (http.server.ThreadingHTTPServer), localhost, ONE table.
Cards are strings as `openhearts.engine.cards.card_name` prints them: rank char + suit char,
ranks "23456789TJQKA", suits "cdsh" (lower case). Examples: "2c" two of clubs, "Qs" queen of
spades, "Th" ten of hearts. Seats are 0..3 clockwise; seat s is followed by (s+1) % 4.
House rules: no passing, no moon; 2c leads trick 1; no penalty card on trick 1 unless forced;
hearts may be led only once broken (or when the hand is all hearts); a heart breaks hearts.

ENDPOINTS (all JSON; every POST returns the same object GET /api/state returns):

  GET  /                      the page (apps/table/static/index.html); /static/<file> assets
  GET  /api/state             current table state (shape below)
  POST /api/new               {"seats": [{"kind": K, "name": str}, x4], "seed": int|null,
                               "advisor_on": bool}  -> deals a new hand; bots play until the first
                               human must act. K in SEAT_KINDS. Seat 0 is at the bottom of the
                               screen, 1 left, 2 top, 3 right (clockwise = 0 -> 1 -> 2 -> 3).
  POST /api/play              {"card": "Qs"}  -> the human whose turn it is plays that card; then
                               bots play until the next human turn or the hand ends. 400 if the
                               card is illegal or it is not a human's turn.
  POST /api/reveal            {} -> hot-seat: show the current human's hand (state.revealed=true).
                               Each new human turn starts hidden when >1 human seat exists;
                               with exactly one human seat the hand is always revealed.
  POST /api/undo              {} -> take back plays until the previous HUMAN decision is
                               undone (bot plays in between are removed too). No-op at start.
  POST /api/advisor           {"on": bool} -> toggle the advisor for human seats.
  POST /api/resume            {} -> load results/live/table_resume.json (autosaved after
                               every play) and continue that hand.

STATE OBJECT (GET /api/state):
{
  "phase": "setup" | "playing" | "over",
  "seats": [{"seat": 0, "kind": "human", "name": "You", "is_human": true}, ...4],
  "to_play": 0..3 | null,                 # null when over or in setup
  "human_turn": bool,                      # true iff phase=playing and seats[to_play] is a KEYBOARD
                                           # human (kind "human"); false on a "table" seat's turn
  "revealed": bool,                        # hot-seat reveal state for the current human turn
  "trick_number": 0..13,                   # completed tricks
  "hearts_broken": bool,
  "scores": [int, int, int, int],          # points so far, by seat
  "current_trick": [{"seat": s, "card": "7h"}, ...],   # 0..3 plays so far this trick
  "last_trick": {"plays": [{"seat": s, "card": c} x4], "winner": s, "points": n} | null,
  "history": [{"seat": s, "card": c, "trick": t}, ...],  # all completed-trick plays in order
  "hand": ["2c", "9d", ...] | null,        # the current human's hand, ONLY when human_turn and
                                           # revealed; null otherwise (never another seat's hand)
  "legal": ["2c", ...],                    # legal cards for the current human (same condition)
  "illegal_reasons": {"Qs": "must follow clubs", ...},   # for the greyed cards in hand
  "advisor_on": bool,
  "advice": null | {                       # only when human_turn, revealed and advisor_on
      "suggested": "9d",
      "candidates": [{"card": "9d", "mean_points": 3.21}, ...],   # one per legal card, sorted
                                           # best (lowest mean points) first; the search's own
                                           # per-candidate averages over its imagined worlds
      "worlds": "50x20", "latency_s": 0.031,
      "belief": {
          "opponents": [{"seat": 1, "name": "West"}, {"seat": 2, ...}, {"seat": 3, ...}],
                                           # the current human's opponents, left, across, right
          "cards": {"Qs": [0.62, 0.38, 0.0], ...},    # P(opponent i holds card), unseen cards
                                           # only, in the same opponent order; columns sum to 1
          "voids": [["c"], [], ["s", "h"]]  # suits each opponent is known to be out of
      },
      "sentences": ["West showed void in clubs on trick 3, so every club is with North or East.", ...]
  },
  "hand_id": int, "seed": int,             # this hand's identity (bots are seeded from it)
  "log_path": "results/live/table_records.txt",
  "latency_budget_s": 3.0,
  "error": null | str                      # last rejected action's message, cleared on success
}

SEAT_KINDS: "human" (at the keyboard), "search" (the shipped bot, honest-FULL 50x20 fused --
DEFAULT for bot seats), "heuristic" (our beginner rulebook player), "random" (uniform legal).
Later (not V0): "table" (human at a real table, transcribed), "personality:<id>", "perilune".
"""
SEAT_KINDS = ("human", "table", "search", "heuristic", "random")   # plain kinds
PARAM_KINDS = ("expert", "personality", "perilune")                  # "kind:param"
BOT_NAME = "TallyHearts"                                                 # the main bot's product name
DEFAULT_BOT = "search"
SEAT_LABELS = {0: "South", 1: "West", 2: "North", 3: "East"}
CARD_RANKS = "23456789TJQKA"
CARD_SUITS = "cdsh"

# ---------------------------------------------------------------------------------------
# V1 ADDITIONS (2026-11-04): real-table seats, more bot kinds, divergence log.
#
# SEAT KINDS (string `kind`):
#   "human"            a person at this keyboard (hand dealt by the server in virtual mode,
#                      typed in by the person in real mode)
#   "table"            a person at a REAL table: the server never knows this hand; their plays
#                      are transcribed (clicked from the candidate list) as they happen
#   "search"           TallyHearts, the main bot (honest-FULL 50x20 fused)  -- DEFAULT bot
#   "heuristic"        the beginner rulebook player
#   "random"           uniform over legal cards
#   "expert:<id>"      Phase-7 expert rulebook player, id 1..200 (training pool ONLY; the
#                      held-out ids 201..250 are never offered)
#   "personality:<n>"  Phase-5 personality player drawn from seed n (any non-negative int)
#   "perilune:v5" / "perilune:v6.1"   Perilune's raw networks (offered only when their
#                      weights are installed; see experiments/perilune/adapter.py)
#   GET /api/kinds -> {"kinds": [{"kind": "search", "label": "TallyHearts (main bot)"}, ...]}
#   lists what this server can seat, in display order, with parametric kinds given once as
#   {"kind": "expert", "label": "Expert bot", "param": "id 1-200"} etc.
#
# MODES. mode = "virtual" (no "table" seat: the server deals) or "real" (at least one "table"
# seat: a physical deck was dealt). In REAL mode POST /api/new must also carry
#   "hands": {"0": ["2c", ...13 cards], "2": [...]}   for EVERY non-table seat
# (keyboard humans and bots); 13 cards each, no duplicates, and the server rejects a set that
# conflicts. Optional "leader": seat index, needed only when several table seats exist and
# no known hand holds 2c (with one table seat the leader is inferred). The state gains:
#   "mode": "virtual" | "real"
#   "table_turn": bool            true when the seat to play is a "table" seat
#   "candidates": ["3c", ...]     in real mode on a table seat's turn: every card that could
#                                 legally be the play (not yet played, not in a known hand,
#                                 and obeying the lead rules) -- the WIDEST legal set, so a true
#                                 play is never rejected for a reason the server cannot see
#   "announce": null | {"seat": s, "card": c, "name": n}
#                                 in real mode, the last BOT play: the person running the
#                                 table plays that card physically for the bot
# POST /api/play {"card": c} on a table seat's turn records that seat's physical play.
# Bots in real mode play automatically from their typed hand (announced), exactly as in
# virtual mode. Undo/resume/log work in both modes; the finished-hand record reconstructs
# every seat's 13 cards from its plays (no passing, so the union of plays IS the hand).
#
# DIVERGENCE LOG. When a keyboard human plays a card different from the advisor's suggestion,
# one line is appended to <results_dir>/table_divergence.txt:
#   hand_id seed seat trick suggested played mean_suggested mean_played
# and state carries "divergences": int (count this hand).

# ---------------------------------------------------------------------------------------
# V2 ADDITIONS (2026-11-06): per-seat coaching, presets, live-table flow.
#
# PER-SEAT ADVISOR. Each seat object sent to /api/new may carry "advisor": bool (default:
# true for seat 0, false for every other human seat). A keyboard human gets advice only when
# BOTH the global advisor_on and their seat's advisor flag are true. The state's "seats"
# entries echo "advisor". POST /api/advisor {"on": bool} still toggles the global switch;
# POST /api/advisor {"seat": s, "on": bool} toggles one seat.
#
# LEADER. In real mode, /api/new accepts "leader": seat index = who holds 2c and leads the
# first trick. Required whenever no typed-in hand holds 2c (i.e. a table player leads);
# the page asks "Who leads?" in that case.
#
# PRESETS (page-side only; the server sees ordinary seats):
#   "Solo vs bots"       you (keyboard, advised) + three bots
#   "Live table"         you (keyboard, advised; you type your 13 cards) + three "table" seats:
#                        the other people play real cards in front of you and you enter each
#                        card as it falls; TallyHearts suggests yours; you play it physically
#   "Pass the laptop"    four keyboard humans, server deals, hands hidden until a reveal tap,
#                        coaching off for everyone but seat 0 unless ticked per seat
#
# TYPED ENTRY. On a table seat's turn (and for a keyboard human's own play) the page accepts
# typing a card such as "7h", "Qs", "10d"/"Td" followed by Enter, as well as clicking.
