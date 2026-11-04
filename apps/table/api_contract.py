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
  "human_turn": bool,                      # true iff phase=playing and seats[to_play] is human
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
SEAT_KINDS = ("human", "search", "heuristic", "random")
DEFAULT_BOT = "search"
SEAT_LABELS = {0: "South", 1: "West", 2: "North", 3: "East"}
CARD_RANKS = "23456789TJQKA"
CARD_SUITS = "cdsh"
