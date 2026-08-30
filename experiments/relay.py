"""Phase 7 Task F1 (6D): the live-play relay.

WHAT THIS IS. A terminal tool for logging REAL Hearts games as they are
played at a real table. House rules only: no passing phase, no shoot-the-
moon rescoring (standard per-trick scoring), trick 0 leads the 2 of clubs.
One seat is "known" to the tool -- the owner's seat in `--advise` mode, or
the bot's own seat in `--seat` mode -- and its hand is entered once at the
start of the hand. All other seats' plays are transcribed by the owner as
they happen at the table; this tool never assumes it knows an unseen hand,
so opponent-entered cards are legality-checked against the WIDEST hand they
could still hold (every card not yet played and not held by the known seat)
-- a strict superset of their true hand, so a truly illegal entry (leading
the wrong card on trick 0, leading a heart before hearts are broken, etc)
is always caught, while a true play is never wrongly rejected for reasons
this tool cannot see (e.g. failing to follow a suit it does not hold).

TWO MODES.
  --advise   The bot (honest-FULL) suggests a card for the known seat; the
             owner then types in what was ACTUALLY played at the table --
             which may differ from the suggestion. That divergence is the
             whole point: this is an advisor, not an autoplayer.
  --seat     The bot plays the known seat itself and announces its move;
             the owner transcribes the other three seats' real plays.

BOT. `HonestSearchPlayer(Level.FULL, n_outer=50, n_inner=20)` -- "honest-
FULL", the reigning champion (Phase 6). Plain honest belief, no
posterior_factory. Every `.choose()` call is timed with
`time.perf_counter()`; the latency is printed, and a budget of <=3s/decision
is reported (never hard-failed -- the point is measurement, not a gate on
whatever hardware happens to be running).

LOGGING. Standard GameRecord rows (`src/openhearts/eval/records.py`
format), appended (never overwritten) to `results/live/relay_records.txt`.
Because Hearts has no passing/discarding, a seat's initial 13-card hand
equals the union of every card that seat plays over the whole hand -- so
even though only the KNOWN seat's starting hand is ever typed in, the
record's `hands` field is reconstructed exactly for all four seats once the
hand completes, and is not lossy. A separate alias-only sidecar row goes to
`results/live/relay_sidecar.txt` per session: date, per-seat ALIASES (never
real names -- that is the whole point of the sidecar), mode, bot config,
notes. `results/` is already gitignored (blanket rule) so `results/live/`
needs no separate entry.

MID-HAND RESUME. After every single play the full hand state is
autosaved to a resume file (default `results/live/relay_resume.json`, or
`--resume <path>`). Passing `--resume <path>` to a NEW invocation loads
that file and continues the hand instead of asking for a fresh deal --
this is what makes a hand survive a typo, a closed terminal, or a crash.

UNDO. Typing `undo` at any card-entry prompt reverts the single most
recent play (whoever made it) and re-prompts for that seat.

No hand target and no clock -- every invocation logs opportunistically.
"""
import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from openhearts.belief.table import Level  # noqa: E402
from openhearts.engine import cards  # noqa: E402
from openhearts.engine.game import legal_moves, trick_points, trick_winner  # noqa: E402
from openhearts.engine.state import PlayerView  # noqa: E402
from openhearts.eval.records import GameRecord, to_line  # noqa: E402
from openhearts.search.honest import HonestSearchPlayer  # noqa: E402

LATENCY_BUDGET_S = 3.0
N_OUTER = 50
N_INNER = 20
BOT_CONFIG = f"HonestSearchPlayer(Level.FULL, n_outer={N_OUTER}, n_inner={N_INNER})"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_LIVE = os.path.join(REPO_ROOT, "results", "live")

UNDO_WORDS = {"undo"}
QUIT_WORDS = {"quit", "q", "save"}


# ------------------------------------------------------------- notation
def parse_card(text: str) -> int:
    """'QS' / '7h' -> card int. Case-insensitive. Raises ValueError."""
    s = text.strip()
    if len(s) != 2:
        raise ValueError(f"'{text}' is not two characters (rank+suit)")
    r_ch, s_ch = s[0].upper(), s[1].lower()
    if r_ch not in cards._RANKS:
        raise ValueError(f"'{text}': unknown rank '{s[0]}'")
    if s_ch not in cards._SUITS:
        raise ValueError(f"'{text}': unknown suit '{s[1]}'")
    return cards._SUITS.index(s_ch) * 13 + cards._RANKS.index(r_ch)


# ------------------------------------------------------------- session
@dataclass
class Session:
    known_seat: int
    mode: str  # "advise" or "seat"
    seed: int
    known_hand: int
    played_by_seat: list = field(default_factory=lambda: [0, 0, 0, 0])
    all_played: int = 0
    current_trick: list = field(default_factory=list)
    history: list = field(default_factory=list)
    hearts_broken: bool = False
    trick_number: int = 0
    scores: list = field(default_factory=lambda: [0, 0, 0, 0])
    to_play: int = 0

    def to_dict(self):
        d = dict(self.__dict__)
        d["current_trick"] = [list(x) for x in self.current_trick]
        d["history"] = [list(x) for x in self.history]
        return d

    @staticmethod
    def from_dict(d):
        s = Session(
            known_seat=d["known_seat"], mode=d["mode"], seed=d["seed"],
            known_hand=d["known_hand"],
        )
        s.played_by_seat = list(d["played_by_seat"])
        s.all_played = d["all_played"]
        s.current_trick = [tuple(x) for x in d["current_trick"]]
        s.history = [tuple(x) for x in d["history"]]
        s.hearts_broken = d["hearts_broken"]
        s.trick_number = d["trick_number"]
        s.scores = list(d["scores"])
        s.to_play = d["to_play"]
        return s

    def is_over(self) -> bool:
        return self.trick_number >= 13 and not self.current_trick


def save_resume(session: Session, path: str) -> None:
    with open(path, "w") as f:
        json.dump(session.to_dict(), f)


def load_resume(path: str) -> Session:
    with open(path) as f:
        return Session.from_dict(json.load(f))


def apply_play(session: Session, seat: int, card: int) -> None:
    if seat == session.known_seat:
        session.known_hand &= ~cards.bit(card)
    session.played_by_seat[seat] |= cards.bit(card)
    session.all_played |= cards.bit(card)
    if cards.suit(card) == cards.HEARTS:
        session.hearts_broken = True
    session.current_trick.append((seat, card))
    if len(session.current_trick) == 4:
        winner = trick_winner(session.current_trick)
        pts = trick_points(session.current_trick)
        session.scores[winner] += pts
        session.history.extend(session.current_trick)
        session.current_trick = []
        session.trick_number += 1
        session.to_play = winner
    else:
        session.to_play = (seat + 1) % 4


def known_legal_mask(session: Session) -> int:
    return legal_moves(session.known_hand, tuple(session.current_trick),
                        session.hearts_broken, session.trick_number)


def other_legal_mask(session: Session) -> int:
    """The widest legal set an unseen seat's entry can be checked against.

    When LEADING (current_trick empty) the house rules that govern a lead
    (must be 2C on trick 0, can't lead hearts before broken, avoid points on
    trick 0 if possible) are hand-shape-independent enough that applying
    them to the full candidate pool matches the real rule for any hand that
    is not the vanishingly rare corner case (e.g. a hand that is entirely
    hearts). When FOLLOWING, "must follow suit if able" genuinely depends
    on whether this seat's true, unseen hand holds the led suit -- since a
    real void is common and undetectable from here, that restriction is
    deliberately NOT applied: any not-yet-played, not-known-seat card is
    accepted as a legal follow. This is the correct maximal superset, not a
    shortcut -- restricting by suit here would reject real, legal void
    discards, which the "never wrongly reject" design goal forbids.
    """
    candidate = cards.FULL_DECK & ~session.all_played & ~session.known_hand
    if not session.current_trick:
        return legal_moves(candidate, (), session.hearts_broken,
                           session.trick_number)
    return candidate


def build_view(session: Session, seat: int) -> PlayerView:
    hand = session.known_hand
    legal = legal_moves(hand, tuple(session.current_trick),
                        session.hearts_broken, session.trick_number)
    return PlayerView(
        seat=seat, hand=hand, history=tuple(session.history),
        current_trick=tuple(session.current_trick),
        hearts_broken=session.hearts_broken, trick_number=session.trick_number,
        scores=tuple(session.scores), legal_moves=legal,
    )


# ------------------------------------------------------------- I/O helpers
def _read_line(prompt: str) -> str:
    print(prompt, end="", flush=True)
    line = sys.stdin.readline()
    if line == "":
        raise EOFError("stdin closed")
    print(line.rstrip("\n"))
    return line.strip()


def prompt_card(prompt: str, legal_mask: int):
    """Returns a card int, or the string 'UNDO' / 'QUIT'."""
    while True:
        raw = _read_line(prompt)
        low = raw.lower()
        if low in UNDO_WORDS:
            return "UNDO"
        if low in QUIT_WORDS:
            return "QUIT"
        try:
            card = parse_card(raw)
        except ValueError as e:
            print(f"  invalid entry: {e}. Try again (or 'undo'/'quit').")
            continue
        if not (legal_mask & cards.bit(card)):
            print(f"  illegal card '{cards.card_name(card)}' for this spot. "
                 f"Try again (or 'undo'/'quit').")
            continue
        return card


def prompt_hand(seat: int) -> int:
    while True:
        try:
            raw = _read_line(f"Enter the 13 cards for seat {seat} "
                             f"(e.g. 'QS 7H TC ...'): ")
        except EOFError:
            raise
        parts = raw.split()
        try:
            cds = [parse_card(p) for p in parts]
        except ValueError as e:
            print(f"  invalid card notation: {e}. Try again.")
            continue
        if len(cds) != 13 or len(set(cds)) != 13:
            print(f"  need exactly 13 distinct cards, got {len(cds)}. Try again.")
            continue
        mask = 0
        for c in cds:
            mask |= cards.bit(c)
        return mask


def make_bot(seed: int, n_outer: int = N_OUTER, n_inner: int = N_INNER):
    return HonestSearchPlayer(Level.FULL, n_outer=n_outer, n_inner=n_inner,
                              rng=np.random.default_rng(seed))


def choose_with_latency(bot, view, latency_budget: float):
    t0 = time.perf_counter()
    card = bot.choose(view)
    dt = time.perf_counter() - t0
    print(f"  [decision latency: {dt:.3f}s]")
    if dt > latency_budget:
        print(f"  WARNING: latency {dt:.3f}s exceeds the "
             f"{latency_budget:.1f}s/decision budget.")
    return card, dt


# ------------------------------------------------------------- main loop
def run_hand(session: Session, bot, mode: str, latency_budget: float,
            resume_path: str):
    stack = []
    while not session.is_over():
        seat = session.to_play
        is_known = (seat == session.known_seat)
        if is_known and mode == "seat":
            view = build_view(session, seat)
            card, _ = choose_with_latency(bot, view, latency_budget)
            print(f"  [bot] seat {seat} plays {cards.card_name(card)}")
            result = card
        elif is_known and mode == "advise":
            view = build_view(session, seat)
            suggestion, _ = choose_with_latency(bot, view, latency_budget)
            print(f"  [advisor] suggests {cards.card_name(suggestion)} "
                 f"for seat {seat}")
            result = prompt_card(
                f"Enter card ACTUALLY played by seat {seat}: ",
                known_legal_mask(session),
            )
        else:
            result = prompt_card(f"Enter card played by seat {seat}: ",
                                 other_legal_mask(session))

        if result == "UNDO":
            if not stack:
                print("  nothing to undo.")
                continue
            session_state = stack.pop()
            session.__dict__.update(session_state.__dict__)
            save_resume(session, resume_path)
            continue
        if result == "QUIT":
            save_resume(session, resume_path)
            print("Saved. Resume later with --resume "
                 f"{resume_path}")
            return False

        stack.append(copy.deepcopy(session))
        apply_play(session, seat, result)
        save_resume(session, resume_path)
    return True


def finalize_record(session: Session) -> GameRecord:
    return GameRecord(
        seed=session.seed,
        hands=tuple(session.played_by_seat),
        plays=tuple(session.history),
        scores=tuple(session.scores),
    )


def append_line(path: str, line: str) -> None:
    with open(path, "a") as f:
        f.write(line + "\n")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Live-play Hearts relay (Task F1).")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--advise", action="store_true",
                       help="bot advises the known seat; owner enters actual play")
    group.add_argument("--seat", action="store_true",
                       help="bot plays the known seat itself")
    p.add_argument("--seat-num", type=int, default=0, choices=[0, 1, 2, 3],
                   help="the known seat: owner's seat (--advise) or bot's "
                        "seat (--seat)")
    p.add_argument("--seed", type=int, default=None,
                   help="rng seed for the bot and the record's seed field "
                        "(default: current time)")
    p.add_argument("--aliases", nargs=4, default=["P0", "P1", "P2", "P3"],
                   metavar=("A0", "A1", "A2", "A3"),
                   help="per-seat aliases for the sidecar (never real names)")
    p.add_argument("--notes", default="", help="free-text sidecar note")
    p.add_argument("--results-dir", default=RESULTS_LIVE)
    p.add_argument("--resume", default=None,
                   help="path to an existing autosave file to resume; "
                        "also used as the autosave path for THIS session "
                        "if given")
    p.add_argument("--n-outer", type=int, default=N_OUTER)
    p.add_argument("--n-inner", type=int, default=N_INNER)
    p.add_argument("--latency-budget", type=float, default=LATENCY_BUDGET_S)
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    mode = "seat" if args.seat else "advise"

    os.makedirs(args.results_dir, exist_ok=True)
    records_path = os.path.join(args.results_dir, "relay_records.txt")
    sidecar_path = os.path.join(args.results_dir, "relay_sidecar.txt")
    default_resume_path = os.path.join(args.results_dir, "relay_resume.json")
    resume_path = args.resume or default_resume_path

    seed = args.seed if args.seed is not None else int(time.time())
    bot = make_bot(seed, args.n_outer, args.n_inner)

    try:
        if args.resume and os.path.exists(args.resume):
            session = load_resume(args.resume)
            print(f"Resumed hand from {args.resume} "
                 f"(trick {session.trick_number}, to_play seat {session.to_play}).")
        else:
            known_hand = prompt_hand(args.seat_num)
            session = Session(known_seat=args.seat_num, mode=mode, seed=seed,
                              known_hand=known_hand)
            if known_hand & cards.bit(cards.TWO_CLUBS):
                session.to_play = args.seat_num
            else:
                while True:
                    raw = _read_line("Which seat leads with the 2 of clubs? "
                                     "(0-3): ")
                    try:
                        lead = int(raw)
                    except ValueError:
                        print("  invalid seat number, try again.")
                        continue
                    if lead in (0, 1, 2, 3) and lead != args.seat_num:
                        session.to_play = lead
                        break
                    print("  invalid seat, try again.")
            save_resume(session, resume_path)
    except EOFError:
        print("\nInput ended before a hand started; nothing logged.")
        return 0

    try:
        completed = run_hand(session, bot, mode, args.latency_budget, resume_path)
    except EOFError:
        save_resume(session, resume_path)
        print(f"\nInput ended early; hand saved to {resume_path}.")
        return 0

    if not completed:
        return 0

    record = finalize_record(session)
    append_line(records_path, to_line(record))
    ts = datetime.now(timezone.utc).isoformat()
    bot_cfg = (f"Level.FULL,n_outer={args.n_outer},n_inner={args.n_inner}"
              f",seed={seed}")
    sidecar_line = "|".join([
        ts, ",".join(args.aliases), mode, bot_cfg,
        args.notes.replace("|", "/"),
    ])
    append_line(sidecar_path, sidecar_line)

    print(f"\nHand complete. Scores: {session.scores}")
    print(f"Record appended to {records_path}")
    print(f"Sidecar appended to {sidecar_path}")
    if os.path.exists(resume_path):
        os.remove(resume_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
