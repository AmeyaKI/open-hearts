"""Phase 7 Task F1 gates for `experiments/relay.py`.

All gates drive the CLI as a subprocess with piped stdin -- the literal
"piped-input end-to-end" path the gate wording asks for -- against a fixed
seed and a tiny bot config (n_outer/n_inner=1) so the whole suite runs fast
and deterministically. A helper builds a real, engine-legal deal + full play
sequence with `HeuristicPlayer` so the transcript fed to the relay is always
a legal game the relay's own legality checks will accept.
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openhearts.engine import cards  # noqa: E402
from openhearts.engine.game import deal, play_game  # noqa: E402
from openhearts.eval.records import record_from, to_line  # noqa: E402
from openhearts.players.heuristic import HeuristicPlayer  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELAY = os.path.join(REPO_ROOT, "experiments", "relay.py")


def _hand_str(mask: int) -> str:
    return " ".join(cards.card_name(c) for c in cards.cards_in(mask))


def _build_reference_game(seed: int):
    """A real, legal, deterministic heuristic-vs-heuristic game."""
    state = deal(np.random.default_rng(seed))
    initial_hands = tuple(state.hands)
    final = play_game(state, [HeuristicPlayer() for _ in range(4)])
    rec = record_from(seed=seed, initial_hands=initial_hands, final_state=final)
    return initial_hands, rec


def _run_relay(args, stdin_text, results_dir):
    return subprocess.run(
        [sys.executable, RELAY, "--results-dir", results_dir] + args,
        input=stdin_text, capture_output=True, text=True, timeout=120,
    )


def _header_line_count(initial_hands, known_seat):
    """1 line for the known seat's hand, plus 1 more only if that seat does
    NOT hold the 2 of clubs (the "who leads?" prompt)."""
    return 1 + (0 if (initial_hands[known_seat] & cards.bit(cards.TWO_CLUBS)) else 1)


def _transcript_lines(initial_hands, plays, known_seat, mode):
    """Build the exact stdin script a human relay session would type,
    given a known reference game. `plays` is the flat (seat, card) order."""
    lines = []
    known_hand = initial_hands[known_seat]
    lines.append(_hand_str(known_hand))
    leader = plays[0][0]
    if not (known_hand & cards.bit(cards.TWO_CLUBS)):
        lines.append(str(leader))
    for seat, card in plays:
        if seat == known_seat:
            if mode == "advise":
                lines.append(cards.card_name(card))
            # in --seat mode the bot plays on its own: nothing to type
        else:
            lines.append(cards.card_name(card))
    return lines


# --------------------------------------------------------------- gate 1
def test_gate1_piped_end_to_end_reproduces_known_record(tmp_path):
    seed = 424242
    initial_hands, rec = _build_reference_game(seed)
    known_seat = rec.plays[0][0]  # seat holding 2C leads trick 0
    lines = _transcript_lines(initial_hands, rec.plays, known_seat, "advise")
    stdin_text = "\n".join(lines) + "\n"

    results_dir = str(tmp_path / "live")
    proc = _run_relay(
        ["--advise", "--seat-num", str(known_seat), "--seed", str(seed),
         "--n-outer", "1", "--n-inner", "1"],
        stdin_text, results_dir,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    records_path = os.path.join(results_dir, "relay_records.txt")
    with open(records_path) as f:
        got_line = f.read().strip()
    assert got_line == to_line(rec)

    # Running the exact same session again must reproduce the exact same
    # line again (append-only, deterministic) -- the reproducibility half
    # of the gate.
    results_dir2 = str(tmp_path / "live2")
    proc2 = _run_relay(
        ["--advise", "--seat-num", str(known_seat), "--seed", str(seed),
         "--n-outer", "1", "--n-inner", "1"],
        stdin_text, results_dir2,
    )
    assert proc2.returncode == 0, proc2.stdout + proc2.stderr
    with open(os.path.join(results_dir2, "relay_records.txt")) as f:
        got_line2 = f.read().strip()
    assert got_line2 == got_line


def test_gate1_non_leading_known_seat(tmp_path):
    """Exercise the '--seat-num' that does NOT hold 2C -- the "which seat
    leads?" prompt path, otherwise never hit by any other test."""
    seed = 271828
    initial_hands, rec = _build_reference_game(seed)
    leader = rec.plays[0][0]
    known_seat = (leader + 1) % 4  # guaranteed not to hold 2C
    lines = _transcript_lines(initial_hands, rec.plays, known_seat, "advise")
    stdin_text = "\n".join(lines) + "\n"

    results_dir = str(tmp_path / "live")
    proc = _run_relay(
        ["--advise", "--seat-num", str(known_seat), "--seed", str(seed),
         "--n-outer", "1", "--n-inner", "1"],
        stdin_text, results_dir,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "leads with the 2 of clubs" in proc.stdout
    with open(os.path.join(results_dir, "relay_records.txt")) as f:
        got_line = f.read().strip()
    assert got_line == to_line(rec)


# --------------------------------------------------------------- gate 2
def test_gate2_illegal_entry_rejected_then_retried(tmp_path):
    seed = 5150
    initial_hands, rec = _build_reference_game(seed)
    known_seat = rec.plays[0][0]
    lines = _transcript_lines(initial_hands, rec.plays, known_seat, "advise")

    # Corrupt the FIRST opponent entry (the trick-0 leader) into an illegal
    # card (something that can never legally lead trick 0), then still send
    # the real, legal entry right after -- reject, then retry, then proceed.
    header = _header_line_count(initial_hands, known_seat)
    pos = header
    for seat, card in rec.plays:
        if seat == known_seat:
            pos += 1
            continue
        break
    illegal_line = lines[pos]
    corrupted = list(lines)
    corrupted[pos] = "3D"  # 3 of diamonds can never legally lead trick 0
    corrupted.insert(pos + 1, illegal_line)

    stdin_text = "\n".join(corrupted) + "\n"
    results_dir = str(tmp_path / "live")
    proc = _run_relay(
        ["--advise", "--seat-num", str(known_seat), "--seed", str(seed),
         "--n-outer", "1", "--n-inner", "1"],
        stdin_text, results_dir,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "illegal card" in proc.stdout
    records_path = os.path.join(results_dir, "relay_records.txt")
    with open(records_path) as f:
        got_line = f.read().strip()
    assert got_line == to_line(rec)


# --------------------------------------------------------------- gate 3
def test_gate3_undo_reverts_last_play(tmp_path):
    seed = 909090
    initial_hands, rec = _build_reference_game(seed)
    known_seat = rec.plays[0][0]
    lines = _transcript_lines(initial_hands, rec.plays, known_seat, "advise")

    # After the very first entered play, type "undo", then re-enter it.
    header = _header_line_count(initial_hands, known_seat)
    first_entry = lines[header]
    with_undo = lines[:header + 1] + ["undo"] + lines[header:]
    stdin_text = "\n".join(with_undo) + "\n"

    results_dir = str(tmp_path / "live")
    proc = _run_relay(
        ["--advise", "--seat-num", str(known_seat), "--seed", str(seed),
         "--n-outer", "1", "--n-inner", "1"],
        stdin_text, results_dir,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    records_path = os.path.join(results_dir, "relay_records.txt")
    with open(records_path) as f:
        got_line = f.read().strip()
    assert got_line == to_line(rec)


# --------------------------------------------------------------- gate 4
def test_gate4_advise_mode_records_actual_divergent_play(tmp_path):
    """--advise mode: the owner's entered card need not match the bot's
    suggestion -- confirm the DIVERGENT entry (not the suggestion) is what
    gets recorded."""
    seed = 123123
    initial_hands, rec = _build_reference_game(seed)
    known_seat = rec.plays[0][0]
    lines = _transcript_lines(initial_hands, rec.plays, known_seat, "advise")
    stdin_text = "\n".join(lines) + "\n"

    results_dir = str(tmp_path / "live")
    proc = _run_relay(
        ["--advise", "--seat-num", str(known_seat), "--seed", str(seed),
         "--n-outer", "1", "--n-inner", "1"],
        stdin_text, results_dir,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[advisor] suggests" in proc.stdout
    with open(os.path.join(results_dir, "relay_records.txt")) as f:
        got_line = f.read().strip()
    assert got_line == to_line(rec)


def test_gate4_seat_mode_bot_plays_itself(tmp_path):
    """--seat mode: the bot decides its own seat's cards, so a static
    transcript from a DIFFERENT (heuristic) game cannot be replayed to
    completion -- the bot's own choices will diverge. Instead: let the
    transcript run for a bounded prefix, then let stdin end, and check
    that the bot announced its own move(s), a latency was printed, and
    mid-hand state was autosaved (the resume file exists with plays in it)."""
    seed = 777001
    initial_hands, rec = _build_reference_game(seed)
    known_seat = rec.plays[0][0]
    lines = _transcript_lines(initial_hands, rec.plays, known_seat, "seat")
    # Feed only the header + first few opponent entries (not the whole
    # hand) since the bot's own plays will diverge from `rec` immediately.
    header = _header_line_count(initial_hands, known_seat)
    prefix = lines[:header + 3]
    stdin_text = "\n".join(prefix) + "\n"

    results_dir = str(tmp_path / "live")
    proc = _run_relay(
        ["--seat", "--seat-num", str(known_seat), "--seed", str(seed),
         "--n-outer", "1", "--n-inner", "1"],
        stdin_text, results_dir,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[bot] seat" in proc.stdout
    assert "decision latency" in proc.stdout
    resume_path = os.path.join(results_dir, "relay_resume.json")
    assert os.path.exists(resume_path)
    import json
    with open(resume_path) as f:
        saved = json.load(f)
    assert len(saved["history"]) + len(saved["current_trick"]) >= 1


# --------------------------------------------------------------- gate 5
def test_gate5_latency_measured_and_printed_per_decision(tmp_path):
    seed = 55501
    initial_hands, rec = _build_reference_game(seed)
    known_seat = rec.plays[0][0]
    lines = _transcript_lines(initial_hands, rec.plays, known_seat, "advise")
    stdin_text = "\n".join(lines) + "\n"

    results_dir = str(tmp_path / "live")
    proc = _run_relay(
        ["--advise", "--seat-num", str(known_seat), "--seed", str(seed),
         "--n-outer", "1", "--n-inner", "1"],
        stdin_text, results_dir,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    latency_lines = [ln for ln in proc.stdout.splitlines()
                     if "decision latency" in ln]
    # known_seat has exactly 13 decisions (one per trick).
    assert len(latency_lines) == 13
    for ln in latency_lines:
        assert "s]" in ln


# --------------------------------------------------------- extra: resume
def test_mid_hand_resume(tmp_path):
    seed = 314159
    initial_hands, rec = _build_reference_game(seed)
    known_seat = rec.plays[0][0]
    lines = _transcript_lines(initial_hands, rec.plays, known_seat, "advise")
    header = _header_line_count(initial_hands, known_seat)
    split = header + 6  # stop partway through the hand

    results_dir = str(tmp_path / "live")
    first_stdin = "\n".join(lines[:split]) + "\n"  # stdin closes early (EOF)
    proc1 = _run_relay(
        ["--advise", "--seat-num", str(known_seat), "--seed", str(seed),
         "--n-outer", "1", "--n-inner", "1"],
        first_stdin, results_dir,
    )
    assert proc1.returncode == 0, proc1.stdout + proc1.stderr
    resume_path = os.path.join(results_dir, "relay_resume.json")
    assert os.path.exists(resume_path)
    assert not os.path.exists(os.path.join(results_dir, "relay_records.txt"))

    rest_stdin = "\n".join(lines[split:]) + "\n"
    proc2 = _run_relay(
        ["--advise", "--seat-num", str(known_seat), "--seed", str(seed),
         "--n-outer", "1", "--n-inner", "1", "--resume", resume_path],
        rest_stdin, results_dir,
    )
    assert proc2.returncode == 0, proc2.stdout + proc2.stderr
    with open(os.path.join(results_dir, "relay_records.txt")) as f:
        got_line = f.read().strip()
    assert got_line == to_line(rec)
    assert not os.path.exists(resume_path)


# ------------------------------------------------------------- sidecar
def test_sidecar_never_contains_full_names_field_and_has_aliases(tmp_path):
    seed = 8675309
    initial_hands, rec = _build_reference_game(seed)
    known_seat = rec.plays[0][0]
    lines = _transcript_lines(initial_hands, rec.plays, known_seat, "advise")
    stdin_text = "\n".join(lines) + "\n"

    results_dir = str(tmp_path / "live")
    proc = _run_relay(
        ["--advise", "--seat-num", str(known_seat), "--seed", str(seed),
         "--n-outer", "1", "--n-inner", "1",
         "--aliases", "Alice", "Bob", "Cara", "Dee", "--notes", "test note"],
        stdin_text, results_dir,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    with open(os.path.join(results_dir, "relay_sidecar.txt")) as f:
        line = f.read().strip()
    parts = line.split("|")
    assert len(parts) == 5
    assert parts[1] == "Alice,Bob,Cara,Dee"
    assert parts[2] == "advise"
    assert "Level.FULL" in parts[3]
    assert parts[4] == "test note"
