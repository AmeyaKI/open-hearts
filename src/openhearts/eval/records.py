"""Flat-file game records: one line per game, replayable offline."""
from dataclasses import dataclass


@dataclass(frozen=True)
class GameRecord:
    seed: int
    hands: tuple   # 4 bitmasks at deal time
    plays: tuple   # 52 (seat, card) in order
    scores: tuple  # 4 final scores


def record_from(seed, initial_hands, final_state) -> GameRecord:
    return GameRecord(
        seed=seed,
        hands=tuple(initial_hands),
        plays=tuple(final_state.history),
        scores=tuple(final_state.scores),
    )


def to_line(rec: GameRecord) -> str:
    hands = ",".join(str(h) for h in rec.hands)
    plays = " ".join(f"{s}:{c}" for s, c in rec.plays)
    scores = ",".join(str(x) for x in rec.scores)
    return f"v1|{rec.seed}|{hands}|{plays}|{scores}"


def from_line(line: str) -> GameRecord:
    ver, seed, hands, plays, scores = line.strip().split("|")
    assert ver == "v1", f"unknown record version {ver}"
    return GameRecord(
        seed=int(seed),
        hands=tuple(int(h) for h in hands.split(",")),
        plays=tuple(
            (int(p.split(":")[0]), int(p.split(":")[1]))
            for p in plays.split(" ")
        ),
        scores=tuple(int(x) for x in scores.split(",")),
    )


def write_records(path, recs) -> None:
    with open(path, "w") as f:
        for r in recs:
            f.write(to_line(r) + "\n")


def read_records(path):
    with open(path) as f:
        return [from_line(ln) for ln in f if ln.strip()]
