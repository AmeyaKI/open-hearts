"""Plain-language reasoning sentences for The Table.

House rule: every sentence cites only numbers computed here from the
inputs (the player's view, the belief table, the search's averages).
No language model, no invented facts. Lower points are better; this
Hearts has no passing and no shooting the moon.

Public entry point: sentences(view, table, candidates, avgs, names).
"""
from openhearts.engine import cards

_SUIT_SYMBOL = "♣♦♠♥"
_SUIT_PLURAL = ["clubs", "diamonds", "spades", "hearts"]
_SUIT_SINGULAR = ["club", "diamond", "spade", "heart"]
_COIN_FLIP_GAP = 0.05
MAX_SENTENCES = 4


def pretty(card: int) -> str:
    """'Qs' -> 'Q♠', 'Th' -> '10♥'."""
    r = cards.card_name(card)[0]
    return ("10" if r == "T" else r) + _SUIT_SYMBOL[cards.suit(card)]


def _join(words: list[str], conj: str = "and") -> str:
    if len(words) <= 1:
        return "".join(words)
    if len(words) == 2:
        return f"{words[0]} {conj} {words[1]}"
    return ", ".join(words[:-1]) + f", {conj} {words[-1]}"


def void_reveals(view) -> list[dict]:
    """For each seat, map suit -> 1-based trick number on which that seat
    first failed to follow that suit. Scans completed tricks then the
    trick in progress, in groups of 4."""
    reveals = [dict() for _ in range(4)]
    plays = list(view.history) + list(view.current_trick)
    for i in range(0, len(plays), 4):
        trick = plays[i:i + 4]
        led = cards.suit(trick[0][1])
        for s, c in trick[1:]:
            if cards.suit(c) != led and led not in reveals[s]:
                reveals[s][led] = i // 4 + 1
    return reveals


def _void_sentences(view, table, names) -> list[tuple[int, str]]:
    """(latest reveal trick, sentence) per opponent with a known void."""
    reveals = void_reveals(view)
    out = []
    for i, seat in enumerate(table.opponent_seats):
        suits = sorted(table.voids[i])
        if not suits:
            continue
        tricks = {s: reveals[seat].get(s) for s in suits}
        if any(t is None for t in tricks.values()):
            # table.voids says void but no reveal found: state the void
            # without a trick number rather than invent one.
            what = "void in " + _join([_SUIT_PLURAL[s] for s in suits])
            latest = -1
        elif len(set(tricks.values())) == 1:
            t = tricks[suits[0]]
            what = ("void in " + _join([_SUIT_PLURAL[s] for s in suits])
                    + f" on trick {t}")
            latest = t
        else:
            parts = [f"{_SUIT_PLURAL[s]} on trick {tricks[s]}"
                     for s in sorted(suits, key=lambda s: tricks[s])]
            what = "void in " + _join(parts)
            latest = max(tricks.values())
        sentence = f"{names[seat]} showed {what}"

        # Where the unseen cards of each void suit must be.
        clauses = []
        for s in suits:
            n = bin(table.unseen_mask & cards.SUIT_MASK[s]).count("1")
            if n == 0:
                continue
            holders = [names[o] for j, o in enumerate(table.opponent_seats)
                       if s not in table.voids[j]]
            if not holders:
                continue
            noun = _SUIT_SINGULAR[s] if n == 1 else _SUIT_PLURAL[s]
            verb = "is" if n == 1 else "are"
            if len(holders) == 1:
                where = f"{verb} all with {holders[0]}" if n > 1 \
                    else f"is with {holders[0]}"
            else:
                where = f"{verb} with {_join(holders, 'or')}"
            clauses.append(f"the {n} {noun} still out {where}")
        if clauses:
            sentence += ", so " + _join(clauses)
        out.append((latest, sentence + "."))
    # most recent reveal first; ties keep left/across/right order
    out.sort(key=lambda p: -p[0])
    return out


def _queen_sentence(view, table, names):
    q = cards.QUEEN_SPADES
    if view.hand & cards.bit(q):
        return "You hold Q♠."
    if not table.unseen_mask & cards.bit(q):
        return None  # already played
    col = table.probs[:, q]
    live = [(i, float(col[i])) for i in range(3) if col[i] > 0]
    if len(live) == 1:
        return f"{names[table.opponent_seats[live[0][0]]]} must hold Q♠."
    parts = [f"{round(100 * p)}% with {names[table.opponent_seats[i]]}"
             for i, p in live]
    return "Q♠ is " + ", ".join(parts) + "."


def _hearts_sentence(view, table):
    plays = list(view.history) + list(view.current_trick)
    heart_played = any(cards.suit(c) == cards.HEARTS for _, c in plays)
    if not (heart_played or view.hearts_broken):
        return None
    out = cards.cards_in(table.unseen_mask & cards.SUIT_MASK[cards.HEARTS])
    if not out:
        return None
    top = pretty(max(out))
    if len(out) == 1:
        return f"1 heart is still out; it is {top}."
    return f"{len(out)} hearts are still out; the highest is {top}."


def _comparison_sentence(view, candidates, avgs):
    if candidates is None or avgs is None or len(candidates) == 0:
        return None
    if len(candidates) == 1:
        if bin(view.legal_moves).count("1") == 1:
            return f"{pretty(candidates[0])} is the only legal play."
        # one group of equivalent cards: not literally the only legal card
        return (f"Every legal play is equivalent to "
                f"{pretty(candidates[0])}.")
    ranked = sorted(zip(candidates, avgs), key=lambda p: (float(p[1]), p[0]))
    (c1, a1), (c2, a2) = ranked[0], ranked[1]
    a1, a2 = float(a1), float(a2)
    gap = a2 - a1
    if gap < _COIN_FLIP_GAP:
        return (f"{pretty(c1)} and {pretty(c2)} are a coin flip "
                f"({a1:.1f} vs {a2:.1f} points).")
    return (f"{pretty(c1)} averages {a1:.1f} points over the imagined "
            f"finishes, {gap:.1f} fewer than {pretty(c2)}.")


def sentences(view, table, candidates, avgs, names) -> list[str]:
    """Up to four plain sentences: voids, Q♠, hearts, best-vs-second."""
    queen = _queen_sentence(view, table, names)
    hearts = _hearts_sentence(view, table)
    compare = _comparison_sentence(view, candidates, avgs)
    others = [s for s in (queen, hearts, compare) if s]
    room = max(0, min(2, MAX_SENTENCES - len(others)))
    voids = [s for _, s in _void_sentences(view, table, names)][:room]
    return voids + others
