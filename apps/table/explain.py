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
MAX_SENTENCES = 6
_MAX_VOID_LINES = 2


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


def _plays(view):
    return list(view.history) + list(view.current_trick)


def _point_word(hand: int) -> str | None:
    """What point card the observer could throw: 'a heart', 'Q♠',
    'a point card' (both), or None (holds no points)."""
    has_heart = bool(hand & cards.HEARTS_MASK)
    has_queen = bool(hand & cards.bit(cards.QUEEN_SPADES))
    if has_heart and has_queen:
        return "a point card"
    if has_heart:
        return "a heart"
    if has_queen:
        return "Q♠"
    return None


def _your_voids_sentence(view):
    """A. Suits (not hearts) the observer holds none of."""
    if not view.hand:
        return None
    void = [s for s in range(3) if not view.hand & cards.SUIT_MASK[s]]
    if not void:
        return None
    lead = "You have no " + _join([_SUIT_PLURAL[s] for s in void], "or") \
        + " left, so "
    throw = _point_word(view.hand)
    # On the first trick points may not be thrown, so only claim the
    # throw from trick 2 on.
    if throw is None or view.trick_number == 0:
        return lead + "you cannot be forced to follow there."
    return (lead + f"you can throw {throw} on any "
            + _join([_SUIT_SINGULAR[s] for s in void], "or") + " trick.")


def _points_phrase(n: int, where: str) -> str:
    return (f"There is 1 point {where}." if n == 1
            else f"There are {n} points {where}.")


def _follow_sentence(view):
    """B. Following suit: can the observer duck under the current winner?"""
    trick = list(view.current_trick)
    if not trick or any(s == view.seat for s, _ in trick):
        return None
    led = cards.suit(trick[0][1])
    mine = cards.cards_in(view.hand & cards.SUIT_MASK[led])
    if not mine:
        return None
    top = max(c for _, c in trick if cards.suit(c) == led)
    ducks = [c for c in mine if c < top]
    out = f"{_SUIT_PLURAL[led].capitalize()} were led; the {pretty(top)} " \
        "is winning so far and you "
    if ducks:
        out += f"can duck with {pretty(ducks[0])}"
        if len(ducks) > 1:
            out += " (or " + ", ".join(pretty(c) for c in ducks[1:]) + ")"
        out += "."
    else:
        out += (f"cannot duck — every {_SUIT_SINGULAR[led]} you hold "
                f"beats the {pretty(top)}.")
    pts = sum(1 if cards.suit(c) == cards.HEARTS else
              13 if c == cards.QUEEN_SPADES else 0 for _, c in trick)
    if pts:
        out += " " + _points_phrase(pts, "on this trick")
    return out


def _lead_sentences(view, table, names) -> list[str]:
    """C. Leading: which suits an opponent is known void in (risky when
    that opponent may still hold points) and which nobody is void in.
    Only said once some opponent has shown a void."""
    if view.current_trick or view.trick_number < 1 or not any(table.voids):
        return []
    suits = [s for s in range(4)
             if view.legal_moves & cards.SUIT_MASK[s]
             and (s != cards.HEARTS or view.hearts_broken)]
    q_unseen = bool(table.unseen_mask & cards.bit(cards.QUEEN_SPADES))
    hearts_unseen = bool(table.unseen_mask & cards.HEARTS_MASK)
    risky, safe = [], []
    for s in suits:
        void_opps = [i for i in range(3) if s in table.voids[i]]
        if not void_opps:
            safe.append(s)
            continue
        dumpers, kinds = [], set()
        for i in void_opps:
            can_heart = hearts_unseen and cards.HEARTS not in table.voids[i]
            can_queen = q_unseen and float(
                table.probs[i, cards.QUEEN_SPADES]) > 0
            if can_heart or can_queen:
                dumpers.append(names[table.opponent_seats[i]])
                kinds.add("heart" if can_heart else "queen")
        if not dumpers:
            continue
        what = ("a heart" if kinds == {"heart"} else
                "Q♠" if kinds == {"queen"} else "a point card")
        verb = "is" if len(dumpers) == 1 else "are"
        risky.append(f"Leading {_SUIT_PLURAL[s]} is risky: "
                     f"{_join(dumpers)} {verb} void and can dump {what}.")
    out = []
    if risky:
        out.append(" ".join(risky))
    if safe:
        words = [_SUIT_PLURAL[s] for s in safe]
        words[0] = words[0].capitalize()
        out.append(f"{_join(words)} are"
                   " safe to lead: nobody is known to be void.")
    return out


def _queen_danger_sentence(view, table):
    """D. Danger around Q♠ for the observer."""
    q = cards.QUEEN_SPADES
    if table.unseen_mask & cards.bit(q):
        high = [c for c in (q + 2, q + 1) if view.hand & cards.bit(c)]
        if not high:
            return None
        return (f"You hold {_join([pretty(c) for c in high])} above the "
                "queen: a spade lead by an opponent could drop Q♠ on you.")
    if not view.hand & cards.bit(q):
        return None
    spades_led = any(cards.suit(c) == cards.SPADES
                     for c in [p[1] for p in _plays(view)[::4]])
    void_somewhere = any(not view.hand & cards.SUIT_MASK[s] for s in range(4))
    if spades_led and (void_somewhere or view.hearts_broken):
        return ("You still hold Q♠; it is safest to dump it on a trick "
                "someone else is winning.")
    return None


def _hearts_broken_sentence(view, names):
    """E. Trick and player that played the first heart."""
    for i, (seat, c) in enumerate(_plays(view)):
        if cards.suit(c) == cards.HEARTS:
            who = "you" if seat == view.seat else names[seat]
            return f"Hearts were broken on trick {i // 4 + 1} by {who}."
    return None


def _points_left_sentence(view):
    """F. Points not yet played (the observer's own hand counts as out)."""
    played = 0
    for _, c in _plays(view):
        played |= cards.bit(c)
    left = cards.POINTS_MASK & ~played
    n = bin(left & cards.HEARTS_MASK).count("1") \
        + (13 if left & cards.bit(cards.QUEEN_SPADES) else 0)
    if n == 0:
        return None
    if n == 1:
        return "1 point is still out of 26."
    return f"{n} points are still out of 26."


def sentences(view, table, candidates, avgs, names) -> list[str]:
    """Up to six plain sentences. When trimming, keep in priority order:
    comparison, Q♠, opponent voids (at most two), your voids, following,
    leading, Q♠ danger, hearts broken, points left, hearts still out.
    Output order: opponent voids, Q♠, the new templates, hearts still
    out, and the comparison last."""
    compare = _comparison_sentence(view, candidates, avgs)
    queen = _queen_sentence(view, table, names)
    voids = [s for _, s in _void_sentences(view, table, names)]
    voids = voids[:_MAX_VOID_LINES]
    extras = [_your_voids_sentence(view), _follow_sentence(view)]
    extras += _lead_sentences(view, table, names)
    extras += [_queen_danger_sentence(view, table),
               _hearts_broken_sentence(view, names),
               _points_left_sentence(view)]
    hearts = _hearts_sentence(view, table)

    # (display slot, sentence) in priority order
    ranked = [(9, compare), (1, queen)]
    ranked += [(0, s) for s in voids]
    ranked += [(2, s) for s in extras]
    ranked += [(8, hearts)]
    kept = [(slot, k, s) for k, (slot, s) in enumerate(ranked) if s]
    kept = kept[:MAX_SENTENCES]
    kept.sort(key=lambda t: (t[0], t[1]))
    return [s for _, _, s in kept]
