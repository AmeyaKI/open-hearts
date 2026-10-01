"""Perilune (JAkoliver/hearts_simulator, MIT) adapter -- ROADMAP B6, Tiers 0 and 1.

Seats Perilune's released RAW networks in OUR engine and reads their belief head, using only a
`PlayerView` (the same information boundary every open-hearts player lives behind).

WHAT IS MIRRORED, AND FROM WHERE (their repo at the pinned commit, see `PIN_COMMIT`):

* Observation: `HeartsEnv::ObserveFor` (550 floats) and `HeartsEnv::ObserveExtFor` (326 floats)
  in HeartsEnv.hpp, re-derived here from a PlayerView. Card ids are IDENTICAL in the two
  projects (suit order clubs, diamonds, spades, hearts; id = 13*suit + rank-2; Q-spades = 36),
  seats advance (s+1)%4 in both, and the legal-move rules are identical (2-clubs leads, no
  penalty card on trick 1 unless forced, hearts led only when broken or forced, a heart -- not
  the queen -- breaks hearts). `tests/test_perilune_adapter.py` proves the encoding bit-for-bit
  against their compiled engine over whole games, every seat, every decision.
* Per-deal convention: `orchestrator._obs_for_net` -- THEIR OWN per-deal raw instrument. A
  550-input net (the v5 champion) gets the plain 550 observation (its match-context term is
  skipped entirely); an 882-input net (the gated ensemble) gets [550 | six ZERO match-context
  dims | 326 extension]. We adopt it unchanged rather than invent a context.
* Raw play: `orchestrator.play_round` -- legality-masked argmax of the policy logits, one
  forward per decision, deterministic.
* Belief: `SearchPlayer::FetchBelief` -- sigmoid of the 3x52 belief logits, rows = the
  observer's RELATIVE opponents left(+1), across(+2), right(+3) -- the same order as our
  `BeliefTable.opponent_seats`. `SearchPlayer::TrySample` (their determinization sampler) is
  ported in `sample_owner_assignment` so their inference can be scored AS DEPLOYED.

VARIANT HONESTY (registry wording applies): their engine has a "hold" pass direction (no
passing), one deal in four of their training rotation, so no-pass is IN-distribution for the
nets; the pass blocks are zero and the direction one-hot says "hold". No-moon is OUR variant:
their nets were trained where taking all 26 points scores 0 for the shooter -- the nets may
defend or attempt moons that do not exist here. That is a disclosed handicap on their side.

Nothing in this module runs a match. Import cost is paid lazily: torch and their `hearts_net`
are imported on the first `load_net` call.
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

from openhearts.engine import cards
from openhearts.engine.state import PlayerView

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PERILUNE_DIR = os.environ.get("PERILUNE_DIR", os.path.join(REPO_ROOT, "external", "perilune"))

PIN_COMMIT = "c395d2a77c139eb9bfa0e548c7a20b85d208c547"

# name -> (file under <PERILUNE_DIR>/weights, sha256, observation width)
NETS = {
    # v5 card-token transformer, 7.59M parameters; the 5th match-era champion, and the DEFAULT
    # component (~95% of decisions) of every later ensemble. md5 8a89da90.
    "v5": ("hearts_model_final.pth",
           "7ce504ce78a728e282437041b7094c3a4eb106b06bb5becf5447e881bf445a5a", 550),
    # v6.1 two-tier gated ensemble, their CURRENT champion (promoted 2026-09-20): v5 default +
    # two moon-defence specialists behind a moon-probability router. md5 710c2102.
    "v6.1": ("hearts_ensemble_710c2102.pth",
             "75d87a2a51f3e47ae5250480fecaf805bfa704d67466840aba2f34c009c7f6a3", 882),
}

OBS_DIM = 550
EXT_DIM = 326
MATCH_CTX_DIM = 6
HOLD = 3                      # pass-direction one-hot index for "hold" (no passing)
BELIEF_FLOOR = 1e-4           # SearchPlayer::TrySample clamps belief weights here

_F26 = np.float32(26.0)
_F13 = np.float32(13.0)
_F4 = np.float32(4.0)


def available() -> bool:
    """True when the pinned checkout and both checkpoints are present."""
    return (os.path.isfile(os.path.join(PERILUNE_DIR, "hearts_simulator", "hearts_net.py"))
            and all(os.path.isfile(os.path.join(PERILUNE_DIR, "weights", f))
                    for f, _, _ in NETS.values()))


# --------------------------------------------------------------------------------------
# observation
# --------------------------------------------------------------------------------------

def _tricks(view: PlayerView):
    """All plays so far as a list of tricks (the last one may be partial)."""
    plays = list(view.history) + list(view.current_trick)
    return [plays[i:i + 4] for i in range(0, len(plays), 4)]


def their_void_tracker(view: PlayerView) -> np.ndarray:
    """Their PUBLIC void tracker, [absolute seat, suit] -> bool.

    Mirrors HeartsEnv::Step: a void is recorded when a FOLLOWER plays off the led suit --
    EXCEPT on the first trick (`!state.isFirstTrick()`), a quirk of their engine: a trick-1
    discard reveals a club void to a human but is not written to their tracker. We reproduce
    the quirk because the nets were trained on it.
    """
    voids = np.zeros((4, 4), dtype=bool)
    for t, trick in enumerate(_tricks(view)):
        if t == 0:
            continue
        led = cards.suit(trick[0][1])
        for seat, card in trick[1:]:
            if cards.suit(card) != led:
                voids[seat, led] = True
    return voids


def encode_obs550(view: PlayerView) -> np.ndarray:
    """HeartsEnv::ObserveFor(view.seat) for a no-pass deal, from public information + own hand."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    me = view.seat

    for c in cards.cards_in(view.hand):                       # block 1: hand
        obs[c] = 1.0
    for _, c in view.current_trick:                           # block 2: current trick
        obs[52 + c] = 1.0
    for _, c in view.history:                                 # block 3: completed tricks
        obs[104 + c] = 1.0
    for i in range(4):                                        # block 4: context
        obs[156 + i] = np.float32(view.scores[i]) / _F26      # ABSOLUTE seats, as theirs
    obs[160 + min(len(view.current_trick), 3)] = 1.0
    obs[164] = 1.0 if view.hearts_broken else 0.0
    obs[165:181] = their_void_tracker(view).reshape(16)       # block 5: [seat*4 + suit]
    obs[181 + HOLD] = 1.0                                     # block 6: pass direction = hold
    # block 7 (in-passing flag), blocks 8/9 (cards passed / received): zero in a hold deal
    for t, trick in enumerate(_tricks(view)):
        when = np.float32(t + 1) / _F13
        for seat, c in trick:
            obs[290 + ((seat - me) % 4) * 52 + c] = 1.0       # block 10: who played what
            obs[498 + c] = when                               # block 11: play timing
    return obs


def encode_ext326(view: PlayerView) -> np.ndarray:
    """HeartsEnv::ObserveExtFor(view.seat): the obs-v2 capture extension (public info only)."""
    ext = np.zeros(EXT_DIM, dtype=np.float32)
    me = view.seat
    tricks_won = [0, 0, 0, 0]
    hearts_seen = 0
    qs_taker = -1
    for trick in _tricks(view):
        for pos, (_, c) in enumerate(trick):
            ext[c] = np.float32(pos + 1) / _F4                # within-trick position
            if pos == 0:
                ext[52 + c] = 1.0                             # led-the-trick flag
            if cards.suit(c) == cards.HEARTS:
                hearts_seen += 1
        if len(trick) == 4:
            led = cards.suit(trick[0][1])
            winner, best = trick[0][0], cards.rank(trick[0][1])
            for seat, c in trick[1:]:
                if cards.suit(c) == led and cards.rank(c) > best:
                    winner, best = seat, cards.rank(c)
            tricks_won[winner] += 1
            for _, c in trick:
                ext[104 + ((winner - me) % 4) * 52 + c] = 1.0  # taken-by planes
                if c == cards.QUEEN_SPADES:
                    qs_taker = winner
    for k in range(4):
        seat = (me + k) % 4
        ext[312 + k] = np.float32(tricks_won[seat]) / _F13
        alive = all(view.scores[t] == 0 for t in range(4) if t != seat)
        ext[316 + k] = 1.0 if alive else 0.0                  # moon-alive (no OTHER seat has points)
    ext[320] = np.float32(13 - hearts_seen) / _F13
    if qs_taker < 0:
        ext[321] = 1.0
    else:
        ext[322 + ((qs_taker - me) % 4)] = 1.0
    return ext


def encode_obs(view: PlayerView, obs_dim: int) -> np.ndarray:
    """Their per-deal observation for a net of the given width (`orchestrator._obs_for_net`)."""
    base = encode_obs550(view)
    if obs_dim == 550:
        return base
    if obs_dim == 882:
        return np.concatenate([base, np.zeros(MATCH_CTX_DIM, dtype=np.float32),
                               encode_ext326(view)])
    raise ValueError(f"unsupported Perilune observation width {obs_dim}")


def legal_mask(view: PlayerView) -> np.ndarray:
    mask = np.zeros(52, dtype=bool)
    for c in cards.cards_in(view.legal_moves):
        mask[c] = True
    return mask


# --------------------------------------------------------------------------------------
# networks
# --------------------------------------------------------------------------------------

_NET_CACHE: dict = {}
_SHA_OK: set = set()


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_path(name: str) -> str:
    return os.path.join(PERILUNE_DIR, "weights", NETS[name][0])


def load_net(name: str):
    """Their network, constructed by THEIR `net_from_checkpoint`, CPU, eval mode.

    The checkpoint's sha256 is verified against the pinned digest on first load in each
    process; a mismatch is a hard error (the registry row names these exact bytes).
    """
    if name in _NET_CACHE:
        return _NET_CACHE[name]
    fname, want, obs_dim = NETS[name]
    path = weights_path(name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} missing -- run scripts/build_perilune.sh (or set PERILUNE_DIR)")
    if path not in _SHA_OK:
        got = sha256_of(path)
        if got != want:
            raise RuntimeError(f"sha256 mismatch for {fname}: got {got}, pinned {want}")
        _SHA_OK.add(path)
    src = os.path.join(PERILUNE_DIR, "hearts_simulator")
    if src not in sys.path:
        sys.path.insert(0, src)
    import torch
    from hearts_net import net_from_checkpoint  # their code, MIT, at PIN_COMMIT

    net = net_from_checkpoint(path, map_location="cpu")
    net.eval()
    assert getattr(net, "obs_dim", 550) == obs_dim, (name, getattr(net, "obs_dim", None))
    for p in net.parameters():
        p.requires_grad_(False)
    _NET_CACHE[name] = (net, obs_dim, torch)
    return _NET_CACHE[name]


def single_thread() -> None:
    """One torch thread per process (pool workers must not oversubscribe the machine)."""
    import torch
    torch.set_num_threads(1)


def policy_logits(name: str, view: PlayerView) -> np.ndarray:
    """Masked policy logits (52,), -inf on illegal cards -- one forward, as their play_round."""
    net, obs_dim, torch = load_net(name)
    obs = torch.from_numpy(encode_obs(view, obs_dim)).unsqueeze(0)
    mask = torch.from_numpy(legal_mask(view)).unsqueeze(0)
    with torch.no_grad():
        logits, _ = net(obs, mask)
    return logits[0].numpy()


class PeriluneRawPlayer:
    """Perilune's raw network as an open-hearts `Player`: legality-masked argmax, no search.

    Deterministic given the view. `decisions` / `seconds` accumulate for cost reporting.
    """

    def __init__(self, net: str = "v5"):
        if net not in NETS:
            raise ValueError(f"unknown Perilune net {net!r}; choose from {sorted(NETS)}")
        self.net = net
        self.decisions = 0
        self.seconds = 0.0
        load_net(net)

    def choose(self, view: PlayerView) -> int:
        import time
        t0 = time.perf_counter()
        card = int(np.argmax(policy_logits(self.net, view)))
        self.seconds += time.perf_counter() - t0
        self.decisions += 1
        assert view.legal_moves & cards.bit(card), f"Perilune chose illegal card {card}"
        return card


# --------------------------------------------------------------------------------------
# belief head (Tier 0)
# --------------------------------------------------------------------------------------

def belief_sigmoid(name: str, view: PlayerView) -> np.ndarray:
    """(3, 52) float64: sigmoid of the belief logits, rows = left(+1), across(+2), right(+3).

    Their search fetches belief with an all-ones mask (the mask touches policy logits only);
    we do the same. For the ensemble, belief comes from its default (v5) component, exactly
    as `HeartsHybrid.forward_all` defines it.
    """
    net, obs_dim, torch = load_net(name)
    obs = torch.from_numpy(encode_obs(view, obs_dim)).unsqueeze(0)
    mask = torch.ones(1, 52, dtype=torch.bool)
    with torch.no_grad():
        _, _, bel = net.forward_all(obs, mask)
    logits = bel[0].double().numpy().reshape(3, 52)
    return 1.0 / (1.0 + np.exp(-logits))


def unseen_mask(view: PlayerView) -> int:
    seen = view.hand
    for _, c in list(view.history) + list(view.current_trick):
        seen |= cards.bit(c)
    return cards.FULL_DECK & ~seen


def opponent_hand_sizes(view: PlayerView) -> list:
    played = [0, 0, 0, 0]
    for s, _ in list(view.history) + list(view.current_trick):
        played[s] += 1
    return [13 - played[(view.seat + 1 + k) % 4] for k in range(3)]


def belief_readings(name: str, view: PlayerView) -> dict:
    """Three analytic readings of the belief head, each (3, 52) over the observer's unseen cards.

    raw        sigmoid as trained (three independent per-card probabilities; columns need not
               sum to 1). The training target; the reading most favourable to what was learned.
    normalized raw, renormalised over the three opponents for each unseen card (a proper
               "who holds it" distribution -- the same object our exact table produces).
    deployed   the per-card weights their sampler actually uses: max(raw, 1e-4), zero where
               THEIR public void tracker rules the opponent out, renormalised. (Their sampler
               additionally enforces hand sizes sequentially; `sample_owner_assignment` covers
               that reading by Monte Carlo.)
    Seen cards are zero in every reading.
    """
    raw = belief_sigmoid(name, view)
    cols = cards.cards_in(unseen_mask(view))
    keep = np.zeros(52, dtype=bool)
    keep[cols] = True

    out_raw = np.where(keep[None, :], raw, 0.0)

    norm = out_raw.copy()
    s = norm.sum(axis=0)
    norm[:, keep] /= s[keep]

    voids = their_void_tracker(view)
    dep = np.maximum(raw, BELIEF_FLOOR)
    for k in range(3):
        seat = (view.seat + 1 + k) % 4
        for suit in range(4):
            if voids[seat, suit]:
                dep[k, 13 * suit:13 * suit + 13] = 0.0
    dep = np.where(keep[None, :], dep, 0.0)
    s = dep.sum(axis=0)
    ok = keep & (s > 0)
    dep[:, ok] /= s[ok]
    return {"raw": out_raw, "normalized": norm, "deployed": dep}


def sample_owner_assignment(raw: np.ndarray, view: PlayerView, rng, use_belief: bool = True):
    """One pass of SearchPlayer::TrySample: owner index 0..2 per unseen card, or None on failure.

    Most-constrained-card-first (ties broken by a coin flip, as theirs), owner drawn with
    weight max(belief, 1e-4) among opponents with remaining capacity and no void in the suit.
    No pinned cards exist in a hold deal. The random stream is numpy's, not their mt19937:
    the DISTRIBUTION is theirs, the individual draws are not reproducible against their binary.
    """
    caps = opponent_hand_sizes(view)
    voids = their_void_tracker(view)
    is_void = [[bool(voids[(view.seat + 1 + k) % 4, s]) for s in range(4)] for k in range(3)]
    todo = cards.cards_in(unseen_mask(view))
    owner = {}
    while todo:
        best_idx, best_count = -1, 4
        for i, c in enumerate(todo):
            count = sum(1 for k in range(3) if caps[k] > 0 and not is_void[k][c // 13])
            if count == 0:
                return None
            if count < best_count or (count == best_count and (rng.integers(2) == 1)):
                best_count, best_idx = count, i
        c = todo.pop(best_idx)
        w = [(max(float(raw[k, c]), BELIEF_FLOOR) if use_belief else 1.0)
             if (caps[k] > 0 and not is_void[k][c // 13]) else 0.0 for k in range(3)]
        total = w[0] + w[1] + w[2]
        r = rng.random() * total
        pick = 2
        for k in range(3):
            if r < w[k]:
                pick = k
                break
            r -= w[k]
        if w[pick] == 0.0:
            return None
        owner[c] = pick
        caps[pick] -= 1
    return owner


def sampler_marginals(name: str, view: PlayerView, rng, n_draws: int = 256):
    """Monte-Carlo marginals of THEIR determinization sampler: (3, 52), plus diagnostics.

    Mirrors SearchPlayer::SampleDeterminization's schedule: belief-weighted attempts first
    (their cap: 100 failed attempts per world), then unweighted attempts. Their final exact
    transportation fallback is replaced by continued unweighted retries (cap 10,000) -- it is
    reached only when the greedy sampler keeps dead-ending, and the fraction of worlds that
    needed any unweighted attempt is returned so the substitution is visible.
    """
    raw = belief_sigmoid(name, view)
    counts = np.zeros((3, 52))
    unweighted = 0
    for _ in range(n_draws):
        owner = None
        for attempt in range(10_000):
            owner = sample_owner_assignment(raw, view, rng, use_belief=attempt < 100)
            if owner is not None:
                if attempt >= 100:
                    unweighted += 1
                break
        if owner is None:
            raise RuntimeError("Perilune sampler port found no consistent world in 10,000 tries")
        for c, k in owner.items():
            counts[k, c] += 1.0
    return counts / n_draws, unweighted / n_draws
