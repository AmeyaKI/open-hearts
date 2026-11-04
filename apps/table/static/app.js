"use strict";
/* The Table - front end. Vanilla JS, no dependencies. See apps/table/api_contract.py. */

var MOCK = new URLSearchParams(location.search).get("mock") === "1";
var SEAT_LABELS = ["South", "West", "North", "East"];
var SEAT_POS = ["bottom", "left", "top", "right"];
var KINDS = [
  ["human", "Human"],
  ["search", "Search bot (main)"],
  ["heuristic", "Heuristic bot"],
  ["random", "Random bot"]
];
var KIND_LABEL = { human: "human", search: "search", heuristic: "heuristic", random: "random" };
var SUITS = { c: "♣", d: "♦", s: "♠", h: "♥" };
var SUIT_NAME = { c: "clubs", d: "diamonds", s: "spades", h: "hearts" };
var HAND_SUIT_ORDER = ["s", "h", "c", "d"];
var RANKS = "23456789TJQKA";

var state = null;
var view = "auto"; // "auto" follows state.phase; "setup" forces the setup screen
var busy = false;
var inflight = false;
var dismissedError = null;
var localError = null;
var lastKey = null;

/* animation timing (ms) */
var THINK_MS = 350;      /* "..." on a bot seat before its card appears */
var PLAY_GAP_MS = 650;   /* spacing between consecutive plays (thinking time included for bots) */
var SLIDE_MS = 300;      /* card slide from seat/hand into the trick */
var HOLD_MS = 700;       /* pause on a completed trick before it is gathered */
var GATHER_MS = 500;     /* four cards converge and slide to the winner */
var SLIDE_FROM = { 0: [0, 200], 1: [-260, 0], 2: [0, -200], 3: [260, 0] };
var HAND_FROM = [0, 320];
var TO_CENTER = { 0: [0, -92], 1: [82, 0], 2: [0, 92], 3: [-82, 0] };
var TO_WINNER = { 0: [0, 260], 1: [-320, 0], 2: [0, -230], 3: [320, 0] };

/* animation state: what the table has actually SHOWN so far (state is what the server last said) */
var rendered = [];       /* flat list of {seat, card}: every play shown on the table so far */
var gTricks = 0;         /* completed tricks already gathered away (points/last-trick are derived from these) */
var handId = null;
var epoch = 0;           /* bumped to cancel a running pump */
var pumping = false;
var optimistic = false;  /* the human's own card was placed before the server replied */
var thinking = null;     /* seat showing "..." */
var enter = null;        /* {idx, at, hand}: the card currently sliding in */
var gather = null;       /* {at, winner}: a gather in progress */
var panelState = null;   /* state the advisor/reasoning panels are showing (only updated when drained) */

var setupDraft = {
  names: ["You", "West", "North", "East"],
  kinds: ["human", "search", "search", "search"],
  seed: "",
  advisor: true
};

/* ---------- helpers ---------- */

function esc(s) {
  return String(s).replace(/[&<>"']/g, function (c) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
  });
}
function rankLabel(r) { return r === "T" ? "10" : r; }
function isRed(suit) { return suit === "d" || suit === "h"; }
function cardLabel(c) { return rankLabel(c[0]) + SUITS[c[1]]; }
function seatName(s) {
  if (state && state.seats && state.seats[s]) return state.seats[s].name || SEAT_LABELS[s];
  return SEAT_LABELS[s];
}
function handLabel(name) { return name === "You" ? "Your hand" : name + "'s hand"; }
function isAnimating() { return pumping || optimistic; }
function reduced() { return !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches); }
function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
function flat(s) {
  var out = [];
  ((s && s.history) || []).forEach(function (h) { out.push({ seat: h.seat, card: h.card }); });
  ((s && s.current_trick) || []).forEach(function (p) { out.push({ seat: p.seat, card: p.card }); });
  return out;
}
function trickWinner(plays) {
  var lead = plays[0].card[1], best = plays[0];
  plays.forEach(function (p) {
    if (p.card[1] === lead && RANKS.indexOf(p.card[0]) > RANKS.indexOf(best.card[0])) best = p;
  });
  return best.seat;
}
function trickPoints(plays) {
  var n = 0;
  plays.forEach(function (p) { if (p.card[1] === "h") n += 1; else if (p.card === "Qs") n += 13; });
  return n;
}
function redraw() {
  lastKey = null;
  render();
}
function animRedraw() {
  lastKey = null;
  if (view === "setup" || (state && state.phase === "setup")) return; /* never rebuild the setup form under the user's fingers */
  render();
}
function settledInSync() {
  return !isAnimating() && state && rendered.length === flat(state).length &&
    gTricks * 4 === ((state.history || []).length);
}
function dispScores() {
  if (settledInSync()) return state.scores;
  var sc = [0, 0, 0, 0];
  for (var t = 0; t < gTricks; t++) {
    var pl = rendered.slice(t * 4, t * 4 + 4);
    if (pl.length === 4) sc[trickWinner(pl)] += trickPoints(pl);
  }
  return sc;
}
function dispToPlay() {
  if (!isAnimating()) return state.phase === "playing" ? state.to_play : null;
  if (rendered.length - gTricks * 4 >= 4) return null;
  var seq = flat(state);
  if (rendered.length < seq.length) return seq[rendered.length].seat;
  if (rendered.length) return (rendered[rendered.length - 1].seat + 1) % 4;
  return state.to_play;
}
function lastTrickData() {
  if (settledInSync()) return state.last_trick || null;
  if (gTricks > 0) {
    var pl = rendered.slice((gTricks - 1) * 4, gTricks * 4);
    if (pl.length === 4) return { plays: pl, winner: trickWinner(pl), points: trickPoints(pl) };
  }
  return null;
}
function cmpCards(a, b) {
  var sa = HAND_SUIT_ORDER.indexOf(a[1]), sb = HAND_SUIT_ORDER.indexOf(b[1]);
  if (sa !== sb) return sa - sb;
  return RANKS.indexOf(a[0]) - RANKS.indexOf(b[0]);
}

function cardHTML(c, cls, extra) {
  var suit = c[1];
  return '<button type="button" class="card ' + (isRed(suit) ? "red " : "black ") + (cls || "") + '" data-card="' + c + '" ' +
    (extra || "") + ">" +
    '<span class="corner"><span class="rank">' + rankLabel(c[0]) + '</span><span class="pip">' + SUITS[suit] + "</span></span>" +
    '<span class="big">' + SUITS[suit] + "</span></button>";
}
function miniCard(c) {
  return '<span class="mini ' + (isRed(c[1]) ? "red" : "black") + '">' + cardLabel(c) + "</span>";
}
function backCard() { return '<span class="card back" aria-hidden="true"></span>'; }

/* ---------- network ---------- */

function apiGet() {
  var url = MOCK ? "/static/fixture_state.json" : "/api/state";
  return fetch(url, { cache: "no-store" }).then(function (r) { return r.json(); });
}

function apiPost(path, body) {
  if (MOCK) return apiGet().then(function (j) { return { ok: true, json: j }; });
  return fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {})
  }).then(function (r) {
    return r.json().catch(function () { return null; }).then(function (j) { return { ok: r.ok, json: j }; });
  });
}

function setState(s) {
  if (!s) return;
  var next = JSON.stringify(s);
  if (state && next === JSON.stringify(state) && !localError) return;
  ingest(s, false);
  redraw();
}

function refresh() {
  if (inflight || busy) return Promise.resolve();
  inflight = true;
  return apiGet().then(function (s) {
    localError = null;
    setState(s);
  }).catch(function (e) {
    localError = "Cannot reach the server (" + e.message + ").";
    render();
  }).then(function () { inflight = false; });
}

/* ---------- animation queue ---------- */

/* Show the server's state at once: cancels any running pump. */
function resetInstant() {
  epoch++;
  pumping = false;
  optimistic = false;
  thinking = null;
  enter = null;
  gather = null;
  rendered = flat(state);
  gTricks = Math.floor(((state && state.history) || []).length / 4);
  handId = state ? state.hand_id : null;
  panelState = state;
}

/* A new server state arrived. If it extends what is shown, queue the difference; otherwise reset. */
function ingest(s, forceReset, fresh) {
  state = s;
  if (fresh && s.phase !== "setup") {
    /* A newly dealt hand: start from an EMPTY table and animate the bots' opening plays too. */
    epoch++;
    pumping = false;
    optimistic = false;
    thinking = null;
    enter = null;
    gather = null;
    rendered = [];
    gTricks = 0;
    handId = s.hand_id;
    panelState = s;
    startPump();
    return;
  }
  var seq = flat(s);
  var ok = !forceReset && s.phase !== "setup" && s.hand_id === handId && rendered.length <= seq.length &&
    rendered.every(function (p, i) { return p.seat === seq[i].seat && p.card === seq[i].card; });
  if (!ok) resetInstant();
  else startPump();
}

function startPump() {
  if (pumping) return;
  pumping = true;
  run(epoch);
}

async function run(my) {
  try {
    while (my === epoch) {
      var seq = flat(state);
      if (rendered.length - gTricks * 4 >= 4) { await doGather(my); continue; }
      if (rendered.length < seq.length) { await doPlay(my, seq[rendered.length]); continue; }
      break;
    }
  } finally {
    if (my === epoch) {
      pumping = false;
      thinking = null;
      enter = null;
      panelState = state;
      animRedraw();
    }
  }
}

async function doPlay(my, play) {
  var info = state.seats && state.seats[play.seat];
  var human = !!(info && info.is_human);
  if (!human) {
    thinking = play.seat;
    animRedraw();
    await sleep(THINK_MS);
    if (my !== epoch) return;
    thinking = null;
  }
  rendered.push({ seat: play.seat, card: play.card });
  enter = { idx: rendered.length - 1, at: Date.now(), hand: human };
  animRedraw();
  if (rendered.length - gTricks * 4 >= 4) return; /* trick complete: the gather's hold covers the pause */
  await sleep(human ? PLAY_GAP_MS : PLAY_GAP_MS - THINK_MS);
}

async function doGather(my) {
  var plays = rendered.slice(gTricks * 4, gTricks * 4 + 4);
  await sleep(HOLD_MS);
  if (my !== epoch) return;
  gather = { at: Date.now(), winner: trickWinner(plays) };
  animRedraw();
  await sleep(GATHER_MS);
  if (my !== epoch) return;
  gather = null;
  enter = null;
  gTricks++;
  animRedraw();
}

function playCard(card) {
  if (MOCK) { act("/api/play", { card: card }); return; }
  rendered.push({ seat: state.to_play, card: card });
  enter = { idx: rendered.length - 1, at: Date.now(), hand: true };
  optimistic = true;
  act("/api/play", { card: card }, null, { optimistic: true });
}

function act(path, body, after, opts) {
  opts = opts || {};
  if (busy) return;
  busy = true;
  localError = null;
  dismissedError = null;
  lastKey = null;
  render();
  apiPost(path, body).then(function (res) {
    busy = false;
    var wasOpt = optimistic;
    optimistic = false;
    var j = res.json;
    if (j && j.phase) {
      ingest(j, !!opts.instant, !!opts.fresh);
      if (res.ok && after) after();
    } else {
      if (!res.ok) localError = (j && j.error) || "The server rejected that action.";
      if (wasOpt) resetInstant();
    }
    redraw();
  }).catch(function (e) {
    busy = false;
    optimistic = false;
    if (state) resetInstant();
    localError = "Cannot reach the server (" + e.message + ").";
    redraw();
  });
}

/* ---------- rendering ---------- */

function render() {
  var key = JSON.stringify([view, state, busy, dismissedError, localError]);
  if (key === lastKey) return;
  lastKey = key;
  if (!isAnimating()) panelState = state;
  var app = document.getElementById("app");
  if (!state) {
    app.innerHTML = '<p class="loading">' + esc(localError || "Loading the table...") + "</p>";
    return;
  }
  var showSetup = view === "setup" || state.phase === "setup";
  var html = errorStrip();
  html += showSetup ? setupHTML() : gameHTML();
  app.innerHTML = html;
}

function errorStrip() {
  var msg = localError || (state && state.error);
  if (!msg || msg === dismissedError) return "";
  return '<div class="error-strip" role="alert"><span>' + esc(msg) +
    '</span><button type="button" data-act="dismiss" aria-label="Dismiss">×</button></div>';
}

function setupHTML() {
  var h = '<div class="setup"><h1>The Table</h1><p class="sub">Four seats, one deck. Choose who sits where.</p><div class="setup-card">';
  for (var s = 0; s < 4; s++) {
    h += '<div class="setup-row"><span class="seat-tag">' + SEAT_LABELS[s] + '</span>' +
      '<input type="text" data-setup="name" data-seat="' + s + '" value="' + esc(setupDraft.names[s]) + '" maxlength="24" aria-label="Name for seat ' + s + '">' +
      '<select data-setup="kind" data-seat="' + s + '" aria-label="Kind for seat ' + s + '">';
    KINDS.forEach(function (k) {
      h += '<option value="' + k[0] + '"' + (setupDraft.kinds[s] === k[0] ? " selected" : "") + ">" + k[1] + "</option>";
    });
    h += "</select></div>";
  }
  h += '<div class="setup-row"><span class="seat-tag">Seed</span>' +
    '<input type="text" inputmode="numeric" data-setup="seed" value="' + esc(setupDraft.seed) + '" placeholder="optional (random if empty)">' +
    "</div>" +
    '<label class="check"><input type="checkbox" data-setup="advisor"' + (setupDraft.advisor ? " checked" : "") + "> Advisor on</label>" +
    '<div class="setup-actions">' +
    (state && state.phase !== "setup" ? '<button type="button" class="btn ghost" data-act="cancel-setup">Back to table</button>' : "") +
    '<button type="button" class="btn primary" data-act="deal"' + (busy ? " disabled" : "") + ">Deal</button></div>" +
    '<p class="hint">' + (busy ? "Dealing..." : "Bots play until the first human must act.") + "</p>" +
    "</div></div>";
  return h;
}

function seatBoxHTML(s) {
  var info = state.seats[s];
  var tp = dispToPlay();
  var turn = state.phase === "playing" && tp === s;
  var h = '<div class="seat seat-' + SEAT_POS[s] + (turn || thinking === s ? " turn" : "") + '">' +
    '<div class="seat-name">' + esc(info.name) + '<span class="badge kind-' + esc(info.kind) + '">' + esc(KIND_LABEL[info.kind] || info.kind) + "</span></div>" +
    '<div class="seat-pts"><b>' + dispScores()[s] + "</b> pts</div>";
  if (thinking === s) h += '<div class="seat-turn thinking" aria-label="thinking">&middot;&middot;&middot;</div>';
  else if (turn && !isAnimating()) h += '<div class="seat-turn">' + (info.is_human ? "your move" : "thinking...") + "</div>";
  return h + "</div>";
}

/* inline animation for the trick card at flat index idx played by seat */
function animStyle(idx, seat) {
  if (reduced()) return "";
  var now = Date.now(), v;
  if (gather) {
    var ge = Math.min(GATHER_MS, now - gather.at), c = TO_CENTER[seat], w = TO_WINNER[gather.winner];
    return ' style="--cx:' + c[0] + "px;--cy:" + c[1] + "px;--wx:" + w[0] + "px;--wy:" + w[1] + "px;animation:gather " +
      GATHER_MS + "ms ease-in-out " + (-ge) + 'ms both"';
  }
  if (enter && enter.idx === idx) {
    var se = now - enter.at;
    if (se < SLIDE_MS) {
      v = enter.hand && seat === 0 ? HAND_FROM : SLIDE_FROM[seat];
      return ' style="--sx:' + v[0] + "px;--sy:" + v[1] + "px;animation:slide-in " + SLIDE_MS + "ms ease-out " + (-se) + 'ms both"';
    }
  }
  return "";
}

function trickHTML() {
  var slots = {}, base = gTricks * 4;
  rendered.slice(base).forEach(function (p, k) { slots[p.seat] = { card: p.card, idx: base + k }; });
  var h = '<div class="trick">';
  for (var s = 0; s < 4; s++) {
    h += '<div class="trick-slot slot-' + SEAT_POS[s] + '">';
    if (slots[s]) h += cardHTML(slots[s].card, "small static", 'tabindex="-1"' + animStyle(slots[s].idx, s));
    h += "</div>";
  }
  return h + "</div>";
}

function lastTrickHTML() {
  var lt = lastTrickData();
  var h = '<div class="last-trick"><h3>Last trick</h3>';
  if (!lt) return h + '<p class="muted">None yet.</p></div>';
  h += '<ol class="lt-plays">';
  lt.plays.forEach(function (p, i) {
    h += '<li class="lt-play' + (p.seat === lt.winner ? " won" : "") + '"><span class="who">' + esc(seatName(p.seat)) +
      (i === 0 ? ' <i class="led">led</i>' : "") + "</span>" + miniCard(p.card) +
      (p.seat === lt.winner ? '<span class="star" title="took the trick">&#9733;</span>' : "") + "</li>";
  });
  h += "</ol><p class=\"lt-result\"><b>" + esc(seatName(lt.winner)) + "</b> took it, " + lt.points + (lt.points === 1 ? " point" : " points") + "</p></div>";
  return h;
}

function handHTML() {
  if (state.phase === "over") return "";
  var anim = isAnimating();
  var tp = dispToPlay();
  var name = tp !== null && tp !== undefined ? seatName(tp) : "";
  var left = Math.max(1, 13 - gTricks);
  if (state.human_turn && !state.revealed) {
    var n = anim ? left : Math.max(1, 13 - state.trick_number), backs = "";
    for (var i = 0; i < n; i++) backs += backCard();
    return '<div class="hand-wrap"><div class="hand hidden">' + backs + "</div>" +
      (anim ? "" : '<button type="button" class="btn primary reveal" data-act="reveal"' + (busy ? " disabled" : "") +
      ">Tap to reveal " + esc(name) + "'s hand</button>") + "</div>";
  }
  if (!state.human_turn || !state.hand) {
    var m = anim ? left : Math.max(1, 13 - state.trick_number), b2 = "";
    for (var j = 0; j < m; j++) b2 += backCard();
    return '<div class="hand-wrap"><div class="hand hidden dim">' + b2 + '</div><p class="hand-note">Waiting for ' + esc(name || "the bots") + "...</p></div>";
  }
  var played = {};
  rendered.forEach(function (p) { played[p.card] = true; });
  var legal = {};
  (state.legal || []).forEach(function (c) { legal[c] = true; });
  var suggested = anim ? null : (state.advice ? state.advice.suggested : null);
  var h = '<div class="hand-wrap"><div class="hand-title">' + esc(handLabel(seatName(state.to_play))) + '</div><div class="hand' + (anim ? " waiting" : "") + '">';
  state.hand.filter(function (c) { return !played[c]; }).sort(cmpCards).forEach(function (c) {
    var cls = anim ? "wait" : (legal[c] ? "legal" : "illegal");
    if (c === suggested) cls += " suggested";
    var reason = (state.illegal_reasons && state.illegal_reasons[c]) || "not legal";
    var extra;
    if (anim) extra = 'disabled aria-disabled="true"';
    else if (legal[c]) extra = (busy ? 'disabled' : '') + ' title="Play ' + cardLabel(c) + '"';
    else extra = 'aria-disabled="true" title="' + esc(reason) + '" data-reason="' + esc(reason) + '"';
    h += cardHTML(c, cls, extra);
  });
  h += '</div><div class="caption" id="caption">' + (busy || anim ? "The table is playing..." : "&nbsp;") + "</div></div>";
  return h;
}

function overHTML() {
  var min = Math.min.apply(null, state.scores);
  var winners = [];
  state.scores.forEach(function (p, s) { if (p === min) winners.push(seatName(s)); });
  var h = '<div class="over"><h2>Hand over</h2><div class="over-scores">';
  state.scores.forEach(function (p, s) {
    h += '<div class="over-seat' + (p === min ? " winner" : "") + '"><span>' + esc(seatName(s)) + "</span><b>" + p + "</b></div>";
  });
  h += "</div><p>" + (winners.length > 1 ? "Tie for the lowest score: " : "Winner (lowest score): ") + "<b>" + esc(winners.join(" & ")) + "</b></p>" +
    '<button type="button" class="btn primary" data-act="new">New hand</button></div>';
  return h;
}

function controlsHTML() {
  var dis = busy ? " disabled" : "";
  return '<div class="controls">' +
    '<button type="button" class="btn" data-act="undo"' + dis + ">Undo</button>" +
    '<button type="button" class="btn" data-act="new">New hand</button>' +
    '<button type="button" class="btn" data-act="resume"' + dis + ">Resume</button>" +
    '<button type="button" class="btn toggle' + (state.advisor_on ? " on" : "") + '" data-act="advisor"' + dis + ">Advisor: " + (state.advisor_on ? "on" : "off") + "</button>" +
    "</div>";
}

function advisorHTML(ps) {
  var a = ps.advice;
  if (!a) return "";
  var cands = a.candidates || [];
  var max = 0;
  cands.forEach(function (c) { if (c.mean_points > max) max = c.mean_points; });
  var h = '<section class="panel advisor"><h3>Advisor</h3><p class="suggest">Suggested: ' + miniCard(a.suggested) + "</p>";
  cands.forEach(function (c) {
    var w = max > 0 ? Math.max(3, (c.mean_points / max) * 100) : 3;
    h += '<div class="bar-row' + (c.card === a.suggested ? " best" : "") + '"><span class="bar-label">' + miniCard(c.card) + " " +
      c.mean_points.toFixed(2) + ' pts</span><span class="bar-track"><span class="bar-fill" style="width:' + w.toFixed(1) + '%"></span></span></div>';
  });
  h += '<p class="meta">worlds ' + esc(a.worlds) + " · " + Number(a.latency_s).toFixed(3) + " s</p></section>";
  return h;
}

function heatColor(p) {
  var light = Math.round(100 - p * 62);
  return "hsl(205, 80%, " + light + "%)";
}

var GRID_SUITS = ["c", "d", "s", "h"];

function reasoningHTML(ps) {
  var a = ps.advice;
  if (!a) return "";
  var b = a.belief;
  var h = '<section class="panel reasoning"><h3>Reasoning</h3>';
  if (b && b.cards && b.opponents) {
    h += '<table class="heat2"><thead><tr><th colspan="2"></th>';
    for (var r = 0; r < RANKS.length; r++) h += "<th>" + rankLabel(RANKS[r]) + "</th>";
    h += "</tr></thead><tbody>";
    b.opponents.forEach(function (o, i) {
      var voids = (b.voids && b.voids[i]) || [];
      GRID_SUITS.forEach(function (suit, k) {
        h += '<tr' + (k === 0 ? ' class="grp-first"' : "") + ">";
        if (k === 0) h += '<th class="opp" rowspan="4">' + esc(o.name) + "</th>";
        h += '<th class="suit ' + (isRed(suit) ? "red" : "black") + '" title="' + SUIT_NAME[suit] + '">' + SUITS[suit] + "</th>";
        for (var q = 0; q < RANKS.length; q++) {
          var card = RANKS[q] + suit, lbl = cardLabel(card);
          if (voids.indexOf(suit) >= 0) {
            h += '<td class="void" title="' + esc(o.name) + " is known void in " + SUIT_NAME[suit] + '"></td>';
          } else if (!Object.prototype.hasOwnProperty.call(b.cards, card)) {
            h += '<td class="seen" title="' + lbl + ' already seen">&ndash;</td>';
          } else {
            var p = b.cards[card][i] || 0;
            h += '<td class="' + (p >= 0.55 ? "hi" : "") + '" style="background:' + (p > 0 ? heatColor(p) : "#fff") +
              '" title="' + esc(o.name) + " holds " + lbl + ": " + Math.round(p * 100) + '%">' +
              (p >= 0.15 ? Math.round(p * 100) + "%" : "") + "</td>";
          }
        }
        h += "</tr>";
      });
    });
    h += '</tbody></table><p class="legend"><span class="sw seen">&ndash;</span> already seen (played or in your hand) &nbsp; ' +
      '<span class="sw void"></span> known void &nbsp; darker blue = more likely</p>';
  }
  (a.sentences || []).forEach(function (s) { h += '<p class="sentence">' + esc(s) + "</p>"; });
  return h + "</section>";
}

function gameHTML() {
  var anim = isAnimating();
  var h = '<header class="topbar"><h1>The Table</h1>' + (MOCK ? '<span class="mock">mock data</span>' : "") + controlsHTML() + "</header>";
  h += '<div class="layout"><main class="felt-col"><div class="felt">';
  var broken = settledInSync() ? state.hearts_broken : rendered.some(function (p) { return p.card[1] === "h"; });
  var trickNo = anim ? Math.min(13, gTricks + 1) : Math.min(13, state.trick_number + (state.phase === "over" ? 0 : 1));
  h += '<div class="statusbar"><span>Trick <b>' + trickNo + "</b> of 13</span>" +
    '<span class="hb ' + (broken ? "broken" : "") + '">♥ ' + (broken ? "hearts broken" : "hearts not broken") + "</span></div>";
  h += '<div class="table-grid">' + seatBoxHTML(2) + seatBoxHTML(1) + '<div class="center">' + trickHTML() + "</div>" + seatBoxHTML(3) + seatBoxHTML(0) + lastTrickHTML() + "</div>";
  if (state.phase === "over" && !anim) h += overHTML();
  h += handHTML() + "</div></main>";
  var ps = panelState;
  var stale = anim ? " stale" : "";
  var adv = ps ? advisorHTML(ps) : "";
  if (adv) h += '<aside class="side' + stale + '">' + adv + "</aside>";
  var rea = ps ? reasoningHTML(ps) : "";
  if (rea) h += '<div class="reasoning-row' + stale + '">' + rea + "</div>";
  h += "</div>";
  h += '<footer class="foot">hand #' + esc(state.hand_id) + " · seed " + esc(state.seed) + " · log " + esc(state.log_path) + "</footer>";
  return h;
}

/* ---------- events ---------- */

function readSeed() {
  var t = String(setupDraft.seed).trim();
  if (t === "") return { ok: true, v: null };
  if (!/^-?\d+$/.test(t)) return { ok: false };
  return { ok: true, v: parseInt(t, 10) };
}

document.addEventListener("click", function (e) {
  var cardEl = e.target.closest(".card.legal");
  if (cardEl && !busy && !isAnimating() && state && state.human_turn) {
    playCard(cardEl.getAttribute("data-card"));
    return;
  }
  var btn = e.target.closest("[data-act]");
  if (!btn || btn.disabled) return;
  switch (btn.getAttribute("data-act")) {
    case "dismiss":
      dismissedError = localError || (state && state.error);
      localError = null;
      lastKey = null;
      render();
      break;
    case "new":
      view = "setup"; lastKey = null; render(); break;
    case "cancel-setup":
      view = "auto"; lastKey = null; render(); break;
    case "deal":
      var sd = readSeed();
      if (!sd.ok) { localError = "Seed must be a whole number."; lastKey = null; render(); break; }
      act("/api/new", {
        seats: [0, 1, 2, 3].map(function (s) {
          return { kind: setupDraft.kinds[s], name: setupDraft.names[s].trim() || SEAT_LABELS[s] };
        }),
        seed: sd.v,
        advisor_on: setupDraft.advisor
      }, function () { view = "auto"; }, { fresh: true });
      break;
    case "reveal": act("/api/reveal", {}); break;
    case "undo": act("/api/undo", {}, null, { instant: true }); break;
    case "resume": act("/api/resume", {}, function () { view = "auto"; }, { instant: true }); break;
    case "advisor": act("/api/advisor", { on: !state.advisor_on }); break;
  }
});

document.addEventListener("input", function (e) {
  var t = e.target, kind = t.getAttribute && t.getAttribute("data-setup");
  if (!kind) return;
  var s = parseInt(t.getAttribute("data-seat"), 10);
  if (kind === "name") setupDraft.names[s] = t.value;
  else if (kind === "kind") setupDraft.kinds[s] = t.value;
  else if (kind === "seed") setupDraft.seed = t.value;
  else if (kind === "advisor") setupDraft.advisor = t.checked;
  lastKey = null; // draft changed; next render must not be skipped, but we do not re-render now (keeps focus)
});

document.addEventListener("mouseover", function (e) {
  var cap = document.getElementById("caption");
  if (!cap) return;
  var c = e.target.closest && e.target.closest(".card.illegal");
  if (c) cap.textContent = cardLabel(c.getAttribute("data-card")) + ": " + (c.getAttribute("data-reason") || "not legal");
  else if (!busy) cap.innerHTML = "&nbsp;";
});

setInterval(function () {
  if (!state || MOCK) return;
  var showSetup = view === "setup" || state.phase === "setup";
  if (showSetup) return;
  if (state.phase === "playing" && !state.human_turn) refresh();
}, 700);

refresh();
