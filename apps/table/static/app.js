"use strict";
/* The Table - front end. Vanilla JS, no dependencies. See apps/table/api_contract.py. */

var MOCK_Q = new URLSearchParams(location.search).get("mock");
var MOCK = MOCK_Q === "1" || MOCK_Q === "real" || MOCK_Q === "live" || MOCK_Q === "pass";
var MOCK_URL = MOCK_Q === "real" ? "/static/fixture_real.json" :
  (MOCK_Q === "live" ? "/static/fixture_live.json" :
  (MOCK_Q === "pass" ? "/static/fixture_pass.json" : "/static/fixture_state.json"));
var SEAT_LABELS = ["South", "West", "North", "East"];
var SEAT_POS = ["bottom", "left", "top", "right"];
var FALLBACK_KINDS = [
  { kind: "human", label: "Human" },
  { kind: "table", label: "Human at the table" },
  { kind: "search", label: "TallyHearts (main bot)" },
  { kind: "heuristic", label: "Heuristic bot" },
  { kind: "random", label: "Random bot" }
];
var kinds = FALLBACK_KINDS.slice();
var KIND_LABEL = { human: "human", table: "TABLE", search: "search", heuristic: "heuristic", random: "random" };
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
var announceBanner = null; /* {seat, card, name, until} */
var lastAnnounceKey = null;
var flashDiv = false;
var lastDiv = 0;
var panelState = null;   /* state the advisor/reasoning panels are showing (only updated when drained) */

var setupDraft = {
  step: "seats",           /* "seats" | "hands" (real-table hand entry) */
  hands: { 0: [], 1: [], 2: [], 3: [] },
  params: ["", "", "", ""],
  names: ["You", "West", "North", "East"],
  kinds: ["human", "search", "search", "search"],
  seed: "",
  advisor: true,
  coach: [true, false, false, false], /* per-seat "Coach" tick for keyboard humans */
  preset: "solo",          /* "solo" | "live" | "pass" | null (edited by hand) */
  leader: null             /* real mode: seat that holds 2c when no typed hand does */
};
var PRESETS = {
  solo: {
    label: "Solo vs bots",
    text: "You at this keyboard (coach on) against three TallyHearts bots. The server deals.",
    kinds: ["human", "search", "search", "search"], names: ["You", "West", "North", "East"], coach: [true, false, false, false]
  },
  live: {
    label: "Live table",
    text: "You sit at a real table with three people and real cards. You type in your 13 cards, enter each card the others play, and TallyHearts suggests yours.",
    kinds: ["human", "table", "table", "table"], names: ["You", "West", "North", "East"], coach: [true, false, false, false]
  },
  pass: {
    label: "Pass the laptop",
    text: "Four people share this laptop. The server deals. Each player taps to see only their own hand. Coaching is on for seat 1 only unless you tick it.",
    kinds: ["human", "human", "human", "human"], names: ["Player 1", "Player 2", "Player 3", "Player 4"], coach: [true, false, false, false]
  }
};
var entryDraft = "";     /* text typed in the card box */
var entryMsg = "";       /* inline message under the card box */
var entryKey = "";       /* which prompt the box belongs to (cleared when the turn changes) */

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
/* "7h", "Qs", "10d", "td", "7♥" -> "7h" / "Qs" / "Td"; null when unreadable */
function parseTypedCard(text) {
  var t = String(text || "").trim().toLowerCase().replace(/\s+/g, "");
  var m = /^(10|[2-9tjqka])([cdsh\u2663\u2666\u2660\u2665])$/.exec(t);
  if (!m) return null;
  var suitMap = { "\u2663": "c", "\u2666": "d", "\u2660": "s", "\u2665": "h" };
  var rank = m[1] === "10" ? "T" : m[1].toUpperCase();
  return rank + (suitMap[m[2]] || m[2]);
}
function keyboardHumans() {
  var n = 0;
  ((state && state.seats) || []).forEach(function (x) { if (x.kind === "human") n++; });
  return n;
}
function cardsHeldBy(seat) {
  var n = 0;
  rendered.forEach(function (p) { if (p.seat === seat) n++; });
  return Math.max(0, 13 - n);
}
function miniLabel(c) { return '<span class="' + (isRed(c[1]) ? "red" : "black") + '">' + cardLabel(c) + "</span>"; }
function pointsText(n) { return n + (n === 1 ? " point" : " points"); }

/* What the tool is waiting for, as HTML for the status strip: {cls, html}. */
function statusInfo() {
  if (!state || state.phase === "setup") return { cls: "", html: "" };
  if (state.phase === "over" && !isAnimating()) return { cls: "over", html: "Hand over. Start a new hand when you are ready." };
  if (gather) {
    var gp = rendered.slice(gTricks * 4, gTricks * 4 + 4);
    if (gp.length === 4) return { cls: "trick", html: "Trick over: " + esc(seatName(trickWinner(gp))) + " took it (" + pointsText(trickPoints(gp)) + ")" };
  }
  if (isAnimating()) return { cls: "busy", html: "Cards are being played..." };
  var tp = state.to_play, nm = esc(seatName(tp)), info = state.seats[tp] || {};
  var lead = "";
  if (state.last_trick && (state.current_trick || []).length === 0 && state.trick_number > 0) {
    lead = "Trick over: " + esc(seatName(state.last_trick.winner)) + " took it (" + pointsText(state.last_trick.points) + "). ";
  }
  if (state.table_turn) return { cls: "wait", html: lead + "Waiting for " + nm + " to play (enter their card)" };
  if (state.human_turn && !state.revealed) return { cls: "hidden", html: lead + "Waiting for " + nm + " to tap \u201cI\u2019m " + nm + ", show my hand\u201d" };
  if (state.human_turn) {
    var who = (state.mode === "real" || keyboardHumans() <= 1) && tp === 0 ? "Your turn" : nm + "\u2019s turn";
    if (state.advice && state.advice.suggested) return { cls: "you", html: lead + who + " \u2014 TallyHearts suggests " + miniLabel(state.advice.suggested) };
    return { cls: "you", html: lead + who + " \u2014 click or type the card played" };
  }
  return { cls: "busy", html: lead + nm + " (" + esc(KIND_LABEL[info.kind] || info.kind) + ") is playing..." };
}

function backCard() { return '<span class="card back" aria-hidden="true"></span>'; }

/* ---------- network ---------- */

function apiGet() {
  var url = MOCK ? MOCK_URL : "/api/state";
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
function trackExtras(s, fresh) {
  if (fresh || (state && state.hand_id !== s.hand_id)) { lastAnnounceKey = null; announceBanner = null; lastDiv = 0; }
  var ak = s.announce ? JSON.stringify(s.announce) : null;
  if (ak && ak !== lastAnnounceKey) {
    var nm = s.announce.name || (s.seats && s.seats[s.announce.seat] && s.seats[s.announce.seat].name) || SEAT_LABELS[s.announce.seat];
    announceBanner = { seat: s.announce.seat, card: s.announce.card, name: nm, until: Date.now() + 4000 };
    setTimeout(function () {
      if (announceBanner && Date.now() >= announceBanner.until - 50) { announceBanner = null; animRedraw(); }
    }, 4100);
  }
  lastAnnounceKey = ak;
  var d = s.divergences || 0;
  if (d > lastDiv && state && state.hand_id === s.hand_id) {
    flashDiv = true;
    setTimeout(function () { flashDiv = false; animRedraw(); }, 1500);
  }
  lastDiv = d;
}

function ingest(s, forceReset, fresh) {
  trackExtras(s, fresh);
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
  var instant = human || !!(info && info.kind === "table");
  if (!instant) {
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
  await sleep(instant ? PLAY_GAP_MS : PLAY_GAP_MS - THINK_MS);
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
  var key = JSON.stringify([view, state, busy, dismissedError, localError, setupDraft, announceBanner, flashDiv, entryMsg, optimistic]);
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
  var keep = document.activeElement && document.activeElement.id === "card-entry";
  app.innerHTML = html;
  focusEntry(keep);
}

/* The card box is the fastest way to transcribe: focus it whenever it is on screen. */
function focusEntry() {
  var el = document.getElementById("card-entry");
  if (!el) return;
  el.value = entryDraft;
  try { el.focus({ preventScroll: true }); el.setSelectionRange(el.value.length, el.value.length); } catch (e) { /* ignore */ }
}

function errorStrip() {
  var msg = localError || (state && state.error);
  if (!msg || msg === dismissedError) return "";
  return '<div class="error-strip" role="alert"><span>' + esc(msg) +
    '</span><button type="button" data-act="dismiss" aria-label="Dismiss">×</button></div>';
}

function kindInfo(k) {
  for (var i = 0; i < kinds.length; i++) if (kinds[i].kind === k) return kinds[i];
  return null;
}
function seatKindString(s) {
  var k = setupDraft.kinds[s], p = String(setupDraft.params[s] || paramDefault(k)).trim();
  var ki = kindInfo(k);
  return ki && ki.param ? k + ":" + p : k;
}
function paramDefault(k) {
  if (k === "expert") return "1";
  if (k === "personality") return "1";
  if (k === "perilune") return "v5";
  return "";
}
function tableSeats() {
  var out = [];
  for (var s = 0; s < 4; s++) if (setupDraft.kinds[s] === "table") out.push(s);
  return out;
}
function isRealSetup() { return tableSeats().length > 0; }
function paramInputHTML(s) {
  var k = setupDraft.kinds[s], ki = kindInfo(k);
  if (!ki || !ki.param) return "";
  var v = setupDraft.params[s] || paramDefault(k);
  if (k === "perilune") {
    return '<select class="param" data-setup="param" data-seat="' + s + '" aria-label="Perilune version for seat ' + s + '">' +
      ["v5", "v6.1"].map(function (o) { return '<option value="' + o + '"' + (v === o ? " selected" : "") + ">" + o + "</option>"; }).join("") + "</select>";
  }
  var ph = k === "expert" ? "id 1-200" : "seed";
  return '<input class="param" type="number" min="' + (k === "expert" ? 1 : 0) + '"' + (k === "expert" ? ' max="200"' : "") +
    ' data-setup="param" data-seat="' + s + '" value="' + esc(v) + '" placeholder="' + ph + '" aria-label="' + ph + ' for seat ' + s + '">';
}

function setupHTML() {
  if (setupDraft.step === "hands" && isRealSetup()) return handsEntryHTML();
  var real = isRealSetup();
  var h = '<div class="setup"><h1>The Table</h1><p class="sub">Four seats, one deck. Pick a way to play, then adjust the seats if you like.</p><div class="setup-card">';
  h += '<div class="presets" role="group" aria-label="Presets">';
  ["solo", "live", "pass"].forEach(function (k) {
    h += '<button type="button" class="btn preset' + (setupDraft.preset === k ? " on" : "") + '" data-act="preset" data-preset="' + k + '">' + esc(PRESETS[k].label) + "</button>";
  });
  h += "</div>";
  h += '<p class="explain preset-text">' + (setupDraft.preset ? esc(PRESETS[setupDraft.preset].text) : "Custom seats.") + "</p>";
  for (var s = 0; s < 4; s++) {
    h += '<div class="setup-row"><span class="seat-tag">' + SEAT_LABELS[s] + '</span>' +
      '<input type="text" data-setup="name" data-seat="' + s + '" value="' + esc(setupDraft.names[s]) + '" maxlength="24" aria-label="Name for seat ' + s + '">' +
      '<select data-setup="kind" data-seat="' + s + '" aria-label="Kind for seat ' + s + '">';
    kinds.forEach(function (k) {
      h += '<option value="' + esc(k.kind) + '"' + (setupDraft.kinds[s] === k.kind ? " selected" : "") + ">" + esc(k.label) + "</option>";
    });
    h += "</select>" + paramInputHTML(s);
    if (setupDraft.kinds[s] === "human") {
      h += '<label class="coach"><input type="checkbox" data-setup="coach" data-seat="' + s + '"' + (setupDraft.coach[s] ? " checked" : "") + "> Coach</label>";
    }
    h += "</div>";
  }
  h += '<p class="explain">Virtual: the server deals. Real table: at least one seat is \u2018Human at the table\u2019; type in every other seat\u2019s 13 cards, then enter the table players\u2019 cards as they fall. Coach: show TallyHearts\u2019s advice to that keyboard player.</p>';
  h += '<div class="setup-row seed-row"><span class="seat-tag">Seed</span>' +
    '<input type="text" inputmode="numeric" data-setup="seed" value="' + esc(setupDraft.seed) + '" placeholder="optional (random if empty)">' +
    "</div>" +
    '<label class="check"><input type="checkbox" data-setup="advisor"' + (setupDraft.advisor ? " checked" : "") + "> Advisor on (master switch)</label>" +
    '<div class="setup-actions">' +
    (state && state.phase !== "setup" ? '<button type="button" class="btn ghost" data-act="cancel-setup">Back to table</button>' : "") +
    '<button type="button" class="btn primary" data-act="deal"' + (busy ? " disabled" : "") + ">" + (real ? "Next: enter your cards" : "Deal") + "</button></div>" +
    '<p class="hint">' + (busy ? "Dealing..." : (real ? "You will enter the cards of every non-table seat next." : (setupDraft.preset === "pass" ? "Each player's hand stays hidden until they tap to show it." : "Bots play until the first human must act."))) + "</p>" +
    "</div></div>";
  return h;
}

function applyPreset(k) {
  var p = PRESETS[k];
  if (!p) return;
  setupDraft.preset = k;
  setupDraft.kinds = p.kinds.slice();
  setupDraft.names = p.names.slice();
  setupDraft.coach = p.coach.slice();
  setupDraft.params = ["", "", "", ""];
  setupDraft.hands = { 0: [], 1: [], 2: [], 3: [] };
  setupDraft.leader = null;
  setupDraft.step = "seats";
}

/* Real mode: which known (non-table) seat holds 2c in the typed hands, or -1. */
function twoClubsHolder() {
  for (var s = 0; s < 4; s++) {
    if (setupDraft.kinds[s] !== "table" && setupDraft.hands[s].indexOf("2c") >= 0) return s;
  }
  return -1;
}
/* The effective leader: the 2c holder, else the chosen table seat, else the lone table seat, else -1. */
function effectiveLeader() {
  var h = twoClubsHolder();
  if (h >= 0) return h;
  var tbl = tableSeats();
  if (tbl.length === 1) return tbl[0];
  if (setupDraft.leader !== null && tbl.indexOf(setupDraft.leader) >= 0) return setupDraft.leader;
  return -1;
}

var GRID_SUITS_PICK = ["s", "h", "c", "d"];

function pickGridHTML(opts) {
  var h = '<div class="pickgrid">';
  GRID_SUITS_PICK.forEach(function (suit) {
    h += '<div class="pickrow"><span class="picksuit ' + (isRed(suit) ? "red" : "black") + '" title="' + SUIT_NAME[suit] + '">' + SUITS[suit] + "</span>";
    for (var r = RANKS.length - 1; r >= 0; r--) {
      var c = RANKS[r] + suit, dis = opts.disabled(c);
      h += '<button type="button" class="pick ' + (isRed(suit) ? "red" : "black") + " " + opts.cls(c) + '"' + (dis ? ' disabled aria-disabled="true"' : "") +
        ' title="' + esc(opts.title(c)) + '" ' + opts.attrs(c) + ">" + rankLabel(RANKS[r]) + SUITS[suit] + "</button>";
    }
    h += "</div>";
  });
  return h + "</div>";
}

function pickedBy(c) {
  for (var s = 0; s < 4; s++) if (setupDraft.hands[s] && setupDraft.hands[s].indexOf(c) >= 0) return s;
  return -1;
}
function handsEntryHTML() {
  var tbl = tableSeats();
  var onlyYou = setupDraft.preset === "live" && tbl.length === 3;
  var h = '<div class="setup hands-entry"><h1>' + (onlyYou ? "Enter your 13 cards" : "Real table: enter the hands") + '</h1><p class="sub">' +
    (onlyYou ? "Click the 13 cards you were dealt. The other three hands stay unknown, as in real life."
      : "For each seat that is not at the real table, click exactly 13 cards. A card chosen for one seat is greyed out for the others.") + "</p>";
  var all = true;
  for (var s = 0; s < 4; s++) {
    if (tbl.indexOf(s) >= 0) continue;
    var n = setupDraft.hands[s].length;
    if (n !== 13) all = false;
    h += '<div class="setup-card pick-card"><div class="pick-head"><b>' + esc(setupDraft.names[s] || SEAT_LABELS[s]) + '</b> <span class="muted">(' + SEAT_LABELS[s] + ")</span>" +
      '<span class="counter' + (n === 13 ? " full" : "") + '">' + n + "/13</span>" +
      '<button type="button" class="btn ghost small" data-act="clear-hand" data-seat="' + s + '">Clear</button></div>' +
      pickGridHTML({
        cls: function (c) { return setupDraft.hands[s].indexOf(c) >= 0 ? "sel" : ""; },
        disabled: function (c) {
          var by = pickedBy(c);
          if (by >= 0 && by !== s) return true;
          return by !== s && n >= 13;
        },
        title: function (c) {
          var by = pickedBy(c);
          return by >= 0 && by !== s ? "already in " + (setupDraft.names[by] || SEAT_LABELS[by]) + "'s hand" : (n >= 13 && by !== s ? "13 cards already chosen" : cardLabel(c));
        },
        attrs: function (c) { return 'data-hand-seat="' + s + '" data-pick="' + c + '"'; }
      }) + "</div>";
  }
  var holder = twoClubsHolder();
  h += '<div class="setup-card leader-card"><b>Who leads the first trick (holds 2\u2663)?</b> ';
  if (holder >= 0) {
    h += '<span class="leader-auto">' + esc(setupDraft.names[holder] || SEAT_LABELS[holder]) + " \u2014 the 2\u2663 is in " +
      (holder === 0 && onlyYou ? "your" : esc(setupDraft.names[holder] || SEAT_LABELS[holder]) + "\u2019s") + " hand, so they lead.</span>";
  } else {
    h += '<select data-setup="leader" aria-label="Who leads the first trick">';
    if (tbl.length > 1) h += '<option value=""' + (setupDraft.leader === null ? " selected" : "") + ">Choose who holds 2\u2663...</option>";
    tbl.forEach(function (t) {
      var sel = (setupDraft.leader === t || (tbl.length === 1)) ? " selected" : "";
      h += '<option value="' + t + '"' + sel + ">" + esc(setupDraft.names[t] || SEAT_LABELS[t]) + "</option>";
    });
    h += "</select>";
    if (all) h += ' <span class="muted">(you do not hold 2\u2663, so one of the table players does)</span>';
  }
  h += "</div>";
  h += '<div class="setup-actions"><button type="button" class="btn ghost" data-act="hands-back">Back</button>' +
    '<button type="button" class="btn primary" data-act="hands-go"' + (busy ? " disabled" : "") + ">" + (busy ? "Starting..." : "Start hand") + "</button></div>" +
    '<p class="hint">' + (all ? "All hands complete." : "Every non-table seat needs exactly 13 cards.") + "</p></div>";
  return h;
}

function seatBoxHTML(s) {
  var info = state.seats[s];
  var tp = dispToPlay();
  var turn = state.phase === "playing" && tp === s;
  var h = '<div class="seat seat-' + SEAT_POS[s] + (turn || thinking === s ? " turn" : "") + (announceBanner && announceBanner.seat === s ? " announce" : "") + '">' +
    '<div class="seat-name">' + esc(info.name) + '<span class="badge kind-' + esc(info.kind) + '">' + esc(KIND_LABEL[info.kind] || info.kind) + "</span></div>" +
    '<div class="seat-pts"><b>' + dispScores()[s] + "</b> pts</div>";
  if (info.kind === "table") {
    var cnt = cardsHeldBy(s);
    h += '<div class="seat-count" title="Cards this person still holds (13 minus the cards they have played)">' + cnt + (cnt === 1 ? " card" : " cards") + " left</div>";
  }
  if (thinking === s) h += '<div class="seat-turn thinking" aria-label="thinking">&middot;&middot;&middot;</div>';
  else if (turn && !isAnimating()) h += '<div class="seat-turn">' + (info.is_human ? "your move" : (info.kind === "table" ? "plays at the table" : "thinking...")) + "</div>";
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
  var hideNow = optimistic && keyboardHumans() > 1; /* never leave a hand on screen after the card is played */
  if ((state.human_turn && !state.revealed) || (hideNow && state.human_turn)) {
    var n = anim ? left : Math.max(1, 13 - state.trick_number), backs = "";
    for (var i = 0; i < n; i++) backs += backCard();
    return '<div class="hand-wrap"><div class="hand hidden">' + backs + "</div>" +
      (anim || !state.human_turn ? "" : '<button type="button" class="btn primary reveal" data-act="reveal"' + (busy ? " disabled" : "") +
      ">I\u2019m " + esc(name) + ", show my hand</button>") + "</div>";
  }
  if (state.mode === "real" && state.table_turn && !anim) {
    var cand = {};
    (state.candidates || []).forEach(function (c) { cand[c] = true; });
    var nmT = seatName(state.to_play);
    return '<div class="hand-wrap table-turn"><div class="entry-prompt">' + esc(nmT) + ' played\u2026</div>' +
      entryBoxHTML("Type their card, e.g. 7h, qs, 10d, td, then press Enter", "table:" + state.to_play + ":" + state.hand_id + ":" + (state.history || []).length) +
      '<div class="hand-title">or click it</div>' +
      pickGridHTML({
        cls: function (c) { return cand[c] ? "cand" : ""; },
        disabled: function (c) { return !cand[c] || busy; },
        title: function (c) { return cand[c] ? "They played " + cardLabel(c) : "already played or held by a known seat"; },
        attrs: function (c) { return 'data-table-card="' + c + '"'; }
      }) + '<p class="entry-undo"><button type="button" class="btn small" data-act="undo"' + (busy ? " disabled" : "") + ">Undo last card</button> " +
      '<span class="muted">entered the wrong card? take it back</span></p></div>';
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
  h += '</div><div class="caption" id="caption">' + (busy || anim ? "The table is playing..." : "&nbsp;") + "</div>";
  if (!anim) {
    h += entryBoxHTML("Or type the card you play, e.g. 9d, then Enter", "own:" + state.to_play + ":" + state.hand_id + ":" + (state.history || []).length + ":" + (state.current_trick || []).length);
  }
  return h + "</div>";
}

function entryBoxHTML(placeholder, key) {
  if (entryKey !== key) { entryKey = key; entryDraft = ""; entryMsg = ""; }
  return '<div class="entry"><input id="card-entry" class="entry-input" type="text" autocomplete="off" autocapitalize="off" spellcheck="false" ' +
    'maxlength="6" value="' + esc(entryDraft) + '" placeholder="' + esc(placeholder) + '" aria-label="Card played">' +
    '<button type="button" class="btn primary" data-act="entry-submit"' + (busy ? " disabled" : "") + '>Enter</button>' +
    '<span id="entry-msg" class="entry-msg" role="alert">' + esc(entryMsg) + "</span></div>";
}

function entryError(msg) {
  entryMsg = msg;
  var m = document.getElementById("entry-msg");
  if (m) m.textContent = msg; else { lastKey = null; render(); }
}

/* Validate typed text locally (no server round trip), then play it. */
function submitEntry() {
  if (!state || busy || isAnimating()) return;
  var raw = entryDraft;
  if (!String(raw).trim()) { entryError("Type a card first, like 7h or qs."); return; }
  var c = parseTypedCard(raw);
  if (!c) { entryError("\u201c" + raw.trim() + "\u201d is not a card. Use rank then suit: 7h, qs, 10d or td."); return; }
  if (state.table_turn) {
    if ((state.candidates || []).indexOf(c) < 0) {
      entryError(cardLabel(c) + " cannot be " + seatName(state.to_play) + "\u2019s card now (already played, in a known hand, or it breaks the lead rules).");
      return;
    }
    entryDraft = ""; entryMsg = "";
    act("/api/play", { card: c });
    return;
  }
  if (state.human_turn && state.hand) {
    if (state.hand.indexOf(c) < 0 || rendered.some(function (p) { return p.card === c; })) {
      entryError(cardLabel(c) + " is not in " + (state.to_play === 0 ? "your" : seatName(state.to_play) + "\u2019s") + " hand.");
      return;
    }
    if ((state.legal || []).indexOf(c) < 0) {
      entryError(cardLabel(c) + ": " + ((state.illegal_reasons && state.illegal_reasons[c]) || "not legal now") + ".");
      return;
    }
    entryDraft = ""; entryMsg = "";
    playCard(c);
  }
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
    '<button type="button" class="btn" data-act="undo"' + dis + ' title="Take back the last card entered">Undo last card</button>' +
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
  var h = '<section class="panel advisor"><h3>TallyHearts suggests</h3><div class="suggest-big">' + cardHTML(a.suggested, "static", 'tabindex="-1" disabled aria-label="Suggested card"') +
    '<span class="suggest-note">play ' + miniCard(a.suggested) + "<br>lowest expected points</span></div>";
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
  if (announceBanner) {
    h += '<div class="announce-banner" role="status"><b>' + esc(announceBanner.name) + "</b> (bot) plays <b>" + miniCard(announceBanner.card) +
      "</b> \u2014 play it at the table for them</div>";
  }
  var st = statusInfo();
  h += '<div class="status-strip ' + st.cls + '" role="status" aria-live="polite">' + st.html + "</div>";
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
  /* coaching panels only for the seat whose hand is on screen; never during a hidden hand or a hot-seat hand-over */
  if ((state.human_turn && !state.revealed) || (keyboardHumans() > 1 && (anim || optimistic)) || (!anim && !state.advice)) ps = null;
  var adv = ps ? advisorHTML(ps) : "";
  if (adv) h += '<aside class="side' + stale + '">' + adv + "</aside>";
  var rea = ps ? reasoningHTML(ps) : "";
  if (rea) h += '<div class="reasoning-row' + stale + '">' + rea + "</div>";
  h += "</div>";
  h += '<footer class="foot">hand #' + esc(state.hand_id) + " · seed " + esc(state.seed) + " · log " + esc(state.log_path) +
    ' · <span class="div' + (flashDiv ? " flash" : "") + '">divergences this hand: ' + (state.divergences || 0) + "</span></footer>";
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
  var pk = e.target.closest && e.target.closest("[data-pick]");
  if (pk && !pk.disabled) {
    var ps_ = parseInt(pk.getAttribute("data-hand-seat"), 10), pc = pk.getAttribute("data-pick");
    var arr = setupDraft.hands[ps_], ix = arr.indexOf(pc);
    if (ix >= 0) arr.splice(ix, 1);
    else if (arr.length < 13 && pickedBy(pc) < 0) arr.push(pc);
    redraw();
    return;
  }
  var tc = e.target.closest && e.target.closest("[data-table-card]");
  if (tc && !tc.disabled && !busy && !isAnimating() && state && state.table_turn) {
    act("/api/play", { card: tc.getAttribute("data-table-card") });
    return;
  }
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
    case "hands-go":
      var sd = readSeed();
      if (!sd.ok) { localError = "Seed must be a whole number."; lastKey = null; render(); break; }
      var perr = null;
      for (var q = 0; q < 4; q++) {
        var kk = setupDraft.kinds[q], kinf = kindInfo(kk), pv = String(setupDraft.params[q] || paramDefault(kk)).trim();
        if (kinf && kinf.param) {
          setupDraft.params[q] = pv;
          if (kk === "expert" && !(/^\d+$/.test(pv) && +pv >= 1 && +pv <= 200)) perr = "Expert id must be a whole number from 1 to 200 (seat " + SEAT_LABELS[q] + ").";
          else if (kk === "personality" && !/^\d+$/.test(pv)) perr = "Personality seed must be a non-negative whole number (seat " + SEAT_LABELS[q] + ").";
          else if (kk === "perilune" && pv !== "v5" && pv !== "v6.1") perr = "Perilune version must be v5 or v6.1.";
        }
      }
      if (perr) { localError = perr; lastKey = null; render(); break; }
      var body = {
        seats: [0, 1, 2, 3].map(function (s) {
          return {
            kind: seatKindString(s), name: setupDraft.names[s].trim() || SEAT_LABELS[s],
            advisor: setupDraft.kinds[s] === "human" ? !!setupDraft.coach[s] : false
          };
        }),
        seed: sd.v,
        advisor_on: setupDraft.advisor
      };
      if (isRealSetup()) {
        if (btn.getAttribute("data-act") === "deal") {
          setupDraft.step = "hands"; localError = null; lastKey = null; render(); break;
        }
        var herr = null, seen = {}, hands = {};
        for (var hs = 0; hs < 4; hs++) {
          if (setupDraft.kinds[hs] === "table") continue;
          var hh = setupDraft.hands[hs];
          if (hh.length !== 13) { herr = SEAT_LABELS[hs] + " has " + hh.length + " cards; each seat needs exactly 13."; break; }
          for (var hi = 0; hi < hh.length; hi++) {
            if (seen[hh[hi]]) { herr = "The card " + cardLabel(hh[hi]) + " is in two hands."; break; }
            seen[hh[hi]] = true;
          }
          if (herr) break;
          hands[String(hs)] = hh.slice();
        }
        if (herr) { localError = herr; lastKey = null; render(); break; }
        var ld = effectiveLeader();
        if (ld < 0) { localError = "Choose who leads the first trick (who holds 2\u2663)."; lastKey = null; render(); break; }
        body.hands = hands;
        body.leader = ld;
      }
      act("/api/new", body, function () { view = "auto"; setupDraft.step = "seats"; }, { fresh: true });
      break;
    case "preset": applyPreset(btn.getAttribute("data-preset")); localError = null; lastKey = null; render(); break;
    case "entry-submit": submitEntry(); break;
    case "hands-back": setupDraft.step = "seats"; localError = null; lastKey = null; render(); break;
    case "clear-hand":
      setupDraft.hands[parseInt(btn.getAttribute("data-seat"), 10)] = [];
      lastKey = null; render(); break;

    case "reveal": act("/api/reveal", {}); break;
    case "undo": act("/api/undo", {}, null, { instant: true }); break;
    case "resume": act("/api/resume", {}, function () { view = "auto"; }, { instant: true }); break;
    case "advisor": act("/api/advisor", { on: !state.advisor_on }); break;
  }
});

document.addEventListener("input", function (e) {
  var t = e.target;
  if (t && t.id === "card-entry") { entryDraft = t.value; if (entryMsg) { entryMsg = ""; var m = document.getElementById("entry-msg"); if (m) m.textContent = ""; } return; }
  var kind = t.getAttribute && t.getAttribute("data-setup");
  if (!kind) return;
  var s = parseInt(t.getAttribute("data-seat"), 10);
  if (kind === "name") setupDraft.names[s] = t.value;
  else if (kind === "leader") {
    setupDraft.leader = t.value === "" ? null : parseInt(t.value, 10);
    redraw();
    return;
  }
  else if (kind === "coach") setupDraft.coach[s] = t.checked;
  else if (kind === "kind") {
    setupDraft.preset = null;
    setupDraft.kinds[s] = t.value;
    setupDraft.params[s] = paramDefault(t.value);
    redraw(); /* the param input and the Deal button label depend on the kinds */
    return;
  }
  else if (kind === "param") setupDraft.params[s] = t.value;
  else if (kind === "seed") setupDraft.seed = t.value;
  else if (kind === "advisor") setupDraft.advisor = t.checked;
  lastKey = null; // draft changed; next render must not be skipped, but we do not re-render now (keeps focus)
});

document.addEventListener("keydown", function (e) {
  var t = e.target;
  if (t && t.id === "card-entry") {
    if (e.key === "Enter") { e.preventDefault(); entryDraft = t.value; submitEntry(); }
    else if (e.key === "Escape") { e.preventDefault(); entryDraft = ""; entryMsg = ""; t.value = ""; var m = document.getElementById("entry-msg"); if (m) m.textContent = ""; }
    return;
  }
  /* typing a card with nothing focused jumps into the card box */
  if (e.ctrlKey || e.metaKey || e.altKey || !e.key || e.key.length !== 1) return;
  if (t && /^(input|textarea|select)$/i.test(t.tagName || "")) return;
  var box = document.getElementById("card-entry");
  if (box) { box.focus(); }
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
  if (state.phase === "playing" && !state.human_turn && !state.table_turn) refresh();
}, 700);

function loadKinds() {
  if (MOCK) return Promise.resolve();
  return fetch("/api/kinds", { cache: "no-store" }).then(function (r) { return r.json(); }).then(function (j) {
    if (!j || !Array.isArray(j.kinds) || !j.kinds.length) return;
    var list = j.kinds.map(function (k) {
      return { kind: k.kind, label: k.kind === "search" ? "TallyHearts (main bot)" : (k.label || k.kind), param: k.param || null };
    });
    var have = {};
    list.forEach(function (k) { have[k.kind] = true; });
    if (!have.table) {
      var at = list.findIndex(function (k) { return k.kind === "human"; });
      list.splice(at + 1, 0, { kind: "table", label: "Human at the table", param: null });
    }
    kinds = list;
    lastKey = null;
    render();
  }).catch(function () { /* keep the built-in kinds */ });
}

loadKinds();
refresh();
