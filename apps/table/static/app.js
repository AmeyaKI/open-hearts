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
  state = s;
  render();
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

function act(path, body, after) {
  if (busy) return;
  busy = true;
  localError = null;
  dismissedError = null;
  render();
  apiPost(path, body).then(function (res) {
    busy = false;
    var j = res.json;
    if (j && j.phase) {
      state = j;
      if (res.ok && after) after();
    } else if (!res.ok) {
      localError = (j && j.error) || "The server rejected that action.";
    }
    lastKey = null;
    render();
  }).catch(function (e) {
    busy = false;
    localError = "Cannot reach the server (" + e.message + ").";
    lastKey = null;
    render();
  });
}

/* ---------- rendering ---------- */

function render() {
  var key = JSON.stringify([view, state, busy, dismissedError, localError]);
  if (key === lastKey) return;
  lastKey = key;
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
  var turn = state.phase === "playing" && state.to_play === s;
  var h = '<div class="seat seat-' + SEAT_POS[s] + (turn ? " turn" : "") + '">' +
    '<div class="seat-name">' + esc(info.name) + '<span class="badge kind-' + esc(info.kind) + '">' + esc(KIND_LABEL[info.kind] || info.kind) + "</span></div>" +
    '<div class="seat-pts"><b>' + state.scores[s] + "</b> pts</div>";
  if (turn) h += '<div class="seat-turn">' + (info.is_human ? "your move" : "thinking...") + "</div>";
  return h + "</div>";
}

function trickHTML() {
  var slots = {};
  (state.current_trick || []).forEach(function (p) { slots[p.seat] = p.card; });
  var h = '<div class="trick">';
  for (var s = 0; s < 4; s++) {
    h += '<div class="trick-slot slot-' + SEAT_POS[s] + '">' + (slots[s] ? cardHTML(slots[s], "small static", 'tabindex="-1"') : "") + "</div>";
  }
  return h + "</div>";
}

function lastTrickHTML() {
  var lt = state.last_trick;
  var h = '<div class="last-trick"><h3>Last trick</h3>';
  if (!lt) return h + '<p class="muted">None yet.</p></div>';
  h += '<div class="lt-plays">';
  lt.plays.forEach(function (p) {
    h += '<div class="lt-play' + (p.seat === lt.winner ? " won" : "") + '"><span class="who">' + esc(seatName(p.seat)) + "</span>" + miniCard(p.card) + "</div>";
  });
  h += "</div><p class=\"lt-result\"><b>" + esc(seatName(lt.winner)) + "</b> took it, " + lt.points + (lt.points === 1 ? " point" : " points") + "</p></div>";
  return h;
}

function handHTML() {
  if (state.phase === "over") return "";
  var name = state.to_play !== null && state.to_play !== undefined ? seatName(state.to_play) : "";
  if (state.human_turn && !state.revealed) {
    var n = Math.max(1, 13 - state.trick_number), backs = "";
    for (var i = 0; i < n; i++) backs += backCard();
    return '<div class="hand-wrap"><div class="hand hidden">' + backs +
      '</div><button type="button" class="btn primary reveal" data-act="reveal"' + (busy ? " disabled" : "") +
      ">Tap to reveal " + esc(name) + "'s hand</button></div>";
  }
  if (!state.human_turn || !state.hand) {
    var m = Math.max(1, 13 - state.trick_number), b2 = "";
    for (var j = 0; j < m; j++) b2 += backCard();
    return '<div class="hand-wrap"><div class="hand hidden dim">' + b2 + '</div><p class="hand-note">Waiting for ' + esc(name || "the bots") + "...</p></div>";
  }
  var legal = {};
  (state.legal || []).forEach(function (c) { legal[c] = true; });
  var suggested = state.advice ? state.advice.suggested : null;
  var h = '<div class="hand-wrap"><div class="hand-title">' + esc(name) + "'s hand</div><div class=\"hand\">";
  state.hand.slice().sort(cmpCards).forEach(function (c) {
    var cls = legal[c] ? "legal" : "illegal";
    if (c === suggested) cls += " suggested";
    var reason = (state.illegal_reasons && state.illegal_reasons[c]) || "not legal";
    var extra = legal[c]
      ? (busy ? 'disabled' : '') + ' title="Play ' + cardLabel(c) + '"'
      : 'aria-disabled="true" title="' + esc(reason) + '" data-reason="' + esc(reason) + '"';
    h += cardHTML(c, cls, extra);
  });
  h += '</div><div class="caption" id="caption">' + (busy ? "Bots are thinking..." : "&nbsp;") + "</div></div>";
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

function advisorHTML() {
  var a = state.advice;
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
  var light = Math.round(100 - p * 58);
  return "hsl(205, 75%, " + light + "%)";
}

function reasoningHTML() {
  var a = state.advice;
  if (!a) return "";
  var b = a.belief;
  var h = '<section class="panel reasoning"><h3>Reasoning</h3>';
  if (b && b.cards) {
    var cards = Object.keys(b.cards).sort(cmpCards);
    h += '<div class="heat-scroll"><table class="heat"><thead><tr><th class="rowhead">Belief</th>';
    var prev = null;
    cards.forEach(function (c) {
      var first = prev !== null && prev !== c[1];
      h += '<th class="' + (first ? "sep " : "") + (isRed(c[1]) ? "red" : "black") + '">' + cardLabel(c) + "</th>";
      prev = c[1];
    });
    h += "</tr></thead><tbody>";
    b.opponents.forEach(function (o, i) {
      var voids = (b.voids && b.voids[i]) || [];
      var vtxt = voids.length ? voids.map(function (v) { return '<span class="' + (isRed(v) ? "red" : "black") + '">' + SUITS[v] + "</span>"; }).join("") : "";
      h += '<tr><th class="rowhead">' + esc(o.name) + (vtxt ? ' <span class="voids" title="known void in ' +
        esc(voids.map(function (v) { return SUIT_NAME[v]; }).join(", ")) + '">void ' + vtxt + "</span>" : "") + "</th>";
      prev = null;
      cards.forEach(function (c) {
        var p = b.cards[c][i] || 0;
        var first = prev !== null && prev !== c[1];
        h += '<td class="' + (first ? "sep " : "") + (p >= 0.7 ? "hi" : "") + '" style="background:' + (p > 0 ? heatColor(p) : "#fff") +
          '" title="' + esc(o.name) + " holds " + cardLabel(c) + ": " + p.toFixed(2) + '">' + (p >= 0.5 ? p.toFixed(2).replace(/^0/, "") : "") + "</td>";
        prev = c[1];
      });
      h += "</tr>";
    });
    h += "</tbody></table></div>";
  }
  (a.sentences || []).forEach(function (s) { h += '<p class="sentence">' + esc(s) + "</p>"; });
  return h + "</section>";
}

function gameHTML() {
  var h = '<header class="topbar"><h1>The Table</h1>' + (MOCK ? '<span class="mock">mock data</span>' : "") + controlsHTML() + "</header>";
  h += '<div class="layout"><main class="felt-col"><div class="felt">';
  h += '<div class="statusbar"><span>Trick <b>' + Math.min(13, state.trick_number + (state.phase === "over" ? 0 : 1)) + "</b> of 13</span>" +
    '<span class="hb ' + (state.hearts_broken ? "broken" : "") + '">♥ ' + (state.hearts_broken ? "hearts broken" : "hearts not broken") + "</span></div>";
  h += '<div class="table-grid">' + seatBoxHTML(2) + seatBoxHTML(1) + '<div class="center">' + trickHTML() + "</div>" + seatBoxHTML(3) + seatBoxHTML(0) + lastTrickHTML() + "</div>";
  if (state.phase === "over") h += overHTML();
  h += handHTML() + "</div></main>";
  var side = advisorHTML() + reasoningHTML();
  if (side) h += '<aside class="side">' + side + "</aside>";
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
  if (cardEl && !busy && state && state.human_turn) {
    act("/api/play", { card: cardEl.getAttribute("data-card") });
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
      }, function () { view = "auto"; });
      break;
    case "reveal": act("/api/reveal", {}); break;
    case "undo": act("/api/undo", {}); break;
    case "resume": act("/api/resume", {}, function () { view = "auto"; }); break;
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
