/* Mixture-of-Experts chapter demos.
   Toy-sized but honest: router scores come from a hand-authored affinity
   table, the softmax, top-2 pick, blend proportions, and compute counts
   are all computed for real. Deterministic per token. */

import { el, svgEl, softmax, loop, spring, reducedMotion, clamp, palette } from "./lib.js";

/* ---------- The panel ---------- */

const EXPERTS = [
  { name: "punctuation & lists", flavor: "commas, bullets, !!!, tidy numbering" },
  { name: "numbers & units", flavor: "42, 3.7 km, percentages, dates" },
  { name: "French-ish", flavor: "bonjour, accents, le and la" },
  { name: "code-ish", flavor: "function, brackets, camelCase" },
  { name: "storytelling", flavor: "once, suddenly, plot glue" },
  { name: "science words", flavor: "photosynthesis, enzyme, quark" },
  { name: "names & places", flavor: "Paris, Ada, proper nouns" },
  { name: "glue words", flavor: "the, of, and, was" },
];

/* Hand-authored affinity logits, one row per tray token, one column per expert */
const AFFINITY = {
  "bonjour":        [-1.0, -2.0,  3.2, -1.5,  0.6, -1.2,  1.1, -0.8],
  "42":             [ 0.4,  3.4, -2.0,  1.2, -1.0,  0.2, -1.5, -1.8],
  "function":       [-0.5, -0.4, -1.8,  3.3, -0.6,  0.3, -1.2,  1.0],
  "once":           [-1.2, -0.8, -1.0, -0.9,  3.0, -1.4, -0.5,  1.4],
  "photosynthesis": [-1.5,  0.5, -1.0, -0.6, -0.8,  3.4, -0.9, -1.6],
  "Paris":          [-1.3, -0.9,  1.5, -1.4,  0.4, -1.0,  3.2, -1.2],
  "the":            [-0.6, -1.5, -0.9, -0.4,  0.8, -1.3, -1.0,  3.1],
  "!!!":            [ 3.5,  0.3, -1.4,  0.2,  1.0, -2.0, -1.6, -0.7],
};

/* ================= Hero demo: the router ================= */

const host = document.getElementById("router-demo");

const trayRow = el("div", { class: "token-row", role: "group", "aria-label": "Token tray" });
const arena = el("div", { class: "moe-arena" });
const overlay = svgEl("svg", { class: "moe-overlay", "aria-hidden": "true" });
const routerNode = el("div", { class: "moe-router mono" }, "router");
const outlet = el("div", { class: "moe-outlet" },
  el("span", { class: "mono moe-outlet-label" }, "blend"));
const hub = el("div", { class: "moe-hub" }, routerNode, outlet);
const expertsRow = el("div", { class: "moe-experts", role: "group", "aria-label": "Eight expert stations" });
const chip = el("div", { class: "token moe-chip", "aria-hidden": "true" });
const whoami = el("p", { class: "hint moe-whoami" },
  "hover, tap, or tab to a station to read its specialty");
const readout = el("p", { class: "moe-readout mono", "aria-live": "polite" },
  "pick a token from the tray");

const stations = EXPERTS.map((ex, i) => {
  const fill = el("div", { class: "moe-gauge-fill" });
  const pct = el("span", { class: "moe-pct mono" }, " ");
  const node = el("button", {
    class: "moe-expert",
    type: "button",
    "aria-label": `Expert ${i + 1}, ${ex.name}. Likes: ${ex.flavor}`,
  },
    el("span", { class: "moe-exnum mono" }, `E${i + 1}`),
    el("span", { class: "moe-exname" }, ex.name),
    el("div", { class: "moe-gauge" }, fill),
    pct,
  );
  const reveal = () => {
    whoami.textContent = `E${i + 1} · ${ex.name}: likes ${ex.flavor}`;
  };
  node.addEventListener("mouseenter", reveal);
  node.addEventListener("focus", reveal);
  node.addEventListener("click", reveal);
  return { node, fill, pct, gauge: spring(0, 9) };
});

expertsRow.append(...stations.map((s) => s.node));
arena.append(overlay, hub, expertsRow, chip);
host.append(
  el("p", { class: "moe-tray-label" }, "the tray"),
  trayRow, arena, whoami, readout,
);

/* Tray */
let busy = false;
for (const word of Object.keys(AFFINITY)) {
  trayRow.append(el("button", { class: "token input", type: "button", onclick: (e) => send(word, e.currentTarget) }, word));
}

/* Chip motion: two springs plus settle promises resolved in the rAF loop */
const cx = spring(0, 8);
const cy = spring(0, 8);
let settlers = [];

function centerOf(node) {
  const a = arena.getBoundingClientRect();
  const r = node.getBoundingClientRect();
  return { x: r.left - a.left + r.width / 2, y: r.top - a.top + r.height / 2 };
}

function moveTo(node) {
  const p = centerOf(node);
  cx.target = p.x;
  cy.target = p.y;
  return new Promise((res) => settlers.push(res));
}

function teleportTo(node) {
  const p = centerOf(node);
  cx.value = cx.target = p.x;
  cy.value = cy.target = p.y;
}

const wait = (ms) => (reducedMotion ? Promise.resolve() : new Promise((r) => setTimeout(r, ms)));

function drawPath(fromNode, toNode, color, width) {
  const p1 = centerOf(fromNode);
  const p2 = centerOf(toNode);
  const mx = (p1.x + p2.x) / 2;
  const my = (p1.y + p2.y) / 2 - 22;
  overlay.append(svgEl("path", {
    d: `M ${p1.x} ${p1.y} Q ${mx} ${my} ${p2.x} ${p2.y}`,
    fill: "none",
    stroke: color,
    "stroke-width": clamp(width, 1.5, 16),
    "stroke-linecap": "round",
    opacity: 0.55,
  }));
}

function pulse(node) {
  node.classList.add("visited");
  setTimeout(() => node.classList.remove("visited"), reducedMotion ? 0 : 420);
}

function clearRun() {
  overlay.replaceChildren();
  stations.forEach((s) => s.node.classList.remove("chosen", "visited"));
  routerNode.classList.remove("thinking");
  outlet.classList.remove("filled");
}

async function send(word, trayBtn) {
  if (busy) return;
  busy = true;
  clearRun();

  /* Real math: softmax over the affinity row, then top-2, then renormalize */
  const scores = softmax(AFFINITY[word]);
  const order = scores.map((w, i) => ({ w, i })).sort((a, b) => b.w - a.w);
  const [first, second] = order;
  const shareA = first.w / (first.w + second.w);
  const shareB = second.w / (first.w + second.w);

  chip.textContent = word;
  chip.classList.add("input");
  chip.classList.remove("output");
  chip.style.opacity = "1";
  teleportTo(trayBtn);
  readout.textContent = `routing “${word}”`;

  await moveTo(routerNode);
  routerNode.classList.add("thinking");
  stations.forEach((s, i) => {
    s.gauge.target = scores[i];
    s.pct.textContent = `${Math.round(scores[i] * 100)}%`;
  });
  stations[first.i].node.classList.add("chosen");
  stations[second.i].node.classList.add("chosen");
  await wait(700);

  drawPath(routerNode, stations[first.i].node, palette.coral, 2 + first.w * 12);
  drawPath(routerNode, stations[second.i].node, palette.coral, 2 + second.w * 12);
  await moveTo(stations[first.i].node);
  pulse(stations[first.i].node);
  await wait(300);
  await moveTo(stations[second.i].node);
  pulse(stations[second.i].node);
  await wait(300);

  drawPath(stations[second.i].node, outlet, palette.mint, 3.5);
  chip.classList.remove("input");
  chip.classList.add("output");
  await moveTo(outlet);
  outlet.classList.add("filled");
  routerNode.classList.remove("thinking");
  readout.textContent =
    `“${word}” = ${Math.round(shareA * 100)}% E${first.i + 1} ${EXPERTS[first.i].name}` +
    ` + ${Math.round(shareB * 100)}% E${second.i + 1} ${EXPERTS[second.i].name}`;
  busy = false;
}

/* Shared rAF loop for chip + gauges */
loop((dt) => {
  cx.step(dt);
  cy.step(dt);
  chip.style.transform = `translate(${cx.value}px, ${cy.value}px) translate(-50%, -50%)`;
  stations.forEach((s) => {
    s.gauge.step(dt);
    s.fill.style.width = `${s.gauge.value * 100}%`;
  });
  if (settlers.length &&
      Math.abs(cx.value - cx.target) < 0.6 &&
      Math.abs(cy.value - cy.target) < 0.6) {
    settlers.forEach((r) => r());
    settlers = [];
  }
}, host);

document.getElementById("router-reset").addEventListener("click", () => {
  if (busy) return;
  clearRun();
  chip.style.opacity = "0";
  stations.forEach((s) => { s.gauge.target = 0; s.pct.textContent = " "; });
  readout.textContent = "pick a token from the tray";
  whoami.textContent = "hover, tap, or tab to a station to read its specialty";
});

window.addEventListener("resize", () => {
  if (!busy) { overlay.replaceChildren(); chip.style.opacity = "0"; }
});

/* ================= The bill: dense vs sparse ================= */

const billHost = document.getElementById("bill-demo");
const N_TOKENS = 20;
const GAP = 0.16;   /* seconds between token launches */
const DUR = 1.5;    /* seconds for one token to cross a lane */
let k = 2;

/* Deterministic per-token expert picks for the sparse lane: a seeded
   permutation of the 8 experts; the first k fire. Varies token to token,
   never randomly between page loads. */
function chosenExperts(tokenIdx, kk) {
  const start = (tokenIdx * 3) % 8;
  const step = [1, 3, 5, 7][tokenIdx % 4];
  return Array.from({ length: kk }, (_, j) => (start + j * step) % 8);
}

function makeLane(title) {
  const cells = EXPERTS.map(() => el("div", { class: "bill-cell" }));
  const bank = el("div", { class: "bill-bank", "aria-hidden": "true" }, cells);
  const dots = Array.from({ length: N_TOKENS }, () =>
    el("div", { class: "bill-dot", style: "opacity:0;" }));
  const strip = el("div", { class: "bill-strip" }, bank, ...dots);
  const count = el("span", { class: "bill-count mono" }, "0");
  const head = el("div", { class: "bill-head" },
    el("span", { class: "bill-title" }, title),
    el("span", { class: "bill-units mono" }, count, " compute units"));
  return { root: el("div", { class: "bill-lane" }, head, strip), cells, strip, dots, count };
}

const denseLane = makeLane("dense: all 8 fire");
const sparseLane = makeLane("sparse: top-k of 8");
const summary = el("p", { class: "hint", "aria-live": "polite" },
  "press run to stream 20 tokens through both models");
billHost.append(el("div", { class: "bill-lanes" }, denseLane.root, sparseLane.root), summary);

let billT = -1;          /* -1 = idle */
let counted = new Array(N_TOKENS).fill(false);
let processed = 0;

function updateCounts() {
  denseLane.count.textContent = String(processed * 8);
  sparseLane.count.textContent = String(processed * k);
}

function finishSummary() {
  const pctWork = Math.round((k / 8) * 100);
  summary.textContent =
    `dense: ${N_TOKENS * 8} units. sparse: ${N_TOKENS * k} units, ${pctWork}% of the dense bill. ` +
    (k === 8
      ? "pick all 8 and the savings vanish: sparse only pays off because most experts sit out."
      : "same 8 experts on the shelf either way; each token just used fewer of them.");
}

const progress = (i) => (billT - i * GAP) / DUR;

function runBill() {
  counted = new Array(N_TOKENS).fill(false);
  processed = 0;
  if (reducedMotion) {
    processed = N_TOKENS;
    billT = -1;
    updateCounts();
    finishSummary();
    return;
  }
  billT = 0;
  updateCounts();
  summary.textContent = "streaming 20 tokens";
}

loop((dt) => {
  if (billT < 0) return;
  billT += dt;
  let allDone = true;
  const denseZone = new Set();
  const sparseZone = new Set();

  for (let i = 0; i < N_TOKENS; i++) {
    const p = progress(i);
    if (p < 1) allDone = false;
    if (p >= 0.55 && !counted[i]) { counted[i] = true; processed++; updateCounts(); }
    const inZone = p > 0.35 && p < 0.65;
    if (inZone) {
      for (let e = 0; e < 8; e++) denseZone.add(e);
      for (const e of chosenExperts(i, k)) sparseZone.add(e);
    }
    for (const lane of [denseLane, sparseLane]) {
      const dot = lane.dots[i];
      if (p < 0 || p > 1) { dot.style.opacity = "0"; continue; }
      const w = lane.strip.clientWidth - 12;
      dot.style.opacity = "1";
      dot.style.transform = `translateX(${p * w}px)`;
    }
  }
  denseLane.cells.forEach((c, e) => c.classList.toggle("on", denseZone.has(e)));
  sparseLane.cells.forEach((c, e) => c.classList.toggle("on", sparseZone.has(e)));

  if (allDone) {
    billT = -1;
    denseLane.cells.forEach((c) => c.classList.remove("on"));
    sparseLane.cells.forEach((c) => c.classList.remove("on"));
    finishSummary();
  }
}, billHost);

document.getElementById("bill-run").addEventListener("click", runBill);

/* k switcher recomputes the counter live */
const kSeg = document.getElementById("k-seg");
for (const kk of [1, 2, 4, 8]) {
  kSeg.append(el("button", {
    type: "button",
    "aria-pressed": String(kk === k),
    onclick: (e) => {
      k = kk;
      [...kSeg.children].forEach((b) => b.setAttribute("aria-pressed", String(b === e.currentTarget)));
      updateCounts();
      if (billT < 0 && processed > 0) finishSummary();
    },
  }, `top-${kk}`));
}
