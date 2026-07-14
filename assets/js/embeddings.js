/* Embeddings chapter demos.
   The 2D coordinates are hand-authored for legibility (the page says so),
   but every distance, neighbour ranking, and vector sum is computed for real. */

import { el, svgEl, loop, clamp, reducedMotion } from "./lib.js";

/* ---------- The hand-drawn map ---------- */
/* [word, x, y] in map units (0..100). Placed so that proximity is truthful
   and the classic analogies land exactly:
   king - man + woman = (84,60) = queen
   paris - france + japan = (88,88) = tokyo
   puppy - dog + cat = (20,19) = kitten */

const WORDS = [
  // animals
  ["dog", 14, 22], ["puppy", 14, 17], ["cat", 20, 24], ["kitten", 20, 19],
  ["wolf", 10, 26], ["fox", 12, 29], ["bear", 6, 22], ["lion", 7, 17],
  ["tiger", 9, 14], ["horse", 17, 31], ["cow", 23, 31], ["sheep", 26, 28],
  ["rabbit", 22, 12], ["mouse", 25, 14], ["bird", 28, 17], ["fish", 29, 23],
  // foods
  ["bread", 42, 14], ["butter", 43, 10], ["cheese", 47, 17], ["milk", 40, 8],
  ["apple", 52, 9], ["banana", 55, 12],
  ["soup", 39, 18], ["pizza", 44, 21], ["pasta", 48, 24], ["rice", 50, 20],
  ["cake", 53, 17], ["coffee", 56, 21], ["tea", 56, 26], ["wine", 59, 18],
  // feelings
  ["happy", 78, 12], ["joy", 81, 10], ["calm", 74, 9], ["proud", 76, 15],
  ["love", 81, 15], ["hate", 84, 18], ["angry", 88, 22], ["fear", 89, 26],
  ["sad", 71, 21], ["lonely", 70, 25], ["bored", 70, 17], ["shy", 69, 30],
  // vehicles
  ["bicycle", 8, 53], ["scooter", 7, 57], ["car", 12, 59], ["taxi", 14, 56],
  ["truck", 10, 62], ["bus", 15, 64], ["train", 19, 67], ["boat", 22, 70],
  ["ship", 24, 72], ["plane", 25, 58], ["rocket", 27, 54],
  // motion
  ["walk", 42, 52], ["run", 46, 50], ["sprint", 48, 47], ["jump", 52, 54],
  ["climb", 41, 46], ["crawl", 39, 56], ["dance", 50, 58], ["swim", 56, 60],
  ["fly", 54, 44],
  // weather
  ["ice", 34, 74], ["cold", 36, 78], ["cool", 41, 79], ["warm", 50, 80],
  ["hot", 56, 81], ["snow", 37, 84], ["rain", 42, 89], ["cloud", 44, 83],
  ["wind", 47, 86], ["storm", 49, 91], ["sun", 55, 86],
  // people & royalty
  ["boy", 72, 50], ["girl", 72, 56], ["man", 76, 52], ["woman", 76, 58],
  ["king", 84, 54], ["queen", 84, 60], ["prince", 89, 51], ["princess", 90, 57],
  ["knight", 80, 48],
  // places
  ["england", 69, 90], ["london", 69, 86], ["france", 76, 92], ["paris", 76, 88],
  ["italy", 83, 95], ["rome", 83, 91], ["japan", 88, 92], ["tokyo", 88, 88],
];

const CLUSTERS = [
  ["animals", 16, 36], ["foods", 48, 3], ["feelings", 80, 3],
  ["vehicles", 15, 47], ["motion", 47, 40], ["weather", 39, 96],
  ["people & royalty", 81, 43], ["places", 74, 99],
];

const COORD = new Map(WORDS.map(([w, x, y]) => [w, { x, y }]));

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

/* Real euclidean nearest neighbours, optionally excluding some words */
function nearest(point, k, exclude = []) {
  return WORDS
    .filter(([w]) => !exclude.includes(w))
    .map(([w, x, y]) => ({ w, d: dist(point, { x, y }) }))
    .sort((a, b) => a.d - b.d)
    .slice(0, k);
}

/* ---------- Map renderer (shared by both demos) ---------- */

const W = 760, H = 620, PAD = 36;
const px = (x) => PAD + (x / 100) * (W - 2 * PAD);
const py = (y) => PAD + (y / 100) * (H - 2 * PAD);
const P = (name) => ({ x: px(COORD.get(name).x), y: py(COORD.get(name).y) });

function makeMap(host, { interactive = false, onPick = null, label = "" } = {}) {
  const svg = svgEl("svg", {
    class: "map-svg",
    viewBox: `0 0 ${W} ${H}`,
    role: interactive ? "group" : "img",
    "aria-label": label,
  });

  // faint dot grid, so it reads as a map with coordinates
  const grid = svgEl("g", { "aria-hidden": "true" });
  for (let gx = 0; gx <= 100; gx += 10) {
    for (let gy = 0; gy <= 100; gy += 10) {
      grid.append(svgEl("circle", {
        cx: px(gx), cy: py(gy), r: 1.1,
        fill: "var(--line)",
      }));
    }
  }
  svg.append(grid);

  for (const [name, cx, cy] of CLUSTERS) {
    svg.append(svgEl("text", {
      class: "cluster-label", x: px(cx), y: py(cy),
      "text-anchor": "middle", "aria-hidden": "true",
    }, name));
  }

  const lineLayer = svgEl("g", { "aria-hidden": "true" });
  const wordLayer = svgEl("g");
  const fxLayer = svgEl("g", { "aria-hidden": "true" });
  svg.append(lineLayer, wordLayer, fxLayer);

  const els = new Map();
  for (const [w, x, y] of WORDS) {
    const g = svgEl("g", {
      class: "word" + (interactive ? " tappable" : ""),
      transform: `translate(${px(x)}, ${py(y)})`,
    });
    if (interactive) {
      g.setAttribute("tabindex", "0");
      g.setAttribute("role", "button");
      g.setAttribute("aria-label", `${w}: show nearest neighbours`);
    }
    g.append(
      svgEl("circle", { class: "halo", r: 10, fill: "transparent", stroke: "none" }),
      svgEl("circle", { class: "dot", cx: 0, cy: 0 }),
      svgEl("text", { class: "wlabel", x: 0, y: -9 }, w),
    );
    if (interactive && onPick) {
      g.addEventListener("click", () => onPick(w));
      g.addEventListener("mouseenter", () => onPick(w));
      g.addEventListener("focus", () => onPick(w));
      g.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onPick(w); }
      });
    }
    els.set(w, g);
    wordLayer.append(g);
  }

  host.replaceChildren(svg);
  return {
    svg, lineLayer, fxLayer, els,
    clearMarks() {
      svg.classList.remove("has-sel");
      lineLayer.replaceChildren();
      fxLayer.replaceChildren();
      for (const g of els.values()) g.classList.remove("sel", "nbr", "op", "ans");
    },
  };
}

/* ================= Demo 1: the map of meaning ================= */

const heroHost = document.getElementById("hero-map");
const panel = document.getElementById("nbr-panel");
const K = 4;

const hero = makeMap(heroHost, {
  interactive: true,
  onPick: selectWord,
  label: "A hand-drawn 2D map of 90 words in semantic clusters",
});

let selected = null;

function selectWord(w) {
  if (w === selected) return;
  selected = w;
  hero.clearMarks();
  hero.svg.classList.add("has-sel");

  const me = COORD.get(w);
  const nbrs = nearest(me, K, [w]);

  hero.els.get(w).classList.add("sel");
  const a = P(w);
  nbrs.forEach((n, rank) => {
    hero.els.get(n.w).classList.add("nbr");
    const b = P(n.w);
    const len = Math.hypot(b.x - a.x, b.y - a.y);
    const line = svgEl("line", {
      class: "nbr-line",
      x1: a.x, y1: a.y, x2: b.x, y2: b.y,
      "stroke-width": [3.2, 2.4, 1.8, 1.4][rank] ?? 1.2,
      opacity: 0.9 - rank * 0.15,
      "stroke-dasharray": len,
      "stroke-dashoffset": len,
    });
    hero.lineLayer.append(line);
    // draw-in: transition dashoffset to 0 (collapses to instant under reduced motion)
    requestAnimationFrame(() => requestAnimationFrame(() => {
      line.setAttribute("stroke-dashoffset", 0);
    }));
  });

  panel.replaceChildren(
    el("div", { class: "token-row" },
      el("span", { class: "token active" }, w),
      el("span", { class: "hint", style: "margin-inline:0.25rem;" }, "nearest neighbours, closest first"),
      ...nbrs.map((n) =>
        el("span", { class: "token weight", "aria-label": `${n.w}, distance ${n.d.toFixed(1)} map units` },
          n.w, el("small", { "aria-hidden": "true" }, n.d.toFixed(1)))
      ),
    ),
  );
}

/* Preset finder buttons */
const PRESETS = ["cat", "king", "hot", "train", "lonely"];
const presetRow = document.getElementById("preset-row");
presetRow.append(...PRESETS.map((w) =>
  el("button", { class: "btn small", onclick: () => {
    selectWord(w);
    if (!reducedMotion) hero.els.get(w).scrollIntoView({ block: "nearest", inline: "nearest", behavior: "smooth" });
  } }, `find ${w}`)
));

selectWord("cat"); // interesting before anyone touches it

/* ================= Demo 2: word arithmetic ================= */

const EQS = [
  { key: "king", a: "king", minus: "man", plus: "woman" },
  { key: "paris", a: "paris", minus: "france", plus: "japan" },
  { key: "puppy", a: "puppy", minus: "dog", plus: "cat" },
];
let eqIndex = 0;

const mathHost = document.getElementById("math-map");
const mathMap = makeMap(mathHost, {
  interactive: false,
  label: "The same word map, used as a canvas for word arithmetic",
});

const eqChips = document.getElementById("eq-chips");
const eqResult = document.getElementById("eq-result");
const runBtn = document.getElementById("eq-run");
const eqSeg = document.getElementById("eq-seg");

let running = null; // handle from loop()

function stopRun() {
  if (running) { running.stop(); running = null; }
  runBtn.disabled = false;
}

function renderChips(answer = null) {
  const eq = EQS[eqIndex];
  eqChips.replaceChildren(
    el("span", { class: "token input" }, eq.a),
    el("span", { class: "sign", "aria-hidden": "true" }, "−"),
    el("span", { class: "token input" }, eq.minus),
    el("span", { class: "sign", "aria-hidden": "true" }, "+"),
    el("span", { class: "token input" }, eq.plus),
    el("span", { class: "sign", "aria-hidden": "true" }, "≈"),
    el("span", { class: answer ? "token output" : "token" },
      answer ?? "?"),
  );
  eqChips.setAttribute("aria-label",
    `${eq.a} minus ${eq.minus} plus ${eq.plus} is approximately ${answer ?? "unknown, press run"}`);
}

function markOperands() {
  const eq = EQS[eqIndex];
  for (const w of [eq.a, eq.minus, eq.plus]) mathMap.els.get(w).classList.add("op");
  mathMap.svg.classList.add("has-sel");
}

function resetEq() {
  stopRun();
  mathMap.clearMarks();
  markOperands();
  renderChips();
  eqResult.textContent = "press run and watch the arrows.";
}

const ease = (t) => 1 - Math.pow(1 - t, 3);

function arrowEl(color, dashed) {
  return svgEl("line", {
    class: "vec-line",
    stroke: color,
    "stroke-width": 3,
    "stroke-dasharray": dashed ? "7 6" : "none",
    "marker-end": dashed ? "url(#arrow-dash)" : "url(#arrow-solid)",
    opacity: 0,
  });
}

/* Arrowhead markers, added once to the math map */
mathMap.svg.prepend(svgEl("defs", {},
  svgEl("marker", { id: "arrow-solid", viewBox: "0 0 10 10", refX: 8, refY: 5,
    markerWidth: 5.5, markerHeight: 5.5, orient: "auto-start-reverse" },
    svgEl("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "var(--cobalt)" })),
  svgEl("marker", { id: "arrow-dash", viewBox: "0 0 10 10", refX: 8, refY: 5,
    markerWidth: 5.5, markerHeight: 5.5, orient: "auto-start-reverse" },
    svgEl("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "var(--cobalt)" })),
));

function setArrow(line, from, to, t) {
  if (t <= 0.02) { line.setAttribute("opacity", 0); return; }
  line.setAttribute("opacity", 1);
  line.setAttribute("x1", from.x);
  line.setAttribute("y1", from.y);
  line.setAttribute("x2", from.x + (to.x - from.x) * t);
  line.setAttribute("y2", from.y + (to.y - from.y) * t);
}

function runEq() {
  stopRun();
  mathMap.clearMarks();
  markOperands();
  renderChips();

  const eq = EQS[eqIndex];
  const A = P(eq.a), B = P(eq.minus), C = P(eq.plus);
  // The genuine sum in map units, then converted to pixels for drawing
  const a = COORD.get(eq.a), b = COORD.get(eq.minus), c = COORD.get(eq.plus);
  const target = { x: a.x - b.x + c.x, y: a.y - b.y + c.y };
  const T = { x: px(target.x), y: py(target.y) };
  const [best] = nearest(target, 1, [eq.a, eq.minus, eq.plus]);

  const arrow1 = arrowEl("var(--cobalt)", false); // measured difference: minus -> plus
  const arrow2 = arrowEl("var(--cobalt)", true);  // same vector, carried to the start word
  const ring = svgEl("circle", {
    cx: T.x, cy: T.y, r: 0, fill: "none",
    stroke: "var(--mint)", "stroke-width": 3, opacity: 0,
  });
  const cross = svgEl("path", {
    d: `M ${T.x - 5} ${T.y - 5} L ${T.x + 5} ${T.y + 5} M ${T.x - 5} ${T.y + 5} L ${T.x + 5} ${T.y - 5}`,
    stroke: "var(--mint)", "stroke-width": 2.5, opacity: 0, "stroke-linecap": "round",
  });
  mathMap.fxLayer.append(arrow1, arrow2, cross, ring);

  const finish = () => {
    mathMap.els.get(best.w).classList.add("ans");
    renderChips(best.w);
    eqResult.textContent =
      `${eq.a} − ${eq.minus} + ${eq.plus} lands ${best.d.toFixed(1)} map units from ${best.w}. Nearest word wins.`;
    runBtn.disabled = false;
  };

  if (reducedMotion) {
    setArrow(arrow1, B, C, 1);
    setArrow(arrow2, A, T, 1);
    cross.setAttribute("opacity", 1);
    ring.setAttribute("opacity", 0.9);
    ring.setAttribute("r", 11);
    finish();
    return;
  }

  runBtn.disabled = true;
  eqResult.textContent = `measuring the ${eq.minus} → ${eq.plus} arrow…`;
  let t = 0;
  running = loop((dt) => {
    t += dt;
    const p1 = ease(clamp(t / 0.8, 0, 1));                // draw minus -> plus
    const p2 = ease(clamp((t - 0.95) / 0.8, 0, 1));       // carry it to the start word
    const p3 = ease(clamp((t - 1.9) / 0.5, 0, 1));        // snap to the answer
    setArrow(arrow1, B, C, p1);
    setArrow(arrow2, A, T, p2);
    if (p2 >= 1 && t > 1.85) {
      cross.setAttribute("opacity", 1);
      ring.setAttribute("opacity", p3 * 0.9);
      ring.setAttribute("r", 26 - p3 * 15);
    }
    if (t >= 2.5) { stopRun(); finish(); }
  }, mathHost);
}

/* Equation picker */
function buildSeg() {
  eqSeg.replaceChildren(...EQS.map((eq, i) =>
    el("button", {
      "aria-pressed": String(i === eqIndex),
      onclick: () => { eqIndex = i; buildSeg(); resetEq(); },
    }, eq.key)
  ));
}

runBtn.addEventListener("click", runEq);
buildSeg();
resetEq();
