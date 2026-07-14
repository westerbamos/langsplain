/* Layers chapter demos.
   Demo 1: one honest neuron. Fixed hand-chosen weights, live weighted sum,
   real ReLU at the valve. Everything on screen is computed from the sliders.
   Demo 2: hand-authored annotations accumulating with depth (illustrative,
   in the spirit of probing research; labelled as such on the page). */

import { el, svgEl, loop, spring, clamp, palette, reducedMotion } from "./lib.js";

/* =====================================================================
   Demo 1 · One neuron as a machine
   ===================================================================== */

const WEIGHTS = [1.6, 2.6, -2.2];
const BIAS = -1.2;
const DEFAULTS = [1, 1, 0];

const INPUTS = [
  { label: "sentence started with a question word (who, what, why)", short: "question-word start" },
  { label: "this word is a question mark", short: "question mark" },
  { label: "we're mid-quotation (someone else's words)", short: "inside quotes" },
];

let x = [...DEFAULTS];

const sums = () => BIAS + WEIGHTS.reduce((s, w, i) => s + w * x[i], 0);
const relu = (s) => Math.max(0, s);

/* Sum range for the tank gauge (bias included) */
const SUM_MIN = BIAS + WEIGHTS.filter((w) => w < 0).reduce((a, b) => a + b, 0); // -3.4
const SUM_MAX = BIAS + WEIGHTS.filter((w) => w > 0).reduce((a, b) => a + b, 0); // +3.0
const sumFrac = (s) => (s - SUM_MIN) / (SUM_MAX - SUM_MIN);

/* ---------- Geometry ---------- */

const VB_W = 720, VB_H = 380;
const PORT_X = 90;
const PORT_Y = [80, 190, 300];
const TANK = { x: 330, y: 90, w: 90, h: 210 };
const INLET_Y = [140, 195, 250];
const VALVE = { x: 500, y: 195, r: 22 };
const OUT_X = 664, OUT_Y = 195;

const fmt = (v, signed = false) =>
  (signed && v >= 0 ? "+" : "") + v.toFixed(2);

const host = document.getElementById("neuron-demo");

const svg = svgEl("svg", {
  class: "neuron-svg",
  viewBox: `0 0 ${VB_W} ${VB_H}`,
  role: "img",
  "aria-label": "A neuron machine: three input pipes with fixed weights feed a tank; a valve releases the output only when the weighted sum is above zero.",
});

/* Input pipes: outer stroke = weight (coral, thickness = size, dashed = negative),
   inner stroke = the signal flowing through (cobalt, opacity = input value). */
const pipePath = (i) =>
  `M ${PORT_X} ${PORT_Y[i]} C 200 ${PORT_Y[i]}, 240 ${INLET_Y[i]}, ${TANK.x} ${INLET_Y[i]}`;

const pipes = WEIGHTS.map((w, i) => {
  const width = 5 + Math.abs(w) * 6;
  const outer = svgEl("path", {
    d: pipePath(i),
    fill: "none",
    stroke: palette.coral,
    "stroke-width": width,
    "stroke-linecap": "round",
    ...(w < 0 ? { "stroke-dasharray": "9 7" } : {}),
    opacity: 0.85,
  });
  const inner = svgEl("path", {
    d: pipePath(i),
    fill: "none",
    stroke: palette.cobalt,
    "stroke-width": Math.max(2, width - 6),
    "stroke-linecap": "round",
  });
  return { outer, inner, w, i };
});

/* Output pipe: mint when flowing */
const outPipe = svgEl("path", {
  d: `M ${TANK.x + TANK.w} ${OUT_Y} L ${OUT_X} ${OUT_Y}`,
  fill: "none",
  stroke: palette.mint,
  "stroke-width": 10,
  "stroke-linecap": "round",
});

/* Tank */
const tankFill = svgEl("rect", {
  x: TANK.x + 4, width: TANK.w - 8,
  y: TANK.y + TANK.h - 4, height: 0,
  fill: palette.coral, opacity: 0.5, rx: 6,
});
const threshY = TANK.y + TANK.h - sumFrac(0) * TANK.h;
const tankBits = [
  tankFill,
  svgEl("rect", {
    x: TANK.x, y: TANK.y, width: TANK.w, height: TANK.h,
    fill: "none", stroke: palette.ink, "stroke-width": 2.5, rx: 10,
  }),
  svgEl("line", {
    x1: TANK.x - 6, x2: TANK.x + TANK.w + 6, y1: threshY, y2: threshY,
    stroke: palette.ink, "stroke-width": 2, "stroke-dasharray": "5 5",
  }),
  svgEl("text", {
    x: TANK.x + TANK.w / 2, y: threshY - 8,
    "font-size": 10.5, fill: palette.inkSoft, "text-anchor": "middle",
  }, "opens above here"),
];

/* Valve: a handle bar that rotates open (along the pipe) or closed (across it) */
const valveHandle = svgEl("rect", {
  x: VALVE.x - 16, y: VALVE.y - 4.5, width: 32, height: 9, rx: 4.5,
  fill: palette.ink,
});
const valveBits = [
  svgEl("circle", {
    cx: VALVE.x, cy: VALVE.y, r: VALVE.r,
    fill: palette.paper, stroke: palette.ink, "stroke-width": 2.5,
  }),
  valveHandle,
];

/* Ports and labels */
const staticBits = [];
INPUTS.forEach((inp, i) => {
  staticBits.push(
    svgEl("circle", { cx: PORT_X, cy: PORT_Y[i], r: 11, fill: palette.cobalt, stroke: palette.ink, "stroke-width": 2.5 }),
    svgEl("text", { x: PORT_X - 20, y: PORT_Y[i] - 22, "font-size": 12, fill: palette.inkSoft }, inp.short),
    svgEl("text", {
      x: 218, y: (PORT_Y[i] + INLET_Y[i]) / 2 - 12,
      "font-size": 13, "font-weight": 600, "text-anchor": "middle",
      fill: palette.coral,
    }, `× ${fmt(WEIGHTS[i], true)}`),
  );
});
staticBits.push(
  svgEl("circle", { cx: OUT_X, cy: OUT_Y, r: 11, fill: palette.mint, stroke: palette.ink, "stroke-width": 2.5 }),
  svgEl("text", { x: OUT_X, y: OUT_Y + 34, "font-size": 12, "text-anchor": "middle", fill: palette.inkSoft }, "output"),
  svgEl("text", { x: TANK.x + TANK.w / 2, y: TANK.y + TANK.h + 28, "font-size": 12, "text-anchor": "middle", fill: palette.inkSoft },
    `starts at ${fmt(BIAS, true)} (skeptical by default)`),
  svgEl("text", { x: VALVE.x, y: VALVE.y - 34, "font-size": 12, "text-anchor": "middle", fill: palette.inkSoft }, "valve (ReLU)"),
);

const sumText = svgEl("text", {
  x: TANK.x + TANK.w / 2, y: TANK.y - 14,
  "font-size": 15, "font-weight": 600, "text-anchor": "middle", fill: palette.ink,
});
const outText = svgEl("text", {
  x: OUT_X, y: OUT_Y - 24,
  "font-size": 15, "font-weight": 600, "text-anchor": "middle", fill: palette.ink,
});

/* Flow dots: input signal (cobalt) and output signal (mint) */
const DOTS_PER_PIPE = 4;
function makeDots(n, color, r) {
  return Array.from({ length: n }, (_, k) =>
    ({ elc: svgEl("circle", { r, fill: color, stroke: palette.ink, "stroke-width": 1.2 }), off: k / n }));
}
const inputDots = pipes.map(() => makeDots(DOTS_PER_PIPE, palette.cobalt, 5));
const outputDots = makeDots(5, palette.mint, 5.5);

svg.append(
  outPipe,
  ...pipes.map((p) => p.outer),
  ...pipes.map((p) => p.inner),
  ...inputDots.flat().map((d) => d.elc),
  ...outputDots.map((d) => d.elc),
  ...tankBits, ...valveBits, ...staticBits, sumText, outText,
);

/* ---------- Controls ---------- */

const sliders = [];
const controls = el("div", { class: "controls-row" },
  ...INPUTS.map((inp, i) => {
    const val = el("output", { class: "value", id: `in-val-${i}` }, fmt(DEFAULTS[i]));
    const range = el("input", {
      type: "range", min: 0, max: 100, step: 1,
      value: String(DEFAULTS[i] * 100),
      id: `in-${i}`,
      "aria-describedby": `in-val-${i}`,
      oninput: (e) => { x[i] = e.target.valueAsNumber / 100; update(); },
    });
    sliders.push(range);
    return el("div", { class: "knob" },
      el("label", { class: "knob-label", for: `in-${i}` }, el("span", {}, inp.label), val),
      range);
  })
);

const roSum = el("output", {}, "");
const roOut = el("output", {}, "");
const roGate = el("output", {}, "");
const gateRo = el("span", { class: "ro" }, el("b", {}, "valve"), roGate);
const readouts = el("div", { class: "neuron-readouts", "aria-live": "polite" },
  el("span", { class: "ro" }, el("b", {}, "weighted sum"), roSum),
  gateRo,
  el("span", { class: "ro" }, el("b", {}, "output"), roOut),
);

host.append(svg, readouts, controls);

/* ---------- State + rendering ---------- */

const levelSpring = spring(sumFrac(sums()), 12);
const valveSpring = spring(sums() > 0 ? 0 : 90, 12); // 0 = open (bar along pipe), 90 = closed
const phases = pipes.map(() => 0);
let outPhase = 0;

const pathLens = pipes.map((p) => p.outer.getTotalLength());
const outLen = outPipe.getTotalLength();

function update() {
  const s = sums();
  const out = relu(s);
  const open = s > 0;

  levelSpring.target = clamp(sumFrac(s), 0, 1);
  valveSpring.target = open ? 0 : 90;

  pipes.forEach((p, i) => {
    p.inner.setAttribute("opacity", 0.12 + 0.88 * x[i]);
  });
  outPipe.setAttribute("opacity", open ? 0.9 : 0.18);

  sumText.textContent = `sum ${fmt(s, true)}`;
  outText.textContent = `out ${fmt(out)}`;
  roSum.textContent = fmt(s, true);
  roOut.textContent = fmt(out);
  roGate.textContent = open ? "open" : "closed";
  gateRo.classList.toggle("is-open", open);
  gateRo.classList.toggle("is-closed", !open);

  if (reducedMotion) paintFrame(0); // instant stepped state
}

function paintFrame(dt) {
  levelSpring.step(dt);
  valveSpring.step(dt);

  const fillH = levelSpring.value * (TANK.h - 8);
  tankFill.setAttribute("y", TANK.y + TANK.h - 4 - fillH);
  tankFill.setAttribute("height", Math.max(0, fillH));
  tankFill.setAttribute("fill", sums() > 0 ? palette.mint : palette.coral);

  valveHandle.setAttribute("transform", `rotate(${valveSpring.value} ${VALVE.x} ${VALVE.y})`);

  const out = relu(sums());
  pipes.forEach((p, i) => {
    if (!reducedMotion) phases[i] = (phases[i] + dt * (0.1 + 0.5 * x[i])) % 1;
    inputDots[i].forEach((d) => {
      const show = x[i] > 0.02;
      d.elc.setAttribute("opacity", show ? 0.25 + 0.75 * x[i] : 0);
      if (!show) return;
      const pt = p.outer.getPointAtLength(((d.off + phases[i]) % 1) * pathLens[i]);
      d.elc.setAttribute("cx", pt.x);
      d.elc.setAttribute("cy", pt.y);
    });
  });

  const flowing = out > 0.02;
  if (!reducedMotion) outPhase = (outPhase + dt * (0.15 + 0.22 * out)) % 1;
  outputDots.forEach((d) => {
    d.elc.setAttribute("opacity", flowing ? clamp(0.3 + out / 3, 0, 1) : 0);
    if (!flowing) return;
    const pt = outPipe.getPointAtLength(((d.off + outPhase) % 1) * outLen);
    d.elc.setAttribute("cx", pt.x);
    d.elc.setAttribute("cy", pt.y);
  });
}

document.getElementById("neuron-reset").addEventListener("click", () => {
  x = [...DEFAULTS];
  sliders.forEach((sl, i) => { sl.value = String(DEFAULTS[i] * 100); });
  controls.querySelectorAll("output.value").forEach((o, i) => { o.textContent = fmt(DEFAULTS[i]); });
  update();
});

controls.addEventListener("input", (e) => {
  const i = sliders.indexOf(e.target);
  if (i >= 0) controls.querySelector(`#in-val-${i}`).textContent = fmt(x[i]);
});

update();
paintFrame(0);
if (!reducedMotion) loop((dt) => paintFrame(dt), host);

/* =====================================================================
   Demo 2 · Stacking layers
   ===================================================================== */

const N_LAYERS = 12;
const SENTENCE = ["The", "chef", "who", "burned", "the", "toast", "apologized"];
const FOCUS = SENTENCE.length - 1; // "apologized"

/* Hand-authored, cumulative annotations. Illustrative of what probing
   research reports at comparable depths; labelled as such in the page copy. */
const TAGS = [
  { layer: 1, phase: "form", text: "a chunk of letters ending in “-ed”" },
  { layer: 2, phase: "form", text: "a real word, correctly spelled" },
  { layer: 3, phase: "form", text: "past-tense verb" },
  { layer: 4, phase: "structure", text: "the main verb of the sentence" },
  { layer: 5, phase: "structure", text: "its subject is “chef”, not “toast”" },
  { layer: 7, phase: "structure", text: "done by the same chef who burned the toast" },
  { layer: 9, phase: "meaning", text: "a social move: smoothing things over" },
  { layer: 10, phase: "meaning", text: "stakes: low, a kitchen mishap" },
  { layer: 11, phase: "predict", text: "“to” or “for” would fit right after" },
  { layer: 12, phase: "predict", text: "ready to bet on the next word" },
];

const PHASES = [
  { key: "form", label: "spelling & grammar", tint: "var(--tag-form)" },
  { key: "structure", label: "who did what", tint: "var(--tag-structure)" },
  { key: "meaning", label: "what it means", tint: "var(--tag-meaning)" },
  { key: "predict", label: "what comes next", tint: "var(--tag-predict)" },
];

let depth = 1;

const stackHost = document.getElementById("stack-demo");

/* Sentence row: the focus word is the one under inspection */
const sentenceRow = el("div", { class: "token-row", role: "group", "aria-label": "Sentence tokens" },
  ...SENTENCE.map((t, i) => el("span", { class: "token" + (i === FOCUS ? " active" : "") }, t)));

/* Legend */
const legend = el("div", { class: "legend", style: "margin-top:0.9rem;" },
  ...PHASES.map((p) => el("span", {},
    el("span", { class: "swatch", style: `background:${p.tint};` }), p.label)));

/* Layer tower */
const blocks = Array.from({ length: N_LAYERS }, (_, k) =>
  el("div", { class: "block", title: `layer ${k + 1}: attention, then think` }, `L${k + 1}`));
const tower = el("div", { class: "layer-tower", "aria-hidden": "true" }, ...blocks);

/* Tag stack */
const tagEls = TAGS.map((t) =>
  el("span", { class: "tag", "data-phase": t.phase },
    el("span", { class: "lyr" }, `L${t.layer}`), t.text));
const knowsTitle = el("p", { class: "knows-title" });
const knows = el("div", { class: "knows" },
  knowsTitle,
  el("div", { class: "tag-stack", "aria-live": "polite" }, ...tagEls));

/* Depth slider */
const depthVal = el("output", { class: "value", id: "depth-val" }, "");
const depthRange = el("input", {
  type: "range", min: 1, max: N_LAYERS, step: 1, value: String(depth),
  id: "depth", "aria-describedby": "depth-val",
  oninput: (e) => { depth = e.target.valueAsNumber; renderStack(); },
});
const depthKnob = el("div", { class: "stack-controls" },
  el("div", { class: "knob" },
    el("label", { class: "knob-label", for: "depth" }, el("span", {}, "depth into the stack"), depthVal),
    depthRange));

stackHost.append(sentenceRow, legend, el("div", { class: "stack-wrap" }, tower, knows), depthKnob);

let prevDepth = 0;
function renderStack() {
  depthVal.textContent = `layer ${depth} / ${N_LAYERS}`;
  knowsTitle.replaceChildren(
    "what layer ", el("span", { class: "mono" }, String(depth)),
    " knows about ", el("span", { class: "mono" }, "“apologized”"));

  blocks.forEach((b, k) => {
    b.classList.toggle("done", k + 1 < depth);
    b.classList.toggle("here", k + 1 === depth);
  });

  TAGS.forEach((t, i) => {
    const on = depth >= t.layer;
    const wasOn = prevDepth >= t.layer;
    tagEls[i].classList.toggle("on", on);
    if (on && !wasOn && !reducedMotion) {
      tagEls[i].classList.add("fresh");
      setTimeout(() => tagEls[i].classList.remove("fresh"), 320);
    }
  });
  prevDepth = depth;
}

document.getElementById("stack-reset").addEventListener("click", () => {
  depth = 1;
  depthRange.value = "1";
  renderStack();
});

renderStack();
