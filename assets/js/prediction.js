/* Prediction chapter demos.
   Toy-sized but honest: one hand-authored score list, a real softmax,
   a real nucleus (top-p) cut, and real sampling from the renormalised
   distribution. The only cheat, declared in the prose: the scoreboard
   does not change between rolls. */

import { el, softmax, rng, spring, loop, reducedMotion, clamp } from "./lib.js";

/* ---------- The scoreboard: candidates for the next word ---------- */

const PROMPT = ["My", "favourite", "thing", "about", "winter", "is", "the"];

const CANDIDATES = [
  ["snow", 8.1],
  ["cold", 6.9],
  ["silence", 6.2],
  ["light", 5.9],
  ["smell", 5.6],
  ["frost", 5.4],
  ["stillness", 4.8],
  ["skiing", 4.4],
  ["darkness", 4.1],
  ["chill", 3.7],
  ["soup", 2.2],
  ["wifi", 1.4],
];

const SCORES = CANDIDATES.map(([, s]) => s);
const N = CANDIDATES.length;
const MAX_ROLLS = 6;

/* Session-seeded PRNG, advanced on every roll */
const rand = rng(((Date.now() & 0xffff) << 1) | 1);

/* Softmax at temperature T, then the real nucleus cut at top-p P.
   SCORES is sorted descending, and softmax is monotone, so the
   nucleus is always a prefix of the list. */
function distribution(T, P) {
  const probs = softmax(SCORES, T);
  const inNucleus = new Array(N).fill(false);
  let cum = 0;
  for (let i = 0; i < N; i++) {
    inNucleus[i] = true;
    cum += probs[i];
    if (cum >= P - 1e-9) break;
  }
  return { probs, inNucleus };
}

/* Sample from the truncated, renormalised distribution */
function sample({ probs, inNucleus }) {
  let total = 0;
  for (let i = 0; i < N; i++) if (inNucleus[i]) total += probs[i];
  let r = rand() * total;
  let last = 0;
  for (let i = 0; i < N; i++) {
    if (!inNucleus[i]) continue;
    last = i;
    r -= probs[i];
    if (r <= 0) return i;
  }
  return last;
}

const fmtPct = (p) => {
  const v = p * 100;
  if (v >= 1) return Math.round(v) + "%";
  if (v >= 0.1) return v.toFixed(1) + "%";
  return "<0.1%";
};

const wait = (ms) => new Promise((r) => setTimeout(r, reducedMotion ? 0 : ms));

/* ================= Hero demo: the sampling playground ================= */

const hero = document.getElementById("hero-demo");

let heroT = 0.8;
let heroP = 0.95;
let sampled = []; // indices of rolled words
let rolling = false;

const sentenceRow = el("div", {
  class: "token-row sentence",
  "aria-live": "polite",
  "aria-label": "Sentence so far",
});

const rollBtn = el("button", { class: "btn primary big", onclick: () => roll() }, "roll the dice");

const tKnob = el("div", { class: "knob" },
  el("div", { class: "knob-label" },
    el("label", { for: "t-slider" }, "temperature"),
    el("span", { class: "value", id: "t-val" }, "0.80")),
  el("input", { type: "range", id: "t-slider", min: "0.1", max: "2", step: "0.05", value: "0.8" }),
);
const pKnob = el("div", { class: "knob" },
  el("div", { class: "knob-label" },
    el("label", { for: "p-slider" }, "top-p"),
    el("span", { class: "value", id: "p-val" }, "0.95")),
  el("input", { type: "range", id: "p-slider", min: "0.1", max: "1", step: "0.01", value: "0.95" }),
);

const controls = el("div", { class: "roll-controls" }, tKnob, pKnob, el("div", {}, rollBtn));
const barsHost = el("div", { class: "prob-list", role: "img", "aria-label": "Probability of each candidate next word" });

hero.append(sentenceRow, controls, barsHost);

/* Build the bar rows once; update them in place for smooth animation */
const rows = CANDIDATES.map(([word]) => {
  const fill = el("div", { class: "fill" });
  const pct = el("span", { class: "pct" });
  const row = el("div", { class: "prob-row" },
    el("span", { class: "word" }, word),
    el("div", { class: "bar" }, fill),
    pct,
  );
  barsHost.append(row);
  return { row, fill, pct };
});

const tSlider = tKnob.querySelector("input");
const pSlider = pKnob.querySelector("input");
const tVal = tKnob.querySelector(".value");
const pVal = pKnob.querySelector(".value");

function updateBars() {
  const d = distribution(heroT, heroP);
  rows.forEach(({ row, fill, pct }, i) => {
    fill.style.width = d.probs[i] * 100 + "%";
    pct.textContent = fmtPct(d.probs[i]);
    row.classList.toggle("cut", !d.inNucleus[i]);
    row.title = d.inNucleus[i]
      ? `${CANDIDATES[i][0]}: ${fmtPct(d.probs[i])}`
      : `${CANDIDATES[i][0]}: cut by top-p, the dice can’t land here`;
  });
  barsHost.setAttribute("aria-label",
    "Probability of each candidate next word: " +
    CANDIDATES.map(([w], i) => d.inNucleus[i] ? `${w} ${fmtPct(d.probs[i])}` : `${w} cut`).join(", "));
  return d;
}

function renderSentence() {
  sentenceRow.replaceChildren(
    ...PROMPT.map((w) => el("span", { class: "token input" }, w)),
    ...sampled.map((i) => el("span", { class: "token output" }, CANDIDATES[i][0])),
  );
  rollBtn.textContent = sampled.length >= MAX_ROLLS ? "start again" : "roll the dice";
}

function markLanded(idx) {
  rows.forEach(({ row }, i) => row.classList.toggle("landed", i === idx));
}

async function roll() {
  if (rolling) return;
  if (sampled.length >= MAX_ROLLS) {
    sampled = [];
    markLanded(-1);
    renderSentence();
    return;
  }
  rolling = true;
  rollBtn.disabled = true;
  const d = distribution(heroT, heroP);
  const idx = sample(d); // honest: decided before the flicker
  if (!reducedMotion) {
    const options = d.inNucleus.map((n, i) => (n ? i : -1)).filter((i) => i >= 0);
    for (let hop = 0; hop < 6; hop++) {
      markLanded(options[Math.floor(rand() * options.length)]);
      await wait(55);
    }
  }
  markLanded(idx);
  sampled.push(idx);
  renderSentence();
  const chip = sentenceRow.lastChild;
  chip.classList.add("pop");
  rolling = false;
  rollBtn.disabled = false;
}

tSlider.addEventListener("input", () => {
  heroT = clamp(+tSlider.value, 0.1, 2);
  tVal.textContent = heroT.toFixed(2);
  updateBars();
});
pSlider.addEventListener("input", () => {
  heroP = clamp(+pSlider.value, 0.1, 1);
  pVal.textContent = heroP.toFixed(2);
  updateBars();
});

document.getElementById("hero-reset").addEventListener("click", () => {
  sampled = [];
  heroT = 0.8;
  heroP = 0.95;
  tSlider.value = "0.8";
  pSlider.value = "0.95";
  tVal.textContent = "0.80";
  pVal.textContent = "0.95";
  markLanded(-1);
  renderSentence();
  updateBars();
});

renderSentence();
updateBars();

/* ================= Preset demo: temperature as personality ================= */

const presetStage = document.getElementById("preset-demo");
const runsHost = document.getElementById("runs");
const t2Slider = document.getElementById("t2-slider");
const p2Slider = document.getElementById("p2-slider");
const t2Val = document.getElementById("t2-val");
const p2Val = document.getElementById("p2-val");
const presetBtns = [...presetStage.querySelectorAll(".preset")];
const customRollBtn = document.getElementById("preset-roll");

/* Springs drive the slider thumbs so presets glide into place */
const tSpring = spring(0.8, 10);
const pSpring = spring(0.95, 10);

function reflectSliders() {
  t2Slider.value = tSpring.value.toFixed(2);
  p2Slider.value = pSpring.value.toFixed(2);
  t2Val.textContent = (+t2Slider.value).toFixed(2);
  p2Val.textContent = (+p2Slider.value).toFixed(2);
}

loop((dt) => {
  const before = tSpring.value + pSpring.value * 1000;
  tSpring.step(dt);
  pSpring.step(dt);
  if (tSpring.value + pSpring.value * 1000 !== before) reflectSliders();
}, presetStage);

t2Slider.addEventListener("input", () => {
  tSpring.value = tSpring.target = clamp(+t2Slider.value, 0.1, 2);
  t2Val.textContent = tSpring.target.toFixed(2);
  presetBtns.forEach((b) => b.setAttribute("aria-pressed", "false"));
});
p2Slider.addEventListener("input", () => {
  pSpring.value = pSpring.target = clamp(+p2Slider.value, 0.1, 1);
  p2Val.textContent = pSpring.target.toFixed(2);
  presetBtns.forEach((b) => b.setAttribute("aria-pressed", "false"));
});

const runRows = []; // newest first, max 3 kept

function newRun(name, T, P) {
  const chipHost = el("div", { class: "token-row" },
    el("span", { class: "lead-in" }, "…the"));
  const row = el("div", { class: "run" },
    el("div", { class: "run-label" },
      el("b", {}, name),
      `T ${T.toFixed(1)} · p ${P.toFixed(2)}`),
    chipHost,
  );
  runRows.unshift(row);
  while (runRows.length > 3) runRows.pop();
  runsHost.replaceChildren(...runRows);
  return chipHost;
}

let running = false;
async function rollLine(name, T, P) {
  if (running) return;
  running = true;
  presetBtns.forEach((b) => (b.disabled = true));
  customRollBtn.disabled = true;
  const chipHost = newRun(name, T, P);
  const d = distribution(T, P);
  for (let k = 0; k < 6; k++) {
    const idx = sample(d);
    const chip = el("span", { class: "token output pop" }, CANDIDATES[idx][0]);
    chipHost.append(chip);
    await wait(240);
  }
  presetBtns.forEach((b) => (b.disabled = false));
  customRollBtn.disabled = false;
  running = false;
}

presetBtns.forEach((btn) => {
  btn.addEventListener("click", async () => {
    if (running) return;
    const T = +btn.dataset.t;
    const P = +btn.dataset.p;
    presetBtns.forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
    tSpring.target = T;
    pSpring.target = P;
    if (reducedMotion) {
      tSpring.value = T;
      pSpring.value = P;
      reflectSliders();
    } else {
      await wait(550); // let the knobs glide into place first
    }
    rollLine(btn.dataset.name, T, P);
  });
});

customRollBtn.addEventListener("click", () => {
  rollLine("custom", tSpring.target, pSpring.target);
});
