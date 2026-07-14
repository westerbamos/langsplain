/* Training chapter demos.
   Toy-sized but honest: demo 1 trains a real bigram counter in the browser,
   demo 2 averages real trait scores of your choices, demo 3 runs a real
   softmax over hand-authored logits. */

import { el, softmax, loop, spring, reducedMotion, clamp } from "./lib.js";

/* =====================================================================
   Demo 1: pretraining as fill-in-the-blank (live bigram counts)
   ===================================================================== */

const CORPUS = `the cat sat on the mat . the cat saw the moon . the moon was
bright . the cat sang to the moon . a dog saw the cat . the dog sat on the
mat . the cat sat on the wall . the moon rose over the wall . the dog sang
to the moon . the cat and the dog sat under the moon . the moon was kind .
the cat slept on the mat . the dog slept on the wall . the moon watched the
cat . the cat dreamed of the moon . a bird saw the dog . the bird sat on
the wall . the bird sang to the moon . the bird and the cat watched the
moon . the moon liked the song . the cat woke and saw the bird . the song
of the bird was kind . the moon knew the song .`;

const TOKENS = CORPUS.trim().split(/\s+/);
const VOCAB = [...new Set(TOKENS)];
const V = VOCAB.length;
const ALPHA = 0.5; // smoothing: the model's open mind before evidence
const SPOTLIGHTS = ["the", "cat", "moon"];
const MAX_SURPRISE = Math.log2(V / ALPHA); // worst case: a never-seen pair

/* Full-corpus counts, computed once, only to fix display order and row sets
   (which follower rows exist). All displayed numbers come from live counts. */
function countBigrams(times = 1) {
  const counts = new Map();
  for (let pass = 0; pass < times; pass++) {
    for (let i = 0; i < TOKENS.length - 1; i++) {
      const a = TOKENS[i], b = TOKENS[i + 1];
      if (!counts.has(a)) counts.set(a, new Map());
      counts.get(a).set(b, (counts.get(a).get(b) ?? 0) + 1);
    }
  }
  return counts;
}
const FINAL = countBigrams();
const followersOf = (w, n) =>
  [...(FINAL.get(w) ?? new Map()).entries()].sort((x, y) => y[1] - x[1]).slice(0, n).map((e) => e[0]);

/* Live model state */
let counts, totals, pos, passes, ema, reading;

function resetModel() {
  counts = new Map();
  totals = new Map();
  pos = 0;
  passes = 0;
  ema = null;
  reading = false;
}

const seen = (a, b) => counts.get(a)?.get(b) ?? 0;
const total = (a) => totals.get(a) ?? 0;
/* Smoothed probability: honest counts plus a whisper of open-mindedness,
   so "never seen" is unlikely rather than impossible. More data, sharper. */
const prob = (a, b) => (seen(a, b) + ALPHA) / (total(a) + ALPHA * V);

function learn(a, b) {
  const s = -Math.log2(prob(a, b)); // surprise BEFORE updating the tally
  ema = ema == null ? s : ema * 0.85 + s * 0.15;
  if (!counts.has(a)) counts.set(a, new Map());
  counts.get(a).set(b, seen(a, b) + 1);
  totals.set(a, total(a) + 1);
}

/* ---------- DOM ---------- */

const corpusBox = document.getElementById("corpus");
const talliesBox = document.getElementById("tallies");
const quizBefore = document.getElementById("quiz-before");
const quizAfter = document.getElementById("quiz-after");
const quizSummary = document.getElementById("quiz-summary");
const surpriseFill = document.getElementById("surprise-fill");
const surpriseVal = document.getElementById("surprise-val");
const readBtn = document.getElementById("read-btn");

const wordSpans = TOKENS.map((t) => el("span", { class: "cw" }, t));
corpusBox.replaceChildren(...wordSpans.flatMap((s) => [s, " "]));

function barRow(label, extraClass = "") {
  const fill = el("div", { class: `bf ${extraClass}` });
  const val = el("span", { class: "bv" }, "");
  const row = el("div", { class: "bar-row" },
    el("span", { class: "bw" }, label),
    el("div", { class: "bt" }, fill),
    val);
  return { row, fill, val };
}

/* Spotlight tally panels: fixed rows (order from full corpus), live counts */
const tallyRows = new Map(); // "the|cat" -> {fill, val}
talliesBox.replaceChildren(...SPOTLIGHTS.map((w) => {
  const panel = el("div", { class: "tally" },
    el("h4", {}, "after ", el("span", { class: "tw" }, w), " comes..."));
  for (const f of followersOf(w, 5)) {
    const r = barRow(f);
    tallyRows.set(`${w}|${f}`, r);
    panel.append(r.row);
  }
  return panel;
}));

function renderTallies() {
  for (const w of SPOTLIGHTS) {
    const rows = followersOf(w, 5).map((f) => ({ f, c: seen(w, f) }));
    const max = Math.max(1, ...rows.map((r) => r.c));
    for (const { f, c } of rows) {
      const r = tallyRows.get(`${w}|${f}`);
      r.fill.style.width = `${(c / max) * 100}%`;
      r.val.textContent = c === 0 ? "" : `×${c}`;
    }
  }
}

/* Quiz: "the ___", uniform shrug vs live learned distribution */
const QUIZ_WORDS = followersOf("the", 6);
const quizRowsAfter = new Map();
quizBefore.replaceChildren(...QUIZ_WORDS.map((f) => {
  const r = barRow(f, "uniform");
  r.fill.style.width = `${(1 / V) * 100}%`;
  r.val.textContent = `${Math.round((1 / V) * 100)}%`;
  return r.row;
}));
quizAfter.replaceChildren(...QUIZ_WORDS.map((f) => {
  const r = barRow(f);
  quizRowsAfter.set(f, r);
  return r.row;
}));

function renderQuiz() {
  for (const f of QUIZ_WORDS) {
    const p = prob("the", f);
    const r = quizRowsAfter.get(f);
    r.fill.style.width = `${p * 100}%`;
    r.val.textContent = `${Math.round(p * 100)}%`;
  }
}

function renderSurprise() {
  if (ema == null) {
    surpriseFill.style.width = "100%";
    surpriseVal.textContent = "not yet";
    return;
  }
  surpriseFill.style.width = `${clamp(ema / MAX_SURPRISE, 0.02, 1) * 100}%`;
  surpriseVal.textContent = ema.toFixed(1);
}

function renderCorpusMarks() {
  wordSpans.forEach((s, i) => {
    s.classList.toggle("read", i < pos);
    s.classList.toggle("now", reading && i === pos - 1);
  });
  const cur = wordSpans[Math.max(0, pos - 1)];
  corpusBox.scrollTop = cur.offsetTop - corpusBox.clientHeight / 2;
}

function announceQuiz() {
  const parts = QUIZ_WORDS.slice(0, 3)
    .map((f) => `${f} ${Math.round(prob("the", f) * 100)}%`).join(", ");
  quizSummary.textContent =
    `After ${passes} reading${passes === 1 ? "" : "s"}, the model's guesses for "the": ${parts}.`;
}

/* ---------- streaming reader ---------- */

let readTimer = null;

function stepWord() {
  if (pos > 0) learn(TOKENS[pos - 1], TOKENS[pos]);
  pos++;
}

function finishPass() {
  reading = false;
  passes++;
  readBtn.disabled = false;
  readBtn.textContent = "read it again";
  renderAllPre();
  announceQuiz();
}

function startReading() {
  if (reading) return;
  reading = true;
  pos = 0;
  readBtn.disabled = true;
  if (reducedMotion) {
    while (pos < TOKENS.length) stepWord();
    finishPass();
    return;
  }
  const tick = () => {
    stepWord();
    renderAllPre();
    if (pos >= TOKENS.length) { finishPass(); return; }
    readTimer = setTimeout(tick, 48);
  };
  tick();
}

function renderAllPre() {
  renderCorpusMarks();
  renderTallies();
  renderQuiz();
  renderSurprise();
}

readBtn.addEventListener("click", startReading);
document.getElementById("reset-pre").addEventListener("click", () => {
  clearTimeout(readTimer);
  resetModel();
  readBtn.disabled = false;
  readBtn.textContent = "read";
  quizSummary.textContent = "Model reset. It knows nothing again.";
  renderAllPre();
});

resetModel();
renderAllPre();

/* =====================================================================
   Demo 2: RLHF, the taste test (traits: warmth, brevity, hedging)
   ===================================================================== */

const TRAITS = ["warmth", "brevity", "hedging"];

const ROUNDS = [
  {
    a: { text: "Rain is a form of precipitation. Atmospheric water vapor condenses into droplets which, upon attaining sufficient mass, descend to the surface.", traits: { warmth: 0.1, brevity: 0.75, hedging: 0.1 } },
    b: { text: "The sky is like a big sponge full of tiny water drops. When the sponge gets too full, it starts to drip. Those drips are rain!", traits: { warmth: 0.9, brevity: 0.7, hedging: 0.1 } },
  },
  {
    a: { text: "Well, it sort of depends, but clouds can hold water, and sometimes, in many cases, if conditions are right, some of that water might fall out, which is roughly what people tend to call rain.", traits: { warmth: 0.5, brevity: 0.2, hedging: 0.9 } },
    b: { text: "Clouds drink up water from puddles and the sea. When a cloud gets too heavy to hold it all, the water falls back down. That falling water is rain.", traits: { warmth: 0.7, brevity: 0.75, hedging: 0.1 } },
  },
  {
    a: { text: "Imagine the ocean taking a warm bath and the steam floating up, up, up until it gets cold and huddles into a cloud. The cloud lugs the water around like a backpack, and when the backpack gets too heavy, whoosh, it all spills onto your umbrella. It has been going around in that circle since before the dinosaurs.", traits: { warmth: 0.9, brevity: 0.2, hedging: 0.1 } },
    b: { text: "Water floats up as invisible steam, bunches into clouds, gets heavy, and falls back down. Rain is the sky returning the water it borrowed.", traits: { warmth: 0.6, brevity: 0.9, hedging: 0.1 } },
  },
];

const FINALS = [
  { traits: { warmth: 0.85, brevity: 0.8, hedging: 0.1 }, text: "Clouds are like fluffy sponges that drink up water from the sea. When a sponge gets too full, drip drip drip, it rains! Then the sun helps the water float back up so it can try again." },
  { traits: { warmth: 0.9, brevity: 0.25, hedging: 0.1 }, text: "Picture the sea taking a warm bath. The steam floats up and up, gets chilly, and snuggles together into a cloud. The cloud carries all that water around the sky like a heavy backpack, and when it just cannot hold any more, whoosh, down it comes, onto roofs and gardens and your umbrella. Then the sun warms the puddles and the whole trip starts over, like the world's slowest merry-go-round." },
  { traits: { warmth: 0.15, brevity: 0.85, hedging: 0.1 }, text: "Rain is water that falls from clouds. Water evaporates, condenses into droplets inside clouds, and when the droplets grow heavy enough, they fall." },
  { traits: { warmth: 0.5, brevity: 0.35, hedging: 0.85 }, text: "Well, it depends a little, but generally speaking clouds hold tiny drops of water, and when there are, you know, quite a lot of them, some of it will usually fall down, and that is more or less what rain is, in most cases." },
];

const roundLabel = document.getElementById("rl-round");
const candPair = document.getElementById("cand-pair");
const metersBox = document.getElementById("meters");
const finalSlot = document.getElementById("final-slot");

let rlRound, rlChoices;

const meterRows = new Map();
metersBox.replaceChildren(
  el("p", { style: "font-size:0.88rem; font-weight:700; margin-bottom:0.2rem;" },
    "what the model is learning about your taste"),
  ...TRAITS.map((t) => {
    const r = barRow(t);
    meterRows.set(t, r);
    return r.row;
  }),
);

function prefVector() {
  const pref = {};
  for (const t of TRAITS) {
    pref[t] = rlChoices.length === 0
      ? 0.5
      : rlChoices.reduce((s, c) => s + c.traits[t], 0) / rlChoices.length;
  }
  return pref;
}

function renderMeters() {
  const pref = prefVector();
  for (const t of TRAITS) {
    const r = meterRows.get(t);
    r.fill.style.width = `${pref[t] * 100}%`;
    r.val.textContent = rlChoices.length === 0 ? "50 · untuned" : `${Math.round(pref[t] * 100)}`;
  }
}

function typeInto(node, text) {
  if (reducedMotion) { node.textContent = text; return; }
  node.textContent = "";
  let i = 0;
  const t = setInterval(() => {
    i = Math.min(text.length, i + 2);
    node.textContent = text.slice(0, i);
    if (i >= text.length) clearInterval(t);
  }, 18);
}

function showFinal() {
  roundLabel.textContent = "taste test · tuned";
  candPair.replaceChildren();
  const pref = prefVector();
  const best = FINALS.reduce((a, b) => {
    const d = (f) => TRAITS.reduce((s, t) => s + (f.traits[t] - pref[t]) ** 2, 0);
    return d(b) < d(a) ? b : a;
  });
  const ft = el("p", { class: "ft" });
  finalSlot.replaceChildren(el("div", { class: "final-panel", "aria-live": "polite" },
    el("p", { class: "fl" }, "the tuned model answers, matched to your picks:"),
    ft,
    el("p", { class: "hint", style: "margin-top:0.6rem;" },
      "same facts it always had. Your three clicks only chose the manner. Real labelers do this thousands of times."),
  ));
  typeInto(ft, best.text);
}

function showRound() {
  if (rlRound >= ROUNDS.length) { showFinal(); return; }
  roundLabel.textContent = `taste test · round ${rlRound + 1} of 3`;
  const { a, b } = ROUNDS[rlRound];
  const make = (cand, label) => el("button", { class: "cand", onclick: (e) => pick(cand, e.currentTarget) },
    el("span", { class: "cl" }, `answer ${label}`),
    el("span", {}, cand.text));
  candPair.replaceChildren(make(a, "A"), make(b, "B"));
}

function pick(cand, btn) {
  if (rlChoices.length > rlRound) return; // already picked this round
  rlChoices.push(cand);
  btn.classList.add("picked");
  [...candPair.children].forEach((c) => {
    c.disabled = true;
    if (c !== btn) c.classList.add("dimmed");
  });
  renderMeters();
  rlRound++;
  setTimeout(showRound, reducedMotion ? 0 : 750);
}

function resetRl() {
  rlRound = 0;
  rlChoices = [];
  finalSlot.replaceChildren();
  renderMeters();
  showRound();
}

document.getElementById("rl-reset").addEventListener("click", resetRl);
resetRl();

/* =====================================================================
   Demo 3: hallucination, the model must answer (real softmax)
   ===================================================================== */

const HALL = [
  { word: "New", honest: false, raw: 2.2, trained: 0.2 },
  { word: "Luna", honest: false, raw: 1.9, trained: 0.5 },
  { word: "Tranquility", honest: false, raw: 1.6, trained: 0.1 },
  { word: "Armstrong", honest: false, raw: 0.6, trained: -0.8 },
  { word: "not a real place", honest: true, raw: 0.2, trained: 2.6 },
  { word: "unknown", honest: true, raw: -0.1, trained: 1.9 },
];

const MODES = {
  raw: "raw pretraining",
  trained: "trained to admit uncertainty",
};
let hallMode = "raw";

const hallRowsBox = document.getElementById("hall-rows");
const hallSummary = document.getElementById("hall-summary");
const hallToggle = document.getElementById("hall-toggle");
const hallSprings = HALL.map(() => spring(0, 9));

const hallRows = HALL.map((c) => {
  const r = barRow(c.word);
  r.row.classList.add(c.honest ? "honest" : "fluent");
  hallRowsBox.append(r.row);
  return r;
});

function hallProbs() {
  return softmax(HALL.map((c) => c[hallMode]));
}

function retargetHall() {
  const ps = hallProbs();
  hallSprings.forEach((s, i) => { s.target = ps[i]; });
  const top = ps.indexOf(Math.max(...ps));
  hallSummary.textContent =
    `${MODES[hallMode]}: most likely continuation is "${HALL[top].word}" at ${Math.round(ps[top] * 100)}%.`;
  hallRowsBox.setAttribute("aria-label",
    `Probability of each continuation, ${MODES[hallMode]}: ` +
    HALL.map((c, i) => `${c.word} ${Math.round(ps[i] * 100)}%`).join(", "));
}

function buildHallToggle() {
  hallToggle.replaceChildren(...Object.entries(MODES).map(([key, label]) =>
    el("button", {
      "aria-pressed": String(key === hallMode),
      onclick: () => { hallMode = key; retargetHall(); buildHallToggle(); },
    }, label)));
}

const hallStage = hallRowsBox.closest(".stage");
loop((dt) => {
  hallSprings.forEach((s, i) => {
    const before = s.value;
    s.step(dt);
    if (s.value === before) return; // settled, skip the DOM write
    hallRows[i].fill.style.width = `${s.value * 100}%`;
    hallRows[i].val.textContent = `${Math.round(s.value * 100)}%`;
  });
}, hallStage);

buildHallToggle();
retargetHall();
