/* Speed chapter demos: KV cache, quantization, speculative decoding.
   Toy-sized but honest: every counter is incremented by the animation
   steps themselves, every snapped value is really rounded to 2^bits levels,
   and the drafter/verifier tallies come from the scripted rounds as played. */

import { el, svgEl, loop, spring, reducedMotion, clamp, palette, rng } from "./lib.js";

/* =====================================================================
   Demo 1 · KV cache
   ===================================================================== */

const SENT = ["The", "robot", "finished", "its", "homework", "and", "went", "straight", "to", "sleep"];

const kv = {
  mode: "nocache",     // "nocache" | "cache"
  pos: 1,              // tokens generated so far ("The" is the prompt)
  count: 0,            // lookbacks performed, incremented per flash
  flashJ: -1,          // token currently being (re)read
  playing: false,
  busy: false,         // a step's animation chain is running
  timer: null,
};
const kvTotals = { nocache: null, cache: null };

const kvHost = document.getElementById("kv-demo");
const kvButtons = el("div", { class: "demo-buttons" });
const kvPlayBtn = el("button", { class: "btn primary", onclick: kvPlayPause }, "play");
const kvStepBtn = el("button", { class: "btn", onclick: () => { if (!kv.busy && !kv.playing) kvStepOnce(() => {}); } }, "step");
const kvResetBtn = el("button", { class: "btn", onclick: () => kvReset(kv.mode) }, "reset");
kvButtons.append(kvPlayBtn, kvStepBtn, kvResetBtn);

const kvStrip = el("div", { class: "token-row", role: "group", "aria-label": "Sentence being generated" });
const kvCounter = el("p", { class: "counter", "aria-live": "polite" });
const kvScore = el("div", { class: "score-row" });
kvHost.append(kvButtons, kvStrip, kvCounter, kvScore);

function kvRender() {
  const parts = [];
  const last = kv.pos - 1;
  if (kv.mode === "cache" && kv.pos > 1) {
    parts.push(el("div", { class: "kv-shelf", role: "group", "aria-label": "Saved tokens" },
      el("span", { class: "kv-shelf-label" }, "saved"),
      ...SENT.slice(0, last).map((w) => el("span", { class: "token output" }, w)),
    ));
  } else {
    for (let i = 0; i < last; i++) {
      parts.push(el("span", { class: "token" + (kv.flashJ === i ? " reading" : "") }, SENT[i]));
    }
  }
  parts.push(el("span", {
    class: "token" + (kv.flashJ === last ? " reading" : "") + (last > 0 && kv.busy && kv.flashJ !== last ? " active" : ""),
  }, SENT[last]));
  kvStrip.replaceChildren(...parts);

  const stepN = kv.pos - 1;
  kvCounter.textContent = kv.pos >= SENT.length
    ? `sentence done: ${kv.count} lookbacks total`
    : stepN === 0 && kv.count === 0
      ? `step 0: 0 lookbacks so far`
      : `step ${stepN}: ${kv.count} lookbacks so far`;
}

function kvRenderScore() {
  const slot = (mode, label) => {
    const t = kvTotals[mode];
    return el("div", { class: "score-slot" + (t != null ? " filled" : "") },
      t != null ? el("span", {}, `${label}: `, el("b", {}, `${t} lookbacks`)) : `${label}: play a full sentence`);
  };
  kvScore.replaceChildren(slot("nocache", "re-read every time"), slot("cache", "with the cache"));
}

function kvFinishCheck() {
  if (kv.pos >= SENT.length) {
    kvTotals[kv.mode] = kv.count;
    kvRenderScore();
  }
}

/* One generation step: honest flashes, one counter tick per flash. */
function kvStepOnce(done) {
  if (kv.pos >= SENT.length) { done(false); return; }
  kv.busy = true;
  kvSyncControls();
  const finish = () => {
    kv.flashJ = -1;
    kv.pos++;
    kv.busy = false;
    kvRender();
    kvFinishCheck();
    kvSyncControls();
    done(true);
  };
  if (reducedMotion) {
    // Stepped instant state: same arithmetic, no flashes.
    kv.count += kv.mode === "nocache" ? kv.pos : 1;
    finish();
    return;
  }
  if (kv.mode === "nocache") {
    let j = 0;
    const flash = () => {
      kv.flashJ = j;
      kv.count++;
      kvRender();
      j++;
      kv.timer = setTimeout(j < kv.pos ? flash : finish, j < kv.pos ? 110 : 150);
    };
    flash();
  } else {
    kv.flashJ = kv.pos - 1; // only the newest token does one lookback pass
    kv.count++;
    kvRender();
    kv.timer = setTimeout(finish, 240);
  }
}

function kvPlayLoop() {
  if (!kv.playing) return;
  kvStepOnce((advanced) => {
    if (!advanced || kv.pos >= SENT.length) {
      kv.playing = false;
      kvSyncControls();
      return;
    }
    kv.timer = setTimeout(kvPlayLoop, reducedMotion ? 700 : 280);
  });
}

function kvPlayPause() {
  if (kv.playing) {
    kv.playing = false; // current step's chain finishes, then we stop
    kvSyncControls();
    return;
  }
  if (kv.pos >= SENT.length) kvReset(kv.mode);
  kv.playing = true;
  kvSyncControls();
  const start = () => {
    if (!kv.playing) return;
    if (kv.busy) setTimeout(start, 60);
    else kvPlayLoop();
  };
  start();
}

function kvSyncControls() {
  kvPlayBtn.textContent = kv.playing ? "pause" : kv.pos >= SENT.length ? "replay" : "play";
  kvStepBtn.disabled = kv.busy || kv.playing || kv.pos >= SENT.length;
  kvResetBtn.disabled = kv.busy;
}

function kvReset(mode) {
  clearTimeout(kv.timer);
  kv.mode = mode;
  kv.pos = 1;
  kv.count = 0;
  kv.flashJ = -1;
  kv.playing = false;
  kv.busy = false;
  kvRender();
  kvSyncControls();
  kvBuildModeSeg();
}

const kvModeSeg = document.getElementById("kv-mode");
const KV_MODES = { nocache: "re-read every time", cache: "remember it (cache)" };
function kvBuildModeSeg() {
  kvModeSeg.replaceChildren(...Object.entries(KV_MODES).map(([key, label]) =>
    el("button", {
      "aria-pressed": String(key === kv.mode),
      onclick: () => { if (!kv.busy) kvReset(key); },
    }, label)));
}

kvBuildModeSeg();
kvRender();
kvRenderScore();
kvSyncControls();

/* =====================================================================
   Demo 2 · Quantization
   ===================================================================== */

const QVAL = 0.7346218;
const Q_BITS = [32, 16, 8, 4];
let qBits = 32;

const quantize = (v, bits) => {
  const levels = 2 ** bits;
  return Math.round(v * (levels - 1)) / (levels - 1);
};
const qDecimals = (bits) => Math.min(7, Math.max(2, Math.ceil(bits * Math.log10(2))));

/* 100 deterministic toy weights */
const qDots = (() => { const r = rng(7); return Array.from({ length: 100 }, () => r()); })();

const qHost = document.getElementById("q-demo");
const qReadout = el("p", { class: "q-readout", "aria-live": "polite" });

/* Ruler SVG */
const QW = 600, QH = 96, QX0 = 14, QX1 = QW - 14, QY = 64;
const qSvg = svgEl("svg", { viewBox: `0 0 ${QW} ${QH}`, class: "q-ruler", role: "img", "aria-label": "Precision ruler showing the weight snapping to the nearest allowed level" });
const qTicksG = svgEl("g");
const qGhostG = svgEl("g");
const qMarkerG = svgEl("g");
qSvg.append(
  svgEl("line", { x1: QX0, y1: QY, x2: QX1, y2: QY, stroke: palette.ink, "stroke-width": 2 }),
  qTicksG, qGhostG, qMarkerG,
  svgEl("text", { x: QX0, y: QY + 22, "font-size": 12, "font-family": "var(--font-mono)", fill: palette.inkSoft }, "0"),
  svgEl("text", { x: QX1, y: QY + 22, "font-size": 12, "font-family": "var(--font-mono)", fill: palette.inkSoft, "text-anchor": "end" }, "1"),
);
const qToX = (v) => QX0 + v * (QX1 - QX0);

/* Ghost of the original, full-precision value */
qGhostG.append(
  svgEl("line", { x1: qToX(QVAL), y1: QY - 14, x2: qToX(QVAL), y2: QY, stroke: palette.inkSoft, "stroke-width": 2, "stroke-dasharray": "3 3" }),
  svgEl("text", { x: qToX(QVAL), y: QY - 40, "font-size": 11, "font-family": "var(--font-mono)", fill: palette.inkSoft, "text-anchor": "middle" }, "original"),
);

/* Marker (springs toward the quantized position) */
const qMarkerStem = svgEl("line", { x1: 0, y1: QY - 22, x2: 0, y2: QY, stroke: palette.ink, "stroke-width": 2.5 });
const qMarkerHead = svgEl("circle", { cx: 0, cy: QY - 26, r: 7, fill: palette.marigold, stroke: palette.ink, "stroke-width": 2 });
qMarkerG.append(qMarkerStem, qMarkerHead);

function qDrawTicks() {
  qTicksG.replaceChildren();
  const levels = 2 ** qBits;
  if (levels <= 256) {
    const major = levels <= 16;
    for (let i = 0; i < levels; i++) {
      const x = qToX(i / (levels - 1));
      qTicksG.append(svgEl("line", {
        x1: x, y1: QY - (major ? 12 : 7), x2: x, y2: QY,
        stroke: major ? palette.cobalt : palette.inkSoft,
        "stroke-width": major ? 2 : 1,
        opacity: major ? 0.9 : 0.55,
      }));
    }
  } else {
    /* Too many ticks to draw one by one: finer than a pixel. */
    qTicksG.append(
      svgEl("rect", { x: QX0, y: QY - 9, width: QX1 - QX0, height: 9, fill: palette.cobalt, opacity: 0.14 }),
      svgEl("text", { x: (QX0 + QX1) / 2, y: QY - 14, "font-size": 11, "font-family": "var(--font-mono)", fill: palette.inkSoft, "text-anchor": "middle" },
        `${levels.toLocaleString("en-US")} levels: finer than this screen`),
    );
  }
}

/* Memory bar */
const qMemFill = el("div", { class: "membar-fill", style: "width:100%;" });
const qMemTrack = el("div", { class: "membar-track", role: "img", "aria-label": "Memory used per weight" }, qMemFill);
const qMemLabel = el("p", { class: "membar-label" });

/* 100 dots */
const qField = el("div", { class: "dotfield", role: "img", "aria-label": "100 toy weights snapping onto the allowed levels" });
const jitter = rng(21);
const qDotEls = qDots.map((v, i) => {
  const d = el("div", { class: "dot" });
  d.style.top = `${6 + (i % 6) * 9 + Math.round(jitter() * 4)}px`;
  d.style.left = `${(v * 100).toFixed(3)}%`;
  qField.append(d);
  return d;
});
const qDistinct = el("p", { class: "membar-label", "aria-live": "polite" });
const qNote = el("p", { class: "hint", style: "margin-top:0.9rem;", "aria-live": "polite" });

qHost.append(qReadout, qSvg, qMemTrack, qMemLabel, qField, qDistinct, qNote);

const qSpring = spring(QVAL, 12);

function qRender() {
  const q = quantize(QVAL, qBits);
  const d = qDecimals(qBits);
  qSpring.target = q;
  qDrawTicks();

  qReadout.replaceChildren(
    el("span", { class: "was" }, `${QVAL} `),
    el("span", {}, "→ stored as "),
    el("span", { class: "now" }, q.toFixed(d)),
    el("span", { class: "was" }, ` at ${qBits} bits`),
  );

  qMemFill.style.width = `${(qBits / 32) * 100}%`;
  qMemLabel.textContent = `${qBits} bits per weight = ${Math.round((qBits / 32) * 100)}% of full-size memory`;

  const snapped = qDots.map((v) => quantize(v, qBits));
  qDotEls.forEach((dEl, i) => { dEl.style.left = `${(snapped[i] * 100).toFixed(3)}%`; });
  const distinct = new Set(snapped).size;
  qDistinct.textContent = `distinct values among these 100 weights: ${distinct}`;

  const maxErr = Math.max(...snapped.map((s, i) => Math.abs(s - qDots[i])));
  qNote.textContent = qBits === 4
    ? `wobble alert: the worst of these 100 weights just moved by ${maxErr.toFixed(3)}. The model still mostly works, but at 4 bits it can get a little dizzy.`
    : `biggest nudge to any of these 100 weights: ${maxErr < 1e-4 ? maxErr.toExponential(1) : maxErr.toFixed(4)}. You cannot tell. That is the trick.`;
}

const qSeg = document.getElementById("q-bits");
function qBuildSeg() {
  qSeg.replaceChildren(...Q_BITS.map((b) =>
    el("button", {
      "aria-pressed": String(b === qBits),
      onclick: () => { qBits = b; qRender(); qBuildSeg(); },
    }, `${b}-bit`)));
}

qBuildSeg();
qRender();
qSpring.value = quantize(QVAL, qBits);

loop((dt) => {
  qSpring.step(dt);
  const x = qToX(clamp(qSpring.value, 0, 1));
  qMarkerStem.setAttribute("x1", x);
  qMarkerStem.setAttribute("x2", x);
  qMarkerHead.setAttribute("cx", x);
}, qHost);

/* =====================================================================
   Demo 3 · Speculative decoding
   ===================================================================== */

/* Hand-authored script: two full accepts, two mid-rejects.
   Each round costs exactly one big-model pass; the big model alone
   would need one pass per committed token. */
const ROUNDS = [
  { draft: ["The", "recipe", "needs", "two"], accept: 4, fix: null },
  { draft: ["cups", "of", "sugar", "and"], accept: 2, fix: "flour" },
  { draft: ["and", "a", "pinch", "of"], accept: 4, fix: null },
  { draft: ["patience", "every", "time", "!"], accept: 1, fix: "." },
];

const sp = {
  round: 0,
  committed: [],
  bigPasses: 0,
  drafts: 0,
  playing: false,
  busy: false,
  runToken: 0,
};

const spHost = document.getElementById("sp-demo");
const spDrafterWho = el("div", { class: "who", id: "sp-drafter" }, el("span", { class: "face small" }), "drafter · small and fast");
const spVerifierWho = el("div", { class: "who", id: "sp-verifier" }, el("span", { class: "face big" }), "verifier · big and careful");
const spCast = el("div", { class: "cast" }, spDrafterWho, spVerifierWho);

const spButtons = el("div", { class: "demo-buttons" });
const spPlayBtn = el("button", { class: "btn primary", onclick: spPlayPause }, "play");
const spStepBtn = el("button", { class: "btn", onclick: () => { if (!sp.busy && !sp.playing) spRunRound(() => {}); } }, "one round");
spButtons.append(spPlayBtn, spStepBtn);

const spStrip = el("div", { class: "token-row", role: "group", "aria-label": "Sentence being generated" });
const spStripWrap = el("div", { class: "stripwrap" }, spStrip);
const spCounter = el("p", { class: "counter", "aria-live": "polite" });
const spScore = el("div", { class: "score-row" });
spHost.append(spCast, spButtons, spStripWrap, spCounter, spScore);

const spT = (fn, ms) => {
  const tok = sp.runToken;
  setTimeout(() => { if (sp.runToken === tok) fn(); }, ms);
};

function spSetActive(who) {
  spDrafterWho.classList.toggle("active", who === "drafter");
  spVerifierWho.classList.toggle("active", who === "verifier");
}

function spRebuildStrip() {
  spStrip.replaceChildren(...sp.committed.map((w) => el("span", { class: "token output" }, w)));
}

function spUpdateCounter() {
  const n = sp.committed.length;
  spCounter.textContent = `${n} tokens committed · ${sp.bigPasses} big-model passes · ${sp.drafts} cheap drafts`;
}

function spRenderScore() {
  const n = sp.committed.length;
  const done = sp.round >= ROUNDS.length;
  const alone = el("div", { class: "score-slot" + (n > 0 ? " filled" : "") },
    n > 0 ? el("span", {}, "big model alone: ", el("b", {}, `${n} slow passes`), ` for ${n} tokens (1.0 per pass)`)
      : "big model alone: press play");
  const helped = el("div", { class: "score-slot" + (sp.bigPasses > 0 ? " filled" : "") },
    sp.bigPasses > 0
      ? el("span", {}, "with drafter: ", el("b", {}, `${sp.bigPasses} slow passes`),
        ` + ${sp.drafts} drafts for ${n} tokens (${(n / sp.bigPasses).toFixed(1)} per pass${done ? "" : " so far"})`)
      : "with drafter: press play");
  spScore.replaceChildren(alone, helped);
}

function spCommit(r) {
  sp.committed.push(...r.draft.slice(0, r.accept));
  if (r.fix != null) sp.committed.push(r.fix);
  sp.round++;
  spUpdateCounter();
  spRenderScore();
}

function spRunRound(done) {
  const r = ROUNDS[sp.round];
  if (!r) { done(false); return; }
  sp.busy = true;
  spSyncControls();

  const finishRound = () => {
    spSetActive(null);
    spRebuildStrip();
    sp.busy = false;
    spSyncControls();
    done(true);
  };

  if (reducedMotion) {
    /* Stepped instant state: same tallies, no animation. */
    sp.drafts += r.draft.length;
    sp.bigPasses++;
    spCommit(r);
    finishRound();
    return;
  }

  const draftChips = [];
  spSetActive("drafter");
  let i = 0;
  const addChip = () => {
    const c = el("span", { class: "token draft" }, r.draft[i]);
    spStrip.append(c);
    draftChips.push(c);
    sp.drafts++;
    spUpdateCounter();
    i++;
    if (i < r.draft.length) spT(addChip, 150);
    else spT(sweep, 380);
  };
  const sweep = () => {
    spSetActive("verifier");
    sp.bigPasses++; // one pass checks all four drafts
    spUpdateCounter();
    const first = draftChips[0];
    const last = draftChips[draftChips.length - 1];
    const ov = el("div", { class: "sweep" });
    ov.style.left = `${first.offsetLeft - 4}px`;
    ov.style.top = `${first.offsetTop - 3}px`;
    ov.style.height = `${first.offsetHeight + 6}px`;
    spStripWrap.append(ov);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      ov.style.left = `${last.offsetLeft + last.offsetWidth - 30}px`;
    }));
    spT(() => { ov.remove(); resolve(); }, 600);
  };
  const resolve = () => {
    draftChips.forEach((c, j) => {
      if (j < r.accept) { c.classList.remove("draft"); c.classList.add("output"); }
      else if (j === r.accept && r.fix != null) { c.classList.add("reject"); }
      else { c.classList.add("discard"); }
    });
    spT(() => {
      draftChips.forEach((c, j) => { if (c.classList.contains("discard")) c.remove(); });
      const rejected = r.fix != null ? draftChips[r.accept] : null;
      if (rejected) {
        rejected.textContent = r.fix; // the big model's own choice
        rejected.classList.remove("reject", "draft");
        rejected.classList.add("output");
      }
      spCommit(r);
      spT(finishRound, 420);
    }, 700);
  };
  addChip();
}

function spPlayLoop() {
  if (!sp.playing) return;
  spRunRound((advanced) => {
    if (!advanced || sp.round >= ROUNDS.length) {
      sp.playing = false;
      spSyncControls();
      return;
    }
    spT(spPlayLoop, reducedMotion ? 800 : 500);
  });
}

function spPlayPause() {
  if (sp.playing) {
    sp.playing = false; // current round finishes, then we stop
    spSyncControls();
    return;
  }
  if (sp.round >= ROUNDS.length) spReset();
  sp.playing = true;
  spSyncControls();
  const start = () => {
    if (!sp.playing) return;
    if (sp.busy) setTimeout(start, 60);
    else spPlayLoop();
  };
  start();
}

function spSyncControls() {
  spPlayBtn.textContent = sp.playing ? "pause" : sp.round >= ROUNDS.length ? "replay" : "play";
  spStepBtn.disabled = sp.busy || sp.playing || sp.round >= ROUNDS.length;
}

function spReset() {
  sp.runToken++;
  sp.round = 0;
  sp.committed = [];
  sp.bigPasses = 0;
  sp.drafts = 0;
  sp.playing = false;
  sp.busy = false;
  spSetActive(null);
  spStripWrap.querySelectorAll(".sweep").forEach((n) => n.remove());
  spRebuildStrip();
  spUpdateCounter();
  spRenderScore();
  spSyncControls();
}

document.getElementById("sp-reset").addEventListener("click", spReset);
spReset();
