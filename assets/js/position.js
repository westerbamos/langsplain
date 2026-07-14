/* Position chapter demos.
   Demo 1: the bag-of-words problem. The "bag" panel is computed from the
   multiset of tokens only (never their order), so it honestly cannot change
   when you reorder, until position tags fold order into the tokens themselves.
   Demo 2: RoPE as clock dials. Every hand angle is computed live as
   position × step; the highlighted relative angle is (posB - posA) × step,
   which is why it survives any global shift. */

import { el, svgEl, loop, spring, reducedMotion, clamp, rng, palette } from "./lib.js";

/* ==========================================================================
   Demo 1: the bag of words
   ========================================================================== */

const PRESETS = [
  { label: "man bites dog", words: ["man", "bites", "dog"] },
  { label: "only the chef", words: ["the", "chef", "only", "burned", "the", "toast"] },
  { label: "can you help", words: ["can", "you", "help"] },
];

const orderHost = document.getElementById("order-demo");

let presetIdx = 0;
let tagsOn = false;
let order = [];          // [{id, text}]; ids are stable, order is what you drag
let originalTexts = [];

const rowEl = el("div", {
  class: "token-row", id: "order-row", style: "position:relative;",
  role: "group", "aria-label": "Sentence tokens. Drag to reorder, or grab with Enter and move with arrow keys.",
});
const bagEl = el("div", { class: "pouch", role: "img", "aria-label": "" });
const verdictEl = el("div", { class: "verdict" });
const liveEl = el("div", { class: "sr-only", "aria-live": "polite" });

orderHost.append(
  el("div", { class: "order-split" },
    el("div", { class: "order-panel" },
      el("p", { class: "panel-label" }, "the sentence · ordered"),
      rowEl),
    el("div", { class: "order-panel" },
      el("p", { class: "panel-label" }, "what attention gets · the bag"),
      bagEl)),
  verdictEl,
  liveEl,
);

function loadPreset(i) {
  presetIdx = i;
  releaseGrab();
  order = PRESETS[i].words.map((text, id) => ({ id, text }));
  originalTexts = order.map((t) => t.text);
}

const wordOf = (id) => order.find((t) => t.id === id).text;

function announce(msg) {
  liveEl.textContent = "";
  liveEl.textContent = msg;
}

/* ---------- ordered row ---------- */

function makeChip(t) {
  const chip = el("button", {
    class: "token input",
    "data-id": t.id,
    "aria-pressed": "false",
  }, t.text, el("span", { class: "pos-tag", "aria-hidden": "true" }, ""));
  chip.addEventListener("pointerdown", (e) => startDrag(e, chip));
  chip.addEventListener("keydown", (e) => onChipKey(e, chip));
  chip.addEventListener("click", () => {
    if (suppressClick) { suppressClick = false; return; }
    toggleGrab(chip);
  });
  return chip;
}

function updateRowMeta() {
  [...rowEl.children].forEach((c, i) => {
    c.querySelector(".pos-tag").textContent = i + 1;
    c.setAttribute("aria-label",
      `${order[i].text}, position ${i + 1} of ${order.length}. Press Enter to grab, arrow keys to move.`);
  });
}

function renderRow({ flip = true } = {}) {
  const prev = new Map([...rowEl.children].map((c) => [c.dataset.id, c.getBoundingClientRect()]));
  rowEl.replaceChildren(...order.map(makeChip));
  updateRowMeta();
  if (flip && !reducedMotion && prev.size) {
    for (const c of rowEl.children) {
      const p = prev.get(c.dataset.id);
      if (!p) continue;
      const n = c.getBoundingClientRect();
      const dx = p.left - n.left, dy = p.top - n.top;
      if (dx || dy) c.animate(
        [{ transform: `translate(${dx}px, ${dy}px)` }, { transform: "none" }],
        { duration: 340, easing: "cubic-bezier(0.16, 1, 0.3, 1)" });
    }
  }
}

/* ---------- the bag (order-blind by construction) ---------- */

const bagTags = new Map(); // token id -> tag element inside the bag

function buildBag() {
  bagTags.clear();
  /* Sort by (text, id): the same multiset gives the same layout for any order */
  const sorted = [...order].sort((a, b) =>
    a.text === b.text ? a.id - b.id : a.text < b.text ? -1 : 1);
  const r = rng(101 + presetIdx * 7);
  const n = sorted.length;
  bagEl.replaceChildren(...sorted.map((t, i) => {
    const frac = Math.sqrt((i + 0.5) / n);
    const ang = i * 2.39996 + r() * 0.9;
    const x = 50 + frac * 34 * Math.cos(ang);
    const y = 52 + frac * 30 * Math.sin(ang);
    const rot = (r() - 0.5) * 38;
    const tag = el("span", { class: "pos-tag", "aria-hidden": "true" }, "");
    bagTags.set(t.id, tag);
    return el("span", {
      class: "token input",
      "aria-hidden": "true",
      style: `left:${x.toFixed(1)}%; top:${y.toFixed(1)}%;` +
        ` transform: translate(-50%, -50%) rotate(${rot.toFixed(1)}deg);`,
    }, t.text, tag);
  }));
}

function updateBag() {
  order.forEach((t, i) => { bagTags.get(t.id).textContent = i + 1; });
  bagEl.setAttribute("aria-label", tagsOn
    ? `Bag of words with position tags: ${order.map((t, i) => `${t.text} tagged ${i + 1}`).join(", ")}.`
    : `Unordered bag of words: ${order.map((t) => t.text).sort().join(", ")}. Order is not represented.`);
}

/* ---------- verdict ---------- */

function renderVerdict() {
  const isOriginal = order.every((t, i) => t.text === originalTexts[i]);
  const sig = tagsOn
    ? order.map((t, i) => `${t.text}#${i + 1}`).sort().join("  ")
    : order.map((t) => t.text).sort().join("  ");
  let call, cls;
  if (!tagsOn) {
    call = isOriginal
      ? "That line is everything raw attention would receive. Now reorder the sentence and watch it."
      : "You rewrote the sentence. The bag is identical, tile for tile. Attention cannot tell.";
    cls = isOriginal ? "" : "blind";
  } else {
    call = isOriginal
      ? "Tags on. Order now travels inside the bag itself."
      : "Different order, different bag. The two sentences are finally distinguishable.";
    cls = "aware";
  }
  verdictEl.replaceChildren(
    el("p", { class: "sig" }, "bag contents:  ", sig),
    el("p", { class: `call ${cls}`.trim() }, call),
  );
}

function orderChanged() {
  updateRowMeta();
  updateBag();
  renderVerdict();
}

/* ---------- pointer drag ---------- */

let drag = null;
let suppressClick = false;

function startDrag(e, chip) {
  if (!e.isPrimary || drag) return;
  drag = { chip, x0: e.clientX, y0: e.clientY, dx: 0, dy: 0, moved: false, pid: e.pointerId };
  chip.setPointerCapture(e.pointerId);
  chip.addEventListener("pointermove", moveDrag);
  chip.addEventListener("pointerup", endDrag);
  chip.addEventListener("pointercancel", endDrag);
}

function moveDrag(e) {
  if (!drag) return;
  drag.dx = e.clientX - drag.x0;
  drag.dy = e.clientY - drag.y0;
  if (!drag.moved && Math.hypot(drag.dx, drag.dy) > 4) {
    drag.moved = true;
    releaseGrab();
    drag.chip.classList.add("active");
    drag.chip.style.zIndex = "5";
  }
  drag.chip.style.transform = `translate(${drag.dx}px, ${drag.dy}px)`;
  if (!drag.moved) return;

  /* Nearest layout slot wins (offsetLeft/Top ignore transforms, so no jitter) */
  const rowRect = rowEl.getBoundingClientRect();
  const px = e.clientX - rowRect.left, py = e.clientY - rowRect.top;
  const chips = [...rowEl.children];
  let best = 0, bd = Infinity;
  chips.forEach((c, i) => {
    const cx = c.offsetLeft + c.offsetWidth / 2;
    const cy = c.offsetTop + c.offsetHeight / 2;
    const d = (cx - px) ** 2 + (cy - py) ** 2;
    if (d < bd) { bd = d; best = i; }
  });
  const cur = chips.indexOf(drag.chip);
  if (best !== cur) reorderDuringDrag(cur, best);
}

function reorderDuringDrag(cur, best) {
  const others = [...rowEl.children].filter((c) => c !== drag.chip);
  const before = new Map(others.map((c) => [c, { l: c.offsetLeft, t: c.offsetTop }]));
  const [tok] = order.splice(cur, 1);
  order.splice(best, 0, tok);

  const oldL = drag.chip.offsetLeft, oldT = drag.chip.offsetTop;
  rowEl.removeChild(drag.chip);
  rowEl.insertBefore(drag.chip, rowEl.children[best] ?? null);

  /* Keep the dragged chip visually pinned under the pointer */
  const shiftX = drag.chip.offsetLeft - oldL;
  const shiftY = drag.chip.offsetTop - oldT;
  drag.x0 += shiftX; drag.y0 += shiftY;
  drag.dx -= shiftX; drag.dy -= shiftY;
  drag.chip.style.transform = `translate(${drag.dx}px, ${drag.dy}px)`;

  if (!reducedMotion) {
    for (const c of others) {
      const p = before.get(c);
      const dx = p.l - c.offsetLeft, dy = p.t - c.offsetTop;
      if (dx || dy) c.animate(
        [{ transform: `translate(${dx}px, ${dy}px)` }, { transform: "none" }],
        { duration: 280, easing: "cubic-bezier(0.16, 1, 0.3, 1)" });
    }
  }
  orderChanged();
}

function endDrag() {
  if (!drag) return;
  const { chip, dx, dy, moved, pid } = drag;
  drag = null;
  try { chip.releasePointerCapture(pid); } catch { /* already released */ }
  chip.removeEventListener("pointermove", moveDrag);
  chip.removeEventListener("pointerup", endDrag);
  chip.removeEventListener("pointercancel", endDrag);
  chip.style.zIndex = "";
  chip.classList.remove("active");
  chip.style.transform = "";
  if (moved) {
    suppressClick = true;
    if (!reducedMotion && (dx || dy)) chip.animate(
      [{ transform: `translate(${dx}px, ${dy}px)` }, { transform: "none" }],
      { duration: 300, easing: "cubic-bezier(0.16, 1, 0.3, 1)" });
    const i = [...rowEl.children].indexOf(chip);
    announce(`${wordOf(+chip.dataset.id)} dropped at position ${i + 1} of ${order.length}.`);
  }
}

/* ---------- keyboard reorder ---------- */

let grabbedId = null;

function setGrab(chip, on) {
  chip.classList.toggle("active", on);
  chip.setAttribute("aria-pressed", String(on));
}

function releaseGrab() {
  grabbedId = null;
  [...rowEl.children].forEach((c) => setGrab(c, false));
}

function toggleGrab(chip) {
  const id = +chip.dataset.id;
  if (grabbedId === id) {
    releaseGrab();
    announce(`Dropped ${wordOf(id)}.`);
  } else {
    releaseGrab();
    grabbedId = id;
    setGrab(chip, true);
    announce(`Grabbed ${wordOf(id)}. Arrow keys to move, Enter to drop.`);
  }
}

function onChipKey(e, chip) {
  if (e.key === "Escape" && grabbedId != null) {
    releaseGrab();
    announce("Dropped.");
    return;
  }
  if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
  e.preventDefault();
  const chips = [...rowEl.children];
  const i = chips.indexOf(chip);
  const dir = e.key === "ArrowLeft" ? -1 : 1;
  if (grabbedId === +chip.dataset.id) {
    const j = clamp(i + dir, 0, order.length - 1);
    if (j === i) return;
    const [tok] = order.splice(i, 1);
    order.splice(j, 0, tok);
    renderRow();
    orderChanged();
    const moved = [...rowEl.children][j];
    setGrab(moved, true);
    moved.focus();
    announce(`${tok.text} moved to position ${j + 1} of ${order.length}.`);
  } else {
    chips[clamp(i + dir, 0, chips.length - 1)].focus();
  }
}

/* ---------- controls ---------- */

const presetSeg = document.getElementById("preset-seg");
function buildPresetSeg() {
  presetSeg.replaceChildren(...PRESETS.map((p, i) =>
    el("button", {
      "aria-pressed": String(i === presetIdx),
      onclick: () => {
        loadPreset(i);
        renderRow({ flip: false });
        buildBag();
        orderChanged();
        buildPresetSeg();
      },
    }, p.label)));
}

const tagBtn = document.querySelector("#tag-toggle button");
tagBtn.addEventListener("click", () => {
  tagsOn = !tagsOn;
  tagBtn.setAttribute("aria-pressed", String(tagsOn));
  orderHost.classList.toggle("tags-on", tagsOn);
  updateBag();
  renderVerdict();
  announce(tagsOn ? "Position tags on." : "Position tags off.");
});

document.getElementById("order-reset").addEventListener("click", () => {
  loadPreset(presetIdx);
  renderRow();
  buildBag();
  orderChanged();
  announce("Original order restored.");
});

loadPreset(0);
renderRow({ flip: false });
buildBag();
orderChanged();
buildPresetSeg();

/* ==========================================================================
   Demo 2: the rotation trick (RoPE)
   ========================================================================== */

const ropeHost = document.getElementById("rope-demo");
const R_WORDS = ["the", "cat", "sat", "on", "the", "mat"];
const COARSE = 22;  // degrees per position, slow dial
const FINE = 71;    // degrees per position, fast dial
const MAX_SHIFT = 18;

let shift = 0;
let selA = 1;  // "cat"
let selB = 5;  // "mat"
let ropeDirty = true;

const coarseS = R_WORDS.map((_, i) => spring(i * COARSE, 12));
const fineS = R_WORDS.map((_, i) => spring(i * FINE, 12));
const handAS = spring(selA * COARSE, 12);
const handBS = spring(selB * COARSE, 12);

function dialSvg(size, cls) {
  const c = size / 2, r = c - 4;
  const hand = svgEl("line", {
    x1: c, y1: c, x2: c, y2: c - r + 7,
    stroke: palette.cobalt, "stroke-width": 3, "stroke-linecap": "round",
  });
  const svg = svgEl("svg", {
    width: size, height: size, viewBox: `0 0 ${size} ${size}`,
    "aria-hidden": "true", class: cls || "",
  },
    svgEl("circle", { cx: c, cy: c, r, fill: palette.paper, stroke: palette.ink, "stroke-width": 2 }),
    svgEl("line", { x1: c, y1: 5, x2: c, y2: 11, stroke: palette.line, "stroke-width": 2 }),
    hand,
    svgEl("circle", { cx: c, cy: c, r: 3, fill: palette.ink }),
  );
  return { svg, hand, c };
}

const cards = R_WORDS.map((w, i) => {
  const coarse = dialSvg(72);
  const fine = dialSvg(44, "fine-dial");
  const deg = el("span", { class: "deg" }, "0°");
  const btn = el("button", {
    class: "dial-card", "data-i": i,
    onclick: () => selectToken(i),
  },
    el("span", { class: "sel-badge", "aria-hidden": "true" }, ""),
    el("span", { class: "dial-word" }, w),
    el("div", { class: "dials" }, coarse.svg, fine.svg),
    deg,
  );
  return { btn, coarse, fine, deg, w };
});

/* Difference dial: both selected hands plus the mint wedge between them */
const DIFF = 150, DC = DIFF / 2, DR = DC - 6, WEDGE_R = DR - 14;
const wedge = svgEl("path", { fill: palette.mint, opacity: 0.4, stroke: palette.mint, "stroke-width": 2, "stroke-linejoin": "round" });
const diffHandA = svgEl("line", { x1: DC, y1: DC, x2: DC, y2: DC - DR * 0.58, stroke: palette.ink, "stroke-width": 3.5, "stroke-linecap": "round" });
const diffHandB = svgEl("line", { x1: DC, y1: DC, x2: DC, y2: DC - DR + 9, stroke: palette.cobalt, "stroke-width": 3.5, "stroke-linecap": "round" });
const diffSvg = svgEl("svg", {
  width: DIFF, height: DIFF, viewBox: `0 0 ${DIFF} ${DIFF}`,
  role: "img", "aria-label": "Both selected dials overlaid; the shaded wedge is the angle between them.",
},
  svgEl("circle", { cx: DC, cy: DC, r: DR, fill: palette.paper, stroke: palette.ink, "stroke-width": 2 }),
  svgEl("line", { x1: DC, y1: 7, x2: DC, y2: 14, stroke: palette.line, "stroke-width": 2 }),
  wedge, diffHandA, diffHandB,
  svgEl("circle", { cx: DC, cy: DC, r: 3.5, fill: palette.ink }),
);

const deltaVal = el("span", { class: "delta-val" }, "");
const deltaCap = el("p", { class: "delta-caption" }, "");
const fineDelta = el("p", { class: "fine-delta" }, "");
const deltaBlock = el("div", { "aria-live": "polite" },
  el("p", { style: "font-size:0.88rem; font-weight:700; margin-bottom:0.45rem;" }, "the angle between the pair"),
  deltaVal, deltaCap, fineDelta,
);

const padWrap = el("span", { class: "pad-wrap", "aria-hidden": "true" });
const dialRow = el("div", {
  class: "dial-row", role: "group",
  "aria-label": "Sentence tokens, each with a position dial. Click two to compare.",
}, padWrap, ...cards.map((c) => c.btn));

const shiftVal = el("span", { class: "value" }, "+0 words");
const shiftRange = el("input", {
  type: "range", min: 0, max: MAX_SHIFT, step: 1, value: 0,
  "aria-label": "Invisible padding words prepended to the sentence",
  oninput: () => {
    shift = +shiftRange.value;
    shiftVal.textContent = `+${shift} word${shift === 1 ? "" : "s"}`;
    retargetRope();
  },
});
const controls = el("div", { class: "controls-row" },
  el("div", { class: "knob" },
    el("div", { class: "knob-label" }, el("span", {}, "shift the whole sentence"), shiftVal),
    shiftRange));

ropeHost.append(dialRow, el("div", { class: "diff-wrap" }, diffSvg, deltaBlock), controls);

function selectToken(i) {
  if (i === selA || i === selB) return;
  selA = selB;
  selB = i;
  if (selA > selB) [selA, selB] = [selB, selA];
  retargetRope();
}

function retargetRope() {
  cards.forEach((_, i) => {
    coarseS[i].target = (shift + i) * COARSE;
    fineS[i].target = (shift + i) * FINE;
  });
  handAS.target = (shift + selA) * COARSE;
  handBS.target = (shift + selB) * COARSE;

  padWrap.replaceChildren(
    ...(shift > 0 ? [el("span", { class: "pad-count" }, `+${shift}`)] : []),
    ...Array.from({ length: shift }, () => el("span", { class: "pad-chip" })));

  cards.forEach((cd, i) => {
    const sel = i === selA ? "A" : i === selB ? "B" : null;
    if (sel) {
      cd.btn.setAttribute("data-sel", sel);
      cd.btn.querySelector(".sel-badge").textContent = sel;
    } else {
      cd.btn.removeAttribute("data-sel");
    }
    cd.btn.setAttribute("aria-pressed", String(sel != null));
    cd.btn.setAttribute("aria-label",
      `${cd.w}, position ${shift + i + 1}, slow dial at ${Math.round(((shift + i) * COARSE) % 360)} degrees` +
      (sel ? `, selected as ${sel}` : "") + ".");
  });

  const dist = selB - selA;
  const d = dist * COARSE;
  const df = dist * FINE;
  deltaVal.textContent = `${d}°`;
  deltaCap.textContent =
    `between "${cards[selA].w}" and "${cards[selB].w}", ${dist} seat${dist === 1 ? "" : "s"} apart. ` +
    "Slide the shift: every hand moves, this number does not.";
  fineDelta.textContent = `fast dial gap: ${df % 360}°` + (df >= 360 ? ` (${df}° wound around)` : "");
  ropeDirty = true;
}

function drawDiff() {
  const aA = handAS.value, aB = handBS.value;
  const span = (((aB - aA) % 360) + 360) % 360;
  const D2R = Math.PI / 180;
  const p1x = DC + WEDGE_R * Math.sin(aA * D2R), p1y = DC - WEDGE_R * Math.cos(aA * D2R);
  const a2 = aA + span;
  const p2x = DC + WEDGE_R * Math.sin(a2 * D2R), p2y = DC - WEDGE_R * Math.cos(a2 * D2R);
  wedge.setAttribute("d",
    `M ${DC} ${DC} L ${p1x.toFixed(1)} ${p1y.toFixed(1)} ` +
    `A ${WEDGE_R} ${WEDGE_R} 0 ${span > 180 ? 1 : 0} 1 ${p2x.toFixed(1)} ${p2y.toFixed(1)} Z`);
  diffHandA.setAttribute("transform", `rotate(${(aA % 360).toFixed(2)} ${DC} ${DC})`);
  diffHandB.setAttribute("transform", `rotate(${(aB % 360).toFixed(2)} ${DC} ${DC})`);
}

const moreBtn = document.querySelector("#more-toggle button");
moreBtn.addEventListener("click", () => {
  const more = !ropeHost.classList.contains("more");
  ropeHost.classList.toggle("more", more);
  moreBtn.setAttribute("aria-pressed", String(more));
  ropeDirty = true;
});

retargetRope();

loop((dt) => {
  let moving = false;
  for (const s of [...coarseS, ...fineS, handAS, handBS]) {
    s.step(dt);
    if (s.value !== s.target) moving = true;
  }
  if (!ropeDirty && !moving) return;
  cards.forEach((cd, i) => {
    cd.coarse.hand.setAttribute("transform",
      `rotate(${(coarseS[i].value % 360).toFixed(2)} ${cd.coarse.c} ${cd.coarse.c})`);
    cd.fine.hand.setAttribute("transform",
      `rotate(${(fineS[i].value % 360).toFixed(2)} ${cd.fine.c} ${cd.fine.c})`);
    cd.deg.textContent = `${Math.round(coarseS[i].value % 360)}°`;
  });
  drawDiff();
  ropeDirty = moving;
}, ropeHost);
