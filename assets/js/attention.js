/* Attention chapter demos.
   Toy-sized but honest: logits come from small hand-labeled word features,
   the softmax, causal mask, and blending are computed for real. */

import { el, svgEl, softmax, loop, spring, palette, reducedMotion } from "./lib.js";

/* ---------- The toy sentence ---------- */

const ADJECTIVES = {
  slippery: { ball: 2.6, robot: 0.2 },
  clumsy: { ball: 0.2, robot: 2.6 },
};

let adjective = "slippery";

const makeTokens = () => [
  { text: "The", tags: ["det"] },
  { text: "robot", tags: ["noun", "agent"] },
  { text: "dropped", tags: ["verb"] },
  { text: "the", tags: ["det"] },
  { text: "ball", tags: ["noun", "object"] },
  { text: "because", tags: ["conj"] },
  { text: "it", tags: ["pron"] },
  { text: "was", tags: ["verb", "aux"] },
  { text: adjective, tags: ["adj"] },
];

const has = (t, tag) => t.tags.includes(tag);

/* Two heads with different personalities */
const HEADS = {
  reference: {
    label: "reference head",
    logit(q, k, dist) {
      if (has(q, "pron")) {
        let s = has(k, "noun") ? 2.2 : -1.5;
        if (has(k, "noun")) s += ADJECTIVES[adjective][k.text] ?? 0;
        return s;
      }
      if (has(q, "noun")) return (has(k, "det") && dist <= 2 ? 1.6 : 0) + (has(k, "verb") ? 1.1 : -0.6);
      if (has(q, "verb")) return has(k, "noun") ? 1.4 : -0.4;
      if (has(q, "adj")) return has(k, "pron") ? 1.8 : has(k, "aux") ? 0.8 : -0.6;
      return -0.4 * dist;
    },
  },
  nearby: {
    label: "nearby head",
    logit(_q, _k, dist) {
      return 2.0 - 0.85 * dist;
    },
  },
};

let head = "reference";

/* Causal attention row for query index qi (self included) */
function attentionRow(tokens, qi) {
  const logits = [];
  for (let j = 0; j <= qi; j++) {
    logits.push(j === qi ? 0.4 : HEADS[head].logit(tokens[qi], tokens[j], qi - j));
  }
  return softmax(logits);
}

/* ================= Hero demo ================= */

const hero = document.getElementById("hero-demo");
const ARC_H = 150;

let tokens = makeTokens();
let selected = 6; // "it"
let weights = tokens.map(() => spring(0, 10));

const arcSvg = svgEl("svg", {
  width: "100%",
  height: ARC_H,
  "aria-hidden": "true",
  style: "display:block; overflow:visible;",
});
const tokenRow = el("div", { class: "token-row", role: "group", "aria-label": "Sentence tokens" });
const blendWrap = el("div", { style: "margin-top:1.5rem;" });

hero.append(arcSvg, tokenRow, blendWrap);

function buildTokenRow() {
  tokenRow.replaceChildren(
    ...tokens.map((t, i) =>
      el("button", {
        class: "token",
        "aria-pressed": String(i === selected),
        onclick: () => { selected = i; retarget(); render(); },
      }, t.text)
    )
  );
}

function retarget() {
  const row = attentionRow(tokens, selected);
  weights.forEach((s, j) => { s.target = j <= selected && j !== selected ? row[j] : 0; });
}

function chipCenters() {
  const base = tokenRow.getBoundingClientRect();
  return [...tokenRow.children].map((c) => {
    const r = c.getBoundingClientRect();
    return { x: r.left - base.left + r.width / 2, y: r.top - base.top - 6 };
  });
}

function drawArcs() {
  const pts = chipCenters();
  arcSvg.replaceChildren();
  const q = pts[selected];
  for (let j = 0; j < selected; j++) {
    const w = weights[j].value;
    if (w < 0.005) continue;
    const t = pts[j];
    const lift = Math.min(30 + Math.abs(q.x - t.x) * 0.28, ARC_H - 14);
    arcSvg.append(svgEl("path", {
      d: `M ${q.x} ${ARC_H + q.y} Q ${(q.x + t.x) / 2} ${ARC_H - lift} ${t.x} ${ARC_H + t.y}`,
      fill: "none",
      stroke: palette.coral,
      "stroke-width": Math.max(1.2, w * 16),
      "stroke-linecap": "round",
      opacity: 0.25 + w * 0.75,
    }));
  }
}

function styleChips() {
  [...tokenRow.children].forEach((chip, i) => {
    chip.classList.toggle("active", i === selected);
    chip.classList.toggle("weight", i < selected && weights[i].value > 0.02);
    chip.style.opacity = i > selected ? 0.35 : 1;
    chip.setAttribute("aria-pressed", String(i === selected));
  });
}

function renderBlend() {
  const row = attentionRow(tokens, selected);
  const parts = row
    .map((w, j) => ({ w, j }))
    .sort((a, b) => b.w - a.w);
  const bar = el("div", {
    style: "display:flex; height:34px; border:2px solid var(--ink); border-radius:10px; overflow:hidden;",
    role: "img",
    "aria-label": `New meaning of “${tokens[selected].text}”: ` +
      parts.slice(0, 3).map((p) => `${Math.round(p.w * 100)}% ${tokens[p.j].text}`).join(", "),
  });
  row.forEach((w, j) => {
    if (w < 0.01) return;
    bar.append(el("div", {
      style: `flex:${w}; background:${j === selected ? "var(--marigold)" : "var(--coral)"};` +
        `opacity:${j === selected ? 1 : 0.35 + w * 0.65}; min-width:0; display:grid; place-items:center;` +
        "font-family:var(--font-mono); font-size:0.72rem; color:var(--ink); overflow:hidden; white-space:nowrap;",
    }, w > 0.09 ? tokens[j].text : ""));
  });
  blendWrap.replaceChildren(
    el("p", { style: "font-size:0.88rem; font-weight:700; margin-bottom:0.4rem;" },
      `the new “${tokens[selected].text}” is a blend of:`),
    bar,
    el("p", { class: "hint", style: "margin-top:0.5rem;" },
      parts.filter((p) => p.w > 0.04).slice(0, 4)
        .map((p) => `${Math.round(p.w * 100)}% ${p.j === selected ? "itself" : `“${tokens[p.j].text}”`}`)
        .join("  ·  ")),
  );
}

function render() {
  styleChips();
  renderBlend();
}

/* Adjective toggle + head switcher */
const adjToggle = document.getElementById("adj-toggle");
function buildToggles() {
  adjToggle.replaceChildren(
    ...Object.keys(ADJECTIVES).map((a) =>
      el("button", {
        "aria-pressed": String(a === adjective),
        onclick: () => {
          adjective = a;
          tokens = makeTokens();
          buildTokenRow();
          retarget();
          render();
          renderGrid();
          buildToggles();
        },
      }, a)
    )
  );
}

/* Head switcher lives under the arcs */
const headSeg = el("div", { class: "seg", role: "group", "aria-label": "Attention head", style: "margin-top:1.25rem;" });
function buildHeadSeg() {
  headSeg.replaceChildren(
    ...Object.entries(HEADS).map(([key, h]) =>
      el("button", {
        "aria-pressed": String(key === head),
        onclick: () => { head = key; retarget(); render(); renderGrid(); buildHeadSeg(); },
      }, h.label)
    )
  );
}
blendWrap.after(headSeg);

buildTokenRow();
buildToggles();
buildHeadSeg();
retarget();
render();

let dirty = true;
loop((dt) => {
  const before = weights.map((s) => s.value);
  weights.forEach((s) => s.step(dt));
  if (weights.some((s, i) => s.value !== before[i])) dirty = true;
  if (!dirty) return;
  drawArcs();
  styleChips();
  dirty = weights.some((s) => s.value !== s.target);
}, hero);

window.addEventListener("resize", () => { dirty = true; });

/* ================= Grid demo ================= */

const gridHost = document.getElementById("grid-demo");

function renderGrid() {
  const n = tokens.length;
  const rows = tokens.map((_, i) => attentionRow(tokens, i));
  const cell = "aspect-ratio:1; border-radius:4px; min-width:0;";
  const grid = el("div", {
    style: `display:grid; grid-template-columns: 5.5rem repeat(${n}, 1fr); gap:4px; max-width:640px; margin-inline:auto;`,
  });
  grid.append(el("div"));
  tokens.forEach((t) => grid.append(el("div", {
    style: "font-family:var(--font-mono); font-size:0.68rem; color:var(--ink-soft); text-align:center; overflow:hidden;",
  }, t.text)));
  rows.forEach((row, i) => {
    grid.append(el("div", {
      style: "font-family:var(--font-mono); font-size:0.78rem; align-self:center; text-align:right; padding-right:0.5rem;",
    }, tokens[i].text));
    for (let j = 0; j < n; j++) {
      const w = j <= i ? row[j] : null;
      grid.append(el("div", {
        class: "grid-cell",
        "data-row": i,
        style: cell + (w == null
          ? "background:repeating-linear-gradient(45deg, var(--paper-deep), var(--paper-deep) 3px, transparent 3px, transparent 7px);"
          : `background:${palette.coral}; opacity:${0.06 + w * 0.94}; transition:opacity var(--t-med) var(--ease-out), outline-color var(--t-fast);`),
        title: w == null ? `${tokens[i].text} can’t see ${tokens[j].text} (it comes later)` :
          `${tokens[i].text} → ${tokens[j].text}: ${Math.round(w * 100)}%`,
      }));
    }
  });
  grid.addEventListener("mouseover", (e) => {
    const r = e.target.getAttribute?.("data-row");
    [...grid.querySelectorAll(".grid-cell")].forEach((c) => {
      c.style.filter = r != null && c.getAttribute("data-row") !== r ? "grayscale(0.9) opacity(0.45)" : "";
    });
  });
  grid.addEventListener("mouseleave", () => {
    [...grid.querySelectorAll(".grid-cell")].forEach((c) => { c.style.filter = ""; });
  });
  gridHost.replaceChildren(grid);
}

document.getElementById("grid-reset").addEventListener("click", renderGrid);
renderGrid();
