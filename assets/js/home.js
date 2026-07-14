/* Homepage hero: a model's entire job on loop — predict, append, repeat. */

import { el, reducedMotion } from "./lib.js";

const PROMPT = ["The", "cat", "sat", "on", "the"];

const STEPS = [
  { candidates: [["mat", 0.61], ["sofa", 0.18], ["roof", 0.09], ["keyboard", 0.07], ["moon", 0.004]], pick: "mat" },
  { candidates: [["and", 0.41], [".", 0.28], ["all", 0.11], ["purring", 0.08], ["quietly", 0.05]], pick: "and" },
  { candidates: [["refused", 0.34], ["fell", 0.29], ["watched", 0.14], ["began", 0.09], ["knitted", 0.01]], pick: "refused" },
  { candidates: [["to", 0.92], ["all", 0.03], ["every", 0.02], ["politely", 0.01], ["the", 0.01]], pick: "to" },
  { candidates: [["move", 0.55], ["budge", 0.21], ["leave", 0.12], ["apologise", 0.04], ["explain", 0.02]], pick: "move" },
  { candidates: [[".", 0.71], ["again", 0.11], ["forever", 0.06], [",", 0.05], ["!", 0.03]], pick: "." },
];

const host = document.getElementById("hero-demo");
const promptRow = el("div", { class: "prompt-row", "aria-label": "Sentence so far" });
const barsHost = el("div", { class: "prob-bars", "aria-live": "polite" });
host.append(promptRow, barsHost);

let timer = null;
const wait = (ms) => new Promise((r) => { timer = setTimeout(r, reducedMotion ? 40 : ms); });

function chip(text, cls = "input") {
  return el("span", { class: `token ${cls}` }, text);
}

function showBars(candidates, winner = null) {
  barsHost.replaceChildren(
    ...candidates.map(([w, p]) => {
      const row = el("div", { class: "prob-bar" + (w === winner ? " winner" : "") },
        el("span", {}, w === "." ? "“.”" : w),
        el("div", { class: "bar" }, el("div", { class: "fill" })),
        el("span", { class: "pct" }, (p * 100).toFixed(p < 0.01 ? 1 : 0) + "%"),
      );
      requestAnimationFrame(() => requestAnimationFrame(() => {
        const fill = row.querySelector(".fill");
        fill.style.transition = reducedMotion ? "none" : "width 600ms cubic-bezier(0.16, 1, 0.3, 1)";
        fill.style.width = (p * 100) + "%";
      }));
      return row;
    })
  );
}

let runId = 0;
async function run() {
  const id = ++runId;
  clearTimeout(timer);
  promptRow.replaceChildren(...PROMPT.map((w) => chip(w)));
  barsHost.replaceChildren();
  await wait(900);
  for (const step of STEPS) {
    if (id !== runId) return;
    showBars(step.candidates);
    await wait(1400);
    if (id !== runId) return;
    showBars(step.candidates, step.pick);
    await wait(700);
    if (id !== runId) return;
    const c = chip(step.pick, "output");
    if (step.pick === ".") c.style.paddingInline = "0.45rem";
    promptRow.append(c);
    await wait(650);
  }
  if (id !== runId) return;
  await wait(2600);
  if (id === runId) run();
}

document.getElementById("hero-again").addEventListener("click", run);
run();
