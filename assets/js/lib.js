/* Langsplain shared demo helpers */

export const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* Critically-damped spring toward a target. Returns an object you nudge via
   .target and read via .value inside a rAF loop. No overshoot. */
export function spring(initial, stiffness = 14) {
  return {
    value: initial,
    target: initial,
    step(dt) {
      if (reducedMotion) { this.value = this.target; return; }
      const k = 1 - Math.exp(-stiffness * dt);
      this.value += (this.target - this.value) * k;
      if (Math.abs(this.target - this.value) < 1e-4) this.value = this.target;
    },
  };
}

/* rAF loop with delta-time in seconds; auto-pauses when tab is hidden
   and when the element scrolls out of view (pass an element to observe). */
export function loop(fn, observeEl = null) {
  let last = null;
  let running = true;
  let visible = true;
  function frame(now) {
    if (!running) return;
    if (last == null) last = now;
    const dt = Math.min((now - last) / 1000, 0.05);
    last = now;
    if (visible) fn(dt);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) last = null;
  });
  if (observeEl) {
    new IntersectionObserver(([e]) => {
      visible = e.isIntersecting;
      if (visible) last = null;
    }).observe(observeEl);
  }
  return { stop() { running = false; } };
}

export const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
export const lerp = (a, b, t) => a + (b - a) * t;

export function softmax(xs, temperature = 1) {
  const t = Math.max(temperature, 1e-6);
  const m = Math.max(...xs);
  const exps = xs.map((x) => Math.exp((x - m) / t));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

export const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);

/* Deterministic PRNG so demos look the same for everyone (mulberry32) */
export function rng(seed = 1) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* el("div", {class: "x", onclick: fn}, child1, child2) */
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

export function svgEl(tag, attrs = {}, ...children) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  for (const c of children.flat()) node.append(c);
  return node;
}

/* Read a CSS custom property from :root (for canvas drawing) */
const rootStyle = getComputedStyle(document.documentElement);
export const cssVar = (name) => rootStyle.getPropertyValue(name).trim();

/* Palette roles for canvas/SVG demos */
export const palette = {
  ink: cssVar("--ink"),
  inkSoft: cssVar("--ink-soft"),
  line: cssVar("--line"),
  paper: cssVar("--paper"),
  coral: cssVar("--coral"),
  cobalt: cssVar("--cobalt"),
  marigold: cssVar("--marigold"),
  mint: cssVar("--mint"),
  coralTint: cssVar("--coral-tint"),
  cobaltTint: cssVar("--cobalt-tint"),
  marigoldTint: cssVar("--marigold-tint"),
  mintTint: cssVar("--mint-tint"),
};

/* Crisp canvas sizing for devicePixelRatio; returns ctx sized in CSS px */
export function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  return { ctx, w: rect.width, h: rect.height };
}
