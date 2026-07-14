# Design

Seed DESIGN.md (pre-implementation). Re-run `$impeccable document` after build to capture real tokens.

## Theme

Light. Scene: a curious adult on a laptop in a bright room, mid-browse, ready to play. A warm paper-white surface makes the demos' saturated colors carry the energy, and firmly rejects the dark "AI aesthetic."

## Color

Strategy: **full palette** (4 named roles used deliberately) over warm tinted neutrals. All OKLCH, no pure black/white.

- `--paper`: oklch(0.97 0.008 85) — warm paper background
- `--ink`: oklch(0.24 0.015 75) — warm near-black text
- `--coral`: oklch(0.66 0.19 25) — primary action / focus token
- `--cobalt`: oklch(0.52 0.19 262) — structure, links, query-side of demos
- `--marigold`: oklch(0.8 0.16 80) — highlights, active states
- `--mint`: oklch(0.72 0.13 165) — success, output-side of demos
- Tints of each role (oklch L≈0.93, C≈0.04) for demo surfaces and chips.

In demos, color encodes meaning consistently across the whole site: cobalt = input/query, coral = attention/weight, mint = output/prediction, marigold = the thing currently under the user's finger. Never hue alone: pair with labels/position.

## Typography

- Display: **Bricolage Grotesque** (Google Fonts) — characterful, adult-playful. Big sizes, tight leading, weight 700–800.
- Body: **Hanken Grotesk** — warm, clean, 1.6 line-height, max 68ch.
- Mono (tokens, logits, numbers): **Spline Sans Mono**.
- Scale ratio ≥1.3; hierarchy via large jumps in size and weight.

## Shape & Surfaces

Toy-like through geometry, not cartoons: generous radii (12–20px) on interactive objects, chunky 2px ink-colored borders on manipulable elements (signals "this is a physical thing you can grab"), soft single-direction shadows only on lifted/dragged items. No glassmorphism, no side-stripes, no gradient text.

## Motion

Smooth and elegant: exponential ease-outs (cubic-bezier(0.16, 1, 0.3, 1)), 250–600ms. FLIP-style position transitions for tokens moving between stages. Continuous demo animation via requestAnimationFrame with spring-toward-target (critically damped, no overshoot). `prefers-reduced-motion`: animations become instant state changes with stepped controls.

## Layout

Full-bleed demo scenes alternating with narrow prose columns; the page breathes between chapters. Spacing rhythm varies deliberately: dense inside demos, airy between sections. No card grids.

## Components

- **Token chip**: rounded rect, mono type, ink border; the atomic visual unit of the whole site.
- **Demo stage**: full-width bordered canvas/SVG region with a title strip and a reset control.
- **Knob/slider**: chunky custom range inputs with big touch targets and live value readouts.
- **Chapter nav**: numbered journey rail, progress persists via URL hash.
