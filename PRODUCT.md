# Product

## Register

brand

## Users

Smart general public: curious adults with no ML or math background who want to genuinely understand how LLMs work. They arrive from a shared link, on a laptop or phone, in a browsing mood. The job to be done: "make attention, MoE, sampling, etc. click for me through play, not prose." No code, no equations required; intuition is the deliverable.

## Product Purpose

Langsplain is an interactive explainer site for modern LLM architecture, published on GitHub Pages (langsplain repo, Cloudflare DNS already wired). Every chapter is built around a hands-on demo the reader plays with; text supports the toy, never the reverse. Success = a portfolio-grade reputation piece people share and link when someone asks "how do LLMs actually work?"

Scope (chapters): tokenization, embeddings, attention, MLP, stacking layers, next-token prediction, Mixture of Experts, RoPE, KV cache, quantization, pretraining, RLHF, hallucination, sampling (temperature / top-p / speculative decoding).

## Brand Personality

Playful, tactile, confident. A beautifully made toy shop for ideas: demos feel like physical objects you can't stop fiddling with. Smooth, elegant motion is a core brand trait. Adult playfulness: wit and color, never cartoonish.

## Anti-references

- Glowing-brain AI cliché: no neon neural nets on black, particle brains, sci-fi gradients.
- Academic paper dryness: no LaTeX walls, no Distill austerity.
- Corporate SaaS landing: no hero-CTA-testimonial template, no icon card grids.
- Kids' edutainment: no mascots, quiz confetti, or bouncy cartoon easing.

## Design Principles

1. **The demo is the explanation.** If the toy can show it, the text shouldn't say it. Every chapter leads with something manipulable.
2. **Touch first.** Every demo responds instantly to hover, drag, and tap; defaults are already interesting before the user touches anything.
3. **Motion teaches.** Animation exists to show cause and effect (a token flowing, a weight shifting), never as decoration.
4. **One idea per screen.** Each scene isolates a single concept; complexity accumulates across chapters, never within one.
5. **Craft is the credibility.** For a general audience with no equations, polish is what signals "this is trustworthy."

## Accessibility & Inclusion

Good defaults: WCAG AA contrast, keyboard-operable demos, prefers-reduced-motion honored (demos degrade to stepped states, not blankness), colorblind-safe encodings (never hue alone; use position/shape/label too).
