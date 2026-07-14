# Langsplain

An interactive tour of how large language models work, at [langsplain.com](https://langsplain.com).

Nine chapters, each built around a demo you can play with: tokenization, embeddings, attention, MLP layers, sampling, positional encoding (RoPE), Mixture of Experts, inference speed tricks, and training. The demos are toy-sized but honest: they compute what they show.

Plain HTML, CSS, and ES modules. No build step. Served by GitHub Pages from the repo root; `PRODUCT.md` and `DESIGN.md` hold the design context. To develop locally, serve the root with any static server, e.g. `python3 -m http.server`.
