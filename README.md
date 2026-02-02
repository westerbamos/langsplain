# Langsplain

An interactive educational web application that explains how modern decoder-only transformer LLMs work, with visual diagrams and hands-on demos.

## Features

- **Interactive Transformer Diagram**: Click on any component to learn how it works
- **Guided Tour**: Step-by-step walkthrough of the entire architecture
- **Attention Demo**: Visualize self-attention patterns with your own text
- **MOE Demo**: See how Mixture of Experts routing works
- **Sampling Demo**: Compare temperature, top-k, and top-p decoding behavior
- **KV Cache Demo**: Visualize why cache reuse speeds up autoregressive generation
- **Glossary**: Searchable reference of key terms
- **Responsive Design**: Works on desktop, tablet, and mobile

## Quick Start

### Option 1: Local Server (Recommended)

Since the app uses ES6 modules, you'll need a local server:

```bash
# Using Python 3
cd langsplain
python -m http.server 8000

# Using Node.js
npx serve .

# Using PHP
php -S localhost:8000
```

Then open http://localhost:8000 in your browser.

### Option 2: VS Code Live Server

If you use VS Code, install the "Live Server" extension and click "Go Live" in the status bar.

## Project Structure

```
langsplain/
├── index.html          # Main HTML structure
├── style.css           # All styles (dark mode, responsive)
├── app.js              # Main application logic
├── modules/
│   ├── diagram.js      # SVG diagram with D3.js
│   ├── attention-demo.js   # Attention visualization
│   ├── moe-demo.js     # MOE routing simulation
│   ├── sampling-demo.js    # Sampling strategies visualization
│   ├── kv-cache-demo.js    # KV cache visualization
│   ├── tour.js         # Guided tour system
│   ├── tokenizer.js    # BPE-style tokenization
│   └── math-utils.js   # Softmax, matrix operations
└── README.md
```

## Technology

- **Pure Vanilla JS** with ES6 modules
- **D3.js v7** for SVG diagram and data visualization
- **Anime.js** for smooth animations
- **No build step required** - just static files

## How It Works

### Toy Model Specifications

The demos use a simplified transformer for visualization:

- Embedding dimension: 64 (real models use 4096-8192)
- Attention heads: 4 per layer
- Layers: 3 (real models have 32-96)
- Vocabulary: ~200 common words + characters
- MOE experts: 8 with top-2 routing

### Key Concepts Covered

1. **Tokenization** - How text becomes tokens
2. **Embeddings** - Converting tokens to vectors
3. **Positional Encoding** - Adding position information
4. **Self-Attention** - Q, K, V and attention weights
5. **Multi-Head Attention** - Parallel attention patterns
6. **Feed-Forward Networks** - Per-token processing
7. **Residual Connections** - Skip connections
8. **Layer Normalization** - Stabilizing activations
9. **Mixture of Experts** - Sparse computation
10. **Output Generation** - Producing next tokens

## Deployment

### GitHub Pages

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-repo-url>
git push -u origin main
```

Enable GitHub Pages in repository settings → Pages → Source: main branch.

### Vercel

```bash
npx vercel --prod
```

### Netlify

Drag and drop the folder to Netlify, or use the CLI:

```bash
npx netlify deploy --prod
```

## Browser Support

- Chrome (recommended)
- Firefox
- Safari
- Edge

## Learning Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual guide
- [Switch Transformers](https://arxiv.org/abs/2101.03961) - MOE paper
- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) - Video course

## License

MIT License - feel free to use for educational purposes.
