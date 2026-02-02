# Langsplain - Context for Claude Code

## Project Overview

Langsplain is an interactive educational web application that teaches how modern decoder-only transformer LLMs work. It uses pure vanilla JavaScript with ES6 modules, D3.js for visualizations, and Anime.js for animations.

**Live Demo**: https://langsplain.com

## Architecture

### Tech Stack
- **Pure Vanilla JS** - No frameworks, ES6 modules only
- **D3.js v7** - SVG diagram rendering and data visualization
- **Anime.js** - Smooth UI animations
- **Static files only** - No build step required

### File Structure
```
langsplain/
├── index.html          # Main HTML structure & UI components
├── style.css           # All styles (dark mode, responsive design)
├── app.js              # Main application logic & event handlers
└── modules/
    ├── diagram.js      # Interactive SVG transformer diagram with D3.js
    ├── attention-demo.js   # Self-attention visualization
    ├── moe-demo.js     # Mixture of Experts routing simulation
    ├── sampling-demo.js    # Sampling strategies visualization
    ├── kv-cache-demo.js    # KV cache visualization
    ├── tour.js         # Guided tour system
    ├── tokenizer.js    # BPE-style tokenization demo
    └── math-utils.js   # Softmax, matrix operations, utilities
```

## Educational Model Specifications

The demos use a simplified transformer model for visualization purposes:
- **Embedding dimension**: 64 (real models: 4096-8192)
- **Attention heads**: 4 per layer (multi-head attention)
- **Layers**: 3 (real models: 32-96)
- **Vocabulary**: ~200 common words + characters
- **MOE experts**: 8 with top-2 routing

## Key Features

1. **Interactive Transformer Diagram** - Clickable components with explanations
2. **Guided Tour** - Step-by-step architecture walkthrough
3. **Attention Demo** - Visualize self-attention with user input
4. **MOE Demo** - Show Mixture of Experts routing decisions
5. **Sampling Demo** - Explore temperature, top-k, and top-p decoding
6. **KV Cache Demo** - Compare cached vs uncached generation cost
7. **Glossary** - Searchable ML/AI terminology reference
8. **Responsive Design** - Works on mobile, tablet, desktop

## Development Guidelines

### Code Style
- Use ES6 module syntax (`import`/`export`)
- Prefer `const` over `let`, avoid `var`
- Use descriptive variable names (e.g., `attentionWeights` not `aw`)
- Keep functions small and focused
- Add comments for complex mathematical operations

### Educational Focus
- Prioritize **clarity** over performance
- Visualizations should be **intuitive** and **accurate**
- Use **simplified versions** of concepts (note the simplifications)
- Provide **contextual explanations** in tooltips/modals
- Link to academic papers and learning resources

### ES6 Modules
Since the app uses ES6 modules, it **requires a local server**:
```bash
python -m http.server 8000
# OR
npx serve .
```

### No Build Step Philosophy
- Keep everything browser-native
- Avoid adding npm dependencies or build tools
- Use CDN for external libraries (D3.js, Anime.js)
- Maintain simplicity for educational transparency

## Common Tasks

### Adding a New Interactive Demo
1. Create new module in `modules/` (e.g., `modules/new-demo.js`)
2. Export initialization function
3. Import in `app.js` and wire up event handlers
4. Add UI controls in `index.html`
5. Style in `style.css` (use existing CSS custom properties)

### Modifying the Diagram
- Edit `modules/diagram.js`
- Use D3.js for SVG manipulation
- Follow existing pattern for interactive elements
- Update tooltips/explanations for clarity

### Adding Mathematical Visualizations
- Use `modules/math-utils.js` for operations (softmax, dot products, etc.)
- Keep calculations simple and well-commented
- Show intermediate steps for educational value
- Use visual representations (matrices, heatmaps)

## Important Considerations

### Educational Accuracy
- When simplifying concepts, note the simplification
- Link to authoritative sources (papers, blog posts)
- Avoid misleading analogies
- Keep terminology consistent with research literature

### Performance
- Limit demo sizes for smooth animations
- Use requestAnimationFrame for rendering
- Throttle expensive calculations (attention computation)
- Profile with browser DevTools for large visualizations

### Accessibility
- Maintain keyboard navigation support
- Use semantic HTML
- Ensure sufficient color contrast
- Provide text alternatives for visualizations

### Browser Compatibility
- Test on Chrome, Firefox, Safari, Edge
- Use standard Web APIs (no experimental features)
- Fallback for older browsers where needed

## Testing

### Manual Testing Checklist
- [ ] Interactive diagram components clickable
- [ ] Guided tour progresses correctly
- [ ] Attention demo works with various inputs
- [ ] MOE demo shows routing decisions
- [ ] Sampling demo updates distributions correctly
- [ ] KV cache demo updates counters/savings correctly
- [ ] Glossary search functional
- [ ] Dark mode toggles properly
- [ ] Responsive on mobile/tablet/desktop
- [ ] All animations smooth (60fps)

### Local Testing
```bash
cd langsplain
python -m http.server 8000
# Visit http://localhost:8000
```

## Deployment

Currently deployed on GitHub Pages. To deploy updates:
```bash
git add .
git commit -m "Description of changes"
git push origin main
```

GitHub Pages automatically rebuilds from the main branch.

## Learning Resources Referenced

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual guide
- [Switch Transformers](https://arxiv.org/abs/2101.03961) - MOE architecture
- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) - Karpathy's course

## Questions to Consider When Making Changes

1. Does this change make the concept **clearer** or more confusing?
2. Is the visualization **accurate** to how real transformers work?
3. Does it maintain the **no-build-step** philosophy?
4. Will it work on **mobile devices**?
5. Is the performance acceptable (smooth animations)?
6. Are explanations at the right level for the target audience?

## Target Audience

- CS students learning ML/AI
- Developers new to transformer architecture
- Technical professionals wanting hands-on understanding
- Anyone curious about how ChatGPT/GPT-4 works under the hood

Educational level: Assumes basic programming knowledge, explains ML concepts from first principles.
