/**
 * Langsplain - Main Application
 * Event handlers, initialization, and view management
 */

import { initDiagram, highlightComponent, clearHighlight, toggleMOE, animateTokenFlow } from './modules/diagram.js';
import { initTour, startTour, exitTour } from './modules/tour.js';
import { AttentionDemoUI } from './modules/attention-demo.js';
import { MOEDemoUI } from './modules/moe-demo.js';
import { SamplingDemoUI } from './modules/sampling-demo.js';
import { KVCacheDemoUI } from './modules/kv-cache-demo.js';
import { tokenize, simulateBPE, getTokenColor } from './modules/tokenizer.js';

// Content for info panel
const INFO_CONTENT = {
    tokenization: {
        title: 'Tokenization',
        simple: `
            <p>Before an LLM can process text, it must be broken into smaller pieces called <strong>tokens</strong>. Think of tokens as the "words" the model understands.</p>
            <p>Common words like "the" or "cat" often become single tokens. Longer or rarer words get split into smaller pieces - "understanding" might become "under" + "standing".</p>
            <p>Most modern LLMs use around 50,000-100,000 different tokens in their vocabulary.</p>
            <div class="bpe-demo-section">
                <h4>Try BPE Tokenization</h4>
                <p class="bpe-note">This is a <strong>simplified simulation</strong> with ~25 common merge rules.
                   Real tokenizers have 30,000-100,000 rules.</p>
                <div class="bpe-suggestions">
                    <span>Try:</span>
                    <button class="bpe-suggest-btn" data-word="the">"the"</button>
                    <button class="bpe-suggest-btn" data-word="cat">"cat"</button>
                    <button class="bpe-suggest-btn" data-word="standing">"standing"</button>
                    <button class="bpe-suggest-btn" data-word="xyz">"xyz"</button>
                </div>
                <div class="bpe-input-row">
                    <input type="text" id="bpe-input" placeholder="Type a word..." value="the" maxlength="30">
                    <button id="bpe-run-btn" class="secondary-btn">Tokenize</button>
                </div>
                <div id="bpe-visualization" class="bpe-visualization"></div>
            </div>
        `,
        technical: `
            <p>Modern LLMs use <strong>Byte Pair Encoding (BPE)</strong> or similar subword tokenization algorithms:</p>
            <div class="formula">
                vocabulary = merge_frequent_pairs(base_characters)
            </div>
            <p>Key properties:</p>
            <ul>
                <li>Handles unknown words by splitting into known subwords</li>
                <li>Balances vocabulary size vs sequence length</li>
                <li>Typical vocab sizes: GPT-4 ~100k, LLaMA ~32k</li>
            </ul>
            <pre><code>// Simplified tokenization
"Hello world" → ["Hello", " world"]
"unbelievable" → ["un", "believ", "able"]</code></pre>
            <p><strong>BPE Algorithm Steps:</strong></p>
            <ol>
                <li>Start with character-level tokens</li>
                <li>Count all adjacent pairs</li>
                <li>Merge most frequent pair into new token</li>
                <li>Repeat until vocabulary size reached</li>
            </ol>
        `
    },
    embeddings: {
        title: 'Token Embeddings',
        simple: `
            <p>Each token is converted into a list of numbers called an <strong>embedding vector</strong>. These numbers capture the meaning and relationships between words.</p>
            <p>Similar words (like "cat" and "dog") will have similar number patterns. The model learns these representations during training.</p>
            <p><strong>Positional encoding</strong> is added so the model knows word order - otherwise "cat sat on mat" would look the same as "mat sat on cat"!</p>
        `,
        technical: `
            <p>Token embeddings are learned lookup tables of dimension <code>d_model</code>:</p>
            <div class="formula">
                E(token) ∈ ℝ^d_model (typically 768-8192)
            </div>
            <p><strong>Positional Encoding</strong> variants:</p>
            <ul>
                <li><strong>Sinusoidal:</strong> PE(pos, 2i) = sin(pos/10000^(2i/d))</li>
                <li><strong>RoPE:</strong> Rotary position embeddings multiply Q,K by rotation matrices</li>
                <li><strong>ALiBi:</strong> Adds position-based bias to attention scores</li>
            </ul>
            <pre><code>embedding = token_embed[token_id] + pos_embed[position]</code></pre>
        `
    },
    attention: {
        title: 'Self-Attention',
        simple: `
            <p>Self-attention is the magic that lets transformers understand context. Each word "looks at" all the other words to figure out what's important.</p>
            <p>For example, in "The cat sat on the mat because it was tired", attention helps the model understand that "it" refers to "cat".</p>
            <p>The model learns three things for each word: what it's looking for (Query), what it has to offer (Key), and its actual content (Value).</p>
        `,
        technical: `
            <p><strong>Scaled Dot-Product Attention:</strong></p>
            <div class="formula">
                Attention(Q, K, V) = softmax(QK^T / √d_k) · V
            </div>
            <p>For each token:</p>
            <ul>
                <li><strong>Query (Q):</strong> What information am I looking for?</li>
                <li><strong>Key (K):</strong> What information do I contain?</li>
                <li><strong>Value (V):</strong> What do I actually output?</li>
            </ul>
            <p><strong>Causal masking</strong> prevents attending to future tokens (autoregressive generation).</p>
            <pre><code>Q = X @ W_q  # Project to queries
K = X @ W_k  # Project to keys
V = X @ W_v  # Project to values
scores = Q @ K.T / sqrt(d_k)
weights = softmax(mask(scores))
output = weights @ V</code></pre>
        `
    },
    multihead: {
        title: 'Multi-Head Attention',
        simple: `
            <p>Instead of doing attention once, transformers do it multiple times in parallel with different "heads". Each head can focus on different types of relationships.</p>
            <p>One head might focus on grammar, another on word meaning, and another on long-range dependencies. The results are combined at the end.</p>
            <p>GPT-3 uses 96 attention heads per layer!</p>
        `,
        technical: `
            <div class="formula">
                MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
            </div>
            <p>Each head has separate projections:</p>
            <div class="formula">
                head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
            </div>
            <p>With <code>d_k = d_model / num_heads</code>, total parameters stay constant.</p>
            <p>Typical configurations:</p>
            <ul>
                <li>GPT-3 175B: 96 heads, d_model=12288</li>
                <li>LLaMA-7B: 32 heads, d_model=4096</li>
            </ul>
        `
    },
    residuals: {
        title: 'Residual + LayerNorm',
        simple: `
            <p>Residual connections are "shortcuts" that add the input directly to the output of each layer. Think of it as saying "keep what you had, plus add this new information".</p>
            <p>Layer normalization (LayerNorm) is usually paired with these residual paths to keep activations in a stable range as signals move through many layers.</p>
            <p>Together, they help gradients flow during training and let the model preserve useful information across deep stacks.</p>
        `,
        technical: `
            <p>Modern decoder-only LLMs usually use <strong>pre-norm</strong> blocks:</p>
            <div class="formula">y = x + Sublayer(LayerNorm(x))</div>
            <p>The original Transformer paper used <strong>post-norm</strong>:</p>
            <div class="formula">y = LayerNorm(x + Sublayer(x))</div>
            <p>Benefits:</p>
            <ul>
                <li><strong>Gradient flow:</strong> Prevents vanishing gradients in deep networks</li>
                <li><strong>Identity mapping:</strong> Easy for layers to learn identity function</li>
                <li><strong>Activation stability:</strong> LayerNorm reduces internal scale drift</li>
            </ul>
            <p><strong>Pre-norm vs Post-norm:</strong></p>
            <pre><code>// Pre-norm (common in modern LLMs)
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))

// Post-norm (original Transformer)
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))</code></pre>
        `
    },
    ffn: {
        title: 'Feed-Forward Network',
        simple: `
            <p>After attention, each token passes through a feed-forward network (FFN) independently. This is a simple neural network that transforms each token.</p>
            <p>Research suggests that FFN layers act like a "memory" - storing factual knowledge about the world that the model learned during training.</p>
            <p>The FFN typically expands the dimension by 4x, then compresses it back down.</p>
        `,
        technical: `
            <div class="formula">
                FFN(x) = W_2 · σ(W_1 · x + b_1) + b_2
            </div>
            <p>Typical architecture:</p>
            <ul>
                <li><strong>Expansion:</strong> d_model → 4×d_model</li>
                <li><strong>Activation:</strong> GELU or SiLU/Swish</li>
                <li><strong>Projection:</strong> 4×d_model → d_model</li>
            </ul>
            <p>SwiGLU variant (LLaMA, PaLM):</p>
            <div class="formula">
                FFN(x) = (Swish(x·W_1) ⊙ (x·V)) · W_2
            </div>
            <p>FFN parameters often dominate total model size (>60%).</p>
        `
    },
    moe: {
        title: 'Mixture of Experts',
        simple: `
            <p>Instead of one big feed-forward network, MOE uses many smaller "expert" networks. A router decides which experts to use for each token.</p>
            <p>This allows models to be much larger while keeping computation manageable - only a few experts are active for each token.</p>
            <p>Mixtral uses 8 experts but only activates 2 per token, achieving better performance than dense models of similar compute cost.</p>
            <p><em>Note:</em> in this demo, expert labels and routing cues are intentionally simplified so behavior is easier to see.</p>
        `,
        technical: `
            <div class="formula">
                MOE(x) = Σ G(x)_i · E_i(x)
            </div>
            <p>Where G(x) is the gating/router function:</p>
            <div class="formula">
                G(x) = TopK(softmax(x · W_g))
            </div>
            <p>Key considerations:</p>
            <ul>
                <li><strong>Load balancing:</strong> Auxiliary loss to prevent expert collapse</li>
                <li><strong>Expert capacity:</strong> Maximum tokens per expert per batch</li>
                <li><strong>Routing strategies:</strong> Top-k, expert choice, or hash-based</li>
            </ul>
            <p>In production MOE models, experts are not hand-labeled by topic; specialization emerges from training.</p>
            <p>Example: Mixtral 8x7B has 46.7B total params but only 12.9B active.</p>
        `
    },
    outputProjection: {
        title: 'Output Projection',
        simple: `
            <p>The final layer converts the model's internal representation back to vocabulary space. It produces a score for every possible next token.</p>
            <p>Higher scores mean the model thinks that token is more likely to come next. These scores are converted to probabilities using softmax.</p>
        `,
        technical: `
            <div class="formula">
                logits = h_final · W_vocab^T
            </div>
            <p>W_vocab is often tied with input embeddings (weight tying) to reduce parameters.</p>
            <div class="formula">
                P(next_token) = softmax(logits / temperature)
            </div>
            <p>Output dimension equals vocabulary size (typically 32k-100k).</p>
        `
    },
    generation: {
        title: 'Token Generation',
        simple: `
            <p>LLMs generate text one token at a time. After predicting a token, it's added to the input, and the process repeats.</p>
            <p>Various sampling strategies control creativity: temperature adjusts randomness, top-k limits choices to the most likely tokens, and top-p (nucleus) samples from tokens covering a probability threshold.</p>
        `,
        technical: `
            <p><strong>Autoregressive generation:</strong></p>
            <div class="formula">
                P(text) = Π P(token_i | token_1...token_{i-1})
            </div>
            <p>Sampling strategies:</p>
            <ul>
                <li><strong>Greedy:</strong> Always pick highest probability</li>
                <li><strong>Temperature:</strong> logits / T (higher = more random)</li>
                <li><strong>Top-k:</strong> Sample from k most likely tokens</li>
                <li><strong>Top-p (nucleus):</strong> Sample from smallest set with cumulative prob > p</li>
            </ul>
            <p><strong>KV Caching:</strong> Store computed K,V for previous tokens to avoid recomputation.</p>
        `
    }
};

// Glossary terms
const GLOSSARY = [
    { term: 'Attention', definition: 'A mechanism that allows each position in a sequence to selectively focus on and aggregate information from other positions, weighted by learned relevance scores.' },
    { term: 'Autoregressive', definition: 'A generation approach where each output token depends only on previously generated tokens, predicting one token at a time from left to right.' },
    { term: 'BPE (Byte Pair Encoding)', definition: 'A tokenization algorithm that iteratively merges the most frequent pairs of bytes/characters to build a vocabulary of subword units.' },
    { term: 'Causal Mask', definition: 'A triangular mask applied to attention scores that prevents tokens from attending to future positions, ensuring autoregressive generation.' },
    { term: 'Decoder-only', definition: 'A transformer architecture (like GPT) that uses only the decoder stack with causal masking, as opposed to encoder-decoder models.' },
    { term: 'Embedding', definition: 'A dense vector representation of a token that captures semantic meaning in a continuous space, typically 768-8192 dimensions.' },
    { term: 'Expert (in MOE)', definition: 'A feed-forward sub-network within a Mixture of Experts layer that processes a subset of tokens selected by the router; experts can specialize during training.' },
    { term: 'Feed-Forward Network', definition: 'A two-layer neural network applied independently to each position, typically expanding then contracting the hidden dimension.' },
    { term: 'GELU', definition: 'Gaussian Error Linear Unit, a smooth activation function commonly used in transformers: GELU(x) = x · Φ(x).' },
    { term: 'KV Cache', definition: 'A memory optimization that stores previously computed Key and Value tensors during generation to avoid redundant computation.' },
    { term: 'Layer Normalization', definition: 'A normalization technique that standardizes activations across the feature dimension, stabilizing training.' },
    { term: 'Logits', definition: 'Raw, unnormalized scores output by the model before applying softmax to get probabilities.' },
    { term: 'Mixture of Experts (MOE)', definition: 'An architecture using multiple expert networks with a learned router that selects which experts process each token.' },
    { term: 'Multi-Head Attention', definition: 'Running multiple attention operations in parallel with different learned projections, then concatenating results.' },
    { term: 'Positional Encoding', definition: 'Information added to embeddings to indicate token positions, since attention is permutation-invariant.' },
    { term: 'Query, Key, Value', definition: 'The three projections in attention: Query asks "what am I looking for?", Key answers "what do I contain?", Value provides "what do I output?".' },
    { term: 'Residual Connection', definition: 'A skip connection that adds the layer input to its output: y = x + f(x), enabling gradient flow in deep networks.' },
    { term: 'RoPE', definition: 'Rotary Position Embedding, a method encoding positions by rotating query and key vectors based on their position.' },
    { term: 'Softmax', definition: 'A function that converts logits to a probability distribution: softmax(x)_i = exp(x_i) / Σexp(x_j).' },
    { term: 'Temperature', definition: 'A hyperparameter that controls randomness in sampling by scaling logits before softmax. Higher = more random.' },
    { term: 'Token', definition: 'The basic unit of text that LLMs process, typically a word, subword, or character depending on the tokenizer.' },
    { term: 'Top-k Sampling', definition: 'A decoding strategy that samples only from the k most probable tokens at each step.' },
    { term: 'Top-p (Nucleus) Sampling', definition: 'A decoding strategy that samples from the smallest set of tokens whose cumulative probability exceeds p.' },
    { term: 'Transformer Block', definition: 'A repeating unit containing self-attention and feed-forward layers with residual connections and normalization.' }
];

// Application state
let currentView = 'home';
let infoPanelOpen = false;
let currentInfoKey = null;
let attentionDemo = null;
let moeDemo = null;
let samplingDemo = null;
let kvCacheDemo = null;

/**
 * Initialize the application
 */
function init() {
    // Initialize diagram
    initDiagram('diagram-container', handleComponentClick);

    // Initialize tour
    initTour();

    // Initialize demos
    attentionDemo = new AttentionDemoUI('attention-demo-content');
    moeDemo = new MOEDemoUI('moe-demo-content');
    samplingDemo = new SamplingDemoUI('sampling-demo-content');
    kvCacheDemo = new KVCacheDemoUI('kvcache-demo-content');

    // Setup event listeners
    setupNavigation();
    setupPanelEvents();
    setupDemoButtons();
    setupTourButton();
    setupModalEvents();
    setupGlossary();

    // Listen for tour MOE toggle event
    window.addEventListener('tour:showMOE', () => {
        toggleMOE();
    });

    // Handle window resize
    window.addEventListener('resize', handleResize);
}

/**
 * Setup navigation events
 */
function setupNavigation() {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const view = link.dataset.view;
            switchView(view);
        });
    });
}

/**
 * Switch between views
 */
function switchView(view) {
    currentView = view;

    // Update nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.toggle('active', link.dataset.view === view);
    });

    // Update view visibility
    document.querySelectorAll('.view').forEach(v => {
        v.classList.toggle('active', v.id === `${view}-view`);
    });

    // Close panel when switching views
    closeInfoPanel();
}

/**
 * Handle component click in diagram
 */
function handleComponentClick(infoKey, component) {
    if (infoKey && INFO_CONTENT[infoKey]) {
        openInfoPanel(infoKey);
    }
}

/**
 * Open info panel with content
 */
function openInfoPanel(infoKey) {
    const content = INFO_CONTENT[infoKey];
    if (!content) return;

    currentInfoKey = infoKey;
    infoPanelOpen = true;

    const panel = document.getElementById('info-panel');
    const title = panel.querySelector('.panel-title');
    const simple = panel.querySelector('.simple-explanation');
    const technical = panel.querySelector('.technical-content');

    title.textContent = content.title;
    simple.innerHTML = content.simple;
    technical.innerHTML = content.technical;

    panel.classList.add('open');
    document.querySelector('.diagram-section').classList.add('panel-open');

    highlightComponent(infoKey);

    // Setup BPE demo if tokenization panel
    if (infoKey === 'tokenization') {
        setupBPEDemo();
    }
}

/**
 * Setup BPE visualization demo in tokenization panel
 */
function setupBPEDemo() {
    const runBtn = document.getElementById('bpe-run-btn');
    const input = document.getElementById('bpe-input');

    if (!runBtn || !input) return;

    const runBPE = () => {
        const text = input.value.trim();
        if (!text) return;

        const result = simulateBPE(text);
        renderBPEVisualization(result);
    };

    runBtn.addEventListener('click', runBPE);
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') runBPE();
    });

    // Setup suggestion button handlers
    document.querySelectorAll('.bpe-suggest-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const word = btn.dataset.word;
            input.value = word;
            runBPE();
        });
    });

    // Run with default value
    runBPE();
}

/**
 * Render BPE merge visualization
 */
function renderBPEVisualization(result) {
    const container = document.getElementById('bpe-visualization');
    if (!container) return;

    let html = '<div class="bpe-steps">';

    result.mergeSteps.forEach((step, i) => {
        const isLast = i === result.mergeSteps.length - 1;

        html += `
            <div class="bpe-step ${isLast ? 'final' : ''}">
                <div class="step-header">
                    <span class="step-number">${step.step === 0 ? 'Start' : `Step ${step.step}`}</span>
                    ${step.frequency ? `<span class="step-freq">freq: ${step.frequency}</span>` : ''}
                </div>
                <div class="step-tokens">
                    ${step.tokens.map((token, j) => {
            const isNew = step.mergedIndex !== undefined && j === step.mergedIndex;
            const depth = token.length > 1 ? Math.min(token.length, 5) : 0;
            const color = getTokenColor(depth);
            return `<span class="bpe-token ${isNew ? 'merged' : ''}" style="border-color: ${color}">${token}</span>`;
        }).join('<span class="token-separator">+</span>')}
                </div>
                ${step.description && step.step > 0 ? `<div class="step-description">${step.description}</div>` : ''}
            </div>
        `;

        if (!isLast) {
            html += '<div class="step-arrow">↓</div>';
        }
    });

    html += '</div>';

    // Add final token count with contextual message
    if (result.totalMerges === 0) {
        html += `
            <div class="bpe-summary">
                <strong>${result.finalTokens.length}</strong> final tokens from
                <strong>${result.input.length}</strong> characters
                (${result.totalMerges} merges)
                <div style="margin-top: 8px; color: var(--accent-orange); font-style: italic;">
                    No merge rules matched - in a real tokenizer, this word might still merge!
                </div>
            </div>
        `;
    } else {
        html += `
            <div class="bpe-summary">
                <strong>${result.finalTokens.length}</strong> final tokens from
                <strong>${result.input.length}</strong> characters
                (${result.totalMerges} merges)
            </div>
        `;
    }

    container.innerHTML = html;
}

/**
 * Close info panel
 */
function closeInfoPanel() {
    infoPanelOpen = false;
    currentInfoKey = null;

    document.getElementById('info-panel').classList.remove('open');
    document.querySelector('.diagram-section').classList.remove('panel-open');

    clearHighlight();
}

/**
 * Setup panel events
 */
function setupPanelEvents() {
    const closeBtn = document.querySelector('.panel-close');
    closeBtn.addEventListener('click', closeInfoPanel);

    // Close on escape
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && infoPanelOpen) {
            closeInfoPanel();
        }
    });
}

/**
 * Setup demo button events
 */
function setupDemoButtons() {
    // Attention demo button
    document.querySelector('[data-demo="attention"]').addEventListener('click', () => {
        openModal('attention-modal');
        attentionDemo.init();
    });

    // MOE demo button
    document.querySelector('[data-demo="moe"]').addEventListener('click', () => {
        openModal('moe-modal');
        moeDemo.init();
    });

    // Sampling demo button
    document.querySelector('[data-demo="sampling"]').addEventListener('click', () => {
        openModal('sampling-modal');
        samplingDemo.init();
    });

    // KV Cache demo button
    document.querySelector('[data-demo="kvcache"]').addEventListener('click', () => {
        openModal('kvcache-modal');
        kvCacheDemo.init();
    });

    // Token flow animation
    document.querySelector('[data-demo="flow"]')?.addEventListener('click', () => {
        const tokens = tokenize('The cat sat');
        animateTokenFlow(tokens);
    });
}

/**
 * Setup tour button
 */
function setupTourButton() {
    const tourBtn = document.querySelector('.tour-start-btn');
    tourBtn.addEventListener('click', () => {
        startTour();
    });
}

/**
 * Setup modal events
 */
function setupModalEvents() {
    // Close buttons
    document.querySelectorAll('.modal-close').forEach(btn => {
        btn.addEventListener('click', () => {
            const modal = btn.closest('.modal-overlay');
            closeModal(modal.id);
        });
    });

    // Click outside to close
    document.querySelectorAll('.modal-overlay').forEach(overlay => {
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                closeModal(overlay.id);
            }
        });
    });

    // Escape to close
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            document.querySelectorAll('.modal-overlay.open').forEach(modal => {
                closeModal(modal.id);
            });
        }
    });
}

/**
 * Open modal
 */
function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.add('open');
        document.body.style.overflow = 'hidden';
    }
}

/**
 * Close modal
 */
function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.remove('open');
        document.body.style.overflow = '';
    }
}

/**
 * Setup glossary
 */
function setupGlossary() {
    const container = document.getElementById('glossary-list');
    const searchInput = document.getElementById('glossary-search');

    // Render glossary items
    function renderGlossary(filter = '') {
        const filtered = GLOSSARY.filter(item =>
            item.term.toLowerCase().includes(filter.toLowerCase()) ||
            item.definition.toLowerCase().includes(filter.toLowerCase())
        );

        container.innerHTML = filtered.map(item => `
            <div class="glossary-item">
                <div class="glossary-term">${item.term}</div>
                <div class="glossary-definition">${item.definition}</div>
            </div>
        `).join('');
    }

    renderGlossary();

    // Search functionality
    searchInput.addEventListener('input', (e) => {
        renderGlossary(e.target.value);
    });
}

/**
 * Handle window resize
 */
function handleResize() {
    // Close panel on small screens
    if (window.innerWidth < 768 && infoPanelOpen) {
        // Keep panel open but remove margin adjustment
        document.querySelector('.diagram-section').classList.remove('panel-open');
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', init);
