/**
 * Attention visualization simulation
 * Interactive demo showing how self-attention works
 */

import { tokenize, getEmbeddings, getEmbedDim } from './tokenizer.js';
import {
    softmax,
    matmul,
    transpose,
    randomMatrix,
    applyCausalMask,
    layerNorm,
    vadd,
    geluVec,
    matvec
} from './math-utils.js';

// Model configuration (toy size)
const CONFIG = {
    numLayers: 3,
    numHeads: 4,
    embedDim: 64,
    headDim: 16, // embedDim / numHeads
    ffnDim: 256  // 4x expansion
};

// Pre-computed weight matrices (seeded for consistency)
let weights = null;

/**
 * Initialize weight matrices
 */
function initWeights() {
    if (weights) return;

    weights = {
        layers: []
    };

    for (let l = 0; l < CONFIG.numLayers; l++) {
        const layer = {
            attention: {
                Wq: [],
                Wk: [],
                Wv: [],
                Wo: randomMatrix(CONFIG.embedDim, CONFIG.embedDim, l * 1000 + 400)
            },
            ffn: {
                W1: randomMatrix(CONFIG.embedDim, CONFIG.ffnDim, l * 1000 + 500),
                W2: randomMatrix(CONFIG.ffnDim, CONFIG.embedDim, l * 1000 + 600)
            }
        };

        // Create head-specific projections
        for (let h = 0; h < CONFIG.numHeads; h++) {
            layer.attention.Wq.push(randomMatrix(CONFIG.embedDim, CONFIG.headDim, l * 1000 + h * 100 + 1));
            layer.attention.Wk.push(randomMatrix(CONFIG.embedDim, CONFIG.headDim, l * 1000 + h * 100 + 2));
            layer.attention.Wv.push(randomMatrix(CONFIG.embedDim, CONFIG.headDim, l * 1000 + h * 100 + 3));
        }

        weights.layers.push(layer);
    }
}

/**
 * Compute attention for one head
 */
function computeHeadAttention(embeddings, Wq, Wk, Wv) {
    const seqLen = embeddings.length;

    // Project to Q, K, V
    const Q = embeddings.map(e => matvec(transpose(Wq), e));
    const K = embeddings.map(e => matvec(transpose(Wk), e));
    const V = embeddings.map(e => matvec(transpose(Wv), e));

    // Compute attention scores: Q @ K^T / sqrt(d_k)
    const scale = Math.sqrt(CONFIG.headDim);
    const scores = [];

    for (let i = 0; i < seqLen; i++) {
        const row = [];
        for (let j = 0; j < seqLen; j++) {
            let score = 0;
            for (let k = 0; k < CONFIG.headDim; k++) {
                score += Q[i][k] * K[j][k];
            }
            row.push(score / scale);
        }
        scores.push(row);
    }

    // Apply causal mask
    const maskedScores = applyCausalMask(scores);

    // Softmax per row
    const attentionWeights = maskedScores.map(row => softmax(row));

    // Apply attention to values: weights @ V
    const output = [];
    for (let i = 0; i < seqLen; i++) {
        const out = new Array(CONFIG.headDim).fill(0);
        for (let j = 0; j < seqLen; j++) {
            for (let k = 0; k < CONFIG.headDim; k++) {
                out[k] += attentionWeights[i][j] * V[j][k];
            }
        }
        output.push(out);
    }

    return {
        Q, K, V,
        scores,
        weights: attentionWeights,
        output
    };
}

/**
 * Run multi-head attention for one layer
 */
function runMultiHeadAttention(embeddings, layerIdx) {
    initWeights();
    const layer = weights.layers[layerIdx];

    const headResults = [];
    for (let h = 0; h < CONFIG.numHeads; h++) {
        headResults.push(computeHeadAttention(
            embeddings,
            layer.attention.Wq[h],
            layer.attention.Wk[h],
            layer.attention.Wv[h]
        ));
    }

    // Concatenate head outputs
    const seqLen = embeddings.length;
    const concatenated = [];
    for (let i = 0; i < seqLen; i++) {
        const concat = [];
        for (let h = 0; h < CONFIG.numHeads; h++) {
            concat.push(...headResults[h].output[i]);
        }
        concatenated.push(concat);
    }

    // Project with Wo
    const attentionOutput = concatenated.map(c => matvec(transpose(layer.attention.Wo), c));

    // Residual + LayerNorm
    const afterAttention = embeddings.map((e, i) => layerNorm(vadd(e, attentionOutput[i])));

    return {
        headResults,
        attentionOutput,
        afterAttention
    };
}

/**
 * Run feed-forward network for one layer
 */
function runFFN(embeddings, layerIdx) {
    initWeights();
    const layer = weights.layers[layerIdx];

    const ffnOutput = embeddings.map(e => {
        // First linear + GELU
        const hidden = geluVec(matvec(transpose(layer.ffn.W1), e));
        // Second linear
        return matvec(transpose(layer.ffn.W2), hidden);
    });

    // Residual + LayerNorm
    const afterFFN = embeddings.map((e, i) => layerNorm(vadd(e, ffnOutput[i])));

    return {
        ffnOutput,
        afterFFN
    };
}

/**
 * Run full forward pass and collect attention data
 */
export function runAttentionDemo(text) {
    initWeights();

    // Tokenize
    const tokens = tokenize(text).slice(0, 10); // Max 10 tokens
    if (tokens.length === 0) {
        return null;
    }

    // Get embeddings with positional encoding
    let embeddings = getEmbeddings(tokens);

    // Store results for all layers
    const layerResults = [];

    for (let l = 0; l < CONFIG.numLayers; l++) {
        // Run attention
        const attentionResult = runMultiHeadAttention(embeddings, l);

        // Run FFN
        const ffnResult = runFFN(attentionResult.afterAttention, l);

        layerResults.push({
            layer: l,
            inputEmbeddings: embeddings,
            attention: attentionResult,
            ffn: ffnResult,
            outputEmbeddings: ffnResult.afterFFN
        });

        // Update embeddings for next layer
        embeddings = ffnResult.afterFFN;
    }

    return {
        tokens,
        config: CONFIG,
        layerResults
    };
}

/**
 * Get attention weights for a specific layer and head
 */
export function getAttentionWeights(result, layerIdx, headIdx) {
    if (!result || !result.layerResults[layerIdx]) return null;
    return result.layerResults[layerIdx].attention.headResults[headIdx].weights;
}

/**
 * Get all head attention weights for a layer
 */
export function getAllHeadWeights(result, layerIdx) {
    if (!result || !result.layerResults[layerIdx]) return null;
    return result.layerResults[layerIdx].attention.headResults.map(h => h.weights);
}

/**
 * Attention Demo UI Controller
 */
export class AttentionDemoUI {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.result = null;
        this.currentLayer = 0;
        this.currentHead = 0;
        this.animating = false;
    }

    /**
     * Initialize the demo UI
     */
    init() {
        this.container.innerHTML = `
            <div class="demo-content">
                <div class="demo-header">
                    <h2>Attention Visualization</h2>
                    <p class="demo-description">
                        See how tokens attend to each other in self-attention.
                        Brighter lines = stronger attention weights.
                    </p>
                </div>

                <div class="demo-input-section">
                    <label for="attention-input">Enter text (max 10 tokens):</label>
                    <div class="input-row">
                        <input type="text" id="attention-input"
                               value="The cat sat on the mat"
                               placeholder="Enter some text..."
                               maxlength="100">
                        <button id="attention-run-btn" class="primary-btn">Run</button>
                    </div>
                </div>

                <div class="demo-controls">
                    <div class="control-group">
                        <label>Layer:</label>
                        <div class="layer-buttons" id="layer-buttons"></div>
                    </div>
                    <div class="control-group">
                        <label>Head:</label>
                        <div class="head-buttons" id="head-buttons"></div>
                    </div>
                    <button id="animate-btn" class="secondary-btn">Animate Flow</button>
                </div>

                <div class="visualization-area">
                    <div class="tokens-display" id="tokens-display"></div>
                    <div class="attention-viz" id="attention-viz">
                        <svg id="attention-svg"></svg>
                    </div>
                    <div class="heatmap-container" id="heatmap-container">
                        <h4>Attention Heatmap</h4>
                        <div id="heatmap"></div>
                    </div>
                </div>

                <div class="demo-explanation" id="demo-explanation">
                    <p>Click "Run" to visualize attention patterns.</p>
                </div>
            </div>
        `;

        this.setupEventListeners();
        this.createLayerHeadButtons();
    }

    /**
     * Create layer and head selection buttons
     */
    createLayerHeadButtons() {
        const layerContainer = this.container.querySelector('#layer-buttons');
        const headContainer = this.container.querySelector('#head-buttons');

        // Layer buttons
        for (let l = 0; l < CONFIG.numLayers; l++) {
            const btn = document.createElement('button');
            btn.className = `selector-btn ${l === 0 ? 'active' : ''}`;
            btn.textContent = `${l + 1}`;
            btn.dataset.layer = l;
            btn.addEventListener('click', () => this.selectLayer(l));
            layerContainer.appendChild(btn);
        }

        // Head buttons
        for (let h = 0; h < CONFIG.numHeads; h++) {
            const btn = document.createElement('button');
            btn.className = `selector-btn ${h === 0 ? 'active' : ''}`;
            btn.textContent = `${h + 1}`;
            btn.dataset.head = h;
            btn.addEventListener('click', () => this.selectHead(h));
            headContainer.appendChild(btn);
        }
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        const runBtn = this.container.querySelector('#attention-run-btn');
        const input = this.container.querySelector('#attention-input');
        const animateBtn = this.container.querySelector('#animate-btn');

        runBtn.addEventListener('click', () => this.run());
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.run();
        });
        animateBtn.addEventListener('click', () => this.animateFlow());
    }

    /**
     * Run the attention demo
     */
    run() {
        const input = this.container.querySelector('#attention-input');
        const text = input.value.trim();

        if (!text) return;

        this.result = runAttentionDemo(text);
        if (!this.result) return;

        this.renderTokens();
        this.renderVisualization();
        this.updateExplanation();
    }

    /**
     * Render token display
     */
    renderTokens() {
        const container = this.container.querySelector('#tokens-display');
        container.innerHTML = '';

        this.result.tokens.forEach((token, i) => {
            const tokenEl = document.createElement('div');
            tokenEl.className = 'token-box';
            tokenEl.dataset.index = i;
            tokenEl.innerHTML = `
                <span class="token-text">${token.text}</span>
                <span class="token-pos">${i}</span>
            `;
            container.appendChild(tokenEl);
        });
    }

    /**
     * Render attention visualization
     */
    renderVisualization() {
        const weights = getAttentionWeights(this.result, this.currentLayer, this.currentHead);
        if (!weights) return;

        this.renderAttentionArcs(weights);
        this.renderHeatmap(weights);
    }

    /**
     * Render attention as arcs between tokens
     */
    renderAttentionArcs(weights) {
        const svg = d3.select(this.container.querySelector('#attention-svg'));
        svg.selectAll('*').remove();

        const tokens = this.result.tokens;
        const n = tokens.length;

        const width = this.container.querySelector('#attention-viz').clientWidth;
        const height = 150;
        const tokenWidth = Math.min(80, (width - 40) / n);
        const startX = (width - n * tokenWidth) / 2 + tokenWidth / 2;

        svg.attr('width', width)
            .attr('height', height)
            .attr('viewBox', `0 0 ${width} ${height}`);

        // Draw arcs for attention weights
        for (let i = 0; i < n; i++) {
            for (let j = 0; j <= i; j++) {
                const weight = weights[i][j];
                if (weight < 0.05) continue; // Skip very small weights

                const x1 = startX + j * tokenWidth;
                const x2 = startX + i * tokenWidth;
                const midX = (x1 + x2) / 2;
                const arcHeight = Math.abs(i - j) * 20 + 30;

                const path = `M ${x1} ${height - 20} Q ${midX} ${height - arcHeight} ${x2} ${height - 20}`;

                svg.append('path')
                    .attr('d', path)
                    .attr('fill', 'none')
                    .attr('stroke', `rgba(0, 212, 255, ${weight})`)
                    .attr('stroke-width', weight * 4 + 1)
                    .attr('class', 'attention-arc')
                    .attr('data-from', j)
                    .attr('data-to', i);
            }
        }

        // Token position indicators
        for (let i = 0; i < n; i++) {
            svg.append('circle')
                .attr('cx', startX + i * tokenWidth)
                .attr('cy', height - 20)
                .attr('r', 6)
                .attr('fill', '#00d4ff')
                .attr('class', 'token-indicator');

            svg.append('text')
                .attr('x', startX + i * tokenWidth)
                .attr('y', height - 5)
                .attr('text-anchor', 'middle')
                .attr('fill', '#a0a0a0')
                .attr('font-size', '10px')
                .text(tokens[i].text.slice(0, 5));
        }
    }

    /**
     * Render attention heatmap
     */
    renderHeatmap(weights) {
        const container = this.container.querySelector('#heatmap');
        container.innerHTML = '';

        const n = weights.length;
        const tokens = this.result.tokens;

        // Create heatmap grid
        const grid = document.createElement('div');
        grid.className = 'heatmap-grid';
        grid.style.gridTemplateColumns = `auto repeat(${n}, 1fr)`;

        // Header row (token labels)
        grid.appendChild(document.createElement('div')); // Empty corner
        for (let j = 0; j < n; j++) {
            const header = document.createElement('div');
            header.className = 'heatmap-header';
            header.textContent = tokens[j].text.slice(0, 4);
            grid.appendChild(header);
        }

        // Data rows
        for (let i = 0; i < n; i++) {
            // Row label
            const rowLabel = document.createElement('div');
            rowLabel.className = 'heatmap-label';
            rowLabel.textContent = tokens[i].text.slice(0, 4);
            grid.appendChild(rowLabel);

            // Cells
            for (let j = 0; j < n; j++) {
                const cell = document.createElement('div');
                cell.className = 'heatmap-cell';

                if (j > i) {
                    // Masked (future tokens)
                    cell.classList.add('masked');
                    cell.textContent = '-';
                } else {
                    const weight = weights[i][j];
                    cell.style.backgroundColor = `rgba(0, 212, 255, ${weight})`;
                    cell.title = `${tokens[i].text} → ${tokens[j].text}: ${weight.toFixed(3)}`;
                    if (weight > 0.1) {
                        cell.textContent = weight.toFixed(2);
                    }
                }

                grid.appendChild(cell);
            }
        }

        container.appendChild(grid);
    }

    /**
     * Select a layer
     */
    selectLayer(layer) {
        this.currentLayer = layer;

        // Update button states
        this.container.querySelectorAll('#layer-buttons .selector-btn').forEach((btn, i) => {
            btn.classList.toggle('active', i === layer);
        });

        if (this.result) {
            this.renderVisualization();
        }
    }

    /**
     * Select a head
     */
    selectHead(head) {
        this.currentHead = head;

        // Update button states
        this.container.querySelectorAll('#head-buttons .selector-btn').forEach((btn, i) => {
            btn.classList.toggle('active', i === head);
        });

        if (this.result) {
            this.renderVisualization();
        }
    }

    /**
     * Animate token flow through layers
     */
    async animateFlow() {
        if (!this.result || this.animating) return;

        // Scroll to heatmap for better visibility during animation
        const heatmapContainer = this.container.querySelector('#heatmap-container');
        if (heatmapContainer) {
            heatmapContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }

        this.animating = true;
        const animateBtn = this.container.querySelector('#animate-btn');
        animateBtn.disabled = true;

        // Cycle through layers
        for (let l = 0; l < CONFIG.numLayers; l++) {
            this.selectLayer(l);
            await this.sleep(800);

            // Cycle through heads
            for (let h = 0; h < CONFIG.numHeads; h++) {
                this.selectHead(h);
                await this.sleep(400);
            }
        }

        this.animating = false;
        animateBtn.disabled = false;
    }

    /**
     * Sleep helper for animations
     */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Update explanation text
     */
    updateExplanation() {
        const container = this.container.querySelector('#demo-explanation');
        const n = this.result.tokens.length;
        const perHeadLayerPairs = (n * (n + 1)) / 2;
        const totalScoreComputations = perHeadLayerPairs * CONFIG.numHeads * CONFIG.numLayers;

        container.innerHTML = `
            <p><strong>Layer ${this.currentLayer + 1}, Head ${this.currentHead + 1}</strong></p>
            <p>
                Each row shows how much a token (right) attends to previous tokens (left).
                The causal mask prevents attending to future tokens (shown as "-").
            </p>
            <p>
                ${perHeadLayerPairs} token pairs/head/layer × ${CONFIG.numHeads} heads × ${CONFIG.numLayers} layers =
                ${totalScoreComputations} attention score computations
            </p>
        `;
    }
}

export default AttentionDemoUI;
