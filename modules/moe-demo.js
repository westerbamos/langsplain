/**
 * Mixture of Experts simulation
 * Interactive demo showing how MOE routing works
 */

import { tokenize, getEmbeddings, getEmbedDim } from './tokenizer.js';
import { softmax, matvec, randomMatrix, transpose } from './math-utils.js';

// MOE configuration
const CONFIG = {
    numExperts: 8,
    topK: 2,
    embedDim: 64,
    expertColors: [
        '#ef4444', // red
        '#f97316', // orange
        '#eab308', // yellow
        '#22c55e', // green
        '#06b6d4', // cyan
        '#3b82f6', // blue
        '#8b5cf6', // violet
        '#ec4899'  // pink
    ],
    expertNames: [
        'Expert 1',
        'Expert 2',
        'Expert 3',
        'Expert 4',
        'Expert 5',
        'Expert 6',
        'Expert 7',
        'Expert 8'
    ]
};

// Token categories for semantic routing
const TOKEN_CATEGORIES = {
    grammar: new Set(['the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need']),
    facts: new Set(['cat', 'dog', 'bird', 'fish', 'tree', 'house', 'car', 'book', 'fox', 'man', 'woman', 'child', 'day', 'night', 'world']),
    math: new Set(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '=']),
    code: new Set(['print', 'function', 'return', 'var', 'let', 'const', '(', ')', '[', ']', '{', '}', ':', ';']),
    creative: new Set(['quick', 'brown', 'good', 'bad', 'big', 'small', 'new', 'old', 'young', 'lazy', 'love', 'like', 'hello']),
    logic: new Set(['and', 'or', 'nor', 'not', 'if', 'then', 'else', 'when', 'why', 'how', 'what', 'which']),
    language: new Set(['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their']),
    general: new Set(['"', "'", '.', ',', '!', '?', '-'])
};

// Category order maps to expert indices (0-7)
const CATEGORY_ORDER = ['grammar', 'facts', 'math', 'code', 'creative', 'logic', 'language', 'general'];

// Secondary affinities: some tokens should boost multiple experts
const SECONDARY_AFFINITIES = {
    3: [[6, 0.8]], // Code → Language (code often involves variable names)
    4: [[6, 0.6]], // Creative → Language (creative text is linguistic)
    6: [[0, 0.5]]  // Language → Grammar (pronouns need grammar)
};

// Semantic bias strength (added to logits for primary expert)
const SEMANTIC_BIAS_STRENGTH = 2.5;

// Router weights (fixed for consistency)
const routerWeights = randomMatrix(CONFIG.embedDim, CONFIG.numExperts, 42424242);

/**
 * Get the expert category for a token based on its text
 * @param {string} tokenText - Token text to classify
 * @returns {number} Expert index (0-7)
 */
function getTokenExpertCategory(tokenText) {
    const text = tokenText.toLowerCase();

    // Single letters go to General expert
    if (text.length === 1 && /[a-z]/.test(text)) {
        return 7; // General
    }

    // Check each category
    for (let i = 0; i < CATEGORY_ORDER.length; i++) {
        const category = CATEGORY_ORDER[i];
        if (TOKEN_CATEGORIES[category].has(text)) {
            return i; // Return expert index
        }
    }

    // Default to General
    return 7;
}

/**
 * Route a single token embedding to experts
 * @param {Float32Array} embedding - Token embedding vector
 * @param {string} tokenText - Optional token text for semantic biasing
 */
function routeToken(embedding, tokenText = null) {
    // Compute router logits
    const logits = matvec(transpose(routerWeights), embedding);

    // Apply semantic biasing if token text is provided
    if (tokenText) {
        const primaryExpert = getTokenExpertCategory(tokenText);

        // Boost primary expert
        logits[primaryExpert] += SEMANTIC_BIAS_STRENGTH;

        // Apply secondary affinities
        const affinities = SECONDARY_AFFINITIES[primaryExpert];
        if (affinities) {
            affinities.forEach(([expertIdx, boost]) => {
                logits[expertIdx] += boost;
            });
        }
    }

    // Softmax to get probabilities
    const probs = softmax(logits);

    // Select top-k experts
    const indexed = probs.map((p, i) => ({ prob: p, index: i }));
    indexed.sort((a, b) => b.prob - a.prob);

    const topExperts = indexed.slice(0, CONFIG.topK);

    // Normalize top-k weights
    const totalWeight = topExperts.reduce((sum, e) => sum + e.prob, 0);
    const normalizedWeights = topExperts.map(e => ({
        ...e,
        weight: e.prob / totalWeight
    }));

    return {
        allProbs: probs,
        topExperts: normalizedWeights,
        expert1: normalizedWeights[0],
        expert2: normalizedWeights[1]
    };
}

/**
 * Run MOE routing for all tokens
 */
export function runMOEDemo(text) {
    // Tokenize
    const tokens = tokenize(text).slice(0, 10);
    if (tokens.length === 0) return null;

    // Get embeddings
    const embeddings = getEmbeddings(tokens);

    // Route each token
    const routingResults = embeddings.map((emb, i) => ({
        token: tokens[i],
        tokenIndex: i,
        routing: routeToken(emb, tokens[i].text)
    }));

    // Compute load balancing statistics
    const expertCounts = new Array(CONFIG.numExperts).fill(0);
    const expertWeights = new Array(CONFIG.numExperts).fill(0);

    routingResults.forEach(r => {
        r.routing.topExperts.forEach(e => {
            expertCounts[e.index]++;
            expertWeights[e.index] += e.weight;
        });
    });

    // Ideal balance would be: numTokens * topK / numExperts per expert
    const idealCount = (tokens.length * CONFIG.topK) / CONFIG.numExperts;
    const loadImbalance = expertCounts.map(c => Math.abs(c - idealCount));
    const avgImbalance = loadImbalance.reduce((a, b) => a + b, 0) / CONFIG.numExperts;

    return {
        tokens,
        config: CONFIG,
        routingResults,
        expertCounts,
        expertWeights,
        loadBalance: {
            idealCount,
            avgImbalance,
            maxImbalance: Math.max(...loadImbalance)
        }
    };
}

/**
 * MOE Demo UI Controller
 */
export class MOEDemoUI {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.result = null;
        this.selectedToken = null;
    }

    /**
     * Initialize the demo UI
     */
    init() {
        this.container.innerHTML = `
            <div class="demo-content">
                <div class="demo-header">
                    <h2>Mixture of Experts</h2>
                    <p class="demo-description">
                        Watch how a router assigns tokens to experts.
                        Each token activates only ${CONFIG.topK} of ${CONFIG.numExperts} experts.
                        This demo adds lightweight rule-based biasing so routing patterns are easier to inspect.
                    </p>
                </div>

                <div class="demo-input-section">
                    <label for="moe-input">Enter text (max 10 tokens):</label>
                    <div class="input-row">
                        <input type="text" id="moe-input"
                               value="print(&quot;The quick brown fox&quot;)"
                               placeholder="Enter some text..."
                               maxlength="100">
                        <button id="moe-run-btn" class="primary-btn">Route</button>
                    </div>
                </div>

                <div class="moe-visualization">
                    <div class="moe-tokens" id="moe-tokens">
                        <h4>Tokens</h4>
                        <div class="token-list" id="moe-token-list"></div>
                    </div>

                    <div class="moe-flow" id="moe-flow">
                        <svg id="moe-svg"></svg>
                    </div>

                    <div class="moe-experts" id="moe-experts">
                        <h4>Experts</h4>
                        <div class="expert-list" id="expert-list"></div>
                    </div>
                </div>

                <div class="moe-stats">
                    <div class="stat-panel" id="routing-detail">
                        <h4>Routing Details</h4>
                        <p class="hint">Click a token to see its routing</p>
                    </div>

                    <div class="stat-panel" id="load-balance">
                        <h4>Load Balance</h4>
                        <div id="load-chart"></div>
                    </div>
                </div>

                <div class="demo-explanation" id="moe-explanation">
                    <p>Click "Route" to see how tokens are assigned to experts.</p>
                </div>
            </div>
        `;

        this.setupEventListeners();
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        const runBtn = this.container.querySelector('#moe-run-btn');
        const input = this.container.querySelector('#moe-input');

        runBtn.addEventListener('click', () => this.run());
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.run();
        });
    }

    /**
     * Run the MOE demo
     */
    run() {
        const input = this.container.querySelector('#moe-input');
        const text = input.value.trim();

        if (!text) return;

        this.result = runMOEDemo(text);
        if (!this.result) return;

        this.selectedToken = null;
        this.renderTokens();
        this.renderExperts();
        this.renderFlowDiagram();
        this.renderLoadBalance();
        this.updateExplanation();
    }

    /**
     * Render token list
     */
    renderTokens() {
        const container = this.container.querySelector('#moe-token-list');
        container.innerHTML = '';

        this.result.routingResults.forEach((r, i) => {
            const tokenEl = document.createElement('div');
            tokenEl.className = 'moe-token';
            tokenEl.dataset.index = i;

            // Create gradient background from top expert colors
            const color1 = CONFIG.expertColors[r.routing.expert1.index];
            const color2 = CONFIG.expertColors[r.routing.expert2.index];

            tokenEl.innerHTML = `
                <span class="token-text">${r.token.text}</span>
                <div class="token-experts">
                    <span class="expert-dot" style="background: ${color1}"></span>
                    <span class="expert-dot" style="background: ${color2}"></span>
                </div>
            `;

            tokenEl.addEventListener('click', () => this.selectToken(i));
            container.appendChild(tokenEl);
        });
    }

    /**
     * Render expert list
     */
    renderExperts() {
        const container = this.container.querySelector('#expert-list');
        container.innerHTML = '';

        for (let i = 0; i < CONFIG.numExperts; i++) {
            const expertEl = document.createElement('div');
            expertEl.className = 'expert-box';
            expertEl.dataset.index = i;
            expertEl.style.borderColor = CONFIG.expertColors[i];

            const count = this.result.expertCounts[i];
            const weight = this.result.expertWeights[i];

            expertEl.innerHTML = `
                <div class="expert-icon" style="background: ${CONFIG.expertColors[i]}">
                    E${i + 1}
                </div>
                <div class="expert-info">
                    <span class="expert-name">${CONFIG.expertNames[i]}</span>
                    <span class="expert-load">${count} tokens (${(weight * 100 / this.result.tokens.length).toFixed(0)}%)</span>
                </div>
            `;

            container.appendChild(expertEl);
        }
    }

    /**
     * Render flow diagram with SVG
     */
    renderFlowDiagram() {
        const svg = d3.select(this.container.querySelector('#moe-svg'));
        svg.selectAll('*').remove();

        const flowContainer = this.container.querySelector('#moe-flow');
        const width = flowContainer.clientWidth;
        const height = flowContainer.clientHeight || 200;

        svg.attr('width', width)
           .attr('height', height)
           .attr('viewBox', `0 0 ${width} ${height}`);

        const n = this.result.tokens.length;
        const tokenSpacing = Math.min(50, (height - 40) / n);
        const tokenStartY = (height - (n - 1) * tokenSpacing) / 2;

        const expertSpacing = Math.min(40, (height - 40) / CONFIG.numExperts);
        const expertStartY = (height - (CONFIG.numExperts - 1) * expertSpacing) / 2;

        // Draw routing paths
        this.result.routingResults.forEach((r, tokenIdx) => {
            const tokenY = tokenStartY + tokenIdx * tokenSpacing;

            r.routing.topExperts.forEach((expert, rank) => {
                const expertY = expertStartY + expert.index * expertSpacing;
                const color = CONFIG.expertColors[expert.index];

                // Create curved path
                const path = svg.append('path')
                    .attr('d', `M 30 ${tokenY} Q ${width / 2} ${(tokenY + expertY) / 2} ${width - 30} ${expertY}`)
                    .attr('fill', 'none')
                    .attr('stroke', color)
                    .attr('stroke-width', expert.weight * 4 + 1)
                    .attr('opacity', expert.weight * 0.8 + 0.2)
                    .attr('class', 'routing-path')
                    .attr('data-token', tokenIdx)
                    .attr('data-expert', expert.index);

                // Animate path drawing
                const pathLength = path.node().getTotalLength();
                path.attr('stroke-dasharray', pathLength)
                    .attr('stroke-dashoffset', pathLength)
                    .transition()
                    .delay(tokenIdx * 100)
                    .duration(500)
                    .ease(d3.easeQuadOut)
                    .attr('stroke-dashoffset', 0);
            });
        });

        // Token position dots
        this.result.tokens.forEach((_, i) => {
            const y = tokenStartY + i * tokenSpacing;
            svg.append('circle')
                .attr('cx', 20)
                .attr('cy', y)
                .attr('r', 6)
                .attr('fill', '#a855f7')
                .attr('class', 'token-dot')
                .attr('data-index', i);
        });

        // Expert position dots
        for (let i = 0; i < CONFIG.numExperts; i++) {
            const y = expertStartY + i * expertSpacing;
            svg.append('circle')
                .attr('cx', width - 20)
                .attr('cy', y)
                .attr('r', 6)
                .attr('fill', CONFIG.expertColors[i])
                .attr('class', 'expert-dot');
        }
    }

    /**
     * Render load balance chart
     */
    renderLoadBalance() {
        const container = this.container.querySelector('#load-chart');
        container.innerHTML = '';

        const maxCount = Math.max(...this.result.expertCounts, this.result.loadBalance.idealCount);

        for (let i = 0; i < CONFIG.numExperts; i++) {
            const count = this.result.expertCounts[i];
            const pct = (count / maxCount) * 100;
            const idealPct = (this.result.loadBalance.idealCount / maxCount) * 100;

            const bar = document.createElement('div');
            bar.className = 'load-bar';
            bar.innerHTML = `
                <span class="bar-label">E${i + 1}</span>
                <div class="bar-track">
                    <div class="bar-fill" style="width: ${pct}%; background: ${CONFIG.expertColors[i]}"></div>
                    <div class="bar-ideal" style="left: ${idealPct}%"></div>
                </div>
                <span class="bar-count">${count}</span>
            `;
            container.appendChild(bar);
        }

        // Add legend
        const legend = document.createElement('div');
        legend.className = 'load-legend';
        legend.innerHTML = `
            <span class="legend-item">
                <span class="legend-line ideal"></span> Ideal balance
            </span>
        `;
        container.appendChild(legend);
    }

    /**
     * Select a token to show routing details
     */
    selectToken(index) {
        this.selectedToken = index;

        // Update token highlighting
        this.container.querySelectorAll('.moe-token').forEach((el, i) => {
            el.classList.toggle('selected', i === index);
        });

        // Highlight paths in SVG
        const svg = d3.select(this.container.querySelector('#moe-svg'));
        svg.selectAll('.routing-path')
            .attr('opacity', function() {
                const tokenIdx = parseInt(this.dataset.token);
                return tokenIdx === index ? 1 : 0.15;
            })
            .attr('stroke-width', function() {
                const tokenIdx = parseInt(this.dataset.token);
                if (tokenIdx === index) {
                    const expertIdx = parseInt(this.dataset.expert);
                    const expert = this.result.routingResults[index].routing.topExperts.find(e => e.index === expertIdx);
                    return expert ? expert.weight * 6 + 2 : 3;
                }
                return 1;
            }.bind(this));

        // Show routing details
        this.showRoutingDetails(index);
    }

    /**
     * Show routing details for selected token
     */
    showRoutingDetails(tokenIndex) {
        const container = this.container.querySelector('#routing-detail');
        const r = this.result.routingResults[tokenIndex];

        let probsHTML = r.routing.allProbs.map((p, i) => `
            <div class="prob-row">
                <span class="prob-expert" style="color: ${CONFIG.expertColors[i]}">
                    E${i + 1} ${CONFIG.expertNames[i]}
                </span>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" style="width: ${p * 100}%; background: ${CONFIG.expertColors[i]}"></div>
                </div>
                <span class="prob-value">${(p * 100).toFixed(1)}%</span>
            </div>
        `).join('');

        container.innerHTML = `
            <h4>Routing: "${r.token.text}"</h4>
            <div class="selected-experts">
                <p>
                    <strong>Selected:</strong>
                    <span style="color: ${CONFIG.expertColors[r.routing.expert1.index]}">
                        E${r.routing.expert1.index + 1} (${(r.routing.expert1.weight * 100).toFixed(0)}%)
                    </span>
                    +
                    <span style="color: ${CONFIG.expertColors[r.routing.expert2.index]}">
                        E${r.routing.expert2.index + 1} (${(r.routing.expert2.weight * 100).toFixed(0)}%)
                    </span>
                </p>
            </div>
            <div class="all-probs">
                <p><strong>All Expert Probabilities:</strong></p>
                ${probsHTML}
            </div>
        `;
    }

    /**
     * Update explanation text
     */
    updateExplanation() {
        const container = this.container.querySelector('#moe-explanation');
        const lb = this.result.loadBalance;

        container.innerHTML = `
            <p><strong>Results:</strong></p>
            <p>
                ${this.result.tokens.length} tokens routed to top-${CONFIG.topK} of ${CONFIG.numExperts} experts.
                Ideal distribution: ${lb.idealCount.toFixed(1)} tokens per expert.
                Load imbalance: ${lb.avgImbalance.toFixed(2)} average, ${lb.maxImbalance.toFixed(1)} max.
            </p>
            <p class="hint">
                This demo intentionally uses token-category biasing so routing is legible.
                Real MOE routers are learned continuous functions of hidden states.
            </p>
            <p class="hint">
                In real MOE models, auxiliary losses encourage load balancing to prevent
                some experts from being overloaded while others are underutilized.
            </p>
        `;
    }
}

export default MOEDemoUI;
