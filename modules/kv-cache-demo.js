/**
 * KV Cache visualization
 * Interactive demo showing how key-value caching speeds up autoregressive generation
 */

// Configuration for the demo
const CONFIG = {
    maxTokens: 8,
    kvDim: 6,      // Simplified K/V dimension for visualization
    numHeads: 2,
    animationSpeed: 800
};

// Sample tokens for the demo
const SAMPLE_TOKENS = ['The', 'cat', 'sat', 'on', 'the', 'warm', 'soft', 'mat'];

/**
 * KV Cache Demo UI Controller
 */
export class KVCacheDemoUI {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.currentStep = 0;
        this.isPlaying = false;
        this.useCache = true;
        this.generatedTokens = [];
        this.cache = { keys: [], values: [] };
        this.stats = {
            withCache: { kComputations: 0, vComputations: 0, attentionOps: 0 },
            withoutCache: { kComputations: 0, vComputations: 0, attentionOps: 0 }
        };
    }

    /**
     * Initialize the demo UI
     */
    init() {
        this.container.innerHTML = `
            <div class="demo-content kv-cache-demo">
                <div class="demo-header">
                    <h2>KV Cache Visualization</h2>
                    <p class="demo-description">
                        See how caching Key and Value tensors eliminates redundant computation during generation.
                        Watch the cache grow as tokens are generated.
                    </p>
                </div>

                <div class="kv-controls">
                    <div class="control-group">
                        <label>Animation Mode:</label>
                        <div class="toggle-buttons">
                            <button id="step-btn" class="toggle-btn active">Step-by-Step</button>
                            <button id="continuous-btn" class="toggle-btn">Continuous</button>
                        </div>
                    </div>
                    <div class="control-group">
                        <button id="play-btn" class="primary-btn">
                            <svg viewBox="0 0 24 24" fill="currentColor" width="18" height="18">
                                <path d="M8 5v14l11-7z"/>
                            </svg>
                            <span id="play-text">Generate Next</span>
                        </button>
                        <button id="reset-kv-btn" class="secondary-btn">Reset</button>
                    </div>
                </div>

                <div class="kv-comparison">
                    <div class="kv-panel with-cache">
                        <div class="panel-header-bar">
                            <h4>With KV Cache</h4>
                            <span class="efficiency-badge">Efficient</span>
                        </div>
                        <div class="token-sequence" id="cached-tokens"></div>
                        <div class="cache-grid-container">
                            <div class="cache-section">
                                <h5>Key Cache</h5>
                                <div class="cache-grid" id="key-cache-grid"></div>
                            </div>
                            <div class="cache-section">
                                <h5>Value Cache</h5>
                                <div class="cache-grid" id="value-cache-grid"></div>
                            </div>
                        </div>
                        <div class="computation-counter" id="cached-counter">
                            <div class="counter-item">
                                <span class="counter-label">K/V Computations</span>
                                <span class="counter-value" id="cached-kv-ops">0</span>
                            </div>
                            <div class="counter-item">
                                <span class="counter-label">Q·K Score Ops</span>
                                <span class="counter-value" id="cached-attn-ops">0</span>
                            </div>
                        </div>
                    </div>

                    <div class="kv-panel without-cache">
                        <div class="panel-header-bar">
                            <h4>Without Cache</h4>
                            <span class="inefficiency-badge">Redundant</span>
                        </div>
                        <div class="token-sequence" id="uncached-tokens"></div>
                        <div class="cache-grid-container">
                            <div class="cache-section">
                                <h5>Recomputed Keys</h5>
                                <div class="cache-grid recompute" id="uncached-key-grid"></div>
                            </div>
                            <div class="cache-section">
                                <h5>Recomputed Values</h5>
                                <div class="cache-grid recompute" id="uncached-value-grid"></div>
                            </div>
                        </div>
                        <div class="computation-counter" id="uncached-counter">
                            <div class="counter-item">
                                <span class="counter-label">K/V Computations</span>
                                <span class="counter-value warning" id="uncached-kv-ops">0</span>
                            </div>
                            <div class="counter-item">
                                <span class="counter-label">Q·K Score Ops</span>
                                <span class="counter-value" id="uncached-attn-ops">0</span>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="savings-display">
                    <h4>Computation Savings</h4>
                    <div class="savings-bar-container">
                        <div class="savings-bar" id="savings-bar">
                            <span class="savings-text" id="savings-text">0% saved</span>
                        </div>
                    </div>
                    <div class="savings-detail" id="savings-detail">
                        Generate tokens to see savings accumulate
                    </div>
                </div>

                <div class="demo-explanation">
                    <p><strong>Why KV Caching Matters:</strong></p>
                    <p>
                        In autoregressive generation, each new token needs to attend to ALL previous tokens.
                        Without caching, we'd recompute K and V for every previous token at every step.
                    </p>
                    <ul>
                        <li><strong>With Cache:</strong> Only compute K,V for the NEW token, reuse cached values for previous tokens</li>
                        <li><strong>Without Cache:</strong> Recompute K,V for ALL tokens at every step (O(n) vs O(1) per token)</li>
                        <li><strong>What this counter tracks:</strong> K/V projection savings only. Q·K attention score work is shown separately.</li>
                        <li><strong>Memory Trade-off:</strong> Cache uses GPU memory proportional to sequence length</li>
                    </ul>
                    <p class="hint">
                        At position n, caching saves (n-1) K/V computations per layer per head.
                        For long sequences, this is a massive speedup.
                    </p>
                </div>
            </div>
        `;

        this.setupEventListeners();
        this.renderInitialState();
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        const playBtn = this.container.querySelector('#play-btn');
        const resetBtn = this.container.querySelector('#reset-kv-btn');
        const stepBtn = this.container.querySelector('#step-btn');
        const continuousBtn = this.container.querySelector('#continuous-btn');

        playBtn.addEventListener('click', () => this.handlePlay());
        resetBtn.addEventListener('click', () => this.reset());

        stepBtn.addEventListener('click', () => {
            this.setMode('step');
            stepBtn.classList.add('active');
            continuousBtn.classList.remove('active');
        });

        continuousBtn.addEventListener('click', () => {
            this.setMode('continuous');
            continuousBtn.classList.add('active');
            stepBtn.classList.remove('active');
        });
    }

    /**
     * Set animation mode
     */
    setMode(mode) {
        this.mode = mode;
        const playText = this.container.querySelector('#play-text');
        playText.textContent = mode === 'step' ? 'Generate Next' : 'Play All';
    }

    /**
     * Handle play button click
     */
    async handlePlay() {
        if (this.isPlaying) {
            this.isPlaying = false;
            this.container.querySelector('#play-text').textContent =
                this.mode === 'step' ? 'Generate Next' : 'Play All';
            return;
        }

        if (this.mode === 'continuous') {
            this.isPlaying = true;
            this.container.querySelector('#play-text').textContent = 'Pause';

            while (this.isPlaying && this.currentStep < CONFIG.maxTokens) {
                await this.generateNextToken();
                await this.sleep(CONFIG.animationSpeed);
            }

            this.isPlaying = false;
            this.container.querySelector('#play-text').textContent = 'Play All';
        } else {
            if (this.currentStep < CONFIG.maxTokens) {
                await this.generateNextToken();
            }
        }
    }

    /**
     * Render initial state
     */
    renderInitialState() {
        this.renderTokenSequence('cached-tokens', []);
        this.renderTokenSequence('uncached-tokens', []);
        this.renderCacheGrid('key-cache-grid', [], false);
        this.renderCacheGrid('value-cache-grid', [], false);
        this.renderCacheGrid('uncached-key-grid', [], true);
        this.renderCacheGrid('uncached-value-grid', [], true);
        this.updateCounters();
        this.updateSavings();
    }

    /**
     * Generate next token with animation
     */
    async generateNextToken() {
        if (this.currentStep >= CONFIG.maxTokens) return;

        const newToken = SAMPLE_TOKENS[this.currentStep];
        this.generatedTokens.push(newToken);

        // Update stats for WITH cache
        // Only compute K,V for the new token
        this.stats.withCache.kComputations += CONFIG.kvDim;
        this.stats.withCache.vComputations += CONFIG.kvDim;
        // Attention with all previous tokens (cached K,V)
        this.stats.withCache.attentionOps += this.currentStep + 1;

        // Update stats for WITHOUT cache
        // Must recompute K,V for ALL tokens
        const seqLen = this.currentStep + 1;
        this.stats.withoutCache.kComputations += seqLen * CONFIG.kvDim;
        this.stats.withoutCache.vComputations += seqLen * CONFIG.kvDim;
        this.stats.withoutCache.attentionOps += seqLen;

        // Add to cache
        this.cache.keys.push(this.generateRandomKV());
        this.cache.values.push(this.generateRandomKV());

        // Animate the update
        await this.animateTokenGeneration(newToken);

        this.currentStep++;
        this.updateCounters();
        this.updateSavings();
    }

    /**
     * Generate random K/V values for visualization
     */
    generateRandomKV() {
        return Array(CONFIG.kvDim).fill(0).map(() => Math.random());
    }

    /**
     * Animate token generation
     */
    async animateTokenGeneration(token) {
        // Update token sequences
        this.renderTokenSequence('cached-tokens', this.generatedTokens, this.currentStep);
        this.renderTokenSequence('uncached-tokens', this.generatedTokens, this.currentStep);

        // Animate cache update (only new row highlights)
        this.renderCacheGrid('key-cache-grid', this.cache.keys, false, this.currentStep);
        this.renderCacheGrid('value-cache-grid', this.cache.values, false, this.currentStep);

        // Animate full recomputation (all rows highlight)
        this.renderCacheGrid('uncached-key-grid', this.cache.keys, true, -1); // -1 = all
        this.renderCacheGrid('uncached-value-grid', this.cache.values, true, -1);

        await this.sleep(300);
    }

    /**
     * Render token sequence
     */
    renderTokenSequence(containerId, tokens, highlightIndex = -1) {
        const container = this.container.querySelector(`#${containerId}`);

        if (tokens.length === 0) {
            container.innerHTML = '<span class="placeholder">Tokens will appear here</span>';
            return;
        }

        container.innerHTML = tokens.map((token, i) => `
            <span class="kv-token ${i === highlightIndex ? 'new' : ''}">${token}</span>
        `).join('');
    }

    /**
     * Render cache grid visualization
     */
    renderCacheGrid(containerId, cacheData, isRecompute, highlightRow = -1) {
        const container = this.container.querySelector(`#${containerId}`);

        if (cacheData.length === 0) {
            container.innerHTML = '<div class="cache-empty">Empty</div>';
            return;
        }

        const gridHTML = cacheData.map((row, rowIdx) => {
            const isHighlighted = highlightRow === -1 || rowIdx === highlightRow;
            const rowClass = isHighlighted ? 'highlighted' : '';
            const recomputeClass = isRecompute && isHighlighted ? 'recomputing' : '';

            return `
                <div class="cache-row ${rowClass} ${recomputeClass}">
                    <span class="row-label">t${rowIdx}</span>
                    ${row.map((val, colIdx) => `
                        <div class="cache-cell"
                             style="background: rgba(0, 212, 255, ${val * 0.8})"
                             title="[${rowIdx}][${colIdx}]: ${val.toFixed(3)}">
                        </div>
                    `).join('')}
                </div>
            `;
        }).join('');

        container.innerHTML = `
            <div class="cache-header">
                <span class="row-label"></span>
                ${Array(CONFIG.kvDim).fill(0).map((_, i) => `<span class="col-label">d${i}</span>`).join('')}
            </div>
            ${gridHTML}
        `;
    }

    /**
     * Update computation counters
     */
    updateCounters() {
        const cachedKV = this.stats.withCache.kComputations + this.stats.withCache.vComputations;
        const uncachedKV = this.stats.withoutCache.kComputations + this.stats.withoutCache.vComputations;

        this.container.querySelector('#cached-kv-ops').textContent = cachedKV;
        this.container.querySelector('#cached-attn-ops').textContent = this.stats.withCache.attentionOps;
        this.container.querySelector('#uncached-kv-ops').textContent = uncachedKV;
        this.container.querySelector('#uncached-attn-ops').textContent = this.stats.withoutCache.attentionOps;
    }

    /**
     * Update savings display
     */
    updateSavings() {
        const cachedTotal = this.stats.withCache.kComputations + this.stats.withCache.vComputations;
        const uncachedTotal = this.stats.withoutCache.kComputations + this.stats.withoutCache.vComputations;

        let savingsPercent = 0;
        if (uncachedTotal > 0) {
            savingsPercent = ((uncachedTotal - cachedTotal) / uncachedTotal * 100);
        }

        const savingsBar = this.container.querySelector('#savings-bar');
        const savingsText = this.container.querySelector('#savings-text');
        const savingsDetail = this.container.querySelector('#savings-detail');

        savingsBar.style.width = `${Math.min(savingsPercent, 100)}%`;
        savingsText.textContent = `${savingsPercent.toFixed(0)}% saved`;

        if (this.currentStep > 0) {
            const saved = uncachedTotal - cachedTotal;
            savingsDetail.innerHTML = `
                <strong>${saved}</strong> K/V computations saved
                (${cachedTotal} with cache vs ${uncachedTotal} without).
                Q·K score ops are unchanged.
            `;
        } else {
            savingsDetail.textContent = 'Generate tokens to see savings accumulate';
        }
    }

    /**
     * Reset the demo
     */
    reset() {
        this.isPlaying = false;
        this.currentStep = 0;
        this.generatedTokens = [];
        this.cache = { keys: [], values: [] };
        this.stats = {
            withCache: { kComputations: 0, vComputations: 0, attentionOps: 0 },
            withoutCache: { kComputations: 0, vComputations: 0, attentionOps: 0 }
        };

        this.container.querySelector('#play-text').textContent =
            this.mode === 'step' ? 'Generate Next' : 'Play All';

        this.renderInitialState();
    }

    /**
     * Sleep helper for animations
     */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

export default KVCacheDemoUI;
