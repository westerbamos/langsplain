/**
 * Langsplain - Main Application
 * Multi-section architecture/training/inference explainer.
 */

import * as architectureDiagram from './modules/diagram.js';
import * as trainingDiagram from './modules/training-diagram.js';
import * as inferenceDiagram from './modules/inference-diagram.js';
import { initTour, startTour } from './modules/tour.js';
import { initSectionSwitcher, setActiveSection as setActiveSectionTab, getActiveSection } from './modules/section-switcher.js';
import { AttentionDemoUI } from './modules/attention-demo.js';
import { MOEDemoUI } from './modules/moe-demo.js';
import { SamplingDemoUI } from './modules/sampling-demo.js';
import { KVCacheDemoUI } from './modules/kv-cache-demo.js';
import { GradientDemoUI } from './modules/gradient-demo.js';
import { LossDemoUI } from './modules/loss-demo.js';
import { GenerationDemoUI } from './modules/generation-demo.js';
import { tokenize, simulateBPE, getTokenColor } from './modules/tokenizer.js';

const ARCHITECTURE_INFO_CONTENT = {
    tokenization: {
        title: 'Tokenization',
        simple: `
            <p>Before the model can process text, it splits text into tokens. Tokens are usually whole words, subwords, punctuation, or bytes.</p>
            <p>Common fragments become reusable units, which helps models handle rare and unseen words.</p>
            <div class="bpe-demo-section">
                <h4>Try BPE Tokenization</h4>
                <p class="bpe-note">This is a <strong>simplified simulation</strong> with a tiny rule set for teaching only.</p>
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
            <p>Modern LLM tokenizers are subword models (often BPE-like variants): frequent pairs merge into new symbols.</p>
            <div class="formula">vocabulary = merge_frequent_pairs(base_symbols)</div>
            <ul>
                <li>Unknown words are decomposed into known pieces.</li>
                <li>Tokenizer choice affects sequence length and efficiency.</li>
                <li>Training/inference details are covered in <a href="#" class="cross-section-link" data-section="training" data-info-key="datasetPrep">Training -> Dataset Prep</a> and <a href="#" class="cross-section-link" data-section="inference" data-info-key="tokenize">Inference -> Tokenize</a>.</li>
            </ul>
        `
    },
    embeddings: {
        title: 'Token Embeddings',
        simple: `
            <p>Each token ID maps to a learned vector. Similar meanings tend to land in nearby regions of vector space.</p>
            <p>Position information is added so the model knows order.</p>
        `,
        technical: `
            <div class="formula">x_0 = token_embedding[token_id] + positional_encoding[position]</div>
            <p>Common positional schemes include RoPE and ALiBi variants, depending on model family.</p>
        `
    },
    attention: {
        title: 'Self-Attention',
        simple: `
            <p>Attention lets each token weigh other earlier tokens to gather useful context.</p>
            <p>This is what enables reference tracking and long-context interactions.</p>
        `,
        technical: `
            <div class="formula">Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V</div>
            <p>Causal masking blocks access to future tokens in decoder-only models.</p>
        `
    },
    residuals: {
        title: 'Residual + LayerNorm',
        simple: `
            <p>Residual connections keep information flowing through deep stacks, while LayerNorm stabilizes activations.</p>
        `,
        technical: `
            <div class="formula">x = x + Sublayer(LayerNorm(x))</div>
            <p>Pre-norm is standard in modern LLMs because it improves optimization stability at scale.</p>
        `
    },
    ffn: {
        title: 'Feed-Forward Network',
        simple: `
            <p>Each token then goes through a per-token MLP that expands and compresses features.</p>
            <p>Many capabilities are stored in these dense parameters.</p>
        `,
        technical: `
            <div class="formula">FFN(x) = W2 * activation(W1 * x + b1) + b2</div>
            <p>Many models use gated variants (for example SwiGLU) for better efficiency/quality.</p>
        `
    },
    moe: {
        title: 'Mixture of Experts (MOE)',
        simple: `
            <p>MOE replaces one dense FFN with multiple experts and a router that picks a subset per token.</p>
            <p>This keeps active compute lower than total parameter count.</p>
        `,
        technical: `
            <div class="formula">MOE(x) = sum_i router(x)_i * expert_i(x)</div>
            <p>Production MOE training includes balancing terms to prevent expert collapse.</p>
        `
    },
    outputProjection: {
        title: 'Output Projection',
        simple: `
            <p>The final hidden state maps back to one score per vocabulary token.</p>
        `,
        technical: `
            <div class="formula">logits = h_final * W_vocab^T</div>
            <p>Those logits feed inference-time sampling in <a href="#" class="cross-section-link" data-section="inference" data-info-key="sampling">Inference -> Sampling</a>.</p>
        `
    },
    generation: {
        title: 'Architecture Output (Hand-off to Inference)',
        simple: `
            <p>At the architecture level, the model emits logits for the next token. The actual decode loop lives in the Inference section.</p>
        `,
        technical: `
            <div class="formula">p(token_t | token_{<t}) = softmax(logits_t)</div>
            <p>Decode details (prefill, KV cache, sampling, stop conditions) are covered in <a href="#" class="cross-section-link" data-section="inference" data-info-key="autoregressiveLoop">Inference -> Autoregressive Loop</a>.</p>
        `
    }
};

const TRAINING_INFO_CONTENT = {
    trainingData: {
        title: 'Training Data',
        simple: `
            <p>Models learn from massive corpora: web text, books, code, docs, and curated instruction data.</p>
        `,
        technical: `
            <p>Quality-sensitive pipelines perform deduplication, filtering, and domain balancing before training.</p>
        `
    },
    datasetPrep: {
        title: 'Dataset Preparation',
        simple: `
            <p>Raw text is tokenized, cleaned, and packed into fixed-length training examples.</p>
        `,
        technical: `
            <p>Sequence packing minimizes padding overhead; sampling/mixing policies control data distribution during training.</p>
            <p>Tokenizer behavior connects to <a href="#" class="cross-section-link" data-section="architecture" data-info-key="tokenization">Architecture -> Tokenization</a>.</p>
        `
    },
    forwardPass: {
        title: 'Forward Pass',
        simple: `
            <p>The model predicts probability distributions for next tokens across the batch.</p>
        `,
        technical: `
            <p>Forward compute is batched matrix math through transformer layers, same core blocks as Architecture.</p>
            <p>See <a href="#" class="cross-section-link" data-section="architecture" data-info-key="attention">Architecture -> Self-Attention</a>.</p>
        `
    },
    lossFunction: {
        title: 'Loss Function',
        simple: `
            <p>Loss measures how wrong predictions are compared with true targets.</p>
        `,
        technical: `
            <div class="formula">L = -sum_t log p(target_t)</div>
            <p>Cross-entropy and perplexity are core diagnostics for language modeling quality.</p>
        `
    },
    backpropagation: {
        title: 'Backpropagation',
        simple: `
            <p>Backprop computes how each parameter contributed to the loss so updates can be directed.</p>
        `,
        technical: `
            <p>Automatic differentiation applies chain rule through the computation graph to produce gradients.</p>
        `
    },
    optimizer: {
        title: 'Optimizer',
        simple: `
            <p>The optimizer updates weights using gradients. This loop repeats across many batches.</p>
        `,
        technical: `
            <p>Adam-family optimizers with warmup/decay schedules are common for large-scale transformer training.</p>
        `
    },
    sft: {
        title: 'Supervised Fine-Tuning (SFT)',
        simple: `
            <p>After base pretraining, SFT teaches instruction following and task-specific behavior.</p>
        `,
        technical: `
            <p>SFT usually uses curated prompt-response pairs and objective weighting tuned for helpfulness and style.</p>
        `
    },
    preferenceTuning: {
        title: 'Preference Tuning (RLHF / DPO / PPO)',
        simple: `
            <p>Preference tuning aligns model outputs with human or policy preferences.</p>
        `,
        technical: `
            <p>Common routes include reward-model RL (PPO) and direct preference objectives (DPO-style methods).</p>
            <p>This affects final inference behavior in <a href="#" class="cross-section-link" data-section="inference" data-info-key="sampling">Inference -> Sampling</a>.</p>
        `
    }
};

const INFERENCE_INFO_CONTENT = {
    inputPrompt: {
        title: 'Input Prompt',
        simple: `
            <p>Inference starts from a structured prompt (for example system + user messages) constrained by context window size.</p>
        `,
        technical: `
            <p>Prompt templates, role formatting, and truncation policies shape the effective context before tokenization.</p>
        `
    },
    tokenize: {
        title: 'Tokenize',
        simple: `
            <p>Prompt text is converted to token IDs before model execution.</p>
        `,
        technical: `
            <p>Tokenizer choice impacts latency and context utilization.</p>
            <p>See tokenizer fundamentals in <a href="#" class="cross-section-link" data-section="architecture" data-info-key="tokenization">Architecture -> Tokenization</a>.</p>
        `
    },
    prefillPhase: {
        title: 'Prefill Phase',
        simple: `
            <p>Prefill runs all prompt tokens in parallel and initializes hidden states plus KV cache.</p>
        `,
        technical: `
            <p>This is usually the highest-throughput phase due to batched matrix operations.</p>
        `
    },
    autoregressiveLoop: {
        title: 'Autoregressive Decode Loop',
        simple: `
            <p>After prefill, generation proceeds one token at a time: forward pass -> logits -> sample -> append token.</p>
        `,
        technical: `
            <div class="formula">P(text) = product_t P(token_t | token_{<t})</div>
            <p>The loop stops at EOS, configured stop sequences, or max token limits.</p>
        `
    },
    logits: {
        title: 'Logits',
        simple: `
            <p>Logits are raw scores for every vocabulary token at the current step.</p>
        `,
        technical: `
            <p>Softmax over logits yields probabilities used by decoding policies.</p>
        `
    },
    sampling: {
        title: 'Sampling',
        simple: `
            <p>Sampling policy controls creativity and determinism when choosing the next token.</p>
        `,
        technical: `
            <ul>
                <li>Temperature rescales logits.</li>
                <li>Top-k keeps only the k highest-probability candidates.</li>
                <li>Top-p (nucleus) keeps the smallest set crossing probability threshold p.</li>
            </ul>
        `
    },
    stopCondition: {
        title: 'Stop Conditions',
        simple: `
            <p>Generation ends when EOS (end of sequence token) appears, a stop sequence matches, or max tokens are reached.</p>
        `,
        technical: `
            <p>Serving stacks often apply custom business stop criteria on top of core model EOS logic.</p>
        `
    },
    kvCache: {
        title: 'KV Cache',
        simple: `
            <p>KV cache stores past key/value tensors so each new token reuses history instead of recomputing it.</p>
        `,
        technical: `
            <p>This reduces per-token decode compute significantly, at the cost of additional memory.</p>
        `
    },
    detokenize: {
        title: 'Detokenize',
        simple: `
            <p>Generated token IDs are converted back into readable text output.</p>
        `,
        technical: `
            <p>Detokenization merges subword units according to tokenizer-specific decoding rules.</p>
        `
    }
};

const SECTION_CONFIG = {
    architecture: {
        diagram: architectureDiagram,
        infoContent: ARCHITECTURE_INFO_CONTENT
    },
    training: {
        diagram: trainingDiagram,
        infoContent: TRAINING_INFO_CONTENT
    },
    inference: {
        diagram: inferenceDiagram,
        infoContent: INFERENCE_INFO_CONTENT
    }
};

const GLOSSARY = [
    { term: 'Attention', definition: 'A mechanism that lets each token aggregate context from other tokens with learned relevance weights.' },
    { term: 'Autoregressive', definition: 'Generation that predicts one token at a time conditioned on previously generated tokens.' },
    { term: 'Backpropagation', definition: 'Gradient computation through the model graph using chain rule so parameters can be updated.' },
    { term: 'Cross-Entropy', definition: 'Loss function measuring mismatch between predicted probability distribution and target token labels.' },
    { term: 'Detokenization', definition: 'Conversion of token IDs back into output text.' },
    { term: 'Embedding', definition: 'A learned dense vector representation for each token ID.' },
    { term: 'Fine-Tuning (SFT)', definition: 'Post-pretraining supervised training on curated prompt-response pairs.' },
    { term: 'KV Cache', definition: 'Stored key/value tensors reused during decode to avoid recomputing history.' },
    { term: 'Layer Normalization', definition: 'Feature-wise normalization used to stabilize optimization in deep networks.' },
    { term: 'Logits', definition: 'Raw model output scores before softmax normalization.' },
    { term: 'Mixture of Experts (MOE)', definition: 'Sparse layer with multiple experts where a router selects active experts per token.' },
    { term: 'Perplexity', definition: 'exp(cross-entropy); a common language-model quality metric where lower is better.' },
    { term: 'Prefill', definition: 'Initial inference pass over all prompt tokens before one-token-at-a-time decoding.' },
    { term: 'RLHF', definition: 'Reinforcement Learning from Human Feedback, a family of preference-alignment approaches.' },
    { term: 'Sampling', definition: 'Policy for selecting next tokens from probability distributions (temperature/top-k/top-p).' },
    { term: 'Softmax', definition: 'Function that converts logits into a probability distribution summing to 1.' },
    { term: 'Tokenizer', definition: 'Algorithm that converts text to tokens and back again for model processing.' },
    { term: 'Transformer Block', definition: 'Core repeated unit: attention + FFN (+ residual/normalization structure).' }
];

const MOBILE_BREAKPOINT = 768;
const PANEL_COLLISION_GAP = 12;

let currentView = 'home';
let currentSection = 'architecture';
let infoPanelOpen = false;
let currentInfoKey = null;
let currentDiagramModule = null;
let sectionRenderToken = 0;

let attentionDemo = null;
let moeDemo = null;
let samplingDemo = null;
let kvCacheDemo = null;
let gradientDemo = null;
let lossDemo = null;
let generationDemo = null;

function init() {
    initTour();

    attentionDemo = new AttentionDemoUI('attention-demo-content');
    moeDemo = new MOEDemoUI('moe-demo-content');
    samplingDemo = new SamplingDemoUI('sampling-demo-content');
    kvCacheDemo = new KVCacheDemoUI('kvcache-demo-content');
    gradientDemo = new GradientDemoUI('gradient-demo-content');
    lossDemo = new LossDemoUI('loss-demo-content');
    generationDemo = new GenerationDemoUI('generation-demo-content');

    setupNavigation();
    setupPanelEvents();
    setupDemoButtons();
    setupTourButton();
    setupModalEvents();
    setupGlossary();
    setupTourEventBridge();

    initSectionSwitcher({
        defaultSection: currentSection,
        onChange: (section) => {
            switchSection(section);
        }
    });

    switchSection(currentSection, { force: true });

    window.addEventListener('resize', handleResize);
}

function setupNavigation() {
    document.querySelectorAll('.nav-link').forEach((link) => {
        link.addEventListener('click', (event) => {
            event.preventDefault();
            switchView(link.dataset.view);
        });
    });
}

function switchView(view) {
    currentView = view;

    document.querySelectorAll('.nav-link').forEach((link) => {
        link.classList.toggle('active', link.dataset.view === view);
    });

    document.querySelectorAll('.view').forEach((node) => {
        node.classList.toggle('active', node.id === `${view}-view`);
    });

    closeInfoPanel();
}

function setupTourEventBridge() {
    window.addEventListener('tour:ensureHome', () => {
        switchView('home');
    });

    window.addEventListener('tour:switchSection', (event) => {
        const section = event.detail?.section;
        if (!section) return;
        switchSection(section);
    });

    window.addEventListener('tour:highlight', (event) => {
        const componentKey = event.detail?.componentKey;
        if (!componentKey) return;
        currentDiagramModule?.highlightComponent?.(componentKey);
    });

    window.addEventListener('tour:clearHighlight', () => {
        currentDiagramModule?.clearHighlight?.();
    });

    window.addEventListener('tour:showMOE', () => {
        if (currentSection === 'architecture') {
            if (!architectureDiagram.isMOEMode()) {
                architectureDiagram.toggleMOE();
            }
        }
    });
}

function switchSection(section, { force = false, afterRender = null } = {}) {
    if (!SECTION_CONFIG[section]) return;
    if (!force && section === currentSection) {
        updateDemoButtonVisibility();
        if (typeof afterRender === 'function') {
            afterRender();
        }
        return;
    }

    closeInfoPanel();

    currentDiagramModule?.destroyDiagram?.();

    currentSection = section;

    if (getActiveSection() !== section) {
        setActiveSectionTab(section);
    }

    updateDemoButtonVisibility();

    const renderToken = ++sectionRenderToken;
    const config = SECTION_CONFIG[currentSection];

    requestAnimationFrame(() => {
        if (renderToken !== sectionRenderToken) return;

        currentDiagramModule = config.diagram;
        currentDiagramModule.initDiagram('diagram-container', handleComponentClick);

        if (typeof afterRender === 'function') {
            afterRender();
        }
    });
}

function updateDemoButtonVisibility() {
    document.querySelectorAll('.demo-btn[data-sections]').forEach((btn) => {
        const sections = btn.dataset.sections.split(',').map((item) => item.trim());
        const visible = sections.includes(currentSection);
        btn.classList.toggle('hidden', !visible);
        btn.setAttribute('aria-hidden', String(!visible));
        btn.disabled = !visible;
    });
}

function handleComponentClick(infoKey) {
    const content = SECTION_CONFIG[currentSection].infoContent;
    if (infoKey && content[infoKey]) {
        openInfoPanel(infoKey);
    }
}

function syncDiagramPanelOffset() {
    const diagramSection = document.querySelector('.diagram-section');
    if (!diagramSection) return;

    if (!infoPanelOpen || window.innerWidth <= MOBILE_BREAKPOINT) {
        diagramSection.classList.remove('panel-open');
        return;
    }

    const diagramContainer = document.getElementById('diagram-container');
    const panel = document.getElementById('info-panel');
    if (!diagramContainer || !panel) {
        diagramSection.classList.remove('panel-open');
        return;
    }

    const { right: diagramRight } = diagramContainer.getBoundingClientRect();
    const panelWidth = panel.offsetWidth;
    const panelLeft = window.innerWidth - panelWidth;
    const shouldOffsetDiagram = diagramRight + PANEL_COLLISION_GAP > panelLeft;

    diagramSection.classList.toggle('panel-open', shouldOffsetDiagram);
}

function openInfoPanel(infoKey) {
    const content = SECTION_CONFIG[currentSection].infoContent[infoKey];
    if (!content) return;

    const panel = document.getElementById('info-panel');
    panel.querySelector('.panel-title').textContent = content.title;
    panel.querySelector('.simple-explanation').innerHTML = content.simple;
    panel.querySelector('.technical-content').innerHTML = content.technical;

    currentInfoKey = infoKey;
    infoPanelOpen = true;

    panel.classList.add('open');
    syncDiagramPanelOffset();

    highlightForInfoKey(infoKey);

    if (infoKey === 'tokenization') {
        setupBPEDemo();
    }

    setupCrossSectionLinks(panel);
}

function highlightForInfoKey(infoKey) {
    if (!currentDiagramModule?.highlightComponent) return;

    const mappedKey = currentDiagramModule.getComponentByInfoKey?.(infoKey);
    const keyToHighlight = mappedKey || infoKey;
    currentDiagramModule.highlightComponent(keyToHighlight);
}

function setupCrossSectionLinks(panel) {
    panel.querySelectorAll('.cross-section-link').forEach((link) => {
        link.addEventListener('click', (event) => {
            event.preventDefault();
            const section = link.dataset.section;
            const infoKey = link.dataset.infoKey;
            if (!SECTION_CONFIG[section]) return;

            switchSection(section, {
                afterRender: () => {
                    if (infoKey && SECTION_CONFIG[section].infoContent[infoKey]) {
                        openInfoPanel(infoKey);
                    }
                }
            });
        });
    });
}

function closeInfoPanel() {
    infoPanelOpen = false;
    currentInfoKey = null;

    document.getElementById('info-panel').classList.remove('open');
    document.querySelector('.diagram-section').classList.remove('panel-open');

    currentDiagramModule?.clearHighlight?.();
}

function setupPanelEvents() {
    const closeBtn = document.querySelector('.panel-close');
    closeBtn.addEventListener('click', closeInfoPanel);

    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && infoPanelOpen) {
            closeInfoPanel();
        }
    });
}

function setupDemoButtons() {
    const demoActions = {
        attention: () => {
            openModal('attention-modal');
            attentionDemo.init();
        },
        moe: () => {
            openModal('moe-modal');
            moeDemo.init();
        },
        sampling: () => {
            openModal('sampling-modal');
            samplingDemo.init();
        },
        kvcache: () => {
            openModal('kvcache-modal');
            kvCacheDemo.init();
        },
        flow: () => {
            if (currentSection !== 'architecture') return;
            const tokens = tokenize('The cat sat');
            architectureDiagram.animateTokenFlow(tokens);
        },
        gradient: () => {
            openModal('gradient-modal');
            gradientDemo.init();
        },
        loss: () => {
            openModal('loss-modal');
            lossDemo.init();
        },
        generation: () => {
            openModal('generation-modal');
            generationDemo.init();
        }
    };

    document.querySelectorAll('.demo-btn[data-demo]').forEach((button) => {
        button.addEventListener('click', () => {
            if (button.classList.contains('hidden')) return;
            const demo = button.dataset.demo;
            const action = demoActions[demo];
            if (action) action();
        });
    });
}

function setupTourButton() {
    const tourBtn = document.querySelector('.tour-start-btn');
    tourBtn.addEventListener('click', () => {
        switchView('home');
        startTour();
    });
}

function setupModalEvents() {
    document.querySelectorAll('.modal-close').forEach((button) => {
        button.addEventListener('click', () => {
            const modal = button.closest('.modal-overlay');
            closeModal(modal.id);
        });
    });

    document.querySelectorAll('.modal-overlay').forEach((overlay) => {
        overlay.addEventListener('click', (event) => {
            if (event.target === overlay) {
                closeModal(overlay.id);
            }
        });
    });

    document.addEventListener('keydown', (event) => {
        if (event.key !== 'Escape') return;

        document.querySelectorAll('.modal-overlay.open').forEach((modal) => {
            closeModal(modal.id);
        });
    });
}

function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (!modal) return;

    modal.classList.add('open');
    document.body.style.overflow = 'hidden';
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (!modal) return;

    // Stop background demo loops when their modal closes.
    switch (modalId) {
        case 'gradient-modal':
            gradientDemo?.stopAutoRun?.();
            break;
        case 'loss-modal':
            lossDemo?.stopAuto?.();
            break;
        case 'generation-modal':
            generationDemo?.stopAuto?.();
            break;
        case 'kvcache-modal':
            kvCacheDemo?.stopPlayback?.();
            break;
        default:
            break;
    }

    modal.classList.remove('open');

    const hasOpenModal = document.querySelector('.modal-overlay.open');
    if (!hasOpenModal) {
        document.body.style.overflow = '';
    }
}

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
    input.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') runBPE();
    });

    document.querySelectorAll('.bpe-suggest-btn').forEach((button) => {
        button.addEventListener('click', () => {
            input.value = button.dataset.word;
            runBPE();
        });
    });

    runBPE();
}

function renderBPEVisualization(result) {
    const container = document.getElementById('bpe-visualization');
    if (!container) return;

    let html = '<div class="bpe-steps">';

    result.mergeSteps.forEach((step, index) => {
        const isLast = index === result.mergeSteps.length - 1;

        html += `
            <div class="bpe-step ${isLast ? 'final' : ''}">
                <div class="step-header">
                    <span class="step-number">${step.step === 0 ? 'Start' : `Step ${step.step}`}</span>
                    ${step.frequency ? `<span class="step-freq">freq: ${step.frequency}</span>` : ''}
                </div>
                <div class="step-tokens">
                    ${step.tokens.map((token, tokenIndex) => {
            const isNew = step.mergedIndex !== undefined && tokenIndex === step.mergedIndex;
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

    html += `
        <div class="bpe-summary">
            <strong>${result.finalTokens.length}</strong> final tokens from
            <strong>${result.input.length}</strong> characters
            (${result.totalMerges} merges)
        </div>
    `;

    container.innerHTML = html;
}

function setupGlossary() {
    const container = document.getElementById('glossary-list');
    const searchInput = document.getElementById('glossary-search');

    function renderGlossary(filter = '') {
        const lowered = filter.toLowerCase();
        const filtered = GLOSSARY.filter((item) =>
            item.term.toLowerCase().includes(lowered) ||
            item.definition.toLowerCase().includes(lowered)
        );

        container.innerHTML = filtered.map((item) => `
            <div class="glossary-item">
                <div class="glossary-term">${item.term}</div>
                <div class="glossary-definition">${item.definition}</div>
            </div>
        `).join('');
    }

    renderGlossary();

    searchInput.addEventListener('input', (event) => {
        renderGlossary(event.target.value);
    });
}

function handleResize() {
    if (infoPanelOpen) {
        syncDiagramPanelOffset();
        return;
    }

    document.querySelector('.diagram-section')?.classList.remove('panel-open');
}

document.addEventListener('DOMContentLoaded', init);
