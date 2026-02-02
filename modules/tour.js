/**
 * Guided tour state machine
 * Walks users through the transformer architecture
 */

import { highlightComponent, clearHighlight } from './diagram.js';

// Tour step definitions
const TOUR_STEPS = [
    {
        target: '#input-box',
        componentKey: 'input',
        title: 'Tokenization',
        content: 'Text input is first broken into tokens - smaller pieces that the model can understand. Words, subwords, and punctuation each become separate tokens.',
        position: 'right'
    },
    {
        target: '#embedding-layer',
        componentKey: 'embeddings',
        title: 'Embeddings',
        content: 'Each token is converted into a vector of numbers (an embedding) that captures its meaning. Positional information is added so the model knows word order.',
        position: 'right'
    },
    {
        target: '#attention-block',
        componentKey: 'attention',
        title: 'Self-Attention',
        content: 'The attention mechanism lets each token look at all other tokens to understand context. This is the key innovation that makes transformers so powerful!',
        position: 'right',
        highlight: true
    },
    {
        target: '#residual-1',
        componentKey: 'residual1',
        title: 'Residual + LayerNorm',
        content: 'Skip connections add the input back to the output, while LayerNorm keeps activations stable. Together they help deep transformer stacks train reliably.',
        position: 'right'
    },
    {
        target: '#ffn-block',
        componentKey: 'ffn',
        title: 'Feed-Forward Network',
        content: 'Each token passes through a neural network independently. This is where much of the "knowledge" is stored in the model\'s parameters.',
        position: 'right'
    },
    {
        target: '#transformer-block',
        componentKey: null,
        title: 'Layer Stacking',
        content: 'This entire transformer block is repeated many times. GPT-3 has 96 layers, while smaller models might have 12-32. More layers = more complex reasoning.',
        position: 'right'
    },
    {
        target: '#output-layer',
        componentKey: 'outputProjection',
        title: 'Output Projection',
        content: 'The final hidden state is projected to vocabulary size to produce logits - raw scores for every possible next token.',
        position: 'right'
    },
    {
        target: '#output-box',
        componentKey: 'output',
        title: 'Token Generation',
        content: 'A probability distribution over all tokens determines the next word. This process repeats to generate text one token at a time.',
        position: 'right'
    },
    {
        target: '#ffn-block .toggle-indicator',
        componentKey: 'moe',
        title: 'Mixture of Experts',
        content: 'Some models use MOE layers instead of standard FFN. A router selects which "expert" networks to use for each token, allowing massive scale with efficient computation.',
        position: 'right',
        action: 'showMOE'
    },
    {
        target: '.demo-btn[data-demo="attention"]',
        componentKey: null,
        title: 'Try the Demos!',
        content: 'Ready to see these concepts in action? Try the interactive Attention Demo to visualize how tokens attend to each other, or the MOE Demo to see expert routing!',
        position: 'left',
        isFinal: true
    }
];

class Tour {
    constructor() {
        this.steps = TOUR_STEPS;
        this.currentStep = 0;
        this.active = false;
        this.tooltip = null;
        this.overlay = null;
        this.onStepChange = null;
        this.onComplete = null;
        this.onExit = null;
    }

    /**
     * Initialize tour UI elements
     */
    init() {
        // Create overlay
        this.overlay = document.createElement('div');
        this.overlay.className = 'tour-overlay';
        this.overlay.innerHTML = `
            <div class="tour-spotlight"></div>
        `;
        document.body.appendChild(this.overlay);

        // Create tooltip
        this.tooltip = document.createElement('div');
        this.tooltip.className = 'tour-tooltip';
        this.tooltip.innerHTML = `
            <button class="tour-close" aria-label="Close tour">&times;</button>
            <h3 class="tour-title"></h3>
            <p class="tour-content"></p>
            <div class="tour-footer">
                <span class="tour-progress"></span>
                <div class="tour-buttons">
                    <button class="tour-btn tour-prev">Previous</button>
                    <button class="tour-btn tour-next">Next</button>
                </div>
            </div>
        `;
        document.body.appendChild(this.tooltip);

        // Event listeners
        this.tooltip.querySelector('.tour-close').addEventListener('click', () => this.exit());
        this.tooltip.querySelector('.tour-prev').addEventListener('click', () => this.prev());
        this.tooltip.querySelector('.tour-next').addEventListener('click', () => this.next());
        this.overlay.addEventListener('click', (e) => {
            if (e.target === this.overlay) this.exit();
        });

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (!this.active) return;

            switch (e.key) {
                case 'ArrowRight':
                case 'Enter':
                    this.next();
                    break;
                case 'ArrowLeft':
                    this.prev();
                    break;
                case 'Escape':
                    this.exit();
                    break;
            }
        });
    }

    /**
     * Start the tour
     */
    start() {
        this.active = true;
        this.currentStep = 0;
        this.overlay.classList.add('active');
        this.tooltip.classList.add('active');
        this.showStep(0);
    }

    /**
     * Show a specific step
     */
    showStep(index) {
        if (index < 0 || index >= this.steps.length) return;

        this.currentStep = index;
        const step = this.steps[index];

        // Update tooltip content
        this.tooltip.querySelector('.tour-title').textContent = step.title;
        this.tooltip.querySelector('.tour-content').textContent = step.content;
        this.tooltip.querySelector('.tour-progress').textContent = `Step ${index + 1} of ${this.steps.length}`;

        // Update buttons
        const prevBtn = this.tooltip.querySelector('.tour-prev');
        const nextBtn = this.tooltip.querySelector('.tour-next');
        prevBtn.style.display = index === 0 ? 'none' : 'inline-block';
        nextBtn.textContent = step.isFinal ? 'Finish' : 'Next';

        // Position spotlight and tooltip
        this.positionElements(step);

        // Highlight component in diagram
        if (step.componentKey) {
            highlightComponent(step.componentKey);
        } else {
            clearHighlight();
        }

        // Execute any step action
        if (step.action === 'showMOE') {
            // Dispatch event to toggle MOE view
            window.dispatchEvent(new CustomEvent('tour:showMOE'));
        }

        // Callback
        if (this.onStepChange) {
            this.onStepChange(index, step);
        }
    }

    /**
     * Position spotlight and tooltip around target
     */
    positionElements(step) {
        const target = document.querySelector(step.target);

        if (!target) {
            // If target not found, center the tooltip
            this.tooltip.style.left = '50%';
            this.tooltip.style.top = '50%';
            this.tooltip.style.transform = 'translate(-50%, -50%)';
            this.overlay.querySelector('.tour-spotlight').style.display = 'none';
            return;
        }

        const rect = target.getBoundingClientRect();
        const spotlight = this.overlay.querySelector('.tour-spotlight');

        // Position spotlight
        spotlight.style.display = 'block';
        spotlight.style.left = `${rect.left - 10}px`;
        spotlight.style.top = `${rect.top - 10}px`;
        spotlight.style.width = `${rect.width + 20}px`;
        spotlight.style.height = `${rect.height + 20}px`;

        // Position tooltip
        const tooltipRect = this.tooltip.getBoundingClientRect();
        let left, top;

        switch (step.position) {
            case 'right':
                left = rect.right + 20;
                top = rect.top + rect.height / 2 - tooltipRect.height / 2;
                break;
            case 'left':
                left = rect.left - tooltipRect.width - 20;
                top = rect.top + rect.height / 2 - tooltipRect.height / 2;
                break;
            case 'bottom':
                left = rect.left + rect.width / 2 - tooltipRect.width / 2;
                top = rect.bottom + 20;
                break;
            case 'top':
                left = rect.left + rect.width / 2 - tooltipRect.width / 2;
                top = rect.top - tooltipRect.height - 20;
                break;
            default:
                left = rect.right + 20;
                top = rect.top;
        }

        // Keep tooltip in viewport
        const padding = 20;
        left = Math.max(padding, Math.min(left, window.innerWidth - tooltipRect.width - padding));
        top = Math.max(padding, Math.min(top, window.innerHeight - tooltipRect.height - padding));

        this.tooltip.style.left = `${left}px`;
        this.tooltip.style.top = `${top}px`;
        this.tooltip.style.transform = 'none';
    }

    /**
     * Go to next step
     */
    next() {
        if (this.currentStep >= this.steps.length - 1) {
            this.complete();
        } else {
            this.showStep(this.currentStep + 1);
        }
    }

    /**
     * Go to previous step
     */
    prev() {
        if (this.currentStep > 0) {
            this.showStep(this.currentStep - 1);
        }
    }

    /**
     * Jump to specific step
     */
    goToStep(index) {
        this.showStep(index);
    }

    /**
     * Complete the tour
     */
    complete() {
        this.active = false;
        this.overlay.classList.remove('active');
        this.tooltip.classList.remove('active');
        clearHighlight();

        if (this.onComplete) {
            this.onComplete();
        }
    }

    /**
     * Exit the tour early
     */
    exit() {
        this.active = false;
        this.overlay.classList.remove('active');
        this.tooltip.classList.remove('active');
        clearHighlight();

        if (this.onExit) {
            this.onExit();
        }
    }

    /**
     * Check if tour is active
     */
    isActive() {
        return this.active;
    }

    /**
     * Get current step
     */
    getCurrentStep() {
        return this.currentStep;
    }
}

// Singleton instance
const tour = new Tour();

export default tour;

export function initTour() {
    tour.init();
    return tour;
}

export function startTour() {
    tour.start();
}

export function exitTour() {
    tour.exit();
}
