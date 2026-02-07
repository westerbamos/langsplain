/**
 * SVG diagram rendering and interactions using D3.js
 * Renders the main transformer architecture diagram
 */

import { clamp, getCenteredBox, getHorizontalBounds } from './diagram-layout-utils.js';

// Component definitions with positions and metadata
const COMPONENTS = {
    input: {
        id: 'input-box',
        label: 'Input Text',
        sublabel: '"The cat sat on the..."',
        y: 0,
        height: 60,
        color: '#a855f7',
        infoKey: 'tokenization'
    },
    embeddings: {
        id: 'embedding-layer',
        label: 'Token Embeddings + Positional Encoding',
        y: 80,
        height: 50,
        color: '#22c55e',
        infoKey: 'embeddings'
    },
    transformerBlock: {
        id: 'transformer-block',
        label: 'Transformer Block',
        sublabel: '×N layers',
        y: 150,
        height: 280,
        color: '#00d4ff',
        isContainer: true,
        children: ['attention', 'residual1', 'ffn', 'residual2']
    },
    attention: {
        id: 'attention-block',
        label: 'Multi-Head Self-Attention',
        sublabel: 'Q, K, V matrices',
        y: 180,
        height: 70,
        color: '#00d4ff',
        infoKey: 'attention',
        parent: 'transformerBlock',
        hasDemo: true
    },
    residual1: {
        id: 'residual-1',
        label: '+ Residual + LayerNorm',
        y: 260,
        height: 30,
        color: '#f97316',
        infoKey: 'residuals',
        parent: 'transformerBlock',
        isSmall: true
    },
    ffn: {
        id: 'ffn-block',
        label: 'Feed-Forward Network',
        sublabel: 'or MOE Layer',
        y: 300,
        height: 70,
        color: '#ec4899',
        infoKey: 'ffn',
        parent: 'transformerBlock',
        hasToggle: true
    },
    moe: {
        id: 'moe-block',
        label: 'Mixture of Experts',
        sublabel: 'Router + 8 Experts',
        y: 300,
        height: 70,
        color: '#eab308',
        infoKey: 'moe',
        parent: 'transformerBlock',
        hidden: true,
        hasDemo: true
    },
    residual2: {
        id: 'residual-2',
        label: '+ Residual + LayerNorm',
        y: 380,
        height: 30,
        color: '#f97316',
        infoKey: 'residuals',
        parent: 'transformerBlock',
        isSmall: true
    },
    outputProjection: {
        id: 'output-layer',
        label: 'Output Projection',
        sublabel: 'to vocabulary',
        y: 450,
        height: 50,
        color: '#06b6d4',
        infoKey: 'outputProjection'
    },
    output: {
        id: 'output-box',
        label: 'Next Token Prediction',
        sublabel: '"mat" → probability distribution',
        y: 520,
        height: 60,
        color: '#a855f7',
        infoKey: 'generation'
    }
};

// Arrow connections between components
const ARROWS = [
    { from: 'input', to: 'embeddings' },
    { from: 'embeddings', to: 'attention' },
    { from: 'attention', to: 'residual1' },
    { from: 'residual1', to: 'ffn' },
    { from: 'ffn', to: 'residual2' },
    { from: 'residual2', to: 'outputProjection' },
    { from: 'outputProjection', to: 'output' }
];

const DIAGRAM_HEIGHT = 620;
const BOX_MAX_WIDTH = 400;
const BOX_SIDE_PADDING = 40;
const LAYOUT_BOUNDS_PADDING = 8;
const CONTAINER_INSET = 15;

let svg = null;
let currentHighlight = null;
let moeMode = false;
let onComponentClick = null;
let resizeHandler = null;

/**
 * Initialize the diagram
 * @param {string} containerId - ID of container element
 * @param {function} clickHandler - Handler for component clicks
 */
export function initDiagram(containerId, clickHandler) {
    destroyDiagram();

    onComponentClick = clickHandler;
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = '';
    const width = container.clientWidth || 700;
    const layout = getArchitectureLayout(width);

    // Create SVG with D3
    svg = d3.select(`#${containerId}`)
        .append('svg')
        .attr('width', '100%')
        .attr('height', DIAGRAM_HEIGHT)
        .attr('viewBox', `0 0 ${width} ${DIAGRAM_HEIGHT}`)
        .attr('class', 'diagram-svg');

    addDefs();
    renderDiagram(layout);
    applyMOEVisualState();

    // Handle resize
    resizeHandler = () => {
        const newWidth = container.clientWidth || width;
        const nextLayout = getArchitectureLayout(newWidth);

        svg.attr('height', DIAGRAM_HEIGHT)
            .attr('viewBox', `0 0 ${newWidth} ${DIAGRAM_HEIGHT}`);

        svg.selectAll('*').remove();
        addDefs();
        renderDiagram(nextLayout);
        applyMOEVisualState();

        if (currentHighlight && COMPONENTS[currentHighlight]) {
            const keyToRestore = currentHighlight;
            currentHighlight = null;
            highlightComponent(keyToRestore);
        }
    };

    window.addEventListener('resize', resizeHandler);
}

function addDefs() {
    if (!svg) return;

    // Add defs for gradients and filters
    const defs = svg.append('defs');

    // Glow filter
    const glow = defs.append('filter')
        .attr('id', 'glow')
        .attr('x', '-50%')
        .attr('y', '-50%')
        .attr('width', '200%')
        .attr('height', '200%');

    glow.append('feGaussianBlur')
        .attr('stdDeviation', '3')
        .attr('result', 'coloredBlur');

    const glowMerge = glow.append('feMerge');
    glowMerge.append('feMergeNode').attr('in', 'coloredBlur');
    glowMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    // Arrow marker
    defs.append('marker')
        .attr('id', 'arrowhead')
        .attr('viewBox', '0 -5 10 10')
        .attr('refX', 8)
        .attr('refY', 0)
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,-5L10,0L0,5')
        .attr('fill', '#606060');
}

function getArchitectureLayout(width) {
    const bounds = getHorizontalBounds(width, LAYOUT_BOUNDS_PADDING);
    const centered = getCenteredBox(width, BOX_MAX_WIDTH, BOX_SIDE_PADDING);
    const containerRightEdge = centered.rightEdge + CONTAINER_INSET;
    const labelFitsRight = (bounds.maxX - containerRightEdge) >= 72;
    const mode = labelFitsRight ? 'wide' : 'narrow';

    return {
        bounds,
        mode,
        boxX: centered.boxX,
        boxWidth: centered.boxWidth,
        centerX: centered.centerX,
        containerLabelX: mode === 'wide'
            ? clamp(containerRightEdge + 8, bounds.minX + 6, bounds.maxX - 6)
            : clamp(centered.leftEdge + 8, bounds.minX + 6, bounds.maxX - 6),
        containerLabelY: COMPONENTS.transformerBlock.y + 20,
        containerLabelAnchor: 'start'
    };
}

function renderDiagram(layout) {
    renderComponents(layout);
    renderArrows(layout);
    renderLayerIndicator(layout);
}

export function destroyDiagram() {
    if (resizeHandler) {
        window.removeEventListener('resize', resizeHandler);
        resizeHandler = null;
    }

    if (svg) {
        svg.remove();
        svg = null;
    }

    currentHighlight = null;
    onComponentClick = null;
    moeMode = false;
}

/**
 * Render all components
 */
function renderComponents(layout) {
    const { boxX, boxWidth } = layout;

    // Create groups for each component
    Object.entries(COMPONENTS).forEach(([key, comp]) => {
        if (comp.isContainer) {
            renderContainer(comp, layout);
        } else if (!comp.hidden || key === 'moe') {
            renderBox(key, comp, boxX, boxWidth);
        }
    });
}

/**
 * Render a container (transformer block)
 */
function renderContainer(comp, layout) {
    const { boxX, boxWidth, containerLabelX, containerLabelY, containerLabelAnchor } = layout;
    const group = svg.append('g')
        .attr('class', 'component-container')
        .attr('id', comp.id);

    // Container rectangle with dashed border
    group.append('rect')
        .attr('x', boxX - CONTAINER_INSET)
        .attr('y', comp.y)
        .attr('width', boxWidth + CONTAINER_INSET * 2)
        .attr('height', comp.height)
        .attr('rx', 8)
        .attr('fill', 'transparent')
        .attr('stroke', comp.color)
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '8,4')
        .attr('opacity', 0.5);

    // Label
    group.append('text')
        .attr('x', containerLabelX)
        .attr('y', containerLabelY)
        .attr('text-anchor', containerLabelAnchor)
        .attr('class', 'container-label')
        .attr('fill', comp.color)
        .attr('font-size', '12px')
        .attr('opacity', 0.7)
        .text(comp.sublabel);
}

/**
 * Render a component box
 */
function renderBox(key, comp, x, width) {
    const group = svg.append('g')
        .attr('class', `component-box ${comp.hidden ? 'hidden' : ''}`)
        .attr('id', comp.id)
        .attr('data-key', key)
        .attr('cursor', 'pointer')
        .on('click', (event) => handleClick(event, key, comp))
        .on('mouseenter', () => handleHover(key, comp, true))
        .on('mouseleave', () => handleHover(key, comp, false));

    const boxHeight = comp.height;
    const innerWidth = comp.parent ? width - 40 : width;
    const innerX = comp.parent ? x + 20 : x;

    // Main rectangle
    group.append('rect')
        .attr('class', 'box-bg')
        .attr('x', innerX)
        .attr('y', comp.y)
        .attr('width', innerWidth)
        .attr('height', boxHeight)
        .attr('rx', comp.isSmall ? 4 : 8)
        .attr('fill', '#1a1a1a')
        .attr('stroke', comp.color)
        .attr('stroke-width', comp.isSmall ? 1 : 2);

    // Label
    group.append('text')
        .attr('class', 'box-label')
        .attr('x', innerX + innerWidth / 2)
        .attr('y', comp.y + (comp.sublabel ? boxHeight / 2 - 6 : boxHeight / 2 + 4))
        .attr('text-anchor', 'middle')
        .attr('fill', '#f0f0f0')
        .attr('font-size', comp.isSmall ? '11px' : '14px')
        .attr('font-weight', '500')
        .text(comp.label);

    // Sublabel
    if (comp.sublabel) {
        group.append('text')
            .attr('class', 'box-sublabel')
            .attr('x', innerX + innerWidth / 2)
            .attr('y', comp.y + boxHeight / 2 + 12)
            .attr('text-anchor', 'middle')
            .attr('fill', '#a0a0a0')
            .attr('font-size', '11px')
            .text(comp.sublabel);
    }

    // Demo button indicator
    if (comp.hasDemo) {
        group.append('circle')
            .attr('cx', innerX + innerWidth - 20)
            .attr('cy', comp.y + boxHeight / 2)
            .attr('r', 8)
            .attr('fill', comp.color)
            .attr('opacity', 0.3);

        group.append('text')
            .attr('x', innerX + innerWidth - 20)
            .attr('y', comp.y + boxHeight / 2 + 4)
            .attr('text-anchor', 'middle')
            .attr('fill', '#fff')
            .attr('font-size', '10px')
            .text('▶');
    }

    // Toggle indicator for FFN/MOE
    if (comp.hasToggle) {
        group.append('rect')
            .attr('class', 'toggle-indicator')
            .attr('x', innerX + innerWidth - 60)
            .attr('y', comp.y + boxHeight - 20)
            .attr('width', 50)
            .attr('height', 14)
            .attr('rx', 7)
            .attr('fill', 'rgba(255,255,255,0.1)')
            .attr('cursor', 'pointer');

        group.append('text')
            .attr('class', 'toggle-text')
            .attr('x', innerX + innerWidth - 35)
            .attr('y', comp.y + boxHeight - 10)
            .attr('text-anchor', 'middle')
            .attr('fill', '#a0a0a0')
            .attr('font-size', '9px')
            .text('→ MOE');
    }
}

/**
 * Render arrows between components
 */
function renderArrows(layout) {
    const { centerX } = layout;

    ARROWS.forEach(arrow => {
        const fromComp = COMPONENTS[arrow.from];
        const toComp = COMPONENTS[arrow.to];

        // Skip if component is hidden (for MOE arrows)
        if (fromComp.hidden || toComp.hidden) return;

        const fromY = fromComp.y + fromComp.height;
        const toY = toComp.y;

        svg.append('line')
            .attr('class', 'arrow')
            .attr('x1', centerX)
            .attr('y1', fromY + 2)
            .attr('x2', centerX)
            .attr('y2', toY - 2)
            .attr('stroke', '#606060')
            .attr('stroke-width', 2)
            .attr('marker-end', 'url(#arrowhead)');
    });
}

/**
 * Render layer stacking indicator
 */
function renderLayerIndicator(layout) {
    const { centerX } = layout;

    // "... more layers ..." text
    svg.append('text')
        .attr('x', centerX)
        .attr('y', COMPONENTS.transformerBlock.y + COMPONENTS.transformerBlock.height + 35)
        .attr('text-anchor', 'middle')
        .attr('fill', '#606060')
        .attr('font-size', '11px')
        .attr('font-style', 'italic')
        .text('... stacked layers ...');
}

/**
 * Handle component click
 */
function handleClick(event, key, comp) {
    const targetClasses = event?.target?.classList;
    if (comp.hasToggle && (targetClasses?.contains('toggle-indicator') || targetClasses?.contains('toggle-text'))) {
        toggleMOE();
        return;
    }

    highlightComponent(key);

    if (onComponentClick) {
        onComponentClick(comp.infoKey || key, comp);
    }
}

/**
 * Handle component hover
 */
function handleHover(key, comp, isEnter) {
    const group = svg.select(`#${comp.id}`);

    if (isEnter) {
        group.select('.box-bg')
            .transition()
            .duration(200)
            .attr('filter', 'url(#glow)')
            .attr('stroke-width', comp.isSmall ? 2 : 3);
    } else if (currentHighlight !== key) {
        group.select('.box-bg')
            .transition()
            .duration(200)
            .attr('filter', null)
            .attr('stroke-width', comp.isSmall ? 1 : 2);
    }
}

/**
 * Highlight a specific component
 */
export function highlightComponent(key) {
    // Remove previous highlight
    if (currentHighlight) {
        const prevComp = COMPONENTS[currentHighlight];
        if (prevComp) {
            svg.select(`#${prevComp.id} .box-bg`)
                .transition()
                .duration(200)
                .attr('filter', null)
                .attr('stroke-width', prevComp.isSmall ? 1 : 2);
        }
    }

    // Apply new highlight
    currentHighlight = key;
    const comp = COMPONENTS[key];

    if (comp) {
        svg.select(`#${comp.id} .box-bg`)
            .transition()
            .duration(200)
            .attr('filter', 'url(#glow)')
            .attr('stroke-width', comp.isSmall ? 2 : 3);
    }
}

/**
 * Clear all highlights
 */
export function clearHighlight() {
    if (currentHighlight) {
        const comp = COMPONENTS[currentHighlight];
        if (comp) {
            svg.select(`#${comp.id} .box-bg`)
                .transition()
                .duration(200)
                .attr('filter', null)
                .attr('stroke-width', comp.isSmall ? 1 : 2);
        }
    }
    currentHighlight = null;
}

/**
 * Toggle between FFN and MOE views
 */
export function toggleMOE() {
    moeMode = !moeMode;
    applyMOEVisualState({ animate: true });
}

function applyMOEVisualState({ animate = false } = {}) {
    if (!svg) return;

    const ffnGroup = svg.select('#ffn-block');
    const moeGroup = svg.select('#moe-block');
    if (ffnGroup.empty() || moeGroup.empty()) return;

    const duration = animate ? 300 : 0;

    if (moeMode) {
        ffnGroup.classed('hidden', true)
            .transition()
            .duration(duration)
            .attr('opacity', 0);

        moeGroup.classed('hidden', false)
            .attr('opacity', 0)
            .transition()
            .duration(duration)
            .attr('opacity', 1);
    } else {
        moeGroup.classed('hidden', true)
            .transition()
            .duration(duration)
            .attr('opacity', 0);

        ffnGroup.classed('hidden', false)
            .attr('opacity', 0)
            .transition()
            .duration(duration)
            .attr('opacity', 1);
    }

    // Update toggle button text
    svg.select('#ffn-block .toggle-text')
        .text(moeMode ? '← FFN' : '→ MOE');
}

/**
 * Get current MOE mode state
 */
export function isMOEMode() {
    return moeMode;
}

/**
 * Animate token flow through the diagram
 * @param {Array<{text: string, id: number}>} tokens - Tokens to animate
 */
export function animateTokenFlow(tokens) {
    const container = svg.node().parentElement;
    const width = container.clientWidth;
    const centerX = width / 2;

    // Create token visuals
    const tokenGroup = svg.append('g').attr('class', 'token-flow');

    const tokenWidth = 50;
    const startX = centerX - (tokens.length * tokenWidth) / 2;

    tokens.forEach((token, i) => {
        const tokenG = tokenGroup.append('g')
            .attr('class', 'flowing-token')
            .attr('transform', `translate(${startX + i * tokenWidth}, -30)`);

        tokenG.append('rect')
            .attr('width', tokenWidth - 4)
            .attr('height', 24)
            .attr('rx', 4)
            .attr('fill', '#a855f7')
            .attr('opacity', 0.8);

        tokenG.append('text')
            .attr('x', (tokenWidth - 4) / 2)
            .attr('y', 16)
            .attr('text-anchor', 'middle')
            .attr('fill', '#fff')
            .attr('font-size', '11px')
            .text(token.text.length > 5 ? token.text.slice(0, 5) + '…' : token.text);
    });

    // Animate through layers
    const layers = ['input', 'embeddings', 'attention', 'residual1', 'ffn', 'residual2', 'outputProjection', 'output'];

    let delay = 0;
    layers.forEach((layer, idx) => {
        const comp = COMPONENTS[layer];
        const targetY = comp.y + comp.height / 2 - 12;

        tokenGroup.selectAll('.flowing-token')
            .transition()
            .delay(delay)
            .duration(500)
            .ease(d3.easeCubicInOut)
            .attr('transform', (d, i) => `translate(${startX + i * tokenWidth}, ${targetY})`);

        delay += 600;
    });

    // Fade out at the end
    tokenGroup
        .transition()
        .delay(delay + 200)
        .duration(500)
        .attr('opacity', 0)
        .remove();
}

/**
 * Get component by info key
 */
export function getComponentByInfoKey(infoKey) {
    return Object.entries(COMPONENTS).find(([_, comp]) => comp.infoKey === infoKey)?.[0] || null;
}

/**
 * Get all interactive components
 */
export function getInteractiveComponents() {
    return Object.entries(COMPONENTS)
        .filter(([_, comp]) => comp.infoKey && !comp.hidden)
        .map(([key, comp]) => ({ key, ...comp }));
}
