/**
 * Training pipeline diagram rendering and interactions using D3.js
 */

import { clamp, getCenteredBox, getHorizontalBounds } from './diagram-layout-utils.js';

const COMPONENTS = {
    trainingData: {
        id: 'training-data',
        label: 'Training Data',
        sublabel: 'Web, books, code, docs',
        y: 20,
        height: 56,
        color: '#22c55e',
        infoKey: 'trainingData'
    },
    datasetPrep: {
        id: 'dataset-prep',
        label: 'Dataset Preparation',
        sublabel: 'Tokenization, filtering, packing',
        y: 95,
        height: 56,
        color: '#14b8a6',
        infoKey: 'datasetPrep'
    },
    forwardPass: {
        id: 'forward-pass',
        label: 'Forward Pass',
        sublabel: 'Predict next-token distribution',
        y: 170,
        height: 56,
        color: '#00d4ff',
        infoKey: 'forwardPass'
    },
    lossFunction: {
        id: 'loss-function',
        label: 'Loss Function',
        sublabel: 'Cross-entropy over targets',
        y: 245,
        height: 56,
        color: '#a855f7',
        infoKey: 'lossFunction'
    },
    backpropagation: {
        id: 'backpropagation',
        label: 'Backpropagation',
        sublabel: 'Compute gradients via chain rule',
        y: 320,
        height: 56,
        color: '#f97316',
        infoKey: 'backpropagation'
    },
    optimizer: {
        id: 'optimizer',
        label: 'Optimizer Update',
        sublabel: 'Adam/SGD + schedule',
        y: 395,
        height: 56,
        color: '#eab308',
        infoKey: 'optimizer'
    },
    sft: {
        id: 'sft',
        label: 'Supervised Fine-Tuning (SFT)',
        sublabel: 'Instruction-following adaptation',
        y: 470,
        height: 52,
        color: '#ec4899',
        infoKey: 'sft'
    },
    preferenceTuning: {
        id: 'preference-tuning',
        label: 'Preference Tuning (DPO/PPO)',
        sublabel: 'Human preference alignment',
        y: 540,
        height: 52,
        color: '#06b6d4',
        infoKey: 'preferenceTuning'
    }
};

const MAIN_ARROWS = [
    { from: 'trainingData', to: 'datasetPrep' },
    { from: 'datasetPrep', to: 'forwardPass' },
    { from: 'forwardPass', to: 'lossFunction' },
    { from: 'lossFunction', to: 'backpropagation' },
    { from: 'backpropagation', to: 'optimizer' }
];

const DIAGRAM_HEIGHT = 680;
const BOX_MAX_WIDTH = 460;
const BOX_SIDE_PADDING = 50;
const LOOP_BOUNDS_PADDING = 8;
const LOOP_WIDE_MIN_RIGHT_SPACE = 92;
const LOOP_WIDE_START_OFFSET = 10;
const LOOP_WIDE_LANE_OFFSET = 58;
const LOOP_NARROW_START_OFFSET = 6;
const LOOP_NARROW_LANE_OFFSET = 24;

let svg = null;
let currentHighlight = null;
let onComponentClick = null;
let resizeHandler = null;

export function initDiagram(containerId, clickHandler) {
    destroyDiagram();

    onComponentClick = clickHandler;

    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = '';

    const width = container.clientWidth || 700;
    const layout = getTrainingLayout(width);

    svg = d3.select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', DIAGRAM_HEIGHT)
        .attr('viewBox', `0 0 ${width} ${DIAGRAM_HEIGHT}`)
        .attr('class', 'diagram-svg');

    addDefs(svg);
    renderDiagram(layout);

    resizeHandler = () => {
        const newWidth = container.clientWidth || width;
        const nextLayout = getTrainingLayout(newWidth);

        svg.attr('height', DIAGRAM_HEIGHT)
            .attr('viewBox', `0 0 ${newWidth} ${DIAGRAM_HEIGHT}`);

        svg.selectAll('*').remove();
        addDefs(svg);
        renderDiagram(nextLayout);

        if (currentHighlight && COMPONENTS[currentHighlight]) {
            const keyToRestore = currentHighlight;
            currentHighlight = null;
            highlightComponent(keyToRestore);
        }
    };

    window.addEventListener('resize', resizeHandler);
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
}

function addDefs(svgEl) {
    const defs = svgEl.append('defs');

    const glow = defs.append('filter')
        .attr('id', 'training-glow')
        .attr('x', '-50%')
        .attr('y', '-50%')
        .attr('width', '200%')
        .attr('height', '200%');

    glow.append('feGaussianBlur')
        .attr('stdDeviation', '3')
        .attr('result', 'coloredBlur');

    const merge = glow.append('feMerge');
    merge.append('feMergeNode').attr('in', 'coloredBlur');
    merge.append('feMergeNode').attr('in', 'SourceGraphic');

    defs.append('marker')
        .attr('id', 'training-arrow')
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

function getTrainingLayout(width) {
    const bounds = getHorizontalBounds(width, LOOP_BOUNDS_PADDING);
    const { boxX, boxWidth, centerX, rightEdge } = getCenteredBox(width, BOX_MAX_WIDTH, BOX_SIDE_PADDING);
    const fromY = COMPONENTS.optimizer.y + COMPONENTS.optimizer.height / 2;
    const toY = COMPONENTS.forwardPass.y + COMPONENTS.forwardPass.height / 2;
    const availableRight = bounds.maxX - rightEdge;
    const mode = availableRight >= LOOP_WIDE_MIN_RIGHT_SPACE ? 'wide' : 'narrow';
    const startOffset = mode === 'wide' ? LOOP_WIDE_START_OFFSET : LOOP_NARROW_START_OFFSET;
    const laneOffset = mode === 'wide' ? LOOP_WIDE_LANE_OFFSET : LOOP_NARROW_LANE_OFFSET;
    const laneX = clamp(rightEdge + laneOffset, bounds.minX + 24, bounds.maxX);
    const startX = clamp(rightEdge + startOffset, bounds.minX + 20, laneX - 6);
    const endX = clamp(rightEdge + LOOP_WIDE_START_OFFSET, bounds.minX + 20, laneX - 6);

    return {
        boxX,
        boxWidth,
        centerX,
        loop: {
            fromY,
            toY,
            startX,
            laneX,
            endX
        }
    };
}

function renderDiagram(layout) {
    renderComponents(layout);
    renderArrows(layout);
    renderLoopArrow(layout);
    renderPostTrainingBranch(layout);
}

function renderComponents(layout) {
    const { boxX, boxWidth, centerX } = layout;

    Object.entries(COMPONENTS).forEach(([key, comp]) => {
        const group = svg.append('g')
            .attr('class', 'component-box')
            .attr('id', comp.id)
            .attr('data-key', key)
            .attr('cursor', 'pointer')
            .on('click', (event) => handleClick(event, key, comp))
            .on('mouseenter', () => handleHover(key, comp, true))
            .on('mouseleave', () => handleHover(key, comp, false));

        group.append('rect')
            .attr('class', 'box-bg')
            .attr('x', boxX)
            .attr('y', comp.y)
            .attr('width', boxWidth)
            .attr('height', comp.height)
            .attr('rx', 8)
            .attr('fill', '#1a1a1a')
            .attr('stroke', comp.color)
            .attr('stroke-width', 2);

        group.append('text')
            .attr('class', 'box-label')
            .attr('x', centerX)
            .attr('y', comp.y + comp.height / 2 - 6)
            .attr('text-anchor', 'middle')
            .attr('fill', '#f0f0f0')
            .attr('font-size', '14px')
            .attr('font-weight', '500')
            .text(comp.label);

        group.append('text')
            .attr('class', 'box-sublabel')
            .attr('x', centerX)
            .attr('y', comp.y + comp.height / 2 + 12)
            .attr('text-anchor', 'middle')
            .attr('fill', '#a0a0a0')
            .attr('font-size', '11px')
            .text(comp.sublabel);
    });
}

function renderArrows(layout) {
    const { centerX } = layout;

    MAIN_ARROWS.forEach((arrow) => {
        const fromComp = COMPONENTS[arrow.from];
        const toComp = COMPONENTS[arrow.to];

        svg.append('line')
            .attr('class', 'arrow')
            .attr('x1', centerX)
            .attr('y1', fromComp.y + fromComp.height + 2)
            .attr('x2', centerX)
            .attr('y2', toComp.y - 2)
            .attr('stroke', '#606060')
            .attr('stroke-width', 2)
            .attr('marker-end', 'url(#training-arrow)');
    });
}

function renderLoopArrow(layout) {
    const { loop } = layout;

    const path = [
        `M ${loop.startX} ${loop.fromY}`,
        `L ${loop.laneX} ${loop.fromY}`,
        `L ${loop.laneX} ${loop.toY}`,
        `L ${loop.endX} ${loop.toY}`
    ].join(' ');

    svg.append('path')
        .attr('d', path)
        .attr('fill', 'none')
        .attr('stroke', '#00d4ff')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '6,4')
        .attr('marker-end', 'url(#training-arrow)')
        .attr('opacity', 0.85);
}

function renderPostTrainingBranch(layout) {
    const { centerX } = layout;

    svg.append('line')
        .attr('x1', centerX)
        .attr('y1', COMPONENTS.optimizer.y + COMPONENTS.optimizer.height + 2)
        .attr('x2', centerX)
        .attr('y2', COMPONENTS.sft.y - 2)
        .attr('stroke', '#606060')
        .attr('stroke-width', 2)
        .attr('marker-end', 'url(#training-arrow)');

    svg.append('line')
        .attr('x1', centerX)
        .attr('y1', COMPONENTS.sft.y + COMPONENTS.sft.height + 2)
        .attr('x2', centerX)
        .attr('y2', COMPONENTS.preferenceTuning.y - 2)
        .attr('stroke', '#606060')
        .attr('stroke-width', 2)
        .attr('marker-end', 'url(#training-arrow)');

    svg.append('text')
        .attr('x', centerX)
        .attr('y', 660)
        .attr('text-anchor', 'middle')
        .attr('fill', '#a0a0a0')
        .attr('font-size', '11px')
        .text('post-training alignment stages');
}

function handleClick(event, key, comp) {
    event.stopPropagation();
    highlightComponent(key);

    if (typeof onComponentClick === 'function') {
        onComponentClick(comp.infoKey || key, comp);
    }
}

function handleHover(key, comp, isEnter) {
    if (!svg) return;

    const group = svg.select(`#${comp.id}`);
    if (group.empty()) return;

    if (isEnter) {
        group.select('.box-bg')
            .transition()
            .duration(180)
            .attr('filter', 'url(#training-glow)')
            .attr('stroke-width', 3);
        return;
    }

    if (currentHighlight !== key) {
        group.select('.box-bg')
            .transition()
            .duration(180)
            .attr('filter', null)
            .attr('stroke-width', 2);
    }
}

export function highlightComponent(key) {
    if (!svg) return;

    if (currentHighlight && COMPONENTS[currentHighlight]) {
        svg.select(`#${COMPONENTS[currentHighlight].id} .box-bg`)
            .transition()
            .duration(180)
            .attr('filter', null)
            .attr('stroke-width', 2);
    }

    currentHighlight = key;

    if (COMPONENTS[key]) {
        svg.select(`#${COMPONENTS[key].id} .box-bg`)
            .transition()
            .duration(180)
            .attr('filter', 'url(#training-glow)')
            .attr('stroke-width', 3);
    }
}

export function clearHighlight() {
    if (!svg || !currentHighlight || !COMPONENTS[currentHighlight]) {
        currentHighlight = null;
        return;
    }

    svg.select(`#${COMPONENTS[currentHighlight].id} .box-bg`)
        .transition()
        .duration(180)
        .attr('filter', null)
        .attr('stroke-width', 2);

    currentHighlight = null;
}

export function getComponentByInfoKey(infoKey) {
    return Object.entries(COMPONENTS).find(([_, comp]) => comp.infoKey === infoKey)?.[0] || null;
}
