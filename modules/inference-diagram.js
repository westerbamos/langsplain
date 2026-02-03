/**
 * Inference pipeline diagram rendering and interactions using D3.js
 */

import { clamp, getCenteredBox, getHorizontalBounds } from './diagram-layout-utils.js';

const COMPONENTS = {
    inputPrompt: {
        id: 'input-prompt',
        label: 'Input Prompt',
        sublabel: 'System + user context',
        y: 20,
        height: 56,
        color: '#22c55e',
        infoKey: 'inputPrompt'
    },
    tokenize: {
        id: 'inference-tokenize',
        label: 'Tokenize',
        sublabel: 'Text -> token IDs',
        y: 95,
        height: 56,
        color: '#14b8a6',
        infoKey: 'tokenize'
    },
    prefillPhase: {
        id: 'prefill-phase',
        label: 'Prefill Phase',
        sublabel: 'Process full prompt in parallel',
        y: 170,
        height: 56,
        color: '#00d4ff',
        infoKey: 'prefillPhase'
    },
    autoregressiveLoop: {
        id: 'decode-loop',
        label: 'Autoregressive Decode Loop',
        sublabel: 'One token at a time',
        y: 250,
        height: 230,
        color: '#a855f7',
        isContainer: true,
        infoKey: 'autoregressiveLoop'
    },
    logits: {
        id: 'decode-logits',
        label: 'Compute Logits',
        sublabel: 'Vocabulary-sized scores',
        y: 285,
        height: 48,
        color: '#06b6d4',
        parent: 'autoregressiveLoop',
        infoKey: 'logits'
    },
    sampling: {
        id: 'decode-sampling',
        label: 'Sample Next Token',
        sublabel: 'Temperature / Top-k / Top-p',
        y: 350,
        height: 48,
        color: '#eab308',
        parent: 'autoregressiveLoop',
        infoKey: 'sampling'
    },
    stopCondition: {
        id: 'decode-stop',
        label: 'Stop Condition Check',
        sublabel: 'EOS / stop seq / max tokens',
        y: 415,
        height: 48,
        color: '#f97316',
        parent: 'autoregressiveLoop',
        infoKey: 'stopCondition'
    },
    kvCache: {
        id: 'kv-cache-box',
        label: 'KV Cache',
        sublabel: 'Reuse past keys/values',
        y: 305,
        height: 78,
        width: 180,
        color: '#ec4899',
        infoKey: 'kvCache',
        isSideBox: true
    },
    detokenize: {
        id: 'detokenize',
        label: 'Detokenize Output',
        sublabel: 'Token IDs -> readable text',
        y: 505,
        height: 56,
        color: '#22c55e',
        infoKey: 'detokenize'
    }
};

const BASE_LAYOUT = {
    kvY: 285,
    detokenizeY: 505,
    height: 600
};

let svg = null;
let onComponentClick = null;
let currentHighlight = null;
let resizeHandler = null;

export function initDiagram(containerId, clickHandler) {
    destroyDiagram();

    onComponentClick = clickHandler;
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = '';

    const width = container.clientWidth || 720;
    const layout = getInferenceLayout(width);

    svg = d3.select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', layout.height)
        .attr('viewBox', `0 0 ${width} ${layout.height}`)
        .attr('class', 'diagram-svg');

    addDefs(svg);
    applyLayout(layout);
    renderDiagram(width, layout);

    resizeHandler = () => {
        const newWidth = container.clientWidth || width;
        const newLayout = getInferenceLayout(newWidth);
        applyLayout(newLayout);

        svg.attr('height', newLayout.height)
            .attr('viewBox', `0 0 ${newWidth} ${newLayout.height}`);

        svg.selectAll('*').remove();
        addDefs(svg);
        renderDiagram(newWidth, newLayout);
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
        .attr('id', 'inference-glow')
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
        .attr('id', 'inference-arrow')
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

function getInferenceLayout(width) {
    const bounds = getHorizontalBounds(width, 8);
    const sidePadding = 20;
    const laneGap = 36;
    const laneInset = 8;
    const decodeInsetRight = 20;
    const kvWritePortRatioWide = 0.5;
    const kvWritePortInsetNarrow = 2;
    const kvReadPortRatio = 0.5;
    const kvMinWidth = 150;
    const kvMaxWidth = COMPONENTS.kvCache.width;
    const kvWidth = clamp(width - 40, kvMinWidth, kvMaxWidth);

    const wideMainWidth = Math.min(430, width - sidePadding * 2 - kvWidth - laneGap);

    if (wideMainWidth >= 320) {
        const totalWidth = wideMainWidth + laneGap + kvWidth;
        const laneLayout = getCenteredBox(width, totalWidth, sidePadding);
        const mainX = laneLayout.leftEdge;
        const mainRight = mainX + wideMainWidth;
        const kvX = clamp(mainRight + laneGap, bounds.minX + 4, bounds.maxX - kvWidth);
        const loopLaneX = clamp(mainRight + laneInset, mainRight + 4, bounds.maxX - 2);
        const kvWritePortX = clamp(kvX + kvWidth * kvWritePortRatioWide, kvX + 2, kvX + kvWidth - 2);
        const kvWriteLaneX = clamp(kvWritePortX, mainRight + 4, bounds.maxX - 2);

        return {
            mode: 'wide',
            mainX,
            mainWidth: wideMainWidth,
            kvX,
            kvY: BASE_LAYOUT.kvY,
            kvWidth,
            loopLaneX,
            kvWriteLaneX,
            kvWritePortX,
            kvReadPortX: clamp(kvX + kvWidth * kvReadPortRatio, kvX + 2, kvX + kvWidth - 2),
            decodeInX: clamp(mainRight - decodeInsetRight, bounds.minX + 20, bounds.maxX - 20),
            decodeInY: COMPONENTS.sampling.y + COMPONENTS.sampling.height / 2,
            detokenizeY: BASE_LAYOUT.detokenizeY,
            height: BASE_LAYOUT.height
        };
    }

    const narrowMain = getCenteredBox(width, 460, 24);
    const mainWidth = narrowMain.boxWidth;
    const mainX = narrowMain.boxX;
    const mainRight = mainX + mainWidth;
    const kvY = COMPONENTS.autoregressiveLoop.y + COMPONENTS.autoregressiveLoop.height + 10;
    const kvX = clamp(width - kvWidth - 20, bounds.minX + 4, bounds.maxX - kvWidth);
    const detokenizeY = kvY + COMPONENTS.kvCache.height + 26;
    const loopLaneX = clamp(mainRight + laneInset + 2, mainRight + 4, bounds.maxX - 2);
    const kvWriteLaneX = clamp(kvX + kvWidth - kvWritePortInsetNarrow, mainRight + 8, bounds.maxX - 2);

    return {
        mode: 'narrow',
        mainX,
        mainWidth,
        kvX,
        kvY,
        kvWidth,
        loopLaneX,
        kvWriteLaneX,
        kvWritePortX: clamp(kvWriteLaneX, kvX + 2, kvX + kvWidth - 2),
        kvReadPortX: clamp(kvX + kvWidth * kvReadPortRatio, kvX + 2, kvX + kvWidth - 2),
        decodeInX: clamp(mainRight - decodeInsetRight, bounds.minX + 20, bounds.maxX - 20),
        decodeInY: COMPONENTS.sampling.y + COMPONENTS.sampling.height / 2,
        detokenizeY,
        height: detokenizeY + COMPONENTS.detokenize.height + 30
    };
}

function applyLayout(layout) {
    COMPONENTS.kvCache.y = layout.kvY;
    COMPONENTS.detokenize.y = layout.detokenizeY;
}

function renderDiagram(width, layout) {
    renderMainBox('inputPrompt', layout.mainX, layout.mainWidth);
    renderMainBox('tokenize', layout.mainX, layout.mainWidth);
    renderMainBox('prefillPhase', layout.mainX, layout.mainWidth);

    renderContainer(layout.mainX, layout.mainWidth);
    renderInnerLoopBoxes(layout);
    renderSideCache(layout);

    renderMainBox('detokenize', layout.mainX, layout.mainWidth);
    renderMainArrows(layout);
}

function renderMainBox(key, x, width) {
    renderBox(key, COMPONENTS[key], x, width);
}

function renderContainer(x, width) {
    const comp = COMPONENTS.autoregressiveLoop;

    const group = svg.append('g')
        .attr('class', 'component-container')
        .attr('id', comp.id)
        .attr('cursor', 'pointer')
        .on('click', (event) => handleClick(event, 'autoregressiveLoop', comp));

    group.append('rect')
        .attr('x', x - 12)
        .attr('y', comp.y)
        .attr('width', width + 24)
        .attr('height', comp.height)
        .attr('rx', 12)
        .attr('fill', 'rgba(168, 85, 247, 0.08)')
        .attr('stroke', comp.color)
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '8,4');

    group.append('text')
        .attr('x', x + width / 2)
        .attr('y', comp.y + 24)
        .attr('text-anchor', 'middle')
        .attr('fill', '#f0f0f0')
        .attr('font-size', '14px')
        .attr('font-weight', '600')
        .text(comp.label);

    group.append('text')
        .attr('x', x + width / 2)
        .attr('y', comp.y + 42)
        .attr('text-anchor', 'middle')
        .attr('fill', '#a0a0a0')
        .attr('font-size', '11px')
        .text(comp.sublabel);
}

function renderInnerLoopBoxes(layout) {
    const x = layout.mainX;
    const width = layout.mainWidth;
    const centerX = x + width / 2;

    renderBox('logits', COMPONENTS.logits, x + 20, width - 40);
    renderBox('sampling', COMPONENTS.sampling, x + 20, width - 40);
    renderBox('stopCondition', COMPONENTS.stopCondition, x + 20, width - 40);

    svg.append('line')
        .attr('x1', centerX)
        .attr('y1', COMPONENTS.logits.y + COMPONENTS.logits.height + 2)
        .attr('x2', centerX)
        .attr('y2', COMPONENTS.sampling.y - 2)
        .attr('stroke', '#606060')
        .attr('stroke-width', 2)
        .attr('marker-end', 'url(#inference-arrow)');

    svg.append('line')
        .attr('x1', centerX)
        .attr('y1', COMPONENTS.sampling.y + COMPONENTS.sampling.height + 2)
        .attr('x2', centerX)
        .attr('y2', COMPONENTS.stopCondition.y - 2)
        .attr('stroke', '#606060')
        .attr('stroke-width', 2)
        .attr('marker-end', 'url(#inference-arrow)');

    const innerRight = x + width - 20;
    const loopStartX = innerRight + 8;
    const loopTurnX = Math.max(loopStartX + 8, layout.loopLaneX);

    drawConnectorPath([
        { x: loopStartX, y: COMPONENTS.stopCondition.y + COMPONENTS.stopCondition.height / 2 },
        { x: loopTurnX, y: COMPONENTS.stopCondition.y + COMPONENTS.stopCondition.height / 2 },
        { x: loopTurnX, y: COMPONENTS.logits.y + COMPONENTS.logits.height / 2 },
        { x: loopStartX, y: COMPONENTS.logits.y + COMPONENTS.logits.height / 2 }
    ], {
        stroke: '#00d4ff',
        dashed: true
    });

}

function renderSideCache(layout) {
    renderBox('kvCache', COMPONENTS.kvCache, layout.kvX, layout.kvWidth);

    const prefillOut = {
        x: layout.mainX + layout.mainWidth,
        y: COMPONENTS.prefillPhase.y + COMPONENTS.prefillPhase.height * 0.35
    };
    const kvWriteIn = {
        x: layout.kvWritePortX,
        y: COMPONENTS.kvCache.y
    };
    const kvReadOut = {
        x: layout.kvReadPortX,
        y: COMPONENTS.kvCache.y + COMPONENTS.kvCache.height
    };
    const decodeIn = {
        x: layout.decodeInX,
        y: layout.decodeInY
    };

    drawConnectorPath([
        prefillOut,
        { x: layout.kvWriteLaneX, y: prefillOut.y },
        kvWriteIn
    ]);

    drawConnectorPath([
        kvReadOut,
        { x: kvReadOut.x, y: decodeIn.y },
        decodeIn
    ]);
}

function renderMainArrows(layout) {
    const centerX = layout.mainX + layout.mainWidth / 2;
    const topFlow = ['inputPrompt', 'tokenize', 'prefillPhase'];

    topFlow.forEach((key, index) => {
        if (index === topFlow.length - 1) return;

        const fromComp = COMPONENTS[key];
        const toComp = COMPONENTS[topFlow[index + 1]];

        svg.append('line')
            .attr('x1', centerX)
            .attr('y1', fromComp.y + fromComp.height + 2)
            .attr('x2', centerX)
            .attr('y2', toComp.y - 2)
            .attr('stroke', '#606060')
            .attr('stroke-width', 2)
            .attr('marker-end', 'url(#inference-arrow)');
    });

    svg.append('line')
        .attr('x1', centerX)
        .attr('y1', COMPONENTS.prefillPhase.y + COMPONENTS.prefillPhase.height + 2)
        .attr('x2', centerX)
        .attr('y2', COMPONENTS.autoregressiveLoop.y - 2)
        .attr('stroke', '#606060')
        .attr('stroke-width', 2)
        .attr('marker-end', 'url(#inference-arrow)');

    svg.append('line')
        .attr('x1', centerX)
        .attr('y1', COMPONENTS.autoregressiveLoop.y + COMPONENTS.autoregressiveLoop.height + 2)
        .attr('x2', centerX)
        .attr('y2', COMPONENTS.detokenize.y - 2)
        .attr('stroke', '#606060')
        .attr('stroke-width', 2)
        .attr('marker-end', 'url(#inference-arrow)');
}

function drawConnectorPath(points, { stroke = '#606060', dashed = false } = {}) {
    const d = points.map((point, index) =>
        `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`
    ).join(' ');

    const path = svg.append('path')
        .attr('d', d)
        .attr('fill', 'none')
        .attr('stroke', stroke)
        .attr('stroke-width', 2)
        .attr('stroke-linejoin', 'round')
        .attr('stroke-linecap', 'round')
        .attr('marker-end', 'url(#inference-arrow)');

    if (dashed) {
        path.attr('stroke-dasharray', '6,4');
    }
}

function renderBox(key, comp, x, width) {
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
        .attr('x', x)
        .attr('y', comp.y)
        .attr('width', width)
        .attr('height', comp.height)
        .attr('rx', 8)
        .attr('fill', '#1a1a1a')
        .attr('stroke', comp.color)
        .attr('stroke-width', 2);

    group.append('text')
        .attr('class', 'box-label')
        .attr('x', x + width / 2)
        .attr('y', comp.y + comp.height / 2 - 6)
        .attr('text-anchor', 'middle')
        .attr('fill', '#f0f0f0')
        .attr('font-size', '13px')
        .attr('font-weight', '500')
        .text(comp.label);

    group.append('text')
        .attr('class', 'box-sublabel')
        .attr('x', x + width / 2)
        .attr('y', comp.y + comp.height / 2 + 12)
        .attr('text-anchor', 'middle')
        .attr('fill', '#a0a0a0')
        .attr('font-size', '11px')
        .text(comp.sublabel);
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
            .attr('filter', 'url(#inference-glow)')
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
            .attr('filter', 'url(#inference-glow)')
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
