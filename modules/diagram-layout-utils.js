/**
 * Shared horizontal layout helpers for SVG diagrams.
 */

export function clamp(value, min, max) {
    const lower = Math.min(min, max);
    const upper = Math.max(min, max);
    return Math.min(Math.max(value, lower), upper);
}

export function getHorizontalBounds(width, padding = 0) {
    const safeWidth = Number.isFinite(width) ? width : 0;
    const safePadding = Math.max(0, padding);
    const minX = safePadding;
    const maxX = Math.max(minX, safeWidth - safePadding);

    return { minX, maxX };
}

export function getCenteredBox(width, maxBoxWidth, sidePadding = 0) {
    const safeWidth = Number.isFinite(width) ? width : 0;
    const safeMaxBoxWidth = Math.max(0, maxBoxWidth);
    const safeSidePadding = Math.max(0, sidePadding);
    const maxAllowedWidth = Math.max(0, safeWidth - safeSidePadding * 2);
    const boxWidth = Math.min(safeMaxBoxWidth, maxAllowedWidth);
    const centerX = safeWidth / 2;
    const boxX = centerX - boxWidth / 2;

    return {
        boxX,
        boxWidth,
        centerX,
        leftEdge: boxX,
        rightEdge: boxX + boxWidth
    };
}
