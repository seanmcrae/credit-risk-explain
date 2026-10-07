"""Render the README's Mermaid flowchart as a static SVG, so the docs site needs no JavaScript.

Supports the subset the README uses: ``flowchart TB`` or ``LR``, nodes written as
``ID["label"]`` or a bare ``ID``, and ``-->`` edges. Nodes are placed in ranks by longest path
from a source and ordered within a rank by the mean position of their parents, which keeps
crossings low for small pipelines.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from html import escape

_NODE = r'(\w+)(?:\["([^"]*)"\])?'
_EDGE = re.compile(rf"^\s*{_NODE}\s*-->\s*{_NODE}\s*$")

NODE_WIDTH, LINE_HEIGHT, PAD_Y = 200, 16, 9
COL_GAP, ROW_GAP, RANK_GAP, MARGIN = 28, 16, 34, 12
WRAP_CHARS = 30


@dataclass
class Flowchart:
    direction: str = "TB"
    labels: dict[str, str] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)


def parse(source: str) -> Flowchart:
    """Parse a Mermaid flowchart; raises ValueError on anything unsupported."""
    lines = [ln.strip() for ln in source.strip().splitlines() if ln.strip()]
    header = re.fullmatch(r"(?:flowchart|graph) (TB|TD|LR)", lines[0]) if lines else None
    if header is None:
        raise ValueError("expected a 'flowchart TB' or 'flowchart LR' diagram")
    chart = Flowchart(direction="LR" if header.group(1) == "LR" else "TB")
    for line in lines[1:]:
        match = _EDGE.match(line)
        if match is None:
            raise ValueError(f"unsupported flowchart line: {line!r}")
        src, src_label, dst, dst_label = match.groups()
        for node, label in ((src, src_label), (dst, dst_label)):
            if label is not None:
                chart.labels[node] = label
            chart.labels.setdefault(node, node)
        chart.edges.append((src, dst))
    return chart


def layers(chart: Flowchart) -> dict[str, int]:
    """Column per node: the length of the longest path reaching it from any source."""
    parents: dict[str, list[str]] = {n: [] for n in chart.labels}
    for src, dst in chart.edges:
        parents[dst].append(src)
    depth: dict[str, int] = {}

    def visit(node: str, trail: frozenset[str]) -> int:
        if node in trail:
            raise ValueError(f"cycle through {node!r}")
        if node not in depth:
            depth[node] = max((visit(p, trail | {node}) + 1 for p in parents[node]), default=0)
        return depth[node]

    for node in chart.labels:
        visit(node, frozenset())
    return depth


def _columns(chart: Flowchart, depth: dict[str, int]) -> list[list[str]]:
    columns: list[list[str]] = [[] for _ in range(max(depth.values()) + 1)]
    for node in chart.labels:  # declaration order is the starting order
        columns[depth[node]].append(node)
    position = {n: float(i) for col in columns for i, n in enumerate(col)}
    for col in columns[1:]:
        for node in col:
            ups = [position[s] for s, d in chart.edges if d == node]
            position[node] = sum(ups) / len(ups) if ups else position[node]
        col.sort(key=lambda n: position[n])
        for i, node in enumerate(col):
            position[node] = float(i)
    return columns


def to_svg(chart: Flowchart, title: str = "Architecture") -> str:
    depth = layers(chart)
    columns = _columns(chart, depth)
    wrapped = {n: textwrap.wrap(label, WRAP_CHARS) or [n] for n, label in chart.labels.items()}
    height = {n: len(lines) * LINE_HEIGHT + 2 * PAD_Y for n, lines in wrapped.items()}
    box: dict[str, tuple[float, float]] = {}  # node -> (x, y) of the top-left corner
    if chart.direction == "LR":
        extents = [sum(height[n] for n in col) + ROW_GAP * (len(col) - 1) for col in columns]
        total_h = max(extents) + 2 * MARGIN
        total_w = len(columns) * NODE_WIDTH + (len(columns) - 1) * COL_GAP + 2 * MARGIN
        for c, col in enumerate(columns):
            x = float(MARGIN + c * (NODE_WIDTH + COL_GAP))
            y = (total_h - extents[c]) / 2
            for node in col:
                box[node] = (x, y)
                y += height[node] + ROW_GAP
    else:
        row_h = [max(height[n] for n in row) for row in columns]
        widths = [len(row) * NODE_WIDTH + (len(row) - 1) * COL_GAP for row in columns]
        total_w = max(widths) + 2 * MARGIN
        total_h = sum(row_h) + RANK_GAP * (len(columns) - 1) + 2 * MARGIN
        y = float(MARGIN)
        for r, row in enumerate(columns):
            x = (total_w - widths[r]) / 2
            for node in row:
                box[node] = (x, y + (row_h[r] - height[node]) / 2)
                x += NODE_WIDTH + COL_GAP
            y += row_h[r] + RANK_GAP

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w:.0f} {total_h:.0f}" '
        f'class="flowchart" role="img" aria-label="{escape(title)}">',
        f"<title>{escape(title)}</title>",
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
        'class="fc-arrow"/></marker></defs>',
    ]
    for src, dst in chart.edges:
        (x1, y1), (x2, y2) = box[src], box[dst]
        if chart.direction == "LR":
            sx, sy = x1 + NODE_WIDTH, y1 + height[src] / 2
            ex, ey = x2, y2 + height[dst] / 2
            bend = (ex - sx) / 2
            c1, c2 = f"{sx + bend:.1f},{sy:.1f}", f"{ex - bend:.1f},{ey:.1f}"
        else:
            sx, sy = x1 + NODE_WIDTH / 2, y1 + height[src]
            ex, ey = x2 + NODE_WIDTH / 2, y2
            bend = (ey - sy) / 2
            c1, c2 = f"{sx:.1f},{sy + bend:.1f}", f"{ex:.1f},{ey - bend:.1f}"
        parts.append(
            f'<path class="fc-edge" d="M{sx:.1f},{sy:.1f} C{c1} {c2} {ex:.1f},{ey:.1f}" '
            'marker-end="url(#arrow)"/>'
        )
    for node, (left, top) in box.items():
        parts.append(
            f'<g class="fc-node"><rect x="{left:.1f}" y="{top:.1f}" width="{NODE_WIDTH}" '
            f'height="{height[node]}" rx="6"/>'
        )
        cx = left + NODE_WIDTH / 2
        for i, line in enumerate(wrapped[node]):
            ty = top + PAD_Y + (i + 0.78) * LINE_HEIGHT
            parts.append(f'<text x="{cx:.1f}" y="{ty:.1f}">{escape(line)}</text>')
        parts.append("</g>")
    parts.append("</svg>")
    return "\n".join(parts)
