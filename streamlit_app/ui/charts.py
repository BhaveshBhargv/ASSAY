"""Plotly charts in the dashboard colours.

Both functions return None when there's nothing to plot.
"""
from __future__ import annotations

import plotly.graph_objects as go

from ui.theme import PALETTE, SEVERITY, STATUS_COLOR

_LAYOUT = dict(
    font=dict(family="IBM Plex Sans, sans-serif", color=PALETTE["ink"], size=13),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=8, r=18, t=10, b=8),
    hoverlabel=dict(font=dict(family="IBM Plex Mono, monospace", size=12)),
)


def _deviation(value, low, high):
    """Distance outside the reference range in multiples of its width (0 if inside)."""
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        return 0.0
    span = max(high - low, 1e-6)
    if value > high:
        return (value - high) / span
    if value < low:
        return (value - low) / span
    return 0.0


def deviation_chart(rule_result):
    items = []
    for b in rule_result.biomarkers:
        rng = b.reference_range or {}
        dev = _deviation(b.value, rng.get("low"), rng.get("high"))
        if abs(dev) > 1e-9:
            items.append((b, dev))
    if not items:
        return None

    items.sort(key=lambda t: t[1])
    names = [b.name for b, _ in items]
    devs = [round(d, 3) for _, d in items]
    colors = [STATUS_COLOR.get(b.status.value, PALETTE["muted"]) for b, _ in items]
    custom = [[f"{b.value:g} {b.unit}", b.status.value] for b, _ in items]

    fig = go.Figure(go.Bar(
        x=devs, y=names, orientation="h", marker=dict(color=colors),
        customdata=custom,
        hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>status: %{customdata[1]}"
                      "<br>range-distance: %{x}<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1.5, line_color=PALETTE["ink2"])
    fig.update_layout(
        height=max(150, 46 * len(items) + 40), showlegend=False, **_LAYOUT,
    )
    fig.update_xaxes(title="deviation outside reference range (× range width)",
                     zeroline=False, gridcolor=PALETTE["line"], title_font=dict(size=11))
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def rf_probability_chart(fused):
    probs = fused.rf_probabilities or {}
    order = ["normal", "borderline", "serious"]
    vals = [probs.get(c) for c in order]
    if not any(isinstance(v, (int, float)) for v in vals):
        return None
    vals = [float(v or 0) for v in vals]
    colors = [SEVERITY[c] for c in order]

    fig = go.Figure(go.Bar(
        x=[c.title() for c in order], y=vals, marker=dict(color=colors),
        text=[f"{v*100:.0f}%" for v in vals], textposition="outside",
        textfont=dict(family="IBM Plex Mono, monospace"),
        hovertemplate="<b>%{x}</b><br>probability: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(height=260, showlegend=False, yaxis_range=[0, 1.12], **_LAYOUT)
    fig.update_yaxes(title="model probability", gridcolor=PALETTE["line"],
                     title_font=dict(size=11), tickformat=".0%")
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
    return fig
