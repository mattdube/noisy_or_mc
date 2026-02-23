"""
visualize_distributions_altair.py
===================================
Visualizes the Beta prior distributions for all risk sources
using Altair 5. Produces three interactive HTML files.

Requires:
    pip install altair numpy scipy

Run:
    python visualize_distributions_altair.py

Outputs:
    source_distributions_grid.html    — small-multiple panel per source
    source_distributions_overlay.html — all sources overlaid, click legend to isolate
    source_distributions_summary.html — mean + 90% CI strip chart
"""

import json
import numpy as np
from scipy.stats import beta as beta_dist

try:
    import altair as alt
except ImportError:
    raise ImportError("Altair is not installed. Run: pip install altair")

# ── Load param store ──────────────────────────────────────────────────────────
PARAM_PATH = "params/param_store.json"

with open(PARAM_PATH) as f:
    store = json.load(f)

SOURCE_ORDER = store["source_order"]
SOURCES      = store["sources"]

# ── Palette ───────────────────────────────────────────────────────────────────
COLORS = {
    "S1": "#22d3ee", "S2": "#a78bfa", "S3": "#34d399", "S4": "#f87171",
    "S5": "#60a5fa", "S6": "#fb923c", "S7": "#e879f9", "S8": "#94a3b8",
}

LABELS = {
    "S1": "Very Low Risk · High Confidence",
    "S2": "Low Risk · Low Confidence",
    "S3": "Moderate Risk · Medium Confidence",
    "S4": "High Risk · Very Uncertain",
    "S5": "Low Risk · Solid Evidence",
    "S6": "Moderate Risk · Limited Data",
    "S7": "High Risk · Strong Evidence",
    "S8": "Minimal Risk · Very Well Validated",
}

N_POINTS = 300

# ── Dark theme — applied via .configure() ─────────────────────────────────────
# Only uses properties that exist in Altair 5's schema.
DARK_CONFIG = {
    "background": "#0b1120",
    "view": {
        "fill":   "#0f172a",
        "stroke": "#1e293b",
    },
    "axis": {
        "domainColor":   "#1e293b",
        "gridColor":     "#1e293b",
        "gridDash":      [3, 3],
        "labelColor":    "#64748b",
        "labelFontSize": 10,
        "tickColor":     "#1e293b",
        "titleColor":    "#94a3b8",
        "titleFontSize": 11,
    },
    "title": {
        "color":      "#f1f5f9",
        "fontSize":   14,
        "fontWeight": "bold",
        "anchor":     "start",
        "offset":     10,
    },
    "legend": {
        "labelColor":    "#cbd5e1",
        "labelFontSize": 10,
        "titleColor":    "#94a3b8",
        "titleFontSize": 10,
        "symbolSize":    120,
        "padding":       10,
    },
    "header": {
        "labelColor":    "#94a3b8",
        "labelFontSize": 12,
        "labelFontWeight": "bold",
    },
}


# ── Data builders ─────────────────────────────────────────────────────────────
def build_pdf_rows():
    """Long-format rows with Beta PDF values. Mean is embedded per row."""
    rows = []
    for src in SOURCE_ORDER:
        s    = SOURCES[src]
        a, b = s["alpha"], s["beta"]
        mean = s["mean"]
        std  = s["std"]
        conc = s["expert_concentration"]
        lo   = max(0.001, mean - 4 * std)
        hi   = min(0.999, mean + 4 * std)
        xs   = np.linspace(lo, hi, N_POINTS)
        ys   = beta_dist.pdf(xs, a, b)
        for x, y in zip(xs, ys):
            rows.append({
                "source":        src,
                "x":             round(float(x), 6),
                "pdf":           round(float(y), 6),
                "mean":          mean,
                "std":           std,
                "alpha":         a,
                "beta":          b,
                "concentration": conc,
                "label":         LABELS[src],
            })
    return rows


def build_summary_rows():
    """One row per source: mean + 90% credible interval."""
    rows = []
    for src in SOURCE_ORDER:
        s     = SOURCES[src]
        a, b  = s["alpha"], s["beta"]
        ci_lo = float(beta_dist.ppf(0.05, a, b))
        ci_hi = float(beta_dist.ppf(0.95, a, b))
        rows.append({
            "source":        src,
            "mean":          s["mean"],
            "std":           s["std"],
            "ci_lo":         round(ci_lo, 5),
            "ci_hi":         round(ci_hi, 5),
            "concentration": s["expert_concentration"],
            "label":         LABELS[src],
        })
    return rows


def color_scale():
    return alt.Scale(
        domain=SOURCE_ORDER,
        range=[COLORS[s] for s in SOURCE_ORDER],
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — Small-multiple grid
# ─────────────────────────────────────────────────────────────────────────────
def chart_small_multiples(save_path="source_distributions_grid.html"):
    """
    One panel per source. All three layers (area, line, mean rule) share the
    same alt.Data object — required by Altair 5 for faceted layer charts.
    Mean is a column in the PDF data so no second dataset is needed.
    """
    data = alt.Data(values=build_pdf_rows())

    color_enc = alt.Color("source:N", scale=color_scale(), legend=None)

    tooltips = [
        alt.Tooltip("source:N",        title="Source"),
        alt.Tooltip("x:Q",             title="P(risk)",      format=".3f"),
        alt.Tooltip("pdf:Q",           title="Density",      format=".2f"),
        alt.Tooltip("mean:Q",          title="Mean",         format=".3f"),
        alt.Tooltip("std:Q",           title="Std",          format=".3f"),
        alt.Tooltip("concentration:Q", title="Concentration"),
        alt.Tooltip("label:N",         title="Description"),
    ]

    x_enc = alt.X("x:Q", axis=alt.Axis(format=".0%", title="P(risk | match)"))
    y_enc = alt.Y("pdf:Q", axis=alt.Axis(title="", labels=False, ticks=False, grid=False))

    area = (
        alt.Chart(data)
        .mark_area(opacity=0.18, interpolate="monotone")
        .encode(x=x_enc, y=y_enc, color=color_enc, tooltip=tooltips)
    )

    line = (
        alt.Chart(data)
        .mark_line(opacity=0.9, strokeWidth=2, interpolate="monotone")
        .encode(x=x_enc, y=y_enc, color=color_enc, tooltip=tooltips)
    )

    # Rule uses `mean` column already in every row — same data source, no mixing.
    mean_rule = (
        alt.Chart(data)
        .mark_rule(strokeDash=[4, 3], opacity=0.6, strokeWidth=1.2)
        .encode(x=alt.X("mean:Q"), color=color_enc)
    )

    chart = (
        alt.layer(area, line, mean_rule)
        .properties(width=220, height=150)
        .facet(
            facet=alt.Facet("source:N", sort=SOURCE_ORDER),
            columns=4,
        )
        .properties(title="Beta Prior Distributions — Risk Sources")
        .configure(**DARK_CONFIG)
        .configure_facet(spacing=14)
    )

    chart.save(save_path)
    print(f"[altair] Saved → {save_path}")
    return chart


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — Overlay
# ─────────────────────────────────────────────────────────────────────────────
def chart_overlay(save_path="source_distributions_overlay.html"):
    """
    All sources on one chart. Click legend to isolate a source.
    Mean rules use the same dataset as curves (mean column embedded per row).
    """
    data = alt.Data(values=build_pdf_rows())

    selection = alt.selection_point(fields=["source"], bind="legend")

    color_enc = alt.Color(
        "source:N",
        scale=color_scale(),
        legend=alt.Legend(title="Source  (click to isolate)", symbolStrokeWidth=3),
    )

    x_enc = alt.X(
        "x:Q",
        scale=alt.Scale(domain=[0, 1]),
        axis=alt.Axis(format=".0%", title="P(risk | match)", tickCount=6),
    )
    y_enc = alt.Y("pdf:Q", axis=alt.Axis(title="Density", labels=False, ticks=False))

    tooltips = [
        alt.Tooltip("source:N",        title="Source"),
        alt.Tooltip("x:Q",             title="P(risk)",      format=".3f"),
        alt.Tooltip("pdf:Q",           title="Density",      format=".2f"),
        alt.Tooltip("mean:Q",          title="Mean",         format=".3f"),
        alt.Tooltip("std:Q",           title="Std",          format=".3f"),
        alt.Tooltip("concentration:Q", title="Concentration"),
        alt.Tooltip("label:N",         title="Description"),
    ]

    line = (
        alt.Chart(data)
        .mark_line(strokeWidth=2.2, interpolate="monotone")
        .encode(
            x=x_enc, y=y_enc, color=color_enc,
            opacity=alt.condition(selection, alt.value(0.9), alt.value(0.08)),
            tooltip=tooltips,
        )
        .add_params(selection)
    )

    area = (
        alt.Chart(data)
        .mark_area(opacity=0.12, interpolate="monotone")
        .encode(
            x=x_enc, y=y_enc, color=color_enc,
            opacity=alt.condition(selection, alt.value(0.15), alt.value(0.04)),
        )
        .add_params(selection)
    )

    mean_rule = (
        alt.Chart(data)
        .mark_rule(strokeDash=[3, 4], strokeWidth=1.1)
        .encode(
            x=alt.X("mean:Q"),
            color=color_enc,
            opacity=alt.condition(selection, alt.value(0.6), alt.value(0.06)),
        )
        .add_params(selection)
    )

    chart = (
        alt.layer(area, line, mean_rule)
        .properties(
            width=760,
            height=360,
            title="All Source Prior Distributions — Overlay",
        )
        .configure(**DARK_CONFIG)
    )

    chart.save(save_path)
    print(f"[altair] Saved → {save_path}")
    return chart


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 — Summary strip (mean + 90% CI)
# ─────────────────────────────────────────────────────────────────────────────
def chart_summary_strip(save_path="source_distributions_summary.html"):
    """Mean dot + 90% CI bar per source, sorted by source order."""
    data = alt.Data(values=build_summary_rows())

    x_scale = alt.Scale(domain=[-0.02, 1.05])
    x_fmt   = alt.Axis(format=".0%", title="P(risk | match)")
    y_sort  = alt.Y("source:N", sort=SOURCE_ORDER, axis=alt.Axis(title="", labelFontSize=12))

    def color_for(field="source:N"):
        return alt.Color(field, scale=color_scale(), legend=None)

    tooltips = [
        alt.Tooltip("source:N",        title="Source"),
        alt.Tooltip("mean:Q",          title="Mean",      format=".3f"),
        alt.Tooltip("std:Q",           title="Std",       format=".3f"),
        alt.Tooltip("ci_lo:Q",         title="CI 90% lo", format=".3f"),
        alt.Tooltip("ci_hi:Q",         title="CI 90% hi", format=".3f"),
        alt.Tooltip("concentration:Q", title="Concentration"),
        alt.Tooltip("label:N",         title="Description"),
    ]

    # CI bar (x = lo, x2 = hi)
    ci_bar = (
        alt.Chart(data)
        .mark_bar(opacity=0.28)
        .encode(
            x=alt.X("ci_lo:Q", scale=x_scale, axis=x_fmt),
            x2=alt.X2("ci_hi:Q"),
            y=y_sort,
            color=color_for(),
            tooltip=tooltips,
        )
    )

    # CI end caps (tick marks)
    ci_lo_tick = (
        alt.Chart(data)
        .mark_tick(thickness=2, size=14, opacity=0.7)
        .encode(
            x=alt.X("ci_lo:Q", scale=x_scale),
            y=alt.Y("source:N", sort=SOURCE_ORDER),
            color=color_for(),
        )
    )

    ci_hi_tick = (
        alt.Chart(data)
        .mark_tick(thickness=2, size=14, opacity=0.7)
        .encode(
            x=alt.X("ci_hi:Q", scale=x_scale),
            y=alt.Y("source:N", sort=SOURCE_ORDER),
            color=color_for(),
        )
    )

    # Mean dot
    mean_dot = (
        alt.Chart(data)
        .mark_point(size=120, filled=True, stroke="white", strokeWidth=1.0)
        .encode(
            x=alt.X("mean:Q", scale=x_scale),
            y=alt.Y("source:N", sort=SOURCE_ORDER),
            color=color_for(),
            tooltip=tooltips,
        )
    )

    # Concentration label to the right of each CI bar
    conc_text = (
        alt.Chart(data)
        .mark_text(align="left", dx=6, fontSize=9, opacity=0.6)
        .encode(
            x=alt.X("ci_hi:Q", scale=x_scale),
            y=alt.Y("source:N", sort=SOURCE_ORDER),
            text=alt.Text("concentration:Q", format=".0f"),
            color=alt.value("#64748b"),
        )
    )

    chart = (
        alt.layer(ci_bar, ci_lo_tick, ci_hi_tick, mean_dot, conc_text)
        .properties(
            width=640,
            height=280,
            title="Source Risk Estimates — Mean & 90% Credible Interval",
        )
        .configure(**DARK_CONFIG)
    )

    chart.save(save_path)
    print(f"[altair] Saved → {save_path}")
    return chart


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating Altair 5 visualizations...")
    chart_small_multiples()
    chart_overlay()
    chart_summary_strip()
    print("\nDone. Open the .html files in any browser.")