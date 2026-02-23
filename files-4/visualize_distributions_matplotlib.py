"""
visualize_distributions_matplotlib.py
======================================
Visualizes the Beta prior distributions for all risk sources
using matplotlib and seaborn.

Requires:
    pip install matplotlib seaborn scipy numpy

Run:
    python visualize_distributions_matplotlib.py

Outputs:
    source_distributions.png  — saved to current directory
    (also displays interactively if a display is available)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from scipy.stats import beta as beta_dist
import seaborn as sns

# ── Load param store ──────────────────────────────────────────────────────────
PARAM_PATH = "params/param_store.json"

with open(PARAM_PATH) as f:
    store = json.load(f)

SOURCE_ORDER = store["source_order"]
SOURCES      = store["sources"]
LEAK         = store["leak"]

# ── Palette (consistent with the React visualizer) ────────────────────────────
COLORS = {
    "S1": "#22d3ee", "S2": "#a78bfa", "S3": "#34d399", "S4": "#f87171",
    "S5": "#60a5fa", "S6": "#fb923c", "S7": "#e879f9", "S8": "#94a3b8",
}

# ── Annotation labels ─────────────────────────────────────────────────────────
LABELS = {
    "S1": "Very Low Risk\nHigh Confidence",
    "S2": "Low Risk\nLow Confidence",
    "S3": "Moderate Risk\nMedium Confidence",
    "S4": "High Risk\nVery Uncertain",
    "S5": "Low Risk\nSolid Evidence",
    "S6": "Moderate Risk\nLimited Data",
    "S7": "High Risk\nStrong Evidence",
    "S8": "Minimal Risk\nVery Well Validated",
}

# ── Style setup ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0b1120",
    "axes.facecolor":    "#0f172a",
    "axes.edgecolor":    "#1e293b",
    "axes.labelcolor":   "#94a3b8",
    "xtick.color":       "#475569",
    "ytick.color":       "#475569",
    "text.color":        "#e2e8f0",
    "grid.color":        "#1e293b",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "font.family":       "monospace",
    "font.size":         9,
})

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Small-multiple grid (one panel per source)
# ─────────────────────────────────────────────────────────────────────────────
def plot_small_multiples(save_path="source_distributions_grid.png"):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.patch.set_facecolor("#0b1120")
    fig.suptitle(
        "Beta Prior Distributions — Risk Sources",
        fontsize=14, fontweight="bold", color="#f1f5f9",
        y=0.98, fontfamily="monospace",
    )

    x = np.linspace(0.001, 0.999, 500)

    for ax, src in zip(axes.flatten(), SOURCE_ORDER):
        s     = SOURCES[src]
        a, b  = s["alpha"], s["beta"]
        mean  = s["mean"]
        std   = s["std"]
        color = COLORS[src]
        conc  = s["expert_concentration"]

        y = beta_dist.pdf(x, a, b)

        # Adaptive x-range: show ±4 std around mean
        lo = max(0.001, mean - 4 * std)
        hi = min(0.999, mean + 4 * std)
        mask = (x >= lo) & (x <= hi)

        ax.set_facecolor("#0f172a")
        ax.fill_between(x[mask], y[mask], alpha=0.25, color=color)
        ax.plot(x[mask], y[mask], color=color, linewidth=2)

        # Mean line
        mean_y = beta_dist.pdf(mean, a, b)
        ax.axvline(mean, color=color, linewidth=1.2, linestyle="--", alpha=0.7)
        ax.text(mean, mean_y * 1.04, f" μ={mean:.2f}",
                color=color, fontsize=8, va="bottom", ha="left")

        # ±1 std shading
        lo_std = max(0.001, mean - std)
        hi_std = min(0.999, mean + std)
        mask_std = (x >= lo_std) & (x <= hi_std)
        ax.fill_between(x[mask_std], y[mask_std], alpha=0.35, color=color)

        # Axes
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, None)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1%}"))
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.4)

        # Title block
        ax.set_title(
            f"{src}   α={a:.1f}  β={b:.1f}  conc={conc}",
            color=color, fontsize=9, fontweight="bold", pad=6,
        )
        ax.set_xlabel(LABELS[src], color="#64748b", fontsize=8, labelpad=4)

        # std annotation
        ax.annotate(
            f"σ={std:.3f}", xy=(0.98, 0.92), xycoords="axes fraction",
            ha="right", va="top", fontsize=8, color="#64748b",
        )

        for spine in ax.spines.values():
            spine.set_edgecolor("#1e293b")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0b1120")
    print(f"[matplotlib] Saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Overlay comparison (all sources on one axes)
# ─────────────────────────────────────────────────────────────────────────────
def plot_overlay(save_path="source_distributions_overlay.png"):
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0f172a")

    x = np.linspace(0.001, 0.999, 800)

    for src in SOURCE_ORDER:
        s     = SOURCES[src]
        a, b  = s["alpha"], s["beta"]
        mean  = s["mean"]
        std   = s["std"]
        color = COLORS[src]

        # Zoom window per source
        lo = max(0.001, mean - 4 * std)
        hi = min(0.999, mean + 4 * std)
        mask = (x >= lo) & (x <= hi)

        y = beta_dist.pdf(x, a, b)

        ax.fill_between(x[mask], y[mask], alpha=0.12, color=color)
        ax.plot(x[mask], y[mask], color=color, linewidth=2, label=src)
        ax.axvline(mean, color=color, linewidth=0.8, linestyle=":", alpha=0.6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_yticks([])
    ax.set_xlabel("P(risk | match)", color="#94a3b8", fontsize=10)
    ax.set_title(
        "All Source Prior Distributions — Overlay",
        color="#f1f5f9", fontsize=13, fontweight="bold", pad=12, fontfamily="monospace",
    )
    ax.grid(True, axis="x", alpha=0.3)

    for spine in ax.spines.values():
        spine.set_edgecolor("#1e293b")

    # Custom legend with mean values
    handles = [
        mpatches.Patch(color=COLORS[s], label=f"{s}  μ={SOURCES[s]['mean']:.2f}  (c={SOURCES[s]['expert_concentration']})")
        for s in SOURCE_ORDER
    ]
    ax.legend(
        handles=handles, loc="upper right", fontsize=8,
        facecolor="#0b1120", edgecolor="#1e293b", labelcolor="#cbd5e1",
        framealpha=0.9, handlelength=1.4,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0b1120")
    print(f"[matplotlib] Saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Summary comparison: mean ± CI strip chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_summary_strip(save_path="source_distributions_summary.png"):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0f172a")

    # Compute 90% credible intervals from the Beta
    ci_lo = np.array([beta_dist.ppf(0.05, SOURCES[s]["alpha"], SOURCES[s]["beta"]) for s in SOURCE_ORDER])
    ci_hi = np.array([beta_dist.ppf(0.95, SOURCES[s]["alpha"], SOURCES[s]["beta"]) for s in SOURCE_ORDER])
    means = np.array([SOURCES[s]["mean"] for s in SOURCE_ORDER])
    concs = np.array([SOURCES[s]["expert_concentration"] for s in SOURCE_ORDER])

    y_pos = np.arange(len(SOURCE_ORDER))

    for i, src in enumerate(SOURCE_ORDER):
        color = COLORS[src]
        lo, hi, m = ci_lo[i], ci_hi[i], means[i]

        # CI bar
        ax.barh(i, hi - lo, left=lo, height=0.45,
                color=color, alpha=0.25, zorder=2)
        # CI whiskers
        ax.plot([lo, hi], [i, i], color=color, linewidth=1.5, alpha=0.6, zorder=3)
        ax.plot([lo, lo], [i - 0.2, i + 0.2], color=color, linewidth=1.5, alpha=0.6, zorder=3)
        ax.plot([hi, hi], [i - 0.2, i + 0.2], color=color, linewidth=1.5, alpha=0.6, zorder=3)
        # Mean dot
        ax.scatter(m, i, color=color, s=80, zorder=5, edgecolors="white", linewidths=0.8)

        # Concentration indicator (dot size legend note)
        ax.text(1.01, i, f"c={concs[i]:>4.0f}", va="center",
                fontsize=8, color="#64748b", transform=ax.get_yaxis_transform())

    ax.set_yticks(y_pos)
    ax.set_yticklabels(SOURCE_ORDER, fontsize=10, color="#e2e8f0")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("P(risk | match)  ·  dot = mean  ·  bar = 90% credible interval",
                  color="#94a3b8", fontsize=9)
    ax.set_title(
        "Source Risk Estimates — Mean & 90% Credible Interval",
        color="#f1f5f9", fontsize=13, fontweight="bold", pad=12, fontfamily="monospace",
    )
    ax.set_xlim(-0.02, 1.0)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    for spine in ax.spines.values():
        spine.set_edgecolor("#1e293b")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0b1120")
    print(f"[matplotlib] Saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Run all three
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating matplotlib/seaborn visualizations...")
    plot_small_multiples()
    plot_overlay()
    plot_summary_strip()
    print("\nDone. Three plots saved to current directory.")
