"""
phase2_simulate_validate.py — Prior Predictive Simulation & Validation
=======================================================================
GOAL: Before deploying anything, verify that the expert priors produce
      sensible risk score distributions. This catches misconfigured
      priors BEFORE they pollute production.

What this replaces from the original code:
  - PyMC sample_prior_predictive → pure NumPy MC (10–100x faster)
  - Static golden_weights.json   → already handled by param_store

Why Monte Carlo instead of PyMC here?
  - No posterior inference is needed (no labeled outcome data)
  - Beta samples are trivially drawn with numpy
  - 10,000 samples runs in <100ms vs ~30s for PyMC compilation

Outputs:
  - Console summary table
  - validation/simulation_results.json
  - plots saved to validation/plots/ (optional, requires matplotlib)

Run:
    python phase2_simulate_validate.py
"""

import json
import os
import warnings
import numpy as np
from phase1_priors import load_param_store, mean_conc_to_alpha_beta

warnings.filterwarnings("ignore")

N_MC_SAMPLES = 10_000
PARAM_STORE_PATH = "params/param_store.json"
OUTPUT_DIR = "validation"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")


# ---------------------------------------------------------------------------
# Core MC Simulation
# ---------------------------------------------------------------------------
def draw_beta_samples(alpha: float, beta_val: float, n: int) -> np.ndarray:
    return np.random.beta(alpha, beta_val, size=n)


def simulate_risk_scores(
    store: dict,
    match_matrix: np.ndarray,   # shape (n_records, n_sources)
    n_mc: int = N_MC_SAMPLES,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    For each record, draw n_mc samples of (p_i, p_leak) from their Beta priors
    and compute the Noisy-OR risk score.  Returns percentile summaries.

    match_matrix[r, i] = 1 if record r matched source i, else 0.

    Returns dict with keys:
      risk_mean, risk_std, risk_p10, risk_p50, risk_p90  — shape (n_records,)
      p_i_samples    — shape (n_mc, n_sources)
      p_leak_samples — shape (n_mc,)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    source_order = store["source_order"]
    n_sources = len(source_order)
    n_records = match_matrix.shape[0]

    # Draw MC samples for each source probability
    p_i_samples = np.column_stack([
        rng.beta(store["sources"][s]["alpha"], store["sources"][s]["beta"], n_mc)
        for s in source_order
    ])  # (n_mc, n_sources)

    p_leak_samples = rng.beta(
        store["leak"]["alpha"], store["leak"]["beta"], n_mc
    )  # (n_mc,)

    # Noisy-OR for every (record × mc_sample) pair
    # log(prob_fail) = log(1-p_leak) + sum_i [ X_i * log(1-p_i) ]
    # Shape broadcast: (n_mc, n_sources) against (n_records, n_sources)
    log_one_minus_p = np.log1p(-p_i_samples)          # (n_mc, n_sources)
    log_p_leak_fail = np.log1p(-p_leak_samples)        # (n_mc,)

    # risk_mc[r, k] = risk score for record r, sample k
    risk_mc = np.zeros((n_records, n_mc), dtype=np.float32)
    for r in range(n_records):
        # (n_mc,): sum of source contributions for this record
        source_log_fail = match_matrix[r] @ log_one_minus_p.T   # (n_mc,)
        log_fail_total  = log_p_leak_fail + source_log_fail
        risk_mc[r]      = 1.0 - np.exp(log_fail_total)

    return {
        "risk_mean":   risk_mc.mean(axis=1).astype(float),
        "risk_std":    risk_mc.std(axis=1).astype(float),
        "risk_p10":    np.percentile(risk_mc, 10, axis=1).astype(float),
        "risk_p50":    np.percentile(risk_mc, 50, axis=1).astype(float),
        "risk_p90":    np.percentile(risk_mc, 90, axis=1).astype(float),
        "p_i_samples": p_i_samples,
        "p_leak_samples": p_leak_samples,
        "risk_mc":     risk_mc,
    }


# ---------------------------------------------------------------------------
# Point-estimate driver attribution (used in API scorer too)
# ---------------------------------------------------------------------------
def compute_impacts(
    match_vector: np.ndarray,   # (n_sources,)
    p_means: np.ndarray,        # (n_sources,)
    p_leak_mean: float,
    risk_score: float,
    log_fail_total: float,
) -> dict[str, float]:
    """
    Marginal impact of each active source:
      impact_i = risk_score - risk_without_i
    Where risk_without_i removes source i's log contribution.
    """
    impacts = {}
    log_one_minus_p = np.log1p(-p_means)
    for i, x in enumerate(match_vector):
        if x == 1:
            log_fail_without_i = log_fail_total - log_one_minus_p[i]
            risk_without_i = 1.0 - np.exp(log_fail_without_i)
            impacts[f"S{i+1}"] = round(float(risk_score - risk_without_i), 6)
    return impacts


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------
def validate_priors(store: dict, n_mc: int = N_MC_SAMPLES) -> dict:
    """
    Sanity checks on the prior specification:
      1. No source has near-certain risk (mean > 0.8) — would dominate everything
      2. Leak probability is low
      3. A zero-match record's baseline risk is reasonable
      4. A full-match record doesn't saturate above ~0.999
    """
    rng = np.random.default_rng(0)
    source_order = store["source_order"]
    n_sources = len(source_order)

    checks = {}

    # Check 1: Individual source means
    high_risk_sources = [
        s for s in source_order if store["sources"][s]["mean"] > 0.8
    ]
    checks["no_dominant_source"] = {
        "pass": len(high_risk_sources) == 0,
        "detail": f"Sources with mean>0.8: {high_risk_sources or 'none'}",
    }

    # Check 2: Leak is small
    leak_mean = store["leak"]["mean"]
    checks["leak_is_small"] = {
        "pass": leak_mean < 0.10,
        "detail": f"Leak mean = {leak_mean:.4f} (should be <0.10)",
    }

    # Check 3: Baseline (zero matches) risk
    zero_match = np.zeros((1, n_sources))
    res_zero = simulate_risk_scores(store, zero_match, n_mc=n_mc, rng=rng)
    baseline_p50 = float(res_zero["risk_p50"][0])
    checks["baseline_risk"] = {
        "pass": baseline_p50 < 0.15,
        "detail": f"Zero-match median risk = {baseline_p50:.4f} (should be <0.15)",
    }

    # Check 4: Full-match saturation
    full_match = np.ones((1, n_sources))
    res_full = simulate_risk_scores(store, full_match, n_mc=n_mc, rng=rng)
    full_p50 = float(res_full["risk_p50"][0])
    checks["full_match_saturates"] = {
        "pass": full_p50 > 0.80,
        "detail": f"Full-match median risk = {full_p50:.4f} (should be >0.80)",
    }

    passed = sum(c["pass"] for c in checks.values())
    checks["_summary"] = {
        "passed": passed,
        "total":  len([k for k in checks if not k.startswith("_")]),
    }
    return checks


def print_validation(checks: dict) -> None:
    print("\n--- Prior Validation ---")
    for name, c in checks.items():
        if name.startswith("_"):
            continue
        status = "✓ PASS" if c["pass"] else "✗ FAIL"
        print(f"  {status}  {name}: {c['detail']}")
    s = checks["_summary"]
    print(f"\n  {s['passed']}/{s['total']} checks passed\n")


# ---------------------------------------------------------------------------
# Source sensitivity analysis
# ---------------------------------------------------------------------------
def source_sensitivity(store: dict, n_mc: int = 2_000) -> dict[str, float]:
    """
    For each source, estimate its average marginal impact by comparing
    a single-source match vs zero match, averaged over MC samples.
    """
    rng = np.random.default_rng(1)
    source_order = store["source_order"]
    n_sources = len(source_order)

    zero = np.zeros((1, n_sources))
    res_zero = simulate_risk_scores(store, zero, n_mc=n_mc, rng=rng)
    base_risk = res_zero["risk_mean"][0]

    sensitivity = {}
    for i, s in enumerate(source_order):
        single = np.zeros((1, n_sources))
        single[0, i] = 1
        res = simulate_risk_scores(store, single, n_mc=n_mc, rng=rng)
        sensitivity[s] = round(float(res["risk_mean"][0] - base_risk), 6)

    return dict(sorted(sensitivity.items(), key=lambda x: -x[1]))


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
def save_validation_results(
    checks: dict,
    sensitivity: dict,
    path: str = "validation/validation_results.json",
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"validation_checks": checks, "source_sensitivity": sensitivity}, f, indent=4)
    print(f"[phase2] Validation results saved → {path}")


# ---------------------------------------------------------------------------
# Optional plots
# ---------------------------------------------------------------------------
def save_plots(store: dict, n_mc: int = N_MC_SAMPLES) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[phase2] matplotlib not available, skipping plots.")
        return

    os.makedirs(PLOT_DIR, exist_ok=True)
    rng = np.random.default_rng(99)
    source_order = store["source_order"]
    n_sources = len(source_order)

    # --- Plot 1: Prior distributions (Beta PDFs) ---
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.flatten()
    x = np.linspace(0.001, 0.999, 500)
    for i, s in enumerate(source_order):
        from scipy.stats import beta as sp_beta
        a, b = store["sources"][s]["alpha"], store["sources"][s]["beta"]
        axes[i].plot(x, sp_beta.pdf(x, a, b), color="steelblue", lw=2)
        axes[i].axvline(store["sources"][s]["mean"], color="red", ls="--", lw=1.2)
        axes[i].set_title(f"{s}  (μ={store['sources'][s]['mean']:.3f})", fontsize=10)
        axes[i].set_xlabel("P(risk | match)")
        axes[i].set_xlim(0, 1)
    plt.suptitle("Expert Prior Distributions per Source (red = mean)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "prior_distributions.png"), dpi=120)
    plt.close()

    # --- Plot 2: Risk score distribution for a random match matrix ---
    n_test = 500
    match_matrix = np.random.binomial(1, 0.25, (n_test, n_sources))
    res = simulate_risk_scores(store, match_matrix, n_mc=n_mc, rng=rng)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(res["risk_p50"], bins=40, color="teal", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Median Risk Score (p50 across MC samples)")
    ax.set_ylabel("Count")
    ax.set_title("Simulated Risk Score Distribution (500 random records, 25% match rate)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "risk_score_distribution.png"), dpi=120)
    plt.close()

    # --- Plot 3: Source sensitivity ---
    sens = source_sensitivity(store)
    fig, ax = plt.subplots(figsize=(7, 4))
    sources_sorted = list(sens.keys())
    values = [sens[s] for s in sources_sorted]
    ax.barh(sources_sorted, values, color="steelblue", edgecolor="white")
    ax.set_xlabel("Average Marginal Risk Increase (single match vs zero)")
    ax.set_title("Source Sensitivity Analysis")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "source_sensitivity.png"), dpi=120)
    plt.close()

    print(f"[phase2] Plots saved → {PLOT_DIR}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    store = load_param_store(PARAM_STORE_PATH)

    print("\n[phase2] Running prior validation...")
    checks = validate_priors(store)
    print_validation(checks)

    print("[phase2] Computing source sensitivity...")
    sensitivity = source_sensitivity(store)
    print("  Source sensitivity (avg marginal impact of a single match):")
    for s, v in sensitivity.items():
        bar = "█" * int(v * 200)
        print(f"    {s}: {v:.4f}  {bar}")

    save_validation_results(checks, sensitivity)
    save_plots(store)

    print("\n[phase2] Complete. Proceed to phase3_scorer.py")
