"""
phase1_priors.py — Expert Prior Specification & Parameter Store
===============================================================
This is the ENTRY POINT of the pipeline. It encodes expert knowledge
about each risk source as Beta distribution parameters and stores them
in a versioned JSON parameter store.

No labeled data is required at this stage.

Beta(alpha, beta) intuition:
  - alpha / (alpha + beta) = mean (your estimated base rate)
  - alpha + beta = concentration (your confidence / effective sample size)

Run:
    python phase1_priors.py
Output:
    params/param_store.json
"""

import json
import os
import numpy as np
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# 1. Expert knowledge: (mean_risk, concentration)
#    concentration ~ "how many virtual observations" back up your estimate
#    Low  (10–30):  wide uncertainty, e.g. novel source with sparse data
#    Med  (50–200): moderate certainty, e.g. source validated on past cases
#    High (300+):   tight prior, e.g. well-studied source with rich history
# ---------------------------------------------------------------------------
SOURCE_SPECS: dict[str, tuple[float, float]] = {
    "S1": (0.02,  500),   # Very low risk, high confidence
    "S2": (0.05,   50),   # Low risk,  low confidence (novel source)
    "S3": (0.12,  100),   # Moderate risk
    "S4": (0.45,   15),   # High risk, very uncertain (sparse evidence)
    "S5": (0.08,  200),   # Low-moderate, solid evidence
    "S6": (0.20,   40),   # Moderate, limited data
    "S7": (0.35,  300),   # High risk, strong evidence
    "S8": (0.01, 1000),   # Tiny risk, very well validated
}

LEAK_SPEC = (0.02, 100)   # ~2% baseline unexplained risk, confident


def mean_conc_to_alpha_beta(mean: float, concentration: float) -> tuple[float, float]:
    """
    Convert (mean, concentration) → (alpha, beta).
    Floor at 1.1 prevents degenerate Beta shapes (U-shaped or half-bathtub).
    """
    alpha = max(mean * concentration, 1.1)
    beta  = max((1.0 - mean) * concentration, 1.1)
    return float(alpha), float(beta)


def alpha_beta_to_mean_conc(alpha: float, beta: float) -> tuple[float, float]:
    conc = alpha + beta
    mean = alpha / conc
    return float(mean), float(conc)


def beta_stats(alpha: float, beta: float) -> dict:
    """Return key statistics of a Beta(alpha, beta) distribution."""
    mean = alpha / (alpha + beta)
    variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    std = variance ** 0.5
    # 90% credible interval via normal approximation (fast, good enough for diagnostics)
    z = 1.645
    lo = max(0.0, mean - z * std)
    hi = min(1.0, mean + z * std)
    return {
        "mean":  round(mean, 6),
        "std":   round(std, 6),
        "ci90_lo": round(lo, 6),
        "ci90_hi": round(hi, 6),
    }


def build_param_store(
    source_specs: dict[str, tuple[float, float]],
    leak_spec: tuple[float, float],
    version: str | None = None,
) -> dict:
    """
    Build a versioned parameter store dict from human-readable specs.
    """
    if version is None:
        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    sources = {}
    for name, (mean, conc) in source_specs.items():
        alpha, beta = mean_conc_to_alpha_beta(mean, conc)
        sources[name] = {
            "alpha": alpha,
            "beta":  beta,
            **beta_stats(alpha, beta),
            "expert_mean": mean,
            "expert_concentration": conc,
        }

    leak_alpha, leak_beta = mean_conc_to_alpha_beta(*leak_spec)
    leak = {
        "alpha": leak_alpha,
        "beta":  leak_beta,
        **beta_stats(leak_alpha, leak_beta),
        "expert_mean": leak_spec[0],
        "expert_concentration": leak_spec[1],
    }

    return {
        "version": version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_order": list(source_specs.keys()),
        "sources": sources,
        "leak": leak,
    }


def save_param_store(store: dict, path: str = "params/param_store.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(store, f, indent=4)
    print(f"[phase1] Parameter store saved → {path}  (version: {store['version']})")


def load_param_store(path: str = "params/param_store.json") -> dict:
    with open(path) as f:
        return json.load(f)


def bayesian_update_source(
    store: dict,
    source_name: str,
    n_successes: int,
    n_trials: int,
) -> dict:
    """
    Conjugate Beta update for a single source given new Bernoulli observations.
    Use this when you accumulate partial labels over time (e.g., analyst reviews).

    updated_alpha = prior_alpha + n_successes
    updated_beta  = prior_beta  + (n_trials - n_successes)

    This is mathematically exact — no MCMC needed.
    """
    src = store["sources"][source_name]
    new_alpha = src["alpha"] + n_successes
    new_beta  = src["beta"]  + (n_trials - n_successes)

    updated = {
        "alpha": new_alpha,
        "beta":  new_beta,
        **beta_stats(new_alpha, new_beta),
        "expert_mean": src["expert_mean"],
        "expert_concentration": src["expert_concentration"],
        "update_note": f"Conjugate update: +{n_successes} successes / {n_trials} trials",
    }
    store["sources"][source_name] = updated
    store["version"] = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_updated"
    return store


# ---------------------------------------------------------------------------
# Diagnostic printout
# ---------------------------------------------------------------------------
def print_prior_summary(store: dict) -> None:
    print("\n" + "=" * 65)
    print(f"  Parameter Store — version: {store['version']}")
    print("=" * 65)
    print(f"  {'Source':<8} {'Mean':>8} {'Std':>8} {'CI90_lo':>9} {'CI90_hi':>9}")
    print("-" * 65)
    for name, s in store["sources"].items():
        print(f"  {name:<8} {s['mean']:>8.4f} {s['std']:>8.4f} "
              f"{s['ci90_lo']:>9.4f} {s['ci90_hi']:>9.4f}")
    lk = store["leak"]
    print(f"  {'Leak':<8} {lk['mean']:>8.4f} {lk['std']:>8.4f} "
          f"{lk['ci90_lo']:>9.4f} {lk['ci90_hi']:>9.4f}")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    store = build_param_store(SOURCE_SPECS, LEAK_SPEC)
    print_prior_summary(store)
    save_param_store(store)

    # --- Example: update S2 after 10 analyst reviews (3 confirmed risk events)
    store_updated = bayesian_update_source(store, "S2", n_successes=3, n_trials=10)
    print("S2 after conjugate update:")
    s2 = store_updated["sources"]["S2"]
    print(f"  mean={s2['mean']:.4f}  std={s2['std']:.4f}  "
          f"(was expert_mean={s2['expert_mean']:.4f})\n")
