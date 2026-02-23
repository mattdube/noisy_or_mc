"""
phase3_scorer.py — Core Vectorized Scorer
==========================================
Single source of truth for all scoring logic.
Both the API and batch pipeline import from here — no duplication.

Key design decisions:
  1. Inputs are always 2D numpy arrays → works for n=1 (API) or n=1M (batch)
  2. Uncertainty via MC sampling from Beta priors (fast, accurate)
  3. Driver attribution is fully vectorized (no Python loops over sources)
  4. Param store is loaded once and cached (pass as argument for testability)

Public API:
    score(match_matrix, store, n_mc) → pd.DataFrame
    score_single(match_vector, store, n_mc) → dict
"""

import json
import numpy as np
import pandas as pd
from typing import Any

PARAM_STORE_PATH = "params/param_store.json"
DEFAULT_N_MC = 1_000   # Fast for API; increase for higher-fidelity uncertainty
BATCH_N_MC   = 500     # Slightly lower for large batch runs (speed vs accuracy tradeoff)


# ---------------------------------------------------------------------------
# Param loading (cached at module level for API reuse)
# ---------------------------------------------------------------------------
_cached_store: dict | None = None

def load_store(path: str = PARAM_STORE_PATH) -> dict:
    global _cached_store
    if _cached_store is None:
        with open(path) as f:
            _cached_store = json.load(f)
    return _cached_store

def reload_store(path: str = PARAM_STORE_PATH) -> dict:
    """Force reload — call when params have been updated."""
    global _cached_store
    with open(path) as f:
        _cached_store = json.load(f)
    return _cached_store


# ---------------------------------------------------------------------------
# Pre-compute parameter arrays from store (fast repeated access)
# ---------------------------------------------------------------------------
def _extract_params(store: dict) -> tuple[np.ndarray, np.ndarray, float, float, list[str]]:
    """
    Returns (alphas, betas, leak_alpha, leak_beta, source_order).
    All as numpy arrays for vectorized sampling.
    """
    source_order = store["source_order"]
    alphas = np.array([store["sources"][s]["alpha"] for s in source_order])
    betas  = np.array([store["sources"][s]["beta"]  for s in source_order])
    return (
        alphas,
        betas,
        store["leak"]["alpha"],
        store["leak"]["beta"],
        source_order,
    )


# ---------------------------------------------------------------------------
# Core scoring engine
# ---------------------------------------------------------------------------
def _score_core(
    match_matrix: np.ndarray,    # (n_records, n_sources), values in {0, 1}
    alphas:       np.ndarray,    # (n_sources,)
    betas:        np.ndarray,    # (n_sources,)
    leak_alpha:   float,
    leak_beta:    float,
    n_mc:         int,
    rng:          np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    Pure numpy scoring with MC uncertainty.

    Returns:
      risk_point  (n_records,)  — point estimate using posterior means
      risk_p50    (n_records,)  — median over MC samples
      risk_p10    (n_records,)  — 10th percentile (lower credible bound)
      risk_p90    (n_records,)  — 90th percentile (upper credible bound)
      risk_std    (n_records,)  — std over MC samples
      impact_matrix (n_records, n_sources) — marginal impact per source
    """
    n_records, n_sources = match_matrix.shape

    # --- Point estimate (posterior means — instantaneous, no MC needed) ---
    p_means     = alphas / (alphas + betas)
    p_leak_mean = leak_alpha / (leak_alpha + leak_beta)

    log_one_minus_p_means  = np.log1p(-p_means)        # (n_sources,)
    log_p_leak_fail_mean   = np.log1p(-p_leak_mean)     # scalar

    # (n_records,)
    source_log_fail_point = match_matrix @ log_one_minus_p_means
    log_fail_point        = log_p_leak_fail_mean + source_log_fail_point
    risk_point            = 1.0 - np.exp(log_fail_point)

    # --- Vectorized driver attribution (point estimate) ---
    # impact[r, i] = risk_point[r] - risk_without_source_i[r]
    # risk_without_i = 1 - exp(log_fail_total - X[r,i]*log(1-p_i))
    # Broadcast: (n_records, n_sources)
    log_one_minus_p_broadcast = log_one_minus_p_means[np.newaxis, :]   # (1, n_sources)
    log_fail_without_i = (
        log_fail_point[:, np.newaxis]                                    # (n_records, 1)
        - match_matrix * log_one_minus_p_broadcast                       # (n_records, n_sources)
    )
    risk_without_i  = 1.0 - np.exp(log_fail_without_i)                  # (n_records, n_sources)
    impact_matrix   = risk_point[:, np.newaxis] - risk_without_i         # (n_records, n_sources)
    impact_matrix  *= match_matrix                                        # zero out non-matches

    # --- MC uncertainty (draws from Beta priors) ---
    p_i_mc    = rng.beta(alphas, betas, size=(n_mc, n_sources))   # (n_mc, n_sources)
    p_leak_mc = rng.beta(leak_alpha, leak_beta, size=n_mc)         # (n_mc,)

    log_1mp_mc       = np.log1p(-p_i_mc)     # (n_mc, n_sources)
    log_leak_fail_mc = np.log1p(-p_leak_mc)  # (n_mc,)

    # risk_mc[r, k] — broadcast (n_records, n_sources) × (n_mc, n_sources).T
    # = (n_records, n_mc)
    source_log_fail_mc = match_matrix @ log_1mp_mc.T    # (n_records, n_mc)
    log_fail_mc        = log_leak_fail_mc[np.newaxis, :] + source_log_fail_mc
    risk_mc            = (1.0 - np.exp(log_fail_mc)).astype(np.float32)

    return {
        "risk_point":    risk_point,
        "risk_p10":      np.percentile(risk_mc, 10, axis=1),
        "risk_p50":      np.percentile(risk_mc, 50, axis=1),
        "risk_p90":      np.percentile(risk_mc, 90, axis=1),
        "risk_std":      risk_mc.std(axis=1),
        "impact_matrix": impact_matrix,
    }


# ---------------------------------------------------------------------------
# Public scoring functions
# ---------------------------------------------------------------------------
def score(
    match_matrix: np.ndarray,
    store:   dict | None = None,
    n_mc:    int  = DEFAULT_N_MC,
    seed:    int  = 0,
    source_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Score a batch of records.

    Parameters
    ----------
    match_matrix : np.ndarray, shape (n_records, n_sources), dtype int/float
        Binary match indicators. Columns must align with store["source_order"].
    store : dict, optional
        Parameter store. Loaded from disk if None.
    n_mc : int
        Number of Monte Carlo samples for uncertainty estimation.
    seed : int
        RNG seed for reproducibility.
    source_names : list[str], optional
        Override column names for the impact columns.

    Returns
    -------
    pd.DataFrame with columns:
        risk_point, risk_p50, risk_p10, risk_p90, risk_std,
        primary_driver, impact_width,
        S1_impact, S2_impact, ..., S8_impact
    """
    if store is None:
        store = load_store()

    rng = np.random.default_rng(seed)
    alphas, betas, leak_alpha, leak_beta, source_order = _extract_params(store)

    if source_names is None:
        source_names = source_order

    match_matrix = np.asarray(match_matrix, dtype=np.float64)
    assert match_matrix.shape[1] == len(source_order), (
        f"match_matrix has {match_matrix.shape[1]} columns but store has {len(source_order)} sources"
    )

    result = _score_core(match_matrix, alphas, betas, leak_alpha, leak_beta, n_mc, rng)

    n_records = match_matrix.shape[0]
    df = pd.DataFrame({
        "risk_point": np.round(result["risk_point"], 4),
        "risk_p50":   np.round(result["risk_p50"],   4),
        "risk_p10":   np.round(result["risk_p10"],   4),
        "risk_p90":   np.round(result["risk_p90"],   4),
        "risk_std":   np.round(result["risk_std"],   4),
        "impact_width": np.round(result["risk_p90"] - result["risk_p10"], 4),
    })

    # Driver attribution columns
    impact_matrix = result["impact_matrix"]
    for i, s in enumerate(source_names):
        df[f"{s}_impact"] = np.round(impact_matrix[:, i], 6)

    # Primary driver (source with highest marginal impact; "Leak" if all zero)
    max_idx     = np.argmax(impact_matrix, axis=1)
    max_val     = impact_matrix[np.arange(n_records), max_idx]
    primary     = np.where(max_val > 0, [source_names[i] for i in max_idx], "Leak")
    df["primary_driver"] = primary

    # Matched sources (human-readable string)
    df["matched_sources"] = [
        ", ".join(s for j, s in enumerate(source_names) if match_matrix[r, j] == 1) or "none"
        for r in range(n_records)
    ]

    return df


def score_single(
    match_vector: np.ndarray,    # (n_sources,) or list
    store:   dict | None = None,
    n_mc:    int  = DEFAULT_N_MC,
    seed:    int  = 0,
) -> dict[str, Any]:
    """
    Score a single record. Wrapper around score() for API use.

    Returns a plain dict (JSON-serializable).
    """
    if store is None:
        store = load_store()

    mv = np.asarray(match_vector, dtype=np.float64).reshape(1, -1)
    df = score(mv, store=store, n_mc=n_mc, seed=seed)
    row = df.iloc[0].to_dict()

    # Build a clean impact sub-dict
    source_order = store["source_order"]
    impacts = {s: row.pop(f"{s}_impact", 0.0) for s in source_order}
    row["impacts"] = impacts

    return row


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from phase1_priors import load_param_store, SOURCE_SPECS, LEAK_SPEC, build_param_store, save_param_store
    import os

    # Build store if it doesn't exist yet
    if not os.path.exists(PARAM_STORE_PATH):
        store = build_param_store(SOURCE_SPECS, LEAK_SPEC)
        save_param_store(store)

    store = load_param_store(PARAM_STORE_PATH)

    print("\n=== Single Record Test ===")
    test_vec = np.array([1, 0, 0, 1, 0, 0, 1, 0])  # S1, S4, S7 matched
    result = score_single(test_vec, store=store, n_mc=2_000)
    print(f"  Matches: S1, S4, S7")
    print(f"  risk_point : {result['risk_point']}")
    print(f"  risk_p50   : {result['risk_p50']}")
    print(f"  risk_p10   : {result['risk_p10']}  →  risk_p90: {result['risk_p90']}")
    print(f"  impact_width: {result['impact_width']} (credible interval width)")
    print(f"  primary_driver: {result['primary_driver']}")
    print(f"  impacts: {result['impacts']}")

    print("\n=== Batch Test (100 records) ===")
    np.random.seed(42)
    batch_matrix = np.random.binomial(1, 0.25, (100, 8))
    df = score(batch_matrix, store=store, n_mc=500)
    print(df[["risk_point", "risk_p50", "risk_p10", "risk_p90", "primary_driver"]].describe())
    print(f"\n  Primary driver distribution:\n{df['primary_driver'].value_counts()}")
    print("\n[phase3] Scorer tests passed.")
