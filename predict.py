"""
predict.py — Single-Record Prediction (Standalone)
====================================================
Drop this file next to your param_store.json and run it.
No other pipeline files needed.

Requirements:
    pip install numpy

Usage — as a script:
    python predict.py                          # runs built-in examples
    python predict.py 1 0 0 1 0 0 1 0         # pass matches as CLI args

Usage — import into a notebook or script:
    from predict import load_model, predict, predict_df

    model = load_model("params/param_store.json")

    # Single record as a dict  (most readable)
    result = predict(model, {"S1":1, "S4":1, "S7":1})

    # Or as a plain list in source order
    result = predict(model, [1, 0, 0, 1, 0, 0, 1, 0])

    print(result["risk_score"])      # e.g. 0.6567
    print(result["tier"])            # e.g. "Moderate"
    print(result["primary_driver"])  # e.g. "S4"
    print(result["explanation"])     # human-readable sentence
"""

import json
import sys
import math
import numpy as np
from pathlib import Path
from typing import Union

# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def load_model(param_store_path: str = "params/param_store.json") -> dict:
    """
    Load the saved param store from disk.
    Returns a lightweight model dict ready for prediction.

    The model dict contains:
        source_order  — list of source names in column order
        p_means       — numpy array of posterior means (one per source)
        p_stds        — numpy array of posterior stds
        alphas        — numpy array of Beta alpha params
        betas         — numpy array of Beta beta params
        leak_mean     — float, baseline leak probability
        leak_alpha    — float
        leak_beta     — float
        version       — string, param store version
    """
    path = Path(param_store_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Param store not found at: {path.resolve()}\n"
            "Run phase1_priors.py first to generate it."
        )

    with open(path) as f:
        store = json.load(f)

    source_order = store["source_order"]

    model = {
        "source_order": source_order,
        "p_means":  np.array([store["sources"][s]["mean"]  for s in source_order]),
        "p_stds":   np.array([store["sources"][s]["std"]   for s in source_order]),
        "alphas":   np.array([store["sources"][s]["alpha"] for s in source_order]),
        "betas":    np.array([store["sources"][s]["beta"]  for s in source_order]),
        "leak_mean":  store["leak"]["mean"],
        "leak_alpha": store["leak"]["alpha"],
        "leak_beta":  store["leak"]["beta"],
        "version":    store["version"],
    }
    return model


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------
def predict(
    model:        dict,
    matches:      Union[dict, list],
    n_mc:         int  = 2000,
    seed:         int  = 0,
) -> dict:
    """
    Score a single record and return a plain dict.

    Parameters
    ----------
    model : dict
        Output of load_model().
    matches : dict or list
        Which sources matched this record.

        As a dict  — keys are source names, values are 0 or 1.
            {"S1": 1, "S4": 1, "S7": 1}
            Unspecified sources default to 0 (no match).

        As a list  — values in the same order as model["source_order"].
            [1, 0, 0, 1, 0, 0, 1, 0]

    n_mc : int
        Monte Carlo samples for uncertainty. 2000 gives stable CIs.
        Use 500 for speed, 5000 for high-precision uncertainty.
    seed : int
        RNG seed. Same seed → identical output (useful for testing).

    Returns
    -------
    dict with keys:
        risk_score      float  — point estimate (uses posterior means)
        risk_p10        float  — 10th percentile over MC samples
        risk_p50        float  — median over MC samples
        risk_p90        float  — 90th percentile over MC samples
        ci_width        float  — risk_p90 - risk_p10 (uncertainty band)
        tier            str    — "Minimal" / "Low" / "Moderate" / "High"
        confidence      str    — "High" / "Moderate" / "Low" / "Very Low"
        primary_driver  str    — source with largest marginal impact
        matched         list   — source names that matched
        impacts         dict   — marginal impact per matched source
        explanation     str    — one-sentence human-readable summary
    """
    source_order = model["source_order"]
    n_sources    = len(source_order)

    # --- Normalise matches to a numpy array ---
    if isinstance(matches, dict):
        vec = np.array(
            [float(matches.get(s, 0)) for s in source_order],
            dtype=np.float64,
        )
    else:
        vec = np.asarray(matches, dtype=np.float64)
        if vec.shape[0] != n_sources:
            raise ValueError(
                f"matches has {vec.shape[0]} values but model has "
                f"{n_sources} sources ({source_order})"
            )

    if not np.all(np.isin(vec, [0.0, 1.0])):
        raise ValueError("matches values must all be 0 or 1")

    # --- Point estimate (deterministic, uses posterior means) ---
    log_one_minus_p = np.log1p(-model["p_means"])
    log_leak_fail   = math.log1p(-model["leak_mean"])
    log_fail_point  = log_leak_fail + float(vec @ log_one_minus_p)
    risk_point      = 1.0 - math.exp(log_fail_point)

    # --- Driver attribution ---
    impacts = {}
    for i, s in enumerate(source_order):
        if vec[i] == 1:
            log_fail_without = log_fail_point - log_one_minus_p[i]
            risk_without     = 1.0 - math.exp(log_fail_without)
            impacts[s]       = round(risk_point - risk_without, 6)

    primary_driver = max(impacts, key=impacts.get) if impacts else "Leak"

    # --- MC uncertainty (draw from Beta priors) ---
    rng          = np.random.default_rng(seed)
    p_i_mc       = rng.beta(model["alphas"], model["betas"], size=(n_mc, n_sources))
    p_leak_mc    = rng.beta(model["leak_alpha"], model["leak_beta"], size=n_mc)

    log_1mp_mc       = np.log1p(-p_i_mc)       # (n_mc, n_sources)
    source_log_fail  = log_1mp_mc @ vec         # (n_mc,)
    log_fail_mc      = np.log1p(-p_leak_mc) + source_log_fail
    risk_mc          = 1.0 - np.exp(log_fail_mc)

    risk_p10 = float(np.percentile(risk_mc, 10))
    risk_p50 = float(np.percentile(risk_mc, 50))
    risk_p90 = float(np.percentile(risk_mc, 90))
    ci_width = round(risk_p90 - risk_p10, 4)

    # --- Tier labels ---
    tier = (
        "High"     if risk_point >= 0.70 else
        "Moderate" if risk_point >= 0.40 else
        "Low"      if risk_point >= 0.15 else
        "Minimal"
    )
    confidence = (
        "High"      if ci_width < 0.05  else
        "Moderate"  if ci_width < 0.15  else
        "Low"       if ci_width < 0.40  else
        "Very Low"
    )

    matched = [s for s in source_order if vec[source_order.index(s)] == 1]

    # --- One-sentence explanation ---
    if not matched:
        explanation = (
            f"No sources matched; risk reflects baseline only "
            f"(score={risk_point:.3f}, {tier})."
        )
    else:
        top_impact = impacts[primary_driver]
        explanation = (
            f"{len(matched)} source(s) matched ({', '.join(matched)}); "
            f"risk score is {risk_point:.3f} ({tier}) with {confidence.lower()} confidence "
            f"[80% CI: {risk_p10:.3f}–{risk_p90:.3f}]. "
            f"Primary driver: {primary_driver} "
            f"(contributes {top_impact:.3f} of the total score)."
        )

    return {
        "risk_score":     round(risk_point, 4),
        "risk_p10":       round(risk_p10, 4),
        "risk_p50":       round(risk_p50, 4),
        "risk_p90":       round(risk_p90, 4),
        "ci_width":       ci_width,
        "tier":           tier,
        "confidence":     confidence,
        "primary_driver": primary_driver,
        "matched":        matched,
        "impacts":        impacts,
        "explanation":    explanation,
    }


# ---------------------------------------------------------------------------
# Convenience: score a list of records → list of dicts (no pandas needed)
# ---------------------------------------------------------------------------
def predict_many(
    model:   dict,
    records: list[Union[dict, list]],
    n_mc:    int = 1000,
    seed:    int = 0,
) -> list[dict]:
    """
    Score multiple records. Returns a list of result dicts.
    Suitable for use in a notebook where you want to loop or inspect results.

    For large files (1000+ records) use phase5_batch.py instead —
    it uses fully vectorised NumPy and is much faster at scale.
    """
    return [predict(model, r, n_mc=n_mc, seed=seed) for r in records]


# ---------------------------------------------------------------------------
# Convenience: score records → pandas DataFrame (if pandas is available)
# ---------------------------------------------------------------------------
def predict_df(
    model:   dict,
    records: list[Union[dict, list]],
    n_mc:    int = 1000,
    seed:    int = 0,
):
    """
    Score multiple records and return a pandas DataFrame.
    Requires pandas (pip install pandas).
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for predict_df. pip install pandas")

    results = predict_many(model, records, n_mc=n_mc, seed=seed)
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------
def print_result(result: dict, label: str = "") -> None:
    """Print a single prediction result in a readable format."""
    header = f"  [{label}]" if label else "  Result"
    print(header)
    print(f"    risk_score    : {result['risk_score']:.4f}  ({result['tier']})")
    print(f"    80% CI        : [{result['risk_p10']:.4f}, {result['risk_p90']:.4f}]"
          f"  width={result['ci_width']:.4f}  ({result['confidence']} confidence)")
    print(f"    primary driver: {result['primary_driver']}")
    print(f"    matched       : {result['matched'] or 'none'}")
    if result["impacts"]:
        print(f"    contributions :")
        for src, imp in sorted(result["impacts"].items(), key=lambda x: -x[1]):
            bar = "█" * max(1, int(imp * 80))
            print(f"      {src}: {imp:.4f}  {bar}")
    print(f"    explanation   : {result['explanation']}")
    print()


# ---------------------------------------------------------------------------
# Main — run examples or score from CLI args
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # --- Load model ---
    MODEL_PATH = "params/param_store.json"
    print(f"\nLoading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print(f"Model version : {model['version']}")
    print(f"Sources       : {model['source_order']}")
    print(f"Leak baseline : {model['leak_mean']:.4f}\n")

    # --- CLI mode: python predict.py 1 0 0 1 0 0 1 0 ---
    if len(sys.argv) > 1:
        try:
            cli_matches = [int(x) for x in sys.argv[1:]]
        except ValueError:
            print("Usage: python predict.py 1 0 0 1 0 0 1 0")
            print("       (one 0/1 value per source, in source order)")
            sys.exit(1)

        result = predict(model, cli_matches, n_mc=2000)
        print_result(result, label="CLI input: " + " ".join(sys.argv[1:]))
        sys.exit(0)

    # --- Built-in examples ---
    print("=" * 55)
    print("  Built-in Prediction Examples")
    print("=" * 55)

    examples = [
        # (matches, label)
        ({"S1": 1, "S4": 1, "S7": 1},            "Dict input: S1+S4+S7"),
        ([1, 0, 0, 1, 0, 0, 1, 0],               "List input: S1+S4+S7"),
        ({},                                       "No matches (baseline)"),
        ({"S7": 1},                                "S7 only (high-confidence source)"),
        ({"S4": 1},                                "S4 only (low-confidence source)"),
        ({"S1":1,"S2":1,"S3":1,"S4":1,
          "S5":1,"S6":1,"S7":1,"S8":1},           "All sources matched"),
    ]

    for matches, label in examples:
        result = predict(model, matches, n_mc=2000, seed=0)
        print_result(result, label=label)

    # --- Notebook-style usage demo ---
    print("=" * 55)
    print("  Notebook Usage Demo")
    print("=" * 55)
    print("""
  # In a Jupyter notebook:

  from predict import load_model, predict, predict_df

  model = load_model("params/param_store.json")

  # Score one record as a dict (clearest intent)
  r = predict(model, {"S1": 1, "S4": 1, "S7": 1})
  print(r["risk_score"])      # 0.6567
  print(r["tier"])            # Moderate
  print(r["explanation"])     # 3 source(s) matched ...

  # Access specific fields
  print(r["risk_p10"], r["risk_p90"])   # 80% CI bounds
  print(r["impacts"])                   # per-source contributions

  # Score several records at once → DataFrame
  records = [
      {"S1": 1, "S4": 1},
      {"S7": 1, "S6": 1},
      {},
  ]
  df = predict_df(model, records)
  print(df[["risk_score", "tier", "primary_driver"]])
""")

    # --- Show predict_df output if pandas is available ---
    try:
        import pandas as pd
        sample_records = [
            {"S1": 1, "S4": 1},
            {"S7": 1, "S6": 1},
            {"S3": 1, "S5": 1},
            {},
        ]
        df = predict_df(model, sample_records, n_mc=1000, seed=0)
        print("  predict_df output:")
        print(df[["risk_score", "risk_p10", "risk_p90",
                   "tier", "confidence", "primary_driver"]].to_string(index=False))
        print()
    except ImportError:
        print("  (pandas not installed — predict_df not shown)")
