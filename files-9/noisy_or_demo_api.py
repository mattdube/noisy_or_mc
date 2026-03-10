"""
noisy_or_demo_api.py
====================
Demonstrates and benchmarks the API-optimized NoisyORModelAPI.

Covers:
  1. Loading the model and inspecting the pre-built pool
  2. Single predictions with precision tiers
  3. Cache behavior on repeated inputs
  4. Vectorized batch prediction
  5. Hot reload after a config update
  6. Side-by-side performance benchmark vs. the base model
  7. Export to Excel / CSV (inherited from base model)

Run:  python noisy_or_demo_api.py
"""

import time
import numpy as np
import pandas as pd

from noisy_or_model     import NoisyORModel
from noisy_or_model_api import NoisyORModelAPI, PRECISION_SAMPLES

# -----------------------------------------------------------------------
# 1.  Load model
# -----------------------------------------------------------------------
print("=" * 60)
print("  NOISY-OR API MODEL DEMO")
print("=" * 60)

model = NoisyORModelAPI("noisy_or_config.json")
print(f"\nLoaded: {model}")
print(f"Pool size : {model.pool_size:,} samples")
print(f"Precision tiers: {PRECISION_SAMPLES}")
print(f"Sources   : {model.source_names}\n")

# -----------------------------------------------------------------------
# 2.  Single predictions — precision tiers
# -----------------------------------------------------------------------
flags = np.array([1, 0, 1, 0, 0, 1, 0, 0])   # Sources 1, 3, 6 matched

print("=== Single Prediction — Precision Tiers ===")
for tier in ("fast", "standard", "high"):
    t0     = time.perf_counter()
    result = model.predict(flags, precision=tier, label=f"Customer_A [{tier}]")
    elapsed = (time.perf_counter() - t0) * 1000
    print(
        f"  {tier:<10}  risk={result['risk_score']:.1%}  "
        f"certainty={result['certainty_label']:<6}  "
        f"p5={result['p5']:.1%}  p95={result['p95']:.1%}  "
        f"({elapsed:.1f} ms)"
    )
print()

# -----------------------------------------------------------------------
# 3.  Cache behavior
# -----------------------------------------------------------------------
print("=== Cache Behavior ===")
# First call — computed and cached
t0 = time.perf_counter()
model.predict(flags, precision="standard")
t1 = (time.perf_counter() - t0) * 1000

# Second call — same flags, same precision → returned from cache instantly
t2 = time.perf_counter()
model.predict(flags, precision="standard")
t3 = (time.perf_counter() - t2) * 1000

print(f"  First call  (computed) : {t1:.2f} ms")
print(f"  Second call (cached)   : {t3:.2f} ms")
print(f"  Cache info             : {model.cache_info}")
print()

# -----------------------------------------------------------------------
# 4.  Other contributors — same as base model
# -----------------------------------------------------------------------
result = model.predict(flags, precision="standard", label="Customer_A")
print("=== Other Contributors (single prediction) ===")
print(f"  Primary driver : {result['primary_driver']}  "
      f"(+{result['primary_driver_contribution']:.1%})")
if result["other_matches"]:
    for m in result["other_matches"]:
        print(f"  • {m['source']:<22}  marginal +{m['marginal_contribution']:.1%}  "
              f"source_mean={m['source_mean']:.3f}")
print()

# -----------------------------------------------------------------------
# 5.  Vectorized batch prediction
# -----------------------------------------------------------------------
batch_inputs = np.array([
    [1, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 1, 0],
    [1, 1, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 0, 0],
])

batch_labels = [
    "Customer_A", "Customer_B", "Customer_C", "Customer_D",
    "Customer_E", "Customer_F", "Customer_G", "Customer_H",
]

print("=== Vectorized Batch Prediction ===")
t0 = time.perf_counter()
df = model.predict_batch(batch_inputs, precision="standard", labels=batch_labels)
elapsed = (time.perf_counter() - t0) * 1000
print(f"  {len(batch_inputs)} records scored in {elapsed:.1f} ms\n")

display_cols = [
    "Label", "Risk_Score", "Certainty_Label",
    "Primary_Driver", "Primary_Driver_Contribution", "Active_Sources",
]
print(df[display_cols].to_string(index=False))
print()

# Marginal contributions
contrib_cols = ["Label"] + [
    col for col in df.columns
    if col.endswith("_MarginalContrib") and df[col].any()
]
print("=== Marginal Contributions ===")
print(df[contrib_cols].to_string(index=False))
print()

# -----------------------------------------------------------------------
# 6.  Performance benchmark — API model vs. base model
# -----------------------------------------------------------------------
print("=== Performance Benchmark ===")

N_BENCHMARK = 200
rng_bench   = np.random.default_rng(0)
bench_flags = rng_bench.integers(0, 2, size=(N_BENCHMARK, 8))

# Base model — looped
base_model = NoisyORModel("noisy_or_config.json")
t0 = time.perf_counter()
for row in bench_flags:
    base_model.predict(np.array(row))
base_elapsed = (time.perf_counter() - t0) * 1000

# API model — vectorized batch
t0 = time.perf_counter()
model.predict_batch(bench_flags, precision="standard")
api_elapsed = (time.perf_counter() - t0) * 1000

print(f"  {N_BENCHMARK} records, standard precision (10k samples)")
print(f"  Base model (looped)        : {base_elapsed:7.1f} ms  "
      f"({base_elapsed/N_BENCHMARK:.2f} ms/record)")
print(f"  API model  (vectorized)    : {api_elapsed:7.1f} ms  "
      f"({api_elapsed/N_BENCHMARK:.2f} ms/record)")
print(f"  Speedup                    : {base_elapsed/api_elapsed:.1f}×")
print()

# -----------------------------------------------------------------------
# 7.  Hot reload — update config and reload without restarting
# -----------------------------------------------------------------------
print("=== Hot Reload ===")
print(f"  Before reload: {model}")

updated_sources = {
    "Watchlist Hit":        {"mu": 0.70, "kappa": 6.0},
    "Adverse Media":        {"mu": 0.40, "kappa": 8.0},
    "PEP Match":            {"mu": 0.55, "kappa": 5.0},
    "Sanctions Screen":     {"mu": 0.80, "kappa": 10.0},
    "High-Risk Geography":  {"mu": 0.35, "kappa": 7.0},
    "Transaction Flag":     {"mu": 0.25, "kappa": 9.0},
    "Industry Risk":        {"mu": 0.20, "kappa": 6.0},
    "Ownership Complexity": {"mu": 0.30, "kappa": 4.0},
}

# Write new config using save_config (inherited from base model)
model.save_config(
    sources_dict=updated_sources,
    output_path="noisy_or_config.json",   # overwrite in place
    bump="minor",
)

# Rebuild pool from new config — zero downtime
model.reload()
print(f"  After reload : {model}")
print(f"  New sources  : {model.source_names}\n")

# Confirm cache was cleared by reload
print(f"  Cache after reload: {model.cache_info}")
print()

# -----------------------------------------------------------------------
# 8.  Export — identical to base model
# -----------------------------------------------------------------------
# Restore original config for clean outputs
original_sources = {
    "Source 1": {"mu": 0.60, "kappa": 5.0},
    "Source 2": {"mu": 0.45, "kappa": 8.0},
    "Source 3": {"mu": 0.30, "kappa": 10.0},
    "Source 4": {"mu": 0.20, "kappa": 6.0},
    "Source 5": {"mu": 0.15, "kappa": 4.0},
    "Source 6": {"mu": 0.50, "kappa": 7.0},
    "Source 7": {"mu": 0.10, "kappa": 12.0},
    "Source 8": {"mu": 0.35, "kappa": 9.0},
}
model.save_config(original_sources, "noisy_or_config.json", version="1.0.0")
model.reload()

df_out = model.export_batch(
    batch_inputs,
    "noisy_or_results_api.xlsx",
    labels=batch_labels,
)
print("\nExport complete.")
print(df_out[["Label", "Risk_Score", "Certainty_Label"]].to_string(index=False))
