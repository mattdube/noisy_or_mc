"""
noisy_or_demo.py
================
Demonstrates all capabilities of NoisyORModel.

  1.  Load model and inspect the batch pool
  2.  Single prediction with pretty-print
  3.  Batch prediction — pandas (default)
  4.  Batch prediction — polars
  5.  Deduplication in action — repeated flag patterns scored once
  6.  Batch performance benchmark vs. old per-record loop
  7.  Export to Excel and CSV (pandas and polars)
  8.  Generate an updated config from a Python dictionary

Run:  python noisy_or_demo.py

Polars support requires:  pip install polars
"""

import time
import numpy as np
from noisy_or_model import NoisyORModel, _run_simulation, _POLARS_AVAILABLE

# -----------------------------------------------------------------------
# 1.  Load model
# -----------------------------------------------------------------------
model = NoisyORModel("noisy_or_config.json")

print(f"Loaded : {model}")
print(f"Sources: {model.source_names}")
print(f"Batch pool: {model.batch_pool_size:,} pre-drawn samples per source\n")

# -----------------------------------------------------------------------
# 2.  Single prediction
#     Array order must match source order in noisy_or_config.json:
#     [Source1, Source2, Source3, Source4, Source5, Source6, Source7, Source8]
# -----------------------------------------------------------------------
flags_single = np.array([1, 0, 1, 0, 0, 1, 0, 0])   # sources 1, 3, 6 matched

result = model.predict(
    active_flags=flags_single,
    label="Customer_A",
    seed=42,       # remove seed= for random results each run
)

model.print_result(result)

# Access individual fields programmatically
print(f"Risk score  : {result['risk_score']:.1%}")
print(f"Certainty   : {result['certainty_label']}")
print(f"Top driver  : {result['primary_driver']}  (+{result['primary_driver_contribution']:.1%})")
print()

# Iterate other contributors
print("=== Other Contributors (single prediction) ===")
if result["other_matches"]:
    for match in result["other_matches"]:
        print(f"  Source     : {match['source']}")
        print(f"  Marginal   : +{match['marginal_contribution']:.1%}")
        print(f"  Source Mean: {match['source_mean']:.3f}")
        print()
else:
    print("  No other contributors (only one source was active).")
print()

# -----------------------------------------------------------------------
# 3.  Batch prediction — pandas (default)
#     predict_batch() automatically uses:
#       * Pre-sampled pool  — no fresh Beta sampling per call
#       * Vectorized scoring — entire batch scored in one NumPy pass
#       * Duplicate-row cache — repeated flag combos scored only once
# -----------------------------------------------------------------------
batch_inputs = np.array([
    [1, 0, 1, 0, 0, 1, 0, 0],   # 3 sources match
    [0, 1, 0, 1, 0, 0, 1, 0],   # 3 different sources
    [1, 1, 0, 0, 1, 0, 0, 1],   # 4 sources — higher risk
    [0, 0, 0, 0, 0, 0, 0, 0],   # no sources — zero risk
    [1, 1, 1, 1, 1, 1, 1, 1],   # all sources — maximum risk
    [1, 0, 0, 0, 0, 0, 0, 0],   # only Source 1
    [0, 0, 0, 0, 0, 0, 0, 1],   # only Source 8
    [1, 1, 1, 0, 0, 0, 0, 0],   # top 3 sources
])

batch_labels = [
    "Customer_A", "Customer_B", "Customer_C", "Customer_D",
    "Customer_E", "Customer_F", "Customer_G", "Customer_H",
]

t0 = time.perf_counter()
df_pd = model.predict_batch(active_matrix=batch_inputs, labels=batch_labels)
                             # backend="pandas" is the default
batch_ms = (time.perf_counter() - t0) * 1000

print(f"=== Batch Results — pandas ({len(batch_inputs)} records, {batch_ms:.1f} ms) ===")
display_cols = [
    "Label", "Risk_Score", "Certainty_Label",
    "Primary_Driver", "Primary_Driver_Contribution", "Active_Sources",
]
print(df_pd[display_cols].to_string(index=False))
print()

# Per-source marginal contributions for active sources
contrib_cols = ["Label"] + [
    col for col in df_pd.columns
    if col.endswith("_MarginalContrib") and df_pd[col].any()
]
print("=== Marginal Contributions ===")
print(df_pd[contrib_cols].to_string(index=False))
print()

# -----------------------------------------------------------------------
# 4.  Batch prediction — polars
#     Pass backend="polars" to get a polars DataFrame instead.
#     Requires:  pip install polars
# -----------------------------------------------------------------------
print("=== Batch Results — polars ===")
if _POLARS_AVAILABLE:
    df_pl = model.predict_batch(
        active_matrix=batch_inputs,
        labels=batch_labels,
        backend="polars",
    )
    print(f"Type : {type(df_pl)}")
    print(f"Shape: {df_pl.shape}")
    print()
    # Polars uses different syntax for column selection
    print(df_pl.select(["Label", "Risk_Score", "Certainty_Label",
                         "Primary_Driver", "Active_Sources"]))
    print()
    # Polars-native operations work directly on the returned DataFrame
    print("High-certainty records (polars filter):")
    print(df_pl.filter(df_pl["Certainty_Label"] == "High")
               .select(["Label", "Risk_Score", "Certainty_Label"]))
    print()
else:
    print("  polars not installed — skipping polars demo.")
    print("  Install with:  pip install polars\n")

# -----------------------------------------------------------------------
# 5.  Deduplication in action
#     Build a large batch where the same 5 flag patterns repeat many
#     times.  The vectorized scorer only runs on the 5 unique rows —
#     results are mapped back to all 200 records instantly.
# -----------------------------------------------------------------------
print("=== Deduplication Demo ===")

unique_patterns = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],   # pattern A — 1 source
    [1, 1, 0, 0, 0, 0, 0, 0],   # pattern B — 2 sources
    [1, 0, 1, 0, 0, 1, 0, 0],   # pattern C — 3 sources
    [1, 1, 0, 0, 1, 0, 0, 1],   # pattern D — 4 sources
    [1, 1, 1, 1, 1, 1, 1, 1],   # pattern E — all sources
])

# Tile to 200 rows and shuffle
rng_dedup = np.random.default_rng(99)
large_batch = np.tile(unique_patterns, (40, 1))   # 200 rows, 5 unique patterns
large_batch = rng_dedup.permutation(large_batch)

t0 = time.perf_counter()
df_large = model.predict_batch(large_batch)
dedup_ms = (time.perf_counter() - t0) * 1000

unique_scores = df_large["Risk_Score"].nunique()
print(f"  {len(large_batch)} records, {len(unique_patterns)} unique flag patterns")
print(f"  Vectorized scorer ran on {len(unique_patterns)} unique rows only")
print(f"  {len(large_batch)} results assembled in {dedup_ms:.1f} ms")
print(f"  Distinct risk scores in output: {unique_scores} "
      f"(one per unique pattern, as expected)\n")

# -----------------------------------------------------------------------
# 6.  Performance benchmark — new vectorized path vs. old per-record loop
# -----------------------------------------------------------------------
print("=== Performance Benchmark ===")

N_BENCH = 500
rng_bench = np.random.default_rng(0)
bench_matrix = rng_bench.integers(0, 2, size=(N_BENCH, 8))

# Old path: Python loop with fresh Beta sampling per record
t0 = time.perf_counter()
for row in bench_matrix:
    _run_simulation(model._mu_values, model._kappa_values, row,
                    model._n_samples_default)
old_ms = (time.perf_counter() - t0) * 1000

# New path: vectorized + pool
t0 = time.perf_counter()
model.predict_batch(bench_matrix)
new_ms = (time.perf_counter() - t0) * 1000

print(f"  {N_BENCH} records  |  {model._n_samples_default:,} samples  |  {model.n_sources} sources")
print(f"  Old (loop + fresh sampling) : {old_ms:7.1f} ms  ({old_ms/N_BENCH:.2f} ms/record)")
print(f"  New (vectorized + pool)     : {new_ms:7.1f} ms  ({new_ms/N_BENCH:.2f} ms/record)")
print(f"  Speedup                     : {old_ms/new_ms:.1f}x\n")

# -----------------------------------------------------------------------
# 7.  Export to Excel and CSV — pandas and polars backends
# -----------------------------------------------------------------------

# pandas — Excel (always uses openpyxl regardless of backend)
model.export_batch(
    active_matrix=batch_inputs,
    output_path="noisy_or_results.xlsx",
    labels=batch_labels,
)

# pandas — CSV
model.export_batch(
    active_matrix=batch_inputs,
    output_path="noisy_or_results.csv",
    labels=batch_labels,
    backend="pandas",
)

# polars — CSV (uses polars' native write_csv — faster on large files)
if _POLARS_AVAILABLE:
    model.export_batch(
        active_matrix=batch_inputs,
        output_path="noisy_or_results_polars.csv",
        labels=batch_labels,
        backend="polars",
    )

# polars — Excel (Polars DataFrame is converted to pandas internally for writing)
if _POLARS_AVAILABLE:
    model.export_batch(
        active_matrix=batch_inputs,
        output_path="noisy_or_results_polars.xlsx",
        labels=batch_labels,
        backend="polars",
    )

print("\nFiles written:")
print("  * noisy_or_results.xlsx          (pandas)")
print("  * noisy_or_results.csv           (pandas)")
if _POLARS_AVAILABLE:
    print("  * noisy_or_results_polars.csv    (polars)")
    print("  * noisy_or_results_polars.xlsx   (polars → pandas for Excel)")
print()

print("Re-use the DataFrame without reloading the file:")
print(df_pd[["Label", "Risk_Score", "Certainty_Label"]].to_string(index=False))

# -----------------------------------------------------------------------
# 8.  Generate an updated config from a Python dictionary
# -----------------------------------------------------------------------
print("\n\n=== Config Generation ===")
print(f"Current config: version {model.version}  ({model.version_date})")

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

model.save_config(
    sources_dict=updated_sources,
    output_path="noisy_or_config_v2.json",
    bump="minor",
    # version="2.0.0",   # or supply an explicit version string
    # n_samples=50000,   # optionally override the sample count
)

model_v2 = NoisyORModel("noisy_or_config_v2.json")
print(f"New config loaded : {model_v2}")
print(f"Sources           : {model_v2.source_names}")
