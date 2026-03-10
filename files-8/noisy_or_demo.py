"""
noisy_or_demo.py
================
Demonstrates all capabilities of NoisyORModel:
  - Single prediction with pretty-print
  - Batch prediction → DataFrame
  - Batch export to Excel and CSV

Run:  python noisy_or_demo.py
"""

import numpy as np
from noisy_or_model import NoisyORModel

# -----------------------------------------------------------------------
# 1.  Load model from config
# -----------------------------------------------------------------------
model = NoisyORModel("noisy_or_config.json")

print(f"Loaded model: {model}")
print(f"Sources: {model.source_names}\n")

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

# Access other contributors and their marginal contributions directly
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
# 3.  Batch predictions
# -----------------------------------------------------------------------
# Each row = one record; columns correspond to sources in config order
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

df = model.predict_batch(
    active_matrix=batch_inputs,
    labels=batch_labels,
    seed=42,
)

# Quick view in notebook / terminal
display_cols = [
    "Label", "Risk_Score", "Certainty_Label",
    "Primary_Driver", "Primary_Driver_Contribution", "Active_Sources",
]
print("=== Batch Results (summary) ===")
print(df[display_cols].to_string(index=False))
print()

# Show per-source marginal contributions for all active sources
contrib_cols = ["Label"] + [
    col for col in df.columns if col.endswith("_MarginalContrib") and df[col].any()
]
print("=== Batch Results (marginal contributions) ===")
print(df[contrib_cols].to_string(index=False))
print()

# -----------------------------------------------------------------------
# 4.  Export to Excel and CSV
# -----------------------------------------------------------------------
df_excel = model.export_batch(
    active_matrix=batch_inputs,
    output_path="noisy_or_results.xlsx",
    labels=batch_labels,
    seed=42,
)

df_csv = model.export_batch(
    active_matrix=batch_inputs,
    output_path="noisy_or_results.csv",
    labels=batch_labels,
    seed=42,
)

print("\nDone. Files written:")
print("  • noisy_or_results.xlsx")
print("  • noisy_or_results.csv")
print()
print("Re-use the DataFrame without reloading the file:")
print(df_excel[["Label", "Risk_Score", "Certainty_Label"]].head())

# -----------------------------------------------------------------------
# 5.  Generate an updated config from a Python dictionary
#     Use this whenever you want to revise sources or parameters and
#     produce a new versioned config file — no manual JSON editing needed.
# -----------------------------------------------------------------------
print("\n\n=== Config Generation ===")
print(f"Current config: version {model.version}  ({model.version_date})")

# Define your updated sources as a plain dictionary:
#   key   = source name (used in all outputs and Excel column headers)
#   value = {"mu": <prior mean 0-1>, "kappa": <concentration > 0>}
updated_sources = {
    "Watchlist Hit":       {"mu": 0.70, "kappa": 6.0},
    "Adverse Media":       {"mu": 0.40, "kappa": 8.0},
    "PEP Match":           {"mu": 0.55, "kappa": 5.0},
    "Sanctions Screen":    {"mu": 0.80, "kappa": 10.0},
    "High-Risk Geography": {"mu": 0.35, "kappa": 7.0},
    "Transaction Flag":    {"mu": 0.25, "kappa": 9.0},
    "Industry Risk":       {"mu": 0.20, "kappa": 6.0},
    "Ownership Complexity":{"mu": 0.30, "kappa": 4.0},
}

# Save as a new config — version is auto-bumped (default: minor bump).
# Today's date is written automatically as version_date.
model.save_config(
    sources_dict=updated_sources,
    output_path="noisy_or_config_v2.json",
    bump="minor",               # "major", "minor", or "patch"
    # version="2.0.0",          # or supply an explicit version string
    # n_samples=50000,          # optionally override the sample count
)

# Load and verify the new config
model_v2 = NoisyORModel("noisy_or_config_v2.json")
print(f"New config loaded: {model_v2}")
print(f"Sources: {model_v2.source_names}")
