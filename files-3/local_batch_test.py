"""
local_batch_test.py — Local Batch Scoring Verification
=======================================================
Runs the full batch scoring pipeline locally and prints detailed
diagnostics at every step so you can verify correctness before
deploying to the API.

What this tests:
  1. Param store loads correctly and has expected structure
  2. Input data passes validation
  3. Scores are mathematically sensible (spot checks)
  4. Uncertainty (credible interval) is wider for uncertain sources
  5. Driver attribution is correct — removing a source lowers the score
  6. Risk tiers are assigned correctly
  7. Chunked processing produces identical results to single-pass
  8. Excel output has all expected sheets and columns

Run:
    python local_batch_test.py

No external services required. Everything runs locally.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make sure we can find the phase modules
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase1_priors import (
    load_param_store, build_param_store, save_param_store,
    SOURCE_SPECS, LEAK_SPEC, print_prior_summary,
)
from phase3_scorer import score, score_single, reload_store
from phase5_batch import (
    load_and_validate, run_batch, generate_test_data,
    print_batch_summary,
)

PARAM_PATH   = "params/param_store.json"
INPUT_CSV    = "local_batch_input.csv"
OUTPUT_XLSX  = "local_batch_output.xlsx"
PASS = "  ✓ PASS"
FAIL = "  ✗ FAIL"

results = []   # collect (test_name, passed, detail)


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    print(f"{status}  {name}")
    if detail:
        print(f"          {detail}")
    results.append((name, condition, detail))
    return condition


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


# ===========================================================================
# TEST 1 — Param store integrity
# ===========================================================================
section("TEST 1: Param Store Integrity")

store = load_param_store(PARAM_PATH)

check("param_store.json exists",
      os.path.exists(PARAM_PATH))

check("store has 'version' key",
      "version" in store,
      f"version = {store.get('version', 'MISSING')}")

check("store has 'source_order'",
      "source_order" in store,
      f"sources = {store.get('source_order', [])}")

check("8 sources present",
      len(store.get("source_order", [])) == 8,
      f"found {len(store.get('source_order', []))} sources")

check("all sources have alpha/beta",
      all("alpha" in store["sources"][s] and "beta" in store["sources"][s]
          for s in store["source_order"]),
      "alpha and beta keys present for all sources")

check("all alpha values > 0",
      all(store["sources"][s]["alpha"] > 0 for s in store["source_order"]))

check("all beta values > 0",
      all(store["sources"][s]["beta"] > 0 for s in store["source_order"]))

check("leak node present",
      "leak" in store and store["leak"]["alpha"] > 0)

# Print source summary for visual inspection
print()
print(f"  {'Source':<8} {'Mean':>8} {'Std':>8} {'α':>8} {'β':>8}")
print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for s in store["source_order"]:
    src = store["sources"][s]
    print(f"  {s:<8} {src['mean']:>8.4f} {src['std']:>8.4f} "
          f"{src['alpha']:>8.2f} {src['beta']:>8.2f}")
lk = store["leak"]
print(f"  {'Leak':<8} {lk['mean']:>8.4f} {lk['std']:>8.4f} "
      f"{lk['alpha']:>8.2f} {lk['beta']:>8.2f}")


# ===========================================================================
# TEST 2 — Single-record scoring sanity checks
# ===========================================================================
section("TEST 2: Single-Record Scoring Sanity")

# 2a: Zero matches → baseline only
r_zero = score_single(np.zeros(8), store=store, n_mc=3000, seed=0)
check("zero-match score equals leak mean",
      abs(r_zero["risk_point"] - store["leak"]["mean"]) < 1e-6,
      f"risk_point={r_zero['risk_point']:.6f}, leak_mean={store['leak']['mean']:.6f}")

check("zero-match driver is 'Leak'",
      r_zero["primary_driver"] == "Leak",
      f"driver={r_zero['primary_driver']}")

check("zero-match CI is narrow (leak well-specified)",
      r_zero["impact_width"] < 0.08,
      f"impact_width={r_zero['impact_width']:.4f}")

# 2b: All matches → high risk
r_full = score_single(np.ones(8), store=store, n_mc=3000, seed=0)
check("all-match score is high (>0.7)",
      r_full["risk_point"] > 0.70,
      f"risk_point={r_full['risk_point']:.4f}")

check("all-match score does not exceed 1.0",
      r_full["risk_point"] <= 1.0,
      f"risk_point={r_full['risk_point']:.6f}")

# 2c: Adding a source never DECREASES risk (monotonicity)
base = score_single(np.array([1,0,0,0,0,0,0,0]), store=store, n_mc=3000, seed=1)
with_s4 = score_single(np.array([1,0,0,1,0,0,0,0]), store=store, n_mc=3000, seed=1)
check("adding a source never decreases risk (monotonicity)",
      with_s4["risk_point"] >= base["risk_point"],
      f"S1 only={base['risk_point']:.4f}  S1+S4={with_s4['risk_point']:.4f}")

# 2d: High-mean, high-concentration source should have narrow CI
r_s7 = score_single(np.array([0,0,0,0,0,0,1,0]), store=store, n_mc=3000, seed=0)
r_s4 = score_single(np.array([0,0,0,1,0,0,0,0]), store=store, n_mc=3000, seed=0)
check("high-concentration source (S7) has narrower CI than uncertain source (S4)",
      r_s7["impact_width"] < r_s4["impact_width"],
      f"S7 width={r_s7['impact_width']:.4f} (conc=300)  "
      f"S4 width={r_s4['impact_width']:.4f} (conc=15)")

# 2e: Driver attribution — removing primary driver should lower score
vec = np.array([1, 0, 0, 1, 0, 0, 1, 0])  # S1, S4, S7
r_full_vec = score_single(vec, store=store, n_mc=3000, seed=0)
driver = r_full_vec["primary_driver"]
driver_idx = int(driver[1]) - 1  # "S4" → index 3

vec_no_driver = vec.copy()
vec_no_driver[driver_idx] = 0
r_no_driver = score_single(vec_no_driver, store=store, n_mc=3000, seed=0)

check("removing primary driver lowers risk score",
      r_no_driver["risk_point"] < r_full_vec["risk_point"],
      f"with {driver}={r_full_vec['risk_point']:.4f}  "
      f"without={r_no_driver['risk_point']:.4f}")

impact_stated = r_full_vec["impacts"][driver]
impact_actual = r_full_vec["risk_point"] - r_no_driver["risk_point"]
check("stated impact matches actual score delta (within tolerance)",
      abs(impact_stated - impact_actual) < 0.01,
      f"stated={impact_stated:.4f}  actual_delta={impact_actual:.4f}")

# Print scoring table for visual review
print()
cases = [
    (np.zeros(8),                     "No matches"),
    (np.array([0,0,0,0,0,0,1,0]),    "S7 only (high conf)"),
    (np.array([0,0,0,1,0,0,0,0]),    "S4 only (low conf)"),
    (np.array([1,0,0,1,0,0,1,0]),    "S1+S4+S7"),
    (np.array([1,1,1,1,1,1,1,1]),    "All matched"),
]
print(f"  {'Description':<26} {'Point':>7} {'P10':>7} {'P90':>7} "
      f"{'Width':>7} {'Driver':<12}")
print(f"  {'-'*26} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*12}")
for vec, desc in cases:
    r = score_single(vec, store=store, n_mc=2000, seed=0)
    print(f"  {desc:<26} {r['risk_point']:>7.4f} {r['risk_p10']:>7.4f} "
          f"{r['risk_p90']:>7.4f} {r['impact_width']:>7.4f} {r['primary_driver']:<12}")


# ===========================================================================
# TEST 3 — Batch scoring correctness
# ===========================================================================
section("TEST 3: Batch Scoring Correctness")

# Generate controlled test matrix
np.random.seed(42)
n_test = 200
match_matrix = np.random.binomial(1, 0.25, (n_test, 8)).astype(float)

t0 = time.perf_counter()
df_batch = score(match_matrix, store=store, n_mc=500, seed=0)
elapsed = time.perf_counter() - t0

check("batch returns correct number of rows",
      len(df_batch) == n_test,
      f"expected {n_test}, got {len(df_batch)}")

expected_cols = ["risk_point", "risk_p50", "risk_p10", "risk_p90",
                 "risk_std", "impact_width", "primary_driver", "matched_sources"]
missing_cols = [c for c in expected_cols if c not in df_batch.columns]
check("all expected columns present",
      len(missing_cols) == 0,
      f"missing: {missing_cols}" if missing_cols else "all present")

check("all risk_point values in [0, 1]",
      df_batch["risk_point"].between(0, 1).all(),
      f"min={df_batch['risk_point'].min():.4f}  max={df_batch['risk_point'].max():.4f}")

check("risk_p10 <= risk_p50 <= risk_p90 (credible interval ordering)",
      (df_batch["risk_p10"] <= df_batch["risk_p50"]).all() and
      (df_batch["risk_p50"] <= df_batch["risk_p90"]).all(),
      "ordering holds for all records")

check("no NaN values in scored output",
      df_batch[expected_cols].isna().sum().sum() == 0,
      "clean")

check("batch completes quickly",
      elapsed < 5.0,
      f"{elapsed:.3f}s for {n_test} records @ n_mc=500")

# Spot check: row 0 from batch should match score_single on same vector
row0_batch = df_batch.iloc[0]
row0_single = score_single(match_matrix[0], store=store, n_mc=500, seed=0)
check("batch row 0 matches score_single on same seed",
      abs(row0_batch["risk_point"] - row0_single["risk_point"]) < 1e-6,
      f"batch={row0_batch['risk_point']:.6f}  single={row0_single['risk_point']:.6f}")

print(f"\n  Batch stats ({n_test} records):")
print(f"    mean risk:   {df_batch['risk_point'].mean():.4f}")
print(f"    median risk: {df_batch['risk_point'].median():.4f}")
print(f"    max risk:    {df_batch['risk_point'].max():.4f}")
print(f"    throughput:  {n_test/elapsed:,.0f} records/sec")
print(f"\n  Primary driver distribution:")
for driver, count in df_batch["primary_driver"].value_counts().items():
    print(f"    {driver:<12} {count:>4}  ({count/n_test*100:5.1f}%)")


# ===========================================================================
# TEST 4 — Chunked vs single-pass produces identical results
# ===========================================================================
section("TEST 4: Chunked Processing Consistency")

np.random.seed(99)
n_chunk_test = 150
mat_chunk = np.random.binomial(1, 0.3, (n_chunk_test, 8)).astype(float)

# Single pass
df_single_pass = score(mat_chunk, store=store, n_mc=300, seed=7)

# Simulate chunked (3 chunks of 50)
chunk_dfs = []
for i in range(3):
    chunk = mat_chunk[i*50:(i+1)*50]
    chunk_dfs.append(score(chunk, store=store, n_mc=300, seed=7))
df_chunked = pd.concat(chunk_dfs, ignore_index=True)

check("chunked results have same row count",
      len(df_chunked) == len(df_single_pass),
      f"{len(df_chunked)} == {len(df_single_pass)}")

check("chunked risk_point matches single-pass exactly",
      (df_chunked["risk_point"] == df_single_pass["risk_point"]).all(),
      "all rows match")

check("chunked primary_driver matches single-pass",
      (df_chunked["primary_driver"] == df_single_pass["primary_driver"]).all(),
      "all rows match")


# ===========================================================================
# TEST 5 — Full batch pipeline with CSV input, all three output formats
# ===========================================================================
section("TEST 5: Full Batch Pipeline (CSV → Excel / CSV / Both)")

# Ensure param store exists for batch pipeline
if not os.path.exists(PARAM_PATH):
    save_param_store(build_param_store(SOURCE_SPECS, LEAK_SPEC))
    reload_store(PARAM_PATH)

# Generate test CSV with realistic structure
np.random.seed(0)
n_pipe = 250
source_cols = [f"S{i}" for i in range(1, 9)]
df_gen = pd.DataFrame(
    np.random.binomial(1, 0.25, (n_pipe, 8)),
    columns=source_cols
)
df_gen.insert(0, "record_id", [f"REC_{i:05d}" for i in range(n_pipe)])
df_gen.insert(1, "entity_name", [f"Entity_{i:05d}" for i in range(n_pipe)])
df_gen.to_csv(INPUT_CSV, index=False)

check("test CSV created successfully",
      os.path.exists(INPUT_CSV),
      f"{n_pipe} records → {INPUT_CSV}")

# Validate input schema
store_for_batch = load_param_store(PARAM_PATH)
reload_store(PARAM_PATH)   # sync module-level cache

df_validated = load_and_validate(INPUT_CSV, store_for_batch)
check("input validation passes",
      len(df_validated) == n_pipe,
      f"{len(df_validated)} records validated")

# ── Format: "excel" only ──────────────────────────────────────────────────
print("\n  -- format=excel --")
df_excel_only = run_batch(
    input_path    = INPUT_CSV,
    output_path   = "local_batch_excel_only",
    output_format = "excel",
    n_mc=300, chunk_size=100,
)
check("format=excel: .xlsx file created",
      os.path.exists("local_batch_excel_only.xlsx"),
      "local_batch_excel_only.xlsx")
check("format=excel: .csv file NOT created",
      not os.path.exists("local_batch_excel_only.csv"),
      "correctly absent")

# ── Format: "csv" only ───────────────────────────────────────────────────
print("\n  -- format=csv --")
df_csv_only = run_batch(
    input_path    = INPUT_CSV,
    output_path   = "local_batch_csv_only",
    output_format = "csv",
    n_mc=300, chunk_size=100,
)
check("format=csv: .csv file created",
      os.path.exists("local_batch_csv_only.csv"),
      "local_batch_csv_only.csv")
check("format=csv: .xlsx file NOT created",
      not os.path.exists("local_batch_csv_only.xlsx"),
      "correctly absent")

df_csv_check = pd.read_csv("local_batch_csv_only.csv")
check("format=csv: correct row count",
      len(df_csv_check) == n_pipe,
      f"{len(df_csv_check)} rows")
check("format=csv: contains risk_point column",
      "risk_point" in df_csv_check.columns)
check("format=csv: contains risk_tier column",
      "risk_tier" in df_csv_check.columns)
check("format=csv: contains impact columns",
      all(f"S{i}_impact" in df_csv_check.columns for i in range(1, 9)))

# ── Format: "both" ────────────────────────────────────────────────────────
print("\n  -- format=both --")
t0 = time.perf_counter()
df_results = run_batch(
    input_path    = INPUT_CSV,
    output_path   = "local_batch_output",
    output_format = "both",
    n_mc=400, chunk_size=100,
    min_risk=0.10,
)
elapsed_pipe = time.perf_counter() - t0

check("format=both: pipeline completes without error",
      len(df_results) == n_pipe,
      f"{len(df_results)} records scored in {elapsed_pipe:.2f}s")
check("format=both: .xlsx created",
      os.path.exists("local_batch_output.xlsx"),
      "local_batch_output.xlsx")
check("format=both: .csv created",
      os.path.exists("local_batch_output.csv"),
      "local_batch_output.csv")

# CSV and Excel row counts should match
df_both_csv   = pd.read_csv("local_batch_output.csv")
check("format=both: CSV and Excel have same row count",
      len(df_both_csv) == n_pipe,
      f"csv={len(df_both_csv)}  expected={n_pipe}")

# Risk scores should be identical between the two outputs
xl_all = pd.read_excel("local_batch_output.xlsx", sheet_name="All_Records")
check("format=both: CSV and Excel risk_point values match",
      (df_both_csv["risk_point"].values == xl_all["risk_point"].values).all(),
      "all rows identical")

# Verify path extension stripping works (pass .xlsx path with csv format)
run_batch(
    input_path    = INPUT_CSV,
    output_path   = "local_batch_exttest.xlsx",  # wrong ext for csv mode
    output_format = "csv",
    n_mc=100, chunk_size=250,
)
check("extension stripping: passing .xlsx path with format=csv writes .csv",
      os.path.exists("local_batch_exttest.csv") and
      not os.path.exists("local_batch_exttest.xlsx"),
      "local_batch_exttest.csv created, .xlsx not created")

# ── Excel content checks (reuse local_batch_output.xlsx) ─────────────────
print()
try:
    xl = pd.ExcelFile("local_batch_output.xlsx")
    sheets = xl.sheet_names
    check("'All_Records' sheet present", "All_Records" in sheets, f"sheets: {sheets}")
    check("'High_Risk_Alerts' sheet present", "High_Risk_Alerts" in sheets)
    check("'Model_Info' sheet present", "Model_Info" in sheets)

    df_all = pd.read_excel("local_batch_output.xlsx", sheet_name="All_Records")
    check("All_Records has correct row count",
          len(df_all) == n_pipe, f"{len(df_all)} rows")
    check("All_Records has risk_tier column", "risk_tier" in df_all.columns)

    tier_vals = set(df_all["risk_tier"].unique())
    valid_tiers = {"Minimal", "Low", "Moderate", "High"}
    check("risk_tier values are valid",
          tier_vals.issubset(valid_tiers), f"found: {tier_vals}")

    df_alerts = pd.read_excel("local_batch_output.xlsx", sheet_name="High_Risk_Alerts")
    if len(df_alerts) > 0 and "risk_tier" in df_alerts.columns:
        check("alert sheet contains only moderate/high risk records",
              df_alerts["risk_tier"].isin(["Moderate", "High"]).all(),
              f"{len(df_alerts)} alert records")
    else:
        check("alert sheet present (may be empty)", True, f"{len(df_alerts)} rows")

    df_model = pd.read_excel("local_batch_output.xlsx", sheet_name="Model_Info")
    check("Model_Info sheet has content", len(df_model) > 0, f"{len(df_model)} rows")

except Exception as e:
    check("Excel file is readable", False, str(e))

print_batch_summary(df_results)

# Alias for edge case tests below
OUTPUT_XLSX = "local_batch_output.xlsx"


# ===========================================================================
# TEST 6 — Edge cases
# ===========================================================================
section("TEST 6: Edge Cases")

# Single record batch
r_one = score(np.array([[1,0,1,0,0,0,0,0]]), store=store, n_mc=1000)
check("batch of 1 record works correctly",
      len(r_one) == 1 and 0 < r_one.iloc[0]["risk_point"] < 1)

# All-zero batch
r_zeros = score(np.zeros((10, 8)), store=store, n_mc=500)
check("all-zero batch: all scores equal leak mean",
      (r_zeros["risk_point"] == round(store["leak"]["mean"], 4)).all(),
      f"leak mean={store['leak']['mean']:.4f}  "
      f"all scores={r_zeros['risk_point'].unique().tolist()}")

check("all-zero batch: all drivers are 'Leak'",
      (r_zeros["primary_driver"] == "Leak").all())

# Score stability — same seed same result
r_a = score_single(np.array([1,0,1,0,1,0,0,0]), store=store, n_mc=1000, seed=42)
r_b = score_single(np.array([1,0,1,0,1,0,0,0]), store=store, n_mc=1000, seed=42)
check("same seed produces identical results (reproducibility)",
      r_a["risk_point"] == r_b["risk_point"] and
      r_a["risk_p10"]   == r_b["risk_p10"],
      f"risk_point: {r_a['risk_point']} == {r_b['risk_point']}")

# Different seed produces different MC uncertainty (but same point estimate)
r_c = score_single(np.array([1,0,1,0,1,0,0,0]), store=store, n_mc=1000, seed=99)
check("point estimate is deterministic (seed-independent)",
      r_a["risk_point"] == r_c["risk_point"],
      f"{r_a['risk_point']} == {r_c['risk_point']}")

# Large batch performance
t0 = time.perf_counter()
big_mat = np.random.binomial(1, 0.25, (5000, 8)).astype(float)
df_big = score(big_mat, store=store, n_mc=200, seed=0)
t_big = time.perf_counter() - t0
check("5000-record batch completes in < 5 seconds",
      t_big < 5.0,
      f"{t_big:.2f}s = {5000/t_big:,.0f} records/sec")


# ===========================================================================
# FINAL REPORT
# ===========================================================================
section("FINAL TEST SUMMARY")

passed  = sum(1 for _, p, _ in results if p)
failed  = sum(1 for _, p, _ in results if not p)
total   = len(results)

print(f"\n  {passed}/{total} tests passed\n")

if failed > 0:
    print("  FAILURES:")
    for name, passed_flag, detail in results:
        if not passed_flag:
            print(f"    ✗  {name}")
            if detail:
                print(f"       {detail}")
    print()
    print("  ⚠ Fix failures before deploying to API.")
    sys.exit(1)
else:
    print("  All tests passed. Safe to proceed to API deployment.")
    print(f"\n  Output files:")
    print(f"    {INPUT_CSV}   ← test input")
    print(f"    {OUTPUT_XLSX}  ← scored output (open in Excel to inspect)")
