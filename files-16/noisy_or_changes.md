# Noisy-OR Model — Change Summary

Changes made across six files in the most recent improvement session.  
Each section describes what changed, why, and how to use it.

---

## Files Changed

| File | Type | Change |
|---|---|---|
| `noisy_or_model.py` | Modified | 5 improvements |
| `noisy_or_model_api.py` | Modified | 3 improvements |
| `noisy_or_updater.py` | Modified | 2 improvements |
| `noisy_or_model_copula.py` | **New file** | Gaussian copula correlated model |
| `noisy_or_config.json` | Modified | Added `risk_bands` example |
| `noisy_or_documentation.md` | Modified | 8 sections added or updated |

---

## noisy_or_model.py

### 1. `reload()` — Standard Model Hot Reload

**Problem:** After `FeedbackUpdater.apply()` writes a new config, the only way to get updated parameters into a `NoisyORModel` was to reinstantiate the object, which rebuilt the 50,000-sample pool from scratch. The API model had `reload()`; the standard model did not.

**Fix:** `__init__` was refactored to extract all config parsing into a shared `_load_from_config()` method. `reload()` calls `_load_from_config()` then rebuilds the batch pool in-place — exactly what reinstantiation did, but without creating a new object.

```python
model   = NoisyORModel("noisy_or_config.json")
updater = FeedbackUpdater("noisy_or_config.json")

updater.apply("feedback_march.csv")
model.reload()   # picks up new parameters immediately
print(model)     # NoisyORModel(v1.0.1, ...)
```

> **Note:** unlike `NoisyORModelAPI.reload()`, the standard model's `reload()` is not thread-safe. Use `NoisyORModelAPI` if concurrent batch calls may be in flight.

---

### 2. Configurable Risk Bands

**Problem:** Batch output gave a raw `Risk_Score` and a `Certainty_Label`, but no way to segment records into named business tiers (e.g. "Critical / High / Medium / Low") without writing post-processing logic separately for every consumer of the output.

**Fix:** An optional `risk_bands` array in the config defines named thresholds. When present, a `Risk_Band` column is automatically added to all batch output — DataFrame, CSV, and Excel.

**Config:**
```json
"risk_bands": [
  {"label": "Critical", "min_score": 0.75},
  {"label": "High",     "min_score": 0.50},
  {"label": "Medium",   "min_score": 0.25},
  {"label": "Low",      "min_score": 0.00}
]
```

**Usage:**
```python
model = NoisyORModel("noisy_or_config.json")
print(model.risk_bands)
# [(0.75, 'Critical'), (0.5, 'High'), (0.25, 'Medium'), (0.0, 'Low')]

df = model.predict_batch(matrix)
print(df[["Label", "Risk_Score", "Risk_Band"]])
#      Label  Risk_Score  Risk_Band
# Customer_A      0.8597   Critical
# Customer_B      0.6045       High
```

`save_config()` accepts a `risk_bands` parameter. If omitted, existing bands are preserved. Pass `risk_bands=[]` to remove all bands from the saved config.

---

### 3. Excel Writer Performance (`_write_excel`)

**Problem:** `_write_excel()` used `df.iterrows()` to populate Excel cells — the slowest DataFrame iteration pattern in pandas. For a 10,000-row export with many source columns, this generated hundreds of thousands of individual `ws.cell()` calls.

**Fix:** All column data is extracted to Python lists in a single pass (`.tolist()` per column). Each row is then written via `ws.append(row_values)` — one call per row for values instead of one call per cell. A second pass applies fills and number formats. Value insertion is now O(n_rows) calls rather than O(n_rows × n_cols).

No change to output — the workbook layout, formatting, and column structure are identical.

---

### 4. `stream_export()` — Memory-Efficient Large Exports

**Problem:** `export_batch()` loads the full result DataFrame into memory before writing a single byte. For 500,000-record screening runs this could require multiple gigabytes of RAM.

**Fix:** `stream_export()` chunks the input matrix into `chunk_size`-record blocks, calls `predict_batch()` on each chunk, and appends incrementally to the output CSV. Peak memory is proportional to `chunk_size`, not the dataset size.

```python
# Score 500,000 records without holding all results in RAM
model.stream_export(
    active_matrix = large_matrix,
    output_path   = "results.csv",
    chunk_size    = 10_000,        # tune to available RAM
)
```

Only CSV output is supported — Excel requires the full workbook in memory. For Excel, use `export_batch()` on datasets that fit comfortably in RAM. Both pandas and polars backends are supported.

---

### 5. Source Parameter Validation at Load Time

**Problem:** `save_config()` validated `mu` and `kappa` before writing, but a config file that was manually edited or corrupted (e.g. `mu=1.5` or `kappa=-2`) would only fail at the first prediction call with a cryptic NumPy error.

**Fix:** `_load_from_config()` now calls `_validate_source_params()` on every source's `mu` and `kappa` when the config is read. A bad config raises a clear `ValueError` at instantiation or `reload()` time, not silently during prediction.

---

## noisy_or_model_api.py

### 1. `reload()` Updated for `_load_from_config()`

The API model's `reload()` was updated to delegate to the shared `_load_from_config()` method (extracted in the base model refactor). This means `risk_bands`, source names, mu/kappa values, and the version string are all reloaded atomically under the thread lock from the same code path as `__init__`. Previously, `reload()` duplicated the attribute-setting logic inline, creating a maintenance risk.

### 2. `stream_export()` Added

`stream_export()` is added to the API model with a `precision` parameter (in addition to the base model's parameters), matching the API model's existing `predict_batch()` interface.

```python
api_model.stream_export(
    active_matrix = large_matrix,
    output_path   = "results.csv",
    precision     = "fast",      # use fast tier for maximum throughput
    chunk_size    = 10_000,
)
```

### 3. Risk Bands Inherited

Because the API model inherits `_format_single()`, `_format_row()`, and `_to_polars()` from the base model, and `reload()` now uses `_load_from_config()`, risk bands work automatically in the API model with no additional changes.

---

## noisy_or_updater.py

### 1. `kappa_cap` — Preventing Model Staleness

**Problem:** As feedback accumulates, `kappa` grows without bound. After hundreds of observations, `kappa` becomes very large — new feedback has almost no effect because the model is treating its prior as equivalent to thousands of data points. A source whose true risk rate shifts over time (e.g. a screening signal that becomes less reliable) would not update meaningfully.

**Fix:** `kappa_cap` is an optional upper bound on `kappa` after each update. When `kappa_new` would exceed the cap, `mu` is preserved exactly but `kappa` is clamped and `alpha`/`beta` are rescaled accordingly. This implements **sliding-window Bayesian updating** — the model retains its learned `mu` estimate but stays responsive to future feedback.

```python
updater = FeedbackUpdater(
    "noisy_or_config.json",
    kappa_cap = 100.0,   # trust at most ~100 observations of prior evidence
)
```

When `kappa_cap` is triggered for a source, the audit log records `"kappa_capped": true` for that source's entry. `kappa_cap` defaults to `None` (unbounded), preserving existing behaviour.

**The math:**
```
if kappa_new > kappa_cap:
    mu_new    = alpha_new / kappa_new   (unchanged)
    kappa_new = kappa_cap               (clamped)
    alpha_cap = mu_new * kappa_cap
    beta_cap  = (1 - mu_new) * kappa_cap
```

### 2. `case_contributions` — Full Audit Trail for Case-Level Feedback

**Problem:** The audit log recorded `n_positive` and `n_negative` totals per source, but not which individual cases contributed what credit. This made it impossible to answer "what would the parameters look like if we removed case X?" or to replay history from scratch without re-running the original feedback files.

**Fix:** The case-level parser (`_parse_case_feedback_csv`) now returns a `case_contributions` list alongside the usual `counts` dict. Each entry records exactly what happened for one case:

```json
{
  "active_sources": ["Source 1", "Source 3"],
  "outcome": 1,
  "weight": 1.0,
  "attribution_method": "marginal",
  "per_source_credits": {
    "Source 1": 0.782341,
    "Source 3": 0.217659
  }
}
```

This list is stored under `"case_contributions"` in the audit log entry. Source-level (Format 1) files record `null` for this field since attribution is not computed. Every log entry is now fully self-contained for reconstruction and what-if analysis.

---

## noisy_or_model_copula.py *(New File)*

### Gaussian Copula — Correlated Source Model

**Problem:** The standard Noisy-OR model assumes source independence. When two sources frequently co-fire because they share an underlying driver (e.g. a watchlist hit and adverse media are often linked to the same high-risk subject), the independent model overestimates combined risk by treating the shared signal as two independent pieces of evidence.

**Solution:** `NoisyORModelCopula` inherits from `NoisyORModel` and overrides only the pool construction method. It uses a **Gaussian copula** to sample source probabilities jointly, preserving each source's Beta marginal distribution exactly while introducing user-specified pairwise correlation.

**How it works:**

For each Monte Carlo draw, instead of sampling each source's Beta distribution independently:
1. Draw correlated standard normals: `z ~ N(0, R)` via Cholesky decomposition of the correlation matrix `R`
2. Map to uniforms: `u_i = Φ(z_i)` via the normal CDF
3. Map to Beta variates: `p_i = Beta⁻¹(u_i; αᵢ, βᵢ)` via the inverse CDF

Each `p_i` still has exactly the right Beta marginal distribution, but sources with positive `rho` tend to draw high or low values together.

**Config:**
```json
"correlations": [
  {"source_a": "Watchlist Hit", "source_b": "Adverse Media", "rho": 0.65},
  {"source_a": "Watchlist Hit", "source_b": "PEP Match",     "rho": 0.40}
]
```

Only non-zero pairs need to be listed. All unlisted pairs default to `rho = 0` (independence). When all correlations are zero, the copula is mathematically identical to the standard model.

**Usage:**
```python
from noisy_or_model_copula import NoisyORModelCopula

model = NoisyORModelCopula("noisy_or_config_correlated.json")
model.print_correlations()

# Side-by-side comparison
flags       = np.array([1, 1, 0, 0])
correlated  = model.predict(flags)["risk_score"]
independent = model.with_independence().predict(flags)["risk_score"]

print(f"Correlated  (rho=0.65): {correlated:.1%}")
print(f"Independent (rho=0.00): {independent:.1%}")
# Correlated risk is lower — correctly accounts for shared signal
```

**Key properties:**
- All batch optimizations inherited unchanged: pool, vectorized scoring, deduplication, chunking, Polars backend, `reload()`, `stream_export()`, risk bands
- `is_independent` property: `True` when all configured rho values are zero
- `correlation_matrix` property: returns the full `(n_sources × n_sources)` matrix
- `with_independence()`: returns a new instance with all correlations stripped — useful for comparison
- `save_config()` overridden to preserve correlations alongside sources and risk bands
- If the correlation matrix is not positive semi-definite (inconsistent rho values), the nearest valid matrix is used automatically with a warning
- Requires `scipy`: `pip install scipy`

---

## noisy_or_config.json

`risk_bands` added with four tiers:

```json
"risk_bands": [
  {"label": "Critical", "min_score": 0.75},
  {"label": "High",     "min_score": 0.50},
  {"label": "Medium",   "min_score": 0.25},
  {"label": "Low",      "min_score": 0.00}
]
```

---

## noisy_or_documentation.md

| Section | Change |
|---|---|
| File Structure | Added `noisy_or_model_copula.py` |
| Installation | Added `pip install scipy` for copula model |
| Overview | Added description of `NoisyORModelCopula` as a third variant |
| Configuration Fields | Added `risk_bands` and `correlations` to the config field reference table |
| Configurable Risk Bands | New section — config format, runtime usage, `save_config()` integration |
| `reload()` Standard Model | New section — usage, thread-safety note |
| `stream_export()` | New section — usage, chunk_size guidance, CSV-only limitation |
| `kappa_cap` | New section — motivation, usage, the rescaling math |
| Correlated Sources | New section — when to use, config format, full usage examples, `with_independence()` comparison, copula math, saving correlated configs |
| Output Reference | Added `risk_band` to single prediction output table and `Risk_Band` to batch DataFrame columns table |
| Mathematical Reference | Added Gaussian copula sampling steps and `kappa_cap` rescaling formula |

---

## Dependency Summary

| Feature | Dependency | Required? |
|---|---|---|
| All existing features | `numpy`, `pandas`, `openpyxl` | Always |
| Polars backend | `polars` | Optional |
| File watcher | `watchdog` | Optional |
| Config locking | `filelock` | Optional (falls back to `fcntl` / threading) |
| Correlated model | `scipy` | Required for `NoisyORModelCopula` only |
