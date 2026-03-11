# Noisy-OR Bayesian Risk Model
## Technical Documentation

---

## Table of Contents

1. [Overview](#overview)
2. [How the Model Works](#how-the-model-works)
3. [File Structure](#file-structure)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Usage — Standard Model](#usage--standard-model)
   - [Single Prediction](#single-prediction)
   - [Batch Prediction](#batch-prediction)
   - [Exporting Results](#exporting-results)
   - [Generating a Config from a Python Dictionary](#generating-a-config-from-a-python-dictionary)
7. [Usage — API-Optimized Model](#usage--api-optimized-model)
   - [Loading the API Model](#loading-the-api-model)
   - [Precision Tiers](#precision-tiers)
   - [Single Prediction](#single-prediction-api)
   - [Vectorized Batch Prediction](#vectorized-batch-prediction)
   - [Input Caching](#input-caching)
   - [Hot Reload](#hot-reload)
   - [Exporting Results](#exporting-results-api)
8. [Standard Model Batch Optimizations](#standard-model-batch-optimizations)
   - [Pre-Sampled Pool (Batch)](#pre-sampled-pool-batch)
   - [Vectorized Batch Scoring (Standard Model)](#vectorized-batch-scoring-standard-model)
   - [Duplicate-Row Cache (Batch)](#duplicate-row-cache-batch)
9. [Multi-Tenant Deployments — Model Registry](#multi-tenant-deployments--model-registry)
   - [The Problem](#the-problem)
   - [Loading the Registry](#loading-the-registry)
   - [Routing Predictions](#routing-predictions)
   - [Cache Isolation](#cache-isolation)
   - [Reloading a Single Config](#reloading-a-single-config)
   - [Adding and Removing Configs at Runtime](#adding-and-removing-configs-at-runtime)
   - [Registry Status and Health Checks](#registry-status-and-health-checks)
   - [API Routing Pattern](#api-routing-pattern)
10. [API Optimizations — Technical Detail](#api-optimizations--technical-detail)
   - [Pre-Sampled Beta Pool](#pre-sampled-beta-pool)
   - [Vectorized Batch Scoring](#vectorized-batch-scoring)
   - [Precision Tiers Detail](#precision-tiers-detail)
   - [Input Caching Detail](#input-caching-detail)
   - [Hot Reload Detail](#hot-reload-detail)
   - [Performance Benchmark](#performance-benchmark)
11. [Output Reference](#output-reference)
    - [Single Prediction Output](#single-prediction-output)
    - [Batch DataFrame Columns](#batch-dataframe-columns)
    - [Excel Workbook Layout](#excel-workbook-layout)
12. [Extending the Model](#extending-the-model)
    - [Adding or Removing Sources](#adding-or-removing-sources)
    - [Tuning Monte Carlo Samples](#tuning-monte-carlo-samples)
    - [Tuning Source Parameters](#tuning-source-parameters)
13. [Config Versioning](#config-versioning)
14. [Interpreting Results](#interpreting-results)
15. [Mathematical Reference](#mathematical-reference)

---

## Overview

This toolkit implements a **Noisy-OR Bayesian Risk Model** in two variants:

**`noisy_or_model.py` — Standard model** designed for server scripts, notebooks, and batch analysis jobs. Single predictions use the original per-call sampling path. Batch predictions automatically use a pre-sampled Beta pool, vectorized NumPy scoring, and a duplicate-row cache — making them suitable for very large datasets without any extra configuration.

**`noisy_or_model_api.py` — API-optimized model** built for deployment behind an API endpoint. It inherits all functionality from the standard model and adds a pre-sampled Beta pool, fully vectorized batch scoring, precision tiers, LRU input caching, zero-downtime hot reload, and a `ModelRegistry` for multi-tenant deployments. Approximately 4× faster on batch workloads in benchmarks.

Both variants share the same output format, the same config file, and the same source management workflow — switching between them requires changing only the import and class name.

The model takes a binary input vector (one flag per source: `1` = matched, `0` = did not match) and returns a probabilistic risk score along with uncertainty bounds, a certainty level, the primary risk driver, and marginal contributions from all other matched sources.

---

## How the Model Works

The model uses **Noisy-OR combination** over a set of independent risk sources. Each source has uncertainty in its own risk probability, represented by a Beta distribution. The simulation proceeds in three steps:

**Step 1 — Sample Beta distributions.** For every source, draw `n` probability samples from a Beta distribution parameterised by `mu` (the prior mean risk) and `kappa` (the concentration, i.e. how confident we are in that mean). Higher `kappa` produces a tighter distribution; lower `kappa` spreads it wider.

```
alpha = mu * kappa
beta  = (1 - mu) * kappa
p_i   ~ Beta(alpha, beta)   [n samples drawn]
```

**Step 2 — Combine active sources via Noisy-OR.** Only sources that matched (`flag = 1`) are combined. The Noisy-OR formula gives the probability that *at least one* source would independently cause the risk event:

```
combined_risk = 1 - ∏(1 - p_i)   for all active sources i
```

**Step 3 — Summarise.** The `n` combined samples yield a distribution of risk scores. Summary statistics (mean, median, P5, P95) are computed from that distribution, and certainty is derived from the width of the 90% interval.

**Marginal contributions** are computed analytically from the source means: each source's contribution is the drop in combined risk if that source were removed.

---

## File Structure

```
noisy_or_config.json         <- Single-tenant config (sources, mu/kappa, version)
configs/
  general.json               <- General config (multi-tenant example)
  customer_1.json            <- Customer-specific config
  customer_2.json            <- Customer-specific config
noisy_or_model.py            <- Standard model (NoisyORModel)
noisy_or_model_api.py        <- API-optimized model (NoisyORModelAPI + ModelRegistry)
noisy_or_demo.py             <- Standard model demonstration (includes batch benchmark)
noisy_or_demo_api.py         <- API model demonstration and benchmark
noisy_or_demo_registry.py    <- ModelRegistry multi-tenant demonstration
```

Output files produced by the demos:
```
noisy_or_results.xlsx       <- Formatted Excel workbook (standard model)
noisy_or_results.csv        <- Flat CSV (standard model)
noisy_or_results_api.xlsx   <- Formatted Excel workbook (API model)
noisy_or_config_v2.json     <- Example of a programmatically generated config
```

---

## Installation

Both models require Python 3.9+ and three packages:

```bash
pip install numpy pandas openpyxl
```

No other dependencies are needed. Neither model requires Flask or any web framework.

---

## Configuration

All model settings live in `noisy_or_config.json`. This is the **only file you need to edit** for routine operation, or you can generate it programmatically using `save_config()` — see [Generating a Config from a Python Dictionary](#generating-a-config-from-a-python-dictionary).

```json
{
  "version": "1.0.0",
  "version_date": "2026-03-09",
  "n_monte_carlo_samples": 20000,
  "sources": [
    {"name": "Source 1", "mu": 0.60, "kappa": 5.0},
    {"name": "Source 2", "mu": 0.45, "kappa": 8.0},
    {"name": "Source 3", "mu": 0.30, "kappa": 10.0},
    {"name": "Source 4", "mu": 0.20, "kappa": 6.0},
    {"name": "Source 5", "mu": 0.15, "kappa": 4.0},
    {"name": "Source 6", "mu": 0.50, "kappa": 7.0},
    {"name": "Source 7", "mu": 0.10, "kappa": 12.0},
    {"name": "Source 8", "mu": 0.35, "kappa": 9.0}
  ]
}
```

### Configuration Fields

| Field | Type | Description |
|---|---|---|
| `version` | string | Semantic version of this config (e.g. `"1.2.0"`). Auto-managed by `save_config()` or set manually. |
| `version_date` | string | ISO date (YYYY-MM-DD) the config was last saved. Auto-set to today by `save_config()`. |
| `n_monte_carlo_samples` | integer | Default Monte Carlo draws for the standard model. Default `20000`. The API model uses precision tiers; see [Precision Tiers Detail](#precision-tiers-detail). |
| `sources` | array | List of source definitions. Order matters — it must match the order of `1`/`0` flags passed to the model. |
| `sources[i].name` | string | Human-readable source name, used in all outputs and Excel column headers. |
| `sources[i].mu` | float (0-1) | Prior mean probability that this source alone would cause the risk event. |
| `sources[i].kappa` | float (> 0) | Concentration parameter. Higher values mean more confidence in `mu`; lower values mean more uncertainty. |

---

## Usage — Standard Model

### Single Prediction

```python
import numpy as np
from noisy_or_model import NoisyORModel

# Load model from config
model = NoisyORModel("noisy_or_config.json")

# Build input array — one entry per source, in config order
# 1 = this source matched the record, 0 = it did not
flags = np.array([1, 0, 1, 0, 0, 1, 0, 0])

# Run prediction
result = model.predict(
    active_flags=flags,
    label="Customer_A",   # optional — used in outputs
    seed=42,              # optional — omit for random results each run
)

# Pretty-print to console
model.print_result(result)

# Access fields directly
print(result["risk_score"])                   # e.g. 0.8606
print(result["certainty_label"])              # "High", "Medium", or "Low"
print(result["primary_driver"])               # e.g. "Source 1"
print(result["primary_driver_contribution"])  # e.g. 0.2106

# Iterate other contributors and their marginal contributions
for match in result["other_matches"]:
    print(match["source"], match["marginal_contribution"], match["source_mean"])
```

**Console output example:**
```
============================================================
  NOISY-OR RISK RESULT  |  Customer_A
============================================================
  Risk Score   : 86.1%
  Median Risk  : 88.0%
  90% Interval : [67.5%,  97.6%]
  Certainty    : 69.8%  (Medium)

  Primary Driver : Source 1
  Primary Contribution : +21.1%

  Other Matched Sources:
    * Source 6              marginal +13.9%
    * Source 3              marginal +6.0%

  Active sources : Source 1, Source 3, Source 6
============================================================
```

The `seed` parameter is optional. When provided, the simulation is fully reproducible. When omitted, a fresh random seed is used on each call.

---

### Batch Prediction

`predict_batch()` applies three optimizations automatically — no extra configuration required:

**Pre-sampled pool** — 50,000 Beta draws per source are computed once at load time and reused across all batch calls. No sampling happens at prediction time.

**Vectorized scoring** — all records are scored simultaneously in a single NumPy pass over a 3-D array. No Python loop over records.

**Duplicate-row cache** — if the same flag combination appears multiple times in a batch (common in large screening datasets where many records have the same single source active), it is scored only once and the result reused. On a 200-record batch with only 5 unique patterns, all 200 rows complete in the time it takes to score 5.

```python
import numpy as np
from noisy_or_model import NoisyORModel

model = NoisyORModel("noisy_or_config.json")

# Each row is one record; columns correspond to sources in config order
inputs = np.array([
    [1, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 1, 0],
    [1, 1, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0],   # no sources — zero risk
    [1, 1, 1, 1, 1, 1, 1, 1],   # all sources — maximum risk
])

labels = ["Customer_A", "Customer_B", "Customer_C", "Customer_D", "Customer_E"]

df = model.predict_batch(active_matrix=inputs, labels=labels, seed=42)

# Summary view
print(df[["Label", "Risk_Score", "Certainty_Label", "Primary_Driver"]])

# Marginal contributions for all active sources
contrib_cols = ["Label"] + [
    col for col in df.columns if col.endswith("_MarginalContrib") and df[col].any()
]
print(df[contrib_cols])
```

The `n_samples` argument controls how many of the 50,000 pre-drawn pool samples are used per record, defaulting to `n_monte_carlo_samples` from the config. The pool is always sized at 50,000, so you can pass up to `n_samples=50000` without rebuilding anything.

---

### Exporting Results

```python
# Export to Excel (.xlsx) or CSV — returns the DataFrame for in-memory use
df = model.export_batch(inputs, "noisy_or_results.xlsx", labels=labels, seed=42)
df = model.export_batch(inputs, "noisy_or_results.csv",  labels=labels, seed=42)
```

The file format is determined automatically from the file extension. The returned DataFrame does not require reloading the file.

---

### Generating a Config from a Python Dictionary

`save_config()` lets you define or update sources entirely in Python — no manual JSON editing required. It writes a new config file with the version auto-bumped and `version_date` set to today's date.

```python
from noisy_or_model import NoisyORModel

model = NoisyORModel("noisy_or_config.json")

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
    bump="minor",        # auto-increment: "major", "minor", or "patch"
    # version="2.0.0",  # or supply an explicit version string
    # n_samples=50000,  # optionally override the Monte Carlo sample count
)
# Config saved -> noisy_or_config_v2.json  (version 1.1.0, 2026-03-10)
```

**`save_config()` parameters:**

| Parameter | Type | Description |
|---|---|---|
| `sources_dict` | dict | Source name to `{"mu": float, "kappa": float}`. Key order determines source order in all outputs. |
| `output_path` | str or Path | Where to write the `.json` file. Can overwrite the loaded config in place. |
| `bump` | str | Which version component to auto-increment: `"major"`, `"minor"` (default), or `"patch"`. |
| `version` | str, optional | Explicit version string. When supplied, `bump` is ignored. |
| `n_samples` | int, optional | Monte Carlo sample count for the new config. Defaults to the currently loaded value. |

---

## Usage — API-Optimized Model

`NoisyORModelAPI` extends `NoisyORModel` with optimizations for high-throughput API deployment. All methods from the standard model (`save_config`, `export_batch`, `print_result`, etc.) are available unchanged.

### Loading the API Model

```python
from noisy_or_model_api import NoisyORModelAPI

# Pool is built automatically at load time
model = NoisyORModelAPI(
    "noisy_or_config.json",
    pool_seed=None,    # optional: fix pool RNG for reproducible pools (testing only)
    cache_size=256,    # LRU cache entries; set to 0 to disable caching
)
# [NoisyORModelAPI] Pool built -- 8 sources x 50,000 samples (1.5 MB)

print(model)
# NoisyORModelAPI(v1.0.0, n_sources=8, pool=50,000, cache=on)
```

---

### Precision Tiers

Instead of specifying raw sample counts, the API model exposes three named tiers that balance speed against result stability:

At load time, 50,000 Beta samples are pre-drawn for each source and stored in the pool. On every prediction call, the tier determines how many of those pre-drawn samples are actually used for the Noisy-OR combination — no new random sampling happens at prediction time.

| Tier | Samples used | Typical Latency | Recommended Use |
|---|---|---|---|
| `"fast"` | 2,000 of 50,000 | ~2-5 ms | Real-time screening, high-throughput pipelines |
| `"standard"` | 10,000 of 50,000 | ~5-20 ms | Default API response quality |
| `"high"` | all 50,000 | ~20-100 ms | Auditable decisions, final risk reports |

More samples used per call means a more stable, less noisy risk estimate — at the cost of slightly more combination math. All three tiers draw from the same pre-built pool, so there is no additional sampling cost for higher tiers.

---

### Single Prediction (API)

```python
import numpy as np
flags = np.array([1, 0, 1, 0, 0, 1, 0, 0])

result = model.predict(
    active_flags=flags,
    precision="standard",   # "fast", "standard", or "high"
    label="Customer_A",     # optional
    # n_samples=20000,      # override tier with an explicit count
    # seed=42,              # optional; disables caching for this call
)

# Output dict is identical to the standard model
model.print_result(result)
```

---

### Vectorized Batch Prediction

```python
import numpy as np

batch = np.array([
    [1, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 1, 0],
    [1, 1, 0, 0, 1, 0, 0, 1],
])
labels = ["Customer_A", "Customer_B", "Customer_C"]

df = model.predict_batch(
    active_matrix=batch,
    precision="standard",
    labels=labels,
)

# DataFrame columns are identical to the standard model
print(df[["Label", "Risk_Score", "Certainty_Label", "Primary_Driver"]])
```

The entire batch is scored in a single NumPy pass — no Python loop over records. See [Vectorized Batch Scoring](#vectorized-batch-scoring) for implementation detail.

---

### Input Caching

Repeated identical inputs are returned from an LRU cache without touching the pool. This is especially effective for risk screening workloads where a small number of source combinations appear frequently across many records.

```python
# First call — computed and stored in cache
result = model.predict(flags, precision="standard")

# Second call with same flags and precision — returned instantly from cache
result = model.predict(flags, precision="standard")

# Inspect cache statistics
print(model.cache_info)
# CacheInfo(hits=1, misses=1, maxsize=256, currsize=1)
```

Passing an explicit `seed=` to `predict()` bypasses the cache for that call, since a seeded call is expected to produce a specific random draw rather than a cached result.

---

### Hot Reload

After saving a new config (via `save_config()` or manual edit), call `reload()` to make the changes live without restarting the server. The pool is rebuilt, the cache is cleared, and all subsequent requests use the new parameters.

```python
# 1. Update sources
model.save_config(updated_sources, "noisy_or_config.json", bump="minor")

# 2. Reload in place — no server restart needed
model.reload()
# [NoisyORModelAPI] Pool built -- 8 sources x 50,000 samples (1.5 MB)
# [NoisyORModelAPI] Reloaded -- NoisyORModelAPI(v1.1.0, ...)

print(model.cache_info)
# CacheInfo(hits=0, misses=0, maxsize=256, currsize=0)  <- cache cleared
```

`reload()` is thread-safe. A lock prevents concurrent requests from seeing a partially rebuilt pool during the reload window.

In a Flask or FastAPI deployment, a lightweight admin endpoint is all that is needed:

```python
# FastAPI example
@app.post("/admin/reload")
def reload_model():
    model.reload()
    return {"version": model.version, "n_sources": model.n_sources}

# Flask example
@app.route("/admin/reload", methods=["POST"])
def reload_model():
    model.reload()
    return {"version": model.version, "n_sources": model.n_sources}
```

---

### Exporting Results (API)

Export methods are inherited from the standard model and work identically:

```python
df = model.export_batch(batch, "noisy_or_results_api.xlsx", labels=labels)
df = model.export_batch(batch, "noisy_or_results_api.csv",  labels=labels)
```

---

## Standard Model Batch Optimizations

`predict_batch()` on `NoisyORModel` applies the same core optimizations as the API model — pre-sampled pool, vectorized NumPy scoring, and deduplication — automatically whenever you call it. No configuration required.

### Pre-Sampled Pool (Batch)

When `NoisyORModel` loads, it draws 50,000 Beta samples per source and stores them as a `(n_sources, 50_000)` float32 matrix — the batch pool. Every call to `predict_batch()` slices columns from this pool rather than sampling fresh Beta draws.

```
Pool matrix:  (n_sources, 50_000) float32  ≈ 1.5 MB for 8 sources
```

Single predictions (`predict()`) are unaffected and continue to draw fresh Beta samples on each call, preserving their existing reproducibility behaviour.

---

### Vectorized Batch Scoring (Standard Model)

The batch is scored in a single NumPy pass — no Python loop over records. A 3-D array of shape `(n_unique_rows, n_sources, n_samples)` is constructed, and `np.prod` computes the Noisy-OR combination across all records simultaneously.

Benchmarked at **~4.5× faster** than the previous per-record loop on 500 records with 20,000 samples.

```
Old: Python loop, fresh Beta sample per record  → ~11 ms/record
New: vectorized pool slice                       →  ~2.5 ms/record
```

---

### Duplicate-Row Cache (Batch)

Before scoring, `predict_batch()` identifies all unique flag combinations in the input matrix. Only the unique rows are passed to the vectorized scorer; results are then mapped back to the original record order. Each unique combination is scored exactly once per batch call regardless of how many times it appears.

This is especially effective for large screening datasets where most records trigger only one or two sources:

```python
# 200 records, only 5 unique flag patterns
# → vectorized scorer runs on a (5, n_sources, n_samples) array
# → all 200 results assembled in microseconds from the 5 sim dicts
df = model.predict_batch(large_matrix)
```

The deduplication is per-call, not persistent between calls. It handles the common real-world case of repeated patterns within a single large batch, without the memory implications of a persistent cross-call cache.

---

## Multi-Tenant Deployments — Model Registry

The `ModelRegistry` class manages multiple `NoisyORModelAPI` instances — one per config — under a single object. It is the recommended pattern when different customers or business units require different source parameters.

### The Problem

The single-model design assumes one config loaded at startup. In a multi-tenant deployment this creates two problems. First, the pre-sampled pool is tied to one set of source parameters — you cannot serve different customers with different `mu`/`kappa` values from the same pool. Second, the LRU cache has no concept of which config produced a result — the same input flags from two different customers could collide in the cache and return the wrong result.

The `ModelRegistry` solves both: each config gets its own pool and its own cache, fully isolated. A cache hit for one customer can never be returned to another.

---

### Loading the Registry

Pass a dictionary of config names to file paths. All pools are built at startup.

```python
from noisy_or_model_api import ModelRegistry

registry = ModelRegistry(
    configs={
        "general":    "configs/general.json",
        "customer_1": "configs/customer_1.json",
        "customer_2": "configs/customer_2.json",
    },
    cache_size=256,   # per-config LRU cache size
)
# [ModelRegistry] Loading 'general' from configs/general.json
# [NoisyORModelAPI] Pool built — 8 sources x 50,000 samples (1.5 MB)
# [ModelRegistry] Loading 'customer_1' from configs/customer_1.json
# [NoisyORModelAPI] Pool built — 8 sources x 50,000 samples (1.5 MB)
# ...
# [ModelRegistry] Ready — 3 config(s) loaded.
```

Memory cost scales linearly — 3 configs × 1.5 MB per pool = ~4.5 MB total, which is negligible.

---

### Routing Predictions

Pass the config name as the first argument to `predict()` or `predict_batch()`. Everything else works identically to the single-model API.

```python
import numpy as np

flags = np.array([1, 0, 1, 0, 0, 1, 0, 0])

# Single prediction routed to a specific config
result = registry.predict("customer_1", flags, precision="standard")

# Batch prediction routed to a specific config
batch = np.array([[1,0,1,0,0,1,0,0], [0,1,0,1,0,0,0,1]])
df = registry.predict_batch("customer_2", batch, precision="standard",
                             labels=["Record_A", "Record_B"])
```

The same input flags produce different scores for different configs because each config has its own `mu` and `kappa` values:

```
flags = [1, 0, 1, 0, 0, 1, 0, 0]  (Watchlist Hit, PEP Match, Transaction Flag)

  general     risk=89.9%  certainty=Medium  driver=Watchlist Hit
  customer_1  risk=97.2%  certainty=High    driver=Watchlist Hit
  customer_2  risk=72.0%  certainty=Low     driver=Watchlist Hit
```

To access any method not exposed directly on the registry (such as `export_batch`, `print_result`, or `save_config`), use `registry.get(config_name)` to retrieve the underlying model instance:

```python
model = registry.get("customer_1")
model.print_result(result)
df = model.export_batch(batch, "customer_1_results.xlsx")
```

---

### Cache Isolation

Each config's LRU cache is completely independent. A cache entry created by a `customer_1` prediction is stored only in `customer_1`'s cache and can never be returned to a `customer_2` request.

```python
# After calling predict() for general and customer_1 twice each:
print(registry.get("general").cache_info)
# CacheInfo(hits=1, misses=1, maxsize=256, currsize=1)

print(registry.get("customer_1").cache_info)
# CacheInfo(hits=1, misses=1, maxsize=256, currsize=1)

print(registry.get("customer_2").cache_info)
# CacheInfo(hits=0, misses=0, maxsize=256, currsize=0)  <- never called, zero entries
```

---

### Reloading a Single Config

`registry.reload(config_name)` rebuilds only that config's pool and clears only that config's cache. Every other config continues serving requests with its existing pool, uninterrupted.

```python
# Update customer_1's parameters
registry.get("customer_1").save_config(updated_sources, "configs/customer_1.json",
                                        bump="minor")

# Reload only customer_1 — general and customer_2 are unaffected
registry.reload("customer_1")

# Verify
print(registry.get("customer_1").version)   # e.g. "1.1.0"  (updated)
print(registry.get("general").version)      # e.g. "1.0.0"  (unchanged)
```

To reload every config at once:

```python
registry.reload_all()
```

---

### Adding and Removing Configs at Runtime

New configs can be added to a running registry without restarting the server. The pool is built immediately on `add()`.

```python
# Add a new customer config
registry.add("customer_3", "configs/customer_3.json")
print(registry.config_names)
# ['general', 'customer_1', 'customer_2', 'customer_3']

# Remove a config and free its pool from memory
registry.remove("customer_3")
print(registry.config_names)
# ['general', 'customer_1', 'customer_2']
```

---

### Registry Status and Health Checks

`registry.status()` returns a dict summarising every loaded config — useful for a `/health` or `/status` API endpoint.

```python
for name, info in registry.status().items():
    print(f"{name}: v{info['version']}  {info['n_sources']} sources  "
          f"pool={info['pool_size']:,}  cache={info['cache_info']}")

# general:    v1.0.0  8 sources  pool=50,000  cache=CacheInfo(hits=1, ...)
# customer_1: v1.1.0  8 sources  pool=50,000  cache=CacheInfo(hits=0, ...)
# customer_2: v1.0.0  8 sources  pool=50,000  cache=CacheInfo(hits=0, ...)
```

---

### API Routing Pattern

A typical deployment maps customer or tenant identifiers to config names and routes each request accordingly.

```python
# FastAPI example

CUSTOMER_CONFIG_MAP = {
    "cust_abc": "customer_1",
    "cust_xyz": "customer_2",
    # All others fall back to the general config
}

@app.post("/score/{customer_id}")
def score_record(customer_id: str, body: ScoreRequest):
    config_name = CUSTOMER_CONFIG_MAP.get(customer_id, "general")
    flags = np.array(body.source_flags)
    result = registry.predict(config_name, flags, precision=body.precision)
    return result

@app.post("/admin/reload/{config_name}")
def reload_config(config_name: str):
    registry.reload(config_name)
    model = registry.get(config_name)
    return {"version": model.version, "n_sources": model.n_sources}

@app.get("/status")
def status():
    return registry.status()
```

---

## API Optimizations — Technical Detail

### Pre-Sampled Beta Pool

**Problem:** In the standard model, Beta distributions are sampled fresh on every prediction call. With 8 sources and 10,000 samples, each call draws 80,000 random numbers before any risk combination logic runs. Under API load this dominates latency and CPU.

**Solution:** When the API model loads, it draws 50,000 Beta samples for every source once and stores them as a matrix. This is the pool. All prediction calls — regardless of tier — work entirely from this pre-drawn data. No random Beta sampling happens at prediction time.

```
Pool matrix shape:  (n_sources, pool_size) = (8, 50_000)
Memory footprint:   8 x 50,000 x 4 bytes (float32) = 1.5 MB
```

Each prediction call randomly selects `n_samples` column indices from the pool (the number is determined by the precision tier) and runs the Noisy-OR combination on that slice. The expensive part — drawing from Beta distributions — has already been done at load time.

The pool is rebuilt from scratch any time `reload()` is called, ensuring updated source parameters take effect immediately.

---

### Vectorized Batch Scoring

**Problem:** The standard model loops over records in Python, running one simulation per record. Python loop overhead is significant at scale.

**Solution:** The API model builds a 3-D array of shape `(n_records, n_sources, n_samples)` and computes the entire Noisy-OR combination in a single `np.prod` call over the source axis. All NumPy operations are pushed to compiled C/Fortran code with no Python loop.

```
Array shape:   (n_records, n_sources, n_samples)
               e.g. (100, 8, 10_000) = 8M float32 values = ~30 MB peak

Noisy-OR step: combined = 1 - np.prod(1 - p3d, axis=1)
               result shape: (n_records, n_samples)

Stats step:    np.mean / np.percentile over axis=1 simultaneously for all records
```

Marginal contributions are also computed in a single vectorized pass using broadcasting, with no record-level loop.

---

### Precision Tiers Detail

The `n_monte_carlo_samples` config value still sets the default for the standard model. For the API model, tiers control how many of the 50,000 pre-drawn pool samples are used per call:

```python
PRECISION_SAMPLES = {
    "fast":     2_000,   # use 2,000 of the 50,000 pool samples
    "standard": 10_000,  # use 10,000 of the 50,000 pool samples
    "high":     50_000,  # use all 50,000 pool samples
}
```

Since all tiers draw from the same pool, there is no extra sampling cost for higher tiers — only slightly more combination math. You can still pass an explicit `n_samples=` argument on any call to override the tier.

---

### Input Caching Detail

Single prediction results are stored in a `functools.lru_cache` keyed on `(flags_tuple, n_samples)`. The cache is most effective for sparse input spaces — screening workflows where most records trigger only one or two sources.

The cache is automatically cleared when `reload()` is called, preventing stale results from persisting after a config update. Cache size defaults to 256 entries and is configurable at init time:

```python
model = NoisyORModelAPI("noisy_or_config.json", cache_size=512)  # larger cache
model = NoisyORModelAPI("noisy_or_config.json", cache_size=0)    # disable caching
```

Caching is bypassed per-call when an explicit `seed=` is passed, since a seeded call is expected to produce a specific random draw rather than a reusable cached result.

---

### Hot Reload Detail

`reload()` performs the following steps under a thread lock:

1. Re-reads the config JSON from disk
2. Updates all source parameters, version, and sample count in memory
3. Rebuilds the Beta sample pool from scratch using the new parameters
4. Replaces the LRU cache with a fresh empty instance

The lock ensures that any in-flight requests complete against the old pool before the swap occurs. There is a brief window during pool construction where new requests will block, but no request will ever see a partially built pool.

---

### Performance Benchmark

Measured on 200 records with 10,000 samples (standard precision), 8 sources:

| Method | Total Time | Per Record |
|---|---|---|
| Standard model — Python loop | ~2,500 ms | ~12.6 ms |
| API model — vectorized batch | ~640 ms | ~3.2 ms |
| **Speedup** | **~4x** | **~4x** |

Single-prediction latency with caching:

| Call | Latency |
|---|---|
| First call (computed) | ~0.05 ms |
| Subsequent identical call (cached) | ~0.02 ms |

The speedup on batch operations comes entirely from eliminating the Python loop and pushing all work into a single NumPy pass. Absolute times will vary by hardware, but the relative improvement is consistent.

---

## Output Reference

### Single Prediction Output

`model.predict()` returns a Python dictionary with the following keys. The structure is identical for both `NoisyORModel` and `NoisyORModelAPI`.

| Key | Type | Description |
|---|---|---|
| `label` | str or None | Label passed in, if any. |
| `risk_score` | float | Mean combined risk across all Monte Carlo samples (0-1). |
| `median_risk` | float | Median combined risk. |
| `p5` | float | 5th percentile of the risk distribution. |
| `p95` | float | 95th percentile of the risk distribution. |
| `certainty` | float | `1 - (p95 - p5)`. Width of the 90% interval subtracted from 1. Higher is more certain. |
| `certainty_label` | str | `"High"` (>= 0.80), `"Medium"` (>= 0.50), or `"Low"` (< 0.50). |
| `primary_driver` | str or None | Name of the active source with the largest marginal contribution. |
| `primary_driver_contribution` | float or None | Marginal contribution of the primary driver. |
| `other_matches` | list of dicts | All other active sources, sorted by marginal contribution descending. Each dict contains `source`, `marginal_contribution`, and `source_mean`. |
| `source_means` | list of floats | Mean sampled probability for every source (active or not), in config order. |
| `active_sources` | list of str | Names of all sources that were flagged active (`1`) for this record. |

---

### Batch DataFrame Columns

`model.predict_batch()` returns a pandas DataFrame with one row per record. Column layout is identical for both model variants.

**Summary columns** (always present):

| Column | Description |
|---|---|
| `Label` | Record label. |
| `Risk_Score` | Mean risk score (0-1). |
| `Median_Risk` | Median risk score. |
| `P5` | 5th percentile. |
| `P95` | 95th percentile. |
| `Certainty` | Certainty score (0-1). |
| `Certainty_Label` | `"High"`, `"Medium"`, or `"Low"`. |
| `Primary_Driver` | Name of the top contributing source. |
| `Primary_Driver_Contribution` | Marginal contribution of the primary driver. |
| `Active_Sources` | Comma-separated list of all matched source names. |

**Per-source columns** (two columns per source, named from the source's `name` field in config):

| Column pattern | Description |
|---|---|
| `<SourceName>_Active` | `1` if this source was flagged active, `0` otherwise. |
| `<SourceName>_MarginalContrib` | Marginal contribution of this source. `0.0` for inactive sources. |

For example, with a source named `"Watchlist Hit"`, the columns would be `Watchlist_Hit_Active` and `Watchlist_Hit_MarginalContrib`.

---

### Excel Workbook Layout

The `.xlsx` export produces a two-sheet workbook:

**Sheet 1 — Risk Results**
- All columns described above, formatted as percentages where appropriate.
- Row 1: workbook title.
- Row 2: frozen header row with auto-filter enabled.
- Data rows: alternating white/light-blue fill for readability. The `Certainty` and `Certainty Level` columns are shaded green (High), yellow (Medium), or red (Low) to allow quick visual triage.

**Sheet 2 — Model Config**
- A snapshot of the source configuration (name, mu, kappa), Monte Carlo sample count, version, and version date at the time the file was generated. Useful for audit purposes and for matching any output file back to the exact model parameters that produced it.

---

## Extending the Model

### Adding or Removing Sources

Open `noisy_or_config.json` and add or remove entries from the `"sources"` array, or use `save_config()` to generate the updated file from a Python dictionary. No other code changes are required.

After changing the config, input arrays passed to `predict()` and `predict_batch()` must have one entry per source in the same order as they appear in the config. A model loaded with 9 sources will reject arrays of length 8 with a clear error message.

For the API model, call `reload()` after saving the new config to make the changes live without restarting the server.

**Adding a source manually:**
```json
{
  "version": "1.1.0",
  "version_date": "2026-03-10",
  "n_monte_carlo_samples": 20000,
  "sources": [
    {"name": "Source 1",       "mu": 0.60, "kappa": 5.0},
    {"name": "Source 2",       "mu": 0.45, "kappa": 8.0},
    {"name": "Adverse Media",  "mu": 0.25, "kappa": 6.0}
  ]
}
```

---

### Tuning Monte Carlo Samples

The `n_monte_carlo_samples` setting controls the default sample count for the standard model. For the API model, precision tiers take precedence; see [Precision Tiers Detail](#precision-tiers-detail).

| Sample Count | Typical Use Case |
|---|---|
| 2,000 - 5,000 | Fast batch screening |
| 10,000 | Balanced API default (`"standard"` tier) |
| 20,000 | Standard model default |
| 50,000 | High-precision decisions; maximum pool size for batch |
| 100,000+ | Research / validation (single predictions only — exceeds pool size) |

The batch pool in the standard model is always built at 50,000 samples. Passing `n_samples` greater than 50,000 to `predict_batch()` is silently capped at 50,000. Single predictions (`predict()`) are not pool-backed and can use any sample count.

To override for a single call:
```python
# Standard model — single prediction (any sample count)
result = model.predict(flags, n_samples=100000)

# Standard model — batch (capped at pool size of 50,000)
df = model.predict_batch(matrix, n_samples=50000)

# API model — use precision tier or explicit override
result = model.predict(flags, precision="high")
result = model.predict(flags, n_samples=50000)
```

---

### Tuning Source Parameters

Each source has two parameters that control the shape of its Beta distribution:

**`mu` — Prior mean (0 to 1)**
The expected probability that this source, alone, would be associated with the risk event. A source with `mu = 0.60` is treated as carrying 60% prior risk on average when it fires.

- Raise `mu` if the source is a strong indicator of risk.
- Lower `mu` if the source is a weak or noisy signal.

**`kappa` — Concentration (> 0)**
How confident the model is in the `mu` value. Think of it as effective sample size.

- Higher `kappa` (e.g. 20-50): the Beta distribution is tight and the source behaves close to its `mu` value on every draw. Use this when the source's base rate is well-established.
- Lower `kappa` (e.g. 2-5): the distribution is wide and diffuse, reflecting genuine uncertainty about the source's true risk rate. Use this for newer or noisier sources.

A practical way to think about the combination: if `mu = 0.30` and `kappa = 10`, you are expressing roughly the same confidence as having observed 3 events in 10 trials. If `kappa = 100`, it is equivalent to 30 events in 100 trials — much more certain.

---

## Config Versioning

Every config file carries a `version` and `version_date` field so that model outputs can always be traced back to the exact parameter set that produced them.

### Version Format

Versions follow **semantic versioning** (`MAJOR.MINOR.PATCH`):

| Component | When to increment |
|---|---|
| `MAJOR` | Breaking change — sources removed, renamed, or reordered in a way that makes old input arrays incompatible |
| `MINOR` | Additive change — new sources added, or significant parameter recalibration |
| `PATCH` | Small adjustment — minor parameter tuning, no structural change |

### Accessing version info at runtime

```python
print(model.version)       # e.g. "1.1.0"
print(model.version_date)  # e.g. "2026-03-10"
print(repr(model))
# NoisyORModel:    NoisyORModel(v1.1.0, n_sources=8, n_samples=20000)
# NoisyORModelAPI: NoisyORModelAPI(v1.1.0, n_sources=8, pool=50,000, cache=on)
```

### Auto-bumping vs. explicit versioning

```python
# Auto-bump the minor version (1.0.0 -> 1.1.0)
model.save_config(sources_dict=my_sources, output_path="config.json", bump="minor")

# Set an explicit version (useful for planned releases)
model.save_config(sources_dict=my_sources, output_path="config.json", version="2.0.0")
```

`version_date` is always set to today's date automatically.

### Version in Excel exports

The **Model Config** sheet of every `.xlsx` export records the version and date alongside the source parameters, providing a built-in audit trail.

---

## Interpreting Results

**Risk Score** is the mean of the simulated combined risk distribution. This is the headline number to use for ranking or thresholding records.

**P5 / P95** bound the 90% credible interval. A narrow interval means the model is confident in its estimate. A wide interval means there is substantial uncertainty, often because individual source probabilities are themselves uncertain (low `kappa`).

**Certainty** is `1 - (P95 - P5)`. It answers: "how wide is the uncertainty band?" — expressed so that higher is better.

| Certainty Level | Value Range | Interpretation |
|---|---|---|
| High | >= 0.80 | Narrow credible interval; model is confident in the risk estimate. |
| Medium | 0.50 - 0.79 | Moderate uncertainty; the interval spans roughly 20-50 percentage points. |
| Low | < 0.50 | Wide credible interval; the point estimate should be treated cautiously. |

**Primary Driver** is the active source whose removal would produce the largest drop in combined risk. It is not simply the source with the highest `mu` — it is the source contributing the most marginal risk *given* all the other sources that also fired, accounting for the Noisy-OR overlap between them.

**Marginal Contribution** values across all active sources will generally not sum to the total risk score. This is expected: Noisy-OR contributions are not additive. Each contribution represents the isolated impact of removing one source while keeping all others active.

---

## Mathematical Reference

**Beta distribution parameterisation:**
```
alpha = max(mu * kappa, 0.001)
beta  = max((1 - mu) * kappa, 0.001)
p_i   ~ Beta(alpha, beta)
```
The `max(..., 0.001)` floor prevents degenerate distributions at the boundaries.

**Noisy-OR combination (Monte Carlo):**
```
combined = 1 - prod(1 - p_i)   for all active sources i
```
Applied element-wise across all `n` simulation samples.

**Summary statistics:**
```
risk_score   = mean(combined)
median_risk  = median(combined)
p5           = percentile(combined, 5)
p95          = percentile(combined, 95)
certainty    = max(0, 1 - (p95 - p5))
```

**Marginal contribution of source i (analytical):**
```
prod_all      = prod(1 - mu_j)   for all active j
overall_risk  = 1 - prod_all
prod_without  = prod_all / (1 - mu_i)
risk_without  = 1 - prod_without
contribution  = overall_risk - risk_without
```
Source means (`mu_j`) are taken as the mean of each source's sampled Beta distribution, not the configured `mu` directly — this ensures the marginal calculation is consistent with the actual simulation draw.

**Vectorized batch Noisy-OR (API model):**
```
Pool slice:  P = pool[:, random_idx]          shape (n_sources, n_samples)
Broadcast:   P3D = P * active_matrix.T        shape (n_records, n_sources, n_samples)
Combined:    C = 1 - prod(1 - P3D, axis=1)   shape (n_records, n_samples)
Stats:       mean/percentile over sample axis simultaneously for all records
```
