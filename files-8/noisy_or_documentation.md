# Noisy-OR Bayesian Risk Model
## Technical Documentation

---

## Table of Contents

1. [Overview](#overview)
2. [How the Model Works](#how-the-model-works)
3. [File Structure](#file-structure)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Usage](#usage)
   - [Single Prediction](#single-prediction)
   - [Batch Prediction](#batch-prediction)
   - [Exporting Results](#exporting-results)
   - [Generating a Config from a Python Dictionary](#generating-a-config-from-a-python-dictionary)
7. [Output Reference](#output-reference)
   - [Single Prediction Output](#single-prediction-output)
   - [Batch DataFrame Columns](#batch-dataframe-columns)
   - [Excel Workbook Layout](#excel-workbook-layout)
8. [Extending the Model](#extending-the-model)
   - [Adding or Removing Sources](#adding-or-removing-sources)
   - [Tuning Monte Carlo Samples](#tuning-monte-carlo-samples)
   - [Tuning Source Parameters](#tuning-source-parameters)
9. [Config Versioning](#config-versioning)
10. [Interpreting Results](#interpreting-results)
11. [Mathematical Reference](#mathematical-reference)

---

## Overview

This toolkit implements a **Noisy-OR Bayesian Risk Model** designed for server or notebook execution. It is the programmatic counterpart to an interactive Flask web application, sharing identical underlying mathematics while adding support for:

- Single record predictions with structured output
- Batch predictions over many records at once
- Human-readable Excel and CSV exports
- Easy configuration of sources, parameters, and simulation settings — all without changing any Python code

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
noisy_or_config.json    ← All configuration: sources, mu/kappa values, simulation size
noisy_or_model.py       ← Core model library (NoisyORModel class)
noisy_or_demo.py        ← Runnable demonstration of all features
```

The demo produces two output files:
```
noisy_or_results.xlsx   ← Formatted Excel workbook (two sheets)
noisy_or_results.csv    ← Flat CSV of the same batch results
```

---

## Installation

The model requires Python 3.9+ and three packages:

```bash
pip install numpy pandas openpyxl
```

No other dependencies are needed. The model does not require Flask or any web framework.

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
| `n_monte_carlo_samples` | integer | Number of Monte Carlo draws per prediction. Default `20000`. See [Tuning Monte Carlo Samples](#tuning-monte-carlo-samples). |
| `sources` | array | List of source definitions. Order matters — it must match the order of `1`/`0` flags passed to the model. |
| `sources[i].name` | string | Human-readable source name, used in all outputs and Excel column headers. |
| `sources[i].mu` | float (0–1) | Prior mean probability that this source alone would cause the risk event. |
| `sources[i].kappa` | float (> 0) | Concentration parameter. Higher values mean more confidence in `mu`; lower values mean more uncertainty. |

---

## Usage

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

# Or access fields directly
print(result["risk_score"])            # e.g. 0.8606
print(result["certainty_label"])       # "High", "Medium", or "Low"
print(result["primary_driver"])        # e.g. "Source 1"
print(result["other_matches"])         # list of dicts for remaining matched sources
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
    • Source 6              marginal +13.9%
    • Source 3              marginal +6.0%

  Active sources : Source 1, Source 3, Source 6
============================================================
```

The `seed` parameter is optional. When provided, the simulation is fully reproducible. When omitted, a fresh random seed is used on each call, which is appropriate for production use where you want natural Monte Carlo variation.

---

### Batch Prediction

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

df = model.predict_batch(
    active_matrix=inputs,
    labels=labels,          # optional; defaults to "Record_1", "Record_2", ...
    seed=42,                # optional
)

# df is a standard pandas DataFrame — use it like any other
print(df[["Label", "Risk_Score", "Certainty_Label", "Primary_Driver"]])
```

---

### Exporting Results

`export_batch` runs the batch prediction and saves the output in a single call, returning the DataFrame so you can continue working with it in memory.

**Export to Excel:**
```python
df = model.export_batch(
    active_matrix=inputs,
    output_path="noisy_or_results.xlsx",
    labels=labels,
    seed=42,
)
```

**Export to CSV:**
```python
df = model.export_batch(
    active_matrix=inputs,
    output_path="noisy_or_results.csv",
    labels=labels,
    seed=42,
)
```

The file format is determined automatically from the file extension. Both `.xlsx` and `.csv` are supported. The returned DataFrame is identical in both cases and does not require reloading the file.

---

### Generating a Config from a Python Dictionary

`save_config()` lets you define or update sources entirely in Python — no manual JSON editing required. It writes a new config file with the version auto-bumped and `version_date` set to today's date.

```python
from noisy_or_model import NoisyORModel

model = NoisyORModel("noisy_or_config.json")

# Define updated sources as a plain dictionary:
#   key   = source name (becomes column headers in all outputs)
#   value = {"mu": <prior mean 0–1>, "kappa": <concentration > 0>}
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
    bump="minor",        # auto-increment "major", "minor", or "patch"
    # version="2.0.0",  # or supply an explicit version string instead
    # n_samples=50000,  # optionally override the Monte Carlo sample count
)
# → Config saved → noisy_or_config_v2.json  (version 1.1.0, 2026-03-10)

# Load the new config immediately
model_v2 = NoisyORModel("noisy_or_config_v2.json")
print(model_v2)
# → NoisyORModel(v1.1.0, n_sources=8, n_samples=20000)
```

**`save_config()` parameters:**

| Parameter | Type | Description |
|---|---|---|
| `sources_dict` | dict | Source definitions — see format above. Key order determines source order in all outputs. |
| `output_path` | str or Path | Where to write the new `.json` file. Can be the same path as the loaded config to overwrite in place. |
| `bump` | str | Which version component to auto-increment when `version` is not supplied: `"major"`, `"minor"` (default), or `"patch"`. |
| `version` | str, optional | Explicit version string (e.g. `"2.0.0"`). When supplied, `bump` is ignored. |
| `n_samples` | int, optional | Monte Carlo sample count for the new config. Defaults to the currently loaded value. |

The generated file always includes `version`, `version_date` (today's date in ISO format), `n_monte_carlo_samples`, and the `sources` array — making it immediately loadable by `NoisyORModel`.

---

## Output Reference

### Single Prediction Output

`model.predict()` returns a Python dictionary with the following keys:

| Key | Type | Description |
|---|---|---|
| `label` | str or None | Label passed in, if any. |
| `risk_score` | float | Mean combined risk across all Monte Carlo samples (0–1). |
| `median_risk` | float | Median combined risk. |
| `p5` | float | 5th percentile of the risk distribution. |
| `p95` | float | 95th percentile of the risk distribution. |
| `certainty` | float | `1 - (p95 - p5)`. Width of the 90% interval subtracted from 1. Higher is more certain. |
| `certainty_label` | str | `"High"` (≥ 0.80), `"Medium"` (≥ 0.50), or `"Low"` (< 0.50). |
| `primary_driver` | str or None | Name of the active source with the largest marginal contribution. |
| `primary_driver_contribution` | float or None | Marginal contribution of the primary driver (percentage point increase in combined risk attributable to that source). |
| `other_matches` | list of dicts | All other active sources, sorted by marginal contribution descending. Each dict contains `source`, `marginal_contribution`, and `source_mean`. |
| `source_means` | list of floats | Mean sampled probability for every source (active or not), in config order. |
| `active_sources` | list of str | Names of all sources that were flagged active (`1`) for this record. |

---

### Batch DataFrame Columns

`model.predict_batch()` returns a pandas DataFrame with one row per record. The columns are:

**Summary columns** (always present):

| Column | Description |
|---|---|
| `Label` | Record label. |
| `Risk_Score` | Mean risk score (0–1). |
| `Median_Risk` | Median risk score. |
| `P5` | 5th percentile. |
| `P95` | 95th percentile. |
| `Certainty` | Certainty score (0–1). |
| `Certainty_Label` | `"High"`, `"Medium"`, or `"Low"`. |
| `Primary_Driver` | Name of the top contributing source. |
| `Primary_Driver_Contribution` | Marginal contribution of the primary driver. |
| `Active_Sources` | Comma-separated list of all matched source names. |

**Per-source columns** (two columns per source, named from the source's `name` field in config):

| Column pattern | Description |
|---|---|
| `<SourceName>_Active` | `1` if this source was flagged active for this record, `0` otherwise. |
| `<SourceName>_MarginalContrib` | Marginal contribution of this source to combined risk. `0.0` for inactive sources. |

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
- A snapshot of the source configuration (name, mu, kappa) and Monte Carlo sample count at the time the file was generated. Useful for audit purposes and for understanding which model version produced a given output.

---

## Extending the Model

### Adding or Removing Sources

Open `noisy_or_config.json` and add or remove entries from the `"sources"` array, or use `save_config()` to generate the updated file from a Python dictionary. No other code changes are required.

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

After editing the config, input arrays passed to `predict()` and `predict_batch()` must have one entry per source in the same order as they appear in the config. A model loaded with 9 sources will reject arrays of length 8 with a clear error message.

**Removing a source:** Delete its entry from the `"sources"` array and update any input arrays accordingly.

---

### Tuning Monte Carlo Samples

The `n_monte_carlo_samples` setting controls the number of simulation draws. This is a speed/accuracy trade-off:

| Sample Count | Typical Use Case |
|---|---|
| 5,000 – 10,000 | Fast exploration, prototyping |
| 20,000 | Default — good balance of speed and stability |
| 50,000 – 100,000 | High-precision runs, final reports |
| 200,000+ | Research / validation |

To change the default globally, edit `noisy_or_config.json`:
```json
{"n_monte_carlo_samples": 50000, "sources": [...]}
```

To override for a single call without changing the config:
```python
result = model.predict(flags, n_samples=50000)
df     = model.predict_batch(matrix, n_samples=50000)
df     = model.export_batch(matrix, "out.xlsx", n_samples=50000)
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

- Higher `kappa` (e.g. 20–50): the Beta distribution is tight and the source behaves close to its `mu` value on every draw. Use this when the source's base rate is well-established.
- Lower `kappa` (e.g. 2–5): the distribution is wide and diffuse, reflecting genuine uncertainty about the source's true risk rate. Use this for newer or noisier sources.

A practical way to think about the combination: if `mu = 0.30` and `kappa = 10`, you are expressing roughly the same confidence as having observed 3 events in 10 trials. If `kappa = 100`, it is equivalent to 30 events in 100 trials — much more certain.

---

## Config Versioning

Every config file carries a `version` and `version_date` field so that model outputs can always be traced back to the exact parameter set that produced them.

### Version format

Versions follow **semantic versioning** (`MAJOR.MINOR.PATCH`):

| Component | When to increment |
|---|---|
| `MAJOR` | Breaking change — sources removed, renamed, or reordered in a way that makes old input arrays incompatible |
| `MINOR` | Additive change — new sources added, or significant parameter recalibration |
| `PATCH` | Small adjustment — minor parameter tuning, no structural change |

### Accessing version info at runtime

```python
model = NoisyORModel("noisy_or_config.json")

print(model.version)       # e.g. "1.1.0"
print(model.version_date)  # e.g. "2026-03-10"
print(repr(model))         # NoisyORModel(v1.1.0, n_sources=8, n_samples=20000)
```

### Auto-bumping vs. explicit versioning

`save_config()` offers two approaches:

```python
# Auto-bump the minor version (1.0.0 → 1.1.0)
model.save_config(sources_dict=my_sources, output_path="config.json", bump="minor")

# Set an explicit version (useful for planned releases)
model.save_config(sources_dict=my_sources, output_path="config.json", version="2.0.0")
```

In both cases, `version_date` is always set to today's date automatically.

### Version in Excel exports

The **Model Config** sheet of every `.xlsx` export records the version and date alongside the source parameters. This provides a built-in audit trail: any output file can be matched back to the config that produced it.

---

## Interpreting Results

**Risk Score** is the mean of the simulated combined risk distribution. This is the headline number to use for ranking or thresholding records.

**P5 / P95** bound the 90% credible interval. A narrow interval (P5 and P95 close together) means the model is confident in its estimate. A wide interval means there is substantial uncertainty, often because individual source probabilities are themselves uncertain (low `kappa`).

**Certainty** is `1 - (P95 - P5)`. It answers: "how wide is the uncertainty band?" — expressed so that higher is better.

| Certainty Level | Value Range | Interpretation |
|---|---|---|
| High | ≥ 0.80 | Narrow credible interval; model is confident in the risk estimate. |
| Medium | 0.50 – 0.79 | Moderate uncertainty; the interval spans roughly 20–50 percentage points. |
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
combined = 1 - ∏ (1 - p_i)   for all active sources i
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
prod_all      = ∏ (1 - mu_j)   for all active j
overall_risk  = 1 - prod_all
prod_without  = prod_all / (1 - mu_i)
risk_without  = 1 - prod_without
contribution  = overall_risk - risk_without
```
Source means (`mu_j`) are taken as the mean of each source's sampled Beta distribution, not the configured `mu` directly — this ensures the marginal calculation is consistent with the actual simulation draw.
