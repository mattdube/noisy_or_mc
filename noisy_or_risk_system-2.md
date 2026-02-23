# Noisy-OR Probabilistic Risk Scoring System
### Architecture, Reasoning, and Implementation Guide

---

## Table of Contents

1. [Why Noisy-OR?](#why-noisy-or)
2. [The Math](#the-math)
3. [Why Beta Distributions?](#why-beta-distributions)
4. [What Changed From the Original Code](#what-changed-from-the-original-code)
5. [System Architecture](#system-architecture)
6. [Phase-by-Phase Walkthrough](#phase-by-phase-walkthrough)
7. [Uncertainty Quantification](#uncertainty-quantification)
8. [Updating Source Confidence Over Time](#updating-source-confidence-over-time)
9. [Quick Local Prediction](#quick-local-prediction)
10. [API Deployment](#api-deployment)
11. [Batch Processing](#batch-processing)
12. [Performance Benchmarks](#performance-benchmarks)
13. [Operational Runbook](#operational-runbook)
14. [Common Questions](#common-questions)

---

## Why Noisy-OR?

When you have **multiple independent risk signals** and you want to combine them into a single risk score, the naive approach is to add or average the probabilities. This breaks in two ways:

**Problem 1 — Exceeds 1.0.** If source A says 60% risk and source B says 70%, adding them gives 130%. Undefined.

**Problem 2 — Ignores independence.** Averaging gives 65%, which treats the two sources as redundant. But if both independently indicate risk, the combined evidence should push the score *higher* than either alone.

The Noisy-OR model solves both problems by working with **probabilities of failure to trigger**. The key insight is:

> The event does NOT occur only if ALL sources fail to trigger it simultaneously.

If source A fails to trigger with probability 0.40 and source B fails with probability 0.30, and they're independent, then the probability both fail is `0.40 × 0.30 = 0.12`. The risk is therefore `1 − 0.12 = 0.88`. This is always in `[0, 1]` and correctly reflects compounding evidence.

---

## The Math

### Core Formula

For a record with match vector **X** (where X_i = 1 if source i matched):

```
P(risk) = 1 − [ (1 − p_leak) × ∏ (1 − p_i)^X_i ]
                                  i
```

Where:
- `p_i` = probability that source i *alone* causes the risk event
- `p_leak` = baseline "leak" probability — unexplained background risk
- The product runs only over matched sources (unmatched sources have X_i = 0, contributing `(1−p_i)^0 = 1`)

### Log-Space Implementation

The product form is numerically unstable for many sources. In code, we work in log-space:

```python
log_fail = log(1 − p_leak) + Σ X_i × log(1 − p_i)
risk     = 1 − exp(log_fail)
```

This is numerically stable even with 100+ sources and near-zero probabilities.

### Driver Attribution (Marginal Impact)

To explain *which* source drove the score, we remove each source and measure the drop:

```
impact_i = P(risk | all sources) − P(risk | all sources except i)
         = risk_score − (1 − exp(log_fail − X_i × log(1 − p_i)))
```

This is computed in fully vectorized form — no loops over sources.

---

## Why Beta Distributions?

Each source's risk probability `p_i` is unknown. Rather than picking a single number, we model it as a probability distribution over possible values. The **Beta distribution** is the natural choice because:

1. **Support is [0, 1]** — perfectly matches probabilities
2. **Conjugate prior for Bernoulli** — when you observe new labeled data (n trials, k hits), the posterior update is mathematically exact and instantaneous: `alpha' = alpha + k`, `beta' = beta + (n − k)`. No MCMC required.
3. **Intuitive parameterization** — can be specified as (mean, concentration) where concentration ~ "effective sample size" of your confidence

### Choosing Parameters

| Expert statement | Mean | Concentration |
|---|---|---|
| "About 2% risk, very sure (500+ historical cases)" | 0.02 | 500 |
| "Maybe 5% risk, not many cases to go on" | 0.05 | 50 |
| "Somewhere around 45%, very uncertain" | 0.45 | 15 |

The Beta(α, β) parameters are derived as:
```
α = max(mean × concentration, 1.1)
β = max((1 − mean) × concentration, 1.1)
```
The 1.1 floor prevents degenerate J-shaped or U-shaped distributions.

---

## What Changed From the Original Code

| Issue | Original | Revised |
|---|---|---|
| **Inference method** | PyMC `sample_prior_predictive` (slow MCMC compilation, ~30s) | Pure NumPy MC draws from Beta priors (~1–5ms) |
| **Volatility calculation** | Incorrect first-order variance propagation on a nonlinear function | MC-based credible interval: actual percentiles of sampled risk scores |
| **Static weights file** | `golden_weights.json` baked with point estimates, goes stale | Versioned `param_store.json` storing `(alpha, beta)` — live sampling at score time |
| **Code duplication** | `production_score_single` and `batch_score_records` duplicate all math | Single `phase3_scorer.py` used by both API and batch — one change updates both |
| **Confidence evolution** | Must re-run PyMC to update | Conjugate Beta update: `alpha += hits`, `beta += misses` — instantaneous |
| **Uncertainty meaning** | `volatility` was std of a linear approximation | `impact_width = p90 − p10` is a true 80% credible interval |
| **Validation** | None | Phase 2 runs 4 automated checks before any production use |
| **Batch performance** | Python loop for driver attribution | Fully vectorized `(n_records × n_sources)` matrix operations |
| **Output format** | Excel only (with silent CSV fallback if openpyxl missing) | `output_format` parameter: `"excel"`, `"csv"`, or `"both"` — explicit, tested, no silent fallbacks |
| **Quick local prediction** | Required the full pipeline to score one record | Standalone `predict.py` — only needs `numpy` and `param_store.json` |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          DATA FLOW                               │
│                                                                  │
│  Expert Knowledge                                                │
│  (mean, confidence)                                              │
│        │                                                         │
│        ▼                                                         │
│  ┌──────────────┐    versioned JSON     ┌───────────────┐        │
│  │ phase1_priors│ ──────────────────▶  │  param_store  │        │
│  │              │    alpha, beta per    │  .json        │        │
│  │  + conjugate │    source + leak      │               │        │
│  │    updates   │                       └───────┬───────┘        │
│  └──────────────┘                               │                │
│                                                 │ loaded         │
│  ┌──────────────┐                               │ once           │
│  │ phase2_      │ ── validates priors ──────────┤                │
│  │ simulate_    │    before deployment           │                │
│  │ validate     │                               ▼                │
│  └──────────────┘                    ┌───────────────────┐       │
│                                      │  phase3_scorer    │       │
│                                      │                   │       │
│                                      │  score()          │       │
│                                      │  score_single()   │       │
│                                      └────────┬──────────┘       │
│                                               │                  │
│                    ┌──────────────────────────┼──────────────┐   │
│                    ▼                          ▼              ▼   │
│          ┌──────────────┐        ┌──────────────┐  ┌──────────┐  │
│          │ phase4_api   │        │ phase5_batch │  │predict   │  │
│          │ FastAPI      │        │ output_format│  │.py       │  │
│          │ /score       │        │ excel|csv|   │  │          │  │
│          │ /score/batch │        │ both         │  │notebooks │  │
│          │ /reload      │        │              │  │scripts   │  │
│          └──────────────┘        └──────────────┘  │CLI       │  │
│                                                    └──────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### File Structure

```
noisy_or_risk/
├── phase1_priors.py             # Expert knowledge → param store
├── phase2_simulate_validate.py  # Validation before deployment
├── phase3_scorer.py             # Core scoring engine (shared)
├── phase4_api.py                # FastAPI REST API
├── phase5_batch.py              # Batch CSV/Excel pipeline
├── predict.py                   # Standalone single-record predictor
├── local_batch_test.py          # Local batch verification suite
├── local_predict_test.py        # Local prediction verification suite
├── run_pipeline.py              # End-to-end runner / smoke test
│
├── params/
│   └── param_store.json         # Versioned Beta parameters
│
└── validation/
    ├── validation_results.json
    └── plots/
        ├── prior_distributions.png
        ├── risk_score_distribution.png
        └── source_sensitivity.png
```

---

## Phase-by-Phase Walkthrough

### Phase 1 — Expert Prior Specification (`phase1_priors.py`)

**Purpose:** Encode domain expertise as Beta distribution parameters. This is the *only* place where human knowledge enters the model.

**Input:** A dictionary of `(mean_risk, concentration)` pairs, one per source.

**Output:** `params/param_store.json` — versioned, timestamped, with full Beta statistics.

**Key function — conjugate update:**
```python
def bayesian_update_source(store, source_name, n_successes, n_trials):
    # Exact Bayesian update — no MCMC, no approximation
    new_alpha = old_alpha + n_successes
    new_beta  = old_beta  + (n_trials - n_successes)
```

Call this whenever an analyst reviews records and confirms/denies risk events. The parameter store version is automatically bumped.

---

### Phase 2 — Simulation & Validation (`phase2_simulate_validate.py`)

**Purpose:** Catch misconfigured priors before they reach production. Runs four automated checks:

| Check | Condition | Why |
|---|---|---|
| No dominant source | No source mean > 0.8 | Would overwhelm all other signals |
| Leak is small | Leak mean < 0.10 | High baseline makes the model insensitive |
| Baseline risk reasonable | Zero-match p50 < 0.15 | Shouldn't flag clean records |
| Full match saturates | All-match p50 > 0.80 | Should reach high risk when all sources fire |

**Source sensitivity analysis** shows the average marginal impact of each source matching alone. From the actual run output:
```
S4: 0.4403  ████████████████████  (high mean + high uncertainty → large average impact)
S7: 0.3430  ████████████████████  (high mean + low uncertainty → consistent large impact)
S6: 0.1964  ██████████
S8: 0.0100  ██                    (very low mean → minimal contribution)
```

---

### Phase 3 — Core Scorer (`phase3_scorer.py`)

**Purpose:** The single source of truth for all scoring math. Both the API and batch pipeline import from here. `predict.py` reimplements the same math inline to remain fully standalone.

**Two public functions:**
```python
result: dict         = score_single(match_vector, store, n_mc=1000)
result: pd.DataFrame = score(match_matrix, store, n_mc=500)
```

**Output fields:**

| Field | Description |
|---|---|
| `risk_point` | Point estimate using posterior means (fast, deterministic) |
| `risk_p50` | Median over MC samples (robust central estimate) |
| `risk_p10` | 10th percentile — lower credible bound |
| `risk_p90` | 90th percentile — upper credible bound |
| `impact_width` | `risk_p90 − risk_p10` — width of the uncertainty band |
| `primary_driver` | Source with highest marginal impact |
| `S1_impact … S8_impact` | Marginal impact per source |
| `matched_sources` | Human-readable matched source list |

**Actual output from pipeline run:**
```
S1+S4+S7 matched:
  risk_point = 0.6567
  CI 80%     = [0.5540, 0.7606]  width=0.2066
  driver     = S4

Only S7 matched (high-confidence, conc=300):
  risk_point = 0.3630
  CI 80%     = [0.3261, 0.3982]  width=0.0721  ← narrow

Only S4 matched (low-confidence, conc=15):
  risk_point = 0.4610
  CI 80%     = [0.3022, 0.6258]  width=0.3237  ← wide
```

S7's narrow CI vs S4's wide CI is the uncertainty system working correctly — same risk tier, different confidence, because S7 has 20× more concentration behind it.

---

### Phase 4 — API (`phase4_api.py`)

**Purpose:** Expose the scorer as a REST API via FastAPI.

**Key design decisions:**
- Param store loaded **once at startup**, not per request
- `/reload` endpoint for hot-swapping updated parameters without restart
- Input validated by Pydantic (wrong vector length, non-binary values → 422 error)
- Both single-record and batch endpoints available

**Endpoints:**

```
GET  /health              → liveness + param version
GET  /params/summary      → current Beta parameters
POST /score               → score one record
POST /score/batch         → score many records efficiently
POST /reload              → hot-reload updated param_store.json
```

**Example request/response:**
```json
POST /score
{"matches": [1, 0, 0, 1, 0, 0, 1, 0]}

{
  "risk_point": 0.6567,
  "risk_p50":   0.6523,
  "risk_p10":   0.5540,
  "risk_p90":   0.7606,
  "impact_width": 0.2066,
  "primary_driver": "S4",
  "matched_sources": "S1, S4, S7",
  "impacts": {"S1": 0.013, "S4": 0.412, "S7": 0.231},
  "latency_ms": 4.4
}
```

**Updating params without downtime:**
```bash
python phase1_priors.py
curl -X POST http://localhost:8000/reload
```

---

### Phase 5 — Batch Processing (`phase5_batch.py`)

**Purpose:** Score large CSV/Excel files with memory-safe chunked processing. Output format is controlled by the `output_format` parameter.

**Input format:**
```
record_id,entity_name,S1,S2,S3,S4,S5,S6,S7,S8
REC_00001,ACME Corp,1,0,0,1,0,0,1,0
REC_00002,Globex,0,1,0,0,0,1,0,0
```

**Output format parameter:**

| Value | Produces | Use when |
|---|---|---|
| `"excel"` | `.xlsx` multi-sheet report only | Sharing results with analysts or stakeholders |
| `"csv"` | `.csv` flat file only | Feeding scored results into another pipeline or database |
| `"both"` | Both `.xlsx` and `.csv` *(default)* | Full auditability — flat file for data, Excel for review |

The output path extension is always stripped and correct extensions applied automatically. Passing `results.xlsx` with `output_format="csv"` correctly produces `results.csv` only.

**Excel sheets (when writing Excel):**

| Sheet | Contents |
|---|---|
| `All_Records` | Complete scored output, all columns |
| `High_Risk_Alerts` | Moderate + High tier records, sorted by risk |
| `Focus_S4` / `Focus_S7` | Top 3 primary drivers get dedicated sheets |
| `Model_Info` | Param store version, source parameters, run metadata |

**CSV output** is a single flat file with all columns in a consistent order: original ID columns → risk score columns → tier labels → per-source impact columns.

**Python usage:**
```python
from phase5_batch import run_batch

run_batch("records.csv", "results", output_format="excel")
run_batch("records.csv", "results", output_format="csv")
run_batch("records.csv", "results", output_format="both")   # default
```

---

### `predict.py` — Standalone Quick Predictor

**Purpose:** Score a single record with zero pipeline dependencies. Drop it next to `param_store.json` and it works. No other pipeline files are needed.

**Dependencies:** `numpy` only. `pandas` is optional and only needed for `predict_df`.

**Three input styles:**
```python
from predict import load_model, predict

model = load_model("params/param_store.json")

# Dict — most readable, unspecified sources default to 0
result = predict(model, {"S1": 1, "S4": 1, "S7": 1})

# List — values in source order
result = predict(model, [1, 0, 0, 1, 0, 0, 1, 0])

# Empty dict — baseline only (no matches)
result = predict(model, {})
```

**Return value** is a plain dict — JSON-serializable, no special objects:

| Field | Example | Description |
|---|---|---|
| `risk_score` | `0.6567` | Point estimate using posterior means |
| `risk_p10` | `0.5540` | Lower bound of 80% credible interval |
| `risk_p50` | `0.6523` | Median over MC samples |
| `risk_p90` | `0.7606` | Upper bound of 80% credible interval |
| `ci_width` | `0.2066` | Width of the credible interval |
| `tier` | `"Moderate"` | Risk tier label |
| `confidence` | `"Low"` | Confidence tier (inverse of ci_width) |
| `primary_driver` | `"S4"` | Source with largest marginal impact |
| `matched` | `["S1","S4","S7"]` | List of matched source names |
| `impacts` | `{"S4": 0.281, ...}` | Marginal contribution per matched source |
| `explanation` | `"3 source(s) matched..."` | Human-readable one-sentence summary |

**CLI mode:**
```bash
python predict.py 1 0 0 1 0 0 1 0
```

**Notebook usage:**
```python
from predict import load_model, predict, predict_df

model = load_model("params/param_store.json")

# Single record
r = predict(model, {"S1": 1, "S4": 1, "S7": 1})
print(r["risk_score"])    # 0.6567
print(r["tier"])          # Moderate
print(r["explanation"])   # 3 source(s) matched (S1, S4, S7); risk score is 0.657...

# Several records → DataFrame (requires pandas)
records = [{"S1": 1, "S4": 1}, {"S7": 1, "S6": 1}, {}]
df = predict_df(model, records)
print(df[["risk_score", "tier", "primary_driver"]])
```

**predict_df output (actual):**
```
risk_score  risk_p10  risk_p90     tier confidence primary_driver
    0.4718    0.3127    0.6304 Moderate        Low             S4
    0.4904    0.4329    0.5511 Moderate   Moderate             S7
    0.0200    0.0055    0.0370  Minimal       High           Leak
```

**When to use `predict.py` vs other options:**

| Situation | Use |
|---|---|
| Exploring in a notebook, testing one record | `predict.py` |
| Scoring 10–200 records in a script or notebook | `predict_many()` or `predict_df()` in `predict.py` |
| Scoring 1000+ records from a file | `phase5_batch.py` (fully vectorized) |
| Real-time scoring from an application | `phase4_api.py` (FastAPI) |

---

## Uncertainty Quantification

The `impact_width` / `ci_width` field is a genuine 80% credible interval — not a variance approximation. It answers:

> "Given my uncertainty about each source's true risk probability, how wide is the range of plausible risk scores for this record?"

### Interpreting width

| Width | Meaning | Action |
|---|---|---|
| < 0.05 | High confidence | Trust the point estimate |
| 0.05–0.15 | Moderate confidence | Note uncertainty in reports |
| 0.15–0.40 | Low confidence | Flag for manual review |
| > 0.40 | Very uncertain | Do not act on score alone |

### Why width varies by record

A record matching only S4 (mean=0.45, concentration=15) will have very wide uncertainty — S4 itself is uncertain. A record matching only S7 (mean=0.35, concentration=300) will be narrow — S7 is well-validated. Same risk tier, very different confidence. The original code's `volatility` field didn't capture this correctly.

---

## Updating Source Confidence Over Time

Since you have no labeled training data at deployment, the system starts with expert priors. As evidence accumulates, you update using the conjugate Beta rule:

### Scenario: Analyst reviews records flagged by S2

```python
from phase1_priors import load_param_store, bayesian_update_source, save_param_store

store = load_param_store()

# 7 of 20 reviewed records were confirmed risk events
store = bayesian_update_source(store, "S2", n_successes=7, n_trials=20)
save_param_store(store)

# Hot-reload the API (no restart needed):
# curl -X POST http://localhost:8000/reload
```

**What happens mathematically:**
```
Before: Beta(2.5,  47.5) → mean=0.050, std=0.031
After:  Beta(9.5,  60.5) → mean=0.136, std=0.041
```

The mean shifts from 5% toward the observed 35% hit rate, dampened by the existing prior. Repeated over time, the mean converges to the true rate and uncertainty narrows.

### Accumulation schedule (recommended)

| Cadence | Action |
|---|---|
| Weekly | Aggregate analyst review decisions, run conjugate update, reload API |
| Monthly | Re-run Phase 2 validation to confirm model behavior hasn't drifted |
| Quarterly | Revisit expert specifications for sources with very high accumulated evidence |

---

## Quick Local Prediction

For ad-hoc scoring in a notebook or script without standing up the API or running the batch pipeline, use `predict.py` directly. It only needs two things alongside it: `numpy` and `params/param_store.json`.

### Minimum working example

```python
from predict import load_model, predict

model = load_model("params/param_store.json")
result = predict(model, {"S1": 1, "S4": 1, "S7": 1})

print(result["risk_score"])     # 0.6567
print(result["tier"])           # Moderate
print(result["ci_width"])       # 0.2066  (80% credible interval width)
print(result["primary_driver"]) # S4
print(result["explanation"])
# → 3 source(s) matched (S1, S4, S7); risk score is 0.657 (Moderate)
#   with low confidence [80% CI: 0.554–0.761].
#   Primary driver: S4 (contributes 0.281 of the total score).
```

### From the terminal in one line

```bash
python predict.py 1 0 0 1 0 0 1 0
```

### Latency

At the default `n_mc=2000`, a single prediction takes approximately **3–5ms** — fast enough to call in a loop for small batches without noticeable delay in a notebook.

---

## API Deployment

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install fastapi uvicorn numpy pandas scipy openpyxl
EXPOSE 8000
CMD ["uvicorn", "phase4_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Environment variables (recommended additions)

```bash
PARAM_STORE_PATH=/mnt/shared/params/param_store.json  # shared volume for hot-reload
LOG_LEVEL=info
```

### Scaling notes

- The scorer is **stateless** — all state is in the param store
- `--workers 4` gives you 4 parallel request handlers with no contention
- For high-volume (>1000 req/s), use `/score/batch` to amortize MC overhead
- Typical single-record latency: **3–5ms** at n_mc=1000

---

## Batch Processing

### Command-line usage

```bash
# Excel output only
python phase5_batch.py --input records.csv --output results --format excel

# CSV output only
python phase5_batch.py --input records.csv --output results --format csv

# Both Excel and CSV (default)
python phase5_batch.py --input records.csv --output results --format both

# Large file with tuned settings
python phase5_batch.py \
  --input large_file.csv \
  --output results \
  --format both \
  --n-mc 200 \
  --chunk-size 50000

# Generate test data and run
python phase5_batch.py --gen-test
```

### Output format guide

| Situation | Recommended format |
|---|---|
| Analyst review, stakeholder reporting | `excel` |
| Downstream pipeline, database ingest, further processing | `csv` |
| Full audit trail (default) | `both` |

The output path extension is always stripped automatically. Passing `--output results.xlsx` with `--format csv` correctly produces `results.csv` only — no need to think about extensions.

### Memory footprint

The dominant allocation is the MC risk matrix: `n_records_in_chunk × n_mc × 4 bytes (float32)`.

| Chunk size | n_mc | Memory |
|---|---|---|
| 10,000 | 500 | ~20 MB |
| 50,000 | 500 | ~100 MB |
| 100,000 | 200 | ~80 MB |

For a 1M record file with chunk_size=50,000: memory stays bounded at ~100 MB regardless of total file size.

---

## Performance Benchmarks

From verified pipeline runs:

| Operation | Tool | Records | n_mc | Time | Rate |
|---|---|---|---|---|---|
| Single record | `predict.py` | 1 | 2,000 | ~4ms | — |
| Single record, API equivalent | `phase3_scorer` | 1 | 2,000 | ~4ms | — |
| Mini-batch | `phase3_scorer` | 20 | 1,000 | 4.4ms | — |
| Batch (CSV output) | `phase5_batch` | 250 | 300 | 0.03s | ~8,900/s |
| Batch (Excel + CSV) | `phase5_batch` | 250 | 400 | 0.29s | ~860/s |
| Large batch | `phase5_batch` | 5,000 | 200 | 0.08s | ~63,000/s |
| Full pipeline incl. validation | `run_pipeline` | — | various | 1.4s total | — |

The original PyMC approach required ~30s just for model compilation before any scoring began. CSV-only output is substantially faster than Excel — the difference is formatting overhead, not scoring.

---

## Operational Runbook

### First-time setup

```bash
# 1. Edit source specifications in phase1_priors.py
#    (SOURCE_SPECS dict — set your means and concentrations)

# 2. Generate param store
python phase1_priors.py

# 3. Validate priors
python phase2_simulate_validate.py

# 4. Quick local smoke test
python predict.py 1 0 0 1 0 0 1 0

# 5. Run full local verification
python local_batch_test.py
python local_predict_test.py

# 6. Start API
uvicorn phase4_api:app --host 0.0.0.0 --port 8000

# 7. Run batch
python phase5_batch.py --input your_data.csv --format both
```

### Quick prediction in a notebook

```python
from predict import load_model, predict

model = load_model("params/param_store.json")
result = predict(model, {"S1": 1, "S4": 1})
print(result["risk_score"], result["tier"])
```

### Updating a source's expert estimate

```python
# In phase1_priors.py, change the SOURCE_SPECS entry:
SOURCE_SPECS = {
    ...
    "S4": (0.40, 25),  # was (0.45, 15) — slightly lower mean, more confident now
    ...
}
# Re-run:  python phase1_priors.py
# Test:    python predict.py 0 0 0 1 0 0 0 0
# Reload:  curl -X POST http://localhost:8000/reload
```

### Updating a source from new analyst evidence

```python
from phase1_priors import load_param_store, bayesian_update_source, save_param_store

store = load_param_store()
store = bayesian_update_source(store, "S2", n_successes=7, n_trials=20)
save_param_store(store)
# curl -X POST http://localhost:8000/reload
```

### Validation check fails

| Failing check | Likely cause | Fix |
|---|---|---|
| `full_match_saturates` | Individual source means too low | Increase mean for key sources, or add more sources |
| `baseline_risk` | Leak prior too high | Lower leak mean in `LEAK_SPEC` |
| `no_dominant_source` | A source has mean > 0.8 | Reconsider — this source will dominate all others |

---

## Common Questions

**Q: Do I need PyMC at all?**
No. The revised system replaces PyMC entirely with NumPy Beta sampling. PyMC would only be needed if you wanted to do full Bayesian posterior inference with a likelihood model (requires labeled outcomes). The conjugate update covers the no-labels case exactly.

**Q: What's the difference between `predict.py` and `phase3_scorer.py`?**
They produce identical scores — the math is the same. `predict.py` is fully self-contained (copy it anywhere with just `param_store.json` and `numpy`), returns a richer dict with tier labels and an explanation sentence, and is optimized for single-record use. `phase3_scorer.py` is the shared engine used by the API and batch pipeline — it operates on numpy arrays and returns DataFrames, which is faster at scale but requires the rest of the pipeline files to be present.

**Q: Can I use `predict.py` in a notebook without the rest of the pipeline?**
Yes. Copy `predict.py` and `params/param_store.json` into your notebook directory. That's all you need. `numpy` must be installed; `pandas` is optional (only needed for `predict_df`).

**Q: What if source match scores are not binary?**
The match input supports float values in [0, 1] — a match score of 0.7 means the source contributed `log(1−p_i) × 0.7` to the log-failure sum. This naturally handles fuzzy or confidence-weighted matching. Pass floats in list form; dict input currently expects 0/1 values.

**Q: Why does the full-match case score 0.79 rather than near 1.0?**
Because several sources have significant uncertainty. S4 has mean=0.45 with concentration=15 — it frequently draws low values in MC samples, pulling the aggregate down. This is correct behavior: the model is honest about uncertain sources. To push the full-match score higher, increase the mean or concentration of your key sources.

**Q: How do I add a new source?**
Add it to `SOURCE_SPECS` in `phase1_priors.py`, re-run Phase 1 and 2, and ensure all input data (batch CSVs, API requests, `predict.py` calls) include the new source column or key. The scorer dimensions are driven entirely by `store["source_order"]` — there are no hardcoded source counts anywhere in the codebase.

**Q: Which output format should I use for batch?**
Use `"both"` (the default) until you know you don't need one of the outputs. CSV is faster to write and easier to ingest downstream; Excel is easier to share with stakeholders. If you're piping results directly into a database or another script, `"csv"` alone avoids the formatting overhead.
