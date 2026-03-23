"""
noisy_or_model.py
=================
Noisy-OR Bayesian Risk Model — server / notebook edition.

Mirrors the logic of the Flask web app exactly, but designed for
programmatic use: single predictions, batch predictions, and
human-readable CSV / Excel exports.

Batch optimizations
-------------------
predict_batch() uses three optimizations automatically — no extra code
required from the caller:

  1. Pre-sampled Beta pool — 50,000 Beta draws per source are computed
     once at load time and reused across all batch calls. No sampling
     happens at prediction time.

  2. Vectorized scoring — the full batch is scored in a single NumPy
     pass over a 3-D array (n_records × n_sources × n_samples). No
     Python loop over records.

  3. Duplicate-row cache — for large datasets that contain repeated
     flag combinations (e.g. many records with the same single source
     active), each unique combination is scored only once and the result
     reused for every matching record.

Single predictions (predict()) are unaffected and use the original
per-call sampling path.

Usage
-----
    from noisy_or_model import NoisyORModel

    model = NoisyORModel("noisy_or_config.json")

    # Single prediction (numpy array: 1 = source matched, 0 = did not match)
    import numpy as np
    result = model.predict(np.array([1, 0, 1, 0, 0, 1, 0, 0]))
    print(result)

    # Batch prediction — automatically uses pool + vectorized path
    inputs = np.array([
        [1, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 0],
        [1, 1, 0, 0, 1, 0, 0, 1],
    ])
    df = model.predict_batch(inputs)                        # pandas (default)
    df = model.predict_batch(inputs, backend="polars")      # polars DataFrame
    print(df)

    # Export batch to Excel or CSV
    model.export_batch(inputs, "results.xlsx")              # always pandas/openpyxl
    model.export_batch(inputs, "results.csv")               # pandas CSV
    model.export_batch(inputs, "results.csv", backend="polars")  # polars CSV

    # Pool is rebuilt automatically if you reload the config
    # (use NoisyORModelAPI for hot reload without reinstantiation)

Adding Sources
--------------
Edit noisy_or_config.json — add a dict to "sources":
    {"name": "My New Source", "mu": 0.25, "kappa": 6.0}
No code changes required.

Changing Monte Carlo sample count
----------------------------------
Edit "n_monte_carlo_samples" in noisy_or_config.json, or pass
n_samples= directly to predict() / predict_batch().
Note: the pool is always built at 50,000 samples. n_samples controls
how many of those are used per batch call.
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

# Polars is optional — install with: pip install polars
try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Batch pool configuration
# ---------------------------------------------------------------------------
# The pool is built at this size at load time.  predict_batch() slices
# n_samples columns per call — so this must be >= any n_samples value used.
_BATCH_POOL_SIZE = 50_000


# ---------------------------------------------------------------------------
# Certainty thresholds — mirrors the web app's colour logic
# ---------------------------------------------------------------------------
def _certainty_label(cert: float) -> str:
    if cert >= 0.80:
        return "High"
    if cert >= 0.50:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Core simulation (identical math to the Flask run_simulation function)
# ---------------------------------------------------------------------------
def _run_simulation(
    mu_values: np.ndarray,
    kappa_values: np.ndarray,
    active_mask: np.ndarray,
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Parameters
    ----------
    mu_values     : shape (n_sources,) — prior mean for each source
    kappa_values  : shape (n_sources,) — concentration for each source
    active_mask   : shape (n_sources,) — 1 if source matched, 0 otherwise
    n_samples     : number of Monte Carlo draws
    rng           : optional numpy RNG for reproducibility

    Returns
    -------
    dict with simulation statistics and per-source marginal contributions
    """
    if rng is None:
        rng = np.random.default_rng()

    n_sources = len(mu_values)

    # Sample Beta distributions for every source
    all_probs: list[np.ndarray] = []
    for mu, kappa in zip(mu_values, kappa_values):
        alpha = max(float(mu) * float(kappa), 0.001)
        beta  = max((1.0 - float(mu)) * float(kappa), 0.001)
        all_probs.append(rng.beta(alpha, beta, n_samples))

    # Combine only active sources via Noisy-OR
    active_probs = [p for p, a in zip(all_probs, active_mask) if a]

    if active_probs:
        combined = 1.0 - np.prod([1.0 - p for p in active_probs], axis=0)
    else:
        combined = np.zeros(n_samples)

    mean_risk   = float(np.mean(combined))
    median_risk = float(np.median(combined))
    p5          = float(np.percentile(combined, 5))
    p95         = float(np.percentile(combined, 95))
    certainty   = max(0.0, 1.0 - (p95 - p5))

    # Per-source marginal contributions (analytical from source means)
    source_means = [float(np.mean(p)) for p in all_probs]
    active_indices = [i for i, a in enumerate(active_mask) if a]

    contribs: dict[int, float] = {}
    if active_indices:
        active_mus = [source_means[i] for i in active_indices]
        prod_all = math.prod(1.0 - m for m in active_mus)
        overall_risk_approx = 1.0 - prod_all

        for idx, mu_i in zip(active_indices, active_mus):
            if mu_i > 0.9999:
                prod_without = 0.0
            else:
                prod_without = prod_all / (1.0 - mu_i)
            risk_without = 1.0 - prod_without
            contribs[idx] = overall_risk_approx - risk_without

    return {
        "mean_risk":      round(mean_risk, 4),
        "median_risk":    round(median_risk, 4),
        "p5":             round(p5, 4),
        "p95":            round(p95, 4),
        "certainty":      round(certainty, 4),
        "certainty_label": _certainty_label(certainty),
        "source_means":   [round(m, 4) for m in source_means],
        "marginal_contribs": contribs,   # {source_index: delta}
    }


# ---------------------------------------------------------------------------
# Vectorized batch simulation using pre-sampled pool
# ---------------------------------------------------------------------------
def _run_batch_vectorized(
    pool: np.ndarray,           # shape (n_sources, pool_size)
    source_means: np.ndarray,   # shape (n_sources,)
    active_matrix: np.ndarray,  # shape (n_records, n_sources)
    n_samples: int,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Score all records in a single vectorized NumPy pass using the pre-built
    pool.  No Python loop over records.  Each unique row combination is
    scored only once (deduplication is handled by the caller via the batch
    cache on NoisyORModel).

    Returns a list of raw simulation dicts in the same order as active_matrix.
    """
    n_records, n_sources = active_matrix.shape

    # Random column indices into the pool — shared across all records
    idx        = rng.integers(0, pool.shape[1], size=n_samples)
    pool_slice = pool[:, idx]   # (n_sources, n_samples)

    # Build 3-D array: (n_records, n_sources, n_samples)
    # Inactive sources are multiplied by 0 → (1 - 0) = 1 in the product,
    # so they have no effect on the combined risk.
    p3d      = pool_slice[np.newaxis, :, :] * active_matrix[:, :, np.newaxis]
    combined = 1.0 - np.prod(1.0 - p3d, axis=1)   # (n_records, n_samples)

    mean_risk   = np.mean(combined, axis=1)
    median_risk = np.median(combined, axis=1)
    p5          = np.percentile(combined, 5,  axis=1)
    p95         = np.percentile(combined, 95, axis=1)
    certainty   = np.clip(1.0 - (p95 - p5), 0.0, None)

    # Vectorized marginal contributions
    active_mus   = source_means[np.newaxis, :] * active_matrix.astype(float)
    inactive     = (active_matrix == 0)
    prod_terms   = np.where(inactive, 1.0, 1.0 - active_mus)
    prod_all     = np.prod(prod_terms, axis=1)                      # (n_records,)
    overall_risk = 1.0 - prod_all
    denom        = np.where(active_mus > 0.9999, np.inf, 1.0 - active_mus)
    prod_without = prod_all[:, np.newaxis] / denom
    contribs_mat = overall_risk[:, np.newaxis] - (1.0 - prod_without)
    contribs_mat = np.where(active_matrix, contribs_mat, 0.0)

    results = []
    for r in range(n_records):
        contribs = {
            int(i): round(float(contribs_mat[r, i]), 4)
            for i in range(n_sources)
            if active_matrix[r, i]
        }
        results.append({
            "mean_risk":         round(float(mean_risk[r]),   4),
            "median_risk":       round(float(median_risk[r]), 4),
            "p5":                round(float(p5[r]),          4),
            "p95":               round(float(p95[r]),         4),
            "certainty":         round(float(certainty[r]),   4),
            "certainty_label":   _certainty_label(float(certainty[r])),
            "source_means":      [round(float(m), 4) for m in source_means],
            "marginal_contribs": contribs,
        })
    return results


# ---------------------------------------------------------------------------
# Memory-safe chunking threshold for the vectorized batch scorer
# ---------------------------------------------------------------------------
# The 3-D array (n_records × n_sources × n_samples) is built in memory.
# When the projected size exceeds this threshold the batch is split into
# chunks automatically.  Override by setting _BATCH_CHUNK_BYTES directly.
_BATCH_CHUNK_BYTES: int = 512 * 1024 * 1024   # 512 MB default


# ---------------------------------------------------------------------------
# Public model class
# ---------------------------------------------------------------------------
class NoisyORModel:
    """
    Noisy-OR Bayesian Risk Model.

    Parameters
    ----------
    config_path : path to JSON config file (see noisy_or_config.json)
    """

    def __init__(self, config_path: Union[str, Path] = "noisy_or_config.json"):
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        self._config_path = config_path
        self._load_from_config(config_path)

        # Build pre-sampled pool for batch operations
        self._build_batch_pool()

    def _load_from_config(self, config_path: Path) -> None:
        """Parse a config file and populate all model attributes."""
        with open(config_path) as f:
            cfg = json.load(f)

        self._sources: list[dict] = cfg["sources"]
        self._n_samples_default: int = int(cfg.get("n_monte_carlo_samples", 20_000))
        self._version: str = cfg.get("version", "1.0.0")
        self._version_date: str = cfg.get("version_date", str(date.today()))

        self._names        = [s["name"]  for s in self._sources]
        self._mu_values    = np.array([s["mu"]    for s in self._sources], dtype=float)
        self._kappa_values = np.array([s["kappa"] for s in self._sources], dtype=float)
        self._n_sources    = len(self._sources)

        # Optional risk bands — sorted descending by threshold so the first
        # match wins.  Format in config:
        #   "risk_bands": [
        #     {"label": "Critical", "min_score": 0.80},
        #     {"label": "High",     "min_score": 0.60},
        #     {"label": "Medium",   "min_score": 0.40},
        #     {"label": "Low",      "min_score": 0.00}
        #   ]
        raw_bands = cfg.get("risk_bands", [])
        self._risk_bands: list[tuple[float, str]] = sorted(
            [(float(b["min_score"]), str(b["label"])) for b in raw_bands],
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Single prediction
    # ------------------------------------------------------------------
    def predict(
        self,
        active_flags: np.ndarray,
        n_samples: Optional[int] = None,
        seed: Optional[int] = None,
        label: Optional[str] = None,
    ) -> dict:
        """
        Run the Noisy-OR model for a single input vector.

        Parameters
        ----------
        active_flags : 1-D numpy array of length n_sources.
                       1 = source matched this record, 0 = did not match.
        n_samples    : Monte Carlo draws (defaults to config value)
        seed         : optional int for reproducible results
        label        : optional string label for this record

        Returns
        -------
        dict with keys:
            label, risk_score, median_risk, p5, p95,
            certainty, certainty_label,
            primary_driver, primary_driver_contribution,
            other_matches  (list of dicts),
            source_means   (list of floats, one per source),
            active_sources (list of source names that matched)
        """
        active_flags = np.asarray(active_flags, dtype=int)
        self._validate_flags(active_flags)

        rng = np.random.default_rng(seed)
        n = n_samples or self._n_samples_default

        sim = _run_simulation(
            self._mu_values,
            self._kappa_values,
            active_flags,
            n,
            rng,
        )

        return self._format_single(sim, active_flags, label)

    # ------------------------------------------------------------------
    # Reload from disk (mirrors NoisyORModelAPI.reload() for the standard model)
    # ------------------------------------------------------------------
    def reload(self) -> None:
        """
        Re-read the config file and rebuild the batch pool in-place.

        Use this after a FeedbackUpdater.apply() call to pick up new
        parameters without reinstantiating the model.

        Note: unlike NoisyORModelAPI.reload(), this method is NOT
        thread-safe.  If concurrent batch calls are in flight, use
        NoisyORModelAPI instead.
        """
        if not self._config_path.exists():
            raise FileNotFoundError(f"Config not found: {self._config_path}")
        self._load_from_config(self._config_path)
        self._build_batch_pool()
        print(f"NoisyORModel reloaded — {self!r}")

    # ------------------------------------------------------------------
    # Risk band label
    # ------------------------------------------------------------------
    def _risk_band_label(self, score: float) -> Optional[str]:
        """Return the risk band label for a given score, or None if no bands configured."""
        for threshold, label in self._risk_bands:
            if score >= threshold:
                return label
        return None
    def _build_batch_pool(self, seed: Optional[int] = None) -> None:
        """
        Draw _BATCH_POOL_SIZE Beta samples for every source and store as a
        (n_sources, pool_size) float32 matrix.  Called once at __init__.
        predict_batch() slices columns from this pool instead of sampling
        fresh Beta draws for every call.
        """
        rng  = np.random.default_rng(seed)
        pool = np.empty((self._n_sources, _BATCH_POOL_SIZE), dtype=np.float32)
        for i, (mu, kappa) in enumerate(zip(self._mu_values, self._kappa_values)):
            alpha = max(float(mu) * float(kappa), 0.001)
            beta  = max((1.0 - float(mu)) * float(kappa), 0.001)
            pool[i] = rng.beta(alpha, beta, _BATCH_POOL_SIZE)

        self._batch_pool         = pool
        self._batch_pool_rng     = np.random.default_rng()
        self._batch_source_means = pool.mean(axis=1).astype(float)  # (n_sources,)

    # ------------------------------------------------------------------
    # Batch prediction — vectorized + pool-backed + duplicate cache
    # ------------------------------------------------------------------
    def predict_batch(
        self,
        active_matrix: np.ndarray,
        n_samples: Optional[int] = None,
        seed: Optional[int] = None,
        labels: Optional[list[str]] = None,
        backend: str = "pandas",
    ) -> "Union[pd.DataFrame, pl.DataFrame]":
        """
        Run the model for many input vectors using three optimizations:

          1. Pre-sampled pool  — Beta draws are reused from the pool built
             at load time; no fresh sampling per call.
          2. Vectorized pass   — all records are scored simultaneously in a
             single NumPy operation; no Python loop over records.
          3. Duplicate cache   — repeated flag combinations (common in large
             screening datasets) are scored only once per batch call.

        Parameters
        ----------
        active_matrix : 2-D numpy array, shape (n_records, n_sources).
                        Each row is one record; values are 0 or 1.
        n_samples     : how many pool samples to use per record (defaults to
                        config value, capped at pool size of 50,000)
        seed          : optional int for a reproducible pool-slice RNG
        labels        : optional list of string labels for each row
        backend       : "pandas" (default) or "polars".
                        "polars" requires polars to be installed:
                            pip install polars

        Returns
        -------
        pandas DataFrame when backend="pandas" (default).
        polars DataFrame when backend="polars".
        Both have identical column names and values.
        """
        if backend not in ("pandas", "polars"):
            raise ValueError(f"backend must be 'pandas' or 'polars', got '{backend}'")
        if backend == "polars" and not _POLARS_AVAILABLE:
            raise ImportError(
                "polars is not installed. Install it with: pip install polars"
            )

        active_matrix = np.atleast_2d(np.asarray(active_matrix, dtype=int))
        n_records = active_matrix.shape[0]
        n = min(n_samples or self._n_samples_default, _BATCH_POOL_SIZE)

        if labels is None:
            labels = [f"Record_{i+1}" for i in range(n_records)]

        # Vectorized validation — two NumPy calls instead of N Python loops
        self._validate_matrix(active_matrix)

        rng = np.random.default_rng(seed)

        # --- Deduplication: find unique flag combinations ---
        keys        = [tuple(row.tolist()) for row in active_matrix]
        unique_keys = list(dict.fromkeys(keys))
        unique_matrix = np.array(unique_keys, dtype=int)

        # --- Memory-safe scoring: chunk if the 3-D array would be too large ---
        projected_bytes = unique_matrix.shape[0] * self._n_sources * n * 4
        if projected_bytes > _BATCH_CHUNK_BYTES:
            chunk_size = max(1, _BATCH_CHUNK_BYTES // (self._n_sources * n * 4))
            unique_sims: list[dict] = []
            for start in range(0, unique_matrix.shape[0], chunk_size):
                chunk = unique_matrix[start : start + chunk_size]
                unique_sims.extend(
                    _run_batch_vectorized(
                        self._batch_pool, self._batch_source_means, chunk, n, rng
                    )
                )
        else:
            unique_sims = _run_batch_vectorized(
                self._batch_pool, self._batch_source_means, unique_matrix, n, rng
            )

        sim_lookup = {k: unique_sims[i] for i, k in enumerate(unique_keys)}

        # --- Assemble output rows in original record order ---
        rows = [
            self._format_row(sim_lookup[key], active_matrix[i], labels[i])
            for i, key in enumerate(keys)
        ]

        if backend == "polars":
            return self._to_polars(rows)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------
    def export_batch(
        self,
        active_matrix: np.ndarray,
        output_path: Union[str, Path],
        n_samples: Optional[int] = None,
        seed: Optional[int] = None,
        labels: Optional[list[str]] = None,
        backend: str = "pandas",
    ) -> "Union[pd.DataFrame, pl.DataFrame]":
        """
        Run batch predictions and save to Excel (.xlsx) or CSV (.csv).

        Returns the DataFrame (pandas or polars depending on backend) so
        you can inspect it in a notebook without loading the file again.

        Parameters
        ----------
        active_matrix : 2-D numpy array, shape (n_records, n_sources)
        output_path   : destination file path (.xlsx or .csv)
        n_samples     : Monte Carlo sample count override
        seed          : optional int for reproducible RNG
        labels        : optional list of record labels
        backend       : "pandas" (default) or "polars".
                        CSV export uses the backend's own writer.
                        Excel export always uses openpyxl regardless of
                        backend (Polars DataFrames are converted to pandas
                        internally for Excel writing only).

        Returns
        -------
        pandas or polars DataFrame (matches backend parameter).
        """
        df = self.predict_batch(active_matrix, n_samples=n_samples,
                                seed=seed, labels=labels, backend=backend)
        output_path = Path(output_path)
        suffix = output_path.suffix.lower()

        if suffix == ".xlsx":
            # Excel always goes through openpyxl (pandas-backed)
            pandas_df = df.to_pandas() if backend == "polars" else df
            self._write_excel(pandas_df, output_path)
        elif suffix == ".csv":
            if backend == "polars":
                df.write_csv(str(output_path))
                print(f"CSV saved → {output_path}  (polars)")
            else:
                df.to_csv(output_path, index=False)
                print(f"CSV saved → {output_path}")
        else:
            raise ValueError(f"Unsupported extension '{suffix}'. Use .xlsx or .csv")

        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_matrix(self, matrix: np.ndarray) -> None:
        """Validate an entire active_matrix in two vectorized NumPy checks."""
        if matrix.ndim != 2 or matrix.shape[1] != self._n_sources:
            raise ValueError(
                f"active_matrix must have shape (n_records, {self._n_sources}), "
                f"got {matrix.shape}."
            )
        if not np.isin(matrix, [0, 1]).all():
            bad_rows = np.where(~np.isin(matrix, [0, 1]).all(axis=1))[0].tolist()
            raise ValueError(
                f"active_matrix must contain only 0 or 1. "
                f"Bad row(s): {bad_rows}"
            )

    def _validate_flags(self, flags: np.ndarray) -> None:
        """Validate a single 1-D flags vector."""
        if flags.shape != (self._n_sources,):
            raise ValueError(
                f"Expected array of length {self._n_sources} "
                f"(one entry per source), got shape {flags.shape}."
            )
        if not np.isin(flags, [0, 1]).all():
            raise ValueError("active_flags must contain only 0 or 1.")

    @staticmethod
    def _validate_source_params(mu: float, kappa: float, name: str) -> None:
        """Raise if mu or kappa are outside their valid ranges."""
        if not (0.0 < mu < 1.0):
            raise ValueError(
                f"Source '{name}': mu={mu} must be strictly between 0 and 1."
            )
        if kappa <= 0.0:
            raise ValueError(
                f"Source '{name}': kappa={kappa} must be greater than 0."
            )

    def _format_single(self, sim: dict, flags: np.ndarray, label: Optional[str]) -> dict:
        contribs = sim["marginal_contribs"]
        active_idx = [i for i, a in enumerate(flags) if a]

        # Rank active sources by marginal contribution
        ranked = sorted(active_idx, key=lambda i: contribs.get(i, 0.0), reverse=True)

        primary_driver = None
        primary_contrib = None
        other_matches = []

        if ranked:
            primary_driver = self._names[ranked[0]]
            primary_contrib = round(contribs.get(ranked[0], 0.0), 4)
            for i in ranked[1:]:
                other_matches.append({
                    "source": self._names[i],
                    "marginal_contribution": round(contribs.get(i, 0.0), 4),
                    "source_mean": sim["source_means"][i],
                })

        return {
            "label":                      label,
            "risk_score":                 sim["mean_risk"],
            "median_risk":                sim["median_risk"],
            "p5":                         sim["p5"],
            "p95":                        sim["p95"],
            "certainty":                  sim["certainty"],
            "certainty_label":            sim["certainty_label"],
            "risk_band":                  self._risk_band_label(sim["mean_risk"]),
            "primary_driver":             primary_driver,
            "primary_driver_contribution": primary_contrib,
            "other_matches":              other_matches,
            "source_means":               sim["source_means"],
            "active_sources":             [self._names[i] for i in active_idx],
        }

    def _format_row(self, sim: dict, flags: np.ndarray, label: str) -> dict:
        """Flatten a single result into a dict of scalar/string columns."""
        res = self._format_single(sim, flags, label)
        contribs = sim["marginal_contribs"]

        row: dict = {
            "Label":                       res["label"],
            "Risk_Score":                  res["risk_score"],
            "Median_Risk":                 res["median_risk"],
            "P5":                          res["p5"],
            "P95":                         res["p95"],
            "Certainty":                   res["certainty"],
            "Certainty_Label":             res["certainty_label"],
            "Primary_Driver":              res["primary_driver"] or "",
            "Primary_Driver_Contribution": res["primary_driver_contribution"] or 0.0,
        }

        if self._risk_bands:
            row["Risk_Band"] = res["risk_band"] or ""

        for i, name in enumerate(self._names):
            col_base = name.replace(" ", "_")
            row[f"{col_base}_Active"]           = int(flags[i])
            row[f"{col_base}_MarginalContrib"]  = round(contribs.get(i, 0.0), 4) if flags[i] else 0.0

        row["Active_Sources"] = ", ".join(res["active_sources"]) if res["active_sources"] else "None"
        return row

    # ------------------------------------------------------------------
    # Streaming export — processes records in chunks, appends to file
    # ------------------------------------------------------------------
    def stream_export(
        self,
        active_matrix: np.ndarray,
        output_path: Union[str, Path],
        n_samples: Optional[int] = None,
        seed: Optional[int] = None,
        labels: Optional[list[str]] = None,
        chunk_size: int = 5_000,
        backend: str = "pandas",
    ) -> None:
        """
        Score a large batch in chunks and write results incrementally,
        keeping peak memory proportional to chunk_size rather than the
        full dataset size.

        Only CSV output is supported (Excel requires the full workbook in
        memory before saving).  For Excel, use export_batch() on datasets
        that fit comfortably in memory.

        Parameters
        ----------
        active_matrix : 2-D numpy array, shape (n_records, n_sources)
        output_path   : destination .csv file path
        n_samples     : Monte Carlo sample count override
        seed          : optional int for reproducible RNG across chunks
        labels        : optional list of record labels
        chunk_size    : records per processing chunk (default 5,000)
        backend       : "pandas" or "polars" (controls CSV writer)
        """
        output_path = Path(output_path)
        if output_path.suffix.lower() != ".csv":
            raise ValueError("stream_export() only supports .csv output. "
                             "Use export_batch() for .xlsx.")

        active_matrix = np.atleast_2d(np.asarray(active_matrix, dtype=int))
        n_records = active_matrix.shape[0]
        self._validate_matrix(active_matrix)

        if labels is None:
            labels = [f"Record_{i+1}" for i in range(n_records)]

        header_written = False
        rng = np.random.default_rng(seed)

        for start in range(0, n_records, chunk_size):
            end         = min(start + chunk_size, n_records)
            chunk_mat   = active_matrix[start:end]
            chunk_lbls  = labels[start:end]

            chunk_df = self.predict_batch(
                chunk_mat, n_samples=n_samples, labels=chunk_lbls,
                backend="pandas",     # always pandas internally; convert after
            )

            if backend == "polars":
                import polars as pl
                chunk_pl = pl.from_pandas(chunk_df)
                if not header_written:
                    chunk_pl.write_csv(str(output_path))
                    header_written = True
                else:
                    with open(output_path, "a") as f:
                        f.write(chunk_pl.write_csv(file=None).split("\n", 1)[1])
            else:
                chunk_df.to_csv(
                    output_path,
                    mode="a" if header_written else "w",
                    header=not header_written,
                    index=False,
                )
                header_written = True

        print(f"Streamed {n_records:,} records → {output_path}  "
              f"(chunk_size={chunk_size:,})")

    def _to_polars(self, rows: list[dict]) -> "pl.DataFrame":
        """
        Convert a list of flat row dicts (as produced by _format_row) to a
        Polars DataFrame with correct dtypes.

        Column dtype mapping:
          Label, Certainty_Label, Primary_Driver, Active_Sources  → Utf8
          *_Active                                                 → Int32
          all other numeric columns                               → Float64
        """
        if not rows:
            # Return an empty Polars DataFrame with the correct schema
            return pl.DataFrame()

        # Collect columns from the first row to infer types
        str_cols  = {"Label", "Certainty_Label", "Primary_Driver", "Active_Sources",
                     "Risk_Band"}
        int_suffixes = ("_Active",)

        series: dict[str, pl.Series] = {}
        for col in rows[0]:
            values = [row[col] for row in rows]
            if col in str_cols:
                series[col] = pl.Series(col, values, dtype=pl.Utf8)
            elif col.endswith(int_suffixes):
                series[col] = pl.Series(col, values, dtype=pl.Int32)
            else:
                series[col] = pl.Series(col, values, dtype=pl.Float64)

        return pl.DataFrame(series)

    def _write_excel(self, df: pd.DataFrame, path: Path) -> None:
        """Write a polished, human-readable Excel workbook."""
        from openpyxl import Workbook
        from openpyxl.styles import (
            Font, PatternFill, Alignment, Border, Side, numbers
        )
        from openpyxl.utils import get_column_letter

        wb = Workbook()

        # ---- Sheet 1: Results summary ----
        ws = wb.active
        ws.title = "Risk Results"

        # Colour palette
        HDR_FILL    = PatternFill("solid", fgColor="1D6FA4")
        SUBHDR_FILL = PatternFill("solid", fgColor="2E86C1")
        HIGH_FILL   = PatternFill("solid", fgColor="D5F5E3")
        MED_FILL    = PatternFill("solid", fgColor="FEF9E7")
        LOW_FILL    = PatternFill("solid", fgColor="FDEDEC")
        ALT_FILL    = PatternFill("solid", fgColor="EBF5FB")
        WHITE_FILL  = PatternFill("solid", fgColor="FFFFFF")

        HDR_FONT    = Font(name="Arial", bold=True, color="FFFFFF", size=10)
        BODY_FONT   = Font(name="Arial", size=10)
        BOLD_FONT   = Font(name="Arial", bold=True, size=10)
        TITLE_FONT  = Font(name="Arial", bold=True, size=13, color="1D6FA4")

        thin = Side(style="thin", color="BDBDBD")
        BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)
        CENTER = Alignment(horizontal="center", vertical="center")
        LEFT   = Alignment(horizontal="left",   vertical="center")

        PCT_FMT   = "0.0%"
        SCORE_FMT = "0.0000"

        # Title row
        ws.merge_cells("A1:O1")
        title_cell = ws["A1"]
        title_cell.value     = "Noisy-OR Bayesian Risk Model — Batch Results"
        title_cell.font      = TITLE_FONT
        title_cell.alignment = LEFT
        ws.row_dimensions[1].height = 24

        # Column groups
        summary_cols = [
            ("Label",                       "Label",                   18, None),
            ("Risk_Score",                  "Risk Score",              11, PCT_FMT),
            ("Median_Risk",                 "Median Risk",             12, PCT_FMT),
            ("P5",                          "P5 (5th pct)",            11, PCT_FMT),
            ("P95",                         "P95 (95th pct)",          13, PCT_FMT),
            ("Certainty",                   "Certainty",               11, PCT_FMT),
            ("Certainty_Label",             "Certainty Level",         14, None),
            ("Primary_Driver",              "Primary Driver",          18, None),
            ("Primary_Driver_Contribution", "Primary Contribution",    18, PCT_FMT),
            ("Active_Sources",              "All Matched Sources",     30, None),
        ]

        # Insert Risk_Band column after Certainty_Label if bands are configured
        if self._risk_bands and "Risk_Band" in df.columns:
            summary_cols.insert(7, ("Risk_Band", "Risk Band", 14, None))

        source_cols = []
        for name in self._names:
            col_base = name.replace(" ", "_")
            source_cols.append((f"{col_base}_Active",          f"{name}\nActive",        10, None))
            source_cols.append((f"{col_base}_MarginalContrib", f"{name}\nMarginal Contrib", 14, PCT_FMT))

        all_cols = summary_cols + source_cols
        n_cols   = len(all_cols)

        # Header row (row 2)
        for col_num, (_, hdr, width, _fmt) in enumerate(all_cols, start=1):
            cell = ws.cell(row=2, column=col_num, value=hdr)
            cell.font      = HDR_FONT
            cell.fill      = HDR_FILL
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border    = BORDER
            ws.column_dimensions[get_column_letter(col_num)].width = width
        ws.row_dimensions[2].height = 36

        # Extract column data as arrays — one pass over the DataFrame,
        # then ws.append() per row instead of per-cell ws.cell() calls.
        col_keys = [df_col for df_col, *_ in all_cols]
        col_fmts = {df_col: fmt for df_col, _, _, fmt in all_cols if fmt}
        str_cols_set = {"Label", "Certainty_Label", "Primary_Driver", "Active_Sources"}
        if self._risk_bands:
            str_cols_set.add("Risk_Band")

        # Convert all needed columns to Python lists up front (fast NumPy path)
        col_data = {k: df[k].tolist() if k in df.columns else [""] * len(df)
                    for k in col_keys}

        cert_fill_map = {"High": HIGH_FILL, "Medium": MED_FILL, "Low": LOW_FILL}
        cert_labels   = col_data.get("Certainty_Label", [""] * len(df))
        cert_col_idx  = next((i for i, (c, *_) in enumerate(all_cols)
                              if c == "Certainty"), None)
        certlbl_col_idx = next((i for i, (c, *_) in enumerate(all_cols)
                                if c == "Certainty_Label"), None)

        for row_num, row_idx in enumerate(range(len(df)), start=3):
            cert_lbl = cert_labels[row_idx] if row_idx < len(cert_labels) else ""
            row_fill = cert_fill_map.get(str(cert_lbl), WHITE_FILL)
            alt_fill = ALT_FILL if (row_num % 2 == 0) else WHITE_FILL

            row_values = [col_data[k][row_idx] if row_idx < len(col_data[k]) else ""
                          for k in col_keys]

            # Append the raw values in one call — much faster than ws.cell() per cell
            ws.append(row_values)

            # Apply per-cell formatting only (fills, borders, number formats)
            for col_num, (df_col, _, _, fmt) in enumerate(all_cols, start=1):
                cell = ws.cell(row=row_num, column=col_num)
                cell.font      = BODY_FONT
                cell.border    = BORDER
                cell.alignment = CENTER if fmt else LEFT

                if df_col in ("Certainty", "Certainty_Label"):
                    cell.fill = row_fill
                    if df_col == "Certainty_Label":
                        cell.font = Font(name="Arial", bold=True, size=10)
                else:
                    cell.fill = alt_fill

                if fmt:
                    cell.number_format = fmt

        ws.freeze_panes = "A3"
        ws.auto_filter.ref = f"A2:{get_column_letter(n_cols)}2"

        # ---- Sheet 2: Config snapshot ----
        ws2 = wb.create_sheet("Model Config")
        ws2["A1"].value = "Noisy-OR Model Configuration"
        ws2["A1"].font  = TITLE_FONT

        cfg_hdrs = ["Source Name", "Mu (Prior Mean)", "Kappa (Concentration)"]
        for c, h in enumerate(cfg_hdrs, start=1):
            cell = ws2.cell(row=2, column=c, value=h)
            cell.font = HDR_FONT
            cell.fill = HDR_FILL
            cell.border = BORDER
            cell.alignment = CENTER

        ws2.column_dimensions["A"].width = 20
        ws2.column_dimensions["B"].width = 16
        ws2.column_dimensions["C"].width = 20

        for r, src in enumerate(self._sources, start=3):
            ws2.cell(row=r, column=1, value=src["name"]).font   = BODY_FONT
            ws2.cell(row=r, column=2, value=src["mu"]).font     = BODY_FONT
            ws2.cell(row=r, column=3, value=src["kappa"]).font  = BODY_FONT
            for c in range(1, 4):
                ws2.cell(row=r, column=c).border = BORDER

        n_row = len(self._sources) + 4
        ws2.cell(row=n_row, column=1, value="Monte Carlo Samples").font = BOLD_FONT
        ws2.cell(row=n_row, column=2, value=self._n_samples_default).font = BODY_FONT

        wb.save(path)
        print(f"Excel workbook saved → {path}")

    # ------------------------------------------------------------------
    # Pretty-print for single prediction (notebook / script use)
    # ------------------------------------------------------------------
    def print_result(self, result: dict) -> None:
        """Pretty-print a single prediction result."""
        lbl = result.get("label") or "Record"
        print(f"\n{'='*60}")
        print(f"  NOISY-OR RISK RESULT  |  {lbl}")
        print(f"{'='*60}")
        print(f"  Risk Score   : {result['risk_score']:.1%}")
        print(f"  Median Risk  : {result['median_risk']:.1%}")
        print(f"  90% Interval : [{result['p5']:.1%},  {result['p95']:.1%}]")
        print(f"  Certainty    : {result['certainty']:.1%}  ({result['certainty_label']})")
        print(f"\n  Primary Driver : {result['primary_driver']}")
        print(f"  Primary Contribution : +{result['primary_driver_contribution']:.1%}")

        if result["other_matches"]:
            print(f"\n  Other Matched Sources:")
            for m in result["other_matches"]:
                print(f"    • {m['source']:<20}  marginal +{m['marginal_contribution']:.1%}")
        else:
            print("\n  No other matched sources.")

        print(f"\n  Active sources : {', '.join(result['active_sources']) or 'None'}")
        print(f"{'='*60}\n")

    @property
    def source_names(self) -> list[str]:
        return self._names

    @property
    def n_sources(self) -> int:
        return self._n_sources

    @property
    def version(self) -> str:
        return self._version

    @property
    def version_date(self) -> str:
        return self._version_date

    def save_config(
        self,
        sources_dict: dict[str, dict],
        output_path: Union[str, Path],
        version: Optional[str] = None,
        n_samples: Optional[int] = None,
        bump: str = "minor",
        risk_bands: Optional[list[dict]] = None,
    ) -> None:
        """
        Build and save an updated JSON config from a plain Python dictionary.

        Parameters
        ----------
        sources_dict : dict mapping source name → {"mu": float, "kappa": float}
        output_path  : where to write the .json file
        version      : explicit version string; if omitted bumped by `bump`
        n_samples    : Monte Carlo sample count; defaults to current value
        bump         : "major", "minor" (default), or "patch"
        risk_bands   : optional list of dicts defining risk band thresholds.
                       Each dict must have "label" (str) and "min_score" (float 0-1).
                       Example:
                           [
                               {"label": "Critical", "min_score": 0.80},
                               {"label": "High",     "min_score": 0.60},
                               {"label": "Medium",   "min_score": 0.40},
                               {"label": "Low",      "min_score": 0.00},
                           ]
                       If None, the existing risk_bands from the loaded config
                       are preserved.  Pass [] to remove all bands.
        """
        # Resolve new version string
        if version:
            new_version = version
        else:
            new_version = self._bump_version(self._version, bump)

        # Build sources list preserving dict order
        sources_list = []
        for name, params in sources_dict.items():
            if "mu" not in params or "kappa" not in params:
                raise ValueError(
                    f"Source '{name}' must have both 'mu' and 'kappa' keys."
                )
            mu    = float(params["mu"])
            kappa = float(params["kappa"])
            self._validate_source_params(mu, kappa, name)
            sources_list.append({"name": name, "mu": mu, "kappa": kappa})

        cfg = {
            "version":               new_version,
            "version_date":          str(date.today()),
            "n_monte_carlo_samples": n_samples or self._n_samples_default,
            "sources":               sources_list,
        }

        # Persist risk_bands: use supplied value, or preserve current bands
        bands_to_save = risk_bands if risk_bands is not None else [
            {"label": lbl, "min_score": thr}
            for thr, lbl in sorted(self._risk_bands, reverse=True)
        ]
        if bands_to_save:
            cfg["risk_bands"] = bands_to_save

        output_path = Path(output_path)
        with open(output_path, "w") as f:
            json.dump(cfg, f, indent=2)

        print(f"Config saved → {output_path}  (version {new_version}, {cfg['version_date']})")

    @staticmethod
    def _bump_version(current: str, part: str) -> str:
        """Increment major, minor, or patch component of a semver string."""
        try:
            major, minor, patch = (int(x) for x in current.split("."))
        except ValueError:
            # If the existing version isn't clean semver, start fresh
            return "1.0.0"

        if part == "major":
            return f"{major + 1}.0.0"
        if part == "minor":
            return f"{major}.{minor + 1}.0"
        return f"{major}.{minor}.{patch + 1}"

    @property
    def risk_bands(self) -> list[tuple[float, str]]:
        """List of (min_score, label) tuples, sorted descending by threshold."""
        return list(self._risk_bands)

    @property
    def batch_pool_size(self) -> int:
        """Number of pre-drawn samples in the batch pool (per source)."""
        return _BATCH_POOL_SIZE

    def __repr__(self) -> str:
        return (
            f"NoisyORModel(v{self._version}, "
            f"n_sources={self._n_sources}, "
            f"n_samples={self._n_samples_default})"
        )
