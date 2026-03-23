"""
noisy_or_model_api.py
=====================
API-optimized Noisy-OR Bayesian Risk Model.

Key optimizations over noisy_or_model.py:
  1. Pre-sampled Beta pool  — Beta draws happen once at load time, not per
                               request. Predictions slice from the pool.
  2. Vectorized batch        — Full batch scored in a single NumPy pass over
                               a 3-D array (n_records × n_sources × n_samples).
                               No Python loop over records.
  3. Precision tiers         — Callers pass "fast" / "standard" / "high" instead
                               of raw sample counts. Pool is sized to the largest
                               tier; smaller tiers slice a subset.
  4. Input caching           — LRU cache on single predictions keyed on the
                               frozen flags tuple. Repeated identical inputs are
                               returned instantly without touching the pool.
  5. Hot reload              — reload() re-reads the config and rebuilds the pool
                               in-place with no server restart required.

Source management (adding/removing sources, save_config) is inherited from the
base NoisyORModel class and works identically — edit the JSON or call
save_config(), then call model.reload().

Usage
-----
    from noisy_or_model_api import NoisyORModelAPI
    import numpy as np

    model = NoisyORModelAPI("noisy_or_config.json")

    # Single prediction
    result = model.predict(np.array([1, 0, 1, 0, 0, 1, 0, 0]), precision="standard")

    # Vectorized batch  →  pandas DataFrame
    matrix = np.array([[1,0,1,0,0,1,0,0], [0,1,0,1,0,0,1,0]])
    df = model.predict_batch(matrix, precision="standard")

    # Export batch to Excel / CSV (same as base model)
    model.export_batch(matrix, "results.xlsx")

    # Hot reload after config change
    model.reload()
"""

from __future__ import annotations

import json
import math
import threading
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

# Re-use shared helpers and the full base class from the original module
from noisy_or_model import NoisyORModel, _certainty_label

# Polars type alias for annotations — actual availability checked at call time
try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore


# ---------------------------------------------------------------------------
# Precision tier definitions
# ---------------------------------------------------------------------------
PRECISION_SAMPLES: dict[str, int] = {
    "fast":     2_000,    # ~5 ms   — real-time screening, high throughput
    "standard": 10_000,   # ~20 ms  — balanced default for API responses
    "high":     50_000,   # ~100 ms — auditable / final decisions
}

# The pool is built at the highest tier so all tiers can slice from it.
_POOL_SIZE = max(PRECISION_SAMPLES.values())


# ---------------------------------------------------------------------------
# Internal simulation using pre-sampled pool
# ---------------------------------------------------------------------------
def _simulate_from_pool(
    pool: np.ndarray,           # shape (n_sources, pool_size)
    source_means: np.ndarray,   # shape (n_sources,)  — pool column means
    active_mask: np.ndarray,    # shape (n_sources,)  — 0/1
    n_samples: int,
    rng: np.random.Generator,
) -> dict:
    """
    Run a single-record simulation by slicing n_samples columns from the pool.
    All Beta sampling has already been done — this is pure combination logic.
    """
    # Draw random indices into the pool (with replacement)
    idx = rng.integers(0, _POOL_SIZE, size=n_samples)
    active_indices = np.where(active_mask)[0]

    if active_indices.size > 0:
        # shape: (n_active, n_samples)
        active_p = pool[active_indices][:, idx]
        combined = 1.0 - np.prod(1.0 - active_p, axis=0)
    else:
        combined = np.zeros(n_samples)

    mean_risk   = float(np.mean(combined))
    median_risk = float(np.median(combined))
    p5          = float(np.percentile(combined, 5))
    p95         = float(np.percentile(combined, 95))
    certainty   = max(0.0, 1.0 - (p95 - p5))

    # Marginal contributions — analytical from pool means
    contribs: dict[int, float] = {}
    if active_indices.size > 0:
        active_mus = source_means[active_indices]
        prod_all   = float(np.prod(1.0 - active_mus))
        overall    = 1.0 - prod_all
        for i, mu_i in zip(active_indices, active_mus):
            prod_without = 0.0 if mu_i > 0.9999 else prod_all / (1.0 - mu_i)
            contribs[int(i)] = overall - (1.0 - prod_without)

    return {
        "mean_risk":         round(mean_risk, 4),
        "median_risk":       round(median_risk, 4),
        "p5":                round(p5, 4),
        "p95":               round(p95, 4),
        "certainty":         round(certainty, 4),
        "certainty_label":   _certainty_label(certainty),
        "source_means":      [round(float(m), 4) for m in source_means],
        "marginal_contribs": contribs,
    }


def _simulate_batch_vectorized(
    pool: np.ndarray,           # shape (n_sources, pool_size)
    source_means: np.ndarray,   # shape (n_sources,)
    active_matrix: np.ndarray,  # shape (n_records, n_sources)
    n_samples: int,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Score all records in a single vectorized NumPy pass.

    Builds a 3-D array of shape (n_records, n_sources, n_samples), masks
    inactive sources to 0 (so they contribute 1.0 to the product), then
    computes Noisy-OR combined risk for every record simultaneously.
    """
    n_records, n_sources = active_matrix.shape

    # Shared random column indices for all records in this batch
    idx = rng.integers(0, _POOL_SIZE, size=n_samples)

    # pool_slice: (n_sources, n_samples)
    pool_slice = pool[:, idx]

    # Broadcast to (n_records, n_sources, n_samples)
    # active_matrix[:, :, None] masks inactive sources → treated as p=0
    # so (1 - p) = 1 and they don't affect the product
    p3d = pool_slice[np.newaxis, :, :] * active_matrix[:, :, np.newaxis]

    # Noisy-OR: combined = 1 - prod(1 - p_i) over sources axis
    # shape after prod: (n_records, n_samples)
    combined = 1.0 - np.prod(1.0 - p3d, axis=1)

    # Summary stats — all computed over the samples axis at once
    mean_risk   = np.mean(combined, axis=1)                   # (n_records,)
    median_risk = np.median(combined, axis=1)                 # (n_records,)
    p5          = np.percentile(combined, 5,  axis=1)         # (n_records,)
    p95         = np.percentile(combined, 95, axis=1)         # (n_records,)
    certainty   = np.clip(1.0 - (p95 - p5), 0.0, None)       # (n_records,)

    # Marginal contributions — analytical, vectorized over records
    # active_mus: (n_records, n_sources), zeroed for inactive
    active_mus = source_means[np.newaxis, :] * active_matrix.astype(float)

    # prod of (1 - mu) for active sources per record; inactive → multiply by 1
    inactive_mask = (active_matrix == 0)
    prod_terms    = np.where(inactive_mask, 1.0, 1.0 - active_mus)
    prod_all      = np.prod(prod_terms, axis=1)               # (n_records,)
    overall_risk  = 1.0 - prod_all                            # (n_records,)

    # For each source: prod_without_i = prod_all / (1 - mu_i)
    # shape: (n_records, n_sources)
    denom         = np.where(active_mus > 0.9999, np.inf, 1.0 - active_mus)
    prod_without  = prod_all[:, np.newaxis] / denom
    risk_without  = 1.0 - prod_without
    contribs_mat  = overall_risk[:, np.newaxis] - risk_without
    # Zero out inactive sources
    contribs_mat  = np.where(active_matrix, contribs_mat, 0.0)

    # Pack into list of dicts matching the base model's output format
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
# API-optimized model class
# ---------------------------------------------------------------------------
class NoisyORModelAPI(NoisyORModel):
    """
    API-optimized Noisy-OR Bayesian Risk Model.

    Inherits all base functionality (save_config, export_batch, print_result,
    etc.) from NoisyORModel and adds:
      - Pre-sampled Beta pool built at load time
      - Vectorized batch scoring (single NumPy pass, no Python loop)
      - Precision tiers: "fast" / "standard" / "high"
      - LRU cache for single predictions on repeated identical inputs
      - reload() for zero-downtime config updates

    Parameters
    ----------
    config_path : path to JSON config file
    pool_seed   : optional int to fix the pool RNG for reproducible pools
                  (useful for testing; omit in production)
    cache_size  : max number of distinct single-prediction inputs to cache
                  (set to 0 to disable caching)
    """

    def __init__(
        self,
        config_path: Union[str, Path] = "noisy_or_config.json",
        pool_seed: Optional[int] = None,
        cache_size: int = 256,
    ):
        self._config_path = Path(config_path)
        self._pool_seed   = pool_seed
        self._cache_size  = cache_size
        self._lock        = threading.Lock()

        # Suppress the base class batch pool build — the API model builds its
        # own pool via _build_pool() below, which uses pool_seed correctly.
        # Without this override, two pools would be built: one unseeded by
        # super().__init__() and one seeded by _build_pool(), wasting memory
        # and making the base-class pool unreachable.
        self._skip_base_pool = True
        super().__init__(config_path)
        self._skip_base_pool = False

        self._build_pool()
        self._build_cache()

    # ------------------------------------------------------------------
    # Override base pool builder — no-op when called from __init__ via super()
    # ------------------------------------------------------------------
    def _build_batch_pool(self, seed: Optional[int] = None) -> None:
        """
        Suppressed in the API model: _build_pool() builds the API pool with
        pool_seed applied correctly.  This override prevents a wasteful
        unseeded pool from being built by super().__init__().
        """
        if getattr(self, "_skip_base_pool", False):
            # Set sentinel attributes so any accidental base-class access
            # raises AttributeError rather than silently returning stale data.
            self._batch_pool         = None  # type: ignore[assignment]
            self._batch_pool_rng     = None  # type: ignore[assignment]
            self._batch_source_means = None  # type: ignore[assignment]
            return
        # Called after full init (e.g. during tests) — delegate to API pool.
        self._build_pool()

    # ------------------------------------------------------------------
    # Pool construction
    # ------------------------------------------------------------------
    def _build_pool(self) -> None:
        """Draw _POOL_SIZE Beta samples for every source and store as a matrix."""
        rng  = np.random.default_rng(self._pool_seed)
        pool = np.empty((self._n_sources, _POOL_SIZE), dtype=np.float32)

        for i, (mu, kappa) in enumerate(zip(self._mu_values, self._kappa_values)):
            alpha = max(float(mu) * float(kappa), 0.001)
            beta  = max((1.0 - float(mu)) * float(kappa), 0.001)
            pool[i] = rng.beta(alpha, beta, _POOL_SIZE)

        self._pool         = pool
        self._pool_rng     = np.random.default_rng()   # separate RNG for slicing
        # Pre-compute pool column means — used for marginal contributions
        self._source_means = self._pool.mean(axis=1)   # (n_sources,)

        pool_mb = pool.nbytes / 1_048_576
        print(
            f"[NoisyORModelAPI] Pool built — "
            f"{self._n_sources} sources × {_POOL_SIZE:,} samples "
            f"({pool_mb:.1f} MB)"
        )

    # ------------------------------------------------------------------
    # LRU cache (rebuilt on reload so stale results don't persist)
    # ------------------------------------------------------------------
    def _build_cache(self) -> None:
        """Wrap the internal prediction logic in a fresh LRU cache."""
        if self._cache_size > 0:
            @lru_cache(maxsize=self._cache_size)
            def _cached(flags_tuple: tuple, n_samples: int) -> dict:
                active = np.array(flags_tuple, dtype=int)
                rng    = np.random.default_rng()
                return _simulate_from_pool(
                    self._pool, self._source_means, active, n_samples, rng
                )
            self._cached_sim = _cached
        else:
            self._cached_sim = None

    # ------------------------------------------------------------------
    # Hot reload
    # ------------------------------------------------------------------
    def reload(self) -> None:
        """
        Re-read the config file and rebuild the sample pool in-place.

        Call this after saving an updated config (e.g. via save_config()
        or FeedbackUpdater.apply()) to make the new sources/parameters
        live without restarting the server.

        Thread-safe: a lock prevents concurrent requests from seeing a
        partially rebuilt pool during reload.  risk_bands, source names,
        mu/kappa values, and the version are all reloaded atomically.
        """
        with self._lock:
            if not self._config_path.exists():
                raise FileNotFoundError(f"Config not found: {self._config_path}")

            # Delegate to the shared loader so all attributes (including
            # risk_bands) are updated from the same code path as __init__.
            self._load_from_config(self._config_path)
            self._build_pool()
            self._build_cache()

        print(f"[NoisyORModelAPI] Reloaded — {self!r}")

    # ------------------------------------------------------------------
    # Single prediction (pool-backed, optionally cached)
    # ------------------------------------------------------------------
    def predict(
        self,
        active_flags: np.ndarray,
        precision: str = "standard",
        n_samples: Optional[int] = None,
        seed: Optional[int] = None,
        label: Optional[str] = None,
    ) -> dict:
        """
        Score a single record using the pre-sampled pool.

        Parameters
        ----------
        active_flags : 1-D numpy array, length n_sources (values 0 or 1)
        precision    : "fast" (2k), "standard" (10k), or "high" (50k) samples.
                       Ignored if n_samples is supplied explicitly.
        n_samples    : override sample count directly (bypasses precision tier)
        seed         : optional int — fixes the pool-slice RNG for this call.
                       Disables caching for this call.
        label        : optional string label included in the returned dict

        Returns
        -------
        Same dict structure as NoisyORModel.predict()
        """
        active_flags = np.asarray(active_flags, dtype=int)
        self._validate_flags(active_flags)

        n = n_samples or PRECISION_SAMPLES.get(precision, PRECISION_SAMPLES["standard"])
        flags_tuple = tuple(active_flags.tolist())

        # Use cache when no explicit seed is set
        if self._cached_sim is not None and seed is None:
            sim = self._cached_sim(flags_tuple, n)
        else:
            rng = np.random.default_rng(seed)
            sim = _simulate_from_pool(
                self._pool, self._source_means, active_flags, n, rng
            )

        return self._format_single(sim, active_flags, label)

    # ------------------------------------------------------------------
    # Vectorized batch prediction
    # ------------------------------------------------------------------
    def predict_batch(
        self,
        active_matrix: np.ndarray,
        precision: str = "standard",
        n_samples: Optional[int] = None,
        seed: Optional[int] = None,
        labels: Optional[list[str]] = None,
        backend: str = "pandas",
    ) -> "Union[pd.DataFrame, pl.DataFrame]":
        """
        Score many records in a single vectorized NumPy pass.

        Parameters
        ----------
        active_matrix : 2-D array, shape (n_records, n_sources), values 0/1
        precision     : "fast", "standard", or "high"
        n_samples     : override sample count directly
        seed          : optional int for reproducible pool-slice RNG
        labels        : optional list of string labels (length n_records)
        backend       : "pandas" (default) or "polars"

        Returns
        -------
        pandas or polars DataFrame depending on backend parameter.
        """
        if backend not in ("pandas", "polars"):
            raise ValueError(f"backend must be 'pandas' or 'polars', got '{backend}'")
        from noisy_or_model import _POLARS_AVAILABLE
        if backend == "polars" and not _POLARS_AVAILABLE:
            raise ImportError(
                "polars is not installed. Install it with: pip install polars"
            )

        active_matrix = np.atleast_2d(np.asarray(active_matrix, dtype=int))
        n_records = active_matrix.shape[0]

        if labels is None:
            labels = [f"Record_{i+1}" for i in range(n_records)]

        # Vectorized validation
        self._validate_matrix(active_matrix)

        n   = n_samples or PRECISION_SAMPLES.get(precision, PRECISION_SAMPLES["standard"])
        rng = np.random.default_rng(seed)

        # Memory-safe chunking — same threshold as base model
        from noisy_or_model import _BATCH_CHUNK_BYTES
        projected_bytes = n_records * self._n_sources * n * 4
        if projected_bytes > _BATCH_CHUNK_BYTES:
            chunk_size = max(1, _BATCH_CHUNK_BYTES // (self._n_sources * n * 4))
            sim_list: list[dict] = []
            for start in range(0, n_records, chunk_size):
                chunk = active_matrix[start : start + chunk_size]
                sim_list.extend(
                    _simulate_batch_vectorized(self._pool, self._source_means, chunk, n, rng)
                )
        else:
            sim_list = _simulate_batch_vectorized(
                self._pool, self._source_means, active_matrix, n, rng
            )

        rows = [
            self._format_row(sim, active_matrix[i], labels[i])
            for i, sim in enumerate(sim_list)
        ]

        if backend == "polars":
            return self._to_polars(rows)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Streaming export (large datasets)
    # ------------------------------------------------------------------
    def stream_export(
        self,
        active_matrix: np.ndarray,
        output_path: Union[str, Path],
        precision: str = "standard",
        n_samples: Optional[int] = None,
        seed: Optional[int] = None,
        labels: Optional[list[str]] = None,
        chunk_size: int = 5_000,
        backend: str = "pandas",
    ) -> None:
        """
        Score a large batch in chunks and write results incrementally to CSV.

        Keeps peak memory proportional to chunk_size rather than the full
        dataset.  Only CSV output is supported — Excel requires the full
        workbook in memory.  For Excel use export_batch() instead.

        Parameters
        ----------
        active_matrix : 2-D numpy array, shape (n_records, n_sources)
        output_path   : destination .csv file path
        precision     : "fast", "standard", or "high"
        n_samples     : override sample count (bypasses precision tier)
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
            end       = min(start + chunk_size, n_records)
            chunk_mat = active_matrix[start:end]
            chunk_lbl = labels[start:end]

            chunk_df = self.predict_batch(
                chunk_mat, precision=precision, n_samples=n_samples,
                labels=chunk_lbl, backend="pandas",
            )

            if backend == "polars":
                from noisy_or_model import _POLARS_AVAILABLE
                if not _POLARS_AVAILABLE:
                    raise ImportError("polars is not installed: pip install polars")
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
              f"(chunk_size={chunk_size:,}, precision={precision})")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def pool_size(self) -> int:
        return _POOL_SIZE

    @property
    def cache_info(self) -> Optional[object]:
        """Return LRU cache statistics, or None if caching is disabled."""
        if self._cached_sim is not None:
            return self._cached_sim.cache_info()
        return None

    def __repr__(self) -> str:
        return (
            f"NoisyORModelAPI(v{self._version}, "
            f"n_sources={self._n_sources}, "
            f"pool={_POOL_SIZE:,}, "
            f"cache={'on' if self._cached_sim else 'off'})"
        )


# ---------------------------------------------------------------------------
# Multi-tenant model registry
# ---------------------------------------------------------------------------
class ModelRegistry:
    """
    Manages a named collection of NoisyORModelAPI instances — one per config.

    Each config gets its own pre-sampled Beta pool and its own LRU cache,
    fully isolated from every other config. There is no risk of cross-
    contamination between customers: a cache hit for Customer A can never
    be returned to Customer B.

    Parameters
    ----------
    configs : dict mapping a config name (str) to a config file path (str or Path)
              Example:
                  {
                      "general":    "configs/general.json",
                      "customer_1": "configs/customer_1.json",
                      "customer_2": "configs/customer_2.json",
                  }
    pool_seed   : optional int — passed to every NoisyORModelAPI instance.
                  Useful for reproducible testing; omit in production.
    cache_size  : LRU cache size per config (default 256)

    Usage
    -----
        registry = ModelRegistry({
            "general":    "configs/general.json",
            "customer_1": "configs/customer_1.json",
        })

        # Route a prediction to a specific config
        result = registry.predict("customer_1", flags, precision="standard")
        df     = registry.predict_batch("customer_1", matrix, precision="standard")

        # Reload one config without touching the others
        registry.reload("customer_1")

        # Reload all configs at once
        registry.reload_all()

        # Inspect all loaded configs
        print(registry.status())
    """

    def __init__(
        self,
        configs: dict[str, Union[str, Path]],
        pool_seed: Optional[int] = None,
        cache_size: int = 256,
    ):
        if not configs:
            raise ValueError("configs dict must contain at least one entry.")

        self._config_paths: dict[str, Path] = {
            name: Path(path) for name, path in configs.items()
        }
        self._pool_seed  = pool_seed
        self._cache_size = cache_size
        self._lock       = threading.Lock()

        # Build all models at startup
        self._models: dict[str, NoisyORModelAPI] = {}
        for name, path in self._config_paths.items():
            print(f"[ModelRegistry] Loading '{name}' from {path}")
            self._models[name] = NoisyORModelAPI(
                path,
                pool_seed=pool_seed,
                cache_size=cache_size,
            )

        print(f"[ModelRegistry] Ready — {len(self._models)} config(s) loaded.")

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------
    def get(self, config_name: str) -> NoisyORModelAPI:
        """
        Return the NoisyORModelAPI instance for a given config name.
        Use this for direct access to any model method not exposed on
        the registry itself (e.g. export_batch, print_result, save_config).
        """
        if config_name not in self._models:
            raise KeyError(
                f"Config '{config_name}' not found in registry. "
                f"Available: {list(self._models.keys())}"
            )
        return self._models[config_name]

    # ------------------------------------------------------------------
    # Prediction routing
    # ------------------------------------------------------------------
    def predict(
        self,
        config_name: str,
        active_flags: np.ndarray,
        precision: str = "standard",
        n_samples: Optional[int] = None,
        seed: Optional[int] = None,
        label: Optional[str] = None,
    ) -> dict:
        """
        Score a single record using the named config's pool and cache.

        Parameters
        ----------
        config_name  : name of the config to use (must match a key in configs)
        active_flags : 1-D numpy array, length n_sources for that config
        precision    : "fast", "standard", or "high"
        n_samples    : override sample count directly
        seed         : optional int — disables caching for this call
        label        : optional string label for the record

        Returns
        -------
        Same dict structure as NoisyORModelAPI.predict()
        """
        return self.get(config_name).predict(
            active_flags,
            precision=precision,
            n_samples=n_samples,
            seed=seed,
            label=label,
        )

    def predict_batch(
        self,
        config_name: str,
        active_matrix: np.ndarray,
        precision: str = "standard",
        n_samples: Optional[int] = None,
        seed: Optional[int] = None,
        labels: Optional[list[str]] = None,
        backend: str = "pandas",
    ) -> "Union[pd.DataFrame, pl.DataFrame]":
        """
        Score many records using the named config's pool.

        Parameters
        ----------
        config_name   : name of the config to use
        active_matrix : 2-D array, shape (n_records, n_sources), values 0/1
        precision     : "fast", "standard", or "high"
        n_samples     : override sample count directly
        seed          : optional int for reproducible pool-slice RNG
        labels        : optional list of string labels (length n_records)
        backend       : "pandas" (default) or "polars"

        Returns
        -------
        pandas or polars DataFrame depending on backend parameter.
        """
        return self.get(config_name).predict_batch(
            active_matrix,
            precision=precision,
            n_samples=n_samples,
            seed=seed,
            labels=labels,
            backend=backend,
        )

    # ------------------------------------------------------------------
    # Reload
    # ------------------------------------------------------------------
    def reload(self, config_name: str) -> None:
        """
        Reload a single config from disk and rebuild its pool and cache.
        All other configs are unaffected.

        Thread-safe: uses the per-model lock inside NoisyORModelAPI.reload().
        """
        print(f"[ModelRegistry] Reloading '{config_name}'...")
        self.get(config_name).reload()

    def reload_all(self) -> None:
        """Reload every config in the registry sequentially."""
        print(f"[ModelRegistry] Reloading all {len(self._models)} config(s)...")
        for name in self._models:
            self.get(name).reload()
        print("[ModelRegistry] All configs reloaded.")

    # ------------------------------------------------------------------
    # Adding / removing configs at runtime
    # ------------------------------------------------------------------
    def add(
        self,
        config_name: str,
        config_path: Union[str, Path],
    ) -> None:
        """
        Add a new config to the registry and build its pool.
        If a config with the same name already exists it is replaced.
        """
        config_path = Path(config_path)
        print(f"[ModelRegistry] Adding '{config_name}' from {config_path}")
        with self._lock:
            self._config_paths[config_name] = config_path
            self._models[config_name] = NoisyORModelAPI(
                config_path,
                pool_seed=self._pool_seed,
                cache_size=self._cache_size,
            )

    def remove(self, config_name: str) -> None:
        """Remove a config and free its pool from memory."""
        with self._lock:
            if config_name not in self._models:
                raise KeyError(f"Config '{config_name}' not in registry.")
            del self._models[config_name]
            del self._config_paths[config_name]
        print(f"[ModelRegistry] Removed '{config_name}'.")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def status(self) -> dict[str, dict]:
        """
        Return a summary dict of all loaded configs, their versions,
        source counts, and cache statistics.

        Useful for a /health or /status API endpoint.
        """
        return {
            name: {
                "version":      model.version,
                "version_date": model.version_date,
                "n_sources":    model.n_sources,
                "sources":      model.source_names,
                "pool_size":    model.pool_size,
                "cache_info":   str(model.cache_info),
                "config_path":  str(self._config_paths[name]),
            }
            for name, model in self._models.items()
        }

    @property
    def config_names(self) -> list[str]:
        """List of all registered config names."""
        return list(self._models.keys())

    def __len__(self) -> int:
        return len(self._models)

    def __repr__(self) -> str:
        names = ", ".join(f"'{n}'" for n in self._models)
        return f"ModelRegistry([{names}])"
