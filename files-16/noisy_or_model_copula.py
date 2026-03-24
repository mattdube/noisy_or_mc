"""
noisy_or_model_copula.py
========================
Gaussian copula extension of the Noisy-OR Bayesian Risk Model.

Why this matters
----------------
The standard NoisyORModel draws each source's Beta probabilities
independently.  This is the Noisy-OR independence assumption — given a
record, knowing that Source 1 fired tells you nothing extra about whether
Source 2 will fire.

In practice many risk sources are correlated:

  • A watchlist hit and adverse media often co-occur for the same entity.
  • A PEP match and high-risk geography frequently appear together.
  • Two transaction anomaly signals from the same underlying pattern will
    both fire or both miss.

When correlated sources are treated as independent, the combined risk is
*overestimated* for cases where those sources co-fire (the model treats two
signals as two independent pieces of evidence when they are really one).

The Gaussian copula approach
----------------------------
A copula separates the marginal distributions (each source's Beta) from
their dependence structure (how they move together).  A Gaussian copula
uses a multivariate normal distribution to induce the correlation:

  1. Draw Z ~ MVN(0, I)  of shape (n_sources, pool_size)
  2. Apply Cholesky factor: Z_corr = L @ Z  where Σ = L L^T
  3. Map to uniform: U_i = Φ(Z_corr_i)  (standard normal CDF)
  4. Map to Beta: p_i = Beta_ppf(U_i; alpha_i, beta_i)

Result: p_i has the correct Beta(alpha_i, beta_i) marginal but samples
across sources are correlated according to the correlation matrix Σ.

A Pearson correlation of ρ = 0.0 reproduces the independent model exactly.
ρ = 0.8 means the two sources tend to produce high probabilities together.

Config format
-------------
Add a "correlations" block to your JSON config.  Unspecified pairs default
to ρ = 0.0 (independent).

    {
      "version": "1.0.0",
      "n_monte_carlo_samples": 20000,
      "sources": [
        {"name": "Watchlist Hit",  "mu": 0.70, "kappa": 6.0},
        {"name": "Adverse Media",  "mu": 0.40, "kappa": 8.0},
        {"name": "PEP Match",      "mu": 0.55, "kappa": 5.0}
      ],
      "correlations": [
        {"source_a": "Watchlist Hit", "source_b": "Adverse Media", "rho": 0.65},
        {"source_a": "Watchlist Hit", "source_b": "PEP Match",     "rho": 0.45},
        {"source_a": "Adverse Media", "source_b": "PEP Match",     "rho": 0.30}
      ]
    }

Usage
-----
    from noisy_or_model_copula import NoisyORModelCopula
    import numpy as np

    model = NoisyORModelCopula("noisy_or_config_copula.json")
    print(model)
    # NoisyORModelCopula(v1.0.0, n_sources=3, n_samples=20000, corr_pairs=3)

    # All standard methods work identically
    result = model.predict(np.array([1, 1, 1]))
    model.print_result(result)

    df = model.predict_batch(np.array([[1,1,0],[1,0,1],[0,0,0]]))

    # Extra copula-specific properties
    print(model.correlation_matrix)
    print(model.correlated_pairs)

    # The FeedbackUpdater works unchanged — correlations live in the config
    from noisy_or_updater import FeedbackUpdater
    updater = FeedbackUpdater("noisy_or_config_copula.json")
    updater.apply("feedback.csv")
    model.reload()   # picks up updated mu/kappa; correlation structure unchanged

Dependencies
------------
scipy is required for the Beta inverse CDF (percent-point function):

    pip install scipy

If scipy is not available, the model falls back to independent sampling
with a RuntimeWarning and behaves identically to NoisyORModel.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np

from noisy_or_model import NoisyORModel, _BATCH_POOL_SIZE


# ---------------------------------------------------------------------------
# Copula pool builder
# ---------------------------------------------------------------------------

def _build_copula_pool(
    mu_values: np.ndarray,
    kappa_values: np.ndarray,
    corr_matrix: np.ndarray,
    pool_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw pool_size correlated Beta samples using a Gaussian copula.

    Returns
    -------
    pool : shape (n_sources, pool_size), dtype float32
    """
    from scipy.stats import norm, beta as beta_dist

    n_sources = len(mu_values)

    # Cholesky decomposition — validated PSD at load time
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        # Nearest PSD via eigenvalue clipping (handles floating-point near-singularity)
        eigvals, eigvecs = np.linalg.eigh(corr_matrix)
        eigvals = np.clip(eigvals, 1e-8, None)
        L = np.linalg.cholesky(eigvecs @ np.diag(eigvals) @ eigvecs.T)

    # Correlated standard normals via Cholesky factor
    Z      = rng.standard_normal((n_sources, pool_size))
    Z_corr = L @ Z                        # (n_sources, pool_size)

    # Map to uniform [0,1] via standard normal CDF
    U = norm.cdf(Z_corr)                  # (n_sources, pool_size)

    # Map each source's uniform draws to its Beta marginal
    pool = np.empty((n_sources, pool_size), dtype=np.float32)
    for i, (mu, kappa) in enumerate(zip(mu_values, kappa_values)):
        alpha     = max(float(mu) * float(kappa), 0.001)
        b         = max((1.0 - float(mu)) * float(kappa), 0.001)
        u_clipped = np.clip(U[i], 1e-7, 1.0 - 1e-7)   # avoid ppf returning ±inf
        pool[i]   = beta_dist.ppf(u_clipped, alpha, b).astype(np.float32)

    return pool


# ---------------------------------------------------------------------------
# Public copula model class
# ---------------------------------------------------------------------------

class NoisyORModelCopula(NoisyORModel):
    """
    Noisy-OR Bayesian Risk Model with Gaussian copula source correlations.

    All methods from NoisyORModel (predict, predict_batch, export_batch,
    stream_export, reload, save_config, print_result, etc.) are available
    unchanged.  The only internal difference is how the Beta sample pool
    is built: correlated draws replace independent sampling.

    Parameters
    ----------
    config_path : path to a JSON config that may contain a "correlations"
                  block (see module docstring for format).

    If no "correlations" block is present, this model behaves identically
    to the standard NoisyORModel (all rho values default to 0.0).
    """

    # ------------------------------------------------------------------
    # Config loading — extended to parse correlations
    # ------------------------------------------------------------------

    def _load_from_config(self, config_path: Path) -> None:
        """Extend base loader to parse the correlations block."""
        super()._load_from_config(config_path)

        with open(config_path) as f:
            cfg = json.load(f)

        self._corr_matrix, self._corr_pairs = self._parse_correlations(
            cfg.get("correlations", []),
            self._names,
        )

    @staticmethod
    def _parse_correlations(
        correlations: list[dict],
        source_names: list[str],
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Build and validate the full (n_sources, n_sources) correlation matrix
        from a sparse list of (source_a, source_b, rho) entries.

        Raises ValueError if a source name is unknown, rho is outside (-1,1),
        or the resulting matrix is not positive semi-definite.
        """
        n   = len(source_names)
        idx = {name: i for i, name in enumerate(source_names)}
        matrix = np.eye(n, dtype=float)
        pairs: list[dict] = []

        for entry in correlations:
            a   = entry["source_a"]
            b   = entry["source_b"]
            rho = float(entry["rho"])

            if a not in idx:
                raise ValueError(
                    f"Correlation references unknown source '{a}'. "
                    f"Known sources: {source_names}"
                )
            if b not in idx:
                raise ValueError(
                    f"Correlation references unknown source '{b}'. "
                    f"Known sources: {source_names}"
                )
            if not (-1.0 < rho < 1.0):
                raise ValueError(
                    f"rho={rho} for ('{a}', '{b}') must be strictly "
                    f"between -1 and 1."
                )

            i, j = idx[a], idx[b]
            matrix[i, j] = rho
            matrix[j, i] = rho
            pairs.append({"source_a": a, "source_b": b, "rho": rho})

        # Positive semi-definiteness check
        eigvals = np.linalg.eigvalsh(matrix)
        if eigvals.min() < -1e-6:
            raise ValueError(
                f"The supplied correlations produce a matrix that is not "
                f"positive semi-definite (min eigenvalue = {eigvals.min():.4f}). "
                f"Reduce some rho values to make the correlation structure consistent."
            )

        return matrix, pairs

    # ------------------------------------------------------------------
    # Effective correlation for a set of active sources
    # ------------------------------------------------------------------

    def _rho_eff_for_active(self, active_indices: list[int]) -> float:
        """
        Compute the mean pairwise correlation across the active sources.

        This is used to blend between Noisy-OR (rho_eff=0) and max
        (rho_eff=1) in the combination step.  When only one source is
        active the blending weight is 0 and Noisy-OR is used unchanged.
        """
        if len(active_indices) < 2:
            return 0.0
        pairs = [
            self._corr_matrix[i, j]
            for k, i in enumerate(active_indices)
            for j in active_indices[k+1:]
        ]
        return float(np.mean(pairs))

    def _blended_risk(self, active_indices: list[int], source_means: list[float]) -> float:
        """
        Compute the blended combined risk for a given set of active source
        indices using their sample means and the correlation matrix.

        This is used to evaluate marginal contributions consistently with
        the blended combination rule.  The same blend used for the full
        prediction is applied when computing 'risk without source i'.
        """
        if not active_indices:
            return 0.0
        if len(active_indices) == 1:
            return source_means[active_indices[0]]

        mus     = [source_means[i] for i in active_indices]
        nor     = 1.0 - float(np.prod([1.0 - m for m in mus]))
        mx      = max(mus)
        rho_eff = self._rho_eff_for_active(active_indices)
        return (1.0 - rho_eff) * nor + rho_eff * mx

    def _blended_marginals(
        self,
        active_indices: list[int],
        source_means: list[float],
    ) -> dict[int, float]:
        """
        Compute marginal contributions consistent with the blended combination
        rule.  For each active source i:

            c_i = R_blended(all active) − R_blended(all active except i)

        When rho_eff = 0 this reduces to the standard Noisy-OR analytical
        formula.  When rho_eff → 1 the contributions correctly reflect
        near-redundancy: the marginal value of adding a correlated source
        on top of an already-matched one approaches zero.
        """
        if not active_indices:
            return {}
        R_all    = self._blended_risk(active_indices, source_means)
        contribs = {}
        for idx in active_indices:
            without       = [i for i in active_indices if i != idx]
            R_without     = self._blended_risk(without, source_means)
            contribs[idx] = round(R_all - R_without, 4)
        return contribs

    # ------------------------------------------------------------------
    # Single prediction — correlated sampling + redundancy-aware blending
    # ------------------------------------------------------------------

    def predict(
        self,
        active_flags,
        n_samples=None,
        seed=None,
        label=None,
    ):
        """
        Run a single correlated Noisy-OR prediction.

        Two corrections are applied over the standard model:

        1. **Correlated sampling** — Beta samples are drawn jointly via a
           Gaussian copula so that highly-correlated sources tend to produce
           high or low values together, rather than independently.

        2. **Redundancy-aware blending** — the combination rule blends
           between Noisy-OR (ρ_eff=0, full independence) and the maximum of
           active source probabilities (ρ_eff→1, full redundancy):

               combined = (1 − ρ_eff) × noisy_or(p_i…) + ρ_eff × max(p_i…)

           At ρ=0 this is identical to standard Noisy-OR.  At ρ→1 the
           combined risk approaches the strongest individual source, correctly
           reflecting that highly-correlated sources carry nearly redundant
           information and should not be double-counted.

           ρ_eff is the mean pairwise correlation across active sources,
           read from the correlation matrix.
        """
        import math as _math  # kept for potential future use
        from scipy.stats import norm as _norm, beta as _beta_dist

        active_flags = np.asarray(active_flags, dtype=int)
        self._validate_flags(active_flags)

        n   = n_samples or self._n_samples_default
        rng = np.random.default_rng(seed)

        # --- Correlated Beta sampling (Gaussian copula) ---
        Z = rng.standard_normal((self._n_sources, n))
        try:
            L = np.linalg.cholesky(self._corr_matrix)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(self._corr_matrix)
            eigvals = np.clip(eigvals, 1e-8, None)
            L = np.linalg.cholesky(eigvecs @ np.diag(eigvals) @ eigvecs.T)

        Z_corr = L @ Z
        U      = np.clip(_norm.cdf(Z_corr), 1e-7, 1.0 - 1e-7)

        all_probs = []
        for i, (mu, kappa) in enumerate(zip(self._mu_values, self._kappa_values)):
            alpha = max(float(mu) * float(kappa), 0.001)
            beta  = max((1.0 - float(mu)) * float(kappa), 0.001)
            all_probs.append(_beta_dist.ppf(U[i], alpha, beta))

        # --- Blended combination (Noisy-OR ↔ max, weighted by ρ_eff) ---
        active_indices = [i for i, a in enumerate(active_flags) if a]
        active_probs   = [all_probs[i] for i in active_indices]

        if not active_probs:
            combined = np.zeros(n)
        elif len(active_probs) == 1:
            combined = active_probs[0].copy()
        else:
            noisy_or = 1.0 - np.prod([1.0 - p for p in active_probs], axis=0)
            mx       = np.max(active_probs, axis=0)
            rho_eff  = self._rho_eff_for_active(active_indices)
            combined = (1.0 - rho_eff) * noisy_or + rho_eff * mx

        mean_risk   = float(np.mean(combined))
        median_risk = float(np.median(combined))
        p5          = float(np.percentile(combined, 5))
        p95         = float(np.percentile(combined, 95))
        certainty   = max(0.0, 1.0 - (p95 - p5))

        # --- Marginal contributions consistent with blended combination ---
        # c_i = R_blended(all active) − R_blended(all active except i)
        # This uses _blended_risk() so the "risk without source i" is also
        # evaluated under the same blending rule (not standard Noisy-OR).
        source_means = [float(np.mean(p)) for p in all_probs]

        contribs: dict[int, float] = {}
        if active_indices:
            R_all = self._blended_risk(active_indices, source_means)
            for idx in active_indices:
                without = [i for i in active_indices if i != idx]
                R_without = self._blended_risk(without, source_means)
                contribs[idx] = round(R_all - R_without, 4)

        from noisy_or_model import _certainty_label
        sim = {
            "mean_risk":         round(mean_risk,   4),
            "median_risk":       round(median_risk, 4),
            "p5":                round(p5,          4),
            "p95":               round(p95,         4),
            "certainty":         round(certainty,   4),
            "certainty_label":   _certainty_label(certainty),
            "source_means":      [round(m, 4) for m in source_means],
            "marginal_contribs": contribs,
        }
        return self._format_single(sim, active_flags, label)

    # ------------------------------------------------------------------
    # Batch prediction — blended combination applied per-record
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        active_matrix,
        n_samples=None,
        seed=None,
        labels=None,
        backend="pandas",
    ):
        """
        Batch prediction with copula sampling and redundancy-aware blending.

        After drawing correlated pool samples, the Noisy-OR combination for
        each record is blended toward the per-record max using that record's
        effective pairwise correlation across its active sources:

            combined_r = (1 − ρ_eff_r) × noisy_or_r + ρ_eff_r × max_r

        Records with no correlation between active sources (ρ_eff=0) get
        standard Noisy-OR.  Records where all active sources are highly
        correlated (ρ_eff→1) get a result approaching the strongest source.
        """
        import pandas as pd
        from noisy_or_model import (
            _run_batch_vectorized, _BATCH_POOL_SIZE, _BATCH_CHUNK_BYTES,
            _POLARS_AVAILABLE, _certainty_label,
        )

        if backend not in ("pandas", "polars"):
            raise ValueError(f"backend must be 'pandas' or 'polars', got '{backend}'")
        if backend == "polars" and not _POLARS_AVAILABLE:
            raise ImportError("polars is not installed: pip install polars")

        active_matrix = np.atleast_2d(np.asarray(active_matrix, dtype=int))
        n_records = active_matrix.shape[0]
        n = min(n_samples or self._n_samples_default, _BATCH_POOL_SIZE)

        if labels is None:
            labels = [f"Record_{i+1}" for i in range(n_records)]

        self._validate_matrix(active_matrix)
        rng = np.random.default_rng(seed)

        # Deduplication
        keys        = [tuple(row.tolist()) for row in active_matrix]
        unique_keys = list(dict.fromkeys(keys))
        unique_matrix = np.array(unique_keys, dtype=int)

        # Vectorized pool-backed scoring (Noisy-OR only, same as base model)
        projected_bytes = unique_matrix.shape[0] * self._n_sources * n * 4
        if projected_bytes > _BATCH_CHUNK_BYTES:
            chunk_size = max(1, _BATCH_CHUNK_BYTES // (self._n_sources * n * 4))
            unique_sims: list[dict] = []
            for start in range(0, unique_matrix.shape[0], chunk_size):
                unique_sims.extend(_run_batch_vectorized(
                    self._batch_pool, self._batch_source_means,
                    unique_matrix[start:start+chunk_size], n, rng,
                ))
        else:
            unique_sims = _run_batch_vectorized(
                self._batch_pool, self._batch_source_means, unique_matrix, n, rng,
            )

        # --- Apply blended combination per unique row ---
        # The vectorized scorer already computed Noisy-OR mean_risk.
        # We need to re-blend using per-row pool samples.
        # For each unique flag pattern, compute rho_eff and blend.

        # Slice the same pool columns the scorer used (approximate — we use
        # a fresh slice of the same pool, which has the correct statistics)
        idx        = rng.integers(0, _BATCH_POOL_SIZE, size=n)
        pool_slice = self._batch_pool[:, idx].astype(float)  # (n_sources, n)

        blended_sims: list[dict] = []
        for r, (row_flags, sim) in enumerate(zip(unique_matrix, unique_sims)):
            active_idx = [i for i, a in enumerate(row_flags) if a]

            if len(active_idx) < 2:
                # Solo or zero active: Noisy-OR and max are identical
                blended_sims.append(sim)
                continue

            rho_eff = self._rho_eff_for_active(active_idx)
            if rho_eff < 1e-6:
                # All active sources are independent — no blending needed
                blended_sims.append(sim)
                continue

            # Re-compute combined using the blended rule on pool samples
            active_p  = pool_slice[active_idx, :]        # (n_active, n)
            noisy_or  = 1.0 - np.prod(1.0 - active_p, axis=0)  # (n,)
            mx        = np.max(active_p, axis=0)                # (n,)
            combined  = (1.0 - rho_eff) * noisy_or + rho_eff * mx

            mean_risk   = round(float(np.mean(combined)),            4)
            median_risk = round(float(np.median(combined)),          4)
            p5          = round(float(np.percentile(combined, 5)),   4)
            p95         = round(float(np.percentile(combined, 95)),  4)
            certainty   = round(max(0.0, 1.0 - (p95 - p5)),         4)

            blended_sims.append({
                **sim,
                "mean_risk":         mean_risk,
                "median_risk":       median_risk,
                "p5":                p5,
                "p95":               p95,
                "certainty":         certainty,
                "certainty_label":   _certainty_label(certainty),
                "marginal_contribs": self._blended_marginals(
                    active_idx, sim["source_means"]
                ),
            })

        sim_lookup = {k: blended_sims[i] for i, k in enumerate(unique_keys)}
        rows = [
            self._format_row(sim_lookup[key], active_matrix[i], labels[i])
            for i, key in enumerate(keys)
        ]

        if backend == "polars":
            return self._to_polars(rows)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Pool construction — uses copula instead of independent sampling
    # ------------------------------------------------------------------

    def _build_batch_pool(self, seed: Optional[int] = None) -> None:
        """
        Build the pre-sampled pool using the Gaussian copula.

        Falls back to independent sampling (standard model behaviour) if:
          - scipy is not installed, or
          - no correlations are configured (all rho == 0)
        """
        try:
            import scipy  # noqa: F401
            _scipy_ok = True
        except ImportError:
            _scipy_ok = False

        has_corr = (
            hasattr(self, "_corr_matrix") and
            not np.allclose(self._corr_matrix, np.eye(self._n_sources))
        )

        if not _scipy_ok:
            if has_corr:
                warnings.warn(
                    "scipy is not installed — falling back to independent sampling "
                    "(correlations will be ignored).  Install with: pip install scipy",
                    RuntimeWarning,
                    stacklevel=3,
                )
            super()._build_batch_pool(seed=seed)
            return

        if not has_corr:
            # No correlations specified — use the faster independent path
            super()._build_batch_pool(seed=seed)
            return

        rng  = np.random.default_rng(seed)
        pool = _build_copula_pool(
            self._mu_values,
            self._kappa_values,
            self._corr_matrix,
            _BATCH_POOL_SIZE,
            rng,
        )

        self._batch_pool         = pool
        self._batch_pool_rng     = np.random.default_rng()
        self._batch_source_means = pool.mean(axis=1).astype(float)

    # ------------------------------------------------------------------
    # Extra properties
    # ------------------------------------------------------------------

    @property
    def correlation_matrix(self) -> np.ndarray:
        """
        Full (n_sources, n_sources) Pearson correlation matrix.
        Diagonal is 1.0.  Unspecified pairs are 0.0 (independent).
        Returns a copy — modifying it does not affect the model.
        """
        return self._corr_matrix.copy()

    @property
    def correlated_pairs(self) -> list[dict]:
        """
        List of explicitly configured correlation pairs.
        Each dict contains "source_a", "source_b", "rho".
        """
        return list(self._corr_pairs)

    @property
    def n_correlated_pairs(self) -> int:
        """Number of explicitly specified source-pair correlations."""
        return len(self._corr_pairs)

    def print_correlations(self) -> None:
        """Pretty-print the correlation matrix and pair list."""
        n = self._n_sources
        max_name = max(len(nm) for nm in self._names)
        header_width = max_name + 2

        print(f"\nCorrelation matrix ({n}×{n}):")
        print(" " * header_width + "  ".join(
            f"{nm[:6]:>6}" for nm in self._names
        ))
        for i, name in enumerate(self._names):
            row = "  ".join(
                f"{self._corr_matrix[i, j]:>6.2f}" for j in range(n)
            )
            print(f"  {name:<{max_name}}  {row}")

        if self._corr_pairs:
            print(f"\nConfigured correlations ({len(self._corr_pairs)}):")
            for p in sorted(self._corr_pairs, key=lambda x: -abs(x["rho"])):
                bar_len = int(abs(p["rho"]) * 20)
                bar     = "█" * bar_len
                sign    = "+" if p["rho"] >= 0 else "-"
                print(f"  {p['source_a']:<20} ↔ {p['source_b']:<20}  "
                      f"ρ = {sign}{abs(p['rho']):.2f}  {bar}")
        else:
            print("\nNo correlations configured — all sources are independent.")
        print()

    def __repr__(self) -> str:
        return (
            f"NoisyORModelCopula(v{self._version}, "
            f"n_sources={self._n_sources}, "
            f"n_samples={self._n_samples_default}, "
            f"corr_pairs={self.n_correlated_pairs})"
        )

    # ------------------------------------------------------------------
    # save_config — persists correlations block alongside sources
    # ------------------------------------------------------------------

    def save_config(
        self,
        sources_dict: dict[str, dict],
        output_path: Union[str, Path],
        version: Optional[str] = None,
        n_samples: Optional[int] = None,
        bump: str = "minor",
        risk_bands: Optional[list[dict]] = None,
        correlations: Optional[list[dict]] = None,
    ) -> None:
        """
        Save an updated config, preserving or replacing the correlations block.

        Parameters
        ----------
        correlations : list of {"source_a": str, "source_b": str, "rho": float}.
                       If None, the current correlations are preserved.
                       Pass [] to remove all correlations from the saved config.
        All other parameters are identical to NoisyORModel.save_config().
        """
        # Parent handles sources, version, risk_bands, and writes the file
        super().save_config(
            sources_dict=sources_dict,
            output_path=output_path,
            version=version,
            n_samples=n_samples,
            bump=bump,
            risk_bands=risk_bands,
        )

        # Re-open and inject the correlations block
        output_path = Path(output_path)
        with open(output_path) as f:
            cfg = json.load(f)

        corr_to_save = (
            correlations if correlations is not None
            else [{"source_a": p["source_a"],
                   "source_b": p["source_b"],
                   "rho":      p["rho"]}
                  for p in self._corr_pairs]
        )

        if corr_to_save:
            cfg["correlations"] = corr_to_save
        elif "correlations" in cfg:
            del cfg["correlations"]

        with open(output_path, "w") as f:
            json.dump(cfg, f, indent=2)
