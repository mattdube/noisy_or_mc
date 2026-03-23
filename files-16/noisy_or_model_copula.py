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
