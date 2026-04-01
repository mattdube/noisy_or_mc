"""
noisy_or_copula.py
==================
Standalone Gaussian copula Noisy-OR Model with Positive Source "Confirmation" logic.
"""

from __future__ import annotations

import json
import math
import warnings
from datetime import date
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False

_BATCH_POOL_SIZE = 50_000
_BATCH_CHUNK_BYTES: int = 512 * 1024 * 1024 

def _certainty_label(cert: float) -> str:
    if cert >= 0.80: return "High"
    if cert >= 0.50: return "Medium"
    return "Low"

def _run_batch_vectorized(pool, source_means, active_matrix, n_samples, rng) -> list[dict]:
    n_records, n_sources = active_matrix.shape
    idx = rng.integers(0, pool.shape[1], size=n_samples)
    pool_slice = pool[:, idx]
    p3d = pool_slice[np.newaxis, :, :] * active_matrix[:, :, np.newaxis]
    combined = 1.0 - np.prod(1.0 - p3d, axis=1)
    mean_risk = np.mean(combined, axis=1)
    median_risk = np.median(combined, axis=1)
    p5, p95 = np.percentile(combined, [5, 95], axis=1)
    certainty = np.clip(1.0 - (p95 - p5), 0.0, None)
    active_mus = source_means[np.newaxis, :] * active_matrix.astype(float)
    prod_all = np.prod(np.where(active_matrix == 0, 1.0, 1.0 - active_mus), axis=1)
    overall_risk = 1.0 - prod_all
    denom = np.where(active_mus > 0.9999, np.inf, 1.0 - active_mus)
    contribs_mat = np.where(active_matrix, overall_risk[:, np.newaxis] - (1.0 - (prod_all[:, np.newaxis] / denom)), 0.0)
    
    results = []
    for r in range(n_records):
        results.append({
            "mean_risk": round(float(mean_risk[r]), 4),
            "median_risk": round(float(median_risk[r]), 4),
            "p5": round(float(p5[r]), 4), "p95": round(float(p95[r]), 4),
            "certainty": round(float(certainty[r]), 4),
            "certainty_label": _certainty_label(float(certainty[r])),
            "source_means": [round(float(m), 4) for m in source_means],
            "marginal_contribs": {i: round(float(contribs_mat[r, i]), 4) for i in range(n_sources) if active_matrix[r, i]}
        })
    return results

def _build_copula_pool(mu_values, kappa_values, corr_matrix, pool_size, rng) -> np.ndarray:
    from scipy.stats import norm, beta as beta_dist
    n = len(mu_values)
    try: L = np.linalg.cholesky(corr_matrix)
    except:
        e, v = np.linalg.eigh(corr_matrix); e = np.clip(e, 1e-8, None)
        L = np.linalg.cholesky(v @ np.diag(e) @ v.T)
    U = norm.cdf(L @ rng.standard_normal((n, pool_size)))
    pool = np.empty((n, pool_size), dtype=np.float32)
    for i, (mu, kappa) in enumerate(zip(mu_values, kappa_values)):
        pool[i] = beta_dist.ppf(np.clip(U[i], 1e-7, 1.0-1e-7), max(mu*kappa, 0.001), max((1-mu)*kappa, 0.001))
    return pool

class NoisyORModelCopula:
    def __init__(self, config_path: Union[str, Path] = "noisy_or_config.json"):
        self._config_path = Path(config_path)
        self._load_from_config(self._config_path)
        self._build_batch_pool()

    def _load_from_config(self, config_path: Path):
        with open(config_path) as f: cfg = json.load(f)
        self._sources = cfg["sources"]
        self._pos_sources = cfg.get("positive_sources", []) # New
        self._n_samples_default = int(cfg.get("n_monte_carlo_samples", 20_000))
        self._version = cfg.get("version", "1.0.0")
        
        self._names = [s["name"] for s in self._sources]
        self._pos_names = [s["name"] for s in self._pos_sources] # New
        self._mu_values = np.array([s["mu"] for s in self._sources], dtype=float)
        self._kappa_values = np.array([s["kappa"] for s in self._sources], dtype=float)
        
        self._n_risk = len(self._sources)
        self._n_pos = len(self._pos_sources)
        self._total_expected = self._n_risk + self._n_pos

        self._risk_bands = sorted([(float(b["min_score"]), str(b["label"])) 
                                   for b in cfg.get("risk_bands", [])], reverse=True)
        self._corr_matrix, self._corr_pairs = self._parse_correlations(cfg.get("correlations", []), self._names)

    @staticmethod
    def _parse_correlations(correlations, names):
        n = len(names); idx = {name: i for i, name in enumerate(names)}
        mat = np.eye(n); pairs = []
        for e in correlations:
            i, j = idx[e["source_a"]], idx[e["source_b"]]
            mat[i, j] = mat[j, i] = float(e["rho"])
            pairs.append(e)
        return mat, pairs

    def _rho_eff_for_active(self, active_indices):
        if len(active_indices) < 2: return 0.0
        return float(np.mean([self._corr_matrix[i, j] for k, i in enumerate(active_indices) for j in active_indices[k+1:]]))

    def _blended_risk(self, active_indices, source_means):
        if not active_indices: return 0.0
        mus = [source_means[i] for i in active_indices]
        if len(mus) == 1: return mus[0]
        rho = self._rho_eff_for_active(active_indices)
        return (1.0 - rho) * (1.0 - math.prod(1.0 - m for m in mus)) + rho * max(mus)

    def predict(self, active_flags: np.ndarray, n_samples=None, seed=None, label=None) -> dict:
        flags = np.asarray(active_flags, dtype=int)
        self._validate_flags(flags)
        
        risk_flags = flags[:self._n_risk]
        pos_flags = flags[self._n_risk:]
        
        # Copula Math
        from scipy.stats import norm as _norm, beta as _beta_dist
        n = n_samples or self._n_samples_default
        rng = np.random.default_rng(seed)
        try: L = np.linalg.cholesky(self._corr_matrix)
        except: 
            e, v = np.linalg.eigh(self._corr_matrix); e = np.clip(e, 1e-8, None)
            L = np.linalg.cholesky(v @ np.diag(e) @ v.T)
        U = np.clip(_norm.cdf(L @ rng.standard_normal((self._n_risk, n))), 1e-7, 1.0-1e-7)
        
        all_probs = [_beta_dist.ppf(U[i], max(self._mu_values[i]*self._kappa_values[i], 0.001), 
                                    max((1-self._mu_values[i])*self._kappa_values[i], 0.001)) for i in range(self._n_risk)]
        
        active_idx = [i for i, a in enumerate(risk_flags) if a]
        if not active_idx: combined = np.zeros(n)
        elif len(active_idx) == 1: combined = all_probs[active_idx[0]].copy()
        else:
            nor = 1.0 - np.prod([1.0 - all_probs[i] for i in active_idx], axis=0)
            rho = self._rho_eff_for_active(active_idx)
            combined = (1.0 - rho) * nor + rho * np.max([all_probs[i] for i in active_idx], axis=0)
            
        p5, p95 = np.percentile(combined, [5, 95])
        source_means = [float(np.mean(p)) for p in all_probs]
        
        sim = {
            "mean_risk": round(float(np.mean(combined)), 4),
            "median_risk": round(float(np.median(combined)), 4),
            "p5": round(float(p5), 4), "p95": round(float(p95), 4),
            "certainty": round(max(0.0, 1.0 - (p95 - p5)), 4),
            "certainty_label": _certainty_label(max(0.0, 1.0 - (p95 - p5))),
            "source_means": [round(m, 4) for m in source_means],
            "marginal_contribs": {i: round(self._blended_risk(active_idx, source_means) - 
                                          self._blended_risk([j for j in active_idx if j != i], source_means), 4) 
                                 for i in active_idx}
        }
        return self._format_single(sim, risk_flags, pos_flags, label)

    def predict_batch(self, active_matrix, n_samples=None, seed=None, labels=None, backend="pandas"):
        mat = np.atleast_2d(np.asarray(active_matrix, dtype=int))
        self._validate_matrix(mat)
        
        risk_mat = mat[:, :self._n_risk]
        pos_mat = mat[:, self._n_risk:]
        
        n = min(n_samples or self._n_samples_default, _BATCH_POOL_SIZE)
        rng = np.random.default_rng(seed)
        
        keys = [tuple(row.tolist()) for row in risk_mat]
        unique_keys = list(dict.fromkeys(keys))
        unique_risk_mat = np.array(unique_keys, dtype=int)
        
        unique_sims = _run_batch_vectorized(self._batch_pool, self._batch_source_means, unique_risk_mat, n, rng)
        
        # Apply Blending to Unique Sims
        pool_slice = self._batch_pool[:, rng.integers(0, _BATCH_POOL_SIZE, size=n)].astype(float)
        blended_lookup = {}
        for row_flags, sim in zip(unique_risk_mat, unique_sims):
            act = [i for i, a in enumerate(row_flags) if a]
            if len(act) > 1 and (rho := self._rho_eff_for_active(act)) > 1e-6:
                p_act = pool_slice[act, :]
                comb = (1.0 - rho) * (1.0 - np.prod(1.0 - p_act, axis=0)) + rho * np.max(p_act, axis=0)
                p5, p95 = np.percentile(comb, [5, 95])
                sim.update({"mean_risk": round(float(np.mean(comb)), 4), "median_risk": round(float(np.median(comb)), 4),
                            "p5": round(float(p5), 4), "p95": round(float(p95), 4), "certainty": round(max(0.0, 1.0-(p95-p5)), 4),
                            "certainty_label": _certainty_label(1.0-(p95-p5))})
            blended_lookup[tuple(row_flags.tolist())] = sim

        labels = labels or [f"Record_{i+1}" for i in range(len(mat))]
        rows = [self._format_row(blended_lookup[keys[i]], risk_mat[i], pos_mat[i], labels[i]) for i in range(len(mat))]
        return self._to_polars(rows) if backend == "polars" else pd.DataFrame(rows)

    def _get_final_label(self, score: float, matched_pos: bool) -> str:
        """New logic for Confirmation label vs Risk Band label."""
        if score > 0:
            for thr, lbl in self._risk_bands:
                if score >= thr: return lbl
            return "Low"
        if matched_pos:
            return "Confirmation"
        return "None"

    def _format_single(self, sim, risk_flags, pos_flags, label) -> dict:
        act_risk = [self._names[i] for i, f in enumerate(risk_flags) if f]
        act_pos = [self._pos_names[i] for i, f in enumerate(pos_flags) if f]
        
        contribs = sim["marginal_contribs"]
        ranked = sorted([i for i, f in enumerate(risk_flags) if f], key=lambda x: contribs.get(x, 0.0), reverse=True)
        
        return {
            "label": label, "risk_score": sim["mean_risk"], "p5": sim["p5"], "p95": sim["p95"],
            "certainty_label": sim["certainty_label"],
            "final_label": self._get_final_label(sim["mean_risk"], any(pos_flags)),
            "primary_driver": self._names[ranked[0]] if ranked else None,
            "active_sources": act_risk, "positive_sources": act_pos,
            "marginal_contribs": contribs, "source_means": sim["source_means"]
        }

    def _format_row(self, sim, risk_flags, pos_flags, label) -> dict:
        res = self._format_single(sim, risk_flags, pos_flags, label)
        row = {"Label": label, "Risk_Score": res["risk_score"], "Final_Label": res["final_label"],
               "Certainty": res["certainty_label"], "Primary_Driver": res["primary_driver"] or ""}
        for i, name in enumerate(self._names):
            row[f"Risk_{name.replace(' ', '_')}"] = int(risk_flags[i])
        for i, name in enumerate(self._pos_names):
            row[f"Pos_{name.replace(' ', '_')}"] = int(pos_flags[i])
        row["All_Risk_Matches"] = ", ".join(res["active_sources"])
        row["All_Pos_Matches"] = ", ".join(res["positive_sources"])
        return row

    def _validate_matrix(self, m):
        if m.shape[1] != self._total_expected: raise ValueError(f"Expected {self._total_expected} columns (Risk+Pos), got {m.shape[1]}")

    def _validate_flags(self, f):
        if f.shape[0] != self._total_expected: raise ValueError(f"Expected {self._total_expected} flags, got {f.shape[0]}")

    def _build_batch_pool(self):
        rng = np.random.default_rng()
        self._batch_pool = _build_copula_pool(self._mu_values, self._kappa_values, self._corr_matrix, _BATCH_POOL_SIZE, rng)
        self._batch_source_means = self._batch_pool.mean(axis=1).astype(float)

    def _to_polars(self, rows):
        if not rows or not _POLARS_AVAILABLE: return pl.DataFrame()
        return pl.DataFrame(rows)

    def export_batch(self, mat, path, **kwargs):
        df = self.predict_batch(mat, **kwargs)
        df.to_csv(path, index=False) if isinstance(df, pd.DataFrame) else df.write_csv(path)
        print(f"Exported to {path}")

    def print_result(self, res):
        print(f"\n--- {res['label'] or 'Result'} ---")
        print(f"Risk Score: {res['risk_score']:.1%}")
        print(f"Final Label: {res['final_label']}")
        if res['active_sources']: print(f"Risk Matches: {', '.join(res['active_sources'])}")
        if res['positive_sources']: print(f"Pos Matches: {', '.join(res['positive_sources'])}")