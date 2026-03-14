"""
noisy_or_updater.py
===================
Bayesian feedback updater and file watcher for NoisyORModel configs.

Overview
--------
Each source in the model has a Beta(alpha, beta) prior parameterised by:
  mu    = prior mean   = alpha / (alpha + beta)
  kappa = sample size  = alpha + beta  (concentration / confidence)

When feedback arrives (confirmed outcomes for cases), we perform an exact
Bayesian update using the Beta conjugate prior:
  alpha_new = alpha_old + n_positive
  beta_new  = beta_old  + n_negative

This translates back to updated mu and kappa:
  kappa_new = kappa_old + n_positive + n_negative
  mu_new    = (alpha_old + n_positive) / kappa_new

The update is additive and commutative: multiple feedback files can be
applied in any order and the result is the same as processing them all at once.

---

Two feedback formats are supported
-----------------------------------

FORMAT 1 — Source-level (simple, pre-attributed)
  Use when you already know which specific source caused the outcome, or
  when cases had only a single source active.

  Required columns: source_name, outcome
  Optional columns: weight

  Example:
    source_name,outcome,weight
    Source 1,1,1.0
    Source 3,1,2.0
    Source 6,0,1.0

FORMAT 2 — Case-level (attributed, for multi-source cases)
  Use when multiple sources fired on the same case and you are uncertain
  which source actually drove the outcome.  The updater computes each
  source's attribution weight analytically from the current model
  parameters and distributes the feedback proportionally.

  Required columns: active_sources, outcome
    active_sources  comma-separated source names that fired on this case
    outcome         1 = confirmed risk, 0 = confirmed non-risk

  Optional columns: weight, attribution_method

  Example:
    active_sources,outcome,weight,attribution_method
    "Source 1,Source 3,Source 6",1,1.0,marginal
    "Source 2,Source 4",0,1.0,marginal

  attribution_method controls how credit is divided among active sources:
    "marginal"  (default) — each source is credited proportionally to its
                marginal contribution to the combined Noisy-OR risk score.
                This is the theoretically correct Bayesian attribution.
    "uniform"   — each active source receives equal credit (1/n_active).
                Use this when you have no basis for distinguishing sources.
    "mu"        — each source is credited proportionally to its current mu.
                Simpler than marginal but ignores inter-source interactions.

  See _compute_attribution() for the full mathematics.

---

Attribution mathematics (FORMAT 2)
------------------------------------
For a case where sources {i, j, k} fired, let:
  R   = combined Noisy-OR risk  = 1 - prod(1 - mu_s) for s in active
  c_i = marginal contribution of source i
      = R - (1 - prod(1 - mu_s) for s != i)

For a confirmed POSITIVE outcome, source i receives fractional credit:
  w_i = c_i / R   (marginal attribution weight)

Each active source i then receives:
  n_positive[i] += w_i * case_weight
  n_negative[i] += 0

Intuition: if R = 90% and source i contributed 60pp of that, it gets
credit for 60/90 ≈ 67% of the confirmed positive observation.

For a confirmed NEGATIVE outcome (false alarm — sources fired but the
case turned out not to be risky), source i was responsible for its share
of the false positive signal, so it receives negative feedback:
  n_positive[i] += 0
  n_negative[i] += w_i * case_weight

The attribution weights always sum to 1 across active sources, so the
total evidence added to the model equals case_weight regardless of how
many sources fired.

When R ≈ 0 (all source mu values are very low), marginal attribution
degenerates.  In that case the updater falls back to uniform weighting
with a warning.

---

Audit trail
-----------
Every update is appended to `feedback_log.jsonl` in the same directory as
the config.  Each line is a JSON object:

    {
      "timestamp":     "2026-03-11T14:23:01",
      "feedback_file": "feedback/march_batch.csv",
      "config_path":   "noisy_or_config.json",
      "version_before": "1.0.0",
      "version_after":  "1.1.0",
      "bump":           "patch",
      "n_records":      42,
      "changes": {
        "Source 1": {"mu_before": 0.60, "mu_after": 0.612,
                     "kappa_before": 5.0, "kappa_after": 8.3,
                     "n_positive": 3.7, "n_negative": 1.2},
        ...
      }
    }

A timestamped backup of the config is written to a `config_backups/`
subdirectory before every update.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Union

# Cross-platform file locking — filelock is preferred (pip install filelock).
# Falls back to fcntl (Unix) or a threading.Lock-only approach on Windows.
try:
    from filelock import FileLock as _FileLock
    def _config_lock(path: Path):
        return _FileLock(str(path) + ".lock")
except ImportError:
    try:
        import fcntl as _fcntl
        import contextlib

        @contextlib.contextmanager
        def _config_lock(path: Path):
            lock_path = str(path) + ".lock"
            with open(lock_path, "w") as lf:
                _fcntl.flock(lf, _fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    _fcntl.flock(lf, _fcntl.LOCK_UN)
    except ImportError:
        # Windows without filelock — process-level lock only
        import contextlib
        _LOCK = threading.Lock()

        @contextlib.contextmanager
        def _config_lock(path: Path):
            with _LOCK:
                yield

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core Bayesian update math
# ---------------------------------------------------------------------------

def _beta_params(mu: float, kappa: float) -> tuple[float, float]:
    """Convert (mu, kappa) to (alpha, beta)."""
    alpha = max(mu * kappa, 1e-6)
    beta  = max((1.0 - mu) * kappa, 1e-6)
    return alpha, beta


def _mu_kappa(alpha: float, beta: float) -> tuple[float, float]:
    """Convert (alpha, beta) back to (mu, kappa), rounded to 6 d.p."""
    kappa = alpha + beta
    mu    = alpha / kappa
    return round(mu, 6), round(kappa, 6)


def _apply_bayesian_update(
    mu: float,
    kappa: float,
    n_positive: float,
    n_negative: float,
) -> tuple[float, float]:
    """
    Exact Beta conjugate update.

    Parameters
    ----------
    mu, kappa    : current prior parameters
    n_positive   : weighted count of confirmed positive outcomes
    n_negative   : weighted count of confirmed negative outcomes

    Returns
    -------
    (mu_new, kappa_new)
    """
    alpha, beta = _beta_params(mu, kappa)
    alpha_new   = alpha + n_positive
    beta_new    = beta  + n_negative
    return _mu_kappa(alpha_new, beta_new)


# ---------------------------------------------------------------------------
# Config integrity helpers
# ---------------------------------------------------------------------------

def _config_checksum(path: Path) -> str:
    """Return a 12-hex-char SHA-256 digest of the config file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _validate_source_params(mu: float, kappa: float, name: str) -> None:
    """
    Raise ValueError if mu or kappa are outside valid ranges.
    Called before writing an updated config so degenerate Beta distributions
    can never be silently committed.
    """
    if not (0.0 < mu < 1.0):
        raise ValueError(
            f"Source '{name}': mu={mu:.6f} must be strictly between 0 and 1. "
            "This can happen with extreme feedback weights — check your input."
        )
    if kappa <= 0.0:
        raise ValueError(
            f"Source '{name}': kappa={kappa:.6f} must be greater than 0."
        )

def _compute_attribution(
    active_source_names: list[str],
    source_mus: dict[str, float],
    method: str = "marginal",
) -> dict[str, float]:
    """
    Compute the attribution weight for each active source on a single case.

    Weights always sum to 1.0 across the active sources, so the total
    evidence added to the model equals case_weight regardless of how many
    sources fired.

    Parameters
    ----------
    active_source_names : names of sources that fired on this case
    source_mus          : current mu value for every source in the config
    method              : "marginal" | "uniform" | "mu"

    Returns
    -------
    dict mapping source_name → attribution_weight (floats summing to 1.0)
    """
    n = len(active_source_names)
    if n == 0:
        return {}
    if n == 1:
        return {active_source_names[0]: 1.0}

    mus = [source_mus[s] for s in active_source_names]

    if method == "uniform":
        w = 1.0 / n
        return {s: w for s in active_source_names}

    if method == "mu":
        total = sum(mus)
        if total < 1e-9:
            # All mus are effectively zero — fall back to uniform
            w = 1.0 / n
            return {s: w for s in active_source_names}
        return {s: mu / total for s, mu in zip(active_source_names, mus)}

    # method == "marginal" (default)
    # Combined Noisy-OR risk from all active sources
    prod_all   = 1.0
    for mu in mus:
        prod_all *= (1.0 - mu)
    combined_risk = 1.0 - prod_all

    if combined_risk < 1e-9:
        # Degenerate case: all mus near zero, marginal attribution undefined
        logger.warning(
            "Combined risk is near zero for sources %s; "
            "falling back to uniform attribution.", active_source_names
        )
        w = 1.0 / n
        return {s: w for s in active_source_names}

    weights: dict[str, float] = {}
    for s, mu_i in zip(active_source_names, mus):
        if mu_i >= 1.0 - 1e-9:
            # Source is near-certain — its marginal contribution is everything
            # that wouldn't exist without it; approximate as prod_all → 0
            prod_without = 0.0
        else:
            prod_without = prod_all / (1.0 - mu_i)
        risk_without   = 1.0 - prod_without
        marginal_contrib = combined_risk - risk_without
        weights[s] = max(marginal_contrib, 0.0)

    total_weight = sum(weights.values())
    if total_weight < 1e-9:
        # Edge case: all marginal contributions are zero (e.g. perfectly
        # correlated sources at identical mu values)
        w = 1.0 / n
        return {s: w for s in active_source_names}

    # Normalise to sum to 1.0
    return {s: w / total_weight for s, w in weights.items()}


# ---------------------------------------------------------------------------
# Case-level feedback file parsing (FORMAT 2)
# ---------------------------------------------------------------------------

def _detect_format(fieldnames: list[str]) -> str:
    """
    Return "case" if the header contains 'active_sources',
    "source" if it contains 'source_name', else raise ValueError.
    """
    lower = {h.lower().strip() for h in fieldnames}
    if "active_sources" in lower:
        return "case"
    if "source_name" in lower:
        return "source"
    raise ValueError(
        "Feedback file must have either a 'source_name' column (source-level "
        "format) or an 'active_sources' column (case-level format). "
        f"Found columns: {fieldnames}"
    )


def _parse_case_feedback_csv(
    path: Union[str, Path],
    known_sources: set[str],
    source_mus: dict[str, float],
    default_method: str = "marginal",
) -> dict[str, dict[str, float]]:
    """
    Parse a case-level feedback CSV and return per-source weighted counts,
    distributing each case's evidence across its active sources according
    to their attribution weights.

    CSV format
    ----------
    Required: active_sources  (comma-separated source names),  outcome (0/1)
    Optional: weight (float, default 1.0),
              attribution_method ("marginal" | "uniform" | "mu",
                                   default = default_method arg)

    Returns
    -------
    dict mapping source_name → {"n_positive": float, "n_negative": float}
    """
    path   = Path(path)
    counts: dict[str, dict[str, float]] = defaultdict(
        lambda: {"n_positive": 0.0, "n_negative": 0.0}
    )
    skipped = 0

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Feedback file is empty or has no header: {path}")

        headers_lower = {h.lower().strip(): h for h in reader.fieldnames}
        missing = {"active_sources", "outcome"} - set(headers_lower)
        if missing:
            raise ValueError(
                f"Case-level feedback file {path} is missing columns: {missing}"
            )

        src_col    = headers_lower["active_sources"]
        out_col    = headers_lower["outcome"]
        wt_col     = headers_lower.get("weight")
        method_col = headers_lower.get("attribution_method")

        for row_num, row in enumerate(reader, start=2):
            # Parse outcome
            try:
                outcome = round(float(row[out_col].strip()))
            except ValueError:
                logger.warning("Row %d: cannot parse outcome — skipping.", row_num)
                skipped += 1
                continue
            if outcome not in (0, 1):
                logger.warning("Row %d: outcome not 0 or 1 — skipping.", row_num)
                skipped += 1
                continue

            # Parse weight
            try:
                weight = float(row[wt_col].strip()) if wt_col else 1.0
            except ValueError:
                logger.warning("Row %d: cannot parse weight — skipping.", row_num)
                skipped += 1
                continue
            if weight <= 0:
                logger.warning("Row %d: weight non-positive — skipping.", row_num)
                skipped += 1
                continue

            # Parse attribution method for this row
            method = default_method
            if method_col:
                m = row[method_col].strip().lower()
                if m in ("marginal", "uniform", "mu"):
                    method = m
                elif m:
                    logger.warning(
                        "Row %d: unknown attribution_method '%s' — using '%s'.",
                        row_num, m, default_method
                    )

            # Parse active source names
            raw_sources = [s.strip() for s in row[src_col].split(",") if s.strip()]
            active = []
            for s in raw_sources:
                if s in known_sources:
                    active.append(s)
                else:
                    logger.warning(
                        "Row %d: unknown source '%s' — excluded from attribution.",
                        row_num, s
                    )
            if not active:
                logger.warning("Row %d: no valid sources — skipping.", row_num)
                skipped += 1
                continue

            # Compute attribution weights and distribute evidence
            attribution = _compute_attribution(active, source_mus, method)

            for source, attr_weight in attribution.items():
                effective = attr_weight * weight
                if outcome == 1:
                    counts[source]["n_positive"] += effective
                else:
                    counts[source]["n_negative"] += effective

    if skipped:
        logger.warning("%d row(s) skipped in %s", skipped, path.name)

    return dict(counts)


# ---------------------------------------------------------------------------
# Source-level feedback file parsing (FORMAT 1)
# ---------------------------------------------------------------------------

def _parse_feedback_csv(
    path: Union[str, Path],
    known_sources: set[str],
) -> dict[str, dict[str, float]]:
    """
    Parse a feedback CSV and return per-source weighted counts.

    Returns
    -------
    dict mapping source_name → {"n_positive": float, "n_negative": float}
    Only sources present in known_sources are included.
    """
    path = Path(path)
    counts: dict[str, dict[str, float]] = defaultdict(
        lambda: {"n_positive": 0.0, "n_negative": 0.0}
    )
    skipped = 0

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # Normalise header names to lower-case for flexible matching
        if reader.fieldnames is None:
            raise ValueError(f"Feedback file is empty or has no header: {path}")

        headers_lower = {h.lower().strip(): h for h in reader.fieldnames}
        required = {"source_name", "outcome"}
        missing  = required - set(headers_lower)
        if missing:
            raise ValueError(
                f"Feedback file {path} is missing required columns: {missing}. "
                f"Found: {list(reader.fieldnames)}"
            )

        src_col     = headers_lower["source_name"]
        outcome_col = headers_lower["outcome"]
        weight_col  = headers_lower.get("weight")

        for i, row in enumerate(reader, start=2):
            source  = row[src_col].strip()
            outcome_raw = row[outcome_col].strip()
            weight  = float(row[weight_col].strip()) if weight_col else 1.0

            if source not in known_sources:
                logger.warning(
                    "Row %d: unknown source '%s' — skipping. "
                    "Known sources: %s", i, source, sorted(known_sources)
                )
                skipped += 1
                continue

            try:
                outcome = round(float(outcome_raw))
            except ValueError:
                logger.warning(
                    "Row %d: cannot parse outcome '%s' — skipping.", i, outcome_raw
                )
                skipped += 1
                continue

            if outcome not in (0, 1):
                logger.warning(
                    "Row %d: outcome %s is not 0 or 1 — skipping.", i, outcome
                )
                skipped += 1
                continue

            if weight <= 0:
                logger.warning(
                    "Row %d: weight %.4f is non-positive — skipping.", i, weight
                )
                skipped += 1
                continue

            if outcome == 1:
                counts[source]["n_positive"] += weight
            else:
                counts[source]["n_negative"] += weight

    if skipped:
        logger.warning("%d row(s) skipped in %s", skipped, path.name)

    return dict(counts)


# ---------------------------------------------------------------------------
# FeedbackUpdater
# ---------------------------------------------------------------------------

class FeedbackUpdater:
    """
    Applies Bayesian feedback updates to a NoisyORModel config file.

    Accepts two CSV formats automatically detected from column headers:

    FORMAT 1 — Source-level (pre-attributed):
      Columns: source_name, outcome[, weight]
      Use when cases had only one active source, or when attribution has
      already been determined externally.

    FORMAT 2 — Case-level (attributed):
      Columns: active_sources, outcome[, weight][, attribution_method]
      Use when multiple sources fired on the same case and you need the
      model to distribute credit across them proportionally.  Each source
      receives a fractional observation weighted by its contribution to
      the combined Noisy-OR risk score.

    Parameters
    ----------
    config_path         : path to the JSON config to update
    bump                : semver component to increment on each update —
                          "patch" (default), "minor", or "major"
    backup_dir          : directory for config backups; defaults to
                          <config_dir>/config_backups/
    log_path            : path to the JSONL audit log; defaults to
                          <config_dir>/feedback_log.jsonl
    min_counts          : minimum total weighted observations required
                          before a source's parameters are updated (default
                          1.0).  Increase this to require more evidence.
    attribution_method  : default attribution method for case-level files —
                          "marginal" (default), "uniform", or "mu".
                          Can be overridden per-row via the
                          attribution_method column in the CSV.
    """

    def __init__(
        self,
        config_path: Union[str, Path],
        bump: str = "patch",
        backup_dir: Optional[Union[str, Path]] = None,
        log_path: Optional[Union[str, Path]] = None,
        min_counts: float = 1.0,
        attribution_method: str = "marginal",
    ):
        self.config_path        = Path(config_path).resolve()
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        self.bump               = bump
        self.min_counts         = min_counts
        self.attribution_method = attribution_method

        cfg_dir = self.config_path.parent
        self.backup_dir = Path(backup_dir).resolve() if backup_dir else cfg_dir / "config_backups"
        self.log_path   = Path(log_path).resolve()   if log_path   else cfg_dir / "feedback_log.jsonl"

        self.backup_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(
        self,
        feedback_path: Union[str, Path],
        dry_run: bool = False,
        bump: Optional[str] = None,
        attribution_method: Optional[str] = None,
    ) -> dict:
        """
        Parse a feedback CSV and update the config with Bayesian posteriors.

        Automatically detects FORMAT 1 (source-level) or FORMAT 2
        (case-level attributed) from the CSV column headers.

        Parameters
        ----------
        feedback_path      : path to the feedback CSV file
        dry_run            : if True, compute updates but do not write any files
        bump               : override the instance-level bump setting for this call
        attribution_method : override the instance-level attribution method for
                             this call ("marginal", "uniform", or "mu").
                             Only used for FORMAT 2 (case-level) files.

        Returns
        -------
        dict with keys:
            version_before, version_after, n_records, n_sources_updated,
            format          ("source" or "case")
            changes         (dict: source_name → {mu_before, mu_after,
                                                   kappa_before, kappa_after,
                                                   n_positive, n_negative})
            skipped_sources (list of sources with insufficient observations)
            dry_run         (bool)
        """
        feedback_path = Path(feedback_path).resolve()
        if not feedback_path.exists():
            raise FileNotFoundError(f"Feedback file not found: {feedback_path}")

        cfg     = self._load_config()
        sources = {s["name"]: s for s in cfg["sources"]}
        source_mus = {s["name"]: s["mu"] for s in cfg["sources"]}

        # Auto-detect format and parse
        method = attribution_method or self.attribution_method
        fmt, counts = self._parse(feedback_path, set(sources), source_mus, method)

        # Compute updates
        bump_to_use = bump or self.bump
        changes, skipped = self._compute_changes(sources, counts)

        n_records = sum(v["n_positive"] + v["n_negative"] for v in counts.values())

        if not changes:
            logger.info("No sources met the minimum observation threshold (%s). "
                        "Config unchanged.", self.min_counts)
            return {
                "version_before":    cfg["version"],
                "version_after":     cfg["version"],
                "n_records":         n_records,
                "n_sources_updated": 0,
                "format":            fmt,
                "changes":           {},
                "skipped_sources":   skipped,
                "dry_run":           dry_run,
            }

        version_before = cfg["version"]
        version_after  = self._bump_version(version_before, bump_to_use)

        if not dry_run:
            with _config_lock(self.config_path):
                # Re-read inside the lock so we apply on top of the latest state,
                # not the snapshot taken before we acquired the lock.
                cfg     = self._load_config()
                sources = {s["name"]: s for s in cfg["sources"]}
                source_mus = {s["name"]: s["mu"] for s in cfg["sources"]}
                fmt, counts = self._parse(
                    feedback_path, set(sources), source_mus, method
                )
                changes, skipped = self._compute_changes(sources, counts)
                n_records = sum(
                    v["n_positive"] + v["n_negative"] for v in counts.values()
                )
                if not changes:
                    return {
                        "version_before":    cfg["version"],
                        "version_after":     cfg["version"],
                        "n_records":         n_records,
                        "n_sources_updated": 0,
                        "format":            fmt,
                        "changes":           {},
                        "skipped_sources":   skipped,
                        "dry_run":           False,
                    }
                version_before = cfg["version"]
                version_after  = self._bump_version(version_before, bump_to_use)

                checksum_before = _config_checksum(self.config_path)
                self._backup_config(cfg, version_before)

                for src in cfg["sources"]:
                    if src["name"] in changes:
                        src["mu"]    = changes[src["name"]]["mu_after"]
                        src["kappa"] = changes[src["name"]]["kappa_after"]

                cfg["version"]      = version_after
                cfg["version_date"] = str(datetime.today().date())

                with open(self.config_path, "w") as f:
                    json.dump(cfg, f, indent=2)

                checksum_after = _config_checksum(self.config_path)

                self._log_update(
                    feedback_path     = feedback_path,
                    version_before    = version_before,
                    version_after     = version_after,
                    bump              = bump_to_use,
                    n_records         = n_records,
                    changes           = changes,
                    fmt               = fmt,
                    checksum_before   = checksum_before,
                    checksum_after    = checksum_after,
                    attribution_used  = method,
                )

            print(
                f"Config updated: {self.config_path.name}  "
                f"v{version_before} → v{version_after}  "
                f"({len(changes)} source(s) updated, {n_records:.1f} observations, "
                f"format={fmt})"
            )
        else:
            print(
                f"[DRY RUN] Would update: {self.config_path.name}  "
                f"v{version_before} → v{version_after}  "
                f"({len(changes)} source(s), {n_records:.1f} observations, "
                f"format={fmt})"
            )

        return {
            "version_before":    version_before,
            "version_after":     version_after,
            "n_records":         n_records,
            "n_sources_updated": len(changes),
            "format":            fmt,
            "changes":           changes,
            "skipped_sources":   skipped,
            "dry_run":           dry_run,
        }

    def apply_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.csv",
        dry_run: bool = False,
        bump_per_file: bool = False,
    ) -> list[dict]:
        """
        Apply all matching CSVs in a directory that have not yet been processed.

        Already-processed files are tracked in the audit log. A file is
        considered processed if its resolved path appears in any log entry.

        Parameters
        ----------
        directory     : directory to scan
        pattern       : glob pattern for feedback files (default "*.csv")
        dry_run       : preview updates without writing
        bump_per_file : if True, bump the version for each file individually;
                        if False (default), a single bump covers all files
                        processed in this call

        Returns
        -------
        list of summary dicts, one per file processed (in discovery order)
        """
        directory = Path(directory).resolve()
        files     = sorted(directory.glob(pattern))

        if not files:
            logger.info("No %s files found in %s", pattern, directory)
            return []

        processed = self._processed_files()
        pending   = [f for f in files if str(f) not in processed]

        if not pending:
            logger.info("All %d file(s) in %s have already been processed.",
                        len(files), directory)
            return []

        print(f"Processing {len(pending)} new feedback file(s) in {directory} ...")

        summaries = []
        for i, fp in enumerate(pending):
            # For single-bump mode: only bump on the last file
            effective_dry_run = dry_run
            if not bump_per_file and not dry_run and i < len(pending) - 1:
                # Apply intermediate files without version bump; last one bumps
                summaries.append(self._apply_no_bump(fp))
            else:
                summaries.append(self.apply(fp, dry_run=effective_dry_run))

        return summaries

    def preview(self, feedback_path: Union[str, Path]) -> dict:
        """Convenience alias for apply(..., dry_run=True)."""
        return self.apply(feedback_path, dry_run=True)

    def print_summary(self, summary: dict) -> None:
        """Pretty-print the result of apply() or preview()."""
        tag = "[DRY RUN] " if summary["dry_run"] else ""
        print(f"\n{'='*62}")
        print(f"  {tag}FEEDBACK UPDATE SUMMARY")
        print(f"{'='*62}")
        print(f"  Version    : {summary['version_before']} → {summary['version_after']}")
        print(f"  Observations: {summary['n_records']:.1f}")
        print(f"  Sources updated: {summary['n_sources_updated']}")

        if summary["changes"]:
            print(f"\n  {'Source':<22} {'mu before':>10} {'mu after':>10} "
                  f"{'kappa before':>13} {'kappa after':>11}")
            print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*13} {'-'*11}")
            for name, ch in summary["changes"].items():
                delta_mu = ch["mu_after"] - ch["mu_before"]
                arrow    = "▲" if delta_mu > 0 else ("▼" if delta_mu < 0 else "─")
                print(f"  {name:<22} {ch['mu_before']:>10.4f} "
                      f"{ch['mu_after']:>10.4f} {arrow}  "
                      f"{ch['kappa_before']:>11.2f}  {ch['kappa_after']:>10.2f}  "
                      f"  (+{ch['n_positive']:.1f} / -{ch['n_negative']:.1f})")

        if summary["skipped_sources"]:
            print(f"\n  Skipped (< {self.min_counts} observations): "
                  f"{', '.join(summary['skipped_sources'])}")

        print(f"{'='*62}\n")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_config(self) -> dict:
        with open(self.config_path) as f:
            return json.load(f)

    def _parse(
        self,
        feedback_path: Path,
        known_sources: set[str],
        source_mus: dict[str, float],
        attribution_method: str,
    ) -> tuple[str, dict]:
        """
        Auto-detect feedback format and return (format_str, counts_dict).
        Peeks at the header row to choose the parser.
        """
        with open(feedback_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"Feedback file is empty or has no header: {feedback_path}")
            fmt = _detect_format(list(reader.fieldnames))

        if fmt == "case":
            counts = _parse_case_feedback_csv(
                feedback_path, known_sources, source_mus, attribution_method
            )
        else:
            counts = _parse_feedback_csv(feedback_path, known_sources)

        return fmt, counts

    def _compute_changes(
        self,
        sources: dict[str, dict],
        counts: dict[str, dict[str, float]],
    ) -> tuple[dict, list[str]]:
        """
        Compute updated (mu, kappa) for each source that has enough observations.
        Validates updated parameters before accepting them.
        Returns (changes_dict, skipped_list).
        """
        changes: dict[str, dict] = {}
        skipped: list[str] = []

        for name, c in counts.items():
            total = c["n_positive"] + c["n_negative"]
            if total < self.min_counts:
                skipped.append(name)
                continue

            src = sources[name]
            mu_before    = src["mu"]
            kappa_before = src["kappa"]

            mu_after, kappa_after = _apply_bayesian_update(
                mu_before, kappa_before, c["n_positive"], c["n_negative"]
            )

            # Validate before accepting — guards against extreme weights
            # producing out-of-range parameters
            try:
                _validate_source_params(mu_after, kappa_after, name)
            except ValueError as e:
                logger.error(
                    "Skipping update for '%s': resulting parameters are invalid. "
                    "(%s)  Consider reducing the weight for this source's feedback.",
                    name, e,
                )
                skipped.append(name)
                continue

            changes[name] = {
                "mu_before":    mu_before,
                "mu_after":     mu_after,
                "kappa_before": kappa_before,
                "kappa_after":  kappa_after,
                "n_positive":   c["n_positive"],
                "n_negative":   c["n_negative"],
            }

        return changes, skipped

    def _apply_no_bump(self, feedback_path: Path) -> dict:
        """
        Apply a feedback file without bumping the version.
        Used internally for apply_directory() when bump_per_file=False.
        Acquires the config file lock before reading or writing.
        """
        feedback_path = Path(feedback_path).resolve()

        with _config_lock(self.config_path):
            cfg        = self._load_config()
            sources    = {s["name"]: s for s in cfg["sources"]}
            source_mus = {s["name"]: s["mu"] for s in cfg["sources"]}
            fmt, counts = self._parse(
                feedback_path, set(sources), source_mus, self.attribution_method
            )
            changes, skipped = self._compute_changes(sources, counts)

            n_records = sum(v["n_positive"] + v["n_negative"] for v in counts.values())
            version   = cfg["version"]

            if changes:
                checksum_before = _config_checksum(self.config_path)

                for src in cfg["sources"]:
                    if src["name"] in changes:
                        src["mu"]    = changes[src["name"]]["mu_after"]
                        src["kappa"] = changes[src["name"]]["kappa_after"]

                with open(self.config_path, "w") as f:
                    json.dump(cfg, f, indent=2)

                checksum_after = _config_checksum(self.config_path)

                self._log_update(
                    feedback_path    = feedback_path,
                    version_before   = version,
                    version_after    = version,
                    bump             = "none",
                    n_records        = n_records,
                    changes          = changes,
                    fmt              = fmt,
                    checksum_before  = checksum_before,
                    checksum_after   = checksum_after,
                    attribution_used = self.attribution_method,
                )
                print(f"  Applied {feedback_path.name}  ({n_records:.1f} obs, "
                      f"{len(changes)} source(s) updated — version held at {version}, "
                      f"format={fmt})")

        return {
            "version_before":    version,
            "version_after":     version,
            "n_records":         n_records,
            "n_sources_updated": len(changes),
            "format":            fmt,
            "changes":           changes,
            "skipped_sources":   skipped,
            "dry_run":           False,
        }

    def _backup_config(self, cfg: dict, version: str) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem      = self.config_path.stem
        backup    = self.backup_dir / f"{stem}_v{version}_{timestamp}.json"
        shutil.copy2(self.config_path, backup)
        logger.debug("Config backed up → %s", backup)

    def _log_update(
        self,
        feedback_path: Path,
        version_before: str,
        version_after: str,
        bump: str,
        n_records: float,
        changes: dict,
        fmt: str = "source",
        checksum_before: str = "",
        checksum_after: str = "",
        attribution_used: str = "n/a",
    ) -> None:
        entry = {
            "timestamp":        datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "feedback_file":    str(feedback_path),
            "config_path":      str(self.config_path),
            "version_before":   version_before,
            "version_after":    version_after,
            "bump":             bump,
            "format":           fmt,
            "attribution":      attribution_used,
            "n_records":        n_records,
            "checksum_before":  checksum_before,
            "checksum_after":   checksum_after,
            "changes":          changes,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _processed_files(self) -> set[str]:
        """Return set of feedback file paths already in the audit log."""
        if not self.log_path.exists():
            return set()
        paths = set()
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        paths.add(json.loads(line)["feedback_file"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        return paths

    @staticmethod
    def _bump_version(current: str, part: str) -> str:
        try:
            major, minor, patch = (int(x) for x in current.split("."))
        except ValueError:
            return "1.0.0"
        if part == "major":
            return f"{major + 1}.0.0"
        if part == "minor":
            return f"{major}.{minor + 1}.0"
        return f"{major}.{minor}.{patch + 1}"


# ---------------------------------------------------------------------------
# FeedbackWatcher  — automatic processing when new feedback files arrive
# ---------------------------------------------------------------------------

class FeedbackWatcher:
    """
    Watches a directory for new feedback CSV files and applies updates
    automatically as they arrive.

    Uses `watchdog` for instant file-system events if installed, or falls
    back to a polling loop (stdlib only) otherwise.

    Parameters
    ----------
    config_path   : path to the JSON config to update
    watch_dir     : directory to watch for new feedback files
    pattern       : glob pattern to match feedback files (default "*.csv")
    poll_interval : seconds between directory polls when watchdog is
                    unavailable (default 30)
    bump          : semver bump per update — "patch" (default), "minor", "major"
    min_counts    : minimum observations before updating a source (default 1.0)
    on_update     : optional callback(summary: dict) called after each
                    successful update
    on_error      : optional callback(path: str, error: Exception) called
                    on any processing error
    backup_dir    : directory for config backups
    log_path      : path to the JSONL audit log
    """

    def __init__(
        self,
        config_path: Union[str, Path],
        watch_dir: Union[str, Path],
        pattern: str = "*.csv",
        poll_interval: int = 30,
        bump: str = "patch",
        min_counts: float = 1.0,
        on_update: Optional[Callable[[dict], None]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
        backup_dir: Optional[Union[str, Path]] = None,
        log_path: Optional[Union[str, Path]] = None,
    ):
        self.watch_dir     = Path(watch_dir).resolve()
        self.pattern       = pattern
        self.poll_interval = poll_interval
        self.on_update     = on_update
        self.on_error      = on_error

        self._updater = FeedbackUpdater(
            config_path = config_path,
            bump        = bump,
            min_counts  = min_counts,
            backup_dir  = backup_dir,
            log_path    = log_path,
        )

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.watch_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the watcher in a background thread (non-blocking)."""
        if self._thread and self._thread.is_alive():
            logger.warning("Watcher already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="FeedbackWatcher")
        self._thread.start()
        print(f"[FeedbackWatcher] Started — watching {self.watch_dir} for {self.pattern}")

    def stop(self) -> None:
        """Signal the watcher thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.poll_interval + 2)
            self._thread = None
        print("[FeedbackWatcher] Stopped.")

    def process_now(self) -> list[dict]:
        """
        Manually trigger a scan and process any unprocessed files immediately.
        Safe to call while the watcher is also running.
        """
        return self._updater.apply_directory(self.watch_dir, pattern=self.pattern)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "FeedbackWatcher":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal run loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Main watcher loop — uses watchdog if available, else polls."""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler, FileCreatedEvent

            class _Handler(FileSystemEventHandler):
                def __init__(self_, updater, pattern, on_update, on_error):
                    self_._updater   = updater
                    self_._pattern   = pattern.lstrip("*").lower()
                    self_._on_update = on_update
                    self_._on_error  = on_error

                def on_created(self_, event):
                    if isinstance(event, FileCreatedEvent):
                        p = Path(event.src_path)
                        if p.suffix.lower() == self_._pattern or self_._pattern == ".*":
                            time.sleep(0.5)  # brief wait for file to be fully written
                            self_._process(p)

                def _process(self_, path):
                    try:
                        summary = self_._updater.apply(path)
                        if self_._on_update:
                            self_._on_update(summary)
                    except Exception as e:
                        logger.error("Error processing %s: %s", path, e)
                        if self_._on_error:
                            self_._on_error(str(path), e)

            handler  = _Handler(self._updater, self.pattern, self.on_update, self.on_error)
            observer = Observer()
            observer.schedule(handler, str(self.watch_dir), recursive=False)
            observer.start()
            print("[FeedbackWatcher] Using watchdog for instant file-event detection.")

            while not self._stop_event.is_set():
                self._stop_event.wait(timeout=1)

            observer.stop()
            observer.join()

        except ImportError:
            print(
                f"[FeedbackWatcher] watchdog not installed — using polling "
                f"(interval: {self.poll_interval}s). "
                f"For instant detection: pip install watchdog"
            )
            self._poll_loop()

    def _poll_loop(self) -> None:
        """Polling fallback: scan for new files every poll_interval seconds."""
        while not self._stop_event.is_set():
            try:
                summaries = self._updater.apply_directory(
                    self.watch_dir, pattern=self.pattern
                )
                for summary in summaries:
                    if self.on_update and summary["n_sources_updated"] > 0:
                        self.on_update(summary)
            except Exception as e:
                logger.error("Error during directory scan: %s", e)
                if self.on_error:
                    self.on_error(str(self.watch_dir), e)

            self._stop_event.wait(timeout=self.poll_interval)
