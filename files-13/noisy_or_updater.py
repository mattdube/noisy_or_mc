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

Feedback file format (CSV)
--------------------------
Required columns:
  source_name   string   must match a source name in the config exactly
  outcome       int      1 = confirmed risk/positive, 0 = confirmed non-risk

Optional columns:
  weight        float    case weight (default 1.0); fractional weights are
                         supported (e.g. 0.5 for lower-confidence labels)

Example:
  source_name,outcome,weight
  Source 1,1,1.0
  Source 1,0,1.0
  Source 3,1,2.0
  Source 6,0,1.0

Column names are case-insensitive. Extra columns are ignored.
Rows with unrecognised source names are skipped with a warning.
Rows with outcome not in {0, 1} after rounding are skipped with a warning.

Usage — manual
--------------
    from noisy_or_updater import FeedbackUpdater

    updater = FeedbackUpdater("noisy_or_config.json")

    # Apply a single feedback file
    updater.apply("feedback_2026_03.csv")

    # Apply all unprocessed CSVs in a directory
    updater.apply_directory("feedback/")

    # Dry-run: preview the updated parameters without saving
    summary = updater.apply("feedback_2026_03.csv", dry_run=True)
    print(summary)

Usage — automatic file watcher
-------------------------------
    from noisy_or_updater import FeedbackWatcher

    watcher = FeedbackWatcher(
        config_path   = "noisy_or_config.json",
        watch_dir     = "feedback/",
        # Optional callbacks:
        on_update     = lambda summary: print("Updated:", summary),
        on_error      = lambda path, err: print(f"Error in {path}: {err}"),
    )

    watcher.start()   # non-blocking — runs in a background thread
    # ... rest of your application ...
    watcher.stop()

    # Or use as a context manager:
    with FeedbackWatcher("noisy_or_config.json", "feedback/") as w:
        time.sleep(3600)   # watch for one hour

The watcher uses the `watchdog` library if installed (pip install watchdog)
for instant file-system event detection, or falls back to a polling loop
using only the standard library.

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
                     "kappa_before": 5.0, "kappa_after": 8.3},
        ...
      }
    }

A timestamped backup of the config is written to a `config_backups/`
subdirectory before every update.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Union

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
# Feedback file parsing
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

    Parameters
    ----------
    config_path : path to the JSON config to update
    bump        : semver component to increment on each update —
                  "patch" (default), "minor", or "major"
    backup_dir  : directory for config backups; defaults to
                  <config_dir>/config_backups/
    log_path    : path to the JSONL audit log; defaults to
                  <config_dir>/feedback_log.jsonl
    min_counts  : minimum total weighted observations required before a
                  source's parameters are updated (default 1.0).
                  Increase this to require more evidence before trusting
                  an update.
    """

    def __init__(
        self,
        config_path: Union[str, Path],
        bump: str = "patch",
        backup_dir: Optional[Union[str, Path]] = None,
        log_path: Optional[Union[str, Path]] = None,
        min_counts: float = 1.0,
    ):
        self.config_path = Path(config_path).resolve()
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        self.bump       = bump
        self.min_counts = min_counts

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
    ) -> dict:
        """
        Parse a feedback CSV and update the config with Bayesian posteriors.

        Parameters
        ----------
        feedback_path : path to the feedback CSV file
        dry_run       : if True, compute updates but do not write any files
        bump          : override the instance-level bump setting for this call

        Returns
        -------
        dict with keys:
            version_before, version_after, n_records, n_sources_updated,
            changes  (dict: source_name → {mu_before, mu_after,
                                            kappa_before, kappa_after,
                                            n_positive, n_negative})
            skipped_sources  (list of sources with insufficient observations)
            dry_run          (bool)
        """
        feedback_path = Path(feedback_path).resolve()
        if not feedback_path.exists():
            raise FileNotFoundError(f"Feedback file not found: {feedback_path}")

        cfg = self._load_config()
        sources = {s["name"]: s for s in cfg["sources"]}

        # Parse feedback
        counts = _parse_feedback_csv(feedback_path, set(sources))

        # Compute updates
        bump_to_use = bump or self.bump
        changes, skipped = self._compute_changes(sources, counts)

        if not changes:
            logger.info("No sources met the minimum observation threshold (%s). "
                        "Config unchanged.", self.min_counts)
            return {
                "version_before":    cfg["version"],
                "version_after":     cfg["version"],
                "n_records":         sum(v["n_positive"] + v["n_negative"]
                                        for v in counts.values()),
                "n_sources_updated": 0,
                "changes":           {},
                "skipped_sources":   skipped,
                "dry_run":           dry_run,
            }

        version_before = cfg["version"]
        version_after  = self._bump_version(version_before, bump_to_use)
        n_records      = sum(v["n_positive"] + v["n_negative"] for v in counts.values())

        if not dry_run:
            # Backup current config
            self._backup_config(cfg, version_before)

            # Apply changes to sources list
            for src in cfg["sources"]:
                if src["name"] in changes:
                    src["mu"]    = changes[src["name"]]["mu_after"]
                    src["kappa"] = changes[src["name"]]["kappa_after"]

            cfg["version"]      = version_after
            cfg["version_date"] = str(datetime.today().date())

            # Write updated config
            with open(self.config_path, "w") as f:
                json.dump(cfg, f, indent=2)

            # Append to audit log
            self._log_update(
                feedback_path  = feedback_path,
                version_before = version_before,
                version_after  = version_after,
                bump           = bump_to_use,
                n_records      = n_records,
                changes        = changes,
            )

            print(
                f"Config updated: {self.config_path.name}  "
                f"v{version_before} → v{version_after}  "
                f"({len(changes)} source(s) updated, {n_records:.1f} observations)"
            )
        else:
            print(
                f"[DRY RUN] Would update: {self.config_path.name}  "
                f"v{version_before} → v{version_after}  "
                f"({len(changes)} source(s), {n_records:.1f} observations)"
            )

        return {
            "version_before":    version_before,
            "version_after":     version_after,
            "n_records":         n_records,
            "n_sources_updated": len(changes),
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

    def _compute_changes(
        self,
        sources: dict[str, dict],
        counts: dict[str, dict[str, float]],
    ) -> tuple[dict, list[str]]:
        """
        Compute updated (mu, kappa) for each source that has enough observations.
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
        """
        feedback_path = Path(feedback_path).resolve()
        cfg     = self._load_config()
        sources = {s["name"]: s for s in cfg["sources"]}
        counts  = _parse_feedback_csv(feedback_path, set(sources))
        changes, skipped = self._compute_changes(sources, counts)

        n_records = sum(v["n_positive"] + v["n_negative"] for v in counts.values())
        version   = cfg["version"]

        if changes:
            for src in cfg["sources"]:
                if src["name"] in changes:
                    src["mu"]    = changes[src["name"]]["mu_after"]
                    src["kappa"] = changes[src["name"]]["kappa_after"]

            with open(self.config_path, "w") as f:
                json.dump(cfg, f, indent=2)

            self._log_update(
                feedback_path  = feedback_path,
                version_before = version,
                version_after  = version,
                bump           = "none",
                n_records      = n_records,
                changes        = changes,
            )
            print(f"  Applied {feedback_path.name}  ({n_records:.1f} obs, "
                  f"{len(changes)} source(s) updated — version held at {version})")

        return {
            "version_before":    version,
            "version_after":     version,
            "n_records":         n_records,
            "n_sources_updated": len(changes),
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
    ) -> None:
        entry = {
            "timestamp":      datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "feedback_file":  str(feedback_path),
            "config_path":    str(self.config_path),
            "version_before": version_before,
            "version_after":  version_after,
            "bump":           bump,
            "n_records":      n_records,
            "changes":        changes,
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
