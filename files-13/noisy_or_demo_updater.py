"""
noisy_or_demo_updater.py
========================
Demonstrates the FeedbackUpdater and FeedbackWatcher.

  1.  Dry-run preview — see what would change before committing
  2.  Single file update — apply one feedback CSV
  3.  Directory update — apply all unprocessed CSVs in one call
  4.  Weighted feedback — higher-confidence labels count more
  5.  Reload model after update (standard and API models)
  6.  Watcher demo — automatic update when a new file is dropped

Run:  python noisy_or_demo_updater.py
"""

import json
import shutil
import time
import threading
from pathlib import Path

import numpy as np

from noisy_or_updater import FeedbackUpdater, FeedbackWatcher
from noisy_or_model import NoisyORModel

# ---------------------------------------------------------------------------
# Setup — work in a temporary directory to keep the real config untouched
# ---------------------------------------------------------------------------
DEMO_DIR    = Path("demo_update_workspace")
CONFIG_SRC  = Path("noisy_or_config.json")
CONFIG_PATH = DEMO_DIR / "noisy_or_config.json"
FEEDBACK_DIR = DEMO_DIR / "feedback"

shutil.rmtree(DEMO_DIR, ignore_errors=True)
DEMO_DIR.mkdir()
FEEDBACK_DIR.mkdir()
shutil.copy2(CONFIG_SRC, CONFIG_PATH)
print(f"Working config: {CONFIG_PATH}\n")

# ---------------------------------------------------------------------------
# Helper: write a feedback CSV
# ---------------------------------------------------------------------------
def write_feedback(path: Path, rows: list[dict]) -> None:
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_name", "outcome", "weight"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Feedback file written: {path.name}")

# ---------------------------------------------------------------------------
# 1.  Dry-run preview
# ---------------------------------------------------------------------------
print("=" * 62)
print("1. DRY-RUN PREVIEW")
print("=" * 62)

feedback_march = FEEDBACK_DIR / "feedback_march.csv"
write_feedback(feedback_march, [
    # Source 1: 8 confirmed positives, 2 negatives → mu should rise
    {"source_name": "Source 1", "outcome": 1, "weight": 1.0},
    {"source_name": "Source 1", "outcome": 1, "weight": 1.0},
    {"source_name": "Source 1", "outcome": 1, "weight": 1.0},
    {"source_name": "Source 1", "outcome": 1, "weight": 1.0},
    {"source_name": "Source 1", "outcome": 1, "weight": 1.0},
    {"source_name": "Source 1", "outcome": 1, "weight": 1.0},
    {"source_name": "Source 1", "outcome": 1, "weight": 1.0},
    {"source_name": "Source 1", "outcome": 1, "weight": 1.0},
    {"source_name": "Source 1", "outcome": 0, "weight": 1.0},
    {"source_name": "Source 1", "outcome": 0, "weight": 1.0},
    # Source 7: 2 negatives → mu should fall (it was already low at 0.10)
    {"source_name": "Source 7", "outcome": 0, "weight": 1.0},
    {"source_name": "Source 7", "outcome": 0, "weight": 1.0},
    # Source 3: only 1 observation — below default min_counts of 1? No, 1.0 == 1.0
    {"source_name": "Source 3", "outcome": 1, "weight": 1.0},
])

updater = FeedbackUpdater(CONFIG_PATH, bump="patch")
summary = updater.preview(feedback_march)
updater.print_summary(summary)

# Verify config unchanged
with open(CONFIG_PATH) as f:
    cfg_after = json.load(f)
assert cfg_after["version"] == "1.0.0", "Dry-run should not modify the config"
print("✓ Config version still 1.0.0 — dry-run made no changes\n")

# ---------------------------------------------------------------------------
# 2.  Apply the same file for real
# ---------------------------------------------------------------------------
print("=" * 62)
print("2. APPLY SINGLE FEEDBACK FILE")
print("=" * 62)

summary = updater.apply(feedback_march)
updater.print_summary(summary)

with open(CONFIG_PATH) as f:
    cfg_after = json.load(f)
print(f"Config version after update: {cfg_after['version']}")

# Show Source 1 mu increased
src1_new = next(s for s in cfg_after["sources"] if s["name"] == "Source 1")
print(f"Source 1 mu: 0.60 → {src1_new['mu']:.4f}  (expected increase from 8 positives)")
assert src1_new["mu"] > 0.60, "Source 1 mu should have increased"

# Show audit log entry
print(f"\nAudit log: {updater.log_path}")
with open(updater.log_path) as f:
    entry = json.loads(f.readline())
print(f"  timestamp     : {entry['timestamp']}")
print(f"  feedback_file : {Path(entry['feedback_file']).name}")
print(f"  version       : {entry['version_before']} → {entry['version_after']}")
print(f"  n_records     : {entry['n_records']}")

# Show backup was created
backups = list(updater.backup_dir.glob("*.json"))
print(f"\nBackup created: {backups[0].name}\n")

# ---------------------------------------------------------------------------
# 3.  Directory update — apply multiple files at once
# ---------------------------------------------------------------------------
print("=" * 62)
print("3. DIRECTORY UPDATE (multiple files)")
print("=" * 62)

feedback_april = FEEDBACK_DIR / "feedback_april.csv"
feedback_may   = FEEDBACK_DIR / "feedback_may.csv"

write_feedback(feedback_april, [
    # Source 2: very strong positive signal
    {"source_name": "Source 2", "outcome": 1, "weight": 1.0},
    {"source_name": "Source 2", "outcome": 1, "weight": 1.0},
    {"source_name": "Source 2", "outcome": 1, "weight": 1.0},
    {"source_name": "Source 2", "outcome": 1, "weight": 1.0},
    {"source_name": "Source 2", "outcome": 0, "weight": 1.0},
    # Source 5: consistently low risk
    {"source_name": "Source 5", "outcome": 0, "weight": 1.0},
    {"source_name": "Source 5", "outcome": 0, "weight": 1.0},
    {"source_name": "Source 5", "outcome": 0, "weight": 1.0},
])

write_feedback(feedback_may, [
    # Source 4: mixed evidence
    {"source_name": "Source 4", "outcome": 1, "weight": 1.0},
    {"source_name": "Source 4", "outcome": 0, "weight": 1.0},
    {"source_name": "Source 4", "outcome": 1, "weight": 1.0},
    # Source 8: strong positive
    {"source_name": "Source 8", "outcome": 1, "weight": 1.0},
    {"source_name": "Source 8", "outcome": 1, "weight": 1.0},
    {"source_name": "Source 8", "outcome": 1, "weight": 1.0},
    {"source_name": "Source 8", "outcome": 1, "weight": 1.0},
    {"source_name": "Source 8", "outcome": 0, "weight": 1.0},
])

# feedback_march is already processed — apply_directory skips it automatically
summaries = updater.apply_directory(FEEDBACK_DIR)
print(f"\n{len(summaries)} file(s) processed")
for s in summaries:
    print(f"  {Path(s.get('feedback_file', '?')).name if 'feedback_file' in s else ''}  "
          f"→  {s['n_sources_updated']} source(s) updated")

with open(CONFIG_PATH) as f:
    cfg_final = json.load(f)
print(f"\nConfig version after directory update: {cfg_final['version']}\n")

# ---------------------------------------------------------------------------
# 4.  Weighted feedback
# ---------------------------------------------------------------------------
print("=" * 62)
print("4. WEIGHTED FEEDBACK")
print("=" * 62)

# A higher-confidence label (e.g. verified by a second analyst) counts as 2
feedback_weighted = FEEDBACK_DIR / "feedback_weighted.csv"
write_feedback(feedback_weighted, [
    # Source 6: one high-confidence positive (weight=3) + one uncertain negative (weight=0.5)
    {"source_name": "Source 6", "outcome": 1, "weight": 3.0},
    {"source_name": "Source 6", "outcome": 0, "weight": 0.5},
])

# Fresh updater, no existing log context needed
updater2 = FeedbackUpdater(CONFIG_PATH, bump="patch")
summary_w = updater2.preview(feedback_weighted)
updater2.print_summary(summary_w)

src6_change = summary_w["changes"].get("Source 6", {})
if src6_change:
    print(f"Effective observations: +{src6_change['n_positive']:.1f} positive, "
          f"-{src6_change['n_negative']:.1f} negative  (weight applied)\n")

# ---------------------------------------------------------------------------
# 5.  Reload model after update
# ---------------------------------------------------------------------------
print("=" * 62)
print("5. RELOAD MODEL AFTER UPDATE")
print("=" * 62)

# Standard model
model_before = NoisyORModel(CONFIG_PATH)
print(f"Model before: {model_before}")

# Apply one more update
feedback_june = FEEDBACK_DIR / "feedback_june.csv"
write_feedback(feedback_june, [
    {"source_name": "Source 1", "outcome": 1, "weight": 1.0},
    {"source_name": "Source 1", "outcome": 1, "weight": 1.0},
    {"source_name": "Source 1", "outcome": 0, "weight": 1.0},
])
updater2.apply(feedback_june)

# Reload: just reinstantiate — the config file has been updated in place
model_after = NoisyORModel(CONFIG_PATH)
print(f"Model after:  {model_after}")

# Confirm parameter shift
src1_before = model_before._mu_values[0]
src1_after  = model_after._mu_values[0]
print(f"\nSource 1 mu:  {src1_before:.4f} → {src1_after:.4f}")

# API model — use hot reload
from noisy_or_model_api import NoisyORModelAPI
api_model = NoisyORModelAPI(CONFIG_PATH)
print(f"\nAPI model: {api_model}")

feedback_july = FEEDBACK_DIR / "feedback_july.csv"
write_feedback(feedback_july, [
    {"source_name": "Source 3", "outcome": 1, "weight": 1.0},
    {"source_name": "Source 3", "outcome": 1, "weight": 1.0},
    {"source_name": "Source 3", "outcome": 0, "weight": 1.0},
])
updater2.apply(feedback_july)

api_model.reload()   # hot reload — no need to reinstantiate
print(f"API model after reload: {api_model}")
print(f"Source 3 pool mean (reflects new mu): {api_model._pool[2].mean():.4f}\n")

# ---------------------------------------------------------------------------
# 6.  FeedbackWatcher demo (polling mode, short interval)
# ---------------------------------------------------------------------------
print("=" * 62)
print("6. FEEDBACK WATCHER (automatic, background thread)")
print("=" * 62)

WATCH_DIR = DEMO_DIR / "incoming"
WATCH_DIR.mkdir(exist_ok=True)

received_summaries = []

def on_update(summary: dict) -> None:
    received_summaries.append(summary)
    print(f"  [callback] Update received: "
          f"{summary['n_sources_updated']} source(s), "
          f"{summary['n_records']:.0f} observations")

watcher = FeedbackWatcher(
    config_path   = CONFIG_PATH,
    watch_dir     = WATCH_DIR,
    poll_interval = 2,      # 2-second poll for demo speed
    bump          = "patch",
    on_update     = on_update,
)

watcher.start()
print("Watcher running. Dropping a feedback file in 1 second...\n")

time.sleep(1)

# Drop a file into the watched directory
feedback_auto = WATCH_DIR / "feedback_auto.csv"
write_feedback(feedback_auto, [
    {"source_name": "Source 2", "outcome": 1, "weight": 1.0},
    {"source_name": "Source 2", "outcome": 1, "weight": 1.0},
    {"source_name": "Source 2", "outcome": 0, "weight": 1.0},
])
print("File dropped. Waiting up to 5 seconds for watcher to process it...")

# Wait for the callback to fire (up to 5 seconds)
for _ in range(25):
    if received_summaries:
        break
    time.sleep(0.2)

watcher.stop()

if received_summaries:
    print(f"\n✓ Watcher processed file automatically "
          f"({received_summaries[0]['n_sources_updated']} source(s) updated)")
else:
    print("\n(Watcher did not fire within timeout — this is normal if the "
          "poll interval was not reached in time. Try watcher.process_now() "
          "to trigger a manual scan.)")

# ---------------------------------------------------------------------------
# Final config snapshot
# ---------------------------------------------------------------------------
print("\n" + "=" * 62)
print("FINAL CONFIG SNAPSHOT")
print("=" * 62)
with open(CONFIG_PATH) as f:
    final_cfg = json.load(f)
print(f"Version: {final_cfg['version']}  ({final_cfg['version_date']})")
print(f"\n{'Source':<12} {'Original mu':>12} {'Final mu':>10} {'Final kappa':>12}")
print(f"{'-'*12} {'-'*12} {'-'*10} {'-'*12}")
original_mus = {"Source 1": 0.60, "Source 2": 0.45, "Source 3": 0.30,
                "Source 4": 0.20, "Source 5": 0.15, "Source 6": 0.50,
                "Source 7": 0.10, "Source 8": 0.35}
for s in final_cfg["sources"]:
    orig = original_mus.get(s["name"], "—")
    print(f"{s['name']:<12} {orig:>12}  {s['mu']:>10.4f}  {s['kappa']:>10.2f}")

# Show audit log
print(f"\nAudit log entries: {updater.log_path}")
with open(updater.log_path) as f:
    entries = [json.loads(l) for l in f if l.strip()]
for e in entries:
    n_changed = len(e["changes"])
    print(f"  {e['timestamp']}  v{e['version_before']} → v{e['version_after']}  "
          f"{n_changed} source(s)  {Path(e['feedback_file']).name}")

# Cleanup
shutil.rmtree(DEMO_DIR)
print("\nDemo workspace cleaned up.\n")
print("Done.")
