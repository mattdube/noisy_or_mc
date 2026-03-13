"""
noisy_or_demo_updater.py
========================
Demonstrates the FeedbackUpdater and FeedbackWatcher.

  1.  Dry-run preview — see what would change before committing
  2.  Single file update — apply one feedback CSV (source-level format)
  3.  Directory update — apply all unprocessed CSVs in one call
  4.  Weighted feedback — higher-confidence labels count more
  5.  Attributed feedback — multi-source cases with automatic credit assignment
  6.  Attribution method comparison — marginal vs uniform vs mu
  7.  Reload model after update (standard and API models)
  8.  Watcher demo — automatic update when a new file is dropped

Run:  python noisy_or_demo_updater.py
"""

import json
import shutil
import time
from pathlib import Path

import numpy as np

from noisy_or_updater import FeedbackUpdater, FeedbackWatcher, _compute_attribution
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
# Helpers: write source-level and case-level feedback CSVs
# ---------------------------------------------------------------------------
def write_feedback(path: Path, rows: list[dict]) -> None:
    """Write a FORMAT 1 (source-level) feedback CSV."""
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_name", "outcome", "weight"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Feedback written (source-level): {path.name}")


def write_case_feedback(path: Path, cases: list[dict]) -> None:
    """
    Write a FORMAT 2 (case-level attributed) feedback CSV.

    Each dict in `cases` should have:
      active_sources      list[str]  sources that fired on this case
      outcome             int        1 = confirmed risk, 0 = confirmed non-risk
      weight              float      optional, default 1.0
      attribution_method  str        optional, default "marginal"
    """
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["active_sources", "outcome", "weight", "attribution_method"])
        for case in cases:
            sources_str = ",".join(case["active_sources"])
            writer.writerow([
                sources_str,
                case["outcome"],
                case.get("weight", 1.0),
                case.get("attribution_method", "marginal"),
            ])
    print(f"Feedback written (case-level):   {path.name}")

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

# ---------------------------------------------------------------------------
# 7 continued — reload demo uses updater2
# ---------------------------------------------------------------------------
summary_w = updater2.preview(feedback_weighted)
updater2.print_summary(summary_w)

src6_change = summary_w["changes"].get("Source 6", {})
if src6_change:
    print(f"Effective observations: +{src6_change['n_positive']:.1f} positive, "
          f"-{src6_change['n_negative']:.1f} negative  (weight applied)\n")

# ---------------------------------------------------------------------------
# 5.  Attributed feedback — multi-source cases
#     When multiple sources fire on the same case, credit is distributed
#     proportionally using the Noisy-OR marginal contribution of each source.
#     The total evidence added always equals case_weight, regardless of how
#     many sources fired — no double-counting.
# ---------------------------------------------------------------------------
print("=" * 62)
print("5. ATTRIBUTED FEEDBACK (case-level format)")
print("=" * 62)

# Load current config to show source parameters before update
with open(CONFIG_PATH) as f:
    cfg_before = json.load(f)
source_mus_before = {s["name"]: s["mu"] for s in cfg_before["sources"]}

print("\nCurrent source parameters (mu):")
for s in cfg_before["sources"]:
    print(f"  {s['name']:<12}  mu={s['mu']:.4f}  kappa={s['kappa']:.2f}")

# Build a case-level feedback file.  Each row is a whole case — the updater
# uses the current model parameters to work out each source's share of credit.
feedback_attributed = FEEDBACK_DIR / "feedback_attributed.csv"
write_case_feedback(feedback_attributed, [
    # Case 1: Sources 1, 3, 6 all fired — confirmed HIGH risk.
    #   Source 1 (mu=0.60) will get the most credit as the strongest driver.
    #   Source 6 (mu=0.50) gets the second-most.
    #   Source 3 (mu=0.30) gets the least as the weakest contributor.
    {
        "active_sources":     ["Source 1", "Source 3", "Source 6"],
        "outcome":            1,
        "weight":             1.0,
        "attribution_method": "marginal",
    },
    # Case 2: Sources 1, 3, 6 all fired — confirmed LOW risk (false alarm).
    #   Negative feedback distributed by the same marginal attribution,
    #   so each source is penalised in proportion to how much it drove
    #   the original (incorrect) risk score.
    {
        "active_sources":     ["Source 1", "Source 3", "Source 6"],
        "outcome":            0,
        "weight":             1.0,
        "attribution_method": "marginal",
    },
    # Case 3: Sources 2 and 4 fired — confirmed HIGH risk.
    #   Source 2 (mu=0.45) is the stronger driver.
    {
        "active_sources":     ["Source 2", "Source 4"],
        "outcome":            1,
        "weight":             1.0,
        "attribution_method": "marginal",
    },
    # Case 4: Single source — full credit goes to Source 8 (no attribution needed).
    {
        "active_sources":     ["Source 8"],
        "outcome":            1,
        "weight":             1.0,
        "attribution_method": "marginal",
    },
    # Case 5: High-confidence case (weight=2.0) — counts as two observations.
    #   The attribution weights still sum to 1, so the effective n_positive
    #   added across all sources still equals 2.0 total.
    {
        "active_sources":     ["Source 1", "Source 6"],
        "outcome":            1,
        "weight":             2.0,
        "attribution_method": "marginal",
    },
])

print()

# Preview with low min_counts so all fractional updates are visible
updater3 = FeedbackUpdater(CONFIG_PATH, bump="patch", min_counts=0.1)
summary_attr = updater3.preview(feedback_attributed)

print("\nAttribution breakdown — fractional observations per source:")
print(f"  {'Source':<12}  {'n_pos':>8}  {'n_neg':>8}  {'net':>8}  {'direction'}")
print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}")
for name, ch in summary_attr["changes"].items():
    net = ch["n_positive"] - ch["n_negative"]
    direction = "▲ mu rises" if net > 0 else ("▼ mu falls" if net < 0 else "─ unchanged")
    print(f"  {name:<12}  {ch['n_positive']:>8.4f}  {ch['n_negative']:>8.4f}  "
          f"{net:>+8.4f}  {direction}")

# Verify total evidence conserved: sum of all n_pos + n_neg should equal
# sum of all case weights (5 cases: 1+1+1+1+2 = 6.0)
total_evidence = sum(
    ch["n_positive"] + ch["n_negative"]
    for ch in summary_attr["changes"].values()
)
print(f"\n  Total evidence across all sources: {total_evidence:.4f}")
print(f"  Sum of case weights:               6.0000")
print(f"  ✓ Conservation check: {abs(total_evidence - 6.0) < 0.001}")

updater3.print_summary(summary_attr)

# Apply for real
updater3.apply(feedback_attributed)
with open(CONFIG_PATH) as f:
    cfg_after_attr = json.load(f)
print("Config updated.  Source 1 vs Source 3 shifts reflect different marginal weights:")
for s in cfg_after_attr["sources"]:
    before = source_mus_before[s["name"]]
    after  = s["mu"]
    delta  = after - before
    if abs(delta) > 0.0001:
        print(f"  {s['name']:<12}  mu: {before:.4f} → {after:.4f}  (Δ{delta:+.4f})")

# ---------------------------------------------------------------------------
# 6.  Attribution method comparison
#     Show how the three attribution methods (marginal, uniform, mu) distribute
#     credit differently across the same set of active sources.
# ---------------------------------------------------------------------------
print()
print("=" * 62)
print("6. ATTRIBUTION METHOD COMPARISON")
print("=" * 62)

# Use current model mus as a reference point
with open(CONFIG_PATH) as f:
    cfg_current = json.load(f)
current_mus = {s["name"]: s["mu"] for s in cfg_current["sources"]}

active_sources = ["Source 1", "Source 3", "Source 6"]
print(f"\nActive sources for comparison: {active_sources}")
print(f"Current mu values:")
for s in active_sources:
    print(f"  {s}: {current_mus[s]:.4f}")

# Combined Noisy-OR risk from these three sources
from math import prod
combined_risk = 1.0 - prod(1.0 - current_mus[s] for s in active_sources)
print(f"\nCombined Noisy-OR risk: {combined_risk:.4f} ({combined_risk:.1%})")

print(f"\n  {'Method':<12}  ", end="")
for s in active_sources:
    print(f"  {s:<14}", end="")
print()
print(f"  {'-'*12}  " + "  ".join(["-"*14]*3))

for method in ("marginal", "uniform", "mu"):
    weights = _compute_attribution(active_sources, current_mus, method)
    print(f"  {method:<12}", end="")
    for s in active_sources:
        w = weights[s]
        print(f"  {w:.4f} ({w:.1%})  ", end="")
    print()

print()
print("Interpretation:")
print("  marginal — Source 1 gets ~51% because removing it would drop risk the most.")
print("             Source 6 (~34%) and Source 3 (~15%) reflect their actual impact.")
print("  uniform  — Equal 33% each; appropriate when you have no basis to distinguish.")
print("  mu       — Weights by raw prior strength; ignores how sources interact.")
print()
print("Recommendation: use 'marginal' (the default) unless you have a specific reason")
print("to prefer one of the others.\n")

# ---------------------------------------------------------------------------
# 7.  Reload model after update
# ---------------------------------------------------------------------------
print("=" * 62)
print("7. RELOAD MODEL AFTER UPDATE")
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
print("8. FEEDBACK WATCHER (automatic, background thread)")
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
