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

# Fresh updater for sections 4 and 7
updater2 = FeedbackUpdater(CONFIG_PATH, bump="patch")

summary_w = updater2.preview(feedback_weighted)
updater2.print_summary(summary_w)

src6_change = summary_w["changes"].get("Source 6", {})
if src6_change:
    print(f"Effective observations: +{src6_change['n_positive']:.1f} positive, "
          f"-{src6_change['n_negative']:.1f} negative  (weight applied)\n")

# ---------------------------------------------------------------------------
# 5.  Attributed feedback — in-depth multi-week screening scenario
#
#     Scenario: a financial crime screening system flags transactions for
#     review.  Each flagged case has one or more sources that fired (e.g.
#     watchlist hit, adverse media, high-risk geography).  After a human
#     analyst reviews each case, they record the confirmed outcome.
#
#     We simulate three weeks of feedback and show:
#       a) how credit is distributed case-by-case using marginal attribution
#       b) how mu and kappa for each source evolve over time
#       c) how the attribution method (marginal vs uniform vs mu) would
#          have produced different parameter trajectories
# ---------------------------------------------------------------------------
print("=" * 62)
print("5. ATTRIBUTED FEEDBACK — IN-DEPTH SCENARIO")
print("=" * 62)

import csv
from math import prod

# Load starting parameters for reference
with open(CONFIG_PATH) as f:
    cfg_start = json.load(f)

# Rename for clarity in this scenario
SOURCE_NAMES = [s["name"] for s in cfg_start["sources"]]
start_mus    = {s["name"]: s["mu"]    for s in cfg_start["sources"]}
start_kappas = {s["name"]: s["kappa"] for s in cfg_start["sources"]}

print("\nStarting parameters:")
print(f"  {'Source':<12}  {'mu':>6}  {'kappa':>7}  {'Interpretation'}")
print(f"  {'-'*12}  {'-'*6}  {'-'*7}  {'-'*40}")
interpretations = {
    "Source 1": "Watchlist hit — strong prior (mu=0.60)",
    "Source 2": "Adverse media — moderate prior (mu=0.45)",
    "Source 3": "High-risk geography — weaker prior (mu=0.30)",
    "Source 4": "PEP match — lower risk (mu=0.20)",
    "Source 5": "Unusual transaction pattern — very low (mu=0.15)",
    "Source 6": "Sanctions adjacent — moderate (mu=0.50)",
    "Source 7": "Dormant account activity — low (mu=0.10)",
    "Source 8": "Cross-border flag — moderate-low (mu=0.35)",
}
for s in cfg_start["sources"]:
    print(f"  {s['name']:<12}  {s['mu']:>6.2f}  {s['kappa']:>7.1f}  "
          f"{interpretations.get(s['name'], '')}")

# ── Week 1: 10 cases, mostly watchlist + adverse media combos ────────────
week1_cases = [
    # Case W1-01: Watchlist + Adverse media → confirmed high risk
    {"active_sources": ["Source 1", "Source 2"], "outcome": 1, "weight": 1.0},
    # Case W1-02: Watchlist + Geography → confirmed high risk
    {"active_sources": ["Source 1", "Source 3"], "outcome": 1, "weight": 1.0},
    # Case W1-03: Adverse media + PEP → confirmed high risk
    {"active_sources": ["Source 2", "Source 4"], "outcome": 1, "weight": 1.0},
    # Case W1-04: Geography only → false alarm (geography was insufficient alone)
    {"active_sources": ["Source 3"], "outcome": 0, "weight": 1.0},
    # Case W1-05: Watchlist + Adverse + Geography → confirmed high risk
    {"active_sources": ["Source 1", "Source 2", "Source 3"], "outcome": 1, "weight": 1.0},
    # Case W1-06: Transaction pattern + Dormant → false alarm
    {"active_sources": ["Source 5", "Source 7"], "outcome": 0, "weight": 1.0},
    # Case W1-07: Sanctions adjacent → confirmed (high-confidence, double weight)
    {"active_sources": ["Source 6"], "outcome": 1, "weight": 2.0},
    # Case W1-08: Cross-border + PEP → false alarm
    {"active_sources": ["Source 8", "Source 4"], "outcome": 0, "weight": 1.0},
    # Case W1-09: Watchlist only → confirmed (solo, full credit to Source 1)
    {"active_sources": ["Source 1"], "outcome": 1, "weight": 1.0},
    # Case W1-10: All 3 strong sources → confirmed (highest-weight case this week)
    {"active_sources": ["Source 1", "Source 2", "Source 6"], "outcome": 1, "weight": 2.0},
]

# ── Week 2: 10 cases, more adverse media and geography signals ────────────
week2_cases = [
    # W2-01: Adverse media → confirmed (getting clearer signal)
    {"active_sources": ["Source 2"], "outcome": 1, "weight": 1.0},
    # W2-02: Watchlist + Geography + Cross-border → confirmed
    {"active_sources": ["Source 1", "Source 3", "Source 8"], "outcome": 1, "weight": 1.0},
    # W2-03: PEP + Dormant → false alarm
    {"active_sources": ["Source 4", "Source 7"], "outcome": 0, "weight": 1.0},
    # W2-04: Sanctions + Adverse → confirmed
    {"active_sources": ["Source 6", "Source 2"], "outcome": 1, "weight": 1.0},
    # W2-05: Transaction pattern only → false alarm
    {"active_sources": ["Source 5"], "outcome": 0, "weight": 1.0},
    # W2-06: Watchlist + Adverse → confirmed
    {"active_sources": ["Source 1", "Source 2"], "outcome": 1, "weight": 1.0},
    # W2-07: Geography + Transaction → false alarm (emerging pattern)
    {"active_sources": ["Source 3", "Source 5"], "outcome": 0, "weight": 1.0},
    # W2-08: Cross-border + Adverse → confirmed
    {"active_sources": ["Source 8", "Source 2"], "outcome": 1, "weight": 1.0},
    # W2-09: PEP only → confirmed (analyst surprised — PEP alone was enough)
    {"active_sources": ["Source 4"], "outcome": 1, "weight": 1.5},
    # W2-10: Dormant only → false alarm
    {"active_sources": ["Source 7"], "outcome": 0, "weight": 1.0},
]

# ── Week 3: 10 cases, confirming patterns from weeks 1-2 ─────────────────
week3_cases = [
    # W3-01: Watchlist solo → confirmed
    {"active_sources": ["Source 1"], "outcome": 1, "weight": 1.0},
    # W3-02: Geography alone → false alarm again (pattern reinforced)
    {"active_sources": ["Source 3"], "outcome": 0, "weight": 1.0},
    # W3-03: PEP + Watchlist → confirmed
    {"active_sources": ["Source 4", "Source 1"], "outcome": 1, "weight": 1.0},
    # W3-04: Adverse + Sanctions + Cross-border → confirmed (complex case)
    {"active_sources": ["Source 2", "Source 6", "Source 8"], "outcome": 1, "weight": 1.0},
    # W3-05: Transaction + PEP → false alarm
    {"active_sources": ["Source 5", "Source 4"], "outcome": 0, "weight": 1.0},
    # W3-06: Watchlist + Adverse + Geography + PEP → confirmed (full house)
    {"active_sources": ["Source 1","Source 2","Source 3","Source 4"], "outcome": 1, "weight": 1.0},
    # W3-07: Sanctions adjacent → confirmed again
    {"active_sources": ["Source 6"], "outcome": 1, "weight": 1.0},
    # W3-08: Dormant + Cross-border → false alarm
    {"active_sources": ["Source 7", "Source 8"], "outcome": 0, "weight": 1.0},
    # W3-09: Adverse media → confirmed (getting strong signal now)
    {"active_sources": ["Source 2"], "outcome": 1, "weight": 2.0},
    # W3-10: All sources → confirmed (extreme case, high-confidence)
    {"active_sources": SOURCE_NAMES, "outcome": 1, "weight": 2.0},
]

# Write the three weekly feedback files in case-level format
def write_weekly(path, cases):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["active_sources", "outcome", "weight", "attribution_method"])
        for c in cases:
            writer.writerow([",".join(c["active_sources"]), c["outcome"],
                             c.get("weight", 1.0), "marginal"])
    n_cases = len(cases)
    n_pos   = sum(c["outcome"] for c in cases)
    total_w = sum(c.get("weight", 1.0) for c in cases)
    print(f"  Written: {path.name}  "
          f"({n_cases} cases, {n_pos} confirmed positive, {total_w:.1f} total weight)")

print("\nGenerating three weeks of attributed feedback:")
fb_week1 = FEEDBACK_DIR / "feedback_week1_attributed.csv"
fb_week2 = FEEDBACK_DIR / "feedback_week2_attributed.csv"
fb_week3 = FEEDBACK_DIR / "feedback_week3_attributed.csv"
write_weekly(fb_week1, week1_cases)
write_weekly(fb_week2, week2_cases)
write_weekly(fb_week3, week3_cases)

# Apply all three weeks and track mu evolution
updater_attr = FeedbackUpdater(CONFIG_PATH, bump="patch", min_counts=0.1)

print("\n── Applying week 1 ──")
s1 = updater_attr.apply(fb_week1)
with open(CONFIG_PATH) as f: cfg_w1 = json.load(f)
mus_after_w1 = {s["name"]: s["mu"] for s in cfg_w1["sources"]}

print("\n── Applying week 2 ──")
s2 = updater_attr.apply(fb_week2)
with open(CONFIG_PATH) as f: cfg_w2 = json.load(f)
mus_after_w2 = {s["name"]: s["mu"] for s in cfg_w2["sources"]}

print("\n── Applying week 3 ──")
s3 = updater_attr.apply(fb_week3)
with open(CONFIG_PATH) as f: cfg_w3 = json.load(f)
mus_after_w3 = {s["name"]: s["mu"] for s in cfg_w3["sources"]}

# Show mu evolution across all three weeks
print(f"\n{'='*62}")
print("MU EVOLUTION — three weeks of marginal attribution updates")
print(f"{'='*62}")
print(f"  {'Source':<12}  {'Start':>7}  {'After W1':>9}  {'After W2':>9}  "
      f"{'After W3':>9}  {'Δ Total':>9}  {'Trend'}")
print(f"  {'-'*12}  {'-'*7}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*6}")
for name in SOURCE_NAMES:
    s0 = start_mus[name]
    w1 = mus_after_w1[name]
    w2 = mus_after_w2[name]
    w3 = mus_after_w3[name]
    delta = w3 - s0
    trend = "▲▲" if delta > 0.05 else ("▲" if delta > 0.01 else
            ("▼▼" if delta < -0.05 else ("▼" if delta < -0.01 else "─")))
    print(f"  {name:<12}  {s0:>7.4f}  {w1:>9.4f}  {w2:>9.4f}  "
          f"{w3:>9.4f}  {delta:>+9.4f}  {trend}")

print("\nKey observations:")
print("  • Sources 1, 2, 6 show the largest mu increases — consistent")
print("    confirmation across many multi-source cases.")
print("  • Source 3 (geography) may have risen less than expected despite")
print("    co-firing frequently, because marginal attribution correctly")
print("    gives it less credit when stronger sources are also active.")
print("  • Sources 5, 7 (low-signal sources) correctly received more")
print("    negative feedback — they appear in false alarms.")

# ── Show per-case attribution math for a single illustrative case ─────────
print(f"\n{'='*62}")
print("CASE-BY-CASE ATTRIBUTION DETAIL (Week 1, Case W1-10)")
print(f"{'='*62}")
print("  Case: Sources 1+2+6 all fired → confirmed HIGH RISK (weight=2.0)")
print(f"  Starting mus: S1={start_mus['Source 1']:.4f}  "
      f"S2={start_mus['Source 2']:.4f}  S6={start_mus['Source 6']:.4f}")

case_sources  = ["Source 1", "Source 2", "Source 6"]
case_mus      = {s: start_mus[s] for s in case_sources}
case_weight   = 2.0
combined_risk = 1.0 - prod(1.0 - case_mus[s] for s in case_sources)

print(f"\n  Combined Noisy-OR risk: "
      f"1 - (1-{case_mus['Source 1']:.2f})(1-{case_mus['Source 2']:.2f})"
      f"(1-{case_mus['Source 6']:.2f}) = {combined_risk:.4f} ({combined_risk:.1%})")

marginal_weights = _compute_attribution(case_sources, case_mus, "marginal")
uniform_weights  = _compute_attribution(case_sources, case_mus, "uniform")

print(f"\n  Attribution weights (marginal vs uniform):")
print(f"  {'Source':<12}  {'mu':>6}  {'Marginal wt':>12}  {'Eff. obs (×2.0)':>16}  "
      f"{'Uniform wt':>11}  {'Eff. obs (×2.0)':>16}")
print(f"  {'-'*12}  {'-'*6}  {'-'*12}  {'-'*16}  {'-'*11}  {'-'*16}")
for s in case_sources:
    mw = marginal_weights[s]
    uw = uniform_weights[s]
    print(f"  {s:<12}  {case_mus[s]:>6.4f}  {mw:>11.4f}  "
          f"{mw*case_weight:>+16.4f}  {uw:>11.4f}  {uw*case_weight:>+16.4f}")

print(f"\n  Marginal attribution concentrates more credit on Source 1 ({marginal_weights['Source 1']:.1%})")
print(f"  because it has the highest mu — removing it would drop combined risk")
print(f"  by {(combined_risk - (1 - prod(1-v for k,v in case_mus.items() if k != 'Source 1'))):.4f}.")
print(f"  Uniform would credit each source equally ({1/3:.1%}) regardless.")

# ── Show what uniform attribution would have produced after 3 weeks ────────
print(f"\n{'='*62}")
print("ATTRIBUTION METHOD COMPARISON — effect on final mu values")
print(f"{'='*62}")

# Re-apply all three weeks using uniform attribution to compare
CONFIG_UNIFORM = DEMO_DIR / "noisy_or_config_uniform.json"
shutil.copy2(CONFIG_SRC, CONFIG_UNIFORM)
updater_uniform = FeedbackUpdater(
    CONFIG_UNIFORM, bump="patch", min_counts=0.1, attribution_method="uniform"
)
for fb in [fb_week1, fb_week2, fb_week3]:
    updater_uniform.apply(fb)
with open(CONFIG_UNIFORM) as f:
    cfg_uniform = json.load(f)
mus_uniform = {s["name"]: s["mu"] for s in cfg_uniform["sources"]}

print(f"\n  {'Source':<12}  {'Start':>7}  {'Marginal':>9}  {'Uniform':>9}  "
      f"{'Difference':>11}  {'Explanation'}")
print(f"  {'-'*12}  {'-'*7}  {'-'*9}  {'-'*9}  {'-'*11}  {'-'*30}")
for name in SOURCE_NAMES:
    s0    = start_mus[name]
    marg  = mus_after_w3[name]
    unif  = mus_uniform[name]
    diff  = marg - unif
    note = ""
    if diff > 0.01:
        note = "marginal rightly credits stronger source more"
    elif diff < -0.01:
        note = "uniform over-credits this weaker source"
    elif abs(diff) <= 0.005:
        note = "similar (often fires alone or not at all)"
    print(f"  {name:<12}  {s0:>7.4f}  {marg:>9.4f}  {unif:>9.4f}  "
          f"{diff:>+11.4f}  {note}")

print("\nConclusion:")
print("  Marginal attribution produces more accurate mu estimates because it")
print("  accounts for how much each source actually contributed to the")
print("  combined risk — not just that it happened to be present in a case.")
print("  Sources that frequently co-fire with stronger signals are correctly")
print("  given less credit under marginal attribution than uniform would give.")

# ---------------------------------------------------------------------------
# 6.  Attribution method comparison (per-case weights, three methods)
# ---------------------------------------------------------------------------
print()
print("=" * 62)
print("6. ATTRIBUTION METHOD COMPARISON (per-case weight tables)")
print("=" * 62)

# Use the current config mus (after week 3) for the comparison
with open(CONFIG_PATH) as f:
    cfg_current = json.load(f)
current_mus = {s["name"]: s["mu"] for s in cfg_current["sources"]}

comparison_cases = [
    (["Source 1", "Source 3", "Source 6"], "Strong + weak + mid"),
    (["Source 1", "Source 2", "Source 6"], "Three strong sources"),
    (["Source 3", "Source 5", "Source 7"], "Three weak sources"),
    (["Source 1"],                          "Solo strong source"),
]

for sources, label in comparison_cases:
    r = 1.0 - prod(1.0 - current_mus[s] for s in sources)
    print(f"\n  Case: {label}")
    print(f"  Active: {sources}   Combined risk: {r:.1%}")
    print(f"  {'Source':<12}  {'mu':>6}  {'marginal':>10}  {'uniform':>10}  {'mu-prop':>10}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")
    wm = _compute_attribution(sources, current_mus, "marginal")
    wu = _compute_attribution(sources, current_mus, "uniform")
    wmu= _compute_attribution(sources, current_mus, "mu")
    for s in sources:
        print(f"  {s:<12}  {current_mus[s]:>6.4f}  "
              f"{wm[s]:>9.1%}  {wu[s]:>9.1%}  {wmu[s]:>9.1%}")

print()
print("Key insight: for the 'three weak sources' case, all three methods")
print("give similar weights because the sources are close to each other.")
print("The difference is largest when source strengths are spread apart.")
print()

# Clean up the uniform config copy
CONFIG_UNIFORM.unlink(missing_ok=True)



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
