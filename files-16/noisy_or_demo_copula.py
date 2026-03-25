"""
noisy_or_demo_copula.py
=======================
Demonstrates NoisyORModelCopula — the Gaussian copula extension that
correctly handles correlated risk sources.

  1.  The problem: why standard Noisy-OR overestimates correlated risk
  2.  Configuring correlations
  3.  Risk score gradient across rho values
  4.  Marginal contributions under correlation
  5.  Batch prediction with mixed correlation patterns
  6.  Multi-source correlation — three correlated sources
  7.  Negative correlation
  8.  Saving and reloading a correlated config

Run:  python noisy_or_demo_copula.py
      (requires: pip install scipy)
"""

import json
import shutil
from math import prod
from pathlib import Path

import numpy as np

from noisy_or_model_copula import NoisyORModelCopula

# ---------------------------------------------------------------------------
# Setup — a workspace config with realistic AML/financial screening sources
# ---------------------------------------------------------------------------
DEMO_DIR = Path("demo_copula_workspace")
shutil.rmtree(DEMO_DIR, ignore_errors=True)
DEMO_DIR.mkdir()

SOURCES = [
    {"name": "Watchlist Hit",   "mu": 0.70, "kappa": 8.0},
    {"name": "Adverse Media",   "mu": 0.55, "kappa": 7.0},
    {"name": "PEP Match",       "mu": 0.60, "kappa": 6.0},
    {"name": "Geography Risk",  "mu": 0.35, "kappa": 8.0},
    {"name": "Transaction Flag","mu": 0.25, "kappa": 6.0},
]
N = len(SOURCES)
NAMES = [s["name"] for s in SOURCES]

# Base config — no correlations (standard independent model)
base_cfg = {
    "version": "1.0.0",
    "n_monte_carlo_samples": 50000,
    "sources": SOURCES,
}
BASE_CONFIG = DEMO_DIR / "config_independent.json"
BASE_CONFIG.write_text(json.dumps(base_cfg, indent=2))

# Correlated config — realistic AML correlation structure
#   Watchlist Hit ↔ Adverse Media:  rho=0.75  (often same entity appears in both)
#   Watchlist Hit ↔ PEP Match:      rho=0.55  (politically exposed persons on lists)
#   Adverse Media ↔ PEP Match:      rho=0.40  (PEPs generate adverse media)
corr_cfg = {
    **base_cfg,
    "correlations": [
        {"source_a": "Watchlist Hit",  "source_b": "Adverse Media", "rho": 0.75},
        {"source_a": "Watchlist Hit",  "source_b": "PEP Match",     "rho": 0.55},
        {"source_a": "Adverse Media",  "source_b": "PEP Match",     "rho": 0.40},
    ]
}
CORR_CONFIG = DEMO_DIR / "config_correlated.json"
CORR_CONFIG.write_text(json.dumps(corr_cfg, indent=2))

# Load both models
indp = NoisyORModelCopula(BASE_CONFIG)
corr = NoisyORModelCopula(CORR_CONFIG)

print(f"Independent model: {indp}")
print(f"Correlated model:  {corr}")
print()

# ---------------------------------------------------------------------------
# 1.  The problem: why standard Noisy-OR overestimates correlated risk
# ---------------------------------------------------------------------------
print("=" * 65)
print("1. THE PROBLEM — standard Noisy-OR with correlated sources")
print("=" * 65)

# Watchlist Hit and Adverse Media co-fired — these are correlated (rho=0.75)
flags_wl_am = np.array([1, 1, 0, 0, 0])

ri = indp.predict(flags_wl_am)
rc = corr.predict(flags_wl_am)

mu_wl = SOURCES[0]["mu"]
mu_am = SOURCES[1]["mu"]
nor_expected = 1.0 - (1.0 - mu_wl) * (1.0 - mu_am)

print(f"\nCase: Watchlist Hit + Adverse Media both fired  (rho=0.75)")
print(f"\n  Source parameters:")
print(f"    Watchlist Hit:  mu={mu_wl:.2f}  (70% prior risk when it fires alone)")
print(f"    Adverse Media:  mu={mu_am:.2f}  (55% prior risk when it fires alone)")
print(f"\n  Standard Noisy-OR (assumes independence):")
print(f"    Expected:  1-(1-{mu_wl})(1-{mu_am}) = {nor_expected:.4f} ({nor_expected:.1%})")
print(f"    Computed:  {ri['risk_score']:.4f} ({ri['risk_score']:.1%})")
print(f"\n  Correlated model (rho=0.75, blended combination):")
print(f"    Computed:  {rc['risk_score']:.4f} ({rc['risk_score']:.1%})")
print(f"    Reduction: {ri['risk_score'] - rc['risk_score']:+.4f} "
      f"({(ri['risk_score'] - rc['risk_score']) / ri['risk_score']:.1%} lower)")
print(f"\n  Intuition: at rho=0.75, both sources often fire because of the")
print(f"  same underlying driver (a high-risk entity appears on watchlists")
print(f"  AND gets adverse media coverage).  Treating them as two independent")
print(f"  pieces of evidence double-counts the same signal.")
print(f"\n  The correlated model recognises this and discounts the combined")
print(f"  score accordingly.")

# ---------------------------------------------------------------------------
# 2.  Configuring correlations and inspecting them
# ---------------------------------------------------------------------------
print()
print("=" * 65)
print("2. CONFIGURING AND INSPECTING CORRELATIONS")
print("=" * 65)

print()
corr.print_correlations()

print(f"Properties:")
print(f"  n_correlated_pairs = {corr.n_correlated_pairs}")
print(f"  correlated_pairs   = {corr.correlated_pairs}")
print()

# ---------------------------------------------------------------------------
# 3.  Risk score gradient across rho values — two sources, both active
# ---------------------------------------------------------------------------
print("=" * 65)
print("3. RISK SCORE GRADIENT ACROSS RHO VALUES")
print("=" * 65)

print(f"\nBoth Watchlist Hit (mu={mu_wl}) and Adverse Media (mu={mu_am}) fired.")
print(f"Showing how combined risk changes with correlation:\n")

print(f"  {'rho':>5}   {'Risk Score':>11}   {'Reduction vs rho=0':>20}   "
      f"{'Interpretation'}")
print(f"  {'-'*5}   {'-'*11}   {'-'*20}   {'-'*30}")

baseline = None
rho_values = [0.00, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
for rho in rho_values:
    cfg_tmp = {**base_cfg}
    if rho > 0:
        cfg_tmp["correlations"] = [
            {"source_a": "Watchlist Hit", "source_b": "Adverse Media", "rho": rho}
        ]
    cfg_path = DEMO_DIR / f"config_rho_{str(rho).replace('.','')}.json"
    cfg_path.write_text(json.dumps(cfg_tmp, indent=2))
    m = NoisyORModelCopula(cfg_path)
    r = m.predict(flags_wl_am)["risk_score"]

    if baseline is None:
        baseline = r
    reduction = baseline - r
    pct_red   = reduction / baseline * 100 if baseline else 0

    note = ""
    if rho == 0.00: note = "Standard Noisy-OR"
    elif rho == 0.75: note = "← configured in demo"
    elif rho == 0.99: note = "≈ max(mu_WL, mu_AM)"

    print(f"  {rho:>5.2f}   {r:>10.1%}   {reduction:>+10.4f}  "
          f"({pct_red:>5.1f}% lower)   {note}")

print(f"\n  At rho→1.0 the combined risk approaches the strongest individual")
print(f"  source (Watchlist Hit at mu=0.70), correctly treating the pair")
print(f"  as a single redundant signal.")

# ---------------------------------------------------------------------------
# 4.  Marginal contributions under correlation
# ---------------------------------------------------------------------------
print()
print("=" * 65)
print("4. MARGINAL CONTRIBUTIONS UNDER CORRELATION")
print("=" * 65)

print(f"\nCase: Watchlist Hit + Adverse Media both fired")
print(f"\n  Standard model (rho=0.00):")

ci = indp.predict(flags_wl_am)
print(f"    risk_score  = {ci['risk_score']:.4f} ({ci['risk_score']:.1%})")
print(f"    contrib WL  = {ci['primary_driver_contribution']:.4f} "
      f"({ci['primary_driver_contribution']:.1%})  ← adding WL on top of AM")
contrib_am_i = ci['other_matches'][0]['marginal_contribution']
print(f"    contrib AM  = {contrib_am_i:.4f} "
      f"({contrib_am_i:.1%})  ← adding AM on top of WL")

print(f"\n  Correlated model (rho=0.75):")
cc = corr.predict(flags_wl_am)
print(f"    risk_score  = {cc['risk_score']:.4f} ({cc['risk_score']:.1%})")
print(f"    contrib WL  = {cc['primary_driver_contribution']:.4f} "
      f"({cc['primary_driver_contribution']:.1%})  ← adding WL on top of AM")
contrib_am_c = cc['other_matches'][0]['marginal_contribution']
print(f"    contrib AM  = {contrib_am_c:.4f} "
      f"({contrib_am_c:.1%})  ← adding AM on top of WL")

print(f"\n  Reduction in contributions from rho=0.75:")
delta_wl = ci['primary_driver_contribution'] - cc['primary_driver_contribution']
delta_am = contrib_am_i - contrib_am_c
print(f"    WL contribution reduced by {delta_wl:.4f} "
      f"({delta_wl / ci['primary_driver_contribution']:.0%})")
print(f"    AM contribution reduced by {delta_am:.4f} "
      f"({delta_am / contrib_am_i:.0%})")

print(f"\n  Why contributions shrink:")
print(f"    c_WL = R(WL+AM) - R(AM alone)")
print(f"    At rho=0.75, R(AM alone) = {cc['other_matches'][0]['source_mean']:.4f}  "
      f"(≈ mu_AM = {mu_am})")
print(f"    R(WL+AM)     = {cc['risk_score']:.4f}")
print(f"    c_WL         = {cc['risk_score']:.4f} - "
      f"{cc['other_matches'][0]['source_mean']:.4f} = "
      f"{cc['risk_score'] - cc['other_matches'][0]['source_mean']:.4f}")
print(f"\n  When AM is already matched, WL adds little incremental risk")
print(f"  because the two signals are largely redundant at rho=0.75.")
print(f"  This is the correct interpretation — and would be invisible in")
print(f"  the standard model, which reports the same contributions as")
print(f"  if the sources were independent.")

# ---------------------------------------------------------------------------
# 5.  Batch prediction with mixed correlation patterns
# ---------------------------------------------------------------------------
print()
print("=" * 65)
print("5. BATCH PREDICTION — MIXED CORRELATION PATTERNS")
print("=" * 65)

batch = np.array([
    [1, 1, 0, 0, 0],   # WL + AM (correlated pair, rho=0.75)
    [1, 0, 1, 0, 0],   # WL + PEP (correlated pair, rho=0.55)
    [0, 0, 0, 1, 1],   # Geo + Transaction (uncorrelated pair, rho=0.00)
    [1, 1, 1, 0, 0],   # WL + AM + PEP (all three correlated)
    [1, 0, 0, 0, 0],   # WL alone
    [0, 1, 0, 0, 0],   # AM alone
    [0, 0, 0, 0, 0],   # No sources
])
labels = [
    "WL+AM (ρ=0.75)",
    "WL+PEP (ρ=0.55)",
    "Geo+Txn (ρ=0.00)",
    "WL+AM+PEP (all corr)",
    "WL only",
    "AM only",
    "No sources",
]

df_c = corr.predict_batch(batch, labels=labels)
df_i = indp.predict_batch(batch, labels=labels)

print(f"\n  {'Case':<22}  {'Corr risk':>10}  {'Indep risk':>11}  "
      f"{'Reduction':>10}  {'Note'}")
print(f"  {'-'*22}  {'-'*10}  {'-'*11}  {'-'*10}  {'-'*30}")

for i, lbl in enumerate(labels):
    rc_s = df_c['Risk_Score'].iloc[i]
    ri_s = df_i['Risk_Score'].iloc[i]
    diff = ri_s - rc_s
    note = ""
    if diff < 0.001:
        note = "no correlation → scores identical"
    elif diff > 0.08:
        note = "high redundancy → large reduction"
    print(f"  {lbl:<22}  {rc_s:>10.1%}  {ri_s:>11.1%}  {diff:>+10.4f}  {note}")

print(f"\n  Observations:")
print(f"  • WL+AM (rho=0.75) shows the largest reduction — highest correlation")
print(f"  • Geo+Transaction (rho=0.00) shows zero reduction — truly independent")
print(f"  • Solo sources are identical — correlation only matters when ≥2 fire")
print(f"  • WL+AM+PEP shows reduction from combined correlation across all pairs")

# ---------------------------------------------------------------------------
# 6.  Multi-source correlation — three correlated sources
# ---------------------------------------------------------------------------
print()
print("=" * 65)
print("6. MULTI-SOURCE CORRELATION — THREE CORRELATED SOURCES")
print("=" * 65)

flags_3 = np.array([1, 1, 1, 0, 0])  # WL + AM + PEP all fired

rc3 = corr.predict(flags_3)
ri3 = indp.predict(flags_3)

print(f"\nCase: Watchlist Hit + Adverse Media + PEP Match all fired")
print(f"  (rho: WL-AM=0.75, WL-PEP=0.55, AM-PEP=0.40)")
print(f"\n  Independent risk:  {ri3['risk_score']:.4f} ({ri3['risk_score']:.1%})")
print(f"  Correlated risk:   {rc3['risk_score']:.4f} ({rc3['risk_score']:.1%})")
print(f"  Reduction:         {ri3['risk_score'] - rc3['risk_score']:+.4f}")

print(f"\n  Marginal contributions:")
print(f"\n  {'Source':<20}  {'Independent':>12}  {'Correlated':>11}  {'Change'}")
print(f"  {'-'*20}  {'-'*12}  {'-'*11}  {'-'*20}")

all_c = {ri3['primary_driver']: ri3['primary_driver_contribution']}
all_c.update({m['source']: m['marginal_contribution'] for m in ri3['other_matches']})
all_c3 = {rc3['primary_driver']: rc3['primary_driver_contribution']}
all_c3.update({m['source']: m['marginal_contribution'] for m in rc3['other_matches']})

for src in ["Watchlist Hit", "Adverse Media", "PEP Match"]:
    ci_s = all_c.get(src, 0.0)
    cc_s = all_c3.get(src, 0.0)
    delta = cc_s - ci_s
    print(f"  {src:<20}  {ci_s:>12.4f}  {cc_s:>11.4f}  {delta:>+10.4f}")

print(f"\n  rho_eff across all three = mean(0.75, 0.55, 0.40) = "
      f"{(0.75+0.55+0.40)/3:.3f}")
print(f"  All three contributions shrink as each source adds less incremental")
print(f"  evidence on top of the other two, which are already highly correlated.")

# ---------------------------------------------------------------------------
# 7.  Negative correlation
# ---------------------------------------------------------------------------
print()
print("=" * 65)
print("7. NEGATIVE CORRELATION")
print("=" * 65)

neg_cfg = {
    **base_cfg,
    "correlations": [
        {"source_a": "Watchlist Hit", "source_b": "Adverse Media", "rho": -0.40}
    ]
}
neg_config = DEMO_DIR / "config_negative.json"
neg_config.write_text(json.dumps(neg_cfg, indent=2))
neg_model = NoisyORModelCopula(neg_config)

flags_wl_am = np.array([1, 1, 0, 0, 0])
r_neg = neg_model.predict(flags_wl_am)["risk_score"]
r_ind = indp.predict(flags_wl_am)["risk_score"]
r_pos = corr.predict(flags_wl_am)["risk_score"]

print(f"\nCase: Watchlist Hit + Adverse Media both fired")
print(f"\n  rho = -0.40 (negatively correlated):  {r_neg:.4f} ({r_neg:.1%})")
print(f"  rho =  0.00 (independent):             {r_ind:.4f} ({r_ind:.1%})")
print(f"  rho = +0.75 (positively correlated):   {r_pos:.4f} ({r_pos:.1%})")
print(f"\n  Negative correlation means when one source fires at high strength,")
print(f"  the other tends to fire at lower strength.  When both fire together,")
print(f"  it's slightly more surprising than expected — the combination is")
print(f"  marginally stronger evidence than the independent model would suggest.")
print(f"  However, the blending effect is also weaker (rho_eff < 0 clamps to 0),")
print(f"  so the risk is close to standard Noisy-OR.")

# ---------------------------------------------------------------------------
# 8.  Saving and reloading a correlated config
# ---------------------------------------------------------------------------
print()
print("=" * 65)
print("8. SAVING AND RELOADING A CORRELATED CONFIG")
print("=" * 65)

sources_updated = {s["name"]: {"mu": s["mu"], "kappa": s["kappa"]}
                   for s in SOURCES}
# Slightly recalibrate Adverse Media upward
sources_updated["Adverse Media"]["mu"] = 0.60

updated_config = DEMO_DIR / "config_updated.json"
corr.save_config(
    sources_dict = sources_updated,
    output_path  = updated_config,
    bump         = "patch",
    correlations = [
        {"source_a": "Watchlist Hit",  "source_b": "Adverse Media", "rho": 0.75},
        {"source_a": "Watchlist Hit",  "source_b": "PEP Match",     "rho": 0.55},
        {"source_a": "Adverse Media",  "source_b": "PEP Match",     "rho": 0.40},
    ],
)

# Reload in place
corr.reload()
print(f"\nSaved + reloaded: {corr}")
saved = json.loads(updated_config.read_text())
assert "correlations" in saved, "Correlations were not persisted"
print(f"✓ Correlations persisted in saved config "
      f"({len(saved['correlations'])} pair(s))")
print(f"✓ Version bumped to {saved['version']}")

# ---------------------------------------------------------------------------
# Final summary table
# ---------------------------------------------------------------------------
print()
print("=" * 65)
print("SUMMARY — CORRELATED VS INDEPENDENT RISK SCORES")
print("=" * 65)

summary_cases = [
    ("WL alone",                  [1,0,0,0,0]),
    ("AM alone",                  [0,1,0,0,0]),
    ("PEP alone",                 [0,0,1,0,0]),
    ("WL + AM  (rho=0.75)",       [1,1,0,0,0]),
    ("WL + PEP (rho=0.55)",       [1,0,1,0,0]),
    ("AM + PEP (rho=0.40)",       [0,1,1,0,0]),
    ("WL + AM + PEP (all corr)",  [1,1,1,0,0]),
    ("All 5 sources",             [1,1,1,1,1]),
]
print(f"\n  {'Case':<28}  {'Independent':>12}  {'Correlated':>11}  {'Δ':>8}")
print(f"  {'-'*28}  {'-'*12}  {'-'*11}  {'-'*8}")
for lbl, f in summary_cases:
    flags = np.array(f)
    ri_s = indp.predict(flags)["risk_score"]
    rc_s = corr.predict(flags)["risk_score"]
    print(f"  {lbl:<28}  {ri_s:>12.1%}  {rc_s:>11.1%}  {rc_s-ri_s:>+8.4f}")

# Cleanup
shutil.rmtree(DEMO_DIR)
print("\nDemo workspace cleaned up.")
print("\nDone.")
