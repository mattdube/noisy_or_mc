"""
noisy_or_demo_registry.py
=========================
Demonstrates the ModelRegistry for multi-tenant / multi-config deployments.

Covers:
  1. Loading a registry with multiple configs at startup
  2. Inspecting registry status
  3. Single predictions routed to specific configs
  4. Showing how the same input produces different scores across configs
  5. Batch prediction routed to a specific config
  6. Cache isolation — a hit in one config never affects another
  7. Reloading a single config without touching others
  8. Adding and removing configs at runtime
  9. Example of how routing would work in a Flask/FastAPI endpoint

Run:  python noisy_or_demo_registry.py
"""

import json
import numpy as np
import pandas as pd

from noisy_or_model_api import ModelRegistry, PRECISION_SAMPLES

# -----------------------------------------------------------------------
# 1.  Load registry — all pools built at startup
# -----------------------------------------------------------------------
print("=" * 60)
print("  MODEL REGISTRY DEMO")
print("=" * 60 + "\n")

registry = ModelRegistry(
    configs={
        "general":    "configs/general.json",
        "customer_1": "configs/customer_1.json",
        "customer_2": "configs/customer_2.json",
    },
    cache_size=256,
)

print(f"\nRegistry: {registry}")
print(f"Loaded configs: {registry.config_names}\n")

# -----------------------------------------------------------------------
# 2.  Registry status — useful for a /health endpoint
# -----------------------------------------------------------------------
print("=== Registry Status ===")
for name, info in registry.status().items():
    print(f"  {name:<12}  v{info['version']}  "
          f"{info['n_sources']} sources  "
          f"pool={info['pool_size']:,}  "
          f"cache={info['cache_info']}")
print()

# -----------------------------------------------------------------------
# 3.  Same input, different configs — scores differ because each config
#     has its own mu/kappa values for the same source names
# -----------------------------------------------------------------------
flags = np.array([1, 0, 1, 0, 0, 1, 0, 0])   # Watchlist, PEP, Transaction Flag

print("=== Same Input, Different Configs ===")
print(f"  Active sources: Watchlist Hit, PEP Match, Transaction Flag\n")

for config_name in registry.config_names:
    result = registry.predict(config_name, flags, precision="standard")
    print(f"  {config_name:<12}  "
          f"risk={result['risk_score']:.1%}  "
          f"certainty={result['certainty_label']:<6}  "
          f"driver={result['primary_driver']}")
print()

# -----------------------------------------------------------------------
# 4.  Full single prediction detail for one config
# -----------------------------------------------------------------------
print("=== Full Result — customer_1 ===")
result = registry.predict("customer_1", flags, precision="standard",
                          label="Customer_1_RecordA")
model = registry.get("customer_1")
model.print_result(result)

# -----------------------------------------------------------------------
# 5.  Batch prediction for a specific config
# -----------------------------------------------------------------------
batch_inputs = np.array([
    [1, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 1, 0],
    [1, 1, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1],
])
batch_labels = ["Record_A", "Record_B", "Record_C", "Record_D", "Record_E"]

print("=== Batch Prediction — general config ===")
df = registry.predict_batch("general", batch_inputs,
                             precision="standard", labels=batch_labels)
print(df[["Label", "Risk_Score", "Certainty_Label",
          "Primary_Driver", "Primary_Driver_Contribution"]].to_string(index=False))
print()

# -----------------------------------------------------------------------
# 6.  Cache isolation — hits in one config don't affect another
# -----------------------------------------------------------------------
print("=== Cache Isolation ===")

# Prime the cache for general and customer_1 with the same flags
registry.predict("general",    flags, precision="standard")
registry.predict("general",    flags, precision="standard")  # cache hit
registry.predict("customer_1", flags, precision="standard")
registry.predict("customer_1", flags, precision="standard")  # cache hit

for name in ["general", "customer_1", "customer_2"]:
    info = registry.get(name).cache_info
    print(f"  {name:<12}  {info}")
print()
print("  Each config's cache is completely independent.")
print("  customer_2 has zero hits — it was never called.\n")

# -----------------------------------------------------------------------
# 7.  Reload a single config — others are unaffected
# -----------------------------------------------------------------------
print("=== Single Config Reload ===")

# Update customer_1's parameters using save_config
updated_sources = {
    "Watchlist Hit":        {"mu": 0.90, "kappa": 12.0},
    "Adverse Media":        {"mu": 0.65, "kappa": 9.0},
    "PEP Match":            {"mu": 0.75, "kappa": 7.0},
    "Sanctions Screen":     {"mu": 0.95, "kappa": 14.0},
    "High-Risk Geography":  {"mu": 0.55, "kappa": 9.0},
    "Transaction Flag":     {"mu": 0.45, "kappa": 11.0},
    "Industry Risk":        {"mu": 0.35, "kappa": 8.0},
    "Ownership Complexity": {"mu": 0.50, "kappa": 6.0},
}

before_version = registry.get("customer_1").version
registry.get("customer_1").save_config(
    updated_sources, "configs/customer_1.json", bump="minor"
)

# Only customer_1 is reloaded — general and customer_2 pools are untouched
registry.reload("customer_1")

after_version = registry.get("customer_1").version
print(f"\n  customer_1 version: {before_version} → {after_version}")
print(f"  general version   : {registry.get('general').version}  (unchanged)")
print(f"  customer_2 version: {registry.get('customer_2').version}  (unchanged)\n")

# Cache cleared for customer_1, intact for others
print("  Cache after reload:")
for name in registry.config_names:
    print(f"    {name:<12}  {registry.get(name).cache_info}")
print()

# -----------------------------------------------------------------------
# 8.  Add and remove configs at runtime
# -----------------------------------------------------------------------
print("=== Add / Remove Configs ===")
print(f"  Before: {registry.config_names}")

# Add a new config (reuse customer_2 file as a stand-in)
registry.add("customer_3", "configs/customer_2.json")
print(f"  After add: {registry.config_names}")

registry.remove("customer_3")
print(f"  After remove: {registry.config_names}\n")

# -----------------------------------------------------------------------
# 9.  API routing pattern — how this maps to real endpoint code
# -----------------------------------------------------------------------
print("=== API Routing Pattern ===")
print("""
  # FastAPI example — route each request to the right config by customer ID

  @app.post("/score/{customer_id}")
  def score_record(customer_id: str, body: ScoreRequest):
      config_name = CUSTOMER_CONFIG_MAP.get(customer_id, "general")
      flags = np.array(body.source_flags)
      result = registry.predict(config_name, flags, precision=body.precision)
      return result

  @app.post("/admin/reload/{config_name}")
  def reload_config(config_name: str):
      registry.reload(config_name)
      model = registry.get(config_name)
      return {"version": model.version, "n_sources": model.n_sources}

  @app.get("/status")
  def status():
      return registry.status()
""")
