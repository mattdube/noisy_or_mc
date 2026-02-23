"""
run_pipeline.py — End-to-End Pipeline Runner
=============================================
Runs all phases in sequence for demonstration and smoke-testing.

Usage:
    python run_pipeline.py              # Full pipeline
    python run_pipeline.py --phase 3    # Only phase 3+
    python run_pipeline.py --skip-plots # Skip matplotlib plots
"""

import argparse
import os
import sys
import time


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def main(start_phase: int = 1, skip_plots: bool = False):
    t0 = time.perf_counter()

    # -----------------------------------------------------------------------
    # Phase 1: Expert priors → param store
    # -----------------------------------------------------------------------
    if start_phase <= 1:
        section("Phase 1: Expert Prior Specification")
        from phase1_priors import SOURCE_SPECS, LEAK_SPEC, build_param_store, save_param_store, print_prior_summary
        store = build_param_store(SOURCE_SPECS, LEAK_SPEC)
        print_prior_summary(store)
        save_param_store(store)

        # Demo: conjugate update on S2
        from phase1_priors import bayesian_update_source
        print("\n[demo] Conjugate update on S2 (3 hits in 10 analyst reviews):")
        store_updated = bayesian_update_source(store.copy(), "S2", 3, 10)
        s2 = store_updated["sources"]["S2"]
        print(f"  S2 mean: {SOURCE_SPECS['S2'][0]:.4f} → {s2['mean']:.4f}  "
              f"(std: {s2['std']:.4f})")

    # -----------------------------------------------------------------------
    # Phase 2: Simulate & validate
    # -----------------------------------------------------------------------
    if start_phase <= 2:
        section("Phase 2: Prior Predictive Simulation & Validation")
        from phase1_priors import load_param_store
        from phase2_simulate_validate import (
            validate_priors, print_validation,
            source_sensitivity, save_validation_results,
        )
        store = load_param_store("params/param_store.json")

        checks = validate_priors(store, n_mc=5_000)
        print_validation(checks)

        print("[phase2] Source sensitivity ranking:")
        sens = source_sensitivity(store, n_mc=2_000)
        for s, v in sens.items():
            bar = "█" * int(v * 200)
            print(f"  {s}: {v:.4f}  {bar}")

        save_validation_results(checks, sens)

        if not skip_plots:
            from phase2_simulate_validate import save_plots
            save_plots(store, n_mc=5_000)

    # -----------------------------------------------------------------------
    # Phase 3: Scorer tests
    # -----------------------------------------------------------------------
    if start_phase <= 3:
        section("Phase 3: Core Scorer Tests")
        import numpy as np
        from phase1_priors import load_param_store
        from phase3_scorer import score_single, score

        store = load_param_store("params/param_store.json")

        test_cases = [
            (np.array([1, 0, 0, 1, 0, 0, 1, 0]), "S1+S4+S7"),
            (np.array([0, 0, 0, 0, 0, 0, 0, 0]), "No matches"),
            (np.array([1, 1, 1, 1, 1, 1, 1, 1]), "All matched"),
        ]

        for vec, label in test_cases:
            r = score_single(vec, store=store, n_mc=2_000)
            print(f"\n  [{label}]")
            print(f"    risk_point={r['risk_point']:.4f}  "
                  f"CI=[{r['risk_p10']:.4f}, {r['risk_p90']:.4f}]  "
                  f"driver={r['primary_driver']}")

        print("\n  Batch scorer (100 records):")
        np.random.seed(42)
        mat = np.random.binomial(1, 0.25, (100, 8)).astype(float)
        df = score(mat, store=store, n_mc=500)
        print(f"    risk_point: mean={df['risk_point'].mean():.4f}  "
              f"max={df['risk_point'].max():.4f}")
        print(f"    Primary drivers:\n{df['primary_driver'].value_counts().to_string()}")

    # -----------------------------------------------------------------------
    # Phase 4: API demo (no server required)
    # -----------------------------------------------------------------------
    if start_phase <= 4:
        section("Phase 4: API Layer Demo")
        from phase4_api import demo_without_api
        demo_without_api()

    # -----------------------------------------------------------------------
    # Phase 5: Batch pipeline
    # -----------------------------------------------------------------------
    if start_phase <= 5:
        section("Phase 5: Batch Processing Pipeline")
        from phase5_batch import generate_test_data, run_batch, print_batch_summary

        if not os.path.exists("batch_input.csv"):
            generate_test_data("batch_input.csv", n=300)

        df = run_batch(
            input_path  = "batch_input.csv",
            output_path = "batch_results.xlsx",
            n_mc        = 300,
            chunk_size  = 100,
        )
        print_batch_summary(df)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total = time.perf_counter() - t0
    section(f"Pipeline Complete — {total:.1f}s total")
    print("  Generated files:")
    for f in [
        "params/param_store.json",
        "validation/validation_results.json",
        "validation/plots/prior_distributions.png",
        "validation/plots/risk_score_distribution.png",
        "validation/plots/source_sensitivity.png",
        "batch_input.csv",
        "batch_results.xlsx",
    ]:
        status = "✓" if os.path.exists(f) else "✗ (not created)"
        print(f"    {status}  {f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",       type=int, default=1, help="Start from phase N")
    parser.add_argument("--skip-plots",  action="store_true")
    args = parser.parse_args()
    main(start_phase=args.phase, skip_plots=args.skip_plots)
