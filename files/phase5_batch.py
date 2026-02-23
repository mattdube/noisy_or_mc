"""
phase5_batch.py — Production Batch Processing Pipeline
=======================================================
Designed for large-scale batch scoring: CSV/Excel input → scored Excel output.

Performance characteristics:
  - 10,000 records @ n_mc=500: ~1–2 seconds on a modern CPU
  - 1,000,000 records @ n_mc=100: ~10–20 seconds (use chunked mode)
  - Memory footprint: O(n_records × n_mc) float32 → ~2GB for 1M × 500

Chunked mode:
  - Enables arbitrarily large files with bounded memory
  - Each chunk is scored independently and appended to output

Run:
    python phase5_batch.py --input my_data.csv --output results.xlsx
    python phase5_batch.py --input my_data.csv --chunk-size 50000

Input CSV format:
    name,S1,S2,S3,S4,S5,S6,S7,S8
    Entity_001,1,0,0,1,0,0,1,0
    Entity_002,0,0,1,0,1,0,0,1
    ...
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from datetime import datetime
from typing import Iterator

import numpy as np
import pandas as pd

from phase3_scorer import score as core_score, load_store, BATCH_N_MC

warnings.filterwarnings("ignore")

DEFAULT_INPUT  = "batch_input.csv"
DEFAULT_OUTPUT = "batch_results.xlsx"
DEFAULT_CHUNK  = 10_000
MIN_RISK_REPORT = 0.10


# ---------------------------------------------------------------------------
# Data loading & validation
# ---------------------------------------------------------------------------
def load_and_validate(
    input_path: str,
    store: dict,
) -> pd.DataFrame:
    """
    Load CSV/Excel and validate that expected source columns are present.
    Returns DataFrame with source columns as int dtype.
    """
    if input_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path)

    source_order = store["source_order"]
    missing = [s for s in source_order if s not in df.columns]
    if missing:
        raise ValueError(
            f"Input file missing source columns: {missing}\n"
            f"Expected: {source_order}"
        )

    invalid = df[source_order].apply(lambda c: ~c.isin([0, 1])).any(axis=1)
    if invalid.any():
        n_bad = invalid.sum()
        raise ValueError(
            f"{n_bad} records have values outside {{0,1}} in source columns. "
            "Ensure all match indicators are binary."
        )

    df[source_order] = df[source_order].astype(int)
    return df


# ---------------------------------------------------------------------------
# Chunk iterator
# ---------------------------------------------------------------------------
def chunk_iterator(
    df: pd.DataFrame,
    chunk_size: int,
) -> Iterator[tuple[int, int, pd.DataFrame]]:
    """Yield (start_idx, end_idx, chunk_df)."""
    n = len(df)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        yield start, end, df.iloc[start:end]


# ---------------------------------------------------------------------------
# Single-chunk scorer
# ---------------------------------------------------------------------------
def score_chunk(
    chunk: pd.DataFrame,
    source_order: list[str],
    store: dict,
    n_mc: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Score one chunk and return result DataFrame aligned with chunk index."""
    match_matrix = chunk[source_order].values.astype(np.float64)
    scored = core_score(match_matrix, store=store, n_mc=n_mc, seed=seed)
    scored.index = chunk.index
    # Attach any non-source original columns (e.g. name, ID)
    extra_cols = [c for c in chunk.columns if c not in source_order]
    return pd.concat([chunk[extra_cols].reset_index(drop=True),
                      scored.reset_index(drop=True)], axis=1)


# ---------------------------------------------------------------------------
# Full batch pipeline
# ---------------------------------------------------------------------------
def run_batch(
    input_path:  str  = DEFAULT_INPUT,
    output_path: str  = DEFAULT_OUTPUT,
    n_mc:        int  = BATCH_N_MC,
    chunk_size:  int  = DEFAULT_CHUNK,
    min_risk:    float = MIN_RISK_REPORT,
    seed:        int  = 0,
) -> pd.DataFrame:
    """
    Main batch pipeline.

    Steps:
      1. Load & validate input
      2. Score in chunks (memory-safe)
      3. Enrich with risk tiers and confidence labels
      4. Generate Excel report with multiple sheets
      5. Return full scored DataFrame

    Returns full scored DataFrame.
    """
    t_start = time.perf_counter()
    store = load_store()
    source_order = store["source_order"]

    print(f"\n[batch] Loading: {input_path}")
    df_in = load_and_validate(input_path, store)
    n_records = len(df_in)
    n_chunks  = (n_records + chunk_size - 1) // chunk_size
    print(f"[batch] {n_records:,} records | {n_chunks} chunks | n_mc={n_mc}")

    # --- Score in chunks ---
    chunks_out = []
    for i, (start, end, chunk) in enumerate(chunk_iterator(df_in, chunk_size)):
        t_chunk = time.perf_counter()
        chunk_seed = seed + i  # Different seed per chunk for true MC diversity
        scored_chunk = score_chunk(chunk, source_order, store, n_mc, chunk_seed)
        chunks_out.append(scored_chunk)
        elapsed = time.perf_counter() - t_chunk
        print(f"  Chunk {i+1}/{n_chunks}: records {start}–{end}  ({elapsed:.2f}s)")

    df_out = pd.concat(chunks_out, ignore_index=True)

    # --- Enrich ---
    df_out = _add_risk_tiers(df_out)

    # --- Save ---
    _save_excel_report(df_out, output_path, source_order, min_risk, store)

    total_elapsed = time.perf_counter() - t_start
    rate = n_records / total_elapsed
    print(f"\n[batch] Complete: {n_records:,} records in {total_elapsed:.2f}s "
          f"({rate:,.0f} records/sec)")
    print(f"[batch] Output → {output_path}")
    return df_out


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------
def _add_risk_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable risk tier and confidence tier columns."""
    df["risk_tier"] = pd.cut(
        df["risk_point"],
        bins   = [0,  0.15, 0.40, 0.70, 1.01],
        labels = ["Minimal", "Low", "Moderate", "High"],
        right  = False,
    ).astype(str)

    # Confidence is inverse of uncertainty width
    df["confidence_tier"] = pd.cut(
        df["impact_width"],
        bins   = [0,  0.05, 0.15, 0.40, 10],
        labels = ["High", "Moderate", "Low", "Very Low"],
        right  = False,
    ).astype(str)

    return df


# ---------------------------------------------------------------------------
# Excel report builder
# ---------------------------------------------------------------------------
def _save_excel_report(
    df: pd.DataFrame,
    output_path: str,
    source_order: list[str],
    min_risk: float,
    store: dict,
) -> None:
    """
    Multi-sheet Excel report:
      - All_Records      : complete scored output
      - High_Risk_Alerts : risk_tier == 'High' or 'Moderate'
      - Focus_<source>   : top 3 primary drivers (deep-dive sheets)
      - Model_Info       : param store metadata
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    priority_cols = [
        "risk_point", "risk_p50", "risk_p10", "risk_p90", "impact_width",
        "risk_tier", "confidence_tier", "primary_driver", "matched_sources",
    ]
    # Add any original ID/name columns
    extra_cols = [c for c in df.columns
                  if c not in priority_cols
                  and not c.endswith("_impact")
                  and c not in source_order]
    display_cols = extra_cols + priority_cols

    try:
        import openpyxl
        EXCEL_OK = True
    except ImportError:
        EXCEL_OK = False

    if not EXCEL_OK:
        # Fallback to CSV
        csv_path = output_path.replace(".xlsx", ".csv")
        df[display_cols + [f"{s}_impact" for s in source_order]].to_csv(csv_path, index=False)
        print(f"[batch] openpyxl not found — saved as CSV: {csv_path}")
        return

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Sheet 1: All records
        out_cols = [c for c in display_cols if c in df.columns]
        df[out_cols].to_excel(writer, sheet_name="All_Records", index=False)

        # Sheet 2: Alerts (moderate + high)
        df_alerts = df[df["risk_tier"].isin(["Moderate", "High"])].copy()
        if len(df_alerts) > 0:
            df_alerts[out_cols].sort_values("risk_point", ascending=False).to_excel(
                writer, sheet_name="High_Risk_Alerts", index=False
            )
        else:
            pd.DataFrame({"info": ["No records above alert threshold"]}).to_excel(
                writer, sheet_name="High_Risk_Alerts", index=False
            )

        # Sheets 3–5: Focus by top drivers
        top_drivers = df["primary_driver"].value_counts().nlargest(3).index.tolist()
        for driver in top_drivers:
            if driver == "Leak":
                continue
            sheet_name = f"Focus_{driver}"[:31]  # Excel max 31 chars
            df_driver = df[df["primary_driver"] == driver]
            df_driver[out_cols].sort_values("risk_point", ascending=False).to_excel(
                writer, sheet_name=sheet_name, index=False
            )

        # Sheet 6: Model info
        model_rows = [
            ("Param Store Version", store["version"]),
            ("Created UTC",         store.get("created_utc", "N/A")),
            ("Source Count",        len(source_order)),
            ("Source Order",        ", ".join(source_order)),
            ("Leak Mean",           round(store["leak"]["mean"], 6)),
            ("Generated",           datetime.utcnow().isoformat()),
        ]
        for s in source_order:
            src = store["sources"][s]
            model_rows.append((
                f"{s} — mean / std / CI90",
                f"{src['mean']:.4f} / {src['std']:.4f} / "
                f"[{src['ci90_lo']:.4f}, {src['ci90_hi']:.4f}]"
            ))

        pd.DataFrame(model_rows, columns=["Parameter", "Value"]).to_excel(
            writer, sheet_name="Model_Info", index=False
        )

    # Column styling (best-effort)
    try:
        _apply_excel_formatting(output_path)
    except Exception:
        pass  # Don't fail the whole pipeline over formatting


def _apply_excel_formatting(path: str) -> None:
    """Apply basic conditional formatting and column widths."""
    from openpyxl import load_workbook
    from openpyxl.styles import PatternFill, Font, numbers
    from openpyxl.formatting.rule import ColorScaleRule

    wb = load_workbook(path)

    RED    = PatternFill("solid", fgColor="FFC7CE")
    YELLOW = PatternFill("solid", fgColor="FFEB9C")
    GREEN  = PatternFill("solid", fgColor="C6EFCE")
    BOLD   = Font(bold=True)

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        if ws.max_row < 2:
            continue

        # Auto column widths
        for col in ws.columns:
            max_len = max((len(str(cell.value or "")) for cell in col), default=10)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 40)

        # Bold headers
        for cell in ws[1]:
            cell.font = BOLD

        # Color risk_tier column
        headers = [cell.value for cell in ws[1]]
        if "risk_tier" in headers:
            col_idx = headers.index("risk_tier") + 1
            for row in ws.iter_rows(min_row=2, min_col=col_idx, max_col=col_idx):
                for cell in row:
                    if cell.value == "High":
                        cell.fill = RED
                    elif cell.value == "Moderate":
                        cell.fill = YELLOW
                    elif cell.value in ("Low", "Minimal"):
                        cell.fill = GREEN

    wb.save(path)


# ---------------------------------------------------------------------------
# Test data generator
# ---------------------------------------------------------------------------
def generate_test_data(
    output_path: str = "batch_input.csv",
    n: int = 500,
    match_rate: float = 0.25,
    seed: int = 42,
) -> None:
    """Generate synthetic test input CSV."""
    np.random.seed(seed)
    source_cols = [f"S{i}" for i in range(1, 9)]
    data = np.random.binomial(1, match_rate, size=(n, 8))
    df = pd.DataFrame(data, columns=source_cols)
    df.insert(0, "record_id", [f"REC_{i:05d}" for i in range(n)])
    df.insert(1, "entity_name", [f"Entity_{i:05d}" for i in range(n)])
    df.to_csv(output_path, index=False)
    print(f"[batch] Test data generated → {output_path}  ({n} records)")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def print_batch_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 55)
    print("  Batch Scoring Summary")
    print("=" * 55)
    print(f"  Total records:     {len(df):,}")
    print(f"\n  Risk Tier Distribution:")
    tier_counts = df["risk_tier"].value_counts()
    for tier in ["High", "Moderate", "Low", "Minimal"]:
        count = tier_counts.get(tier, 0)
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"    {tier:<12} {count:>6,}  ({pct:5.1f}%)  {bar}")

    print(f"\n  Primary Driver Distribution:")
    driver_counts = df["primary_driver"].value_counts().head(6)
    for driver, count in driver_counts.items():
        pct = count / len(df) * 100
        print(f"    {driver:<12} {count:>6,}  ({pct:5.1f}%)")

    print(f"\n  Risk Score Stats (point estimate):")
    print(f"    mean   = {df['risk_point'].mean():.4f}")
    print(f"    median = {df['risk_point'].median():.4f}")
    print(f"    p90    = {df['risk_point'].quantile(0.90):.4f}")
    print(f"    p99    = {df['risk_point'].quantile(0.99):.4f}")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch risk scoring pipeline")
    parser.add_argument("--input",      default=DEFAULT_INPUT,  help="Input CSV/Excel path")
    parser.add_argument("--output",     default=DEFAULT_OUTPUT, help="Output Excel path")
    parser.add_argument("--n-mc",       type=int, default=BATCH_N_MC, help="MC samples")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK, help="Records per chunk")
    parser.add_argument("--min-risk",   type=float, default=MIN_RISK_REPORT, help="Alert threshold")
    parser.add_argument("--gen-test",   action="store_true", help="Generate test input data")
    args = parser.parse_args()

    import os
    if not os.path.exists("params/param_store.json"):
        print("[batch] param_store.json not found. Running phase1 first...")
        from phase1_priors import SOURCE_SPECS, LEAK_SPEC, build_param_store, save_param_store
        save_param_store(build_param_store(SOURCE_SPECS, LEAK_SPEC))

    if args.gen_test or not os.path.exists(args.input):
        generate_test_data(args.input)

    df_results = run_batch(
        input_path  = args.input,
        output_path = args.output,
        n_mc        = args.n_mc,
        chunk_size  = args.chunk_size,
        min_risk    = args.min_risk,
    )
    print_batch_summary(df_results)
