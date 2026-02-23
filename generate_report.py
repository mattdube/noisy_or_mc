"""
generate_report.py — Model & Prediction Report Generator
=========================================================
Produces a fully-formatted Excel workbook with:

  Summary          — model metadata, overall stats, top drivers, risk tier breakdown
  Source_S1 … S8   — one tab per source: Beta parameters, CI, sensitivity analysis,
                     and every prediction where that source was matched or primary driver

Usage:

    from generate_report import generate_report
    from phase1_priors import load_param_store
    from phase3_scorer import score
    import numpy as np, pandas as pd

    store       = load_param_store()
    predictions = score(match_matrix, store=store, n_mc=1000)
    # Optionally attach your own ID/name columns before passing:
    # predictions.insert(0, "entity", ["ACME", ...])

    generate_report(store, predictions, output_path="model_report.xlsx")

Or run standalone to generate a report on synthetic test data:

    python generate_report.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

from openpyxl import Workbook
from openpyxl.styles import (
    Alignment, Border, Font, GradientFill, PatternFill, Side
)
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule, DataBarRule

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY       = "0B1120"
DARK_PANEL = "0F172A"
SLATE      = "1E293B"
MID_GREY   = "475569"
LIGHT_GREY = "94A3B8"
OFF_WHITE  = "F1F5F9"
WHITE      = "FFFFFF"

# Risk tier colours (ARGB — no leading #)
TIER_FILLS = {
    "High":     "FFC7CE",   # red tint
    "Moderate": "FFEB9C",   # amber tint
    "Low":      "C6EFCE",   # green tint
    "Minimal":  "EFF6FF",   # blue tint
}
TIER_FONTS = {
    "High":     "9C0006",
    "Moderate": "7D6608",
    "Low":      "375623",
    "Minimal":  "1D4ED8",
}

SOURCE_COLORS = {
    "S1": "22D3EE", "S2": "A78BFA", "S3": "34D399", "S4": "F87171",
    "S5": "60A5FA", "S6": "FB923C", "S7": "E879F9", "S8": "94A3B8",
}


# ── Style helpers ─────────────────────────────────────────────────────────────
def _font(bold=False, size=10, color=None, italic=False):
    return Font(name="Arial", size=size, bold=bold, italic=italic,
                color=color or "000000")


def _fill(hex_color: str):
    return PatternFill("solid", fgColor=hex_color)


def _border(style="thin", color="D1D5DB"):
    s = Side(style=style, color=color)
    return Border(left=s, right=s, top=s, bottom=s)


def _bottom_border(color="D1D5DB"):
    s = Side(style="thin", color=color)
    return Border(bottom=s)


def _align(h="left", v="center", wrap=False):
    return Alignment(horizontal=h, vertical=v, wrap_text=wrap)


def _set_col_width(ws, col_idx: int, width: float):
    ws.column_dimensions[get_column_letter(col_idx)].width = width


def _header_row(ws, row: int, values: list[str], fill_hex: str,
                font_color: str = WHITE, font_size: int = 10):
    """Write a styled header row and return the row."""
    for ci, val in enumerate(values, 1):
        c = ws.cell(row=row, column=ci, value=val)
        c.font      = _font(bold=True, color=font_color, size=font_size)
        c.fill      = _fill(fill_hex)
        c.alignment = _align("center")
        c.border    = _border(color=SLATE)
    return row


def _section_title(ws, row: int, col: int, text: str,
                   fill_hex: str = SLATE, span: int = 1):
    # Write title to the first cell only — no merging.
    # Merging causes MergedCell conflicts when later rows write to the same columns.
    c = ws.cell(row=row, column=col, value=text)
    c.font      = _font(bold=True, size=10, color=OFF_WHITE)
    c.fill      = _fill(fill_hex)
    c.alignment = _align("left")
    # Fill remaining span cells with the same background (no merge)
    for ci in range(col + 1, col + span):
        ec = ws.cell(row=row, column=ci, value=None)
        ec.fill = _fill(fill_hex)
    return row + 1


def _kv_row(ws, row: int, label: str, value, label_col: int = 1,
            pct: bool = False, dp: int = 4):
    """Write a label-value pair."""
    lc = ws.cell(row=row, column=label_col, value=label)
    lc.font      = _font(color=MID_GREY)
    lc.alignment = _align("left")
    lc.fill      = _fill("F8FAFC")

    vc = ws.cell(row=row, column=label_col + 1, value=value)
    vc.font      = _font(bold=True)
    vc.alignment = _align("right")
    vc.fill      = _fill("F8FAFC")
    if pct and isinstance(value, float):
        vc.number_format = "0.0%"
    elif isinstance(value, float):
        vc.number_format = f"0.{'0'*dp}"
    vc.border = _bottom_border()
    return row + 1


def _risk_tier(score_val: float) -> str:
    if score_val >= 0.70: return "High"
    if score_val >= 0.40: return "Moderate"
    if score_val >= 0.15: return "Low"
    return "Minimal"


# ── Main report function ──────────────────────────────────────────────────────
def generate_report(
    store:        dict,
    predictions:  pd.DataFrame,
    output_path:  str = "model_report.xlsx",
    model_name:   str = "Noisy-OR Risk Model",
    n_mc_used:    int = 1000,
) -> str:
    """
    Generate a full Excel report from a param store and a scored predictions
    DataFrame (as returned by phase3_scorer.score or phase5_batch.run_batch).

    Parameters
    ----------
    store        : dict  — loaded param store (load_param_store())
    predictions  : pd.DataFrame — scored records; must contain risk_point,
                   primary_driver, matched_sources, and S{n}_impact columns
    output_path  : str   — where to save the .xlsx file
    model_name   : str   — display name shown in the report header
    n_mc_used    : int   — MC samples used when scoring (for documentation only)

    Returns
    -------
    str — absolute path of the saved file
    """
    source_order = store["source_order"]
    sources      = store["sources"]
    leak         = store["leak"]
    df           = predictions.copy()

    # Ensure risk_tier column exists
    if "risk_tier" not in df.columns:
        df["risk_tier"] = df["risk_point"].apply(_risk_tier)

    wb = Workbook()
    wb.remove(wb.active)   # remove default blank sheet

    _build_summary_sheet(wb, store, df, source_order, sources, leak,
                         model_name, n_mc_used)

    for src in source_order:
        _build_source_sheet(wb, src, sources[src], df, source_order)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    wb.save(output_path)
    print(f"[report] Saved → {output_path}")
    return os.path.abspath(output_path)


# ── Summary sheet ─────────────────────────────────────────────────────────────
def _build_summary_sheet(wb, store, df, source_order, sources, leak,
                         model_name, n_mc_used):
    ws = wb.create_sheet("Summary")
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "A6"

    # ── Banner ────────────────────────────────────────────────────────────────
    ws.merge_cells("A1:L1")
    title_cell = ws["A1"]
    title_cell.value     = model_name
    title_cell.font      = _font(bold=True, size=16, color=OFF_WHITE)
    title_cell.fill      = _fill(NAVY)
    title_cell.alignment = _align("left", "center")
    ws.row_dimensions[1].height = 36

    ws.merge_cells("A2:L2")
    sub = ws["A2"]
    sub.value     = (f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}  ·  "
                     f"Param store v{store['version']}  ·  {len(df):,} records scored  ·  "
                     f"n_mc={n_mc_used}")
    sub.font      = _font(size=9, color=LIGHT_GREY, italic=True)
    sub.fill      = _fill(DARK_PANEL)
    sub.alignment = _align("left", "center")
    ws.row_dimensions[2].height = 20

    for row in [3, 4]:
        ws.merge_cells(f"A{row}:L{row}")
        ws[f"A{row}"].fill = _fill(NAVY)
        ws.row_dimensions[row].height = 6

    # ── Section A: Model metadata (cols A–D) ──────────────────────────────────
    row = 5
    row = _section_title(ws, row, 1, "  Model Configuration", NAVY, span=4)

    meta = [
        ("Param Store Version", store["version"]),
        ("Created (UTC)",       store.get("created_utc", "N/A")[:19].replace("T", " ")),
        ("Source Count",        len(source_order)),
        ("Source Order",        ", ".join(source_order)),
        ("Leak — Mean",         leak["mean"]),
        ("Leak — Std",          leak["std"]),
        ("Leak — α",            leak["alpha"]),
        ("Leak — β",            leak["beta"]),
        ("Records Scored",      len(df)),
        ("MC Samples (n_mc)",   n_mc_used),
    ]
    for label, val in meta:
        row = _kv_row(ws, row, f"  {label}", val, label_col=1)

    # ── Section B: Risk tier summary (cols A–D, continuing) ───────────────────
    row += 1
    row = _section_title(ws, row, 1, "  Risk Tier Distribution", SLATE, span=4)
    _header_row(ws, row, ["Tier", "Count", "Pct", "Avg Score"], NAVY)
    row += 1

    tier_order = ["High", "Moderate", "Low", "Minimal"]
    for tier in tier_order:
        sub_df = df[df["risk_tier"] == tier]
        count  = len(sub_df)
        pct    = count / len(df) if len(df) else 0
        avg    = sub_df["risk_point"].mean() if count else 0

        cells = [tier, count, pct, avg]
        for ci, val in enumerate(cells, 1):
            c = ws.cell(row=row, column=ci, value=val)
            c.fill      = _fill(TIER_FILLS.get(tier, "FFFFFF"))
            c.font      = _font(color=TIER_FONTS.get(tier, "000000"),
                                bold=(ci == 1))
            c.alignment = _align("center" if ci > 1 else "left")
            c.border    = _border(color="D1D5DB")
            if ci == 3:
                c.number_format = "0.0%"
            elif ci == 4:
                c.number_format = "0.0000"
        row += 1

    # ── Section C: Portfolio risk stats (cols A–D, continuing) ───────────────
    row += 1
    row = _section_title(ws, row, 1, "  Portfolio Risk Statistics", SLATE, span=4)
    stats = [
        ("Mean Risk Score",    df["risk_point"].mean()),
        ("Median Risk Score",  df["risk_point"].median()),
        ("Std Dev",            df["risk_point"].std()),
        ("P90 Risk Score",     df["risk_point"].quantile(0.90)),
        ("P99 Risk Score",     df["risk_point"].quantile(0.99)),
        ("Max Risk Score",     df["risk_point"].max()),
        ("Min Risk Score",     df["risk_point"].min()),
        ("Mean CI Width",      df["impact_width"].mean() if "impact_width" in df.columns else "N/A"),
    ]
    for label, val in stats:
        row = _kv_row(ws, row, f"  {label}", val, label_col=1)

    # ── Section D: Source sensitivity (cols F–I) ──────────────────────────────
    col_offset = 6  # column F
    row_d = 5
    row_d = _section_title(ws, row_d, col_offset, "  Source Prior Summary", NAVY, span=5)
    _header_row(ws, row_d, ["Source", "Mean", "Std", "α", "β", "Conc.", "90% CI Lo", "90% CI Hi"],
                NAVY, font_color=WHITE)
    row_d += 1

    for src in source_order:
        s     = sources[src]
        ci_lo = beta_dist.ppf(0.05, s["alpha"], s["beta"])
        ci_hi = beta_dist.ppf(0.95, s["alpha"], s["beta"])
        vals  = [src, s["mean"], s["std"], s["alpha"], s["beta"],
                 s["expert_concentration"], ci_lo, ci_hi]
        src_color = SOURCE_COLORS.get(src, "94A3B8")
        for ci, val in enumerate(vals, col_offset):
            c = ws.cell(row=row_d, column=ci, value=val)
            c.alignment = _align("center" if ci > col_offset else "left")
            c.border    = _border(color="E2E8F0")
            c.number_format = "0.0000"
            if ci == col_offset:  # source name cell
                c.font = _font(bold=True, color=src_color)
                c.fill = _fill("F8FAFC")
            else:
                c.font = _font(color="1E293B")
                c.fill = _fill("FFFFFF" if row_d % 2 == 0 else "F8FAFC")
        row_d += 1

    # ── Section E: Primary driver frequency (cols F–I, continuing) ───────────
    row_d += 1
    row_d = _section_title(ws, row_d, col_offset, "  Primary Driver Frequency", SLATE, span=4)
    _header_row(ws, row_d, ["Driver", "Count", "Pct of Records", "Avg Impact"],
                NAVY, font_color=WHITE)
    row_d += 1

    driver_counts = df["primary_driver"].value_counts()
    for driver, count in driver_counts.items():
        pct      = count / len(df)
        src_df   = df[df["primary_driver"] == driver]
        imp_col  = f"{driver}_impact"
        avg_imp  = src_df[imp_col].mean() if imp_col in df.columns else 0
        src_color = SOURCE_COLORS.get(driver, MID_GREY)

        for ci, val in enumerate([driver, count, pct, avg_imp], col_offset):
            c = ws.cell(row=row_d, column=ci, value=val)
            c.alignment = _align("center" if ci > col_offset else "left")
            c.border    = _border(color="E2E8F0")
            c.fill      = _fill("FFFFFF" if row_d % 2 == 0 else "F8FAFC")
            if ci == col_offset:
                c.font = _font(bold=True, color=src_color)
            elif ci == col_offset + 2:
                c.number_format = "0.0%"
                c.font = _font()
            elif ci == col_offset + 3:
                c.number_format = "0.0000"
                c.font = _font()
            else:
                c.font = _font()
        row_d += 1

    # ── Column widths ─────────────────────────────────────────────────────────
    widths = {1: 28, 2: 16, 3: 10, 4: 10,
              6: 10, 7: 10, 8: 10, 9: 10, 10: 10, 11: 12, 12: 12, 13: 12}
    for col, w in widths.items():
        _set_col_width(ws, col, w)

    # Spacer column E
    _set_col_width(ws, 5, 3)


# ── Per-source sheet ──────────────────────────────────────────────────────────
def _build_source_sheet(wb, src: str, src_params: dict,
                         df: pd.DataFrame, source_order: list[str]):
    ws = wb.create_sheet(f"Source_{src}")
    ws.sheet_view.showGridLines = False

    src_color  = SOURCE_COLORS.get(src, "94A3B8")
    imp_col    = f"{src}_impact"

    # ── Banner ────────────────────────────────────────────────────────────────
    ws.merge_cells("A1:M1")
    banner = ws["A1"]
    banner.value     = f"Source {src}  —  Prior Distribution & Prediction Analysis"
    banner.font      = _font(bold=True, size=14, color=WHITE)
    banner.fill      = _fill(src_color)
    banner.alignment = _align("left", "center")
    ws.row_dimensions[1].height = 30

    ws.merge_cells("A2:M2")
    sub = ws["A2"]
    sub.value     = (f"{src_params.get('expert_concentration', 'N/A')} effective observations  ·  "
                     f"Mean {src_params['mean']:.4f}  ·  "
                     f"Std {src_params['std']:.4f}  ·  "
                     f"Beta(α={src_params['alpha']:.2f}, β={src_params['beta']:.2f})")
    sub.font      = _font(size=9, color="1E293B", italic=True)
    sub.fill      = _fill("F0F9FF")
    sub.alignment = _align("left", "center")
    ws.row_dimensions[2].height = 18

    # ── Section A: Beta parameters ────────────────────────────────────────────
    row = 4
    row = _section_title(ws, row, 1, "  Beta Distribution Parameters", src_color, span=3)

    ci_lo = beta_dist.ppf(0.05, src_params["alpha"], src_params["beta"])
    ci_hi = beta_dist.ppf(0.95, src_params["alpha"], src_params["beta"])
    ci_80_lo = beta_dist.ppf(0.10, src_params["alpha"], src_params["beta"])
    ci_80_hi = beta_dist.ppf(0.90, src_params["alpha"], src_params["beta"])

    params = [
        ("Expert Mean",          src_params["mean"],                   False),
        ("Expert Std",           src_params["std"],                    False),
        ("Alpha (α)",            src_params["alpha"],                  False),
        ("Beta (β)",             src_params["beta"],                   False),
        ("Concentration",        src_params["expert_concentration"],   False),
        ("90% CI — Lower",       ci_lo,                                False),
        ("90% CI — Upper",       ci_hi,                                False),
        ("80% CI — Lower",       ci_80_lo,                             False),
        ("80% CI — Upper",       ci_80_hi,                             False),
        ("CI Width (80%)",       ci_80_hi - ci_80_lo,                  False),
    ]
    for label, val, pct in params:
        row = _kv_row(ws, row, f"  {label}", val, label_col=1, pct=pct)

    # ── Section B: Confidence interpretation ──────────────────────────────────
    row += 1
    row = _section_title(ws, row, 1, "  Confidence Interpretation", SLATE, span=3)

    conc = src_params["expert_concentration"]
    ci_w = ci_80_hi - ci_80_lo
    conf_tier = (
        "High"      if ci_w < 0.05  else
        "Moderate"  if ci_w < 0.15  else
        "Low"       if ci_w < 0.40  else
        "Very Low"
    )
    interp = {
        "High":     "Narrow CI — score is reliable, trust the point estimate.",
        "Moderate": "Moderate CI — flag uncertainty in high-stakes decisions.",
        "Low":      "Wide CI — recommend manual review for high-risk records.",
        "Very Low": "Very wide CI — do not act on score alone; gather more data.",
    }
    interp_rows = [
        ("Confidence Tier",      conf_tier),
        ("80% CI Width",         f"{ci_w:.4f}"),
        ("Interpretation",       interp[conf_tier]),
        ("Effective Obs.",       f"{conc} (higher = narrower distribution)"),
    ]
    for label, val in interp_rows:
        lc = ws.cell(row=row, column=1, value=f"  {label}")
        lc.font      = _font(color=MID_GREY)
        lc.fill      = _fill("F8FAFC")
        vc = ws.cell(row=row, column=2, value=val)
        vc.font      = _font(bold=True, color=src_color if label == "Confidence Tier" else "000000")
        vc.fill      = _fill("F8FAFC")
        vc.alignment = _align("left")
        ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=3)
        row += 1

    # ── Section C: Prediction summary for this source ─────────────────────────
    row += 1
    row = _section_title(ws, row, 1, "  Prediction Summary for this Source", src_color, span=3)

    matched_df  = df[df["matched_sources"].str.contains(src, na=False)] if "matched_sources" in df.columns else df.iloc[0:0]
    primary_df  = df[df["primary_driver"] == src]
    n_matched   = len(matched_df)
    n_primary   = len(primary_df)
    n_total     = len(df)
    avg_impact  = matched_df[imp_col].mean() if imp_col in matched_df.columns and n_matched else 0
    avg_risk_when_matched = matched_df["risk_point"].mean() if n_matched else 0
    avg_risk_overall      = df["risk_point"].mean()

    pred_stats = [
        ("Records where matched",        n_matched),
        ("Match rate",                   n_matched / n_total if n_total else 0),
        ("Records as primary driver",    n_primary),
        ("Primary driver rate",          n_primary / n_total if n_total else 0),
        ("Avg impact when matched",      avg_impact),
        ("Avg risk when matched",        avg_risk_when_matched),
        ("Avg risk overall (all records)", avg_risk_overall),
        ("Risk lift from match",         avg_risk_when_matched - avg_risk_overall),
    ]
    for i, (label, val) in enumerate(pred_stats):
        pct_fmt = "match rate" in label or "driver rate" in label
        row = _kv_row(ws, row, f"  {label}", val, label_col=1, pct=pct_fmt)

    # ── Section D: Predictions table ─────────────────────────────────────────
    row += 1
    row = _section_title(ws, row, 1,
                         f"  Records Where {src} Matched  ({n_matched} records)",
                         SLATE, span=13)

    # Determine which extra columns exist (e.g. record_id, entity_name)
    score_cols  = ["risk_point", "risk_p10", "risk_p50", "risk_p90",
                   "impact_width", "risk_tier", "primary_driver",
                   "matched_sources", imp_col]
    extra_cols  = [c for c in df.columns
                   if c not in score_cols
                   and not c.endswith("_impact")
                   and c not in source_order
                   and c not in ["risk_std", "risk_tier"]]
    table_cols  = extra_cols + score_cols
    table_cols  = [c for c in table_cols if c in df.columns]

    header_labels = {
        "risk_point":      "Risk Score",
        "risk_p10":        "P10",
        "risk_p50":        "P50 (Median)",
        "risk_p90":        "P90",
        "impact_width":    "CI Width",
        "risk_tier":       "Tier",
        "primary_driver":  "Primary Driver",
        "matched_sources": "Matched Sources",
        imp_col:           f"{src} Impact",
    }
    display_headers = [header_labels.get(c, c.replace("_", " ").title())
                       for c in table_cols]

    _header_row(ws, row, display_headers, src_color, font_color=WHITE)
    header_row_num = row
    row += 1

    # Write matched records, sorted by impact descending
    table_data = matched_df[table_cols].copy() if n_matched else df[table_cols].iloc[0:0]
    if imp_col in table_data.columns:
        table_data = table_data.sort_values(imp_col, ascending=False)

    data_start_row = row
    for rec_row in table_data.itertuples(index=False):
        values = list(rec_row)
        for ci, (col_name, val) in enumerate(zip(table_cols, values), 1):
            c = ws.cell(row=row, column=ci, value=val)
            c.alignment = _align("center" if col_name != "matched_sources" else "left")
            c.font       = _font(size=9)
            c.border     = _border(color="E2E8F0")
            c.fill       = _fill("FFFFFF" if row % 2 == 0 else "F8FAFC")

            # Tier colouring
            if col_name == "risk_tier" and val in TIER_FILLS:
                c.fill = _fill(TIER_FILLS[val])
                c.font = _font(size=9, color=TIER_FONTS[val], bold=True)
            elif col_name == "risk_point":
                c.number_format = "0.0000"
            elif col_name in ("risk_p10", "risk_p50", "risk_p90", imp_col, "impact_width"):
                c.number_format = "0.0000"
            elif col_name == "primary_driver":
                c.font = _font(size=9, color=SOURCE_COLORS.get(val, MID_GREY), bold=True)
        row += 1

    # Conditional colour scale on Risk Score column
    if n_matched > 0:
        risk_col_idx   = table_cols.index("risk_point") + 1 if "risk_point" in table_cols else None
        impact_col_idx = table_cols.index(imp_col) + 1 if imp_col in table_cols else None
        data_end_row   = row - 1

        if risk_col_idx and data_end_row >= data_start_row:
            risk_col_letter = get_column_letter(risk_col_idx)
            ws.conditional_formatting.add(
                f"{risk_col_letter}{data_start_row}:{risk_col_letter}{data_end_row}",
                ColorScaleRule(
                    start_type="min",  start_color="C6EFCE",
                    mid_type="num",    mid_value=0.4, mid_color="FFEB9C",
                    end_type="max",    end_color="FFC7CE",
                ),
            )
        if impact_col_idx and data_end_row >= data_start_row:
            imp_col_letter = get_column_letter(impact_col_idx)
            ws.conditional_formatting.add(
                f"{imp_col_letter}{data_start_row}:{imp_col_letter}{data_end_row}",
                DataBarRule(start_type="min", end_type="max",
                            color=src_color.lstrip("#") if src_color.startswith("#") else src_color),
            )

    # ── Column widths ─────────────────────────────────────────────────────────
    base_widths = {1: 26, 2: 16}
    score_widths = {
        "risk_point": 12, "risk_p10": 9, "risk_p50": 12, "risk_p90": 9,
        "impact_width": 10, "risk_tier": 11, "primary_driver": 14,
        "matched_sources": 22, imp_col: 13,
    }
    for ci, col_name in enumerate(table_cols, 1):
        w = score_widths.get(col_name, base_widths.get(ci, 14))
        _set_col_width(ws, ci, w)

    ws.freeze_panes = f"A{header_row_num + 1}"


# ── Standalone smoke test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from phase1_priors import load_param_store, build_param_store, SOURCE_SPECS, LEAK_SPEC
    from phase3_scorer import score, reload_store

    PARAM_PATH = "params/param_store.json"
    if os.path.exists(PARAM_PATH):
        store = load_param_store(PARAM_PATH)
    else:
        from phase1_priors import save_param_store
        store = build_param_store(SOURCE_SPECS, LEAK_SPEC)
        save_param_store(store)

    reload_store(PARAM_PATH)

    # Generate 300 test records
    np.random.seed(42)
    n = 300
    source_order = store["source_order"]
    mat = np.random.binomial(1, 0.28, (n, 8)).astype(float)

    print("Scoring test records...")
    predictions = score(mat, store=store, n_mc=1000, seed=0)

    # Attach synthetic ID columns
    predictions.insert(0, "record_id",   [f"REC_{i:05d}" for i in range(n)])
    predictions.insert(1, "entity_name", [f"Entity_{i:05d}" for i in range(n)])

    from phase5_batch import _add_risk_tiers
    predictions = _add_risk_tiers(predictions)

    output = "model_report.xlsx"
    generate_report(store, predictions, output_path=output, n_mc_used=1000)
    print(f"Report saved to: {output}")
