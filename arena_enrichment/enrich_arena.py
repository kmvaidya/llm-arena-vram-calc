#!/usr/bin/env python3
"""
Arena.ai Open-Source LLM Leaderboard Enrichment with GPU VRAM Feasibility.

Scrapes the Arena.ai open-source text leaderboard, enriches each model with
parameter counts and VRAM estimates, and outputs an enriched CSV/XLSX showing
GPU deployment feasibility.

Usage:
    python enrich_arena.py                     # Scrape live from arena.ai
    python enrich_arena.py --input data.csv    # Use a pre-downloaded CSV
"""

import argparse
import logging
import os
import re
import sys

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

# Add parent directory to path so imports work when running directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from known_models import ARENA_TO_HF, KNOWN_MODELS
from scrapers.arena_scraper import scrape_arena_leaderboard
from scrapers.aa_resolver import resolve_from_artificial_analysis
from scrapers.hf_resolver import resolve_from_huggingface
from vram_calculator import (
    GPU_CONFIGS,
    GPU_DISPLAY_NAMES,
    GPUS_BY_VRAM,
    add_all_vram_and_gpu_columns,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


# ---------------------------------------------------------------------------
# Parameter resolution
# ---------------------------------------------------------------------------

def resolve_from_known(model_name):
    """
    Strategy 1: Look up model in hardcoded KNOWN_MODELS database.

    Tries exact match first, then longest substring match.
    """
    # Exact match
    if model_name in KNOWN_MODELS:
        total, active, arch = KNOWN_MODELS[model_name]
        return {
            "total_params_b": total,
            "active_params_b": active,
            "architecture": arch,
        }

    # Substring match — find the longest matching key
    best_match = None
    best_len = 0
    name_lower = model_name.lower()
    for key in KNOWN_MODELS:
        key_lower = key.lower()
        if key_lower in name_lower and len(key) > best_len:
            best_match = key
            best_len = len(key)

    if best_match:
        total, active, arch = KNOWN_MODELS[best_match]
        return {
            "total_params_b": total,
            "active_params_b": active,
            "architecture": arch,
        }

    return None


def resolve_from_name(model_name):
    """
    Strategy 2: Parse parameter count from model name using regex.

    Patterns:
        - Qwen3-30B-A3B → total=30, active=3, moe
        - Llama-3.3-70B-Instruct → total=70, active=70, dense
        - Mistral-Small-2506-24B → total=24, active=24, dense
        - Gemma 3 27B IT → total=27, active=27, dense
    """
    # Look for total size: {N}B pattern (case insensitive)
    # Must be preceded by a separator (hyphen, space, start) and followed by
    # B (not part of a longer word)
    total_match = re.search(
        r'(?:^|[\s\-_])(\d+(?:\.\d+)?)\s*B(?:[\s\-_]|$)',
        model_name,
        re.IGNORECASE,
    )

    if not total_match:
        return None

    total_b = float(total_match.group(1))

    # Look for active params: A{N}B pattern (MoE indicator)
    active_match = re.search(
        r'(?:^|[\s\-_])A(\d+(?:\.\d+)?)\s*B(?:[\s\-_]|$)',
        model_name,
        re.IGNORECASE,
    )

    if active_match:
        active_b = float(active_match.group(1))
        return {
            "total_params_b": total_b,
            "active_params_b": active_b,
            "architecture": "moe",
        }

    return {
        "total_params_b": total_b,
        "active_params_b": total_b,
        "architecture": "dense",
    }


def resolve_model_params(model_name, use_network=True):
    """
    Resolve parameter counts for a model using the priority chain.

    Priority:
        1. Hardcoded known values
        2. Parse from model name
        3. Artificial Analysis (if use_network)
        4. HuggingFace API (if use_network)
        5. UNKNOWN

    Args:
        model_name: Arena model name
        use_network: Whether to use network-based resolution

    Returns:
        tuple of (params_dict, source_name)
        params_dict has keys: total_params_b, active_params_b, architecture
        source_name is one of: "hardcoded", "name_parsing", "artificial_analysis",
                               "huggingface", "UNKNOWN"
    """
    # Strategy 1: Hardcoded known values
    result = resolve_from_known(model_name)
    if result:
        return result, "hardcoded"

    # Strategy 2: Parse from model name
    result = resolve_from_name(model_name)
    if result:
        return result, "name_parsing"

    if not use_network:
        return None, "UNKNOWN"

    # Strategy 3: Artificial Analysis
    try:
        result = resolve_from_artificial_analysis(model_name)
        if result:
            return result, "artificial_analysis"
    except Exception as e:
        logger.debug(f"AA resolution failed for {model_name}: {e}")

    # Strategy 4: HuggingFace API
    try:
        hf_id = ARENA_TO_HF.get(model_name)
        # Also try common name patterns for HF ID
        if not hf_id:
            for key, hf_repo in ARENA_TO_HF.items():
                if key.lower() in model_name.lower():
                    hf_id = hf_repo
                    break
        result = resolve_from_huggingface(model_name, hf_id=hf_id)
        if result:
            return result, "huggingface"
    except Exception as e:
        logger.debug(f"HF resolution failed for {model_name}: {e}")

    return None, "UNKNOWN"


# ---------------------------------------------------------------------------
# XLSX formatting
# ---------------------------------------------------------------------------

GREEN_FILL = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
RED_FILL = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
BOLD_FONT = Font(bold=True)
HEADER_FILL = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
HEADER_FONT = Font(bold=True, color="FFFFFF")


def write_xlsx(df, resolution_log, output_path):
    """
    Write enriched data to a formatted XLSX file with 3 sheets.

    Sheet 1: Full Table — all data with conditional formatting
    Sheet 2: GPU Summary — best model per GPU at each precision
    Sheet 3: Resolution Log — how each model was resolved
    """
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Sheet 1: Full Table
        _write_full_table(df, writer)

        # Sheet 2: GPU Summary
        _write_gpu_summary(df, writer)

        # Sheet 3: Resolution Log
        _write_resolution_log(resolution_log, writer)

    logger.info(f"XLSX saved to: {output_path}")


def _write_full_table(df, writer):
    """Write the full enriched table with conditional formatting."""
    df.to_excel(writer, sheet_name="Full Table", index=False)
    ws = writer.sheets["Full Table"]

    # Format headers
    for col_idx in range(1, len(df.columns) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", wrap_text=True)

    # Bold top 10 ranked models
    rank_col_idx = list(df.columns).index("rank") + 1 if "rank" in df.columns else None
    if rank_col_idx:
        for row_idx in range(2, min(12, len(df) + 2)):  # rows 2-11 (top 10)
            for col_idx in range(1, len(df.columns) + 1):
                ws.cell(row=row_idx, column=col_idx).font = BOLD_FONT

    # Conditional formatting on GPU fit columns (green/red)
    fit_cols = [c for c in df.columns if c.startswith("fits_")]
    for col_name in fit_cols:
        col_idx = list(df.columns).index(col_name) + 1
        for row_idx in range(2, len(df) + 2):
            cell = ws.cell(row=row_idx, column=col_idx)
            if cell.value is True:
                cell.fill = GREEN_FILL
                cell.value = "Yes"
            elif cell.value is False:
                cell.fill = RED_FILL
                cell.value = "No"

    # Auto-adjust column widths
    for col_idx in range(1, len(df.columns) + 1):
        col_letter = get_column_letter(col_idx)
        max_len = max(
            len(str(ws.cell(row=r, column=col_idx).value or ""))
            for r in range(1, min(20, len(df) + 2))
        )
        ws.column_dimensions[col_letter].width = min(max_len + 3, 30)


def _write_gpu_summary(df, writer):
    """Write GPU summary sheet: best model per GPU at each precision."""
    rows = []
    for gpu_key, gpu_vram in GPUS_BY_VRAM:
        display = GPU_DISPLAY_NAMES[gpu_key]
        for precision in ["BF16", "FP8", "INT4"]:
            p_lower = precision.lower()
            serving_col = f"vram_{p_lower}_serving_gb"

            if serving_col not in df.columns:
                continue

            # Find highest-ranked model that fits
            fits = df[df[serving_col].notna() & (df[serving_col] <= gpu_vram)]
            if not fits.empty:
                best = fits.iloc[0]  # Already sorted by rank
                note = ""
                if gpu_key == "RTX_PRO_6000" and precision == "FP8":
                    note = "FP8 requires software emulation (no native Tensor Core support)"
                rows.append({
                    "GPU": f"{display} ({gpu_vram} GB)",
                    "Precision": precision,
                    "Best Rank": int(best["rank"]) if pd.notna(best["rank"]) else "?",
                    "Model": best["model_name"],
                    "Total Params (B)": best.get("total_params_b", "?"),
                    "Active Params (B)": best.get("active_params_b", "?"),
                    "Architecture": best.get("architecture", "?"),
                    "Serving VRAM (GB)": round(best[serving_col], 1),
                    "Note": note,
                })
            else:
                rows.append({
                    "GPU": f"{display} ({gpu_vram} GB)",
                    "Precision": precision,
                    "Best Rank": "N/A",
                    "Model": "No model fits",
                    "Total Params (B)": "",
                    "Active Params (B)": "",
                    "Architecture": "",
                    "Serving VRAM (GB)": "",
                    "Note": "",
                })

    summary_df = pd.DataFrame(rows)
    summary_df.to_excel(writer, sheet_name="GPU Summary", index=False)

    # Format headers
    ws = writer.sheets["GPU Summary"]
    for col_idx in range(1, len(summary_df.columns) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", wrap_text=True)

    # Auto-adjust column widths
    for col_idx in range(1, len(summary_df.columns) + 1):
        col_letter = get_column_letter(col_idx)
        max_len = max(
            len(str(ws.cell(row=r, column=col_idx).value or ""))
            for r in range(1, min(30, len(summary_df) + 2))
        )
        ws.column_dimensions[col_letter].width = min(max_len + 3, 40)


def _write_resolution_log(resolution_log, writer):
    """Write resolution log sheet."""
    log_df = pd.DataFrame(resolution_log)
    log_df.to_excel(writer, sheet_name="Resolution Log", index=False)

    ws = writer.sheets["Resolution Log"]
    for col_idx in range(1, len(log_df.columns) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", wrap_text=True)

    for col_idx in range(1, len(log_df.columns) + 1):
        col_letter = get_column_letter(col_idx)
        max_len = max(
            len(str(ws.cell(row=r, column=col_idx).value or ""))
            for r in range(1, min(30, len(log_df) + 2))
        )
        ws.column_dimensions[col_letter].width = min(max_len + 3, 40)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_console_summary(df, resolution_counts):
    """Print a nicely formatted console summary."""
    total = len(df)
    resolved = total - resolution_counts.get("UNKNOWN", 0)
    pct = (resolved / total * 100) if total > 0 else 0

    print("\n" + "=" * 60)
    print("  Arena Leaderboard Enrichment Complete")
    print("=" * 60)
    print(f"Total models: {total}")
    print(f"Parameters resolved: {resolved} ({pct:.1f}%)")
    print(f"  - From hardcoded:           {resolution_counts.get('hardcoded', 0)}")
    print(f"  - From name parsing:        {resolution_counts.get('name_parsing', 0)}")
    print(f"  - From Artificial Analysis: {resolution_counts.get('artificial_analysis', 0)}")
    print(f"  - From HuggingFace:         {resolution_counts.get('huggingface', 0)}")
    print(f"  - Unresolved:               {resolution_counts.get('UNKNOWN', 0)}")

    # Best model per GPU at each precision
    for precision in ["BF16", "FP8"]:
        p_lower = precision.lower()
        serving_col = f"vram_{p_lower}_serving_gb"

        if serving_col not in df.columns:
            continue

        print(f"\n=== Best Model Per GPU ({precision} serving) ===")
        for gpu_key, gpu_vram in GPUS_BY_VRAM:
            display = GPU_DISPLAY_NAMES[gpu_key]
            fits = df[df[serving_col].notna() & (df[serving_col] <= gpu_vram)]
            if not fits.empty:
                best = fits.iloc[0]
                rank = int(best["rank"]) if pd.notna(best["rank"]) else "?"
                name = best["model_name"]
                total_b = best.get("total_params_b", "?")
                arch = best.get("architecture", "?")
                serving = round(best[serving_col], 1)
                arch_label = f" {arch.upper()}" if arch and arch != "dense" else ""
                print(
                    f"  {display:20s} ({gpu_vram:3d} GB): "
                    f"#{rank:<4} {name} ({total_b}B{arch_label}, ~{serving} GB serving)"
                )
            else:
                print(f"  {display:20s} ({gpu_vram:3d} GB): No model fits")

    print()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Enrich Arena.ai leaderboard with VRAM estimates"
    )
    parser.add_argument(
        "--input", "-i",
        help="Path to a pre-downloaded arena leaderboard CSV",
        default=None,
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Skip network-based resolution (AA, HF). Use only hardcoded + name parsing.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory (default: arena_enrichment/output/)",
        default=OUTPUT_DIR,
    )
    args = parser.parse_args()

    # Step 1: Get the leaderboard data
    print("Step 1: Loading Arena.ai leaderboard...")
    try:
        df = scrape_arena_leaderboard(input_csv=args.input)
    except RuntimeError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    print(f"  Loaded {len(df)} models")

    # Ensure sorted by rank
    if "rank" in df.columns and df["rank"].notna().any():
        df = df.sort_values("rank").reset_index(drop=True)

    # Step 2: Resolve parameter counts
    print("\nStep 2: Resolving model parameters...")
    resolution_log = []
    resolution_counts = {}
    use_network = not args.no_network

    for idx, row in df.iterrows():
        model_name = row["model_name"]
        if pd.isna(model_name):
            continue

        params, source = resolve_model_params(model_name, use_network=use_network)

        resolution_counts[source] = resolution_counts.get(source, 0) + 1

        if params:
            df.at[idx, "total_params_b"] = params["total_params_b"]
            df.at[idx, "active_params_b"] = params["active_params_b"]
            df.at[idx, "architecture"] = params["architecture"]
        else:
            df.at[idx, "total_params_b"] = None
            df.at[idx, "active_params_b"] = None
            df.at[idx, "architecture"] = None

        resolution_log.append({
            "model_name": model_name,
            "total_params_b": params["total_params_b"] if params else None,
            "active_params_b": params["active_params_b"] if params else None,
            "architecture": params["architecture"] if params else None,
            "resolution_source": source,
        })

        # Progress indicator
        if (idx + 1) % 20 == 0 or idx == len(df) - 1:
            print(f"  Resolved {idx + 1}/{len(df)} models...", end="\r")

    print(f"  Resolved {len(df)}/{len(df)} models      ")

    # Step 3 & 4: Compute VRAM and GPU fit columns
    print("\nStep 3: Computing VRAM estimates and GPU fit...")
    df = add_all_vram_and_gpu_columns(df)
    print("  Done")

    # Step 5: Output
    print("\nStep 4: Writing output files...")
    os.makedirs(args.output_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(args.output_dir, "arena_leaderboard_enriched.csv")
    df.to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path}")

    # XLSX
    xlsx_path = os.path.join(args.output_dir, "arena_leaderboard_enriched.xlsx")
    write_xlsx(df, resolution_log, xlsx_path)
    print(f"  XLSX: {xlsx_path}")

    # Console summary
    print_console_summary(df, resolution_counts)

    # List unresolved models
    unresolved = [r for r in resolution_log if r["resolution_source"] == "UNKNOWN"]
    if unresolved:
        print(f"Unresolved models ({len(unresolved)}):")
        for r in unresolved:
            print(f"  - {r['model_name']}")


if __name__ == "__main__":
    main()
