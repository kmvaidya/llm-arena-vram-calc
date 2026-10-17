#!/usr/bin/env python3
"""
Arena.ai Open-Source LLM Leaderboard Enrichment with GPU VRAM Feasibility.

Loads the Arena.ai open-source text leaderboard (published by Arena as a
Hugging Face dataset), enriches each model with
parameter counts and VRAM estimates, and outputs an enriched CSV/XLSX showing
GPU deployment feasibility. Optionally updates the repo README.md with tables.

Usage:
    python enrich_arena.py                          # Download the latest leaderboard
    python enrich_arena.py --input data.csv         # Use a pre-downloaded CSV
    python enrich_arena.py --update-readme          # Also update root README.md
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

# Add parent directory to path so imports work when running directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from known_models import MODEL_OVERRIDES
from scrapers.aa_resolver import (
    ModelParams,
    get_aa_models,
    resolve_from_aa,
    resolve_single_from_aa,
)
from scrapers.arena_scraper import scrape_arena_leaderboard
from vram_calculator import (
    DECODE_MBU,
    DEFAULT_OVERHEAD_PCT,
    FIT_PRECISIONS,
    FITS_ON_GPUS,
    GPUS,
    NODE_GPU_COUNT,
    NODE_GPUS,
    PRECISION_BYTES,
    add_all_vram_and_gpu_columns,
    decode_tokens_per_s,
    node_label,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

# Number of models to show directly in README (rest goes in <details>)
README_TOP_N = 50

# Arena publishes every few days; older data means the source has stalled
LEADERBOARD_STALE_DAYS = 14


# ---------------------------------------------------------------------------
# Parameter resolution
# ---------------------------------------------------------------------------

def resolve_from_overrides(model_name: str) -> ModelParams | None:
    """
    Strategy 1: Look up model in the small MODEL_OVERRIDES corrections dict.

    Only contains models where automated resolution gives wrong/no answers.
    Tries case-insensitive exact match first, then normalized substring match
    with safeguards to avoid false positives on distilled variants.
    """
    name_lower = model_name.lower()

    # Build case-insensitive lookup (lazy, rebuilt each call — dict is small)
    overrides_lower = {k.lower(): k for k in MODEL_OVERRIDES}

    # Exact match (case-insensitive)
    if name_lower in overrides_lower:
        orig_key = overrides_lower[name_lower]
        total, active, arch = MODEL_OVERRIDES[orig_key]
        return {
            "total_params_b": total,
            "active_params_b": active,
            "architecture": arch,
        }

    # Normalized substring match — normalize separators so "Command R+" matches
    # "command-r-plus" and "GLM-4.5-Air" matches "glm-4.5-air".
    def _normalize(s):
        return re.sub(r'[\s\-_]+', '-', s.lower()).replace('+', '-plus-').strip('-')

    name_norm = _normalize(model_name)

    # Skip substring matching if the name contains a size indicator (like
    # "Qwen3-8B" in "DeepSeek-R1-0528-Qwen3-8B") — that suggests a
    # different/distilled model. Let name parsing handle it instead.
    has_size_in_name = bool(re.search(
        r'(?:^|-)(\d+(?:\.\d+)?)b(?:-|$)',
        name_norm, re.IGNORECASE
    ))
    if has_size_in_name:
        return None

    best_match = None
    best_len = 0
    for key in MODEL_OVERRIDES:
        key_norm = _normalize(key)
        if key_norm in name_norm and len(key_norm) > best_len:
            best_match = key
            best_len = len(key_norm)

    if best_match:
        total, active, arch = MODEL_OVERRIDES[best_match]
        return {
            "total_params_b": total,
            "active_params_b": active,
            "architecture": arch,
        }

    return None


def resolve_from_name(model_name: str) -> ModelParams | None:
    """
    Strategy 2: Parse parameter count from model name using regex.

    Patterns handled:
        - Qwen3-30B-A3B       → total=30, active=3, moe
        - Llama-3.3-70B-Instruct → total=70, active=70, dense
        - Mistral-Small-2506-24B → total=24, active=24, dense
        - Gemma 3 27B IT       → total=27, active=27, dense
        - Mixtral-8x22B        → total=141 (8*22*0.8 rough), active=39, moe
        - Mixtral-8x7B         → total=47 (8*7*0.84), active=13, moe
    """
    # Special handling for Mixtral-style NxMB patterns
    mixtral_match = re.search(
        r'(?:^|[\s\-_])(\d+)x(\d+(?:\.\d+)?)\s*B(?:[\s\-_]|$)',
        model_name,
        re.IGNORECASE,
    )
    if mixtral_match:
        num_experts = int(mixtral_match.group(1))
        expert_size = float(mixtral_match.group(2))
        # MoE total ≈ num_experts * expert_size * overhead_factor (shared layers add ~20%)
        # Active ≈ 2 experts active (typical for Mixtral) + shared params
        total_b = round(num_experts * expert_size * 0.8, 1)  # rough estimate
        active_b = round(expert_size * 2 * 0.9, 1)  # 2 active experts typical
        return {
            "total_params_b": total_b,
            "active_params_b": active_b,
            "architecture": "moe",
        }

    # Look for active params first: A{N}B pattern (MoE indicator)
    active_match = re.search(
        r'(?:^|[\s\-_])A(\d+(?:\.\d+)?)\s*B(?:[\s\-_]|$)',
        model_name,
        re.IGNORECASE,
    )

    # Look for total size: {N}B pattern
    # Must be preceded by a separator and followed by B (not part of longer word)
    total_match = re.search(
        r'(?:^|[\s\-_])(\d+(?:\.\d+)?)\s*B(?:[\s\-_]|$)',
        model_name,
        re.IGNORECASE,
    )

    if not total_match:
        return None

    total_b = float(total_match.group(1))

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


def resolve_model_params(model_name: str, use_network: bool = True, aa_lookup: dict[str, Any] | None = None) -> tuple[ModelParams | None, str]:
    """
    Resolve parameter counts for a model using the priority chain.

    Priority:
        1. MODEL_OVERRIDES (corrections for AA mistakes, misleading names, etc.)
        2. Artificial Analysis cache (bulk data, covers most open-source models)
        3. Parse from model name (fast, no network — last resort)
        4. AA single-model scrape (if use_network, for models missing from bulk)
        5. UNKNOWN

    Args:
        model_name: Arena model name
        use_network: Whether to allow network requests (AA single-model fallback)
        aa_lookup: Pre-loaded AA lookup dict (pass for efficiency in batch calls)

    Returns:
        tuple of (params_dict, source_name)

    A result that cannot be right (active parameters above the total, or a
    non-positive count) is logged and skipped in favour of the next strategy.
    """
    strategies: list[tuple[str, Callable[[], ModelParams | None]]] = [
        # Strategy 1: Override corrections (highest priority — fixes AA mistakes)
        ("override", lambda: resolve_from_overrides(model_name)),
    ]
    if aa_lookup is not None:
        # Strategy 2: Artificial Analysis list match + model page (primary automated source)
        strategies.append(("artificial_analysis", lambda: resolve_from_aa(
            model_name, aa_lookup=aa_lookup, fetch_params=use_network,
        )))
    # Strategy 3: Parse from model name
    strategies.append(("name_parsing", lambda: resolve_from_name(model_name)))
    if use_network:
        # Strategy 4: AA single-model page from a guessed slug (expensive, last resort)
        strategies.append(("aa_single_scrape", lambda: resolve_single_from_aa(model_name)))

    for source, strategy in strategies:
        try:
            result = strategy()
        except Exception as e:
            logger.debug(f"{source} resolution failed for {model_name}: {e}")
            continue
        if not result:
            continue
        if not _is_plausible(result):
            logger.warning(f"Ignoring implausible {source} result for {model_name}: {result}")
            continue
        return result, source

    return None, "UNKNOWN"


def _is_plausible(params: ModelParams) -> bool:
    """Whether resolved parameter counts can describe a real model."""
    total = params["total_params_b"]
    active = params["active_params_b"]
    # Allow 1% for rounding in sources that report the two counts separately
    return total > 0 and 0 < active <= total * 1.01


# ---------------------------------------------------------------------------
# XLSX formatting
# ---------------------------------------------------------------------------

GREEN_FILL = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
RED_FILL = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
BOLD_FONT = Font(bold=True)
HEADER_FILL = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
HEADER_FONT = Font(bold=True, color="FFFFFF")


def write_xlsx(df: pd.DataFrame, resolution_log: list[dict[str, Any]], output_path: str) -> None:
    """Write enriched data to a formatted XLSX file with 3 sheets."""
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        _write_full_table(df, writer)
        _write_gpu_summary(df, writer)
        _write_resolution_log(resolution_log, writer)
    logger.info(f"XLSX saved to: {output_path}")


def _write_full_table(df: pd.DataFrame, writer: pd.ExcelWriter) -> None:
    """Write the full enriched table with conditional formatting."""
    df.to_excel(writer, sheet_name="Full Table", index=False)
    ws = writer.sheets["Full Table"]

    for col_idx in range(1, len(df.columns) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", wrap_text=True)

    for row_idx in range(2, min(12, len(df) + 2)):
        for col_idx in range(1, len(df.columns) + 1):
            ws.cell(row=row_idx, column=col_idx).font = BOLD_FONT

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

    for col_idx in range(1, len(df.columns) + 1):
        col_letter = get_column_letter(col_idx)
        max_len = max(
            len(str(ws.cell(row=r, column=col_idx).value or ""))
            for r in range(1, min(20, len(df) + 2))
        )
        ws.column_dimensions[col_letter].width = min(max_len + 3, 30)


def _write_gpu_summary(df: pd.DataFrame, writer: pd.ExcelWriter) -> None:
    """Write GPU summary sheet."""
    rows = _build_gpu_summary_rows(df)
    summary_df = pd.DataFrame(rows)
    summary_df.to_excel(writer, sheet_name="GPU Summary", index=False)

    ws = writer.sheets["GPU Summary"]
    for col_idx in range(1, len(summary_df.columns) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", wrap_text=True)

    for col_idx in range(1, len(summary_df.columns) + 1):
        col_letter = get_column_letter(col_idx)
        max_len = max(
            len(str(ws.cell(row=r, column=col_idx).value or ""))
            for r in range(1, min(30, len(summary_df) + 2))
        )
        ws.column_dimensions[col_letter].width = min(max_len + 3, 40)


def _write_resolution_log(resolution_log: list[dict[str, Any]], writer: pd.ExcelWriter) -> None:
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


def _hardware_configs(gpu_count: int) -> list[tuple[str, str, int, int | None]]:
    """
    (tier, label, total VRAM in GB, memory bandwidth in GB/s) for each single
    GPU (gpu_count 1), in display order, or for each 8-GPU node, smallest
    first.  Nodes carry no bandwidth: the speed formula only holds on one GPU.
    """
    if gpu_count == 1:
        return [(gpu.tier, gpu.name, gpu.vram_gb, gpu.bandwidth_gbs) for gpu in GPUS]
    return [
        (f"{NODE_GPU_COUNT}-GPU server", node_label(gpu), gpu.vram_gb * gpu_count, None)
        for gpu in NODE_GPUS
    ]


def _estimated_speed(model: pd.Series, precision: str, bandwidth_gbs: int | None) -> float | None:
    """Single-request decode speed on one GPU, or None (8-GPU node or unknown architecture)."""
    if bandwidth_gbs is None or model.get("architecture") not in DECODE_MBU:
        return None
    return decode_tokens_per_s(
        model["active_params_b"], model["architecture"], precision, bandwidth_gbs
    )


def _format_speed(tokens_per_s: float) -> str:
    """An estimate, so two significant figures: '~65 tok/s', '~280 tok/s'."""
    return f"~{float(f'{tokens_per_s:.2g}'):g} tok/s"


def _best_fit(df: pd.DataFrame, serving_col: str, vram_gb: int) -> pd.Series | None:
    """The highest-ranked model whose serving VRAM fits, or None (df is in rank order)."""
    if serving_col not in df.columns:
        return None
    fits = df[df[serving_col].notna() & (df[serving_col] <= vram_gb)]
    return None if fits.empty else fits.iloc[0]


def _build_gpu_summary_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Build GPU summary data for the XLSX: single GPUs, then 8-GPU nodes."""
    rows = []
    for gpu_count in (1, NODE_GPU_COUNT):
        for tier, display, gpu_vram, bandwidth in _hardware_configs(gpu_count):
            for precision in FIT_PRECISIONS:
                serving_col = f"vram_{precision.lower()}_serving_gb"
                if serving_col not in df.columns:
                    continue

                best = _best_fit(df, serving_col, gpu_vram)
                if best is not None:
                    speed = _estimated_speed(best, precision, bandwidth)
                    rows.append({
                        "Tier": tier,
                        "GPU": f"{display} ({gpu_vram} GB)",
                        "Precision": precision,
                        "Best Rank": int(best["rank"]) if pd.notna(best["rank"]) else "?",
                        "Model": best["model_name"],
                        "Total Params (B)": best.get("total_params_b", "?"),
                        "Active Params (B)": best.get("active_params_b", "?"),
                        "Architecture": best.get("architecture", "?"),
                        "Serving VRAM (GB)": round(best[serving_col], 1),
                        "Est. tok/s": round(speed, 1) if speed is not None else "",
                    })
                else:
                    rows.append({
                        "Tier": tier,
                        "GPU": f"{display} ({gpu_vram} GB)",
                        "Precision": precision,
                        "Best Rank": "N/A",
                        "Model": "No model fits",
                        "Total Params (B)": "",
                        "Active Params (B)": "",
                        "Architecture": "",
                        "Serving VRAM (GB)": "",
                        "Est. tok/s": "",
                    })
    return rows


# ---------------------------------------------------------------------------
# README generation
# ---------------------------------------------------------------------------

def generate_readme(
    df: pd.DataFrame,
    resolution_counts: dict[str, int],
    aa_is_stale: bool = False,
    leaderboard_date: str | None = None,
) -> str:
    """
    Generate the full README.md content with:
    1. Best Model Per GPU — one row per GPU, one column per precision
    2. Best Model Per 8-GPU Node — the same for servers of eight GPUs
    3. Enriched Leaderboard — compact table with collapsible rest
    4. How It Works / Usage
    """
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    total = len(df)
    resolved = total - resolution_counts.get("UNKNOWN", 0)
    pct = (resolved / total * 100) if total > 0 else 0

    lines = []

    # Header
    lines.append("# LLM Arena VRAM Calculator")
    lines.append("")
    lines.append("Enriches the [Arena.ai](https://arena.ai/leaderboard/text?license=open-source) "
                 "open-source LLM leaderboard with parameter counts and VRAM estimates "
                 "for single-GPU and 8-GPU server deployment feasibility.")
    lines.append("")
    lines.append("Most LLM leaderboards rank models by quality but ignore deployment constraints. "
                 "This tool answers: *\"What's the best model I can actually run on my hardware?\"* "
                 "by cross-referencing Arena rankings with VRAM requirements across precisions.")
    lines.append("")
    data_date = f"**Arena data:** {leaderboard_date} | " if leaderboard_date else ""
    lines.append(f"> **Last updated:** {now} | "
                 f"{data_date}"
                 f"**Models:** {total} | "
                 f"**Resolved:** {resolved} ({pct:.1f}%)")
    lines.append("")
    data_age = _days_since(leaderboard_date)
    if data_age is not None and data_age > LEADERBOARD_STALE_DAYS:
        lines.append(f"> **Warning:** The Arena leaderboard data is {data_age} days old; "
                     "Arena may have stopped publishing updates.")
        lines.append("")
    if aa_is_stale:
        lines.append("> **Warning:** The Artificial Analysis model list could not be fetched, "
                     "so some parameter counts may be stale or missing.")
        lines.append("")

    # Table 1: Best Model Per GPU — one row per GPU, one column per precision
    lines.append("## Best Model Per GPU")
    lines.append("")
    lines.append("Highest-ranked Arena model that fits on one GPU at each precision, "
                 "with its Arena rank, serving VRAM and estimated decode speed for one "
                 "request (see [Memory formula](#memory-formula) and "
                 "[Speed formula](#speed-formula)).")
    lines.append("")
    lines.extend(_generate_best_model_table(df))
    lines.append("")

    # Table 2: Best Model Per 8-GPU Node — where the top of the leaderboard lives
    lines.append(f"## Best Model Per {NODE_GPU_COUNT}-GPU Node")
    lines.append("")
    lines.append(f"The same for a server with {NODE_GPU_COUNT} of one GPU, the usual way to "
                 "serve models too large for one GPU (weights split across the GPUs). "
                 f"Only GPUs sold in {NODE_GPU_COUNT}-GPU servers are listed; see [GPUs](#gpus).")
    lines.append("")
    lines.extend(_generate_best_model_table(df, gpu_count=NODE_GPU_COUNT))
    lines.append("")

    # Table 3: Enriched Leaderboard
    lines.append("## Full Leaderboard")
    lines.append("")
    lines.append("*Fits on* is the smallest widely deployed GPU (or "
                 f"{NODE_GPU_COUNT}-GPU server) that holds the model at FP8; *Speed* is the "
                 "estimated decode speed for one request on that GPU "
                 "(see [Speed formula](#speed-formula)).")
    lines.append("")
    lines.extend(_generate_leaderboard_table(df))
    lines.append("")

    # Architecture section
    lines.append("## Architecture")
    lines.append("")
    lines.append("**Data flow:** Arena.ai leaderboard → parameter resolution "
                 "(4-strategy fallback) → VRAM calculation → GPU feasibility matrix")
    lines.append("")
    lines.append("The parameter resolution chain prioritizes accuracy: manual overrides "
                 "catch known errors, [Artificial Analysis](https://artificialanalysis.ai) "
                 "supplies sizes from its model pages (its model list maps Arena names to "
                 "pages, and page results are cached in the repository so each run only "
                 "reads new models), name parsing extracts `{N}B` patterns as a fallback, "
                 "and guessed page names handle the long tail. Results where active "
                 "parameters exceed the total are rejected.")
    lines.append("")
    overhead = 1 + DEFAULT_OVERHEAD_PCT
    lines.append("### Memory formula")
    lines.append("")
    lines.append("```")
    lines.append(f"Serving VRAM (GB) = total parameters (billions) × bytes per parameter × {overhead:g}")
    lines.append("```")
    lines.append("")
    lines.append("| Precision | Bytes per parameter | 70B model: weights | 70B model: serving |")
    lines.append("|-----------|---------------------|--------------------|--------------------|")
    for precision in FIT_PRECISIONS:
        weights = 70 * PRECISION_BYTES[precision]
        lines.append(f"| {precision} | {PRECISION_BYTES[precision]:g} | {weights:g} GB "
                     f"| {round(weights * overhead, 1):g} GB |")
    lines.append("")
    lines.append(f"- The ×{overhead:g} adds {DEFAULT_OVERHEAD_PCT:.0%} for the KV cache (the attention "
                 "keys and values kept for every token in context), activations and the serving "
                 "framework. The KV cache grows with context length and with the number of "
                 "requests served at once, so long contexts or many users need more.")
    lines.append("- **MoE models** count all their parameters: every expert must sit in memory, "
                 "even though each token uses only a few.")
    lines.append(f"- A model fits a GPU when its serving VRAM is at most the GPU's memory, and an "
                 f"{NODE_GPU_COUNT}-GPU server when it is at most {NODE_GPU_COUNT} × that memory.")
    lines.append("")

    h100 = next(g for g in GPUS if g.key == "h100")
    example = decode_tokens_per_s(30, "dense", "FP8", h100.bandwidth_gbs)
    lines.append("### Speed formula")
    lines.append("")
    lines.append("Generating each token reads every active weight from memory once, so with one "
                 "request at a time the GPU's memory bandwidth, not its compute, sets the speed:")
    lines.append("")
    lines.append("```")
    lines.append("Decode speed (tokens/s) ≈ MBU × memory bandwidth (GB/s) "
                 "÷ (active parameters (billions) × bytes per parameter)")
    lines.append("```")
    lines.append("")
    lines.append("- **Active parameters** are all the parameters of a dense model, and only those "
                 "used per token for an MoE model (for example 37B of DeepSeek V3's 671B).")
    lines.append(f"- **MBU** (model bandwidth utilization) is the share of peak bandwidth real "
                 f"software reaches: {DECODE_MBU['dense']:g} for dense models (Databricks measured "
                 f"55–60% with one request; RTX 4090 llama.cpp results work out to 57–70%) and "
                 f"{DECODE_MBU['moe']:g} for MoE models, which read their experts in many small "
                 f"pieces (Qwen3-30B-A3B on an RTX 4090 works out to 25–40%).")
    lines.append(f"- Example: a 30B dense model at FP8 on an H100 SXM: {DECODE_MBU['dense']:g} × "
                 f"{h100.bandwidth_gbs:,} ÷ (30 × 1) ≈ {example:.0f} tokens/s.")
    lines.append("- This assumes one request and a short context. Reading a long context's KV "
                 "cache lowers the speed; serving many requests at once raises a GPU's total "
                 "throughput, not the speed each request sees.")
    lines.append(f"- **No speed is shown for {NODE_GPU_COUNT}-GPU servers.** With one request, the "
                 "GPUs wait on each other at every layer and use only a small part of their "
                 "combined bandwidth: DeepSeek-R1 on 8× H200 has been measured at about 47 "
                 "tokens/s, where the formula's ceiling is about 1,000.")
    lines.append("")
    lines.append("### GPUs")
    lines.append("")
    lines.append(f"| Tier | GPU | VRAM | Bandwidth | Architecture | Native FP8 "
                 f"| In {NODE_GPU_COUNT}-GPU servers |")
    lines.append("|------|-----|------|-----------|--------------|------------|------------------|")
    for gpu in GPUS:
        if not gpu.in_8gpu_servers:
            servers = "No"
        elif gpu.server_variant:
            servers = f"{gpu.server_variant} only"
        else:
            servers = "Yes"
        fp8 = "Yes" if gpu.native_fp8 else "No (weight-only FP8 kernels)"
        lines.append(f"| {gpu.tier} | {gpu.name} | {gpu.vram_gb} GB "
                     f"| {gpu.bandwidth_gbs / 1000:.3g} TB/s | {gpu.architecture} "
                     f"| {fp8} | {servers} |")
    lines.append("")
    ladder = ", ".join(g.name for g in FITS_ON_GPUS)
    lines.append(f"The leaderboard's *Fits on* names the smallest of the most widely deployed "
                 f"GPUs that fits ({ladder}), or else an {NODE_GPU_COUNT}-GPU server of the "
                 "smallest of them that fits; the tables above cover every GPU.")
    lines.append("")
    lines.append(f"GeForce cards are not in the {NODE_GPU_COUNT}-GPU tables because NVIDIA's "
                 "GeForce driver license bars data-center deployment; workstation cards are "
                 "built for desktops with up to four cards. Rack-scale systems "
                 "(GB200/GB300 NVL72, AMD Helios with MI455X) come as 72-GPU racks rather "
                 f"than single GPUs or {NODE_GPU_COUNT}-GPU servers, so they are not listed. "
                 "B200 shows the 180 GB per GPU usable on an HGX B200 board.")
    lines.append("")

    # Usage section
    lines.append("## Usage")
    lines.append("")
    lines.append("```bash")
    lines.append("# Install dependencies")
    lines.append("uv sync")
    lines.append("")
    lines.append("# Full pipeline (downloads the latest leaderboard, updates README)")
    lines.append("uv run python arena_enrichment/enrich_arena.py --update-readme")
    lines.append("")
    lines.append("# Use a pre-downloaded CSV")
    lines.append("uv run python arena_enrichment/enrich_arena.py --input data.csv --update-readme")
    lines.append("")
    lines.append("# Skip network resolution (overrides + name parsing only)")
    lines.append("uv run python arena_enrichment/enrich_arena.py --no-network --update-readme")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def _days_since(day: str | None) -> int | None:
    """Days from a YYYY-MM-DD date to today (UTC), or None if there is no valid date."""
    try:
        return (datetime.now(UTC).date() - date.fromisoformat(day or "")).days
    except ValueError:
        return None


def _generate_best_model_table(df: pd.DataFrame, gpu_count: int = 1) -> list[str]:
    """
    One row per GPU (or 8-GPU node), one column per precision, each cell the
    best model that fits: "name (#rank, serving GB, ~speed)", with no speed
    for nodes.
    """
    single = gpu_count == 1
    lines = []
    if single:
        lines.append("| Tier | GPU | VRAM | " + " | ".join(FIT_PRECISIONS) + " |")
        lines.append("|------|-----|------|" + "|".join("------" for _ in FIT_PRECISIONS) + "|")
    else:
        lines.append("| Node | VRAM | " + " | ".join(FIT_PRECISIONS) + " |")
        lines.append("|------|------|" + "|".join("------" for _ in FIT_PRECISIONS) + "|")

    previous_tier = None
    for tier, display, gpu_vram, bandwidth in _hardware_configs(gpu_count):
        cells = []
        for precision in FIT_PRECISIONS:
            serving_col = f"vram_{precision.lower()}_serving_gb"
            best = _best_fit(df, serving_col, gpu_vram)
            if best is None:
                cells.append("—")
                continue
            rank = int(best["rank"]) if pd.notna(best["rank"]) else "?"
            details = [f"#{rank}", f"{round(best[serving_col], 1):g} GB"]
            speed = _estimated_speed(best, precision, bandwidth)
            if speed is not None:
                details.append(_format_speed(speed))
            cells.append(f"{best['model_name']} ({', '.join(details)})")
        if single:
            # Name each tier once, on its first row
            tier_cell = tier if tier != previous_tier else ""
            previous_tier = tier
            lines.append(f"| {tier_cell} | {display} | {gpu_vram} GB | " + " | ".join(cells) + " |")
        else:
            lines.append(f"| {display} | {gpu_vram} GB | " + " | ".join(cells) + " |")

    return lines


def _generate_leaderboard_table(df: pd.DataFrame) -> list[str]:
    """Generate the enriched leaderboard markdown table."""
    lines = []

    header = ("| Rank | Model | Score | Params (B) | Arch | VRAM BF16 | VRAM FP8 | VRAM INT4 "
              "| Fits on | Speed |")
    sep = "|------|-------|-------|------------|------|-----------|----------|-----------|---------|-------|"

    def _format_row(row):
        rank = int(row["rank"]) if pd.notna(row.get("rank")) else "?"
        name = row.get("model_name", "?")
        score = int(row["arena_score"]) if pd.notna(row.get("arena_score")) else "?"
        total = row.get("total_params_b")
        active = row.get("active_params_b")
        # Show "total (active)" for MoE, just "total" for dense
        if pd.notna(total) and pd.notna(active) and total != active:
            params_s = f"{total:g} ({active:g})"
        elif pd.notna(total):
            params_s = f"{total:g}"
        else:
            params_s = "?"
        arch = (row.get("architecture") or "?").upper() if pd.notna(row.get("architecture")) else "?"
        if arch == "DENSE":
            arch = "Dense"
        elif arch == "MOE":
            arch = "MoE"
        bf16 = row.get("vram_bf16_serving_gb")
        bf16_s = f"{bf16:g}" if pd.notna(bf16) else "?"
        fp8 = row.get("vram_fp8_serving_gb")
        fp8_s = f"{fp8:g}" if pd.notna(fp8) else "?"
        int4 = row.get("vram_int4_serving_gb")
        int4_s = f"{int4:g}" if pd.notna(int4) else "?"
        # "Fits on" — show the smallest reference GPU at FP8, the most
        # practical precision, or else the smallest reference 8-GPU node
        best_fp8 = row.get("best_gpu_fp8", "?")
        best_node_fp8 = row.get("best_node_fp8", "?")
        if best_fp8 == "UNKNOWN":
            fits_s = "?"
        elif best_fp8 != "MULTI-GPU":
            fits_s = f"{best_fp8} (FP8)"
        elif best_node_fp8 == "MULTI-NODE":
            fits_s = "Multi-node"
        else:
            fits_s = f"{best_node_fp8} (FP8)"
        # Speed — one request on the "Fits on" GPU; none for 8-GPU servers
        tok_s = row.get("est_tok_s_fp8")
        if pd.notna(tok_s):
            speed_s = _format_speed(tok_s)
        elif best_fp8 == "MULTI-GPU":
            speed_s = "—"
        else:
            speed_s = "?"
        return (f"| {rank} | {name} | {score} | {params_s} | {arch} | {bf16_s} | {fp8_s} "
                f"| {int4_s} | {fits_s} | {speed_s} |")

    # Top N rows shown directly
    top_df = df.head(README_TOP_N)
    rest_df = df.iloc[README_TOP_N:]

    lines.append(header)
    lines.append(sep)
    for _, row in top_df.iterrows():
        lines.append(_format_row(row))

    # Remaining rows in collapsible section
    if len(rest_df) > 0:
        lines.append("")
        lines.append(f"<details><summary>Show remaining {len(rest_df)} models</summary>")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for _, row in rest_df.iterrows():
            lines.append(_format_row(row))
        lines.append("")
        lines.append("</details>")

    return lines


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_console_summary(df: pd.DataFrame, resolution_counts: dict[str, int]) -> None:
    """Print a nicely formatted console summary."""
    total = len(df)
    resolved = total - resolution_counts.get("UNKNOWN", 0)
    pct = (resolved / total * 100) if total > 0 else 0

    print("\n" + "=" * 60)
    print("  Arena Leaderboard Enrichment Complete")
    print("=" * 60)
    print(f"Total models: {total}")
    print(f"Parameters resolved: {resolved} ({pct:.1f}%)")
    print(f"  - From AA cache:            {resolution_counts.get('artificial_analysis', 0)}")
    print(f"  - From overrides:           {resolution_counts.get('override', 0)}")
    print(f"  - From name parsing:        {resolution_counts.get('name_parsing', 0)}")
    print(f"  - From AA single scrape:    {resolution_counts.get('aa_single_scrape', 0)}")
    print(f"  - Unresolved:               {resolution_counts.get('UNKNOWN', 0)}")

    for precision in ["BF16", "FP8"]:
        p_lower = precision.lower()
        serving_col = f"vram_{p_lower}_serving_gb"
        if serving_col not in df.columns:
            continue

        print(f"\n=== Best Model Per GPU ({precision} serving) ===")
        for gpu in GPUS:
            display, gpu_vram = gpu.name, gpu.vram_gb
            best = _best_fit(df, serving_col, gpu_vram)
            if best is not None:
                speed = _estimated_speed(best, precision, gpu.bandwidth_gbs)
                speed_s = f", {_format_speed(speed)}" if speed is not None else ""
                rank = int(best["rank"]) if pd.notna(best["rank"]) else "?"
                name = best["model_name"]
                total_b = best.get("total_params_b", "?")
                arch = best.get("architecture", "?")
                serving = round(best[serving_col], 1)
                arch_label = f" {arch.upper()}" if arch and arch != "dense" else ""
                print(
                    f"  {display:24s} ({gpu_vram:3d} GB): "
                    f"#{rank:<4} {name} ({total_b}B{arch_label}, ~{serving} GB serving{speed_s})"
                )
            else:
                print(f"  {display:24s} ({gpu_vram:3d} GB): No model fits")

    print()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
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
        help="Skip network-based resolution (AA). Use only overrides + name parsing.",
    )
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="Update the root README.md with leaderboard tables.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Force re-fetch of the Artificial Analysis model cache.",
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

    leaderboard_date = df.attrs.get("leaderboard_date")
    published = f" (Arena published {leaderboard_date})" if leaderboard_date else ""
    print(f"  Loaded {len(df)} models{published}")

    # Ensure sorted by rank
    if "rank" in df.columns and df["rank"].notna().any():
        df = df.sort_values("rank").reset_index(drop=True)

    # Step 2: Resolve parameter counts
    print("\nStep 2: Resolving model parameters...")
    use_network = not args.no_network

    # Pre-load AA model cache (one network request, or from disk)
    aa_lookup = None
    aa_is_stale = False
    if use_network:
        print("  Loading Artificial Analysis model database...")
        aa_lookup, aa_is_stale = get_aa_models(force_refresh=args.refresh_cache)
        print(f"  AA database: {len(aa_lookup)} entries loaded")
        if aa_is_stale:
            fallback = "using the last cached list" if aa_lookup else "no cached list either"
            print(f"  WARNING: AA model list fetch failed, {fallback}")

    resolution_log: list[dict[str, Any]] = []
    resolution_counts: dict[str, int] = {}

    for idx, row in df.iterrows():
        model_name = row["model_name"]
        if pd.isna(model_name):
            continue

        params, source = resolve_model_params(
            model_name, use_network=use_network, aa_lookup=aa_lookup
        )
        resolution_counts[source] = resolution_counts.get(source, 0) + 1

        if params:
            df.at[idx, "total_params_b"] = params["total_params_b"]
            df.at[idx, "active_params_b"] = params["active_params_b"]
            df.at[idx, "architecture"] = params["architecture"]
        else:
            df.at[idx, "total_params_b"] = None
            df.at[idx, "active_params_b"] = None
            df.at[idx, "architecture"] = None
        df.at[idx, "resolution_source"] = source

        resolution_log.append({
            "model_name": model_name,
            "total_params_b": params["total_params_b"] if params else None,
            "active_params_b": params["active_params_b"] if params else None,
            "architecture": params["architecture"] if params else None,
            "resolution_source": source,
        })

        if (idx + 1) % 20 == 0 or idx == len(df) - 1:
            print(f"  Resolved {idx + 1}/{len(df)} models...", end="\r")

    print(f"  Resolved {len(df)}/{len(df)} models      ")

    # Step 3: Compute VRAM and GPU fit columns
    print("\nStep 3: Computing VRAM estimates and GPU fit...")
    df = add_all_vram_and_gpu_columns(df)
    print("  Done")

    # Step 4: Output
    print("\nStep 4: Writing output files...")
    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, "arena_leaderboard_enriched.csv")
    df.to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path}")

    xlsx_path = os.path.join(args.output_dir, "arena_leaderboard_enriched.xlsx")
    write_xlsx(df, resolution_log, xlsx_path)
    print(f"  XLSX: {xlsx_path}")

    # Step 5: Update README if requested
    if args.update_readme:
        print("\nStep 5: Updating README.md...")
        readme_content = generate_readme(
            df, resolution_counts, aa_is_stale=aa_is_stale, leaderboard_date=leaderboard_date
        )
        readme_path = os.path.join(REPO_ROOT, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        print(f"  README: {readme_path}")

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
