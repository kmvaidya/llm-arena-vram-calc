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
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

# Add parent directory to path so imports work when running directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from known_models import MODEL_OVERRIDES, NOT_RELEASED
from measurements import MEASUREMENTS
from scrapers import aa_resolver
from scrapers.aa_resolver import (
    RELEASE_TAGS,
    AAUnavailable,
    ModelParams,
    get_aa_models,
    resolve_from_aa,
    resolve_single_from_aa,
)
from scrapers.arena_scraper import scrape_arena_leaderboard
from vram_calculator import (
    BANDWIDTH_EFFICIENCY,
    BYTES_PER_GIB,
    CONTEXT_TOKENS,
    FIT_HEADROOM,
    FIT_PRECISIONS,
    FITS_ON_GPUS,
    FIXED_MS_PER_TOKEN,
    HARDWARE,
    KV_BYTES_PER_TOKEN,
    KV_BYTES_PER_TOKEN_SMALL,
    MOE_BYTES_FACTOR,
    NODE_GPU_COUNT,
    NODE_GPUS,
    PRECISION_BYTES,
    RUNTIME_GIB,
    SMALL_MODEL_PARAMS_B,
    SYNC_MS_PER_TOKEN,
    Hardware,
    add_all_vram_and_gpu_columns,
    decode_tokens_per_s,
    fits,
    hardware_speed,
    kv_cache_gib,
    needed_gib,
    node_label,
    serving_gib,
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

# Arena publishes every few days, and has gone up to 19 days between
# publications (2024-08 to 2026-10); older data means the source has stalled
LEADERBOARD_STALE_DAYS = 21
# Health checks: the daily run still publishes when one fails, and the
# workflow opens an issue listing the problems
MIN_RESOLVED_SHARE = 0.85      # models with parameter counts, ~93% in 2026-10
AA_LIST_MAX_AGE_DAYS = 3       # an outage of a day or two is not worth an issue


# ---------------------------------------------------------------------------
# Parameter resolution
# ---------------------------------------------------------------------------

_RELEASE_PART = re.compile(RELEASE_TAGS)


def _normalize_override(name: str) -> str:
    """Lowercase with "-" between parts, so "Command R+" reads "command-r-plus"."""
    return re.sub(r"[\s\-_]+", "-", name.lower().replace("+", "-plus-")).strip("-")


_OVERRIDE_KEYS = {_normalize_override(key): key for key in MODEL_OVERRIDES}


def resolve_from_overrides(model_name: str) -> ModelParams | None:
    """
    Strategy 1: Look up model in the small MODEL_OVERRIDES corrections dict.

    A key matches the name exactly, or as its start followed only by
    release parts: "kimi-k2" covers "kimi-k2-0905-preview" but not
    "kimi-k2.6", "step-3" not "step-3.5-flash", "mimo-v2.5" not
    "mimo-v2.5-pro".  The longest matching key wins, so new names of other
    models fall through to the automated sources instead of taking a
    neighbour's size.
    """
    name = _normalize_override(model_name)
    best = None
    for key_norm in _OVERRIDE_KEYS:
        if not name.startswith(key_norm):
            continue
        rest = name[len(key_norm):]
        if rest and not (rest.startswith("-")
                         and all(_RELEASE_PART.fullmatch(part) for part in rest[1:].split("-"))):
            continue
        if best is None or len(key_norm) > len(best):
            best = key_norm

    if best is None:
        return None
    total, active, arch = MODEL_OVERRIDES[_OVERRIDE_KEYS[best]]
    return {
        "total_params_b": total,
        "active_params_b": active,
        "architecture": arch,
    }


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
        except AAUnavailable as e:
            logger.debug(f"{source} resolution failed for {model_name}: {e}")
            continue
        except Exception:
            # A bug or an unexpected page, not an outage: say so in the log
            logger.warning(f"{source} resolution crashed for {model_name}", exc_info=True)
            continue
        if not result:
            continue
        if not _is_plausible(result):
            logger.warning(f"Ignoring implausible {source} result for {model_name}: {result}")
            continue
        return result, source

    return None, "UNKNOWN"


def drop_unreleased(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Leave out models whose weights were never released, and rank the rest 1, 2, ..."""
    unreleased = df["model_name"].str.lower().isin(NOT_RELEASED)
    if not unreleased.any():
        return df, []
    left_out = list(df.loc[unreleased, "model_name"])
    df = df[~unreleased].reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df, left_out


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


@dataclass(frozen=True)
class HardwareRow:
    """One row of the hardware table: a single device, or an 8-GPU server of one GPU."""

    hardware: Hardware
    gpu_count: int

    @property
    def tier(self) -> str:
        return self.hardware.tier if self.gpu_count == 1 else f"{NODE_GPU_COUNT}-GPU server"

    @property
    def label(self) -> str:
        return self.hardware.name if self.gpu_count == 1 else node_label(self.hardware)

    @property
    def memory_label(self) -> str:
        """The memory a model can use, in GiB ("96 of 128 GiB")."""
        hw = self.hardware
        if self.gpu_count > 1:
            return f"{self.gpu_count * hw.usable_gib:,.0f} GiB"
        if hw.usable_memory_gb is not None:
            return f"{hw.usable_memory_gb:g} of {hw.memory_gb:g} GiB"
        return f"{hw.memory_gb:g} GiB"

    @property
    def bandwidth_label(self) -> str:
        tb_s = f"{self.hardware.bandwidth_for(self.gpu_count) / 1000:.3g} TB/s"
        return tb_s if self.gpu_count == 1 else f"{self.gpu_count} × {tb_s}"

    def needed_gib(self, single_serving_gib: float) -> float:
        """Memory a model needs here, from its single-GPU serving memory."""
        return needed_gib(single_serving_gib, self.gpu_count)

    def fits(self, single_serving_gib: float) -> bool:
        return fits(self.hardware, single_serving_gib, self.gpu_count)


def _hardware_rows() -> list[HardwareRow]:
    """Every single device in display order, then every 8-GPU server, smallest first."""
    return ([HardwareRow(hw, 1) for hw in HARDWARE]
            + [HardwareRow(gpu, NODE_GPU_COUNT) for gpu in NODE_GPUS])


def _format_gib(gib: float | None) -> str:
    """Whole GiB from 100 up ("1,568"), one decimal below ("22.3")."""
    if gib is None or pd.isna(gib):
        return "?"
    return f"{gib:,.0f}" if gib >= 100 else f"{gib:.1f}"


def _format_speed(tokens_per_s: float) -> str:
    """An estimate, so two significant figures: '~65 tok/s', '~280 tok/s'."""
    return f"~{float(f'{tokens_per_s:.2g}'):g} tok/s"


def _best_fit(df: pd.DataFrame, serving_col: str, row: HardwareRow) -> pd.Series | None:
    """The highest-ranked model that fits the hardware, or None (df is in rank order)."""
    if serving_col not in df.columns:
        return None
    serving = df[serving_col]
    fits = df[serving.notna() & serving.apply(lambda x: pd.notna(x) and row.fits(x))]
    return None if fits.empty else fits.iloc[0]


def _build_gpu_summary_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Build the hardware summary for the XLSX: single devices, then 8-GPU servers."""
    rows = []
    for hw_row in _hardware_rows():
        for precision in FIT_PRECISIONS:
            serving_col = f"vram_{precision.lower()}_serving_gib"
            if serving_col not in df.columns:
                continue

            base = {
                "Tier": hw_row.tier,
                "Hardware": hw_row.label,
                "Usable memory": hw_row.memory_label,
                "Bandwidth": hw_row.bandwidth_label,
                "Precision": precision,
            }
            best = _best_fit(df, serving_col, hw_row)
            if best is not None:
                speed = hardware_speed(hw_row.hardware, best, precision, hw_row.gpu_count)
                rows.append({
                    **base,
                    "Best Rank": int(best["rank"]) if pd.notna(best["rank"]) else "?",
                    "Model": best["model_name"],
                    "Total Params (B)": best.get("total_params_b", "?"),
                    "Active Params (B)": best.get("active_params_b", "?"),
                    "Architecture": best.get("architecture", "?"),
                    "Memory needed (GiB)": round(hw_row.needed_gib(best[serving_col]), 1),
                    "Est. tok/s": round(speed, 1) if speed is not None else "",
                })
            else:
                rows.append({
                    **base,
                    "Best Rank": "N/A",
                    "Model": "No model fits",
                    "Total Params (B)": "",
                    "Active Params (B)": "",
                    "Architecture": "",
                    "Memory needed (GiB)": "",
                    "Est. tok/s": "",
                })
    return rows


def _speed_accuracy() -> tuple[list[tuple[Any, float]], float, float]:
    """Each measurement with its estimate, the median error, and the share within ±20%."""
    pairs = [
        (m, decode_tokens_per_s(m.active_params_b, m.architecture, m.precision,
                                m.bandwidth_gbs, m.platform, m.gpu_count))
        for m in MEASUREMENTS
    ]
    errors = [abs(estimate / m.tokens_per_s - 1) for m, estimate in pairs]
    median = statistics.median(errors)
    within_20 = sum(e <= 0.2 for e in errors) / len(errors)
    return pairs, median, within_20


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
    1. Best Model Per Hardware — one row per device or 8-GPU server,
       one column per precision
    2. Enriched Leaderboard — compact table with collapsible rest
    3. How It Works (memory and speed formulas, accuracy) / Usage
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
                 "and speed estimates for consumer GPUs, Macs and other unified-memory "
                 "computers, workstations, data-center GPUs and 8-GPU servers.")
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

    # Table 1: Best Model Per Hardware — one row per device or 8-GPU server
    lines.append("## Best Model Per Hardware")
    lines.append("")
    lines.append("Highest-ranked Arena model that fits each piece of hardware at each precision, "
                 "with its Arena rank, the memory it needs and its estimated generation speed "
                 "for one request (see [Memory formula](#memory-formula) and "
                 "[Speed formula](#speed-formula)). *Memory* is what a model can use: the "
                 "GPU's memory, or for computers whose GPU shares system RAM, the share the "
                 f"system gives it. A model fits a GPU when it needs at most {FIT_HEADROOM:.0%} "
                 "of its memory.")
    lines.append("")
    lines.extend(_generate_hardware_table(df))
    lines.append("")
    lines.extend(_hardware_notes())
    lines.append("")

    # Table 2: Enriched Leaderboard
    ladder = ", ".join(g.name for g in FITS_ON_GPUS)
    lines.append("## Full Leaderboard")
    lines.append("")
    lines.append("*GiB* columns are the memory each model needs (see [Memory formula]"
                 f"(#memory-formula)). *Fits on* is the smallest of the most widely deployed GPUs ({ladder}) "
                 f"that holds the model at FP8, or else an {NODE_GPU_COUNT}-GPU server of the "
                 "smallest of them; *Speed* is the estimated generation speed for one request "
                 "there (see [Speed formula](#speed-formula)).")
    lines.append("")
    lines.extend(_generate_leaderboard_table(df))
    lines.append("")

    # Architecture section
    lines.append("## Architecture")
    lines.append("")
    lines.append("**Data flow:** Arena.ai leaderboard → parameter resolution "
                 "(4-strategy fallback) → memory and speed estimates → hardware table")
    lines.append("")
    lines.append("The parameter resolution chain prioritizes accuracy: manual overrides "
                 "catch known errors, [Artificial Analysis](https://artificialanalysis.ai) "
                 "supplies sizes from its model pages (its model list maps Arena names to "
                 "pages, and page results are cached in the repository so each run only "
                 "reads new models), name parsing extracts `{N}B` patterns as a fallback, "
                 "and guessed page names handle the long tail. Results where active "
                 "parameters exceed the total are rejected.")
    lines.append("")
    lines.extend(_memory_formula_section())
    lines.append("")
    lines.extend(_speed_formula_section())
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


def _generate_hardware_table(df: pd.DataFrame) -> list[str]:
    """
    One row per device or 8-GPU server, one column per precision, each cell
    the best model that fits: "name (#rank, GiB needed, ~speed)".
    """
    lines = ["| Tier | Hardware | Memory | Bandwidth | " + " | ".join(FIT_PRECISIONS) + " |",
             "|------|----------|--------|-----------|" + "|".join("------" for _ in FIT_PRECISIONS) + "|"]

    previous_tier = None
    for hw_row in _hardware_rows():
        cells = []
        for precision in FIT_PRECISIONS:
            serving_col = f"vram_{precision.lower()}_serving_gib"
            best = _best_fit(df, serving_col, hw_row)
            if best is None:
                cells.append("—")
                continue
            rank = int(best["rank"]) if pd.notna(best["rank"]) else "?"
            details = [f"#{rank}", f"{round(hw_row.needed_gib(best[serving_col])):,} GiB"]
            speed = hardware_speed(hw_row.hardware, best, precision, hw_row.gpu_count)
            if speed is not None:
                details.append(_format_speed(speed))
            cells.append(f"{best['model_name']} ({', '.join(details)})")
        # Name each tier once, on its first row
        tier_cell = hw_row.tier if hw_row.tier != previous_tier else ""
        previous_tier = hw_row.tier
        lines.append(f"| {tier_cell} | {hw_row.label} | {hw_row.memory_label} "
                     f"| {hw_row.bandwidth_label} | " + " | ".join(cells) + " |")

    return lines


def _hardware_notes() -> list[str]:
    """What the hardware table includes and leaves out, and why."""
    no_fp8 = ", ".join(hw.name.split(",")[0] for hw in HARDWARE if not hw.native_fp8
                       and not hw.name.startswith("Mac"))
    pcie = ", ".join(hw.server_name or hw.name for hw in NODE_GPUS if not hw.gpu_fabric)
    return [
        f"- **8-bit without FP8 hardware** ({no_fp8}, Macs): the FP8 column means 8-bit "
        "weights such as GGUF Q8_0 or MLX 8-bit, which need the same memory.",
        "- **Unified memory**: the GPU shares system RAM, which is several times slower than "
        "GPU memory; the speeds show it. macOS gives the GPU 75% of RAM by default, and "
        "`sudo sysctl iogpu.wired_limit_mb=<MB>` raises that until the next restart (leave "
        "8–16 GiB for macOS). The Ryzen AI Max+ 395 gives its GPU up to 96 GiB on Windows, and "
        "more on Linux, where the kernel's limit can be raised (AMD's guides use about 120 GiB); "
        "the DGX Spark about 115 GiB after its operating system.",
        f"- **{NODE_GPU_COUNT}-GPU servers** list only GPUs sold that way: GeForce cards are "
        "left out because NVIDIA's GeForce driver license bars data-center deployment, as are "
        "workstation cards, built for desktops of up to four; the RTX PRO 6000 appears as its "
        "Server Edition. Speeds are estimated for boards that link every GPU to every other "
        f"(NVLink, Infinity Fabric, Gaudi's on-board network); servers that connect GPUs over "
        f"PCIe ({pcie}) show none, for lack of measurements. For MoE models they assume a "
        "latency-tuned engine: the measurement behind them is TensorRT-LLM's fastest DeepSeek-R1 "
        "setup, and default setups have run at about half that speed.",
        "- **Not listed**: rack-scale systems (GB200/GB300 NVL72, AMD Helios with MI455X), which "
        "come as 72-GPU racks, and AMD's MI440X 8-GPU platform, whose memory is not yet "
        "published. B200, B300 and Rubin show the memory an HGX board gives each GPU; Rubin's "
        "figures (2 TB and 130 TB/s per board) are from NVIDIA's US page of 2026-10-15, while "
        "its other pages list 2.3 TB and 176 TB/s. The DGX Station counts its 252 GiB of GPU "
        "memory; its 496 GiB of CPU memory is about 18× slower.",
    ]


def _memory_formula_section() -> list[str]:
    kv_large = CONTEXT_TOKENS * KV_BYTES_PER_TOKEN / BYTES_PER_GIB
    kv_small = CONTEXT_TOKENS * KV_BYTES_PER_TOKEN_SMALL / BYTES_PER_GIB
    h100 = next(hw for hw in HARDWARE if hw.key == "h100")
    fp8_gib = PRECISION_BYTES["FP8"] * 1e9 / BYTES_PER_GIB
    example = serving_gib(70, "FP8")

    def gib_per_billion(precision: str) -> str:
        return f"{PRECISION_BYTES[precision] * 1e9 / BYTES_PER_GIB:.3g}"

    return [
        "### Memory formula",
        "",
        "```",
        "Memory (GiB) = total parameters (billions) × GiB per billion parameters",
        f"             + KV cache: {kv_large:g} GiB ({CONTEXT_TOKENS:,} tokens × "
        f"{KV_BYTES_PER_TOKEN // 1024} KiB; {kv_small:g} GiB under {SMALL_MODEL_PARAMS_B}B parameters)",
        f"             + {RUNTIME_GIB:g} GiB per GPU",
        "```",
        "",
        "| Precision | Bytes per parameter | GiB per billion parameters | Files it covers |",
        "|-----------|---------------------|----------------------------|-----------------|",
        f"| BF16 | {PRECISION_BYTES['BF16']:g} | {gib_per_billion('BF16')} | 16-bit weights |",
        f"| FP8 | {PRECISION_BYTES['FP8']:g} | {gib_per_billion('FP8')} | FP8, GGUF Q8_0, MLX 8-bit |",
        f"| INT4 | {PRECISION_BYTES['INT4']:g} | {gib_per_billion('INT4')} "
        "| GGUF Q4_K_M, AWQ, GPTQ, NVFP4, MLX 4-bit |",
        "",
        "- **GiB** (1,073,741,824 bytes) is the unit GPU and Mac memory is sold in, though it is "
        "labelled GB: an \"80 GB\" H100 holds 80 GiB. A billion parameters at 2 bytes each is "
        "2 billion bytes, or 1.86 GiB.",
        "- **Bytes per parameter** are those of real files: quantized checkpoints also store "
        "scale factors and usually keep the embeddings and output layer in 16-bit, so 4-bit "
        "files of 30B-and-up models come to 0.56–0.64 bytes per parameter rather than 0.5, "
        "and 8-bit files to 1.00–1.07.",
        "- **Released precision**: many current models are published only in 8-bit or 4-bit "
        "(DeepSeek, GLM-5, Kimi, gpt-oss). For them the BF16 column describes weights nobody "
        "publishes, and the lower column holds their full quality.",
        "- **MoE models** count all their parameters: every expert must sit in memory, even "
        "though each token uses only a few.",
        f"- **KV cache**: the attention keys and values kept for every token in context, "
        f"reserved for one {CONTEXT_TOKENS:,}-token conversation. It does not shrink when the "
        "weights are quantized, and depends on the attention design: 20–100 KiB per token for "
        "most current MoE models (DeepSeek's and Kimi's MLA, hybrid and sliding-window "
        "attention), 190–370 KiB for models such as Qwen3-235B, GLM-4.6 and Llama 3.3 70B, "
        "and 500 KiB for Llama 3.1 405B. This tool does not know it per model; "
        f"{KV_BYTES_PER_TOKEN // 1024} KiB is a middle value, generous for the newest models and "
        "short for the oldest large ones. Longer contexts and several users at once need more.",
        f"- **{RUNTIME_GIB:g} GiB per GPU** covers the CUDA or Metal context, the framework and "
        f"activations. Split across {NODE_GPU_COUNT} GPUs, a model needs it on each.",
        f"- **{FIT_HEADROOM:.0%}**: a model fits a GPU when it needs at most {FIT_HEADROOM:.0%} of "
        "its memory. GPUs report a little less than their label (an L40S with ECC on shows 45 of "
        "48 GiB), and engines want headroom: vLLM uses 90% unless told otherwise "
        f"(`--gpu-memory-utilization {FIT_HEADROOM:g}`). On computers whose GPU shares system RAM "
        "the whole share the system gives the GPU counts, as it already leaves the system its own.",
        f"- Example: a 70B dense model at FP8 needs 70 × {fp8_gib:.3g} + {kv_cache_gib(70):g} + "
        f"{RUNTIME_GIB:g} = {example:.1f} GiB, so it fits an H100 "
        f"({FIT_HEADROOM * h100.usable_gib:g} of {h100.usable_gib:g} GiB).",
    ]


def _speed_formula_section() -> list[str]:
    efficiency = BANDWIDTH_EFFICIENCY
    h100 = next(hw for hw in HARDWARE if hw.key == "h100")
    example_gb = 30 * PRECISION_BYTES["FP8"]
    example_ms = example_gb / (efficiency * h100.bandwidth_gbs) * 1000
    example = decode_tokens_per_s(30, "dense", "FP8", h100.bandwidth_gbs)
    gpu_ms = FIXED_MS_PER_TOKEN[("gpu", "dense")], FIXED_MS_PER_TOKEN[("gpu", "moe")]
    apple_ms = FIXED_MS_PER_TOKEN[("apple", "dense")], FIXED_MS_PER_TOKEN[("apple", "moe")]
    pairs, median, within_20 = _speed_accuracy()

    lines = [
        "### Speed formula",
        "",
        "Generating each token reads every active weight from memory once, so with one "
        "request at a time memory bandwidth, not compute, sets the speed:",
        "",
        "```",
        f"Time per token (ms) = bytes per token ÷ ({efficiency:g} × bandwidth × GPUs) + fixed cost",
        f"Bytes per token     = active parameters × bytes per parameter (× {MOE_BYTES_FACTOR:g} for MoE)",
        "Speed (tokens/s)    = 1000 ÷ time per token",
        "```",
        "",
        "- **Active parameters** are all the parameters of a dense model, and only those used "
        "per token for an MoE model (37B of DeepSeek V3's 671B). MoE models read somewhat "
        "more per token than that (the router, and shared layers that quantized files keep at "
        f"higher precision); ×{MOE_BYTES_FACTOR:g} is fitted on the measurements.",
        f"- **{efficiency:g}**: software reads weights at about {efficiency:.0%} of the "
        "memory's peak bandwidth.",
        f"- **Fixed cost per token** (starting the work for every layer, synchronising, "
        f"sampling): {gpu_ms[0]:g} ms for dense and {gpu_ms[1]:g} ms for MoE models on "
        f"NVIDIA, AMD and Intel hardware, {apple_ms[0]:g} and {apple_ms[1]:g} ms on Apple "
        f"Silicon. When {NODE_GPU_COUNT} GPUs split one request they also exchange results "
        f"at every layer: {SYNC_MS_PER_TOKEN['dense']:g} ms more for dense and "
        f"{SYNC_MS_PER_TOKEN['moe']:g} ms for MoE models. This is why small and MoE models "
        "fall short of their bandwidth limit on fast GPUs, and why 8 GPUs do not make one "
        "request 8× faster.",
        f"- Example: a 30B dense model at FP8 on an H100 SXM reads {example_gb:.1f} GB per "
        f"token: {example_gb:.1f} ÷ ({efficiency:g} × {h100.bandwidth_gbs:,.0f} GB/s) = "
        f"{example_ms:.1f} ms, plus {gpu_ms[0]:g} ms, so about {example:.0f} tokens/s.",
        "- **Assumes** one request, a short context (a long context's KV cache is read for "
        "every token too: a small MoE model runs about 20% slower at 8K tokens and half as "
        "fast at 32K), no speculative decoding or multi-token prediction (typically 1.5–2.5× "
        "faster), and an engine suited to the hardware: vLLM, SGLang or TensorRT-LLM on "
        "data-center GPUs, where llama.cpp runs 30–55% below these estimates for small models.",
        "- **Reading the prompt** is a separate cost the tables do not estimate, limited by "
        "compute rather than memory (about 2 × active parameters floating-point operations "
        "per prompt token). It is where computers whose GPU shares system RAM fall furthest "
        "behind: llama.cpp reads a 7B model's prompt at about 1,700 tokens/s on a Ryzen AI "
        "Max+ 395, 3,200 on an M5 Max and 3,300 on a DGX Spark, against 14,800 on an RTX 4090 "
        "([Apple](https://github.com/ggml-org/llama.cpp/discussions/4167), "
        "[Vulkan](https://github.com/ggml-org/llama.cpp/discussions/10879), "
        "[CUDA](https://github.com/ggml-org/llama.cpp/discussions/15013)). A 10,000-token "
        "prompt to a 70B model takes about half a minute on an M5 Max or a DGX Spark and a "
        "minute on the Ryzen, against a few seconds on data-center GPUs.",
        "",
        "#### How close are the estimates?",
        "",
        f"Checked against {len(pairs)} published single-request speeds "
        "(`arena_enrichment/measurements.py`): the median estimate is within "
        f"{median:.0%} of the measurement, and {within_20:.0%} are within ±20%. The constants "
        "were chosen with these same measurements, so this shows the formula fits them rather "
        "than proving it on new hardware. Most are of small models: large models have few "
        "measurements, and large dense models on a Mac none from llama-bench (LM Studio users "
        "report speeds 25–50% below these estimates). A selection:",
        "",
        "| Hardware | Model | Engine | Measured | Estimate |",
        "|----------|-------|--------|----------|----------|",
    ]
    for m, estimate in pairs:
        if m.in_readme:
            lines.append(f"| {m.hardware} | [{m.model}]({m.source}) | {m.engine} "
                         f"| {float(f'{m.tokens_per_s:.3g}'):g} tok/s | {_format_speed(estimate).lstrip('~')} |")
    lines.append("")
    lines.append("The largest misses are llama.cpp on data-center GPUs (the A100, H100 and "
                 "MI300X run 30–55% below the estimate with a 7B model) and an early-2025 SGLang "
                 "run of DeepSeek-R1 on 8× H200 at less than half the estimate; reported "
                 "DeepSeek speeds have varied widely between engine versions and settings.")
    return lines


def _generate_leaderboard_table(df: pd.DataFrame) -> list[str]:
    """Generate the enriched leaderboard markdown table."""
    lines = []

    header = ("| Rank | Model | Score | Params (B) | Arch | GiB BF16 | GiB FP8 | GiB INT4 "
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
        bf16_s, fp8_s, int4_s = (_format_gib(row.get(f"vram_{p.lower()}_serving_gib"))
                                 for p in FIT_PRECISIONS)
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
        # Speed — one request on the "Fits on" hardware; none for PCIe servers
        tok_s = row.get("est_tok_s_fp8")
        if pd.notna(tok_s):
            speed_s = _format_speed(tok_s)
        elif best_fp8 == "UNKNOWN":
            speed_s = "?"
        else:
            speed_s = "—"
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
        serving_col = f"vram_{p_lower}_serving_gib"
        if serving_col not in df.columns:
            continue

        print(f"\n=== Best Model Per Hardware ({precision} serving) ===")
        for hw_row in _hardware_rows():
            display, gpu_vram = hw_row.label, hw_row.memory_label
            best = _best_fit(df, serving_col, hw_row)
            if best is not None:
                speed = hardware_speed(hw_row.hardware, best, precision, hw_row.gpu_count)
                speed_s = f", {_format_speed(speed)}" if speed is not None else ""
                rank = int(best["rank"]) if pd.notna(best["rank"]) else "?"
                name = best["model_name"]
                total_b = best.get("total_params_b", "?")
                arch = best.get("architecture", "?")
                serving = round(hw_row.needed_gib(best[serving_col]), 1)
                arch_label = f" {arch.upper()}" if arch and arch != "dense" else ""
                print(
                    f"  {display:42s} ({gpu_vram:>11s}): "
                    f"#{rank:<4} {name} ({total_b}B{arch_label}, ~{serving} GiB serving{speed_s})"
                )
            else:
                print(f"  {display:42s} ({gpu_vram:>11s}): No model fits")

    print()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def check_health(
    total: int,
    resolved: int,
    leaderboard_date: str | None,
    expect_date: bool = False,
    aa_list_problem: bool = False,
    aa_list_age_days: float | None = None,
    lost_sizes: list[str] | None = None,
    unreadable_pages: list[str] | None = None,
) -> list[str]:
    """
    What is wrong with this run's data, in plain words with where to look.

    The run publishes regardless, on the last good data where a source
    failed; the daily workflow turns a non-empty list into an issue.
    """
    problems = []
    if expect_date and leaderboard_date is None:
        problems.append(
            "The leaderboard download has no publication date (column `leaderboard_publish_date`), "
            "so a leaderboard that stops updating would go unnoticed. Check the dataset's columns "
            "against `_fetch_hf_leaderboard` in `arena_enrichment/scrapers/arena_scraper.py`."
        )
    data_age = _days_since(leaderboard_date)
    if data_age is not None and data_age > LEADERBOARD_STALE_DAYS:
        problems.append(
            f"The Arena leaderboard was last published {data_age} days ago ({leaderboard_date}). "
            "Check that Arena still updates the Hugging Face dataset lmarena-ai/leaderboard-dataset "
            "read by `arena_enrichment/scrapers/arena_scraper.py`."
        )
    if aa_list_problem:
        copy = (f"the last good copy is {aa_list_age_days:.0f} days old"
                if aa_list_age_days is not None else "there is no saved copy")
        problems.append(
            f"The Artificial Analysis model list could not be read, and {copy}. Check "
            "`_extract_models_from_rsc` in `arena_enrichment/scrapers/aa_resolver.py` against "
            "https://artificialanalysis.ai/leaderboards/models."
        )
    if total and resolved / total < MIN_RESOLVED_SHARE:
        problems.append(
            f"Only {resolved} of {total} models ({resolved / total:.0%}) have parameter counts, "
            f"below the {MIN_RESOLVED_SHARE:.0%} minimum. The run log lists the unresolved models; "
            "add their sizes to `arena_enrichment/known_models.py`."
        )
    if unreadable_pages:
        problems.append(
            f"{len(unreadable_pages)} Artificial Analysis page(s) could not be read "
            f"({', '.join(sorted(unreadable_pages))}): the site blocked the request or the pages "
            "lost their specifications table. Open one in a browser, then check `_fetch_single_model` "
            "in `arena_enrichment/scrapers/aa_resolver.py`."
        )
    if lost_sizes:
        problems.append(
            f"{len(lost_sizes)} Artificial Analysis page(s) no longer show parameter counts "
            f"({', '.join(sorted(lost_sizes))}); the cached counts are used meanwhile. If several "
            "pages are affected, the page layout changed: check `_fetch_single_model` in "
            "`arena_enrichment/scrapers/aa_resolver.py`."
        )
    return problems


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
    parser.add_argument(
        "--health-file",
        help="Write the run's health problems here, one Markdown list item each "
             "(empty when healthy)",
        default=None,
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

    df, left_out = drop_unreleased(df)
    if left_out:
        print(f"  Left out (weights never released): {', '.join(left_out)}")

    # Step 2: Resolve parameter counts
    print("\nStep 2: Resolving model parameters...")
    use_network = not args.no_network

    # A corrupt page cache would fail every lookup quietly: stop here instead
    aa_resolver.validate_params_cache()

    # Pre-load AA model cache (one network request, or from disk)
    aa_lookup = None
    aa_list_problem = False
    aa_list_age = None
    if use_network:
        print("  Loading Artificial Analysis model database...")
        aa_lookup, aa_is_stale = get_aa_models(force_refresh=args.refresh_cache)
        print(f"  AA database: {len(aa_lookup)} entries loaded")
        if aa_is_stale:
            fallback = "using the last cached list" if aa_lookup else "no cached list either"
            print(f"  WARNING: AA model list fetch failed, {fallback}")
            aa_list_age = aa_resolver.model_list_age_days() if aa_lookup else None
            aa_list_problem = aa_list_age is None or aa_list_age > AA_LIST_MAX_AGE_DAYS

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
            df, resolution_counts, aa_is_stale=aa_list_problem, leaderboard_date=leaderboard_date
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

    problems = check_health(
        total=len(df),
        resolved=len(df) - len(unresolved),
        leaderboard_date=leaderboard_date,
        expect_date=args.input is None,
        aa_list_problem=aa_list_problem,
        aa_list_age_days=aa_list_age,
        lost_sizes=sorted(aa_resolver.LOST_SIZES),
        unreadable_pages=sorted(aa_resolver.UNREADABLE_PAGES),
    )
    for problem in problems:
        print(f"HEALTH: {problem}")
    if args.health_file:
        with open(args.health_file, "w", encoding="utf-8") as f:
            f.writelines(f"- {problem}\n" for problem in problems)


if __name__ == "__main__":
    main()
