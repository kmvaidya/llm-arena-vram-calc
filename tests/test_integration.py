"""
Integration test — runs the full pipeline with mocked network calls.

Verifies that scraping, parameter resolution, VRAM calculation,
GPU feasibility, and README generation work end-to-end.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from enrich_arena import (
    generate_readme,
    resolve_model_params,
)
from vram_calculator import add_all_vram_and_gpu_columns

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def integration_leaderboard() -> pd.DataFrame:
    """Simulate a scraped Arena leaderboard (5 models, diverse cases)."""
    return pd.DataFrame([
        {"rank": 1, "model_name": "DeepSeek-V3", "arena_score": 1380,
         "ci_lower": 1375, "ci_upper": 1385, "votes": 50000,
         "organization": "DeepSeek", "license": "MIT"},
        {"rank": 2, "model_name": "Qwen2.5-72B-Instruct", "arena_score": 1350,
         "ci_lower": 1345, "ci_upper": 1355, "votes": 40000,
         "organization": "Alibaba", "license": "Apache 2.0"},
        {"rank": 3, "model_name": "Llama-4-Maverick", "arena_score": 1320,
         "ci_lower": 1315, "ci_upper": 1325, "votes": 30000,
         "organization": "Meta", "license": "Llama 4"},
        {"rank": 4, "model_name": "Gemma-3-27B-IT", "arena_score": 1290,
         "ci_lower": 1285, "ci_upper": 1295, "votes": 20000,
         "organization": "Google", "license": "Apache 2.0"},
        {"rank": 5, "model_name": "mystery-model", "arena_score": 1200,
         "ci_lower": 1195, "ci_upper": 1205, "votes": 10000,
         "organization": "Unknown", "license": "MIT"},
    ])


@pytest.fixture
def integration_aa_lookup() -> dict:
    """Load the fixture AA cache for integration tests."""
    with open(os.path.join(FIXTURES_DIR, "aa_cache_sample.json"), encoding="utf-8") as f:
        data = json.load(f)
    return data["models"]


class TestFullPipeline:
    """End-to-end: resolve params → compute VRAM → generate README."""

    def test_resolution_and_vram_columns(
        self, integration_leaderboard, integration_aa_lookup
    ):
        df = integration_leaderboard.copy()
        resolution_counts: dict[str, int] = {}

        for idx, row in df.iterrows():
            params, source = resolve_model_params(
                row["model_name"], use_network=False, aa_lookup=integration_aa_lookup
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

        df = add_all_vram_and_gpu_columns(df)

        # All expected columns exist
        for col in [
            "total_params_b", "active_params_b", "architecture",
            "vram_bf16_weights_gib", "vram_fp8_serving_gib",
            "best_gpu_fp8", "fits_h100_fp8",
        ]:
            assert col in df.columns, f"Missing column: {col}"

        # DeepSeek-V3: should resolve via AA (MoE, 671B total)
        dsv3 = df[df["model_name"] == "DeepSeek-V3"].iloc[0]
        assert dsv3["total_params_b"] == 671.0
        assert dsv3["architecture"] == "moe"
        assert dsv3["vram_fp8_serving_gib"] > 600  # 671B x 1.06 bytes = 662 GiB of weights

        # Llama-4-Maverick: should resolve via override (400B MoE)
        mav = df[df["model_name"] == "Llama-4-Maverick"].iloc[0]
        assert mav["total_params_b"] == 400
        assert mav["active_params_b"] == 17
        assert mav["architecture"] == "moe"

        # Gemma-3-27B-IT: resolves via AA (27.2B) or name parsing (27B), both dense
        gemma = df[df["model_name"] == "Gemma-3-27B-IT"].iloc[0]
        assert 27.0 <= gemma["total_params_b"] <= 27.2
        assert gemma["architecture"] == "dense"
        assert gemma["fits_h100_fp8"] is True  # ~27 * 1.0 * 1.25 ≈ 34 < 80

        # mystery-model: should be unresolved
        mystery = df[df["model_name"] == "mystery-model"].iloc[0]
        assert pd.isna(mystery["total_params_b"])
        assert mystery["best_gpu_fp8"] == "UNKNOWN"

        # Resolution source counts
        assert resolution_counts.get("UNKNOWN", 0) == 1  # only mystery-model

    def test_readme_generation(
        self, integration_leaderboard, integration_aa_lookup
    ):
        df = integration_leaderboard.copy()
        resolution_counts: dict[str, int] = {}

        for idx, row in df.iterrows():
            params, source = resolve_model_params(
                row["model_name"], use_network=False, aa_lookup=integration_aa_lookup
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

        df = add_all_vram_and_gpu_columns(df)
        readme = generate_readme(df, resolution_counts)

        # Contains expected sections
        assert "# LLM Arena VRAM Calculator" in readme
        assert "## Best Model Per Hardware" in readme
        assert "## Full Leaderboard" in readme
        assert "## Architecture" in readme
        assert "## Usage" in readme

        # Contains model names
        assert "DeepSeek-V3" in readme
        assert "Llama-4-Maverick" in readme

        # Contains resolution stats
        assert "Resolved:" in readme
        assert "4 (80.0%)" in readme  # 4 out of 5 resolved

    def test_stale_warning_in_readme(
        self, integration_leaderboard, integration_aa_lookup
    ):
        df = integration_leaderboard.copy()
        for idx, row in df.iterrows():
            params, _ = resolve_model_params(
                row["model_name"], use_network=False, aa_lookup=integration_aa_lookup
            )
            if params:
                df.at[idx, "total_params_b"] = params["total_params_b"]
                df.at[idx, "active_params_b"] = params["active_params_b"]
                df.at[idx, "architecture"] = params["architecture"]

        df = add_all_vram_and_gpu_columns(df)

        readme_stale = generate_readme(df, {}, aa_is_stale=True)
        assert "stale" in readme_stale.lower()

        readme_fresh = generate_readme(df, {}, aa_is_stale=False)
        assert "stale" not in readme_fresh.lower()

    def test_arena_date_in_readme(self, integration_leaderboard):
        df = add_all_vram_and_gpu_columns(integration_leaderboard.copy().assign(
            total_params_b=None, active_params_b=None, architecture=None,
        ))
        today = datetime.now(UTC).date()

        fresh = generate_readme(df, {}, leaderboard_date=str(today - timedelta(days=2)))
        assert f"**Arena data:** {today - timedelta(days=2)}" in fresh
        assert "days old" not in fresh

        old = generate_readme(df, {}, leaderboard_date=str(today - timedelta(days=30)))
        assert "data is 30 days old" in old

        undated = generate_readme(df, {})
        assert "Arena data" not in undated

    def _two_models(self, integration_leaderboard):
        df = integration_leaderboard.copy().head(2)
        # A 671B MoE (37B active) that needs 8 GPUs, and a 72B dense model
        df["total_params_b"] = [671.0, 72.0]
        df["active_params_b"] = [37.0, 72.0]
        df["architecture"] = ["moe", "dense"]
        return generate_readme(add_all_vram_and_gpu_columns(df), {})

    def test_one_hardware_table(self, integration_leaderboard):
        readme = self._two_models(integration_leaderboard)
        assert "## Best Model Per Hardware" in readme
        assert "## Best Model Per 8-GPU Node" not in readme
        assert "### GPUs" not in readme
        # 8x H200: BF16 DeepSeek needs 1,266 GiB of 1,128, so Qwen; FP8 DeepSeek
        # needs 667.9 + 7 x 1.5 = 678 GiB.  Speeds: 144 GB / 30,720 GB/s + 1 + 3 ms;
        # 45.1 GB / 30,720 GB/s + 2 + 6 ms
        assert ("| 8x H200 SXM | 1,128 GiB | 8 × 4.8 TB/s "
                "| Qwen2.5-72B-Instruct (#2, 150 GiB, ~120 tok/s) "
                "| DeepSeek-V3 (#1, 678 GiB, ~110 tok/s) "
                "| DeepSeek-V3 (#1, 391 GiB, ~110 tok/s) |") in readme
        # PCIe servers get no speed
        l40s_row = next(line for line in readme.splitlines() if "| 8x L40S |" in line)
        assert "Qwen2.5-72B-Instruct (#2, 56 GiB)" in l40s_row
        assert "tok/s" not in l40s_row

    def test_fits_on_and_speed_in_leaderboard(self, integration_leaderboard):
        readme = self._two_models(integration_leaderboard)
        # DeepSeek at FP8 needs 678 GiB: over 8x H100 (640), within 8x RTX PRO
        # 6000 (768), a PCIe server with no speed estimate
        assert "| 8x RTX PRO 6000 Blackwell Server Edition (FP8) | — |" in readme
        # Qwen 72B at FP8 needs 76.6 GiB of an H100's 80: 76.3 GB / 2,680 GB/s + 1 ms
        assert "| H100 SXM (FP8) | ~34 tok/s |" in readme
        # Whole GiB from 100 up, one decimal below
        assert "| 671 (37) | MoE | 1,255 | 668 | 380 |" in readme
        assert "| 72 | Dense | 140 | 76.6 | 45.7 |" in readme
        assert "*Speed* is the estimated generation speed for one request there" in readme

    def test_hardware_tiers_in_readme(self, integration_leaderboard):
        df = integration_leaderboard.copy().head(1)
        df["total_params_b"], df["active_params_b"], df["architecture"] = 8.0, 8.0, "dense"
        readme = generate_readme(add_all_vram_and_gpu_columns(df), {})

        # 8B at BF16: 14.9 + 2 + 1.5 = 18.4 GiB; 16 GB / 806 GB/s + 1 ms = 48 tok/s
        assert ("| Consumer GPU | RTX 4090 | 24 GiB | 1.01 TB/s "
                "| DeepSeek-V3 (#1, 18 GiB, ~48 tok/s) |") in readme
        # Macs: 75% of RAM, Apple's fixed cost: 16 GB / 245.6 GB/s + 3 ms
        assert ("| Unified memory | Mac mini / MacBook Pro, M5 Pro 64 GB | 48 of 64 GiB "
                "| 0.307 TB/s | DeepSeek-V3 (#1, 18 GiB, ~15 tok/s) |") in readme
        assert "|  | DGX Spark (GB10), 128 GB | 115 of 128 GiB | 0.273 TB/s |" in readme
        # 8-GPU servers: no desktop hardware, the RTX PRO 6000 as Server Edition
        for desktop in ("8x RTX 4090", "8x Radeon AI PRO R9700", "8x Mac", "8x DGX Spark",
                        "8x DGX Station"):
            assert desktop not in readme
        assert "| 8x RTX PRO 6000 Blackwell Server Edition | 768 GiB |" in readme
        # The notes say why
        assert "GeForce driver license bars data-center deployment" in readme
        assert "sudo sysctl iogpu.wired_limit_mb" in readme

    def test_formulas_and_accuracy_in_readme(self, integration_leaderboard):
        readme = self._two_models(integration_leaderboard)

        assert "### Memory formula" in readme
        assert "Memory (GiB) = total parameters (billions) × GiB per billion parameters" in readme
        assert ("+ KV cache: 4 GiB (16,384 tokens × 256 KiB; 2 GiB under 15B parameters)"
                in readme)
        assert "+ 1.5 GiB per GPU" in readme
        assert "| FP8 | 1.06 | 0.987 | FP8, GGUF Q8_0, MLX 8-bit |" in readme
        assert "70 × 0.987 + 4 + 1.5 = 74.6 GiB, so it fits an H100 (80 GiB)" in readme

        assert "### Speed formula" in readme
        assert ("Time per token (ms) = bytes per token ÷ (0.8 × bandwidth × GPUs) + fixed cost"
                in readme)
        assert ("31.8 ÷ (0.8 × 3,350 GB/s) = 11.9 ms, plus 1 ms, so about 78 tokens/s"
                in readme)

        assert "#### How close are the estimates?" in readme
        assert "Checked against 49 published single-request speeds" in readme
        assert "| DGX Spark | [gpt-oss-120b MXFP4](" in readme
