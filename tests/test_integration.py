"""
Integration test — runs the full pipeline with mocked network calls.

Verifies that scraping, parameter resolution, VRAM calculation,
GPU feasibility, and README generation work end-to-end.
"""

from __future__ import annotations

import json
import os

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
            "vram_bf16_weights_gb", "vram_fp8_serving_gb",
            "best_gpu_fp8", "fits_h100_fp8",
        ]:
            assert col in df.columns, f"Missing column: {col}"

        # DeepSeek-V3: should resolve via AA (MoE, 671B total)
        dsv3 = df[df["model_name"] == "DeepSeek-V3"].iloc[0]
        assert dsv3["total_params_b"] == 671.0
        assert dsv3["architecture"] == "moe"
        assert dsv3["vram_fp8_serving_gb"] > 600  # 671 * 1.0 * 1.25 = 838.75

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
        assert "## Best Model Per GPU" in readme
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
