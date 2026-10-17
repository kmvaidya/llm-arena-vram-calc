import json
import os

import pandas as pd
import pytest

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def aa_lookup():
    """Load the hand-crafted AA cache fixture for matching tests."""
    with open(os.path.join(FIXTURES_DIR, "aa_cache_sample.json"), encoding="utf-8") as f:
        data = json.load(f)
    return data["models"]


@pytest.fixture
def sample_leaderboard_df():
    """Minimal leaderboard DataFrame for VRAM/GPU column tests."""
    return pd.DataFrame([
        {"rank": 1, "model_name": "big-model-70b", "arena_score": 1300,
         "total_params_b": 70.0, "active_params_b": 70.0, "architecture": "dense"},
        {"rank": 2, "model_name": "moe-model-600b", "arena_score": 1280,
         "total_params_b": 600.0, "active_params_b": 30.0, "architecture": "moe"},
        {"rank": 3, "model_name": "small-model-8b", "arena_score": 1200,
         "total_params_b": 8.0, "active_params_b": 8.0, "architecture": "dense"},
        {"rank": 4, "model_name": "unknown-model", "arena_score": 1100,
         "total_params_b": None, "active_params_b": None, "architecture": None},
    ])


@pytest.fixture
def sample_rsc_payload():
    """Minimal RSC-style text containing a model array (no parameter counts since 2026-04)."""
    models = [
        {
            "slug": "test-model-70b",
            "name": "Test Model 70B",
            "isOpenWeights": True,
            "paramClass": "medium",
            "contextWindowTokens": 8192,
        },
        {
            "slug": "test-moe",
            "name": "Test MoE (Max)",
            "isOpenWeights": False,
            "paramClass": "large",
            "contextWindowTokens": 32768,
        },
    ]
    # Wrap in RSC-like text with some prefix junk
    return f"some_prefix_data[{json.dumps(models[0])},{json.dumps(models[1])}]some_suffix"
