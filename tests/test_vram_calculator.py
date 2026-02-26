import pandas as pd
import pytest
from vram_calculator import (
    _find_best_gpu,
    add_all_vram_and_gpu_columns,
    compute_vram_columns,
    practical_serving_gb,
    vram_estimate_gb,
)


class TestVramEstimateGb:
    def test_bf16_70b(self):
        assert vram_estimate_gb(70, "BF16") == 140.0

    def test_fp8_70b(self):
        assert vram_estimate_gb(70, "FP8") == 70.0

    def test_int4_70b(self):
        assert vram_estimate_gb(70, "INT4") == 35.0

    def test_mxfp4(self):
        assert vram_estimate_gb(70, "MXFP4") == pytest.approx(37.1, abs=0.1)

    def test_small_model(self):
        assert vram_estimate_gb(1, "BF16") == 2.0

    def test_fractional_params(self):
        assert vram_estimate_gb(7.6, "FP8") == 7.6

    def test_zero_params(self):
        assert vram_estimate_gb(0, "BF16") == 0.0

    def test_unknown_precision_raises(self):
        with pytest.raises(ValueError, match="Unknown precision"):
            vram_estimate_gb(70, "FP3")


class TestPracticalServingGb:
    def test_default_overhead_bf16(self):
        assert practical_serving_gb(70, "BF16") == 175.0

    def test_default_overhead_fp8(self):
        assert practical_serving_gb(70, "FP8") == 87.5

    def test_default_overhead_int4(self):
        assert practical_serving_gb(70, "INT4") == 43.75

    def test_custom_overhead(self):
        assert practical_serving_gb(70, "BF16", overhead_pct=0.5) == 210.0

    def test_zero_overhead(self):
        result = practical_serving_gb(70, "BF16", overhead_pct=0)
        assert result == vram_estimate_gb(70, "BF16")

    def test_moe_uses_total_params(self):
        # MoE model: all 671B params must be loaded, not just 37B active
        result = practical_serving_gb(671, "FP8")
        assert result == 671 * 1.0 * 1.25


class TestFindBestGpu:
    def test_fits_h100(self):
        assert _find_best_gpu(75) == "H100 SXM"

    def test_exactly_h100(self):
        assert _find_best_gpu(80) == "H100 SXM"

    def test_exceeds_h100_fits_rtx_pro(self):
        assert _find_best_gpu(85) == "RTX PRO 6000"

    def test_fits_h200(self):
        assert _find_best_gpu(100) == "H200 SXM"

    def test_fits_b200(self):
        assert _find_best_gpu(150) == "B200 SXM"

    def test_fits_b300(self):
        assert _find_best_gpu(200) == "B300 SXM"

    def test_multi_gpu(self):
        assert _find_best_gpu(300) == "MULTI-GPU"

    def test_tiny_model(self):
        assert _find_best_gpu(5) == "H100 SXM"


class TestComputeVramColumns:
    def test_columns_added(self, sample_leaderboard_df):
        df = compute_vram_columns(sample_leaderboard_df)
        for precision in ["bf16", "fp8", "int4"]:
            assert f"vram_{precision}_weights_gb" in df.columns
            assert f"vram_{precision}_serving_gb" in df.columns

    def test_nan_params_produce_none(self, sample_leaderboard_df):
        df = compute_vram_columns(sample_leaderboard_df)
        unknown_row = df[df["model_name"] == "unknown-model"].iloc[0]
        assert pd.isna(unknown_row["vram_bf16_weights_gb"])
        assert pd.isna(unknown_row["vram_bf16_serving_gb"])

    def test_values_correct(self, sample_leaderboard_df):
        df = compute_vram_columns(sample_leaderboard_df)
        row = df[df["model_name"] == "big-model-70b"].iloc[0]
        assert row["vram_bf16_weights_gb"] == 140.0
        assert row["vram_bf16_serving_gb"] == 175.0
        assert row["vram_fp8_weights_gb"] == 70.0
        assert row["vram_int4_weights_gb"] == 35.0


class TestAddAllColumns:
    def test_gpu_fit_columns_added(self, sample_leaderboard_df):
        df = add_all_vram_and_gpu_columns(sample_leaderboard_df)
        assert "fits_h100_bf16" in df.columns
        assert "fits_h100_fp8" in df.columns
        assert "best_gpu_fp8" in df.columns

    def test_small_model_fits_h100(self, sample_leaderboard_df):
        df = add_all_vram_and_gpu_columns(sample_leaderboard_df)
        row = df[df["model_name"] == "small-model-8b"].iloc[0]
        assert row["fits_h100_fp8"] is True
        assert row["best_gpu_fp8"] == "H100 SXM"

    def test_huge_model_needs_multi_gpu(self, sample_leaderboard_df):
        df = add_all_vram_and_gpu_columns(sample_leaderboard_df)
        row = df[df["model_name"] == "moe-model-600b"].iloc[0]
        assert row["fits_h100_bf16"] is False
        assert row["best_gpu_bf16"] == "MULTI-GPU"

    def test_unknown_model_has_unknown_gpu(self, sample_leaderboard_df):
        df = add_all_vram_and_gpu_columns(sample_leaderboard_df)
        row = df[df["model_name"] == "unknown-model"].iloc[0]
        assert row["best_gpu_fp8"] == "UNKNOWN"
