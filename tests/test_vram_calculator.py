import pandas as pd
import pytest
from vram_calculator import (
    GPUS,
    NODE_GPUS,
    _find_best_gpu,
    _find_best_node,
    add_all_vram_and_gpu_columns,
    compute_vram_columns,
    decode_tokens_per_s,
    node_label,
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
    def test_tiny_model(self):
        assert _find_best_gpu(5) == "RTX 4090"

    def test_exactly_rtx_4090(self):
        assert _find_best_gpu(24) == "RTX 4090"

    def test_fits_rtx_5090(self):
        assert _find_best_gpu(30) == "RTX 5090"

    def test_fits_l40s(self):
        assert _find_best_gpu(40) == "L40S"

    def test_fits_h100(self):
        assert _find_best_gpu(75) == "H100 SXM"

    def test_70b_at_fp8_names_rtx_pro_not_h100_nvl(self):
        # 87.5 GB also fits a 94 GB H100 NVL, which is not a reference GPU
        assert _find_best_gpu(87.5) == "RTX PRO 6000 Blackwell"

    def test_fits_h200(self):
        assert _find_best_gpu(140) == "H200 SXM / NVL"

    def test_fits_b200(self):
        assert _find_best_gpu(150) == "B200 SXM"

    def test_fits_b300(self):
        assert _find_best_gpu(200) == "B300 SXM"

    def test_multi_gpu(self):
        assert _find_best_gpu(300) == "MULTI-GPU"


class TestFindBestNode:
    def test_fits_l40s_node(self):
        assert _find_best_node(300) == "8x L40S"

    def test_exactly_h100_node(self):
        assert _find_best_node(640) == "8x H100 SXM"

    def test_rtx_pro_node_is_server_edition(self):
        assert _find_best_node(700) == "8x RTX PRO 6000 Blackwell Server Edition"

    def test_deepseek_v3_at_fp8_names_h200_not_gaudi(self):
        # 856 GB also fits 8x Gaudi 3 (1024 GB), which is not a reference GPU
        assert _find_best_node(856) == "8x H200 SXM / NVL"

    def test_fits_b200_node(self):
        assert _find_best_node(1400) == "8x B200 SXM"

    def test_fits_b300_node(self):
        # A 1.6T model at FP8 needs 2000 GB with serving overhead
        assert _find_best_node(2000) == "8x B300 SXM"

    def test_multi_node(self):
        assert _find_best_node(3000) == "MULTI-NODE"

    def test_small_model_still_gets_smallest_node(self):
        # Desktop cards are not 8-GPU servers, so the smallest node is 8x L40S
        assert _find_best_node(10) == "8x L40S"


class TestGpuList:
    def test_keys_and_names_unique(self):
        assert len({g.key for g in GPUS}) == len(GPUS)
        assert len({g.name for g in GPUS}) == len(GPUS)

    def test_display_order_is_tier_then_memory(self):
        tiers = ["Consumer", "Workstation", "Data center"]
        order = [(tiers.index(g.tier), g.vram_gb) for g in GPUS]
        assert order == sorted(order)

    def test_nodes_exclude_desktop_cards(self):
        # GeForce cards (license bars data-center use) and workstation cards
        # are not sold in 8-GPU servers; the RTX PRO 6000 is, as Server Edition
        node_names = {g.name for g in NODE_GPUS}
        assert not any(g.tier == "Consumer" for g in NODE_GPUS)
        assert "Radeon AI PRO R9700" not in node_names
        rtx_pro = next(g for g in NODE_GPUS if g.key == "rtx_pro_6000")
        assert node_label(rtx_pro) == "8x RTX PRO 6000 Blackwell Server Edition"

    def test_nodes_sorted_by_memory(self):
        assert [g.vram_gb for g in NODE_GPUS] == sorted(g.vram_gb for g in NODE_GPUS)

    def test_reference_gpus_have_distinct_sizes(self):
        sizes = [g.vram_gb for g in GPUS if g.reference]
        assert len(sizes) == len(set(sizes))

    def test_reference_nodes_are_8gpu_servers(self):
        assert all(g.in_8gpu_servers for g in GPUS if g.reference and g.tier == "Data center")

    def test_a100_has_no_native_fp8(self):
        assert next(g for g in GPUS if g.key == "a100").native_fp8 is False


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

    def test_small_model_fits_consumer_gpu(self, sample_leaderboard_df):
        df = add_all_vram_and_gpu_columns(sample_leaderboard_df)
        row = df[df["model_name"] == "small-model-8b"].iloc[0]
        assert row["fits_h100_fp8"] is True
        assert row["fits_rtx_4090_fp8"] is True   # 10 GB
        assert row["best_gpu_fp8"] == "RTX 4090"

    def test_fit_columns_for_every_gpu(self, sample_leaderboard_df):
        df = add_all_vram_and_gpu_columns(sample_leaderboard_df)
        for gpu in GPUS:
            assert f"fits_{gpu.key}_fp8" in df.columns

    def test_huge_model_needs_multi_gpu(self, sample_leaderboard_df):
        df = add_all_vram_and_gpu_columns(sample_leaderboard_df)
        row = df[df["model_name"] == "moe-model-600b"].iloc[0]
        assert row["fits_h100_bf16"] is False
        assert row["best_gpu_bf16"] == "MULTI-GPU"

    def test_unknown_model_has_unknown_gpu(self, sample_leaderboard_df):
        df = add_all_vram_and_gpu_columns(sample_leaderboard_df)
        row = df[df["model_name"] == "unknown-model"].iloc[0]
        assert row["best_gpu_fp8"] == "UNKNOWN"

    def test_huge_model_gets_a_node(self, sample_leaderboard_df):
        df = add_all_vram_and_gpu_columns(sample_leaderboard_df)
        row = df[df["model_name"] == "moe-model-600b"].iloc[0]
        assert row["best_node_bf16"] == "8x B300 SXM"   # 1500 GB: over 8x B200's 1440
        assert row["best_node_fp8"] == "8x RTX PRO 6000 Blackwell Server Edition"  # 750 GB

    def test_unknown_model_has_unknown_node(self, sample_leaderboard_df):
        df = add_all_vram_and_gpu_columns(sample_leaderboard_df)
        row = df[df["model_name"] == "unknown-model"].iloc[0]
        assert row["best_node_fp8"] == "UNKNOWN"


class TestDecodeTokensPerS:
    def test_dense_formula(self):
        # 0.6 x 3350 GB/s / (30B x 1 byte) = 67 tokens/s
        assert decode_tokens_per_s(30, "dense", "FP8", 3350) == pytest.approx(67.0)

    def test_moe_reads_only_active_params_at_lower_mbu(self):
        # 0.3 x 4800 / (37B x 1 byte) for DeepSeek V3 on an H200
        assert decode_tokens_per_s(37, "moe", "FP8", 4800) == pytest.approx(0.3 * 4800 / 37)

    def test_int4_is_twice_fp8(self):
        fp8 = decode_tokens_per_s(30, "dense", "FP8", 3350)
        assert decode_tokens_per_s(30, "dense", "INT4", 3350) == pytest.approx(2 * fp8)

    def test_every_gpu_has_a_bandwidth(self):
        assert all(g.bandwidth_gbs > 0 for g in GPUS)


class TestComputeSpeedColumns:
    def test_speed_on_fits_on_gpu(self, sample_leaderboard_df):
        df = add_all_vram_and_gpu_columns(sample_leaderboard_df)
        row = df[df["model_name"] == "small-model-8b"].iloc[0]
        # FP8: 10 GB fits the RTX 4090: 0.6 x 1008 / 8 = 75.6
        assert row["est_tok_s_fp8"] == pytest.approx(75.6)
        # BF16: 20 GB also fits the RTX 4090: 0.6 x 1008 / 16 = 37.8
        assert row["est_tok_s_bf16"] == pytest.approx(37.8)

    def test_no_speed_when_several_gpus_needed(self, sample_leaderboard_df):
        df = add_all_vram_and_gpu_columns(sample_leaderboard_df)
        row = df[df["model_name"] == "moe-model-600b"].iloc[0]
        assert pd.isna(row["est_tok_s_fp8"])

    def test_no_speed_for_unknown_model(self, sample_leaderboard_df):
        df = add_all_vram_and_gpu_columns(sample_leaderboard_df)
        row = df[df["model_name"] == "unknown-model"].iloc[0]
        assert pd.isna(row["est_tok_s_fp8"])
