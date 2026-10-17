import pandas as pd
import pytest
from vram_calculator import (
    HARDWARE,
    NODE_GPUS,
    TIERS,
    _find_best_gpu,
    _find_best_node,
    add_all_vram_and_gpu_columns,
    compute_vram_columns,
    decode_tokens_per_s,
    fits,
    hardware_speed,
    kv_cache_gib,
    node_label,
    node_serving_gib,
    serving_gib,
    weights_gib,
)

GIB = 2**30


def _hw(key):
    return next(h for h in HARDWARE if h.key == key)


class TestWeightsGib:
    def test_bf16_70b(self):
        # 140 billion bytes
        assert weights_gib(70, "BF16") == pytest.approx(140e9 / GIB)

    def test_8bit_70b_includes_scales_and_16bit_layers(self):
        assert weights_gib(70, "FP8") == pytest.approx(74.2e9 / GIB)

    def test_4bit_70b_includes_scales_and_16bit_layers(self):
        assert weights_gib(70, "INT4") == pytest.approx(42e9 / GIB)

    def test_mxfp4(self):
        assert weights_gib(70, "MXFP4") == pytest.approx(39.2e9 / GIB)

    def test_zero_params(self):
        assert weights_gib(0, "BF16") == 0.0

    def test_unknown_precision_raises(self):
        with pytest.raises(ValueError, match="Unknown precision"):
            weights_gib(70, "FP3")


class TestServingGib:
    def test_kv_cache_for_16k_tokens(self):
        # 16,384 tokens x 256 KiB = 4 GiB
        assert kv_cache_gib(70) == 4.0

    def test_small_models_get_half_the_kv_cache(self):
        assert kv_cache_gib(8) == 2.0

    def test_kv_cache_does_not_shrink_with_precision(self):
        bf16 = serving_gib(70, "BF16") - weights_gib(70, "BF16")
        int4 = serving_gib(70, "INT4") - weights_gib(70, "INT4")
        assert bf16 == pytest.approx(int4)

    def test_70b_fp8(self):
        # 69.1 weights + 4 KV + 1.5 runtime
        assert serving_gib(70, "FP8") == pytest.approx(74.604, abs=0.001)

    def test_runtime_per_gpu(self):
        assert serving_gib(70, "FP8", gpu_count=8) - serving_gib(70, "FP8") == pytest.approx(10.5)
        assert node_serving_gib(serving_gib(70, "FP8")) == pytest.approx(serving_gib(70, "FP8", gpu_count=8))

    def test_moe_uses_total_params(self):
        # Memory is about total params, not active
        assert serving_gib(600, "BF16") > 1100


class TestFindBestGpu:
    def test_tiny_model(self):
        assert _find_best_gpu(5) == "RTX 4090"

    def test_rtx_4090_holds_95_percent_of_24_gib(self):
        # A model fits within FIT_HEADROOM of the memory: 22.8 of 24 GiB
        assert _find_best_gpu(22.7) == "RTX 4090"
        assert _find_best_gpu(22.9) == "RTX 5090"

    def test_fits_l40s(self):
        assert _find_best_gpu(40) == "L40S"

    def test_70b_at_fp8_fits_h100(self):
        assert _find_best_gpu(serving_gib(70, "FP8")) == "H100 SXM"

    def test_just_over_h100_names_rtx_pro_not_h100_nvl(self):
        # 81 GiB also fits a 94 GiB H100 NVL, which is not a reference GPU
        assert _find_best_gpu(81) == "RTX PRO 6000 Blackwell"

    def test_fits_h200(self):
        assert _find_best_gpu(133) == "H200 SXM / NVL"
        assert _find_best_gpu(135) == "B200"  # over 95% of 141 GiB

    def test_fits_b200(self):
        assert _find_best_gpu(160) == "B200"

    def test_fits_b300(self):
        assert _find_best_gpu(254) == "B300"  # 95% of 268 GiB is 254.6

    def test_multi_gpu(self):
        assert _find_best_gpu(255) == "MULTI-GPU"


class TestFindBestNode:
    def test_small_model_gets_smallest_node(self):
        # Desktop cards are not 8-GPU servers, so the smallest node is 8x L40S
        assert _find_best_node(10) == "8x L40S"

    def test_every_gpu_carries_its_runtime(self):
        # 8x L40S holds 364.8 GiB (95% of 384), less 7 more runtimes (10.5 GiB)
        assert _find_best_node(354) == "8x L40S"
        assert _find_best_node(355) == "8x H100 SXM"

    def test_deepseek_v3_at_fp8_fits_rtx_pro_server(self):
        # 667.9 + 10.5 GiB: over 95% of 8x H100's 640, within 95% of 8x RTX PRO 6000's 768
        assert _find_best_node(serving_gib(671, "FP8")) == "8x RTX PRO 6000 Blackwell Server Edition"

    def test_names_h200_not_gaudi(self):
        # 910.5 GiB also fits 8x Gaudi 3 (1,024 GiB), which is not a reference GPU
        assert _find_best_node(900) == "8x H200 SXM"

    def test_fits_b200_node(self):
        assert _find_best_node(1350) == "8x B200"  # 1,360.5 of 1,368 GiB

    def test_1_6t_model_at_fp8_fits_b300_node(self):
        assert _find_best_node(serving_gib(1600, "FP8")) == "8x B300"

    def test_multi_node(self):
        assert _find_best_node(3000) == "MULTI-NODE"


class TestHardwareList:
    def test_keys_and_names_unique(self):
        assert len({h.key for h in HARDWARE}) == len(HARDWARE)
        assert len({h.name for h in HARDWARE}) == len(HARDWARE)

    def test_display_order_is_tier_then_usable_memory(self):
        order = [(TIERS.index(h.tier), h.usable_gib) for h in HARDWARE]
        assert order == sorted(order)

    def test_memory_counted_in_gib(self):
        # An "80 GB" H100 holds 80 GiB
        assert _hw("h100").usable_gib == 80

    def test_macs_give_the_gpu_75_percent(self):
        mac = _hw("m5_max_128")
        assert mac.platform == "apple"
        assert mac.usable_gib == 96

    def test_unified_memory_keeps_some_for_the_system(self):
        for hw in HARDWARE:
            if hw.tier == "Unified memory":
                assert hw.usable_memory_gb < hw.memory_gb

    def test_unified_memory_needs_no_extra_headroom(self):
        # 48 GiB is already what macOS gives the GPU of a 64 GB Mac
        mac = _hw("m5_pro_64")
        assert fits(mac, 48) and not fits(mac, 48.1)
        h100 = _hw("h100")
        assert fits(h100, 76) and not fits(h100, 76.1)

    def test_hgx_boards_give_less_than_the_package(self):
        assert _hw("b200").usable_memory_gb == 180
        assert _hw("b300").usable_memory_gb == 268

    def test_nodes_exclude_desktop_hardware(self):
        # GeForce cards (license bars data-center use), unified-memory
        # computers and workstation cards are not sold in 8-GPU servers; the
        # RTX PRO 6000 is, as Server Edition
        assert all(h.tier in ("Data center GPU", "Workstation") for h in NODE_GPUS)
        assert {h.key for h in NODE_GPUS if h.tier == "Workstation"} == {"rtx_pro_6000"}
        assert node_label(_hw("rtx_pro_6000")) == "8x RTX PRO 6000 Blackwell Server Edition"

    def test_pcie_servers_have_no_gpu_fabric(self):
        assert {h.key for h in NODE_GPUS if not h.gpu_fabric} == {"l40s", "h100_nvl", "rtx_pro_6000"}

    def test_nodes_sorted_by_memory(self):
        assert [g.usable_gib for g in NODE_GPUS] == sorted(g.usable_gib for g in NODE_GPUS)

    def test_reference_gpus_have_distinct_sizes(self):
        sizes = [h.usable_gib for h in HARDWARE if h.reference]
        assert len(sizes) == len(set(sizes))

    def test_every_device_has_a_bandwidth(self):
        assert all(h.bandwidth_gbs > 0 for h in HARDWARE)

    def test_a100_has_no_native_fp8(self):
        assert _hw("a100").native_fp8 is False


class TestDecodeTokensPerS:
    def test_dense_formula(self):
        # 31.8 GB / (0.8 x 3350 GB/s) = 11.87 ms, + 1 ms
        assert decode_tokens_per_s(30, "dense", "FP8", 3350) == pytest.approx(1000 / 12.866, rel=1e-3)

    def test_moe_reads_active_params_plus_15_percent(self):
        # 37 x 1.06 x 1.15 = 45.1 GB / 3840 GB/s = 11.75 ms, + 2 ms
        assert decode_tokens_per_s(37, "moe", "FP8", 4800) == pytest.approx(1000 / 13.746, rel=1e-3)

    def test_apple_has_a_larger_fixed_cost_for_moe(self):
        # Current llama.cpp builds run dense models on Apple with the GPUs' 1 ms
        gpu, apple = (decode_tokens_per_s(6.74, "dense", "BF16", 614, p) for p in ("gpu", "apple"))
        assert apple == pytest.approx(gpu)
        gpu, apple = (decode_tokens_per_s(5.1, "moe", "INT4", 614, p) for p in ("gpu", "apple"))
        assert 1000 / apple - 1000 / gpu == pytest.approx(3.0)

    def test_eight_gpus_add_sync_cost(self):
        # 74.8 GB / (0.8 x 4800 x 8) = 2.44 ms, + 1 ms, + 3 ms sync
        assert decode_tokens_per_s(70.6, "dense", "FP8", 4800, gpu_count=8) == pytest.approx(155.4, abs=0.1)

    def test_fixed_cost_caps_tiny_models(self):
        assert decode_tokens_per_s(0.001, "dense", "INT4", 8000) == pytest.approx(1000, rel=1e-3)

    def test_lower_precision_is_faster(self):
        fp8 = decode_tokens_per_s(30, "dense", "FP8", 3350)
        assert decode_tokens_per_s(30, "dense", "INT4", 3350) > fp8


class TestHardwareSpeed:
    MODEL = pd.Series({"active_params_b": 30.0, "architecture": "dense"})

    def test_single_gpu(self):
        assert hardware_speed(_hw("h100"), self.MODEL, "FP8") == pytest.approx(77.7, abs=0.1)

    def test_no_speed_for_pcie_servers(self):
        assert hardware_speed(_hw("l40s"), self.MODEL, "FP8", gpu_count=8) is None

    def test_speed_for_fabric_servers(self):
        assert hardware_speed(_hw("h100"), self.MODEL, "FP8", gpu_count=8) is not None

    def test_no_speed_for_unknown_architecture(self):
        assert hardware_speed(_hw("h100"), pd.Series({"active_params_b": 30.0, "architecture": None}),
                              "FP8") is None


class TestComputeVramColumns:
    def test_columns_added(self, sample_leaderboard_df):
        df = compute_vram_columns(sample_leaderboard_df)
        for p in ["bf16", "fp8", "int4"]:
            assert f"vram_{p}_weights_gib" in df.columns
            assert f"vram_{p}_serving_gib" in df.columns

    def test_nan_params_produce_none(self, sample_leaderboard_df):
        df = compute_vram_columns(sample_leaderboard_df)
        row = df[df["model_name"] == "unknown-model"].iloc[0]
        assert pd.isna(row["vram_bf16_weights_gib"])
        assert pd.isna(row["vram_fp8_serving_gib"])

    def test_values_correct(self, sample_leaderboard_df):
        df = compute_vram_columns(sample_leaderboard_df)
        row = df[df["model_name"] == "big-model-70b"].iloc[0]
        assert row["vram_bf16_weights_gib"] == 130.4
        assert row["vram_fp8_serving_gib"] == 74.6


class TestAddAllColumns:
    def test_fit_columns_for_every_device(self, sample_leaderboard_df):
        df = add_all_vram_and_gpu_columns(sample_leaderboard_df)
        for hw in HARDWARE:
            assert f"fits_{hw.key}_fp8" in df.columns
        assert "best_gpu_fp8" in df.columns
        assert "best_node_fp8" in df.columns

    def test_small_model_fits_consumer_gpu(self, sample_leaderboard_df):
        df = add_all_vram_and_gpu_columns(sample_leaderboard_df)
        row = df[df["model_name"] == "small-model-8b"].iloc[0]
        assert row["fits_rtx_4090_fp8"] is True    # 11.4 GiB
        assert row["fits_m5_pro_64_bf16"] is True  # 18.4 GiB of 48
        assert row["best_gpu_fp8"] == "RTX 4090"

    def test_70b_fp8_fits_h100(self, sample_leaderboard_df):
        df = add_all_vram_and_gpu_columns(sample_leaderboard_df)
        row = df[df["model_name"] == "big-model-70b"].iloc[0]
        assert row["best_gpu_fp8"] == "H100 SXM"   # 74.6 GiB of 76
        assert row["best_gpu_bf16"] == "B200"      # 135.9 GiB: over 95% of an H200's 141

    def test_huge_model_gets_a_node(self, sample_leaderboard_df):
        df = add_all_vram_and_gpu_columns(sample_leaderboard_df)
        row = df[df["model_name"] == "moe-model-600b"].iloc[0]
        assert row["best_gpu_fp8"] == "MULTI-GPU"
        # 608.3 GiB: just over 95% of 8x H100's 640, so the 768 GiB PCIe server
        assert row["best_node_fp8"] == "8x RTX PRO 6000 Blackwell Server Edition"
        assert row["best_node_bf16"] == "8x B200"      # 1,134 GiB: over 95% of 8x H200

    def test_unknown_model_has_unknown_gpu(self, sample_leaderboard_df):
        df = add_all_vram_and_gpu_columns(sample_leaderboard_df)
        row = df[df["model_name"] == "unknown-model"].iloc[0]
        assert row["best_gpu_fp8"] == "UNKNOWN"
        assert row["best_node_fp8"] == "UNKNOWN"
        assert pd.isna(row["est_tok_s_fp8"])


class TestComputeSpeedColumns:
    def test_speed_on_fits_on_gpu(self, sample_leaderboard_df):
        df = add_all_vram_and_gpu_columns(sample_leaderboard_df)
        row = df[df["model_name"] == "small-model-8b"].iloc[0]
        # RTX 4090: 8.48 GB / 806.4 GB/s = 10.5 ms, + 1 ms
        assert row["est_tok_s_fp8"] == pytest.approx(86.8)
        assert row["est_tok_s_bf16"] == pytest.approx(48.0)

    def test_speed_on_fits_on_node(self):
        df = add_all_vram_and_gpu_columns(pd.DataFrame([
            {"total_params_b": 1000.0, "active_params_b": 32.0, "architecture": "moe"},
        ]))
        row = df.iloc[0]
        # 1,003 GiB at FP8: 8x H200.  39.0 GB / 30,720 GB/s = 1.27 ms, + 2 ms, + 6 ms sync
        assert row["best_node_fp8"] == "8x H200 SXM"
        assert row["est_tok_s_fp8"] == pytest.approx(107.9)

    def test_no_speed_on_pcie_server(self, sample_leaderboard_df):
        df = add_all_vram_and_gpu_columns(sample_leaderboard_df)
        row = df[df["model_name"] == "moe-model-600b"].iloc[0]
        assert pd.isna(row["est_tok_s_fp8"])
