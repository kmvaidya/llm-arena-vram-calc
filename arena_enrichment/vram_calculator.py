"""
VRAM estimation and GPU fit calculations.

Computes estimated VRAM requirements at multiple precisions, determines
which single-GPU and 8-GPU node configurations can serve each model, and
estimates single-request decode speed on one GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Bytes per parameter for each precision
PRECISION_BYTES = {
    "BF16": 2.0,      # 16-bit = 2 bytes
    "FP8": 1.0,       # 8-bit = 1 byte
    "INT4": 0.5,      # 4-bit = 0.5 bytes
    "MXFP4": 0.53,    # ~4.25 bits (used by GPT-OSS natively)
}

@dataclass(frozen=True)
class Gpu:
    """A GPU the tables report on."""

    key: str                # column id, as in fits_<key>_<precision>
    name: str
    vram_gb: int            # memory a server can use
    bandwidth_gbs: int      # peak memory bandwidth, GB/s
    tier: str               # "Consumer", "Workstation" or "Data center"
    architecture: str
    native_fp8: bool
    # Sold in 8-GPU servers.  GeForce cards are not (NVIDIA's GeForce driver
    # license bars data-center deployment), nor are workstation cards, which
    # are built for desktops with up to four cards
    in_8gpu_servers: bool = False
    # The version that goes in 8-GPU servers, when not every version does
    server_variant: str = ""
    # The most widely deployed GPU at its memory size.  The leaderboard's
    # "Fits on" names the smallest reference GPU (or 8-GPU server) that fits,
    # so it reads "8x H200" rather than a rarer part that happens to be smaller
    reference: bool = False


# Every GPU in the tables, in display order: by tier, then memory.  Ties in
# memory keep this order when picking the smallest GPU that fits.
#
# Checked 2026-10 against vendor and spec sources, memory and bandwidth both.
# B200 lists the 180 GB per GPU usable on an HGX B200 board
# (192 GB on the package).  Rubin ships as HGX Rubin NVL8 from H2 2026.  Left
# out: rack-scale-only parts (GB200/GB300 NVL72, MI455X in AMD Helios), which
# come as 72-GPU racks rather than single GPUs or 8-GPU servers, and AMD's
# MI440X 8-GPU platform, whose memory is not yet published.
GPUS: list[Gpu] = [
    Gpu("rtx_4090", "RTX 4090", 24, 1008, "Consumer", "Ada Lovelace", True, reference=True),
    Gpu("rtx_5090", "RTX 5090", 32, 1792, "Consumer", "Blackwell", True, reference=True),
    Gpu("r9700", "Radeon AI PRO R9700", 32, 640, "Workstation", "RDNA 4", True),
    Gpu("rtx_pro_6000", "RTX PRO 6000 Blackwell", 96, 1792, "Workstation", "Blackwell", True,
        in_8gpu_servers=True, server_variant="Server Edition", reference=True),
    Gpu("l40s", "L40S", 48, 864, "Data center", "Ada Lovelace", True,
        in_8gpu_servers=True, reference=True),
    Gpu("h100", "H100 SXM", 80, 3350, "Data center", "Hopper", True,
        in_8gpu_servers=True, reference=True),
    Gpu("a100", "A100 80GB", 80, 2039, "Data center", "Ampere", False, in_8gpu_servers=True),
    Gpu("h100_nvl", "H100 NVL", 94, 3900, "Data center", "Hopper", True, in_8gpu_servers=True),
    Gpu("gaudi3", "Gaudi 3", 128, 3700, "Data center", "Gaudi 3", True, in_8gpu_servers=True),
    Gpu("h200", "H200 SXM / NVL", 141, 4800, "Data center", "Hopper", True,
        in_8gpu_servers=True, reference=True),
    Gpu("b200", "B200 SXM", 180, 8000, "Data center", "Blackwell", True,
        in_8gpu_servers=True, reference=True),
    Gpu("mi300x", "MI300X", 192, 5300, "Data center", "CDNA 3", True, in_8gpu_servers=True),
    Gpu("mi325x", "MI325X", 256, 6000, "Data center", "CDNA 3", True, in_8gpu_servers=True),
    Gpu("b300", "B300 SXM", 288, 8000, "Data center", "Blackwell Ultra", True,
        in_8gpu_servers=True, reference=True),
    Gpu("rubin", "Rubin", 288, 22000, "Data center", "Rubin", True, in_8gpu_servers=True),
    Gpu("mi355x", "MI350X / MI355X", 288, 8000, "Data center", "CDNA 4", True, in_8gpu_servers=True),
]

# GPUs sorted by VRAM ascending (for finding smallest fitting GPU)
GPUS_BY_VRAM = sorted(GPUS, key=lambda g: g.vram_gb)

# GPUs in a standard server node (HGX/DGX-style), the usual way to serve
# models too large for one GPU: weights are split across the node's GPUs
NODE_GPU_COUNT = 8

# GPUs sold in 8-GPU servers, sorted by VRAM ascending
NODE_GPUS = [g for g in GPUS_BY_VRAM if g.in_8gpu_servers]

# The reference GPUs and 8-GPU servers that "Fits on" chooses from, smallest first
FITS_ON_GPUS = [g for g in GPUS_BY_VRAM if g.reference]
FITS_ON_NODES = [g for g in NODE_GPUS if g.reference]

# Precisions to compute for GPU fit columns
FIT_PRECISIONS = ["BF16", "FP8", "INT4"]

# Default serving overhead (KV cache, activations, framework overhead)
DEFAULT_OVERHEAD_PCT = 0.25

# Share of peak memory bandwidth that decoding one request achieves (MBU,
# model bandwidth utilization).  Dense: Databricks measured 55-60% at batch
# size 1, and RTX 4090 llama.cpp results work out to 57-70%.  MoE models read
# their active experts in many small pieces: single-GPU results for
# Qwen3-30B-A3B on an RTX 4090 work out to 25-40%.
DECODE_MBU = {"dense": 0.6, "moe": 0.3}


def vram_estimate_gb(total_params_b: float, precision: str) -> float:
    """
    Estimate VRAM in GB to load model weights.

    For MoE models, ALL parameters must be loaded regardless of active count.
    The active_params only affects compute, not memory.

    Args:
        total_params_b: Total parameters in billions
        precision: One of "BF16", "FP8", "INT4", "MXFP4"

    Returns:
        VRAM estimate in GB
    """
    bytes_per = PRECISION_BYTES.get(precision)
    if bytes_per is None:
        raise ValueError(f"Unknown precision: {precision}")
    return total_params_b * bytes_per


def practical_serving_gb(total_params_b: float, precision: str, overhead_pct: float = DEFAULT_OVERHEAD_PCT) -> float:
    """
    Estimate total VRAM needed for serving with moderate batch size.

    Adds overhead for KV cache, activations, and framework overhead.

    Args:
        total_params_b: Total parameters in billions
        precision: One of "BF16", "FP8", "INT4", "MXFP4"
        overhead_pct: Overhead fraction (default 0.25 = 25%)

    Returns:
        Practical serving VRAM estimate in GB
    """
    weight_gb = vram_estimate_gb(total_params_b, precision)
    return weight_gb * (1 + overhead_pct)


def decode_tokens_per_s(
    active_params_b: float, architecture: str, precision: str, bandwidth_gbs: float
) -> float:
    """
    Estimate single-request decode speed on one GPU, in tokens per second.

    Each new token reads every active weight from memory once, so with one
    request the speed is set by memory bandwidth, not compute:
    MBU x bandwidth / (active parameters x bytes per parameter).  Assumes a
    short context; reading a long context's KV cache lowers the speed.

    Args:
        active_params_b: Parameters used per token, in billions (all of them
            for a dense model)
        architecture: "dense" or "moe", which sets the MBU
        precision: One of "BF16", "FP8", "INT4", "MXFP4"
        bandwidth_gbs: Peak memory bandwidth of the GPU in GB/s

    Returns:
        Estimated tokens per second
    """
    gb_per_token = vram_estimate_gb(active_params_b, precision)
    return DECODE_MBU[architecture] * bandwidth_gbs / gb_per_token


def compute_vram_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add VRAM estimate columns to the DataFrame.

    Adds columns:
        vram_bf16_weights_gb, vram_fp8_weights_gb, vram_int4_weights_gb
        vram_bf16_serving_gb, vram_fp8_serving_gb, vram_int4_serving_gb

    Args:
        df: DataFrame with 'total_params_b' column

    Returns:
        DataFrame with VRAM columns added
    """
    for precision in FIT_PRECISIONS:
        p_lower = precision.lower()
        df[f"vram_{p_lower}_weights_gb"] = df["total_params_b"].apply(
            lambda x, p=precision: round(vram_estimate_gb(x, p), 1) if pd.notna(x) else None
        )
        df[f"vram_{p_lower}_serving_gb"] = df["total_params_b"].apply(
            lambda x, p=precision: round(practical_serving_gb(x, p), 1) if pd.notna(x) else None
        )

    return df


def compute_gpu_fit_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add GPU fit boolean columns to the DataFrame.

    For each GPU and precision, adds a column like 'fits_h100_bf16' (True/False).
    Uses the serving estimate (with overhead), not just weights.

    Args:
        df: DataFrame with vram_*_serving_gb columns

    Returns:
        DataFrame with GPU fit columns added
    """
    for gpu in GPUS:
        for precision in FIT_PRECISIONS:
            p_lower = precision.lower()
            serving_col = f"vram_{p_lower}_serving_gb"
            fit_col = f"fits_{gpu.key}_{p_lower}"
            df[fit_col] = df[serving_col].apply(
                lambda x, vram=gpu.vram_gb: bool(x <= vram) if pd.notna(x) else None
            )

    return df


def compute_best_gpu_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add summary columns for the smallest reference GPU that can serve each model.

    Adds columns: best_gpu_bf16, best_gpu_fp8

    Args:
        df: DataFrame with vram_*_serving_gb columns

    Returns:
        DataFrame with best_gpu columns added
    """
    for precision in ["BF16", "FP8"]:
        p_lower = precision.lower()
        serving_col = f"vram_{p_lower}_serving_gb"
        col_name = f"best_gpu_{p_lower}"

        df[col_name] = df[serving_col].apply(
            lambda x: _find_best_gpu(x) if pd.notna(x) else "UNKNOWN"
        )

    return df


def _find_best_gpu(serving_gb: float) -> str:
    """Find the smallest reference GPU that can fit the given serving VRAM requirement."""
    gpu = _fits_on_gpu(serving_gb)
    return gpu.name if gpu else "MULTI-GPU"


def _fits_on_gpu(serving_gb: float) -> Gpu | None:
    """The smallest reference GPU that fits, or None when no single one does."""
    for gpu in FITS_ON_GPUS:
        if serving_gb <= gpu.vram_gb:
            return gpu
    return None


def compute_speed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add estimated decode speed on the "Fits on" GPU (the smallest reference
    GPU that fits), or None when the model needs more than one GPU.

    Adds columns: est_tok_s_bf16, est_tok_s_fp8

    Args:
        df: DataFrame with vram_*_serving_gb, active_params_b and
            architecture columns

    Returns:
        DataFrame with est_tok_s columns added
    """
    for precision in ["BF16", "FP8"]:
        p_lower = precision.lower()
        serving_col = f"vram_{p_lower}_serving_gb"

        def speed(row: pd.Series, precision: str = precision, serving_col: str = serving_col) -> float | None:
            if pd.isna(row[serving_col]) or row.get("architecture") not in DECODE_MBU:
                return None
            gpu = _fits_on_gpu(row[serving_col])
            if gpu is None:
                return None
            return round(decode_tokens_per_s(
                row["active_params_b"], row["architecture"], precision, gpu.bandwidth_gbs
            ), 1)

        df[f"est_tok_s_{p_lower}"] = df.apply(speed, axis=1)

    return df


def node_label(gpu: Gpu) -> str:
    """Display name for an 8-GPU node, e.g. '8x H100 SXM'."""
    variant = f" {gpu.server_variant}" if gpu.server_variant else ""
    return f"{NODE_GPU_COUNT}x {gpu.name}{variant}"


def compute_best_node_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add summary columns for the smallest reference 8-GPU node that can serve each model.

    Adds columns: best_node_bf16, best_node_fp8

    Args:
        df: DataFrame with vram_*_serving_gb columns

    Returns:
        DataFrame with best_node columns added
    """
    for precision in ["BF16", "FP8"]:
        p_lower = precision.lower()
        serving_col = f"vram_{p_lower}_serving_gb"
        col_name = f"best_node_{p_lower}"

        df[col_name] = df[serving_col].apply(
            lambda x: _find_best_node(x) if pd.notna(x) else "UNKNOWN"
        )

    return df


def _find_best_node(serving_gb: float) -> str:
    """Find the smallest reference 8-GPU node that can fit the given serving VRAM requirement."""
    for gpu in FITS_ON_NODES:
        if serving_gb <= gpu.vram_gb * NODE_GPU_COUNT:
            return node_label(gpu)
    return "MULTI-NODE"


def add_all_vram_and_gpu_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all VRAM and GPU-related columns to the DataFrame.

    This is the main entry point for enriching a DataFrame with VRAM data.

    Args:
        df: DataFrame with 'total_params_b' column

    Returns:
        DataFrame with all VRAM and GPU columns added
    """
    df = compute_vram_columns(df)
    df = compute_gpu_fit_columns(df)
    df = compute_best_gpu_columns(df)
    df = compute_best_node_columns(df)
    df = compute_speed_columns(df)
    return df
