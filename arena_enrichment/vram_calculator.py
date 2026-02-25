"""
VRAM estimation and GPU fit calculations.

Computes estimated VRAM requirements at multiple precisions and
determines which single-GPU configurations can serve each model.
"""

import pandas as pd

# Bytes per parameter for each precision
PRECISION_BYTES = {
    "BF16": 2.0,      # 16-bit = 2 bytes
    "FP8": 1.0,       # 8-bit = 1 byte
    "INT4": 0.5,      # 4-bit = 0.5 bytes
    "MXFP4": 0.53,    # ~4.25 bits (used by GPT-OSS natively)
}

# GPU configurations: name -> VRAM in GB
GPU_CONFIGS = {
    "RTX_PRO_6000": 96,
    "H100": 80,
    "H200": 141,
    "B200": 180,
    "B300": 288,
}

# Display names for GPUs
GPU_DISPLAY_NAMES = {
    "RTX_PRO_6000": "RTX PRO 6000",
    "H100": "H100 SXM",
    "H200": "H200 SXM",
    "B200": "B200 SXM",
    "B300": "B300 SXM",
}

# GPUs sorted by VRAM ascending (for finding smallest fitting GPU)
GPUS_BY_VRAM = sorted(GPU_CONFIGS.items(), key=lambda x: x[1])

# Precisions to compute for GPU fit columns
FIT_PRECISIONS = ["BF16", "FP8", "INT4"]

# Default serving overhead (KV cache, activations, framework overhead)
DEFAULT_OVERHEAD_PCT = 0.25


def vram_estimate_gb(total_params_b, precision):
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


def practical_serving_gb(total_params_b, precision, overhead_pct=DEFAULT_OVERHEAD_PCT):
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


def compute_vram_columns(df):
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
            lambda x: round(vram_estimate_gb(x, precision), 1) if pd.notna(x) else None
        )
        df[f"vram_{p_lower}_serving_gb"] = df["total_params_b"].apply(
            lambda x: round(practical_serving_gb(x, precision), 1) if pd.notna(x) else None
        )

    return df


def compute_gpu_fit_columns(df):
    """
    Add GPU fit boolean columns to the DataFrame.

    For each GPU and precision, adds a column like 'fits_h100_bf16' (True/False).
    Uses the serving estimate (with overhead), not just weights.

    Args:
        df: DataFrame with vram_*_serving_gb columns

    Returns:
        DataFrame with GPU fit columns added
    """
    for gpu_key, gpu_vram in GPU_CONFIGS.items():
        gpu_lower = gpu_key.lower()
        for precision in FIT_PRECISIONS:
            p_lower = precision.lower()
            serving_col = f"vram_{p_lower}_serving_gb"
            fit_col = f"fits_{gpu_lower}_{p_lower}"
            df[fit_col] = df[serving_col].apply(
                lambda x: bool(x <= gpu_vram) if pd.notna(x) else None
            )

    return df


def compute_best_gpu_columns(df):
    """
    Add summary columns for the smallest GPU that can serve each model.

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


def _find_best_gpu(serving_gb):
    """Find the smallest GPU that can fit the given serving VRAM requirement."""
    for gpu_key, gpu_vram in GPUS_BY_VRAM:
        if serving_gb <= gpu_vram:
            return GPU_DISPLAY_NAMES[gpu_key]
    return "MULTI-GPU"


def add_all_vram_and_gpu_columns(df):
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
    return df
