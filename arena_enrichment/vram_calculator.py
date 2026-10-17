"""
Memory and speed estimates, and which hardware can serve each model.

Memory, in GiB (the unit GPU and Mac memory is sold in): the weights (total
parameters × bytes per parameter of real checkpoints), plus the KV cache for
one long conversation, plus a fixed runtime overhead per GPU.  Speed: generation for one request, where each new
token reads the model's active weights from memory once, so the time per
token is those bytes over the usable bandwidth plus a fixed per-token cost.
The constants come from published measurements (measurements.py), and the
README's "Memory formula" and "Speed formula" sections explain them.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Bytes per parameter of real checkpoints, including quantization scales and
# the layers kept in 16-bit (embeddings, output head).  FP8 and GGUF Q8_0
# files of current models come to 1.00-1.07; 4-bit files (GGUF Q4_K_M,
# AWQ/GPTQ, NVFP4) to 0.56-0.61 for models of 30B and up
PRECISION_BYTES = {
    "BF16": 2.0,
    "FP8": 1.06,      # any 8-bit format: FP8, GGUF Q8_0, MLX 8-bit
    "INT4": 0.6,      # any 4-bit format: GGUF Q4_K_M, AWQ, GPTQ, NVFP4, MLX 4-bit
    "MXFP4": 0.56,    # gpt-oss style: MXFP4 experts, the rest in 16-bit
}

# The KV cache (attention keys and values kept for every token in context)
# is reserved for one conversation of this many tokens: a long prompt, or a
# reasoning model's thinking
CONTEXT_TOKENS = 16_384
# KV cache per token, in bytes, when a model's attention layout is unknown.
# 16-bit GQA models of 20-70B store 160-330 KiB per token, MLA, sliding-window
# and hybrid models far less, so 256 KiB errs on the safe side; dense models
# under 15B store about 128 KiB
KV_BYTES_PER_TOKEN = 256 * 1024
KV_BYTES_PER_TOKEN_SMALL = 128 * 1024
SMALL_MODEL_PARAMS_B = 15
# Runtime memory per GPU, GiB: CUDA or Metal context, framework, activations
RUNTIME_GIB = 1.5
# Memory is counted in GiB, the unit hardware memory is sold in although it
# is labelled "GB": an "80 GB" H100 holds 80 GiB
BYTES_PER_GIB = 2**30

# Generation speed for one request: the share of peak bandwidth spent reading
# weights, the extra bytes MoE models read (attention and shared layers stay
# at higher precision), the fixed cost of each token (kernel launches,
# synchronisation, sampling), and the extra cost per token when 8 GPUs split
# a request (an all-reduce at every layer).  Fitted on measurements.py
BANDWIDTH_EFFICIENCY = 0.8
MOE_BYTES_FACTOR = 1.15
FIXED_MS_PER_TOKEN = {
    ("gpu", "dense"): 1.0,
    ("gpu", "moe"): 2.0,
    ("apple", "dense"): 1.0,
    ("apple", "moe"): 5.0,
}
SYNC_MS_PER_TOKEN = {"dense": 3.0, "moe": 6.0}

# Precisions to compute for fit columns
FIT_PRECISIONS = ["BF16", "FP8", "INT4"]

# GPUs in a standard server node (HGX/DGX-style), the usual way to serve
# models too large for one GPU: weights are split across the node's GPUs
NODE_GPU_COUNT = 8

# A model fits when it needs at most this share of the usable memory: the
# memory a GPU reports is a little below its label (an L40S with ECC on
# shows 45 of 48 GiB), and engines want headroom (vLLM uses 90% by default
# and runs these fits at 0.95)
FIT_HEADROOM = 0.95

TIERS = ["Consumer GPU", "Unified memory", "Workstation", "Data center GPU"]


@dataclass(frozen=True)
class Hardware:
    """A GPU, or a computer whose GPU shares system memory, that the tables report on."""

    key: str                    # column id, as in fits_<key>_<precision>
    name: str
    tier: str                   # one of TIERS
    memory_gb: float            # as the vendor states it ("80 GB", which is 80 GiB)
    bandwidth_gbs: float        # peak memory bandwidth, GB/s
    # The part of memory_gb a model can use, when the OS or the board keeps
    # some: macOS gives the GPU 75% of RAM by default
    usable_memory_gb: float | None = None
    platform: str = "gpu"       # "gpu" or "apple": sets the fixed cost per token
    native_fp8: bool = True
    # Sold in 8-GPU servers.  GeForce cards are not (NVIDIA's GeForce driver
    # license bars data-center deployment), nor are workstation cards, which
    # are built for desktops with up to four cards
    in_8gpu_servers: bool = False
    # Its name and bandwidth in an 8-GPU server, when they differ (the RTX
    # PRO 6000 Server Edition runs its memory at 1,597 GB/s)
    server_name: str = ""
    server_bandwidth_gbs: float | None = None
    # The 8-GPU board links every GPU to every other (NVLink switch, AMD
    # Infinity Fabric, Gaudi's on-board Ethernet).  Single-request speed is
    # measured on such boards; PCIe servers have no data to go on
    gpu_fabric: bool = False
    # The most widely deployed GPU at its memory size.  The leaderboard's
    # "Fits on" names the smallest reference GPU (or 8-GPU server) that fits,
    # so it reads "8x H200" rather than a rarer part that happens to be smaller
    reference: bool = False

    @property
    def usable_gib(self) -> float:
        """Memory a model can use, in GiB."""
        return self.usable_memory_gb if self.usable_memory_gb is not None else self.memory_gb

    def bandwidth_for(self, gpu_count: int) -> float:
        """One GPU's peak bandwidth, GB/s, alone or in an 8-GPU server."""
        if gpu_count > 1 and self.server_bandwidth_gbs is not None:
            return self.server_bandwidth_gbs
        return self.bandwidth_gbs


def _mac(key: str, name: str, ram_gb: int, bandwidth_gbs: float) -> Hardware:
    """An Apple Silicon Mac: macOS lets the GPU use 75% of RAM unless raised."""
    return Hardware(key, name, "Unified memory", ram_gb, bandwidth_gbs,
                    usable_memory_gb=ram_gb * 0.75, platform="apple", native_fp8=False)


# Every piece of hardware in the table, in display order: by tier, then
# usable memory.  Ties in memory keep this order when picking the smallest
# GPU that fits.
#
# Checked 2026-10 against vendor pages and datasheets, memory and bandwidth
# both.  B200, B300 and Rubin list what an 8-GPU HGX board gives each GPU
# (B200 180 of 192 GB, B300 about 268 of 288 GB; Rubin about 250 GB and
# 16 TB/s from NVIDIA's US HGX page of 2026-10-15, while other NVIDIA pages
# list 288 GB and 22 TB/s).  DGX Station counts its 252 GB of GPU memory;
# its 496 GB of CPU memory runs at 396 GB/s.  Left out: rack-scale-only parts
# (GB200/GB300 NVL72, MI455X in AMD Helios), which come as 72-GPU racks, and
# AMD's MI440X 8-GPU platform, whose memory is not yet published.
HARDWARE: list[Hardware] = [
    Hardware("rtx_5060_ti", "RTX 5060 Ti 16 GB", "Consumer GPU", 16, 448),
    Hardware("rtx_5080", "RTX 5080", "Consumer GPU", 16, 960),
    Hardware("rtx_3090", "RTX 3090", "Consumer GPU", 24, 936, native_fp8=False),
    Hardware("rtx_4090", "RTX 4090", "Consumer GPU", 24, 1008, reference=True),
    Hardware("rtx_5090", "RTX 5090", "Consumer GPU", 32, 1792, reference=True),
    _mac("m5_pro_64", "Mac mini / MacBook Pro, M5 Pro 64 GB", 64, 307),
    Hardware("ryzen_ai_max", "Ryzen AI Max+ 395, 128 GB", "Unified memory", 128, 256,
             usable_memory_gb=96, native_fp8=False),
    _mac("m5_max_128", "MacBook Pro / Mac Studio, M5 Max 128 GB", 128, 614),
    Hardware("dgx_spark", "DGX Spark (GB10), 128 GB", "Unified memory", 128, 273,
             usable_memory_gb=115),
    _mac("m5_ultra_256", "Mac Studio, M5 Ultra 256 GB", 256, 1229),
    _mac("m5_ultra_512", "Mac Studio, M5 Ultra 512 GB", 512, 1229),
    Hardware("r9700", "Radeon AI PRO R9700", "Workstation", 32, 640),
    Hardware("rtx_pro_6000", "RTX PRO 6000 Blackwell", "Workstation", 96, 1792,
             in_8gpu_servers=True, server_name="RTX PRO 6000 Blackwell Server Edition",
             server_bandwidth_gbs=1597, reference=True),
    Hardware("dgx_station", "DGX Station (GB300)", "Workstation", 252, 7100),
    Hardware("l40s", "L40S", "Data center GPU", 48, 864, in_8gpu_servers=True, reference=True),
    Hardware("h100", "H100 SXM", "Data center GPU", 80, 3350,
             in_8gpu_servers=True, gpu_fabric=True, reference=True),
    Hardware("a100", "A100 80GB SXM", "Data center GPU", 80, 2039, native_fp8=False,
             in_8gpu_servers=True, gpu_fabric=True),
    Hardware("h100_nvl", "H100 NVL", "Data center GPU", 94, 3900, in_8gpu_servers=True),
    Hardware("gaudi3", "Gaudi 3", "Data center GPU", 128, 3700,
             in_8gpu_servers=True, gpu_fabric=True),
    Hardware("h200", "H200 SXM / NVL", "Data center GPU", 141, 4800,
             in_8gpu_servers=True, server_name="H200 SXM", gpu_fabric=True, reference=True),
    Hardware("b200", "B200", "Data center GPU", 192, 8000, usable_memory_gb=180,
             in_8gpu_servers=True, gpu_fabric=True, reference=True),
    Hardware("mi300x", "MI300X", "Data center GPU", 192, 5300,
             in_8gpu_servers=True, gpu_fabric=True),
    Hardware("rubin", "Rubin", "Data center GPU", 288, 16250, usable_memory_gb=250,
             in_8gpu_servers=True, gpu_fabric=True),
    Hardware("mi325x", "MI325X", "Data center GPU", 256, 6000,
             in_8gpu_servers=True, gpu_fabric=True),
    Hardware("b300", "B300", "Data center GPU", 288, 8000, usable_memory_gb=268,
             in_8gpu_servers=True, gpu_fabric=True, reference=True),
    Hardware("mi355x", "MI350X / MI355X", "Data center GPU", 288, 8000,
             in_8gpu_servers=True, gpu_fabric=True),
]

# Hardware sorted by usable memory ascending (for finding the smallest that fits)
HARDWARE_BY_MEMORY = sorted(HARDWARE, key=lambda h: h.usable_gib)

# GPUs sold in 8-GPU servers, sorted by memory ascending
NODE_GPUS = [h for h in HARDWARE_BY_MEMORY if h.in_8gpu_servers]

# The reference GPUs and 8-GPU servers that "Fits on" chooses from, smallest first
FITS_ON_GPUS = [h for h in HARDWARE_BY_MEMORY if h.reference]
FITS_ON_NODES = [h for h in NODE_GPUS if h.reference]


def weights_gib(total_params_b: float, precision: str) -> float:
    """
    Memory for the weights, in GiB.

    MoE models count ALL parameters: every expert must sit in memory even
    though each token uses only a few.

    Args:
        total_params_b: Total parameters in billions
        precision: One of "BF16", "FP8", "INT4", "MXFP4"

    Returns:
        Weight memory in GiB
    """
    bytes_per = PRECISION_BYTES.get(precision)
    if bytes_per is None:
        raise ValueError(f"Unknown precision: {precision}")
    return total_params_b * 1e9 * bytes_per / BYTES_PER_GIB


def kv_cache_gib(total_params_b: float) -> float:
    """KV cache for one conversation of CONTEXT_TOKENS tokens, in GiB."""
    per_token = KV_BYTES_PER_TOKEN_SMALL if total_params_b < SMALL_MODEL_PARAMS_B else KV_BYTES_PER_TOKEN
    return CONTEXT_TOKENS * per_token / BYTES_PER_GIB


def serving_gib(total_params_b: float, precision: str, gpu_count: int = 1) -> float:
    """
    Memory needed to serve a model: weights + KV cache + runtime per GPU, in GiB.

    Args:
        total_params_b: Total parameters in billions
        precision: One of "BF16", "FP8", "INT4", "MXFP4"
        gpu_count: GPUs the model is split across (each needs its runtime)

    Returns:
        Serving memory in GiB
    """
    return weights_gib(total_params_b, precision) + kv_cache_gib(total_params_b) + RUNTIME_GIB * gpu_count


def node_serving_gib(single_serving_gib: float) -> float:
    """The memory an 8-GPU server needs, from the single-GPU figure: 7 more runtimes."""
    return single_serving_gib + RUNTIME_GIB * (NODE_GPU_COUNT - 1)


def needed_gib(single_serving_gib: float, gpu_count: int = 1) -> float:
    """The memory a model needs on one GPU or an 8-GPU server, from the single-GPU figure."""
    return single_serving_gib if gpu_count == 1 else node_serving_gib(single_serving_gib)


def fits(hardware: Hardware, single_serving_gib: float, gpu_count: int = 1) -> bool:
    """Whether a model fits one device or an 8-GPU server of it, with FIT_HEADROOM."""
    return (needed_gib(single_serving_gib, gpu_count)
            <= hardware.usable_gib * gpu_count * FIT_HEADROOM)


def decode_tokens_per_s(
    active_params_b: float,
    architecture: str,
    precision: str,
    bandwidth_gbs: float,
    platform: str = "gpu",
    gpu_count: int = 1,
) -> float:
    """
    Estimate generation speed for one request, in tokens per second.

    Each new token reads every active weight from memory once, so:
    time per token = bytes per token / (efficiency × bandwidth × GPUs)
                     + fixed cost per token (+ sync cost across GPUs).
    Bytes and bandwidth are decimal (GB, GB/s), as bandwidth is quoted.
    Assumes a short context and no speculative decoding.

    Args:
        active_params_b: Parameters used per token, in billions (all of them
            for a dense model)
        architecture: "dense" or "moe"
        precision: One of "BF16", "FP8", "INT4", "MXFP4"
        bandwidth_gbs: Peak memory bandwidth of one GPU, GB/s
        platform: "gpu" or "apple"
        gpu_count: GPUs splitting the request (tensor parallel)

    Returns:
        Estimated tokens per second
    """
    gb_per_token = active_params_b * PRECISION_BYTES[precision]
    if architecture == "moe":
        gb_per_token *= MOE_BYTES_FACTOR
    ms = gb_per_token / (BANDWIDTH_EFFICIENCY * bandwidth_gbs * gpu_count) * 1000
    ms += FIXED_MS_PER_TOKEN[(platform, architecture)]
    if gpu_count > 1:
        ms += SYNC_MS_PER_TOKEN[architecture]
    return 1000 / ms


def hardware_speed(
    hardware: Hardware, model: pd.Series, precision: str, gpu_count: int = 1
) -> float | None:
    """
    Estimated generation speed of a model on a piece of hardware, or None:
    unknown architecture, or 8 GPUs without an all-to-all GPU fabric.
    """
    if model.get("architecture") not in ("dense", "moe") or pd.isna(model.get("active_params_b")):
        return None
    if gpu_count > 1 and not hardware.gpu_fabric:
        return None
    return decode_tokens_per_s(
        model["active_params_b"], model["architecture"], precision,
        hardware.bandwidth_for(gpu_count), hardware.platform, gpu_count,
    )


def compute_vram_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add memory estimate columns to the DataFrame, in GiB.

    Adds columns:
        vram_bf16_weights_gib, vram_fp8_weights_gib, vram_int4_weights_gib
        vram_bf16_serving_gib, vram_fp8_serving_gib, vram_int4_serving_gib
        (serving: weights + KV cache + runtime, on one GPU)

    Args:
        df: DataFrame with 'total_params_b' column

    Returns:
        DataFrame with memory columns added
    """
    for precision in FIT_PRECISIONS:
        p_lower = precision.lower()
        df[f"vram_{p_lower}_weights_gib"] = df["total_params_b"].apply(
            lambda x, p=precision: round(weights_gib(x, p), 1) if pd.notna(x) else None
        )
        df[f"vram_{p_lower}_serving_gib"] = df["total_params_b"].apply(
            lambda x, p=precision: round(serving_gib(x, p), 1) if pd.notna(x) else None
        )

    return df


def compute_gpu_fit_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add fit boolean columns to the DataFrame.

    For each piece of hardware and precision, adds a column like
    'fits_h100_bf16' (True/False): serving memory within usable memory.

    Args:
        df: DataFrame with vram_*_serving_gib columns

    Returns:
        DataFrame with fit columns added
    """
    for hardware in HARDWARE:
        for precision in FIT_PRECISIONS:
            p_lower = precision.lower()
            serving_col = f"vram_{p_lower}_serving_gib"
            fit_col = f"fits_{hardware.key}_{p_lower}"
            df[fit_col] = df[serving_col].apply(
                lambda x, hw=hardware: fits(hw, x) if pd.notna(x) else None
            )

    return df


def compute_best_gpu_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add summary columns for the smallest reference GPU that can serve each model.

    Adds columns: best_gpu_bf16, best_gpu_fp8

    Args:
        df: DataFrame with vram_*_serving_gib columns

    Returns:
        DataFrame with best_gpu columns added
    """
    for precision in ["BF16", "FP8"]:
        p_lower = precision.lower()
        serving_col = f"vram_{p_lower}_serving_gib"
        col_name = f"best_gpu_{p_lower}"

        df[col_name] = df[serving_col].apply(
            lambda x: _find_best_gpu(x) if pd.notna(x) else "UNKNOWN"
        )

    return df


def _find_best_gpu(serving: float) -> str:
    """Find the smallest reference GPU that can fit the given serving memory."""
    gpu = _fits_on_gpu(serving)
    return gpu.name if gpu else "MULTI-GPU"


def _fits_on_gpu(serving: float) -> Hardware | None:
    """The smallest reference GPU that fits, or None when no single one does."""
    for gpu in FITS_ON_GPUS:
        if fits(gpu, serving):
            return gpu
    return None


def node_label(gpu: Hardware) -> str:
    """Display name for an 8-GPU node, e.g. '8x H100 SXM'."""
    return f"{NODE_GPU_COUNT}x {gpu.server_name or gpu.name}"


def compute_best_node_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add summary columns for the smallest reference 8-GPU node that can serve each model.

    Adds columns: best_node_bf16, best_node_fp8

    Args:
        df: DataFrame with vram_*_serving_gib columns

    Returns:
        DataFrame with best_node columns added
    """
    for precision in ["BF16", "FP8"]:
        p_lower = precision.lower()
        serving_col = f"vram_{p_lower}_serving_gib"
        col_name = f"best_node_{p_lower}"

        df[col_name] = df[serving_col].apply(
            lambda x: _find_best_node(x) if pd.notna(x) else "UNKNOWN"
        )

    return df


def _find_best_node(serving: float) -> str:
    """Find the smallest reference 8-GPU node that fits, from the single-GPU serving memory."""
    gpu = _fits_on_node(serving)
    return node_label(gpu) if gpu else "MULTI-NODE"


def _fits_on_node(serving: float) -> Hardware | None:
    """The smallest reference 8-GPU node that fits, or None."""
    for gpu in FITS_ON_NODES:
        if fits(gpu, serving, NODE_GPU_COUNT):
            return gpu
    return None


def compute_speed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add estimated generation speed on the "Fits on" hardware: the smallest
    reference GPU that fits, else the smallest reference 8-GPU server (None
    for PCIe servers, which have no single-request data).

    Adds columns: est_tok_s_bf16, est_tok_s_fp8

    Args:
        df: DataFrame with vram_*_serving_gib, active_params_b and
            architecture columns

    Returns:
        DataFrame with est_tok_s columns added
    """
    for precision in ["BF16", "FP8"]:
        p_lower = precision.lower()
        serving_col = f"vram_{p_lower}_serving_gib"

        def speed(row: pd.Series, precision: str = precision, serving_col: str = serving_col) -> float | None:
            if pd.isna(row[serving_col]):
                return None
            gpu = _fits_on_gpu(row[serving_col])
            if gpu is not None:
                estimate = hardware_speed(gpu, row, precision)
            else:
                node = _fits_on_node(row[serving_col])
                estimate = hardware_speed(node, row, precision, NODE_GPU_COUNT) if node else None
            return round(estimate, 1) if estimate is not None else None

        df[f"est_tok_s_{p_lower}"] = df.apply(speed, axis=1)

    return df


def add_all_vram_and_gpu_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all memory, fit and speed columns to the DataFrame.

    This is the main entry point for enriching a DataFrame with VRAM data.

    Args:
        df: DataFrame with 'total_params_b' column

    Returns:
        DataFrame with all memory, fit and speed columns added
    """
    df = compute_vram_columns(df)
    df = compute_gpu_fit_columns(df)
    df = compute_best_gpu_columns(df)
    df = compute_best_node_columns(df)
    df = compute_speed_columns(df)
    return df
