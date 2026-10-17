"""
Published single-request generation speeds, used to fit and check the speed
formula in vram_calculator.py.

Each entry describes the model the way the estimate sees it: active
parameters, dense or MoE, and the precision class of its weights (BF16,
8-bit as FP8, 4-bit as INT4).  Collected 2026-10 from the sources given, and
checked against them.

Only runs with a known engine and current software are kept: llama-bench's
generation figure (tg128 or tg32), or an engine benchmark at batch size 1
without speculative decoding.  Left out: the 2023 llama.cpp build behind
the older rows of the Apple discussion (the same Macs run 20-33% faster on
current builds), and LM Studio figures that do not say how they were timed
(an M5 Max running a 70B model at half the bandwidth llama-bench reaches).
"""

from __future__ import annotations

from dataclasses import dataclass

LLAMA_CPP_GPUS = "https://github.com/ggml-org/llama.cpp/discussions/15013"
LLAMA_CPP_ROCM = "https://github.com/ggml-org/llama.cpp/discussions/15021"
LLAMA_CPP_APPLE = "https://github.com/ggml-org/llama.cpp/discussions/4167"
LLAMA_CPP_VULKAN = "https://github.com/ggml-org/llama.cpp/discussions/10879"
LLAMA_CPP_GPT_OSS = "https://github.com/ggml-org/llama.cpp/discussions/15396"
LLAMA_CPP_GPT_OSS_120B = "https://github.com/ggml-org/llama.cpp/discussions/18005"
LLAMA_CPP_SPARK = "https://github.com/ggml-org/llama.cpp/blob/master/benches/dgx-spark/dgx-spark.md"
LMSYS_SPARK = "https://lmsys.org/blog/2025-10-13-nvidia-dgx-spark/"
STRIX_HALO = "https://github.com/lhl/strix-halo-testing"
STRIX_HALO_FORUM = "https://community.frame.work/t/amd-strix-halo-ryzen-ai-max-395-gpu-llm-performance-tests/72521"
QWEN_SPEED = "https://qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html"
TRT_LLM_H200 = ("https://developer.nvidia.com/blog/"
                "boost-llama-3-3-70b-inference-throughput-3x-with-nvidia-tensorrt-llm-speculative-decoding")
TRT_LLM_H200_TP8 = ("https://developer.nvidia.com/blog/low-latency-inference-chapter-1-up-to-1-9x-higher-"
                    "llama-3-1-performance-with-medusa-on-nvidia-hgx-h200-with-nvlink-switch/")
TRT_LLM_B200 = ("https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/"
                "blog01_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.html")
GPT_FAST = "https://github.com/meta-pytorch/gpt-fast"
SGLANG_H200 = "https://verda.com/blog/deploy-deepseek-r1-on-8x-nvidia-h200"


@dataclass(frozen=True)
class Measurement:
    """One published generation speed for a single request."""

    hardware: str
    model: str
    active_params_b: float
    architecture: str       # "dense" or "moe"
    precision: str          # "BF16", "FP8" (any 8-bit) or "INT4" (any 4-bit)
    bandwidth_gbs: float    # one GPU's peak memory bandwidth
    platform: str           # "gpu" or "apple"
    gpu_count: int
    tokens_per_s: float
    engine: str
    source: str
    in_readme: bool = False


MEASUREMENTS: list[Measurement] = [
    # Discrete GPUs, llama.cpp, Llama-2-7B Q4_0
    Measurement("RTX 4090", "Llama-2-7B Q4_0", 6.74, "dense", "INT4", 1008, "gpu", 1, 188.96,
                "llama.cpp", LLAMA_CPP_GPUS, in_readme=True),
    Measurement("RTX 5090", "Llama-2-7B Q4_0", 6.74, "dense", "INT4", 1792, "gpu", 1, 300.40,
                "llama.cpp", LLAMA_CPP_GPUS),
    Measurement("RTX PRO 6000", "Llama-2-7B Q4_0", 6.74, "dense", "INT4", 1792, "gpu", 1, 281.11,
                "llama.cpp", LLAMA_CPP_GPUS),
    Measurement("A100 80GB", "Llama-2-7B Q4_0", 6.74, "dense", "INT4", 2039, "gpu", 1, 200.90,
                "llama.cpp", LLAMA_CPP_GPUS),
    Measurement("H100 SXM", "Llama-2-7B Q4_0", 6.74, "dense", "INT4", 3350, "gpu", 1, 280.74,
                "llama.cpp", LLAMA_CPP_GPUS, in_readme=True),
    Measurement("MI300X", "Llama-2-7B Q4_0", 6.74, "dense", "INT4", 5300, "gpu", 1, 232.92,
                "llama.cpp", LLAMA_CPP_ROCM),
    # Data-center engines
    Measurement("H200", "Llama-3.3-70B FP8", 70.6, "dense", "FP8", 4800, "gpu", 1, 51.14,
                "TensorRT-LLM", TRT_LLM_H200, in_readme=True),
    Measurement("A100 80GB", "Qwen2.5-7B BF16", 7.6, "dense", "BF16", 2039, "gpu", 1, 84.28,
                "vLLM", QWEN_SPEED),
    Measurement("A100 80GB", "Qwen2.5-32B BF16", 32.5, "dense", "BF16", 2039, "gpu", 1, 22.13,
                "vLLM", QWEN_SPEED, in_readme=True),
    Measurement("A100 80GB", "Qwen2.5-7B GPTQ-Int4", 7.6, "dense", "INT4", 2039, "gpu", 1, 154.05,
                "vLLM", QWEN_SPEED),
    Measurement("A100 80GB", "Qwen2.5-32B GPTQ-Int4", 32.5, "dense", "INT4", 2039, "gpu", 1, 55.83,
                "vLLM", QWEN_SPEED),
    # Apple Silicon, llama.cpp (current builds)
    Measurement("M5 Max", "Llama-2-7B F16", 6.74, "dense", "BF16", 614, "apple", 1, 37.11,
                "llama.cpp", LLAMA_CPP_APPLE),
    Measurement("M5 Max", "Llama-2-7B Q8_0", 6.74, "dense", "FP8", 614, "apple", 1, 72.42,
                "llama.cpp", LLAMA_CPP_APPLE, in_readme=True),
    Measurement("M5 Max", "Llama-2-7B Q4_0", 6.74, "dense", "INT4", 614, "apple", 1, 119.92,
                "llama.cpp", LLAMA_CPP_APPLE, in_readme=True),
    Measurement("M5 Ultra", "Llama-2-7B F16", 6.74, "dense", "BF16", 1229, "apple", 1, 73.60,
                "llama.cpp", LLAMA_CPP_APPLE, in_readme=True),
    Measurement("M5 Ultra", "Llama-2-7B Q4_0", 6.74, "dense", "INT4", 1229, "apple", 1, 179.10,
                "llama.cpp", LLAMA_CPP_APPLE),
    # DGX Spark and Ryzen AI Max+ 395
    Measurement("DGX Spark", "Qwen2.5-Coder-7B Q8_0", 7.62, "dense", "FP8", 273, "gpu", 1, 29.43,
                "llama.cpp", LLAMA_CPP_SPARK),
    Measurement("DGX Spark", "Llama-2-7B Q4_0", 6.74, "dense", "INT4", 273, "gpu", 1, 60.39,
                "llama.cpp (Vulkan)", LLAMA_CPP_VULKAN),
    Measurement("DGX Spark", "Llama-3.1-70B FP8", 70.6, "dense", "FP8", 273, "gpu", 1, 2.7,
                "SGLang", LMSYS_SPARK, in_readme=True),
    Measurement("Ryzen AI Max+ 395", "Llama-2-7B Q4_0", 6.74, "dense", "INT4", 256, "gpu", 1, 54.38,
                "llama.cpp (Vulkan)", LLAMA_CPP_VULKAN),
    Measurement("Ryzen AI Max+ 395", "70B Q4_K_M", 70.6, "dense", "INT4", 256, "gpu", 1, 5.0,
                "llama.cpp (Vulkan)", STRIX_HALO_FORUM, in_readme=True),
    Measurement("Ryzen AI Max+ 395", "Mistral-Small-24B Q4_K_XL", 23.6, "dense", "INT4", 256, "gpu", 1,
                14.3, "llama.cpp (Vulkan)", STRIX_HALO_FORUM),
    Measurement("Ryzen AI Max+ 395", "Qwen3-32B Q8_0", 32.8, "dense", "FP8", 256, "gpu", 1, 6.4,
                "llama.cpp (Vulkan)", STRIX_HALO_FORUM),
    # MoE
    Measurement("RTX 4090", "gpt-oss-20b MXFP4", 3.6, "moe", "INT4", 1008, "gpu", 1, 221.95,
                "llama.cpp", LLAMA_CPP_GPT_OSS),
    Measurement("RTX 5090", "gpt-oss-20b MXFP4", 3.6, "moe", "INT4", 1792, "gpu", 1, 282.51,
                "llama.cpp", LLAMA_CPP_GPT_OSS, in_readme=True),
    Measurement("DGX Spark", "gpt-oss-20b MXFP4", 3.6, "moe", "INT4", 273, "gpu", 1, 83.43,
                "llama.cpp", LLAMA_CPP_SPARK),
    Measurement("M3 Ultra", "gpt-oss-20b MXFP4", 3.6, "moe", "INT4", 819, "apple", 1, 115.52,
                "llama.cpp", LLAMA_CPP_GPT_OSS, in_readme=True),
    Measurement("RTX PRO 6000", "gpt-oss-120b MXFP4", 5.1, "moe", "INT4", 1792, "gpu", 1, 196.31,
                "llama.cpp", LLAMA_CPP_GPT_OSS),
    Measurement("DGX Spark", "gpt-oss-120b MXFP4", 5.1, "moe", "INT4", 273, "gpu", 1, 58.72,
                "llama.cpp", LLAMA_CPP_SPARK, in_readme=True),
    Measurement("GH200", "gpt-oss-120b MXFP4", 5.1, "moe", "INT4", 4900, "gpu", 1, 209.01,
                "llama.cpp", LLAMA_CPP_GPT_OSS_120B),
    Measurement("M2 Ultra", "gpt-oss-120b MXFP4", 5.1, "moe", "INT4", 800, "apple", 1, 79.68,
                "llama.cpp", LLAMA_CPP_GPT_OSS),
    Measurement("Ryzen AI Max+ 395", "gpt-oss-120b MXFP4", 5.1, "moe", "INT4", 256, "gpu", 1, 50.05,
                "llama.cpp (Vulkan)", STRIX_HALO, in_readme=True),
    Measurement("DGX Spark", "Qwen3-Coder-30B-A3B Q8_0", 3.3, "moe", "FP8", 273, "gpu", 1, 61.06,
                "llama.cpp", LLAMA_CPP_SPARK),
    Measurement("Ryzen AI Max+ 395", "Qwen3-30B-A3B Q4_K_XL", 3.3, "moe", "INT4", 256, "gpu", 1, 72.0,
                "llama.cpp (Vulkan)", STRIX_HALO_FORUM),
    # 3.9B active counting embeddings and output layer, as Qwen3-30B-A3B's 3.3B does
    Measurement("DGX Spark", "GLM-4.7-Flash Q8_0", 3.9, "moe", "FP8", 273, "gpu", 1, 48.68,
                "llama.cpp", LLAMA_CPP_SPARK),
    # 8-GPU servers, one request split across the GPUs
    Measurement("8x A100", "Llama-2-70B BF16", 69.0, "dense", "BF16", 2039, "gpu", 8, 62.5,
                "gpt-fast", GPT_FAST),
    Measurement("8x H200", "Llama-3.1-70B FP8", 70.6, "dense", "FP8", 4800, "gpu", 8, 184.0,
                "TensorRT-LLM", TRT_LLM_H200_TP8),
    Measurement("8x H200", "Llama-3.1-405B FP8", 405.0, "dense", "FP8", 4800, "gpu", 8, 56.0,
                "TensorRT-LLM", TRT_LLM_H200_TP8, in_readme=True),
    Measurement("8x B200", "DeepSeek-R1 FP4", 37.0, "moe", "INT4", 8000, "gpu", 8, 111.0,
                "TensorRT-LLM, no MTP", TRT_LLM_B200, in_readme=True),
    Measurement("8x H200", "DeepSeek-R1 FP8", 37.0, "moe", "FP8", 4800, "gpu", 8, 47.7,
                "SGLang (Jan 2025)", SGLANG_H200, in_readme=True),
]
