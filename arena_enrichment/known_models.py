"""
Model parameter overrides — corrections for when Artificial Analysis data
is wrong, missing, or when name-based parsing gives incorrect results.

This dict is intentionally small.  The primary source of truth is the
Artificial Analysis model database (cached locally).  Only add entries here
when:
  1. AA has wrong parameter counts for a model
  2. AA doesn't list the model at all and name parsing can't resolve it
  3. The model name is misleading (e.g., per-expert size instead of total)
"""

# Model name fragment -> (total_params_B, active_params_B, architecture)
MODEL_OVERRIDES = {
    # === Models with misleading names ===
    "GPT-OSS-120B": (117, 5.1, "moe"),    # Name says 120B, actual is 117B
    "GPT-OSS-20B": (21, 3.6, "moe"),      # Name says 20B, actual is 21B

    # === MoE models where name contains per-expert size, not total ===
    # Llama-4: arena names contain "17b" (per-expert size), not total params.
    "Llama-4-Maverick": (400, 17, "moe"),
    "llama-4-maverick-17b-128e-instruct": (400, 17, "moe"),
    "Llama-4-Scout": (109, 17, "moe"),
    "llama-4-scout-17b-16e-instruct": (109, 17, "moe"),

    # === Models not in AA and no size in name ===
    "Kimi-K2.5": (1000, 32, "moe"),
    "Kimi-K2": (1000, 32, "moe"),
    "LongCat-Flash": (560, 27, "moe"),

    # === AA maps to wrong variant or wrong data ===
    "Phi-4": (14, 14, "dense"),             # AA has multimodal 5.6B, arena means text 14B
    "phi-4": (14, 14, "dense"),
    "deepseek-llm-67b-chat": (67, 67, "dense"),  # AA incorrectly lists as 7B

    # === Older models not in AA, no parseable size in name ===
    "phi-3-medium": (14, 14, "dense"),
    "phi-3-small": (7.4, 7.4, "dense"),
    "phi-3-mini": (3.8, 3.8, "dense"),
    "athene-v2": (72, 72, "dense"),            # Fine-tune of Qwen2.5-72B
    "trinity-large": (70, 70, "dense"),        # InfiniAI fine-tune of Llama-3.1-70B
    "command-r-plus-08-2024": (104, 104, "dense"),  # Cohere Command R+ Aug 2024
    "command-r-08-2024": (35, 35, "dense"),    # Cohere Command R Aug 2024
    "ibm-granite-h-small": (8, 8, "dense"),    # IBM Granite 3.2 8B variant
}
