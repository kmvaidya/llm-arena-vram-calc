"""
Model parameter overrides — corrections for when Artificial Analysis data
is wrong, missing, or when name-based parsing gives incorrect results.

This dict is intentionally small.  The primary source of truth is the
Artificial Analysis model database (cached locally).  Only add entries here
when:
  1. AA has wrong parameter counts for a model
  2. AA doesn't list the model at all and name parsing can't resolve it
  3. The model name is misleading (e.g., per-expert size instead of total)

A key matches an Arena name exactly, or as its start followed only by
release tags ("kimi-k2" covers "kimi-k2-0905-preview" and
"kimi-k2-thinking-turbo", not "kimi-k2.6" or a "-flash" sibling).  Sizes are
checkpoint totals from Hugging Face where the weights are published.
"""

from __future__ import annotations

# (total_params_B, active_params_B, architecture)
ModelOverride = tuple[float, float, str]

MODEL_OVERRIDES: dict[str, ModelOverride] = {
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
    "Kimi-K2.6": (1000, 32, "moe"),
    "Kimi-K2": (1000, 32, "moe"),
    "LongCat-Flash": (560, 27, "moe"),

    # === AA maps to wrong variant or wrong data ===
    "Phi-4": (14, 14, "dense"),             # AA has multimodal 5.6B, arena means text 14B
    "deepseek-llm-67b-chat": (67, 67, "dense"),  # AA incorrectly lists as 7B
    # AA lists the 552B backbone; the checkpoint adds 196B of Engram memory,
    # a vision encoder and a draft model, all loaded by default
    "deepseek-v4.1-flash": (763, 16, "moe"),
    # MatFormer with per-layer embeddings, not MoE
    "gemma-3n-e4b-it": (7.85, 7.85, "dense"),

    # === Older models not in AA, no parseable size in name ===
    "phi-3-medium": (14, 14, "dense"),
    "phi-3-small": (7.4, 7.4, "dense"),
    "phi-3-mini": (3.8, 3.8, "dense"),
    "athene-v2": (72, 72, "dense"),            # Fine-tune of Qwen2.5-72B
    "trinity-large": (398.6, 13, "moe"),       # Arcee AI's sparse MoE (Trinity-Large-Preview)
    "command-r-plus": (104, 104, "dense"),     # Cohere c4ai-command-r-plus
    "command-r-plus-08-2024": (104, 104, "dense"),  # Cohere Command R+ Aug 2024
    "command-r": (35, 35, "dense"),            # Cohere c4ai-command-r-v01
    "command-r-08-2024": (32.3, 32.3, "dense"),  # Cohere Command R Aug 2024
    "ibm-granite-h-small": (32.2, 9, "moe"),   # Granite-4.0-H-Small, hybrid Mamba-2 and MoE

    # === Pages AA lists without sizes ===
    "mimo-v2.5": (310.8, 15, "moe"),           # XiaomiMiMo/MiMo-V2.5
    "deepseek-r1-0528": (684.5, 37, "moe"),    # same checkpoint as DeepSeek R1
    "hunyuan-hy3-preview": (298.8, 21, "moe"),  # tencent/Hy3-preview
    "step-3.5-flash": (199.4, 11, "moe"),      # stepfun-ai/Step-3.5-Flash
    "step-3": (321, 38, "moe"),                # stepfun-ai/step3
    "minimax-m1": (456.1, 45.9, "moe"),        # MiniMaxAI/MiniMax-M1-80k
    "mistral-small-2506": (24, 24, "dense"),   # Mistral-Small-3.2-24B-Instruct-2506
    "mistral-large-2411": (122.6, 122.6, "dense"),  # Mistral-Large-Instruct-2411
    "deepseek-v2.5-1210": (235.7, 21, "moe"),  # DeepSeek-V2.5-1210
    "openchat-3.5": (7.24, 7.24, "dense"),     # Mistral-7B fine-tune (and its 0106 release)
    "snowflake-arctic-instruct": (478.6, 17, "moe"),  # Snowflake/snowflake-arctic-instruct
}

# Models Arena lists under an open license whose weights were never
# released: nobody can run them, so they are left off the tables
NOT_RELEASED: dict[str, str] = {
    "qwen-max-0919": "Qwen-Max is API-only; Arena lists it under the Qwen license",
}
