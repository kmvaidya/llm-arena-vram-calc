"""
Model parameter overrides and Arena-to-HuggingFace name mappings.

MODEL_OVERRIDES contains ONLY models where automated resolution (name parsing
or HuggingFace API) gives wrong answers or fails entirely. This is intentionally
small — most models should be resolved automatically.

ARENA_TO_HF maps arena display names to HuggingFace repo IDs, helping the
HF resolver find repos without guessing.
"""

# Model name fragment -> (total_params_B, active_params_B, architecture)
#
# ONLY include models here if:
#   1. The name is misleading (e.g., GPT-OSS-120B is actually 117B)
#   2. There's no size in the name AND no reliable HF repo (e.g., GLM-5, Kimi)
#   3. The model is MoE but the name doesn't encode total/active correctly
#
# Do NOT include models that:
#   - Have size in their name (e.g., Qwen3-32B, Llama-3.3-70B)
#   - Can be resolved from HuggingFace safetensors metadata
#   - Have A{X}B patterns in their name (e.g., Qwen3-30B-A3B)
MODEL_OVERRIDES = {
    # === Models with misleading names ===
    "GPT-OSS-120B": (117, 5.1, "moe"),    # Name says 120B, actual is 117B
    "GPT-OSS-20B": (21, 3.6, "moe"),      # Name says 20B, actual is 21B

    # === Models with no size in name and no/unreliable HF repo ===
    "GLM-5": (744, 40, "moe"),
    "GLM-4.7": (355, 32, "moe"),
    "GLM-4.6": (355, 32, "moe"),
    "GLM-4.5": (355, 32, "moe"),
    "GLM-4.5-Air": (106, 12, "moe"),
    "Kimi-K2.5": (1000, 32, "moe"),       # 1T total
    "Kimi-K2": (1000, 32, "moe"),
    "DeepSeek-V3": (671, 37, "moe"),
    "DeepSeek-V3.1": (671, 37, "moe"),
    "DeepSeek-V3.2": (671, 37, "moe"),
    "DeepSeek-R1": (671, 37, "moe"),
    "Mistral-Large-3": (675, 41, "moe"),
    "LongCat-Flash": (560, 27, "moe"),
    "MiniMax-M2.5": (230, 10, "moe"),
    "MiniMax-M2.1": (230, 10, "moe"),
    "MiniMax-M2": (230, 10, "moe"),
    "MiniMax-M1": (456, 46, "moe"),

    # === Dense models with no size in name (resolved via HF when online) ===
    "Phi-4": (14, 14, "dense"),            # No B in name; HF fallback in CI

    # === MoE models where name doesn't encode total/active correctly ===
    "Llama-4-Maverick": (400, 17, "moe"),  # 17B per expert × 128 experts
    "Llama-4-Scout": (109, 17, "moe"),     # 17B per expert × 16 experts
    "Command A": (111, 11, "moe"),         # No size in name
    "Command R+": (104, 104, "dense"),     # No size in name
    "Command R": (35, 35, "dense"),        # No size in name
    "Jamba-1.5-Large": (398, 94, "moe"),   # Jamba SSM-MoE hybrid
    "Jamba-1.5-Mini": (52, 12, "moe"),
    "DBRX-Instruct": (132, 36, "moe"),
    "Snowflake-Arctic": (480, 17, "moe"),
    "Grok-1": (314, 86, "moe"),
    "Yi-Large": (102, 102, "moe"),
}


# Arena model name -> HuggingFace repo ID
# Helps the HF resolver find repos by exact ID rather than search-guessing.
# This mapping is NOT for parameter values — just for locating the right repo.
ARENA_TO_HF = {
    # GLM
    "GLM-5": "zai-org/GLM-5",
    "GLM-4.7": "zai-org/GLM-4.7",
    "GLM-4.6": "zai-org/GLM-4.6",
    "GLM-4.5": "zai-org/GLM-4.5",
    # Kimi
    "Kimi-K2.5-Thinking": "moonshotai/Kimi-K2.5-Thinking",
    "Kimi-K2-Thinking": "moonshotai/Kimi-K2-Instruct",
    # DeepSeek
    "DeepSeek-V3.2": "deepseek-ai/DeepSeek-V3.2",
    "DeepSeek-V3.1": "deepseek-ai/DeepSeek-V3-0324",
    "DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
    "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
    "DeepSeek-R1-0528": "deepseek-ai/DeepSeek-R1-0528",
    # Mistral
    "Mistral-Large-3": "mistralai/Mistral-Large-3-675B-Instruct-2512",
    "Mistral-Small-2506": "mistralai/Mistral-Small-2506",
    # Others
    "LongCat-Flash-Chat": "meituan-longcat/LongCat-Flash-Chat",
    "MiniMax-M2.5": "MiniMaxAI/MiniMax-M2.5",
    "MiniMax-M2.1": "MiniMaxAI/MiniMax-M2.1",
    "GPT-OSS-120B": "openai/gpt-oss-120b",
    "GPT-OSS-20B": "openai/gpt-oss-20b",
    # Google Gemma
    "Gemma 3 27B IT": "google/gemma-3-27b-it",
    "Gemma 3 12B IT": "google/gemma-3-12b-it",
    "Gemma 3 4B IT": "google/gemma-3-4b-it",
    "Gemma 3 1B IT": "google/gemma-3-1b-it",
    "Gemma 2 27B IT": "google/gemma-2-27b-it",
    "Gemma 2 9B IT": "google/gemma-2-9b-it",
    # Meta Llama
    "Llama-3.3-70B-Instruct": "meta-llama/Llama-3.3-70B-Instruct",
    "Llama-3.1-405B-Instruct": "meta-llama/Llama-3.1-405B-Instruct",
    "Llama-3.1-70B-Instruct": "meta-llama/Llama-3.1-70B-Instruct",
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama-4-Maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "Llama-4-Scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    # Microsoft Phi
    "Phi-4": "microsoft/phi-4",
    "Phi-4-Mini": "microsoft/Phi-4-mini-instruct",
    # Qwen
    "Qwen3-235B-A22B": "Qwen/Qwen3-235B-A22B",
    "Qwen3-32B": "Qwen/Qwen3-32B",
    "Qwen3-30B-A3B": "Qwen/Qwen3-30B-A3B",
    "Qwen3-14B": "Qwen/Qwen3-14B",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen3-4B": "Qwen/Qwen3-4B",
    "Qwen3-1.7B": "Qwen/Qwen3-1.7B",
    "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
    "Qwen2.5-72B-Instruct": "Qwen/Qwen2.5-72B-Instruct",
    "Qwen2.5-32B-Instruct": "Qwen/Qwen2.5-32B-Instruct",
    "Qwen2.5-14B-Instruct": "Qwen/Qwen2.5-14B-Instruct",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "QwQ-32B": "Qwen/QwQ-32B",
    # Cohere
    "Command R+": "CohereForAI/c4ai-command-r-plus",
    "Command R": "CohereForAI/c4ai-command-r-v01",
    "Command A": "CohereForAI/c4ai-command-a-03-2025",
    # AI21
    "Jamba-1.5-Large": "ai21labs/AI21-Jamba-1.5-Large",
    "Jamba-1.5-Mini": "ai21labs/AI21-Jamba-1.5-Mini",
    # Others
    "DBRX-Instruct": "databricks/dbrx-instruct",
    "OLMo-2-32B": "allenai/OLMo-2-1124-32B-Instruct",
    "OLMo-2-13B": "allenai/OLMo-2-1124-13B-Instruct",
    "OLMo-2-7B": "allenai/OLMo-2-1124-7B-Instruct",
    "Tulu-3-70B": "allenai/Tulu-3-Llama-3.1-70B",
    "Tulu-3-8B": "allenai/Tulu-3-Llama-3.1-8B",
    "Falcon-H1-34B": "tiiuae/Falcon-H1-34B-Instruct",
    "Falcon-H1-7B": "tiiuae/Falcon-H1-7B-Instruct",
    "Mixtral-8x22B": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "Mixtral-8x7B": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Nemotron-4-340B": "nvidia/Nemotron-4-340B-Instruct",
    "Llama-3.1-Nemotron-70B": "nvidia/Llama-3.1-Nemotron-70B-Instruct",
}
