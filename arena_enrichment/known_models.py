"""
Hardcoded model parameter databases and Arena-to-HuggingFace name mappings.

These are the highest-priority resolution source for model parameters.
Values are sourced from official model cards, technical reports, and
verified community documentation.
"""

# Model name fragment -> (total_params_B, active_params_B, architecture)
# For dense models, active == total.
# For MoE models, active < total (active = params per forward pass).
KNOWN_MODELS = {
    # === GLM family (Zhipu AI) ===
    "GLM-5": (744, 40, "moe"),
    "GLM-4.7": (355, 32, "moe"),
    "GLM-4.6": (355, 32, "moe"),
    "GLM-4.5": (355, 32, "moe"),
    "GLM-4.5-Air": (106, 12, "moe"),

    # === Kimi family (Moonshot AI) ===
    "Kimi-K2.5": (1000, 32, "moe"),
    "Kimi-K2": (1000, 32, "moe"),

    # === DeepSeek family ===
    "DeepSeek-V3": (671, 37, "moe"),
    "DeepSeek-V3.1": (671, 37, "moe"),
    "DeepSeek-V3.2": (671, 37, "moe"),
    "DeepSeek-R1": (671, 37, "moe"),

    # === Mistral family ===
    "Mistral-Large-3": (675, 41, "moe"),

    # === LongCat (Meituan) ===
    "LongCat-Flash": (560, 27, "moe"),

    # === MiniMax family ===
    "MiniMax-M2.5": (230, 10, "moe"),
    "MiniMax-M2.1": (230, 10, "moe"),
    "MiniMax-M2": (230, 10, "moe"),
    "MiniMax-M1": (456, 46, "moe"),

    # === OpenAI OSS ===
    "GPT-OSS-120B": (117, 5.1, "moe"),
    "GPT-OSS-20B": (21, 3.6, "moe"),

    # === Qwen MoE models (not parseable from name alone) ===
    "Qwen3.5-397B-A17B": (397, 17, "moe"),
    "Qwen3-Next-80B-A3B": (80, 3, "moe"),
    "Qwen3-Coder-480B-A35B": (480, 35, "moe"),

    # === Dense models commonly on the leaderboard ===
    "Llama-4-Maverick": (400, 17, "moe"),
    "Llama-4-Scout": (109, 17, "moe"),
    "Falcon-H1-34B": (34, 3.5, "moe"),
    "Falcon-H1-7B": (7.3, 0.8, "moe"),

    # === NVIDIA ===
    "Llama-3.1-Nemotron-70B": (70, 70, "dense"),
    "Nemotron-4-340B": (340, 340, "dense"),

    # === Command R family (Cohere) ===
    "Command R+": (104, 104, "dense"),
    "Command R": (35, 35, "dense"),
    "Command A": (111, 11, "moe"),

    # === Yi family (01.AI) ===
    "Yi-Large": (102, 102, "moe"),

    # === Jamba (AI21) ===
    "Jamba-1.5-Large": (398, 94, "moe"),
    "Jamba-1.5-Mini": (52, 12, "moe"),

    # === DBRX (Databricks) ===
    "DBRX-Instruct": (132, 36, "moe"),

    # === Arctic (Snowflake) ===
    "Snowflake-Arctic": (480, 17, "moe"),

    # === Grok (xAI) ===
    "Grok-1": (314, 86, "moe"),

    # === OLMo (AI2) ===
    "OLMo-2-32B": (32, 32, "dense"),
    "OLMo-2-13B": (13, 13, "dense"),
    "OLMo-2-7B": (7, 7, "dense"),

    # === Tulu (AI2) ===
    "Tulu-3-70B": (70, 70, "dense"),
    "Tulu-3-8B": (8, 8, "dense"),

    # === Map Neo ===
    "MAP-Neo-7B": (7, 7, "dense"),

    # === InternLM ===
    "InternLM3-8B": (8, 8, "dense"),
    "InternLM2.5-20B": (20, 20, "dense"),

    # === Microsoft Phi family ===
    "Phi-4": (14, 14, "dense"),
    "Phi-4-Mini": (3.8, 3.8, "dense"),
    "Phi-3.5-MoE": (42, 6.6, "moe"),
    "Phi-3-Medium": (14, 14, "dense"),
    "Phi-3-Small": (7, 7, "dense"),
    "Phi-3-Mini": (3.8, 3.8, "dense"),

    # === Gemma family (Google) ===
    "Gemma 3 27B": (27, 27, "dense"),
    "Gemma 3 12B": (12, 12, "dense"),
    "Gemma 3 4B": (4, 4, "dense"),
    "Gemma 3 1B": (1, 1, "dense"),
    "Gemma 2 27B": (27, 27, "dense"),
    "Gemma 2 9B": (9, 9, "dense"),

    # === Mistral smaller models ===
    "Mistral-Small": (24, 24, "dense"),
    "Mistral-Nemo": (12, 12, "dense"),
    "Mixtral-8x22B": (141, 39, "moe"),
    "Mixtral-8x7B": (47, 13, "moe"),

    # === Megrez (InfiniAI) ===
    "Megrez-4-Scout": (109, 17, "moe"),

    # === Qwen2.5 dense models (for name parser backup) ===
    "QwQ-32B": (32, 32, "dense"),

    # === StarCoder / BigCode ===
    "StarCoder2-15B": (15, 15, "dense"),

    # === Vicuna / LMSys ===
    "Vicuna-33B": (33, 33, "dense"),
    "Vicuna-13B": (13, 13, "dense"),
    "Vicuna-7B": (7, 7, "dense"),
}


# Arena model name -> HuggingFace repo ID
# Used for HuggingFace API resolution when name parsing fails.
ARENA_TO_HF = {
    "GLM-5": "zai-org/GLM-5",
    "GLM-4.7": "zai-org/GLM-4.7",
    "GLM-4.6": "zai-org/GLM-4.6",
    "GLM-4.5": "zai-org/GLM-4.5",
    "Kimi-K2.5-Thinking": "moonshotai/Kimi-K2.5-Thinking",
    "Kimi-K2-Thinking": "moonshotai/Kimi-K2-Instruct",
    "DeepSeek-V3.2": "deepseek-ai/DeepSeek-V3.2",
    "DeepSeek-V3.1": "deepseek-ai/DeepSeek-V3-0324",
    "DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
    "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
    "DeepSeek-R1-0528": "deepseek-ai/DeepSeek-R1-0528",
    "Mistral-Large-3": "mistralai/Mistral-Large-3-675B-Instruct-2512",
    "Mistral-Small-2506": "mistralai/Mistral-Small-2506",
    "LongCat-Flash-Chat": "meituan-longcat/LongCat-Flash-Chat",
    "MiniMax-M2.5": "MiniMaxAI/MiniMax-M2.5",
    "MiniMax-M2.1": "MiniMaxAI/MiniMax-M2.1",
    "GPT-OSS-120B": "openai/gpt-oss-120b",
    "GPT-OSS-20B": "openai/gpt-oss-20b",
    "Gemma-3-27B-IT": "google/gemma-3-27b-it",
    "Gemma-3-12B-IT": "google/gemma-3-12b-it",
    "Gemma-3-4B-IT": "google/gemma-3-4b-it",
    "Gemma-3-1B-IT": "google/gemma-3-1b-it",
    "Llama-3.3-70B-Instruct": "meta-llama/Llama-3.3-70B-Instruct",
    "Llama-3.1-405B-Instruct": "meta-llama/Llama-3.1-405B-Instruct",
    "Llama-3.1-70B-Instruct": "meta-llama/Llama-3.1-70B-Instruct",
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama-4-Maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "Llama-4-Scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "Phi-4": "microsoft/phi-4",
    "Phi-4-Mini": "microsoft/Phi-4-mini-instruct",
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
    "Command R+": "CohereForAI/c4ai-command-r-plus",
    "Command R": "CohereForAI/c4ai-command-r-v01",
    "Command A": "CohereForAI/c4ai-command-a-03-2025",
    "Jamba-1.5-Large": "ai21labs/AI21-Jamba-1.5-Large",
    "Jamba-1.5-Mini": "ai21labs/AI21-Jamba-1.5-Mini",
    "DBRX-Instruct": "databricks/dbrx-instruct",
    "OLMo-2-32B": "allenai/OLMo-2-1124-32B-Instruct",
    "OLMo-2-13B": "allenai/OLMo-2-1124-13B-Instruct",
    "OLMo-2-7B": "allenai/OLMo-2-1124-7B-Instruct",
    "Tulu-3-70B": "allenai/Tulu-3-Llama-3.1-70B",
    "Tulu-3-8B": "allenai/Tulu-3-Llama-3.1-8B",
    "Falcon-H1-34B": "tiiuae/Falcon-H1-34B-Instruct",
    "Falcon-H1-7B": "tiiuae/Falcon-H1-7B-Instruct",
}
