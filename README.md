# LLM Arena VRAM Calculator

Enriches the [Arena.ai](https://arena.ai/leaderboard/text?license=open-source) open-source LLM leaderboard with parameter counts and VRAM estimates for single-GPU deployment feasibility.

> **Last updated:** 2026-02-25 22:45 UTC | **Models:** 35 | **Resolved:** 35 (100.0%)

## Best Model Per GPU

Highest-ranked Arena model that fits on each single GPU at each precision (with 25% serving overhead for KV cache, activations, and framework).

| GPU | VRAM | Best Model (FP8) | Rank | VRAM Est. | Best Model (INT4) | Rank | VRAM Est. |
|-----|------|------------------|------|-----------|-------------------|------|-----------|
| H100 SXM | 80 GB | Qwen3-32B | #8 | 40.0 GB | Qwen3-32B | #8 | 20.0 GB |
| RTX PRO 6000 | 96 GB | Qwen3-32B | #8 | 40.0 GB | Qwen3-32B | #8 | 20.0 GB |
| H200 SXM | 141 GB | Qwen3-32B | #8 | 40.0 GB | Qwen3-32B | #8 | 20.0 GB |
| B200 SXM | 180 GB | Qwen3-32B | #8 | 40.0 GB | Qwen3-235B-A22B | #5 | 146.9 GB |
| B300 SXM | 288 GB | Qwen3-32B | #8 | 40.0 GB | Qwen3-235B-A22B | #5 | 146.9 GB |

## Open-Source LLM Leaderboard (Enriched)

| Rank | Model | Score | Params (B) | Active (B) | Arch | VRAM FP8 | VRAM INT4 | Best GPU (FP8) |
|------|-------|-------|------------|------------|------|----------|-----------|----------------|
| 1 | GLM-5 | 1455 | 744 | 40 | MoE | 930 GB | 465 GB | MULTI-GPU |
| 2 | Kimi-K2.5-Thinking | 1448 | 1000 | 32 | MoE | 1250 GB | 625 GB | MULTI-GPU |
| 3 | DeepSeek-R1 | 1440 | 671 | 37 | MoE | 838.8 GB | 419.4 GB | MULTI-GPU |
| 4 | DeepSeek-V3.2 | 1435 | 671 | 37 | MoE | 838.8 GB | 419.4 GB | MULTI-GPU |
| 5 | Qwen3-235B-A22B | 1430 | 235 | 22 | MoE | 293.8 GB | 146.9 GB | MULTI-GPU |
| 6 | Llama-3.1-405B-Instruct | 1425 | 405 | 405 | Dense | 506.2 GB | 253.1 GB | MULTI-GPU |
| 7 | Mistral-Large-3 | 1420 | 675 | 41 | MoE | 843.8 GB | 421.9 GB | MULTI-GPU |
| 8 | Qwen3-32B | 1415 | 32 | 32 | Dense | 40 GB | 20 GB | H100 SXM |
| 9 | Gemma 3 27B IT | 1410 | 27 | 27 | Dense | 33.8 GB | 16.9 GB | H100 SXM |
| 10 | GPT-OSS-120B | 1408 | 117 | 5.1 | MoE | 146.2 GB | 73.1 GB | B200 SXM |
| 11 | Qwen3-30B-A3B | 1405 | 30 | 3 | MoE | 37.5 GB | 18.8 GB | H100 SXM |
| 12 | Llama-3.3-70B-Instruct | 1400 | 70 | 70 | Dense | 87.5 GB | 43.8 GB | RTX PRO 6000 |
| 13 | MiniMax-M2.5 | 1395 | 230 | 10 | MoE | 287.5 GB | 143.8 GB | B300 SXM |
| 14 | Phi-4 | 1390 | 14 | 14 | Dense | 17.5 GB | 8.8 GB | H100 SXM |
| 15 | Falcon-H1-34B | 1385 | 34 | 34 | Dense | 42.5 GB | 21.2 GB | H100 SXM |
| 16 | Command A | 1380 | 111 | 11 | MoE | 138.8 GB | 69.4 GB | H200 SXM |
| 17 | LongCat-Flash-Chat | 1375 | 560 | 27 | MoE | 700 GB | 350 GB | MULTI-GPU |
| 18 | OLMo-2-32B | 1370 | 32 | 32 | Dense | 40 GB | 20 GB | H100 SXM |
| 19 | Qwen3-14B | 1365 | 14 | 14 | Dense | 17.5 GB | 8.8 GB | H100 SXM |
| 20 | GPT-OSS-20B | 1360 | 21 | 3.6 | MoE | 26.2 GB | 13.1 GB | H100 SXM |
| 21 | DeepSeek-R1-0528-Qwen3-8B | 1355 | 8 | 8 | Dense | 10 GB | 5 GB | H100 SXM |
| 22 | Llama-3.1-8B-Instruct | 1340 | 8 | 8 | Dense | 10 GB | 5 GB | H100 SXM |
| 23 | Qwen3-8B | 1335 | 8 | 8 | Dense | 10 GB | 5 GB | H100 SXM |
| 24 | Qwen3-4B | 1320 | 4 | 4 | Dense | 5 GB | 2.5 GB | H100 SXM |
| 25 | Qwen3-Coder-480B-A35B | 1450 | 480 | 35 | MoE | 600 GB | 300 GB | MULTI-GPU |
| 26 | Qwen3.5-397B-A17B | 1445 | 397 | 17 | MoE | 496.2 GB | 248.1 GB | MULTI-GPU |
| 27 | Jamba-1.5-Large | 1350 | 398 | 94 | MoE | 497.5 GB | 248.8 GB | MULTI-GPU |
| 28 | DBRX-Instruct | 1330 | 132 | 36 | MoE | 165 GB | 82.5 GB | B200 SXM |
| 29 | Llama-4-Maverick | 1442 | 400 | 17 | MoE | 500 GB | 250 GB | MULTI-GPU |
| 30 | Llama-4-Scout | 1418 | 109 | 17 | MoE | 136.2 GB | 68.1 GB | H200 SXM |
| 31 | Mixtral-8x22B | 1380 | 140.8 | 39.6 | MoE | 176 GB | 88 GB | B200 SXM |
| 32 | Mixtral-8x7B | 1350 | 44.8 | 12.6 | MoE | 56 GB | 28 GB | H100 SXM |
| 33 | Qwen3-0.6B | 1280 | 0.6 | 0.6 | Dense | 0.8 GB | 0.4 GB | H100 SXM |
| 34 | Vicuna-33B | 1310 | 33 | 33 | Dense | 41.2 GB | 20.6 GB | H100 SXM |
| 35 | Mistral-Small-2506-24B | 1370 | 24 | 24 | Dense | 30 GB | 15 GB | H100 SXM |

## How It Works

### VRAM Calculation
- **Weight VRAM** = total_params x bytes_per_param (BF16: 2B, FP8: 1B, INT4: 0.5B)
- **Serving VRAM** = weight_VRAM x 1.25 (25% overhead for KV cache, activations, framework)
- For **MoE models**, VRAM is based on **total** parameters (all experts must be loaded into memory)

### Parameter Resolution
Parameters are resolved automatically via a priority chain:
1. **Override corrections** - small dict for models with misleading names or no public metadata
2. **Name parsing** - regex extraction of `{N}B` and `A{N}B` patterns from model names
3. **HuggingFace API** - safetensors metadata and config.json for architecture details
4. **Artificial Analysis** - web scraping as backup

### GPU Configurations

| GPU | VRAM | Architecture | Native FP8 |
|-----|------|-------------|------------|
| H100 SXM | 80 GB | Hopper | Yes |
| RTX PRO 6000 | 96 GB | Ada Lovelace | No (software emulation) |
| H200 SXM | 141 GB | Hopper | Yes |
| B200 SXM | 180 GB | Blackwell | Yes |
| B300 SXM | 288 GB | Blackwell Ultra | Yes |

## Usage

```bash
cd arena_enrichment
pip install -r requirements.txt

# Full pipeline (scrapes arena.ai live)
python enrich_arena.py --update-readme

# Use a pre-downloaded CSV
python enrich_arena.py --input data.csv --update-readme

# Skip network resolution (overrides + name parsing only)
python enrich_arena.py --input data.csv --no-network --update-readme
```
