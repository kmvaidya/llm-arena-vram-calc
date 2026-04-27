# LLM Arena VRAM Calculator

Enriches the [Arena.ai](https://arena.ai/leaderboard/text?license=open-source) open-source LLM leaderboard with parameter counts and VRAM estimates for single-GPU deployment feasibility.

Most LLM leaderboards rank models by quality but ignore deployment constraints. This tool answers: *"What's the best model I can actually run on my hardware?"* by cross-referencing Arena rankings with VRAM requirements across precisions.

> **Last updated:** 2026-04-27 08:14 UTC | **Models:** 200 | **Resolved:** 158 (79.0%)

> **Warning:** AA data may be stale (RSC fetch failed, using cached data).

## Best Model Per GPU

Highest-ranked Arena model that fits on each single GPU (includes 25% serving overhead for KV cache, activations, and framework).

### BF16 (Full Precision)

| GPU | VRAM | Best Model | Arena Rank | Params | Arch | Serving VRAM |
|-----|------|------------|------------|--------|------|--------------|
| H100 SXM | 80 GB | gemma-4-31b | #6 | 31B | Dense | 77.5 GB |
| RTX PRO 6000 | 96 GB | gemma-4-31b | #6 | 31B | Dense | 77.5 GB |
| H200 SXM | 141 GB | gemma-4-31b | #6 | 31B | Dense | 77.5 GB |
| B200 SXM | 180 GB | gemma-4-31b | #6 | 31B | Dense | 77.5 GB |
| B300 SXM | 288 GB | gemma-4-31b | #6 | 31B | Dense | 77.5 GB |

### FP8 (8-bit)

| GPU | VRAM | Best Model | Arena Rank | Params | Arch | Serving VRAM |
|-----|------|------------|------------|--------|------|--------------|
| H100 SXM | 80 GB | gemma-4-31b | #6 | 31B | Dense | 38.8 GB |
| RTX PRO 6000 | 96 GB | gemma-4-31b | #6 | 31B | Dense | 38.8 GB |
| H200 SXM | 141 GB | gemma-4-31b | #6 | 31B | Dense | 38.8 GB |
| B200 SXM | 180 GB | gemma-4-31b | #6 | 31B | Dense | 38.8 GB |
| B300 SXM | 288 GB | gemma-4-31b | #6 | 31B | Dense | 38.8 GB |

### INT4 (4-bit)

| GPU | VRAM | Best Model | Arena Rank | Params | Arch | Serving VRAM |
|-----|------|------------|------------|--------|------|--------------|
| H100 SXM | 80 GB | gemma-4-31b | #6 | 31B | Dense | 19.4 GB |
| RTX PRO 6000 | 96 GB | gemma-4-31b | #6 | 31B | Dense | 19.4 GB |
| H200 SXM | 141 GB | gemma-4-31b | #6 | 31B | Dense | 19.4 GB |
| B200 SXM | 180 GB | gemma-4-31b | #6 | 31B | Dense | 19.4 GB |
| B300 SXM | 288 GB | gemma-4-31b | #6 | 31B | Dense | 19.4 GB |

## Full Leaderboard

| Rank | Model | Score | Params (B) | Arch | VRAM BF16 | VRAM FP8 | VRAM INT4 | Fits on |
|------|-------|-------|------------|------|-----------|----------|-----------|---------|
| 1 | glm-5.1 | 1469 | ? | ? | ? | ? | ? | ? |
| 2 | deepseek-v4-pro | 1463 | 1600 (49) | MoE | 4000 | 2000 | 1000 | Multi-GPU |
| 3 | deepseek-v4-pro-thinking | 1462 | ? | ? | ? | ? | ? | ? |
| 4 | kimi-k2.6 | 1458 | 1000 (32) | MoE | 2500 | 1250 | 625 | Multi-GPU |
| 5 | glm-5 | 1457 | 744 (40) | MoE | 1860 | 930 | 465 | Multi-GPU |
| 6 | gemma-4-31b | 1451 | 31 | Dense | 77.5 | 38.8 | 19.4 | H100 SXM (FP8) |
| 7 | kimi-k2.5-thinking | 1449 | 1000 (32) | MoE | 2500 | 1250 | 625 | Multi-GPU |
| 8 | qwen3.5-397b-a17b | 1446 | 397 (17) | MoE | 992.5 | 496.2 | 248.1 | Multi-GPU |
| 9 | glm-4.7 | 1442 | ? | ? | ? | ? | ? | ? |
| 10 | deepseek-v4-flash-thinking | 1439 | ? | ? | ? | ? | ? | ? |
| 11 | gemma-4-26b-a4b | 1438 | 26 (4) | MoE | 65 | 32.5 | 16.2 | H100 SXM (FP8) |
| 12 | deepseek-v4-flash | 1433 | 284 (13) | MoE | 710 | 355 | 177.5 | Multi-GPU |
| 13 | kimi-k2.5-instant | 1432 | 1000 (32) | MoE | 2500 | 1250 | 625 | Multi-GPU |
| 14 | kimi-k2-thinking-turbo | 1429 | 1000 (32) | MoE | 2500 | 1250 | 625 | Multi-GPU |
| 15 | glm-4.6 | 1425 | ? | ? | ? | ? | ? | ? |
| 16 | deepseek-v3.2-exp-thinking | 1424 | ? | ? | ? | ? | ? | ? |
| 17 | deepseek-v3.2 | 1424 | ? | ? | ? | ? | ? | ? |
| 18 | deepseek-v3.2-exp | 1423 | ? | ? | ? | ? | ? | ? |
| 19 | qwen3-235b-a22b-instruct-2507 | 1422 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |
| 20 | deepseek-v3.2-thinking | 1422 | ? | ? | ? | ? | ? | ? |
| 21 | deepseek-r1-0528 | 1421 | ? | ? | ? | ? | ? | ? |
| 22 | qwen3.5-122b-a10b | 1418 | 122 (10) | MoE | 305 | 152.5 | 76.2 | B200 SXM (FP8) |
| 23 | kimi-k2-0905-preview | 1418 | 1000 (32) | MoE | 2500 | 1250 | 625 | Multi-GPU |
| 24 | deepseek-v3.1 | 1417 | ? | ? | ? | ? | ? | ? |
| 25 | kimi-k2-0711-preview | 1417 | 1000 (32) | MoE | 2500 | 1250 | 625 | Multi-GPU |
| 26 | deepseek-v3.1-terminus-thinking | 1417 | ? | ? | ? | ? | ? | ? |
| 27 | deepseek-v3.1-thinking | 1416 | ? | ? | ? | ? | ? | ? |
| 28 | deepseek-v3.1-terminus | 1415 | ? | ? | ? | ? | ? | ? |
| 29 | qwen3-vl-235b-a22b-instruct | 1415 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |
| 30 | mistral-large-3 | 1415 | 675 (41) | MoE | 1687.5 | 843.8 | 421.9 | Multi-GPU |
| 31 | glm-4.5 | 1411 | 355 (32) | MoE | 887.5 | 443.8 | 221.9 | Multi-GPU |
| 32 | qwen3.5-27b | 1405 | 27 | Dense | 67.5 | 33.8 | 16.9 | H100 SXM (FP8) |
| 33 | minimax-m2.7 | 1403 | ? | ? | ? | ? | ? | ? |
| 34 | qwen3-235b-a22b-no-thinking | 1402 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |
| 35 | qwen3-next-80b-a3b-instruct | 1401 | 80 (3) | MoE | 200 | 100 | 50 | H200 SXM (FP8) |
| 36 | longcat-flash-chat | 1401 | 560 (27) | MoE | 1400 | 700 | 350 | Multi-GPU |
| 37 | minimax-m2.5 | 1399 | ? | ? | ? | ? | ? | ? |
| 38 | qwen3-235b-a22b-thinking-2507 | 1399 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |
| 39 | deepseek-r1 | 1397 | 685 (37) | MoE | 1712.5 | 856.2 | 428.1 | Multi-GPU |
| 40 | qwen3-vl-235b-a22b-thinking | 1395 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |
| 41 | qwen3.5-35b-a3b | 1395 | 35 (3) | MoE | 87.5 | 43.8 | 21.9 | H100 SXM (FP8) |
| 42 | deepseek-v3-0324 | 1395 | 671 (37) | MoE | 1677.5 | 838.8 | 419.4 | Multi-GPU |
| 43 | mimo-v2-flash (non-thinking) | 1392 | ? | ? | ? | ? | ? | ? |
| 44 | step-3.5-flash | 1392 | ? | ? | ? | ? | ? | ? |
| 45 | mimo-v2-flash (thinking) | 1387 | ? | ? | ? | ? | ? | ? |
| 46 | qwen3-coder-480b-a35b-instruct | 1387 | 480 (35) | MoE | 1200 | 600 | 300 | Multi-GPU |
| 47 | minimax-m2.1-preview | 1385 | ? | ? | ? | ? | ? | ? |
| 48 | qwen3-30b-a3b-instruct-2507 | 1383 | 30 (3) | MoE | 75 | 37.5 | 18.8 | H100 SXM (FP8) |
| 49 | glm-4.6v | 1377 | ? | ? | ? | ? | ? | ? |
| 50 | qwen3-235b-a22b | 1374 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |

<details><summary>Show remaining 150 models</summary>

| Rank | Model | Score | Params (B) | Arch | VRAM BF16 | VRAM FP8 | VRAM INT4 | Fits on |
|------|-------|-------|------------|------|-----------|----------|-----------|---------|
| 51 | trinity-large-preview | 1374 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 52 | glm-4.5-air | 1372 | ? | ? | ? | ? | ? | ? |
| 53 | qwen3-next-80b-a3b-thinking | 1369 | 80 (3) | MoE | 200 | 100 | 50 | H200 SXM (FP8) |
| 54 | glm-4.7-flash | 1368 | ? | ? | ? | ? | ? | ? |
| 55 | gemma-3-27b-it | 1365 | 27 | Dense | 67.5 | 33.8 | 16.9 | H100 SXM (FP8) |
| 56 | minimax-m1 | 1363 | ? | ? | ? | ? | ? | ? |
| 57 | nvidia-nemotron-3-super-120b-a12b | 1361 | 120 (12) | MoE | 300 | 150 | 75 | B200 SXM (FP8) |
| 58 | deepseek-v3 | 1358 | 671 (37) | MoE | 1677.5 | 838.8 | 419.4 | Multi-GPU |
| 59 | mistral-small-2506 | 1357 | ? | ? | ? | ? | ? | ? |
| 60 | intellect-3 | 1356 | 107 | Dense | 267.5 | 133.8 | 66.9 | H200 SXM (FP8) |
| 61 | command-a-03-2025 | 1353 | ? | ? | ? | ? | ? | ? |
| 62 | gpt-oss-120b | 1353 | 117 (5.1) | MoE | 292.5 | 146.2 | 73.1 | B200 SXM (FP8) |
| 63 | glm-4.5v | 1353 | ? | ? | ? | ? | ? | ? |
| 64 | step-3 | 1347 | ? | ? | ? | ? | ? | ? |
| 65 | qwen3-32b | 1347 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 66 | llama-3.1-nemotron-ultra-253b-v1 | 1347 | 253 | Dense | 632.5 | 316.2 | 158.1 | Multi-GPU |
| 67 | minimax-m2 | 1346 | 230 (10) | MoE | 575 | 287.5 | 143.8 | B300 SXM (FP8) |
| 68 | ling-flash-2.0 | 1345 | ? | ? | ? | ? | ? | ? |
| 69 | nvidia-llama-3.3-nemotron-super-49b-v1.5 | 1342 | 49 | Dense | 122.5 | 61.2 | 30.6 | H100 SXM (FP8) |
| 70 | gemma-3-12b-it | 1341 | 12 | Dense | 30 | 15 | 7.5 | H100 SXM (FP8) |
| 71 | qwq-32b | 1335 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 72 | llama-3.1-405b-instruct-bf16 | 1334 | 405 | Dense | 1012.5 | 506.2 | 253.1 | Multi-GPU |
| 73 | llama-3.1-405b-instruct-fp8 | 1332 | 405 | Dense | 1012.5 | 506.2 | 253.1 | Multi-GPU |
| 74 | olmo-3.1-32b-instruct | 1330 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 75 | molmo-2-8b | 1327 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 76 | llama-3.3-nemotron-49b-super-v1 | 1327 | 49 | Dense | 122.5 | 61.2 | 30.6 | H100 SXM (FP8) |
| 77 | qwen3-30b-a3b | 1327 | 30 (3) | MoE | 75 | 37.5 | 18.8 | H100 SXM (FP8) |
| 78 | llama-4-maverick-17b-128e-instruct | 1326 | 400 (17) | MoE | 1000 | 500 | 250 | Multi-GPU |
| 79 | deepseek-v2.5-1210 | 1323 | ? | ? | ? | ? | ? | ? |
| 80 | llama-4-scout-17b-16e-instruct | 1322 | 109 (17) | MoE | 272.5 | 136.2 | 68.1 | H200 SXM (FP8) |
| 81 | ring-flash-2.0 | 1320 | ? | ? | ? | ? | ? | ? |
| 82 | llama-3.3-70b-instruct | 1318 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 83 | gemma-3n-e4b-it | 1318 | 8.4 (4) | MoE | 21 | 10.5 | 5.2 | H100 SXM (FP8) |
| 84 | qwen-max-0919 | 1317 | ? | ? | ? | ? | ? | ? |
| 85 | gpt-oss-20b | 1317 | 21 (3.6) | MoE | 52.5 | 26.2 | 13.1 | H100 SXM (FP8) |
| 86 | nvidia-nemotron-3-nano-30b-a3b-bf16 | 1317 | 30 (3) | MoE | 75 | 37.5 | 18.8 | H100 SXM (FP8) |
| 87 | athene-v2-chat | 1314 | 72 | Dense | 180 | 90 | 45 | RTX PRO 6000 (FP8) |
| 88 | mistral-large-2407 | 1313 | 123 | Dense | 307.5 | 153.8 | 76.9 | B200 SXM (FP8) |
| 89 | deepseek-v2.5 | 1306 | ? | ? | ? | ? | ? | ? |
| 90 | athene-70b-0725 | 1305 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 91 | olmo-3-32b-think | 1305 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 92 | mistral-large-2411 | 1304 | ? | ? | ? | ? | ? | ? |
| 93 | mistral-small-3.1-24b-instruct-2503 | 1303 | 24 | Dense | 60 | 30 | 15 | H100 SXM (FP8) |
| 94 | gemma-3-4b-it | 1303 | 4 | Dense | 10 | 5 | 2.5 | H100 SXM (FP8) |
| 95 | qwen2.5-72b-instruct | 1302 | 72 | Dense | 180 | 90 | 45 | RTX PRO 6000 (FP8) |
| 96 | llama-3.1-nemotron-70b-instruct | 1298 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 97 | llama-3.1-70b-instruct | 1293 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 98 | jamba-1.5-large | 1288 | ? | ? | ? | ? | ? | ? |
| 99 | gemma-2-27b-it | 1288 | 27 | Dense | 67.5 | 33.8 | 16.9 | H100 SXM (FP8) |
| 100 | ibm-granite-h-small | 1286 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 101 | llama-3.1-tulu-3-70b | 1285 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 102 | llama-3.1-nemotron-51b-instruct | 1285 | 51 | Dense | 127.5 | 63.8 | 31.9 | H100 SXM (FP8) |
| 103 | olmo-3.1-32b-think | 1285 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 104 | gemma-2-9b-it-simpo | 1279 | 9 | Dense | 22.5 | 11.2 | 5.6 | H100 SXM (FP8) |
| 105 | nemotron-4-340b-instruct | 1276 | 340 | Dense | 850 | 425 | 212.5 | Multi-GPU |
| 106 | command-r-plus-08-2024 | 1275 | 104 | Dense | 260 | 130 | 65 | H200 SXM (FP8) |
| 107 | llama-3-70b-instruct | 1275 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 108 | mistral-small-24b-instruct-2501 | 1273 | 24 | Dense | 60 | 30 | 15 | H100 SXM (FP8) |
| 109 | qwen2.5-coder-32b-instruct | 1270 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 110 | c4ai-aya-expanse-32b | 1266 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 111 | gemma-2-9b-it | 1265 | 9 | Dense | 22.5 | 11.2 | 5.6 | H100 SXM (FP8) |
| 112 | deepseek-coder-v2 | 1263 | 236 (21) | MoE | 590 | 295 | 147.5 | Multi-GPU |
| 113 | command-r-plus | 1261 | ? | ? | ? | ? | ? | ? |
| 114 | qwen2-72b-instruct | 1261 | 72 | Dense | 180 | 90 | 45 | RTX PRO 6000 (FP8) |
| 115 | phi-4 | 1255 | 14 | Dense | 35 | 17.5 | 8.8 | H100 SXM (FP8) |
| 116 | olmo-2-0325-32b-instruct | 1251 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 117 | command-r-08-2024 | 1249 | 35 | Dense | 87.5 | 43.8 | 21.9 | H100 SXM (FP8) |
| 118 | jamba-1.5-mini | 1238 | ? | ? | ? | ? | ? | ? |
| 119 | ministral-8b-2410 | 1236 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 120 | qwen1.5-110b-chat | 1233 | 110 | Dense | 275 | 137.5 | 68.8 | H200 SXM (FP8) |
| 121 | qwen1.5-72b-chat | 1232 | 72 | Dense | 180 | 90 | 45 | RTX PRO 6000 (FP8) |
| 122 | mixtral-8x22b-instruct-v0.1 | 1228 | 140.8 (39.6) | MoE | 352 | 176 | 88 | B200 SXM (FP8) |
| 123 | command-r | 1226 | ? | ? | ? | ? | ? | ? |
| 124 | llama-3-8b-instruct | 1222 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 125 | c4ai-aya-expanse-8b | 1222 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 126 | llama-3.1-tulu-3-8b | 1220 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 127 | yi-1.5-34b-chat | 1212 | 34 | Dense | 85 | 42.5 | 21.2 | H100 SXM (FP8) |
| 128 | zephyr-orpo-141b-A35b-v0.1 | 1211 | 141 (35) | MoE | 352.5 | 176.2 | 88.1 | B200 SXM (FP8) |
| 129 | llama-3.1-8b-instruct | 1211 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 130 | granite-3.1-8b-instruct | 1207 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 131 | qwen1.5-32b-chat | 1202 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 132 | gemma-2-2b-it | 1198 | 2 | Dense | 5 | 2.5 | 1.2 | H100 SXM (FP8) |
| 133 | phi-3-medium-4k-instruct | 1197 | 14 | Dense | 35 | 17.5 | 8.8 | H100 SXM (FP8) |
| 134 | mixtral-8x7b-instruct-v0.1 | 1196 | 44.8 (12.6) | MoE | 112 | 56 | 28 | H100 SXM (FP8) |
| 135 | dbrx-instruct-preview | 1194 | ? | ? | ? | ? | ? | ? |
| 136 | internlm2_5-20b-chat | 1190 | 20 | Dense | 50 | 25 | 12.5 | H100 SXM (FP8) |
| 137 | qwen1.5-14b-chat | 1190 | 14 | Dense | 35 | 17.5 | 8.8 | H100 SXM (FP8) |
| 138 | wizardlm-70b | 1183 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 139 | deepseek-llm-67b-chat | 1183 | 67 | Dense | 167.5 | 83.8 | 41.9 | RTX PRO 6000 (FP8) |
| 140 | yi-34b-chat | 1183 | 34 | Dense | 85 | 42.5 | 21.2 | H100 SXM (FP8) |
| 141 | openchat-3.5-0106 | 1181 | ? | ? | ? | ? | ? | ? |
| 142 | openchat-3.5 | 1181 | ? | ? | ? | ? | ? | ? |
| 143 | granite-3.0-8b-instruct | 1181 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 144 | gemma-1.1-7b-it | 1180 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 145 | snowflake-arctic-instruct | 1178 | ? | ? | ? | ? | ? | ? |
| 146 | granite-3.1-2b-instruct | 1178 | 2 | Dense | 5 | 2.5 | 1.2 | H100 SXM (FP8) |
| 147 | tulu-2-dpo-70b | 1177 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 148 | openhermes-2.5-mistral-7b | 1174 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 149 | vicuna-33b | 1171 | 33 | Dense | 82.5 | 41.2 | 20.6 | H100 SXM (FP8) |
| 150 | starling-lm-7b-beta | 1170 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 151 | phi-3-small-8k-instruct | 1170 | 7.4 | Dense | 18.5 | 9.2 | 4.6 | H100 SXM (FP8) |
| 152 | llama-2-70b-chat | 1169 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 153 | starling-lm-7b-alpha | 1166 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 154 | llama-3.2-3b-instruct | 1165 | 3 | Dense | 7.5 | 3.8 | 1.9 | H100 SXM (FP8) |
| 155 | nous-hermes-2-mixtral-8x7b-dpo | 1163 | 44.8 (12.6) | MoE | 112 | 56 | 28 | H100 SXM (FP8) |
| 156 | qwq-32b-preview | 1155 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 157 | granite-3.0-2b-instruct | 1155 | 2 | Dense | 5 | 2.5 | 1.2 | H100 SXM (FP8) |
| 158 | llama2-70b-steerlm-chat | 1154 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 159 | solar-10.7b-instruct-v1.0 | 1151 | 10.7 | Dense | 26.8 | 13.4 | 6.7 | H100 SXM (FP8) |
| 160 | dolphin-2.2.1-mistral-7b | 1151 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 161 | mpt-30b-chat | 1149 | 30 | Dense | 75 | 37.5 | 18.8 | H100 SXM (FP8) |
| 162 | mistral-7b-instruct-v0.2 | 1148 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 163 | wizardlm-13b | 1148 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 164 | falcon-180b-chat | 1146 | 180 | Dense | 450 | 225 | 112.5 | B300 SXM (FP8) |
| 165 | qwen1.5-7b-chat | 1142 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 166 | phi-3-mini-4k-instruct-june-2024 | 1142 | 3.8 | Dense | 9.5 | 4.8 | 2.4 | H100 SXM (FP8) |
| 167 | llama-2-13b-chat | 1140 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 168 | vicuna-13b | 1140 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 169 | qwen-14b-chat | 1137 | 14 | Dense | 35 | 17.5 | 8.8 | H100 SXM (FP8) |
| 170 | gemma-7b-it | 1135 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 171 | codellama-34b-instruct | 1135 | 34 | Dense | 85 | 42.5 | 21.2 | H100 SXM (FP8) |
| 172 | zephyr-7b-beta | 1130 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 173 | phi-3-mini-128k-instruct | 1128 | 3.8 | Dense | 9.5 | 4.8 | 2.4 | H100 SXM (FP8) |
| 174 | phi-3-mini-4k-instruct | 1127 | 3.8 | Dense | 9.5 | 4.8 | 2.4 | H100 SXM (FP8) |
| 175 | guanaco-33b | 1126 | 33 | Dense | 82.5 | 41.2 | 20.6 | H100 SXM (FP8) |
| 176 | zephyr-7b-alpha | 1125 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 177 | stripedhyena-nous-7b | 1120 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 178 | codellama-70b-instruct | 1118 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 179 | gemma-1.1-2b-it | 1114 | 2 | Dense | 5 | 2.5 | 1.2 | H100 SXM (FP8) |
| 180 | vicuna-7b | 1113 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 181 | smollm2-1.7b-instruct | 1113 | 1.7 | Dense | 4.2 | 2.1 | 1.1 | H100 SXM (FP8) |
| 182 | llama-3.2-1b-instruct | 1110 | 1 | Dense | 2.5 | 1.2 | 0.6 | H100 SXM (FP8) |
| 183 | mistral-7b-instruct | 1108 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 184 | llama-2-7b-chat | 1107 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 185 | gemma-2b-it | 1091 | 2 | Dense | 5 | 2.5 | 1.2 | H100 SXM (FP8) |
| 186 | qwen1.5-4b-chat | 1089 | 4 | Dense | 10 | 5 | 2.5 | H100 SXM (FP8) |
| 187 | olmo-7b-instruct | 1073 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 188 | koala-13b | 1069 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 189 | alpaca-13b | 1066 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 190 | gpt4all-13b-snoozy | 1065 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 191 | mpt-7b-chat | 1061 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 192 | chatglm3-6b | 1055 | 6 | Dense | 15 | 7.5 | 3.8 | H100 SXM (FP8) |
| 193 | RWKV-4-Raven-14B | 1040 | 14 | Dense | 35 | 17.5 | 8.8 | H100 SXM (FP8) |
| 194 | chatglm2-6b | 1023 | 6 | Dense | 15 | 7.5 | 3.8 | H100 SXM (FP8) |
| 195 | oasst-pythia-12b | 1021 | 12 | Dense | 30 | 15 | 7.5 | H100 SXM (FP8) |
| 196 | chatglm-6b | 994 | 6 | Dense | 15 | 7.5 | 3.8 | H100 SXM (FP8) |
| 197 | fastchat-t5-3b | 990 | 3 | Dense | 7.5 | 3.8 | 1.9 | H100 SXM (FP8) |
| 198 | dolly-v2-12b | 979 | 12 | Dense | 30 | 15 | 7.5 | H100 SXM (FP8) |
| 199 | llama-13b | 971 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 200 | stablelm-tuned-alpha-7b | 951 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |

</details>

## Architecture

**Data flow:** Arena.ai leaderboard → parameter resolution (4-strategy fallback) → VRAM calculation → GPU feasibility matrix

The parameter resolution chain prioritizes accuracy: manual overrides catch known errors, [Artificial Analysis](https://artificialanalysis.ai) provides bulk data for 400+ models via a single RSC stream request (cached locally, 24h TTL), name parsing extracts `{N}B` patterns as a fallback, and per-model page scraping handles the long tail.

### VRAM Estimation

| Precision | Bytes/Param | Example: 70B model |
|-----------|-------------|---------------------|
| BF16 | 2.0 | 140 GB weights, 175 GB serving |
| FP8 | 1.0 | 70 GB weights, 87.5 GB serving |
| INT4 | 0.5 | 35 GB weights, 43.8 GB serving |

**Serving VRAM** = weight VRAM × 1.25 (25% overhead for KV cache, activations, framework). For **MoE models**, all experts must be loaded regardless of active count.

### GPUs

| GPU | VRAM | Architecture | Native FP8 |
|-----|------|-------------|------------|
| H100 SXM | 80 GB | Hopper | Yes |
| RTX PRO 6000 | 96 GB | Ada Lovelace | No (software emulation) |
| H200 SXM | 141 GB | Hopper | Yes |
| B200 SXM | 180 GB | Blackwell | Yes |
| B300 SXM | 288 GB | Blackwell Ultra | Yes |

## Usage

```bash
# Install dependencies
uv sync

# Full pipeline (scrapes arena.ai live, updates README)
uv run python arena_enrichment/enrich_arena.py --update-readme

# Use a pre-downloaded CSV
uv run python arena_enrichment/enrich_arena.py --input data.csv --update-readme

# Skip network resolution (overrides + name parsing only)
uv run python arena_enrichment/enrich_arena.py --no-network --update-readme
```
