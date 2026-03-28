# LLM Arena VRAM Calculator

Enriches the [Arena.ai](https://arena.ai/leaderboard/text?license=open-source) open-source LLM leaderboard with parameter counts and VRAM estimates for single-GPU deployment feasibility.

Most LLM leaderboards rank models by quality but ignore deployment constraints. This tool answers: *"What's the best model I can actually run on my hardware?"* by cross-referencing Arena rankings with VRAM requirements across precisions.

> **Last updated:** 2026-03-28 06:52 UTC | **Models:** 191 | **Resolved:** 186 (97.4%)

## Best Model Per GPU

Highest-ranked Arena model that fits on each single GPU (includes 25% serving overhead for KV cache, activations, and framework).

### BF16 (Full Precision)

| GPU | VRAM | Best Model | Arena Rank | Params | Arch | Serving VRAM |
|-----|------|------------|------------|--------|------|--------------|
| H100 SXM | 80 GB | qwen3.5-27b | #25 | 27.8B | Dense | 69.5 GB |
| RTX PRO 6000 | 96 GB | qwen3.5-27b | #25 | 27.8B | Dense | 69.5 GB |
| H200 SXM | 141 GB | qwen3.5-27b | #25 | 27.8B | Dense | 69.5 GB |
| B200 SXM | 180 GB | qwen3.5-27b | #25 | 27.8B | Dense | 69.5 GB |
| B300 SXM | 288 GB | qwen3.5-27b | #25 | 27.8B | Dense | 69.5 GB |

### FP8 (8-bit)

| GPU | VRAM | Best Model | Arena Rank | Params | Arch | Serving VRAM |
|-----|------|------------|------------|--------|------|--------------|
| H100 SXM | 80 GB | qwen3.5-27b | #25 | 27.8B | Dense | 34.8 GB |
| RTX PRO 6000 | 96 GB | qwen3.5-27b | #25 | 27.8B | Dense | 34.8 GB |
| H200 SXM | 141 GB | qwen3.5-27b | #25 | 27.8B | Dense | 34.8 GB |
| B200 SXM | 180 GB | qwen3.5-122b-a10b | #15 | 125B | MoE | 156.2 GB |
| B300 SXM | 288 GB | qwen3.5-122b-a10b | #15 | 125B | MoE | 156.2 GB |

### INT4 (4-bit)

| GPU | VRAM | Best Model | Arena Rank | Params | Arch | Serving VRAM |
|-----|------|------------|------------|--------|------|--------------|
| H100 SXM | 80 GB | qwen3.5-122b-a10b | #15 | 125B | MoE | 78.1 GB |
| RTX PRO 6000 | 96 GB | qwen3.5-122b-a10b | #15 | 125B | MoE | 78.1 GB |
| H200 SXM | 141 GB | qwen3.5-122b-a10b | #15 | 125B | MoE | 78.1 GB |
| B200 SXM | 180 GB | qwen3-235b-a22b-instruct-2507 | #11 | 235B | MoE | 146.9 GB |
| B300 SXM | 288 GB | qwen3.5-397b-a17b | #3 | 397B | MoE | 248.1 GB |

## Full Leaderboard

| Rank | Model | Score | Params (B) | Arch | VRAM BF16 | VRAM FP8 | VRAM INT4 | Fits on |
|------|-------|-------|------------|------|-----------|----------|-----------|---------|
| 1 | glm-5 | 1455 | 744 (40) | MoE | 1860 | 930 | 465 | Multi-GPU |
| 2 | kimi-k2.5-thinking | 1453 | 1000 (32) | MoE | 2500 | 1250 | 625 | Multi-GPU |
| 3 | qwen3.5-397b-a17b | 1450 | 397 (17) | MoE | 992.5 | 496.2 | 248.1 | Multi-GPU |
| 4 | glm-4.7 | 1442 | 357 (32) | MoE | 892.5 | 446.2 | 223.1 | Multi-GPU |
| 5 | kimi-k2.5-instant | 1433 | 1000 (32) | MoE | 2500 | 1250 | 625 | Multi-GPU |
| 6 | kimi-k2-thinking-turbo | 1429 | 1000 (32) | MoE | 2500 | 1250 | 625 | Multi-GPU |
| 7 | glm-4.6 | 1425 | 357 (32) | MoE | 892.5 | 446.2 | 223.1 | Multi-GPU |
| 8 | deepseek-v3.2-exp-thinking | 1424 | 685 (37) | MoE | 1712.5 | 856.2 | 428.1 | Multi-GPU |
| 9 | deepseek-v3.2 | 1423 | 685 (37) | MoE | 1712.5 | 856.2 | 428.1 | Multi-GPU |
| 10 | deepseek-v3.2-exp | 1422 | 685 (37) | MoE | 1712.5 | 856.2 | 428.1 | Multi-GPU |
| 11 | qwen3-235b-a22b-instruct-2507 | 1422 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |
| 12 | deepseek-v3.2-thinking | 1421 | 685 (37) | MoE | 1712.5 | 856.2 | 428.1 | Multi-GPU |
| 13 | deepseek-r1-0528 | 1421 | 685 (37) | MoE | 1712.5 | 856.2 | 428.1 | Multi-GPU |
| 14 | deepseek-v3.1 | 1417 | 685 (37) | MoE | 1712.5 | 856.2 | 428.1 | Multi-GPU |
| 15 | qwen3.5-122b-a10b | 1417 | 125 (10) | MoE | 312.5 | 156.2 | 78.1 | B200 SXM (FP8) |
| 16 | kimi-k2-0905-preview | 1417 | 1000 (32) | MoE | 2500 | 1250 | 625 | Multi-GPU |
| 17 | kimi-k2-0711-preview | 1417 | 1000 (32) | MoE | 2500 | 1250 | 625 | Multi-GPU |
| 18 | deepseek-v3.1-thinking | 1416 | 685 (37) | MoE | 1712.5 | 856.2 | 428.1 | Multi-GPU |
| 19 | deepseek-v3.1-terminus-thinking | 1416 | 685 (37) | MoE | 1712.5 | 856.2 | 428.1 | Multi-GPU |
| 20 | qwen3-vl-235b-a22b-instruct | 1415 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |
| 21 | mistral-large-3 | 1415 | 675 (41) | MoE | 1687.5 | 843.8 | 421.9 | Multi-GPU |
| 22 | deepseek-v3.1-terminus | 1415 | 685 (37) | MoE | 1712.5 | 856.2 | 428.1 | Multi-GPU |
| 23 | glm-4.5 | 1410 | 355 (32) | MoE | 887.5 | 443.8 | 221.9 | Multi-GPU |
| 24 | minimax-m2.5 | 1405 | 230 (10) | MoE | 575 | 287.5 | 143.8 | B300 SXM (FP8) |
| 25 | qwen3.5-27b | 1405 | 27.8 | Dense | 69.5 | 34.8 | 17.4 | H100 SXM (FP8) |
| 26 | qwen3-235b-a22b-no-thinking | 1402 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |
| 27 | qwen3.5-35b-a3b | 1402 | 36 (3) | MoE | 90 | 45 | 22.5 | H100 SXM (FP8) |
| 28 | qwen3-next-80b-a3b-instruct | 1401 | 80 (3) | MoE | 200 | 100 | 50 | H200 SXM (FP8) |
| 29 | longcat-flash-chat | 1400 | 560 (27) | MoE | 1400 | 700 | 350 | Multi-GPU |
| 30 | qwen3-235b-a22b-thinking-2507 | 1399 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |
| 31 | deepseek-r1 | 1397 | 685 (37) | MoE | 1712.5 | 856.2 | 428.1 | Multi-GPU |
| 32 | qwen3-vl-235b-a22b-thinking | 1395 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |
| 33 | deepseek-v3-0324 | 1394 | 671 (37) | MoE | 1677.5 | 838.8 | 419.4 | Multi-GPU |
| 34 | mimo-v2-flash (non-thinking) | 1391 | 309 (15) | MoE | 772.5 | 386.2 | 193.1 | Multi-GPU |
| 35 | step-3.5-flash | 1390 | 196 (11) | MoE | 490 | 245 | 122.5 | B300 SXM (FP8) |
| 36 | qwen3-coder-480b-a35b-instruct | 1387 | 480 (35) | MoE | 1200 | 600 | 300 | Multi-GPU |
| 37 | mimo-v2-flash (thinking) | 1387 | 309 (15) | MoE | 772.5 | 386.2 | 193.1 | Multi-GPU |
| 38 | minimax-m2.1-preview | 1386 | 230 (10) | MoE | 575 | 287.5 | 143.8 | B300 SXM (FP8) |
| 39 | qwen3-30b-a3b-instruct-2507 | 1383 | 30.5 (3.3) | MoE | 76.2 | 38.1 | 19.1 | H100 SXM (FP8) |
| 40 | glm-4.6v | 1377 | 108 | Dense | 270 | 135 | 67.5 | H200 SXM (FP8) |
| 41 | trinity-large | 1376 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 42 | qwen3-235b-a22b | 1374 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |
| 43 | glm-4.5-air | 1372 | 106 (12) | MoE | 265 | 132.5 | 66.2 | H200 SXM (FP8) |
| 44 | qwen3-next-80b-a3b-thinking | 1369 | 80 (3) | MoE | 200 | 100 | 50 | H200 SXM (FP8) |
| 45 | glm-4.7-flash | 1368 | 31.2 (3) | MoE | 78 | 39 | 19.5 | H100 SXM (FP8) |
| 46 | gemma-3-27b-it | 1365 | 27.4 | Dense | 68.5 | 34.2 | 17.1 | H100 SXM (FP8) |
| 47 | nvidia-nemotron-3-super-120b-a12b | 1364 | 120.6 (12.7) | MoE | 301.5 | 150.8 | 75.4 | B200 SXM (FP8) |
| 48 | minimax-m1 | 1363 | 456 (45.9) | MoE | 1140 | 570 | 285 | Multi-GPU |
| 49 | deepseek-v3 | 1358 | 671 (37) | MoE | 1677.5 | 838.8 | 419.4 | Multi-GPU |
| 50 | mistral-small-2506 | 1356 | 22 | Dense | 55 | 27.5 | 13.8 | H100 SXM (FP8) |

<details><summary>Show remaining 141 models</summary>

| Rank | Model | Score | Params (B) | Arch | VRAM BF16 | VRAM FP8 | VRAM INT4 | Fits on |
|------|-------|-------|------------|------|-----------|----------|-----------|---------|
| 51 | intellect-3 | 1356 | 107 | Dense | 267.5 | 133.8 | 66.9 | H200 SXM (FP8) |
| 52 | gpt-oss-120b | 1353 | 117 (5.1) | MoE | 292.5 | 146.2 | 73.1 | B200 SXM (FP8) |
| 53 | command-a-03-2025 | 1353 | 111 | Dense | 277.5 | 138.8 | 69.4 | H200 SXM (FP8) |
| 54 | glm-4.5v | 1353 | 108 (12) | MoE | 270 | 135 | 67.5 | H200 SXM (FP8) |
| 55 | step-3 | 1347 | ? | ? | ? | ? | ? | ? |
| 56 | qwen3-32b | 1347 | 32.8 | Dense | 82 | 41 | 20.5 | H100 SXM (FP8) |
| 57 | llama-3.1-nemotron-ultra-253b-v1 | 1346 | 253 | Dense | 632.5 | 316.2 | 158.1 | Multi-GPU |
| 58 | minimax-m2 | 1346 | 230 (10) | MoE | 575 | 287.5 | 143.8 | B300 SXM (FP8) |
| 59 | ling-flash-2.0 | 1346 | 103 (6.1) | MoE | 257.5 | 128.8 | 64.4 | H200 SXM (FP8) |
| 60 | nvidia-llama-3.3-nemotron-super-49b-v1.5 | 1342 | 49 | Dense | 122.5 | 61.2 | 30.6 | H100 SXM (FP8) |
| 61 | gemma-3-12b-it | 1341 | 12.2 | Dense | 30.5 | 15.2 | 7.6 | H100 SXM (FP8) |
| 62 | qwq-32b | 1335 | 32.8 | Dense | 82 | 41 | 20.5 | H100 SXM (FP8) |
| 63 | llama-3.1-405b-instruct-bf16 | 1334 | 405 | Dense | 1012.5 | 506.2 | 253.1 | Multi-GPU |
| 64 | llama-3.1-405b-instruct-fp8 | 1332 | 405 | Dense | 1012.5 | 506.2 | 253.1 | Multi-GPU |
| 65 | olmo-3.1-32b-instruct | 1330 | 32.2 | Dense | 80.5 | 40.2 | 20.1 | H100 SXM (FP8) |
| 66 | qwen3-30b-a3b | 1327 | 30.5 (3.3) | MoE | 76.2 | 38.1 | 19.1 | H100 SXM (FP8) |
| 67 | llama-3.3-nemotron-49b-super-v1 | 1327 | 49 | Dense | 122.5 | 61.2 | 30.6 | H100 SXM (FP8) |
| 68 | llama-4-maverick-17b-128e-instruct | 1326 | 400 (17) | MoE | 1000 | 500 | 250 | Multi-GPU |
| 69 | molmo-2-8b | 1326 | 8.7 | Dense | 21.8 | 10.9 | 5.4 | H100 SXM (FP8) |
| 70 | deepseek-v2.5-1210 | 1323 | 236 (21) | MoE | 590 | 295 | 147.5 | Multi-GPU |
| 71 | llama-4-scout-17b-16e-instruct | 1322 | 109 (17) | MoE | 272.5 | 136.2 | 68.1 | H200 SXM (FP8) |
| 72 | ring-flash-2.0 | 1320 | 103 (6.1) | MoE | 257.5 | 128.8 | 64.4 | H200 SXM (FP8) |
| 73 | llama-3.3-70b-instruct | 1318 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 74 | gpt-oss-20b | 1318 | 21 (3.6) | MoE | 52.5 | 26.2 | 13.1 | H100 SXM (FP8) |
| 75 | gemma-3n-e4b-it | 1318 | 8.4 (4) | MoE | 21 | 10.5 | 5.2 | H100 SXM (FP8) |
| 76 | nvidia-nemotron-3-nano-30b-a3b-bf16 | 1317 | 31.6 (3.6) | MoE | 79 | 39.5 | 19.8 | H100 SXM (FP8) |
| 77 | qwen-max-0919 | 1317 | ? | ? | ? | ? | ? | ? |
| 78 | athene-v2-chat | 1314 | 72 | Dense | 180 | 90 | 45 | RTX PRO 6000 (FP8) |
| 79 | mistral-large-2407 | 1313 | 123 | Dense | 307.5 | 153.8 | 76.9 | B200 SXM (FP8) |
| 80 | deepseek-v2.5 | 1306 | 236 (21) | MoE | 590 | 295 | 147.5 | Multi-GPU |
| 81 | athene-70b-0725 | 1305 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 82 | olmo-3-32b-think | 1305 | 32.2 | Dense | 80.5 | 40.2 | 20.1 | H100 SXM (FP8) |
| 83 | mistral-large-2411 | 1304 | 123 | Dense | 307.5 | 153.8 | 76.9 | B200 SXM (FP8) |
| 84 | gemma-3-4b-it | 1302 | 4.3 | Dense | 10.8 | 5.4 | 2.7 | H100 SXM (FP8) |
| 85 | mistral-small-3.1-24b-instruct-2503 | 1302 | 24 | Dense | 60 | 30 | 15 | H100 SXM (FP8) |
| 86 | qwen2.5-72b-instruct | 1302 | 72 | Dense | 180 | 90 | 45 | RTX PRO 6000 (FP8) |
| 87 | llama-3.1-nemotron-70b-instruct | 1298 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 88 | llama-3.1-70b-instruct | 1292 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 89 | jamba-1.5-large | 1288 | 398 (94) | MoE | 995 | 497.5 | 248.8 | Multi-GPU |
| 90 | gemma-2-27b-it | 1287 | 27 | Dense | 67.5 | 33.8 | 16.9 | H100 SXM (FP8) |
| 91 | ibm-granite-h-small | 1286 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 92 | llama-3.1-tulu-3-70b | 1286 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 93 | llama-3.1-nemotron-51b-instruct | 1285 | 51 | Dense | 127.5 | 63.8 | 31.9 | H100 SXM (FP8) |
| 94 | olmo-3.1-32b-think | 1285 | 32.2 | Dense | 80.5 | 40.2 | 20.1 | H100 SXM (FP8) |
| 95 | gemma-2-9b-it-simpo | 1278 | 9 | Dense | 22.5 | 11.2 | 5.6 | H100 SXM (FP8) |
| 96 | nemotron-4-340b-instruct | 1276 | 340 | Dense | 850 | 425 | 212.5 | Multi-GPU |
| 97 | command-r-plus-08-2024 | 1275 | 104 | Dense | 260 | 130 | 65 | H200 SXM (FP8) |
| 98 | llama-3-70b-instruct | 1275 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 99 | mistral-small-24b-instruct-2501 | 1273 | 24 | Dense | 60 | 30 | 15 | H100 SXM (FP8) |
| 100 | qwen2.5-coder-32b-instruct | 1270 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 101 | c4ai-aya-expanse-32b | 1266 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 102 | gemma-2-9b-it | 1265 | 9 | Dense | 22.5 | 11.2 | 5.6 | H100 SXM (FP8) |
| 103 | deepseek-coder-v2 | 1263 | 236 (21) | MoE | 590 | 295 | 147.5 | Multi-GPU |
| 104 | command-r-plus | 1261 | 104 | Dense | 260 | 130 | 65 | H200 SXM (FP8) |
| 105 | qwen2-72b-instruct | 1261 | 72 | Dense | 180 | 90 | 45 | RTX PRO 6000 (FP8) |
| 106 | phi-4 | 1255 | 14 | Dense | 35 | 17.5 | 8.8 | H100 SXM (FP8) |
| 107 | olmo-2-0325-32b-instruct | 1251 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 108 | command-r-08-2024 | 1249 | 35 | Dense | 87.5 | 43.8 | 21.9 | H100 SXM (FP8) |
| 109 | jamba-1.5-mini | 1238 | 52 (12) | MoE | 130 | 65 | 32.5 | H100 SXM (FP8) |
| 110 | ministral-8b-2410 | 1236 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 111 | qwen1.5-110b-chat | 1233 | 110 | Dense | 275 | 137.5 | 68.8 | H200 SXM (FP8) |
| 112 | qwen1.5-72b-chat | 1232 | 72 | Dense | 180 | 90 | 45 | RTX PRO 6000 (FP8) |
| 113 | mixtral-8x22b-instruct-v0.1 | 1228 | 141 (39) | MoE | 352.5 | 176.2 | 88.1 | B200 SXM (FP8) |
| 114 | command-r | 1226 | 35 | Dense | 87.5 | 43.8 | 21.9 | H100 SXM (FP8) |
| 115 | llama-3-8b-instruct | 1222 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 116 | c4ai-aya-expanse-8b | 1222 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 117 | llama-3.1-tulu-3-8b | 1220 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 118 | yi-1.5-34b-chat | 1212 | 34 | Dense | 85 | 42.5 | 21.2 | H100 SXM (FP8) |
| 119 | zephyr-orpo-141b-A35b-v0.1 | 1211 | 141 (35) | MoE | 352.5 | 176.2 | 88.1 | B200 SXM (FP8) |
| 120 | llama-3.1-8b-instruct | 1211 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 121 | granite-3.1-8b-instruct | 1207 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 122 | qwen1.5-32b-chat | 1203 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 123 | gemma-2-2b-it | 1198 | 2 | Dense | 5 | 2.5 | 1.2 | H100 SXM (FP8) |
| 124 | phi-3-medium-4k-instruct | 1197 | 14 | Dense | 35 | 17.5 | 8.8 | H100 SXM (FP8) |
| 125 | mixtral-8x7b-instruct-v0.1 | 1196 | 46.7 (12.9) | MoE | 116.8 | 58.4 | 29.2 | H100 SXM (FP8) |
| 126 | dbrx-instruct-preview | 1194 | 132 (36) | MoE | 330 | 165 | 82.5 | B200 SXM (FP8) |
| 127 | internlm2_5-20b-chat | 1190 | 20 | Dense | 50 | 25 | 12.5 | H100 SXM (FP8) |
| 128 | qwen1.5-14b-chat | 1190 | 14 | Dense | 35 | 17.5 | 8.8 | H100 SXM (FP8) |
| 129 | wizardlm-70b | 1183 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 130 | deepseek-llm-67b-chat | 1183 | 67 | Dense | 167.5 | 83.8 | 41.9 | RTX PRO 6000 (FP8) |
| 131 | yi-34b-chat | 1183 | 34 | Dense | 85 | 42.5 | 21.2 | H100 SXM (FP8) |
| 132 | openchat-3.5-0106 | 1181 | ? | ? | ? | ? | ? | ? |
| 133 | openchat-3.5 | 1181 | ? | ? | ? | ? | ? | ? |
| 134 | granite-3.0-8b-instruct | 1181 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 135 | gemma-1.1-7b-it | 1179 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 136 | snowflake-arctic-instruct | 1178 | ? | ? | ? | ? | ? | ? |
| 137 | granite-3.1-2b-instruct | 1178 | 2 | Dense | 5 | 2.5 | 1.2 | H100 SXM (FP8) |
| 138 | tulu-2-dpo-70b | 1177 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 139 | openhermes-2.5-mistral-7b | 1174 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 140 | vicuna-33b | 1171 | 33 | Dense | 82.5 | 41.2 | 20.6 | H100 SXM (FP8) |
| 141 | starling-lm-7b-beta | 1170 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 142 | phi-3-small-8k-instruct | 1170 | 7.4 | Dense | 18.5 | 9.2 | 4.6 | H100 SXM (FP8) |
| 143 | llama-2-70b-chat | 1169 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 144 | starling-lm-7b-alpha | 1166 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 145 | llama-3.2-3b-instruct | 1165 | 3 | Dense | 7.5 | 3.8 | 1.9 | H100 SXM (FP8) |
| 146 | nous-hermes-2-mixtral-8x7b-dpo | 1163 | 44.8 (12.6) | MoE | 112 | 56 | 28 | H100 SXM (FP8) |
| 147 | qwq-32b-preview | 1156 | 32.8 | Dense | 82 | 41 | 20.5 | H100 SXM (FP8) |
| 148 | granite-3.0-2b-instruct | 1155 | 2 | Dense | 5 | 2.5 | 1.2 | H100 SXM (FP8) |
| 149 | llama2-70b-steerlm-chat | 1154 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 150 | solar-10.7b-instruct-v1.0 | 1151 | 10.7 | Dense | 26.8 | 13.4 | 6.7 | H100 SXM (FP8) |
| 151 | dolphin-2.2.1-mistral-7b | 1151 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 152 | mpt-30b-chat | 1149 | 30 | Dense | 75 | 37.5 | 18.8 | H100 SXM (FP8) |
| 153 | mistral-7b-instruct-v0.2 | 1148 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 154 | wizardlm-13b | 1148 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 155 | falcon-180b-chat | 1146 | 180 | Dense | 450 | 225 | 112.5 | B300 SXM (FP8) |
| 156 | qwen1.5-7b-chat | 1142 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 157 | phi-3-mini-4k-instruct-june-2024 | 1142 | 3.8 | Dense | 9.5 | 4.8 | 2.4 | H100 SXM (FP8) |
| 158 | llama-2-13b-chat | 1140 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 159 | vicuna-13b | 1140 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 160 | qwen-14b-chat | 1137 | 14 | Dense | 35 | 17.5 | 8.8 | H100 SXM (FP8) |
| 161 | codellama-34b-instruct | 1135 | 34 | Dense | 85 | 42.5 | 21.2 | H100 SXM (FP8) |
| 162 | gemma-7b-it | 1135 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 163 | zephyr-7b-beta | 1130 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 164 | phi-3-mini-128k-instruct | 1128 | 3.8 | Dense | 9.5 | 4.8 | 2.4 | H100 SXM (FP8) |
| 165 | phi-3-mini-4k-instruct | 1127 | 3.8 | Dense | 9.5 | 4.8 | 2.4 | H100 SXM (FP8) |
| 166 | guanaco-33b | 1126 | 33 | Dense | 82.5 | 41.2 | 20.6 | H100 SXM (FP8) |
| 167 | zephyr-7b-alpha | 1126 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 168 | stripedhyena-nous-7b | 1120 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 169 | codellama-70b-instruct | 1118 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 170 | vicuna-7b | 1113 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 171 | gemma-1.1-2b-it | 1113 | 2 | Dense | 5 | 2.5 | 1.2 | H100 SXM (FP8) |
| 172 | smollm2-1.7b-instruct | 1113 | 1.7 | Dense | 4.2 | 2.1 | 1.1 | H100 SXM (FP8) |
| 173 | llama-3.2-1b-instruct | 1110 | 1 | Dense | 2.5 | 1.2 | 0.6 | H100 SXM (FP8) |
| 174 | mistral-7b-instruct | 1108 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 175 | llama-2-7b-chat | 1107 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 176 | gemma-2b-it | 1091 | 2 | Dense | 5 | 2.5 | 1.2 | H100 SXM (FP8) |
| 177 | qwen1.5-4b-chat | 1089 | 4 | Dense | 10 | 5 | 2.5 | H100 SXM (FP8) |
| 178 | olmo-7b-instruct | 1073 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 179 | koala-13b | 1069 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 180 | alpaca-13b | 1066 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 181 | gpt4all-13b-snoozy | 1065 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 182 | mpt-7b-chat | 1061 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 183 | chatglm3-6b | 1055 | 6 | Dense | 15 | 7.5 | 3.8 | H100 SXM (FP8) |
| 184 | RWKV-4-Raven-14B | 1040 | 14 | Dense | 35 | 17.5 | 8.8 | H100 SXM (FP8) |
| 185 | chatglm2-6b | 1023 | 6 | Dense | 15 | 7.5 | 3.8 | H100 SXM (FP8) |
| 186 | oasst-pythia-12b | 1021 | 12 | Dense | 30 | 15 | 7.5 | H100 SXM (FP8) |
| 187 | chatglm-6b | 994 | 6 | Dense | 15 | 7.5 | 3.8 | H100 SXM (FP8) |
| 188 | fastchat-t5-3b | 990 | 3 | Dense | 7.5 | 3.8 | 1.9 | H100 SXM (FP8) |
| 189 | dolly-v2-12b | 979 | 12 | Dense | 30 | 15 | 7.5 | H100 SXM (FP8) |
| 190 | llama-13b | 971 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 191 | stablelm-tuned-alpha-7b | 951 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |

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
