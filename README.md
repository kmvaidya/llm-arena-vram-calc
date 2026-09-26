# LLM Arena VRAM Calculator

Enriches the [Arena.ai](https://arena.ai/leaderboard/text?license=open-source) open-source LLM leaderboard with parameter counts and VRAM estimates for single-GPU deployment feasibility.

Most LLM leaderboards rank models by quality but ignore deployment constraints. This tool answers: *"What's the best model I can actually run on my hardware?"* by cross-referencing Arena rankings with VRAM requirements across precisions.

> **Last updated:** 2026-09-26 10:39 UTC | **Models:** 225 | **Resolved:** 171 (76.0%)

> **Warning:** AA data may be stale (RSC fetch failed, using cached data).

## Best Model Per GPU

Highest-ranked Arena model that fits on each single GPU (includes 25% serving overhead for KV cache, activations, and framework).

### BF16 (Full Precision)

| GPU | VRAM | Best Model | Arena Rank | Params | Arch | Serving VRAM |
|-----|------|------------|------------|--------|------|--------------|
| H100 SXM | 80 GB | deepseek-v4-pro | #11 | 1.6B | Dense | 4.0 GB |
| RTX PRO 6000 | 96 GB | deepseek-v4-pro | #11 | 1.6B | Dense | 4.0 GB |
| H200 SXM | 141 GB | deepseek-v4-pro | #11 | 1.6B | Dense | 4.0 GB |
| B200 SXM | 180 GB | deepseek-v4-pro | #11 | 1.6B | Dense | 4.0 GB |
| B300 SXM | 288 GB | deepseek-v4-pro | #11 | 1.6B | Dense | 4.0 GB |

### FP8 (8-bit)

| GPU | VRAM | Best Model | Arena Rank | Params | Arch | Serving VRAM |
|-----|------|------------|------------|--------|------|--------------|
| H100 SXM | 80 GB | deepseek-v4-pro | #11 | 1.6B | Dense | 2.0 GB |
| RTX PRO 6000 | 96 GB | deepseek-v4-pro | #11 | 1.6B | Dense | 2.0 GB |
| H200 SXM | 141 GB | deepseek-v4-pro | #11 | 1.6B | Dense | 2.0 GB |
| B200 SXM | 180 GB | deepseek-v4-pro | #11 | 1.6B | Dense | 2.0 GB |
| B300 SXM | 288 GB | deepseek-v4-pro | #11 | 1.6B | Dense | 2.0 GB |

### INT4 (4-bit)

| GPU | VRAM | Best Model | Arena Rank | Params | Arch | Serving VRAM |
|-----|------|------------|------------|--------|------|--------------|
| H100 SXM | 80 GB | deepseek-v4-pro | #11 | 1.6B | Dense | 1.0 GB |
| RTX PRO 6000 | 96 GB | deepseek-v4-pro | #11 | 1.6B | Dense | 1.0 GB |
| H200 SXM | 141 GB | deepseek-v4-pro | #11 | 1.6B | Dense | 1.0 GB |
| B200 SXM | 180 GB | deepseek-v4-pro | #11 | 1.6B | Dense | 1.0 GB |
| B300 SXM | 288 GB | deepseek-v4-pro | #11 | 1.6B | Dense | 1.0 GB |

## Full Leaderboard

| Rank | Model | Score | Params (B) | Arch | VRAM BF16 | VRAM FP8 | VRAM INT4 | Fits on |
|------|-------|-------|------------|------|-----------|----------|-----------|---------|
| 1 | kimi-k3-max | 1487 | ? | ? | ? | ? | ? | ? |
| 2 | mimo-v2.6-pro | 1479 | ? | ? | ? | ? | ? | ? |
| 3 | glm-5.3-max | 1479 | ? | ? | ? | ? | ? | ? |
| 4 | deepseek-v4.1-flash-max | 1476 | ? | ? | ? | ? | ? | ? |
| 5 | glm-5.2-max | 1475 | ? | ? | ? | ? | ? | ? |
| 6 | glm-5.3-flash | 1474 | ? | ? | ? | ? | ? | ? |
| 7 | mimo-v2.5-pro | 1467 | ? | ? | ? | ? | ? | ? |
| 8 | glm-5.1 | 1465 | ? | ? | ? | ? | ? | ? |
| 9 | deepseek-v4-pro-high-20260813 | 1464 | ? | ? | ? | ? | ? | ? |
| 10 | kimi-k2.6 | 1460 | 1000 (32) | MoE | 2500 | 1250 | 625 | Multi-GPU |
| 11 | deepseek-v4-pro | 1457 | 1.6 (49) | Dense | 4 | 2 | 1 | H100 SXM (FP8) |
| 12 | glm-5 | 1457 | 744 (40) | MoE | 1860 | 930 | 465 | Multi-GPU |
| 13 | hy3 | 1456 | 299 (21) | MoE | 747.5 | 373.8 | 186.9 | Multi-GPU |
| 14 | deepseek-v4-pro-high-preview | 1454 | ? | ? | ? | ? | ? | ? |
| 15 | mimo-v2.6-flash | 1453 | ? | ? | ? | ? | ? | ? |
| 16 | gemma-4-31b | 1452 | 31 | Dense | 77.5 | 38.8 | 19.4 | H100 SXM (FP8) |
| 17 | kimi-k2.5-thinking | 1450 | 1000 (32) | MoE | 2500 | 1250 | 625 | Multi-GPU |
| 18 | inkling | 1442 | 975 (41) | MoE | 2437.5 | 1218.8 | 609.4 | Multi-GPU |
| 19 | qwen3.5-397b-a17b | 1441 | 397 (17) | MoE | 992.5 | 496.2 | 248.1 | Multi-GPU |
| 20 | glm-4.7 | 1441 | ? | ? | ? | ? | ? | ? |
| 21 | minimax-m3 | 1440 | 428 (23) | MoE | 1070 | 535 | 267.5 | Multi-GPU |
| 22 | deepseek-v4-flash-high-preview | 1438 | ? | ? | ? | ? | ? | ? |
| 23 | qwen3.8-27b | 1437 | 27 | Dense | 67.5 | 33.8 | 16.9 | H100 SXM (FP8) |
| 24 | gemma-4-26b-a4b | 1437 | 26 (4) | MoE | 65 | 32.5 | 16.2 | H100 SXM (FP8) |
| 25 | deepseek-v4-flash | 1436 | 284 (13) | MoE | 710 | 355 | 177.5 | Multi-GPU |
| 26 | mimo-v2.5 | 1433 | ? | ? | ? | ? | ? | ? |
| 27 | kimi-k2.5-instant | 1430 | 1000 (32) | MoE | 2500 | 1250 | 625 | Multi-GPU |
| 28 | kimi-k2-thinking-turbo | 1430 | 1000 (32) | MoE | 2500 | 1250 | 625 | Multi-GPU |
| 29 | mistral-medium-3.5 | 1426 | ? | ? | ? | ? | ? | ? |
| 30 | nvidia-nemotron-3-ultra-550b-a55b-nvfp4 | 1425 | 550 (55) | MoE | 1375 | 687.5 | 343.8 | Multi-GPU |
| 31 | deepseek-v3.2-exp-thinking | 1424 | ? | ? | ? | ? | ? | ? |
| 32 | deepseek-v3.2 | 1424 | ? | ? | ? | ? | ? | ? |
| 33 | muse-glimmer | 1424 | 30 | Dense | 75 | 37.5 | 18.8 | H100 SXM (FP8) |
| 34 | glm-4.6 | 1424 | ? | ? | ? | ? | ? | ? |
| 35 | deepseek-v3.2-thinking | 1422 | ? | ? | ? | ? | ? | ? |
| 36 | qwen3-235b-a22b-instruct-2507 | 1422 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |
| 37 | deepseek-v3.2-exp | 1421 | ? | ? | ? | ? | ? | ? |
| 38 | deepseek-r1-0528 | 1421 | ? | ? | ? | ? | ? | ? |
| 39 | kimi-k2-0905-preview | 1418 | 1000 (32) | MoE | 2500 | 1250 | 625 | Multi-GPU |
| 40 | kimi-k2-0711-preview | 1418 | 1000 (32) | MoE | 2500 | 1250 | 625 | Multi-GPU |
| 41 | deepseek-v3.1-terminus-thinking | 1417 | ? | ? | ? | ? | ? | ? |
| 42 | deepseek-v3.1 | 1417 | ? | ? | ? | ? | ? | ? |
| 43 | qwen3.5-122b-a10b | 1416 | 122 (10) | MoE | 305 | 152.5 | 76.2 | B200 SXM (FP8) |
| 44 | deepseek-v3.1-thinking | 1415 | ? | ? | ? | ? | ? | ? |
| 45 | minimax-m2.7 | 1414 | ? | ? | ? | ? | ? | ? |
| 46 | deepseek-v3.1-terminus | 1414 | ? | ? | ? | ? | ? | ? |
| 47 | qwen3-vl-235b-a22b-instruct | 1413 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |
| 48 | mistral-large-3 | 1413 | 675 (41) | MoE | 1687.5 | 843.8 | 421.9 | Multi-GPU |
| 49 | glm-4.5 | 1411 | 355 (32) | MoE | 887.5 | 443.8 | 221.9 | Multi-GPU |
| 50 | hunyuan-hy3-preview | 1409 | ? | ? | ? | ? | ? | ? |

<details><summary>Show remaining 175 models</summary>

| Rank | Model | Score | Params (B) | Arch | VRAM BF16 | VRAM FP8 | VRAM INT4 | Fits on |
|------|-------|-------|------------|------|-----------|----------|-----------|---------|
| 51 | qwen3.5-27b | 1408 | 27 | Dense | 67.5 | 33.8 | 16.9 | H100 SXM (FP8) |
| 52 | Inkling Small | 1405 | 266 (12) | MoE | 665 | 332.5 | 166.2 | Multi-GPU |
| 53 | qwen3-235b-a22b-no-thinking | 1402 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |
| 54 | longcat-flash-chat | 1401 | 560 (27) | MoE | 1400 | 700 | 350 | Multi-GPU |
| 55 | qwen3-235b-a22b-thinking-2507 | 1400 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |
| 56 | qwen3-next-80b-a3b-instruct | 1399 | 80 (3) | MoE | 200 | 100 | 50 | H200 SXM (FP8) |
| 57 | deepseek-r1 | 1398 | 685 (37) | MoE | 1712.5 | 856.2 | 428.1 | Multi-GPU |
| 58 | deepseek-v3-0324 | 1395 | 671 (37) | MoE | 1677.5 | 838.8 | 419.4 | Multi-GPU |
| 59 | qwen3-vl-235b-a22b-thinking | 1394 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |
| 60 | qwen3.5-35b-a3b | 1394 | 35 (3) | MoE | 87.5 | 43.8 | 21.9 | H100 SXM (FP8) |
| 61 | step-3.5-flash | 1393 | ? | ? | ? | ? | ? | ? |
| 62 | mimo-v2-flash (non-thinking) | 1391 | ? | ? | ? | ? | ? | ? |
| 63 | minimax-m2.5 | 1390 | ? | ? | ? | ? | ? | ? |
| 64 | qwen3-coder-480b-a35b-instruct | 1387 | 480 (35) | MoE | 1200 | 600 | 300 | Multi-GPU |
| 65 | mimo-v2-flash (thinking) | 1386 | ? | ? | ? | ? | ? | ? |
| 66 | minimax-m2.1-preview | 1384 | ? | ? | ? | ? | ? | ? |
| 67 | qwen3-30b-a3b-instruct-2507 | 1382 | 30 (3) | MoE | 75 | 37.5 | 18.8 | H100 SXM (FP8) |
| 68 | trinity-large-preview | 1378 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 69 | glm-4.6v | 1378 | ? | ? | ? | ? | ? | ? |
| 70 | qwen3-235b-a22b | 1375 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |
| 71 | glm-4.5-air | 1373 | ? | ? | ? | ? | ? | ? |
| 72 | qwen3-next-80b-a3b-thinking | 1369 | 80 (3) | MoE | 200 | 100 | 50 | H200 SXM (FP8) |
| 73 | trinity-large-thinking | 1367 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 74 | gemma-3-27b-it | 1365 | 27 | Dense | 67.5 | 33.8 | 16.9 | H100 SXM (FP8) |
| 75 | glm-4.7-flash | 1365 | ? | ? | ? | ? | ? | ? |
| 76 | minimax-m1 | 1363 | ? | ? | ? | ? | ? | ? |
| 77 | nvidia-nemotron-3-super-120b-a12b | 1360 | 120 (12) | MoE | 300 | 150 | 75 | B200 SXM (FP8) |
| 78 | deepseek-v3 | 1358 | 671 (37) | MoE | 1677.5 | 838.8 | 419.4 | Multi-GPU |
| 79 | mistral-small-2506 | 1356 | ? | ? | ? | ? | ? | ? |
| 80 | intellect-3 | 1355 | 107 (12) | MoE | 267.5 | 133.8 | 66.9 | H200 SXM (FP8) |
| 81 | command-a-03-2025 | 1353 | ? | ? | ? | ? | ? | ? |
| 82 | glm-4.5v | 1352 | ? | ? | ? | ? | ? | ? |
| 83 | gpt-oss-120b | 1351 | 117 (5.1) | MoE | 292.5 | 146.2 | 73.1 | B200 SXM (FP8) |
| 84 | step-3 | 1349 | ? | ? | ? | ? | ? | ? |
| 85 | llama-3.1-nemotron-ultra-253b-v1 | 1347 | 253 | Dense | 632.5 | 316.2 | 158.1 | Multi-GPU |
| 86 | qwen3-32b | 1346 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 87 | nvidia-nemotron-3.5-lightning-30b-a3b-nvfp4 | 1346 | 30 (3) | MoE | 75 | 37.5 | 18.8 | H100 SXM (FP8) |
| 88 | ling-flash-2.0 | 1343 | ? | ? | ? | ? | ? | ? |
| 89 | minimax-m2 | 1343 | 230 (10) | MoE | 575 | 287.5 | 143.8 | B300 SXM (FP8) |
| 90 | nvidia-llama-3.3-nemotron-super-49b-v1.5 | 1342 | 49 | Dense | 122.5 | 61.2 | 30.6 | H100 SXM (FP8) |
| 91 | gemma-3-12b-it | 1341 | 12 | Dense | 30 | 15 | 7.5 | H100 SXM (FP8) |
| 92 | granite-4.2-30b | 1341 | 30 | Dense | 75 | 37.5 | 18.8 | H100 SXM (FP8) |
| 93 | qwq-32b | 1335 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 94 | llama-3.1-405b-instruct-bf16 | 1335 | 405 | Dense | 1012.5 | 506.2 | 253.1 | Multi-GPU |
| 95 | llama-3.1-405b-instruct-fp8 | 1333 | 405 | Dense | 1012.5 | 506.2 | 253.1 | Multi-GPU |
| 96 | olmo-3.1-32b-instruct | 1329 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 97 | llama-3.3-nemotron-49b-super-v1 | 1327 | 49 | Dense | 122.5 | 61.2 | 30.6 | H100 SXM (FP8) |
| 98 | llama-4-maverick-17b-128e-instruct | 1326 | 400 (17) | MoE | 1000 | 500 | 250 | Multi-GPU |
| 99 | qwen3-30b-a3b | 1326 | 30 (3) | MoE | 75 | 37.5 | 18.8 | H100 SXM (FP8) |
| 100 | deepseek-v2.5-1210 | 1323 | ? | ? | ? | ? | ? | ? |
| 101 | ring-flash-2.0 | 1322 | ? | ? | ? | ? | ? | ? |
| 102 | molmo-2-8b | 1321 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 103 | llama-4-scout-17b-16e-instruct | 1321 | 109 (17) | MoE | 272.5 | 136.2 | 68.1 | H200 SXM (FP8) |
| 104 | qwen-max-0919 | 1318 | ? | ? | ? | ? | ? | ? |
| 105 | llama-3.3-70b-instruct | 1317 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 106 | gpt-oss-20b | 1317 | 21 (3.6) | MoE | 52.5 | 26.2 | 13.1 | H100 SXM (FP8) |
| 107 | gemma-3n-e4b-it | 1317 | 8.4 (4) | MoE | 21 | 10.5 | 5.2 | H100 SXM (FP8) |
| 108 | mistral-large-2407 | 1314 | 123 | Dense | 307.5 | 153.8 | 76.9 | B200 SXM (FP8) |
| 109 | athene-v2-chat | 1314 | 72 | Dense | 180 | 90 | 45 | RTX PRO 6000 (FP8) |
| 110 | nvidia-nemotron-3-nano-30b-a3b-bf16 | 1313 | 30 (3) | MoE | 75 | 37.5 | 18.8 | H100 SXM (FP8) |
| 111 | deepseek-v2.5 | 1307 | ? | ? | ? | ? | ? | ? |
| 112 | athene-70b-0725 | 1306 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 113 | olmo-3-32b-think | 1306 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 114 | mistral-large-2411 | 1305 | ? | ? | ? | ? | ? | ? |
| 115 | granite-4.1-8b | 1304 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 116 | gemma-3-4b-it | 1303 | 4 | Dense | 10 | 5 | 2.5 | H100 SXM (FP8) |
| 117 | mistral-small-3.1-24b-instruct-2503 | 1303 | 24 | Dense | 60 | 30 | 15 | H100 SXM (FP8) |
| 118 | qwen2.5-72b-instruct | 1302 | 72 | Dense | 180 | 90 | 45 | RTX PRO 6000 (FP8) |
| 119 | llama-3.1-nemotron-70b-instruct | 1298 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 120 | llama-3.1-70b-instruct | 1293 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 121 | granite-4.2-3b | 1291 | 3 | Dense | 7.5 | 3.8 | 1.9 | H100 SXM (FP8) |
| 122 | gemma-2-27b-it | 1289 | 27 | Dense | 67.5 | 33.8 | 16.9 | H100 SXM (FP8) |
| 123 | jamba-1.5-large | 1289 | ? | ? | ? | ? | ? | ? |
| 124 | granite-4.2-8b | 1289 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 125 | ibm-granite-h-small | 1286 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 126 | llama-3.1-nemotron-51b-instruct | 1286 | 51 | Dense | 127.5 | 63.8 | 31.9 | H100 SXM (FP8) |
| 127 | olmo-3.1-32b-think | 1286 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 128 | llama-3.1-tulu-3-70b | 1286 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 129 | gemma-2-9b-it-simpo | 1280 | 9 | Dense | 22.5 | 11.2 | 5.6 | H100 SXM (FP8) |
| 130 | nemotron-4-340b-instruct | 1277 | 340 | Dense | 850 | 425 | 212.5 | Multi-GPU |
| 131 | llama-3-70b-instruct | 1276 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 132 | command-r-plus-08-2024 | 1276 | 104 | Dense | 260 | 130 | 65 | H200 SXM (FP8) |
| 133 | mistral-small-24b-instruct-2501 | 1274 | 24 | Dense | 60 | 30 | 15 | H100 SXM (FP8) |
| 134 | qwen2.5-coder-32b-instruct | 1270 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 135 | c4ai-aya-expanse-32b | 1267 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 136 | gemma-2-9b-it | 1267 | 9 | Dense | 22.5 | 11.2 | 5.6 | H100 SXM (FP8) |
| 137 | deepseek-coder-v2 | 1265 | 236 (21) | MoE | 590 | 295 | 147.5 | Multi-GPU |
| 138 | qwen2-72b-instruct | 1261 | 72 | Dense | 180 | 90 | 45 | RTX PRO 6000 (FP8) |
| 139 | command-r-plus | 1261 | ? | ? | ? | ? | ? | ? |
| 140 | phi-4 | 1256 | 14 | Dense | 35 | 17.5 | 8.8 | H100 SXM (FP8) |
| 141 | olmo-2-0325-32b-instruct | 1251 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 142 | command-r-08-2024 | 1250 | 35 | Dense | 87.5 | 43.8 | 21.9 | H100 SXM (FP8) |
| 143 | jamba-1.5-mini | 1239 | ? | ? | ? | ? | ? | ? |
| 144 | ministral-8b-2410 | 1237 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 145 | qwen1.5-110b-chat | 1234 | 110 | Dense | 275 | 137.5 | 68.8 | H200 SXM (FP8) |
| 146 | qwen1.5-72b-chat | 1233 | 72 | Dense | 180 | 90 | 45 | RTX PRO 6000 (FP8) |
| 147 | mixtral-8x22b-instruct-v0.1 | 1229 | 140.8 (39.6) | MoE | 352 | 176 | 88 | B200 SXM (FP8) |
| 148 | command-r | 1226 | ? | ? | ? | ? | ? | ? |
| 149 | llama-3-8b-instruct | 1223 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 150 | c4ai-aya-expanse-8b | 1223 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 151 | llama-3.1-tulu-3-8b | 1220 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 152 | zephyr-orpo-141b-A35b-v0.1 | 1212 | 141 (35) | MoE | 352.5 | 176.2 | 88.1 | B200 SXM (FP8) |
| 153 | yi-1.5-34b-chat | 1212 | 34 | Dense | 85 | 42.5 | 21.2 | H100 SXM (FP8) |
| 154 | llama-3.1-8b-instruct | 1211 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 155 | granite-3.1-8b-instruct | 1208 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 156 | qwen1.5-32b-chat | 1203 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 157 | gemma-2-2b-it | 1200 | 2 | Dense | 5 | 2.5 | 1.2 | H100 SXM (FP8) |
| 158 | phi-3-medium-4k-instruct | 1197 | 14 | Dense | 35 | 17.5 | 8.8 | H100 SXM (FP8) |
| 159 | mixtral-8x7b-instruct-v0.1 | 1197 | 44.8 (12.6) | MoE | 112 | 56 | 28 | H100 SXM (FP8) |
| 160 | dbrx-instruct-preview | 1195 | ? | ? | ? | ? | ? | ? |
| 161 | qwen1.5-14b-chat | 1191 | 14 | Dense | 35 | 17.5 | 8.8 | H100 SXM (FP8) |
| 162 | internlm2_5-20b-chat | 1190 | 20 | Dense | 50 | 25 | 12.5 | H100 SXM (FP8) |
| 163 | deepseek-llm-67b-chat | 1184 | 67 | Dense | 167.5 | 83.8 | 41.9 | RTX PRO 6000 (FP8) |
| 164 | wizardlm-70b | 1184 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 165 | yi-34b-chat | 1183 | 34 | Dense | 85 | 42.5 | 21.2 | H100 SXM (FP8) |
| 166 | granite-3.0-8b-instruct | 1183 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 167 | openchat-3.5 | 1183 | ? | ? | ? | ? | ? | ? |
| 168 | openchat-3.5-0106 | 1182 | ? | ? | ? | ? | ? | ? |
| 169 | gemma-1.1-7b-it | 1182 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 170 | snowflake-arctic-instruct | 1180 | ? | ? | ? | ? | ? | ? |
| 171 | granite-3.1-2b-instruct | 1178 | 2 | Dense | 5 | 2.5 | 1.2 | H100 SXM (FP8) |
| 172 | tulu-2-dpo-70b | 1177 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 173 | openhermes-2.5-mistral-7b | 1175 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 174 | vicuna-33b | 1172 | 33 | Dense | 82.5 | 41.2 | 20.6 | H100 SXM (FP8) |
| 175 | phi-3-small-8k-instruct | 1171 | 7.4 | Dense | 18.5 | 9.2 | 4.6 | H100 SXM (FP8) |
| 176 | starling-lm-7b-beta | 1170 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 177 | llama-2-70b-chat | 1170 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 178 | starling-lm-7b-alpha | 1167 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 179 | llama-3.2-3b-instruct | 1166 | 3 | Dense | 7.5 | 3.8 | 1.9 | H100 SXM (FP8) |
| 180 | nous-hermes-2-mixtral-8x7b-dpo | 1164 | 44.8 (12.6) | MoE | 112 | 56 | 28 | H100 SXM (FP8) |
| 181 | granite-3.0-2b-instruct | 1156 | 2 | Dense | 5 | 2.5 | 1.2 | H100 SXM (FP8) |
| 182 | llama2-70b-steerlm-chat | 1154 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 183 | qwq-32b-preview | 1154 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 184 | solar-10.7b-instruct-v1.0 | 1152 | 10.7 | Dense | 26.8 | 13.4 | 6.7 | H100 SXM (FP8) |
| 185 | dolphin-2.2.1-mistral-7b | 1152 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 186 | mpt-30b-chat | 1150 | 30 | Dense | 75 | 37.5 | 18.8 | H100 SXM (FP8) |
| 187 | wizardlm-13b | 1149 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 188 | mistral-7b-instruct-v0.2 | 1149 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 189 | falcon-180b-chat | 1148 | 180 | Dense | 450 | 225 | 112.5 | B300 SXM (FP8) |
| 190 | qwen1.5-7b-chat | 1143 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 191 | phi-3-mini-4k-instruct-june-2024 | 1143 | 3.8 | Dense | 9.5 | 4.8 | 2.4 | H100 SXM (FP8) |
| 192 | vicuna-13b | 1141 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 193 | llama-2-13b-chat | 1141 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 194 | qwen-14b-chat | 1139 | 14 | Dense | 35 | 17.5 | 8.8 | H100 SXM (FP8) |
| 195 | gemma-7b-it | 1137 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 196 | codellama-34b-instruct | 1136 | 34 | Dense | 85 | 42.5 | 21.2 | H100 SXM (FP8) |
| 197 | zephyr-7b-beta | 1130 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 198 | phi-3-mini-128k-instruct | 1129 | 3.8 | Dense | 9.5 | 4.8 | 2.4 | H100 SXM (FP8) |
| 199 | phi-3-mini-4k-instruct | 1128 | 3.8 | Dense | 9.5 | 4.8 | 2.4 | H100 SXM (FP8) |
| 200 | guanaco-33b | 1127 | 33 | Dense | 82.5 | 41.2 | 20.6 | H100 SXM (FP8) |
| 201 | zephyr-7b-alpha | 1126 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 202 | stripedhyena-nous-7b | 1121 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 203 | codellama-70b-instruct | 1119 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 204 | gemma-1.1-2b-it | 1116 | 2 | Dense | 5 | 2.5 | 1.2 | H100 SXM (FP8) |
| 205 | vicuna-7b | 1115 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 206 | smollm2-1.7b-instruct | 1114 | 1.7 | Dense | 4.2 | 2.1 | 1.1 | H100 SXM (FP8) |
| 207 | llama-3.2-1b-instruct | 1111 | 1 | Dense | 2.5 | 1.2 | 0.6 | H100 SXM (FP8) |
| 208 | mistral-7b-instruct | 1110 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 209 | llama-2-7b-chat | 1107 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 210 | gemma-2b-it | 1093 | 2 | Dense | 5 | 2.5 | 1.2 | H100 SXM (FP8) |
| 211 | qwen1.5-4b-chat | 1091 | 4 | Dense | 10 | 5 | 2.5 | H100 SXM (FP8) |
| 212 | olmo-7b-instruct | 1073 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 213 | koala-13b | 1070 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 214 | alpaca-13b | 1070 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 215 | gpt4all-13b-snoozy | 1067 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 216 | mpt-7b-chat | 1063 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 217 | chatglm3-6b | 1056 | 6 | Dense | 15 | 7.5 | 3.8 | H100 SXM (FP8) |
| 218 | RWKV-4-Raven-14B | 1042 | 14 | Dense | 35 | 17.5 | 8.8 | H100 SXM (FP8) |
| 219 | chatglm2-6b | 1024 | 6 | Dense | 15 | 7.5 | 3.8 | H100 SXM (FP8) |
| 220 | oasst-pythia-12b | 1023 | 12 | Dense | 30 | 15 | 7.5 | H100 SXM (FP8) |
| 221 | chatglm-6b | 995 | 6 | Dense | 15 | 7.5 | 3.8 | H100 SXM (FP8) |
| 222 | fastchat-t5-3b | 992 | 3 | Dense | 7.5 | 3.8 | 1.9 | H100 SXM (FP8) |
| 223 | dolly-v2-12b | 982 | 12 | Dense | 30 | 15 | 7.5 | H100 SXM (FP8) |
| 224 | llama-13b | 975 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 225 | stablelm-tuned-alpha-7b | 953 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |

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
