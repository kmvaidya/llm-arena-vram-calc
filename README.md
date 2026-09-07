# LLM Arena VRAM Calculator

Enriches the [Arena.ai](https://arena.ai/leaderboard/text?license=open-source) open-source LLM leaderboard with parameter counts and VRAM estimates for single-GPU deployment feasibility.

Most LLM leaderboards rank models by quality but ignore deployment constraints. This tool answers: *"What's the best model I can actually run on my hardware?"* by cross-referencing Arena rankings with VRAM requirements across precisions.

> **Last updated:** 2026-09-07 11:19 UTC | **Models:** 222 | **Resolved:** 171 (77.0%)

> **Warning:** AA data may be stale (RSC fetch failed, using cached data).

## Best Model Per GPU

Highest-ranked Arena model that fits on each single GPU (includes 25% serving overhead for KV cache, activations, and framework).

### BF16 (Full Precision)

| GPU | VRAM | Best Model | Arena Rank | Params | Arch | Serving VRAM |
|-----|------|------------|------------|--------|------|--------------|
| H100 SXM | 80 GB | deepseek-v4-pro | #9 | 1.6B | Dense | 4.0 GB |
| RTX PRO 6000 | 96 GB | deepseek-v4-pro | #9 | 1.6B | Dense | 4.0 GB |
| H200 SXM | 141 GB | deepseek-v4-pro | #9 | 1.6B | Dense | 4.0 GB |
| B200 SXM | 180 GB | deepseek-v4-pro | #9 | 1.6B | Dense | 4.0 GB |
| B300 SXM | 288 GB | deepseek-v4-pro | #9 | 1.6B | Dense | 4.0 GB |

### FP8 (8-bit)

| GPU | VRAM | Best Model | Arena Rank | Params | Arch | Serving VRAM |
|-----|------|------------|------------|--------|------|--------------|
| H100 SXM | 80 GB | deepseek-v4-pro | #9 | 1.6B | Dense | 2.0 GB |
| RTX PRO 6000 | 96 GB | deepseek-v4-pro | #9 | 1.6B | Dense | 2.0 GB |
| H200 SXM | 141 GB | deepseek-v4-pro | #9 | 1.6B | Dense | 2.0 GB |
| B200 SXM | 180 GB | deepseek-v4-pro | #9 | 1.6B | Dense | 2.0 GB |
| B300 SXM | 288 GB | deepseek-v4-pro | #9 | 1.6B | Dense | 2.0 GB |

### INT4 (4-bit)

| GPU | VRAM | Best Model | Arena Rank | Params | Arch | Serving VRAM |
|-----|------|------------|------------|--------|------|--------------|
| H100 SXM | 80 GB | deepseek-v4-pro | #9 | 1.6B | Dense | 1.0 GB |
| RTX PRO 6000 | 96 GB | deepseek-v4-pro | #9 | 1.6B | Dense | 1.0 GB |
| H200 SXM | 141 GB | deepseek-v4-pro | #9 | 1.6B | Dense | 1.0 GB |
| B200 SXM | 180 GB | deepseek-v4-pro | #9 | 1.6B | Dense | 1.0 GB |
| B300 SXM | 288 GB | deepseek-v4-pro | #9 | 1.6B | Dense | 1.0 GB |

## Full Leaderboard

| Rank | Model | Score | Params (B) | Arch | VRAM BF16 | VRAM FP8 | VRAM INT4 | Fits on |
|------|-------|-------|------------|------|-----------|----------|-----------|---------|
| 1 | kimi-k3-max | 1488 | ? | ? | ? | ? | ? | ? |
| 2 | glm-5.3-max | 1482 | ? | ? | ? | ? | ? | ? |
| 3 | glm-5.3-flash | 1474 | ? | ? | ? | ? | ? | ? |
| 4 | glm-5.2-max | 1471 | ? | ? | ? | ? | ? | ? |
| 5 | mimo-v2.5-pro | 1468 | ? | ? | ? | ? | ? | ? |
| 6 | glm-5.1 | 1466 | ? | ? | ? | ? | ? | ? |
| 7 | kimi-k2.6 | 1460 | 1000 (32) | MoE | 2500 | 1250 | 625 | Multi-GPU |
| 8 | deepseek-v4-pro-high-20260813 | 1459 | ? | ? | ? | ? | ? | ? |
| 9 | deepseek-v4-pro | 1457 | 1.6 (49) | Dense | 4 | 2 | 1 | H100 SXM (FP8) |
| 10 | glm-5 | 1457 | 744 (40) | MoE | 1860 | 930 | 465 | Multi-GPU |
| 11 | hy3 | 1455 | 299 (21) | MoE | 747.5 | 373.8 | 186.9 | Multi-GPU |
| 12 | deepseek-v4-pro-high-preview | 1455 | ? | ? | ? | ? | ? | ? |
| 13 | gemma-4-31b | 1451 | 31 | Dense | 77.5 | 38.8 | 19.4 | H100 SXM (FP8) |
| 14 | kimi-k2.5-thinking | 1450 | 1000 (32) | MoE | 2500 | 1250 | 625 | Multi-GPU |
| 15 | minimax-m3 | 1442 | 428 (23) | MoE | 1070 | 535 | 267.5 | Multi-GPU |
| 16 | glm-4.7 | 1441 | ? | ? | ? | ? | ? | ? |
| 17 | qwen3.5-397b-a17b | 1441 | 397 (17) | MoE | 992.5 | 496.2 | 248.1 | Multi-GPU |
| 18 | inkling | 1439 | 975 (41) | MoE | 2437.5 | 1218.8 | 609.4 | Multi-GPU |
| 19 | deepseek-v4-flash-high-preview | 1438 | ? | ? | ? | ? | ? | ? |
| 20 | gemma-4-26b-a4b | 1438 | 26 (4) | MoE | 65 | 32.5 | 16.2 | H100 SXM (FP8) |
| 21 | qwen3.8-27b | 1435 | 27 | Dense | 67.5 | 33.8 | 16.9 | H100 SXM (FP8) |
| 22 | deepseek-v4-flash | 1435 | 284 (13) | MoE | 710 | 355 | 177.5 | Multi-GPU |
| 23 | mimo-v2.5 | 1433 | ? | ? | ? | ? | ? | ? |
| 24 | kimi-k2.5-instant | 1430 | 1000 (32) | MoE | 2500 | 1250 | 625 | Multi-GPU |
| 25 | kimi-k2-thinking-turbo | 1430 | 1000 (32) | MoE | 2500 | 1250 | 625 | Multi-GPU |
| 26 | muse-glimmer | 1427 | 30 | Dense | 75 | 37.5 | 18.8 | H100 SXM (FP8) |
| 27 | mistral-medium-3.5 | 1426 | ? | ? | ? | ? | ? | ? |
| 28 | nvidia-nemotron-3-ultra-550b-a55b-nvfp4 | 1426 | 550 (55) | MoE | 1375 | 687.5 | 343.8 | Multi-GPU |
| 29 | deepseek-v3.2 | 1425 | ? | ? | ? | ? | ? | ? |
| 30 | deepseek-v3.2-exp-thinking | 1425 | ? | ? | ? | ? | ? | ? |
| 31 | glm-4.6 | 1424 | ? | ? | ? | ? | ? | ? |
| 32 | qwen3-235b-a22b-instruct-2507 | 1423 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |
| 33 | deepseek-v3.2-thinking | 1422 | ? | ? | ? | ? | ? | ? |
| 34 | deepseek-v3.2-exp | 1422 | ? | ? | ? | ? | ? | ? |
| 35 | deepseek-r1-0528 | 1421 | ? | ? | ? | ? | ? | ? |
| 36 | kimi-k2-0905-preview | 1417 | 1000 (32) | MoE | 2500 | 1250 | 625 | Multi-GPU |
| 37 | kimi-k2-0711-preview | 1417 | 1000 (32) | MoE | 2500 | 1250 | 625 | Multi-GPU |
| 38 | deepseek-v3.1-terminus-thinking | 1417 | ? | ? | ? | ? | ? | ? |
| 39 | deepseek-v3.1 | 1417 | ? | ? | ? | ? | ? | ? |
| 40 | qwen3.5-122b-a10b | 1417 | 122 (10) | MoE | 305 | 152.5 | 76.2 | B200 SXM (FP8) |
| 41 | deepseek-v3.1-thinking | 1415 | ? | ? | ? | ? | ? | ? |
| 42 | minimax-m2.7 | 1415 | ? | ? | ? | ? | ? | ? |
| 43 | deepseek-v3.1-terminus | 1414 | ? | ? | ? | ? | ? | ? |
| 44 | qwen3-vl-235b-a22b-instruct | 1413 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |
| 45 | mistral-large-3 | 1413 | 675 (41) | MoE | 1687.5 | 843.8 | 421.9 | Multi-GPU |
| 46 | hunyuan-hy3-preview | 1413 | ? | ? | ? | ? | ? | ? |
| 47 | glm-4.5 | 1411 | 355 (32) | MoE | 887.5 | 443.8 | 221.9 | Multi-GPU |
| 48 | qwen3.5-27b | 1407 | 27 | Dense | 67.5 | 33.8 | 16.9 | H100 SXM (FP8) |
| 49 | Inkling Small | 1406 | 266 (12) | MoE | 665 | 332.5 | 166.2 | Multi-GPU |
| 50 | qwen3-235b-a22b-no-thinking | 1402 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |

<details><summary>Show remaining 172 models</summary>

| Rank | Model | Score | Params (B) | Arch | VRAM BF16 | VRAM FP8 | VRAM INT4 | Fits on |
|------|-------|-------|------------|------|-----------|----------|-----------|---------|
| 51 | longcat-flash-chat | 1401 | 560 (27) | MoE | 1400 | 700 | 350 | Multi-GPU |
| 52 | qwen3-235b-a22b-thinking-2507 | 1399 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |
| 53 | qwen3-next-80b-a3b-instruct | 1399 | 80 (3) | MoE | 200 | 100 | 50 | H200 SXM (FP8) |
| 54 | deepseek-r1 | 1397 | 685 (37) | MoE | 1712.5 | 856.2 | 428.1 | Multi-GPU |
| 55 | deepseek-v3-0324 | 1395 | 671 (37) | MoE | 1677.5 | 838.8 | 419.4 | Multi-GPU |
| 56 | qwen3.5-35b-a3b | 1395 | 35 (3) | MoE | 87.5 | 43.8 | 21.9 | H100 SXM (FP8) |
| 57 | qwen3-vl-235b-a22b-thinking | 1395 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |
| 58 | step-3.5-flash | 1394 | ? | ? | ? | ? | ? | ? |
| 59 | mimo-v2-flash (non-thinking) | 1392 | ? | ? | ? | ? | ? | ? |
| 60 | minimax-m2.5 | 1390 | ? | ? | ? | ? | ? | ? |
| 61 | qwen3-coder-480b-a35b-instruct | 1387 | 480 (35) | MoE | 1200 | 600 | 300 | Multi-GPU |
| 62 | mimo-v2-flash (thinking) | 1386 | ? | ? | ? | ? | ? | ? |
| 63 | minimax-m2.1-preview | 1383 | ? | ? | ? | ? | ? | ? |
| 64 | qwen3-30b-a3b-instruct-2507 | 1382 | 30 (3) | MoE | 75 | 37.5 | 18.8 | H100 SXM (FP8) |
| 65 | trinity-large-preview | 1378 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 66 | glm-4.6v | 1378 | ? | ? | ? | ? | ? | ? |
| 67 | qwen3-235b-a22b | 1374 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |
| 68 | glm-4.5-air | 1373 | ? | ? | ? | ? | ? | ? |
| 69 | trinity-large-thinking | 1369 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 70 | qwen3-next-80b-a3b-thinking | 1369 | 80 (3) | MoE | 200 | 100 | 50 | H200 SXM (FP8) |
| 71 | glm-4.7-flash | 1366 | ? | ? | ? | ? | ? | ? |
| 72 | gemma-3-27b-it | 1365 | 27 | Dense | 67.5 | 33.8 | 16.9 | H100 SXM (FP8) |
| 73 | minimax-m1 | 1363 | ? | ? | ? | ? | ? | ? |
| 74 | nvidia-nemotron-3-super-120b-a12b | 1360 | 120 (12) | MoE | 300 | 150 | 75 | B200 SXM (FP8) |
| 75 | deepseek-v3 | 1358 | 671 (37) | MoE | 1677.5 | 838.8 | 419.4 | Multi-GPU |
| 76 | mistral-small-2506 | 1356 | ? | ? | ? | ? | ? | ? |
| 77 | intellect-3 | 1356 | 107 (12) | MoE | 267.5 | 133.8 | 66.9 | H200 SXM (FP8) |
| 78 | nvidia-nemotron-3.5-lightning-30b-a3b-nvfp4 | 1355 | 30 (3) | MoE | 75 | 37.5 | 18.8 | H100 SXM (FP8) |
| 79 | command-a-03-2025 | 1353 | ? | ? | ? | ? | ? | ? |
| 80 | glm-4.5v | 1352 | ? | ? | ? | ? | ? | ? |
| 81 | gpt-oss-120b | 1352 | 117 (5.1) | MoE | 292.5 | 146.2 | 73.1 | B200 SXM (FP8) |
| 82 | step-3 | 1349 | ? | ? | ? | ? | ? | ? |
| 83 | llama-3.1-nemotron-ultra-253b-v1 | 1347 | 253 | Dense | 632.5 | 316.2 | 158.1 | Multi-GPU |
| 84 | qwen3-32b | 1346 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 85 | minimax-m2 | 1345 | 230 (10) | MoE | 575 | 287.5 | 143.8 | B300 SXM (FP8) |
| 86 | ling-flash-2.0 | 1344 | ? | ? | ? | ? | ? | ? |
| 87 | nvidia-llama-3.3-nemotron-super-49b-v1.5 | 1343 | 49 | Dense | 122.5 | 61.2 | 30.6 | H100 SXM (FP8) |
| 88 | granite-4.2-30b | 1341 | 30 | Dense | 75 | 37.5 | 18.8 | H100 SXM (FP8) |
| 89 | gemma-3-12b-it | 1341 | 12 | Dense | 30 | 15 | 7.5 | H100 SXM (FP8) |
| 90 | qwq-32b | 1335 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 91 | llama-3.1-405b-instruct-bf16 | 1334 | 405 | Dense | 1012.5 | 506.2 | 253.1 | Multi-GPU |
| 92 | llama-3.1-405b-instruct-fp8 | 1333 | 405 | Dense | 1012.5 | 506.2 | 253.1 | Multi-GPU |
| 93 | olmo-3.1-32b-instruct | 1329 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 94 | llama-3.3-nemotron-49b-super-v1 | 1327 | 49 | Dense | 122.5 | 61.2 | 30.6 | H100 SXM (FP8) |
| 95 | molmo-2-8b | 1327 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 96 | qwen3-30b-a3b | 1326 | 30 (3) | MoE | 75 | 37.5 | 18.8 | H100 SXM (FP8) |
| 97 | llama-4-maverick-17b-128e-instruct | 1326 | 400 (17) | MoE | 1000 | 500 | 250 | Multi-GPU |
| 98 | deepseek-v2.5-1210 | 1323 | ? | ? | ? | ? | ? | ? |
| 99 | llama-4-scout-17b-16e-instruct | 1321 | 109 (17) | MoE | 272.5 | 136.2 | 68.1 | H200 SXM (FP8) |
| 100 | ring-flash-2.0 | 1321 | ? | ? | ? | ? | ? | ? |
| 101 | qwen-max-0919 | 1317 | ? | ? | ? | ? | ? | ? |
| 102 | llama-3.3-70b-instruct | 1317 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 103 | gemma-3n-e4b-it | 1317 | 8.4 (4) | MoE | 21 | 10.5 | 5.2 | H100 SXM (FP8) |
| 104 | gpt-oss-20b | 1317 | 21 (3.6) | MoE | 52.5 | 26.2 | 13.1 | H100 SXM (FP8) |
| 105 | nvidia-nemotron-3-nano-30b-a3b-bf16 | 1314 | 30 (3) | MoE | 75 | 37.5 | 18.8 | H100 SXM (FP8) |
| 106 | mistral-large-2407 | 1314 | 123 | Dense | 307.5 | 153.8 | 76.9 | B200 SXM (FP8) |
| 107 | athene-v2-chat | 1314 | 72 | Dense | 180 | 90 | 45 | RTX PRO 6000 (FP8) |
| 108 | deepseek-v2.5 | 1306 | ? | ? | ? | ? | ? | ? |
| 109 | olmo-3-32b-think | 1306 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 110 | athene-70b-0725 | 1306 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 111 | granite-4.1-8b | 1306 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 112 | mistral-large-2411 | 1305 | ? | ? | ? | ? | ? | ? |
| 113 | gemma-3-4b-it | 1303 | 4 | Dense | 10 | 5 | 2.5 | H100 SXM (FP8) |
| 114 | mistral-small-3.1-24b-instruct-2503 | 1302 | 24 | Dense | 60 | 30 | 15 | H100 SXM (FP8) |
| 115 | qwen2.5-72b-instruct | 1302 | 72 | Dense | 180 | 90 | 45 | RTX PRO 6000 (FP8) |
| 116 | llama-3.1-nemotron-70b-instruct | 1298 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 117 | granite-4.2-3b | 1295 | 3 | Dense | 7.5 | 3.8 | 1.9 | H100 SXM (FP8) |
| 118 | llama-3.1-70b-instruct | 1293 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 119 | jamba-1.5-large | 1289 | ? | ? | ? | ? | ? | ? |
| 120 | gemma-2-27b-it | 1289 | 27 | Dense | 67.5 | 33.8 | 16.9 | H100 SXM (FP8) |
| 121 | granite-4.2-8b | 1288 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 122 | llama-3.1-nemotron-51b-instruct | 1286 | 51 | Dense | 127.5 | 63.8 | 31.9 | H100 SXM (FP8) |
| 123 | llama-3.1-tulu-3-70b | 1285 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 124 | olmo-3.1-32b-think | 1285 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 125 | ibm-granite-h-small | 1285 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 126 | gemma-2-9b-it-simpo | 1279 | 9 | Dense | 22.5 | 11.2 | 5.6 | H100 SXM (FP8) |
| 127 | nemotron-4-340b-instruct | 1276 | 340 | Dense | 850 | 425 | 212.5 | Multi-GPU |
| 128 | llama-3-70b-instruct | 1276 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 129 | command-r-plus-08-2024 | 1275 | 104 | Dense | 260 | 130 | 65 | H200 SXM (FP8) |
| 130 | mistral-small-24b-instruct-2501 | 1274 | 24 | Dense | 60 | 30 | 15 | H100 SXM (FP8) |
| 131 | qwen2.5-coder-32b-instruct | 1270 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 132 | c4ai-aya-expanse-32b | 1266 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 133 | gemma-2-9b-it | 1266 | 9 | Dense | 22.5 | 11.2 | 5.6 | H100 SXM (FP8) |
| 134 | deepseek-coder-v2 | 1264 | 236 (21) | MoE | 590 | 295 | 147.5 | Multi-GPU |
| 135 | qwen2-72b-instruct | 1261 | 72 | Dense | 180 | 90 | 45 | RTX PRO 6000 (FP8) |
| 136 | command-r-plus | 1261 | ? | ? | ? | ? | ? | ? |
| 137 | phi-4 | 1255 | 14 | Dense | 35 | 17.5 | 8.8 | H100 SXM (FP8) |
| 138 | olmo-2-0325-32b-instruct | 1251 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 139 | command-r-08-2024 | 1249 | 35 | Dense | 87.5 | 43.8 | 21.9 | H100 SXM (FP8) |
| 140 | jamba-1.5-mini | 1239 | ? | ? | ? | ? | ? | ? |
| 141 | ministral-8b-2410 | 1237 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 142 | qwen1.5-110b-chat | 1233 | 110 | Dense | 275 | 137.5 | 68.8 | H200 SXM (FP8) |
| 143 | qwen1.5-72b-chat | 1232 | 72 | Dense | 180 | 90 | 45 | RTX PRO 6000 (FP8) |
| 144 | mixtral-8x22b-instruct-v0.1 | 1228 | 140.8 (39.6) | MoE | 352 | 176 | 88 | B200 SXM (FP8) |
| 145 | command-r | 1226 | ? | ? | ? | ? | ? | ? |
| 146 | llama-3-8b-instruct | 1223 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 147 | c4ai-aya-expanse-8b | 1222 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 148 | llama-3.1-tulu-3-8b | 1220 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 149 | yi-1.5-34b-chat | 1212 | 34 | Dense | 85 | 42.5 | 21.2 | H100 SXM (FP8) |
| 150 | zephyr-orpo-141b-A35b-v0.1 | 1212 | 141 (35) | MoE | 352.5 | 176.2 | 88.1 | B200 SXM (FP8) |
| 151 | llama-3.1-8b-instruct | 1211 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 152 | granite-3.1-8b-instruct | 1207 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 153 | qwen1.5-32b-chat | 1203 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 154 | gemma-2-2b-it | 1199 | 2 | Dense | 5 | 2.5 | 1.2 | H100 SXM (FP8) |
| 155 | phi-3-medium-4k-instruct | 1197 | 14 | Dense | 35 | 17.5 | 8.8 | H100 SXM (FP8) |
| 156 | mixtral-8x7b-instruct-v0.1 | 1196 | 44.8 (12.6) | MoE | 112 | 56 | 28 | H100 SXM (FP8) |
| 157 | dbrx-instruct-preview | 1194 | ? | ? | ? | ? | ? | ? |
| 158 | qwen1.5-14b-chat | 1190 | 14 | Dense | 35 | 17.5 | 8.8 | H100 SXM (FP8) |
| 159 | internlm2_5-20b-chat | 1190 | 20 | Dense | 50 | 25 | 12.5 | H100 SXM (FP8) |
| 160 | deepseek-llm-67b-chat | 1184 | 67 | Dense | 167.5 | 83.8 | 41.9 | RTX PRO 6000 (FP8) |
| 161 | wizardlm-70b | 1184 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 162 | yi-34b-chat | 1183 | 34 | Dense | 85 | 42.5 | 21.2 | H100 SXM (FP8) |
| 163 | granite-3.0-8b-instruct | 1182 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 164 | openchat-3.5 | 1182 | ? | ? | ? | ? | ? | ? |
| 165 | openchat-3.5-0106 | 1182 | ? | ? | ? | ? | ? | ? |
| 166 | gemma-1.1-7b-it | 1181 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 167 | snowflake-arctic-instruct | 1179 | ? | ? | ? | ? | ? | ? |
| 168 | granite-3.1-2b-instruct | 1178 | 2 | Dense | 5 | 2.5 | 1.2 | H100 SXM (FP8) |
| 169 | tulu-2-dpo-70b | 1177 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 170 | openhermes-2.5-mistral-7b | 1175 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 171 | vicuna-33b | 1172 | 33 | Dense | 82.5 | 41.2 | 20.6 | H100 SXM (FP8) |
| 172 | starling-lm-7b-beta | 1170 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 173 | phi-3-small-8k-instruct | 1170 | 7.4 | Dense | 18.5 | 9.2 | 4.6 | H100 SXM (FP8) |
| 174 | llama-2-70b-chat | 1170 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 175 | starling-lm-7b-alpha | 1166 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 176 | llama-3.2-3b-instruct | 1166 | 3 | Dense | 7.5 | 3.8 | 1.9 | H100 SXM (FP8) |
| 177 | nous-hermes-2-mixtral-8x7b-dpo | 1163 | 44.8 (12.6) | MoE | 112 | 56 | 28 | H100 SXM (FP8) |
| 178 | granite-3.0-2b-instruct | 1156 | 2 | Dense | 5 | 2.5 | 1.2 | H100 SXM (FP8) |
| 179 | llama2-70b-steerlm-chat | 1154 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 180 | qwq-32b-preview | 1154 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 181 | solar-10.7b-instruct-v1.0 | 1151 | 10.7 | Dense | 26.8 | 13.4 | 6.7 | H100 SXM (FP8) |
| 182 | dolphin-2.2.1-mistral-7b | 1151 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 183 | mpt-30b-chat | 1150 | 30 | Dense | 75 | 37.5 | 18.8 | H100 SXM (FP8) |
| 184 | wizardlm-13b | 1148 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 185 | mistral-7b-instruct-v0.2 | 1148 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 186 | falcon-180b-chat | 1147 | 180 | Dense | 450 | 225 | 112.5 | B300 SXM (FP8) |
| 187 | qwen1.5-7b-chat | 1143 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 188 | phi-3-mini-4k-instruct-june-2024 | 1142 | 3.8 | Dense | 9.5 | 4.8 | 2.4 | H100 SXM (FP8) |
| 189 | llama-2-13b-chat | 1140 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 190 | vicuna-13b | 1140 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 191 | qwen-14b-chat | 1138 | 14 | Dense | 35 | 17.5 | 8.8 | H100 SXM (FP8) |
| 192 | gemma-7b-it | 1137 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 193 | codellama-34b-instruct | 1136 | 34 | Dense | 85 | 42.5 | 21.2 | H100 SXM (FP8) |
| 194 | zephyr-7b-beta | 1130 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 195 | phi-3-mini-128k-instruct | 1129 | 3.8 | Dense | 9.5 | 4.8 | 2.4 | H100 SXM (FP8) |
| 196 | phi-3-mini-4k-instruct | 1127 | 3.8 | Dense | 9.5 | 4.8 | 2.4 | H100 SXM (FP8) |
| 197 | guanaco-33b | 1126 | 33 | Dense | 82.5 | 41.2 | 20.6 | H100 SXM (FP8) |
| 198 | zephyr-7b-alpha | 1126 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 199 | stripedhyena-nous-7b | 1120 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 200 | codellama-70b-instruct | 1118 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 201 | gemma-1.1-2b-it | 1116 | 2 | Dense | 5 | 2.5 | 1.2 | H100 SXM (FP8) |
| 202 | vicuna-7b | 1114 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 203 | smollm2-1.7b-instruct | 1114 | 1.7 | Dense | 4.2 | 2.1 | 1.1 | H100 SXM (FP8) |
| 204 | llama-3.2-1b-instruct | 1110 | 1 | Dense | 2.5 | 1.2 | 0.6 | H100 SXM (FP8) |
| 205 | mistral-7b-instruct | 1109 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 206 | llama-2-7b-chat | 1107 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 207 | gemma-2b-it | 1093 | 2 | Dense | 5 | 2.5 | 1.2 | H100 SXM (FP8) |
| 208 | qwen1.5-4b-chat | 1090 | 4 | Dense | 10 | 5 | 2.5 | H100 SXM (FP8) |
| 209 | olmo-7b-instruct | 1073 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 210 | koala-13b | 1070 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 211 | alpaca-13b | 1069 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 212 | gpt4all-13b-snoozy | 1066 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 213 | mpt-7b-chat | 1062 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 214 | chatglm3-6b | 1055 | 6 | Dense | 15 | 7.5 | 3.8 | H100 SXM (FP8) |
| 215 | RWKV-4-Raven-14B | 1041 | 14 | Dense | 35 | 17.5 | 8.8 | H100 SXM (FP8) |
| 216 | chatglm2-6b | 1024 | 6 | Dense | 15 | 7.5 | 3.8 | H100 SXM (FP8) |
| 217 | oasst-pythia-12b | 1022 | 12 | Dense | 30 | 15 | 7.5 | H100 SXM (FP8) |
| 218 | chatglm-6b | 995 | 6 | Dense | 15 | 7.5 | 3.8 | H100 SXM (FP8) |
| 219 | fastchat-t5-3b | 991 | 3 | Dense | 7.5 | 3.8 | 1.9 | H100 SXM (FP8) |
| 220 | dolly-v2-12b | 981 | 12 | Dense | 30 | 15 | 7.5 | H100 SXM (FP8) |
| 221 | llama-13b | 974 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 222 | stablelm-tuned-alpha-7b | 952 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |

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
