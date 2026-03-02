# LLM Arena VRAM Calculator

Enriches the [Arena.ai](https://arena.ai/leaderboard/text?license=open-source) open-source LLM leaderboard with parameter counts and VRAM estimates for single-GPU deployment feasibility.

Most LLM leaderboards rank models by quality but ignore deployment constraints. This tool answers: *"What's the best model I can actually run on my hardware?"* by cross-referencing Arena rankings with VRAM requirements across precisions.

> **Last updated:** 2026-03-02 06:50 UTC | **Models:** 187 | **Resolved:** 184 (98.4%)

## Best Model Per GPU

Highest-ranked Arena model that fits on each single GPU (includes 25% serving overhead for KV cache, activations, and framework).

### BF16 (Full Precision)

| GPU | VRAM | Best Model | Arena Rank | Params | Arch | Serving VRAM |
|-----|------|------------|------------|--------|------|--------------|
| H100 SXM | 80 GB | qwen3-30b-a3b-instruct-2507 | #36 | 30.5B | MoE | 76.2 GB |
| RTX PRO 6000 | 96 GB | qwen3-30b-a3b-instruct-2507 | #36 | 30.5B | MoE | 76.2 GB |
| H200 SXM | 141 GB | qwen3-30b-a3b-instruct-2507 | #36 | 30.5B | MoE | 76.2 GB |
| B200 SXM | 180 GB | qwen3-30b-a3b-instruct-2507 | #36 | 30.5B | MoE | 76.2 GB |
| B300 SXM | 288 GB | qwen3-next-80b-a3b-instruct | #24 | 80B | MoE | 200.0 GB |

### FP8 (8-bit)

| GPU | VRAM | Best Model | Arena Rank | Params | Arch | Serving VRAM |
|-----|------|------------|------------|--------|------|--------------|
| H100 SXM | 80 GB | qwen3-30b-a3b-instruct-2507 | #36 | 30.5B | MoE | 38.1 GB |
| RTX PRO 6000 | 96 GB | qwen3-30b-a3b-instruct-2507 | #36 | 30.5B | MoE | 38.1 GB |
| H200 SXM | 141 GB | qwen3-next-80b-a3b-instruct | #24 | 80B | MoE | 100.0 GB |
| B200 SXM | 180 GB | qwen3-next-80b-a3b-instruct | #24 | 80B | MoE | 100.0 GB |
| B300 SXM | 288 GB | qwen3-next-80b-a3b-instruct | #24 | 80B | MoE | 100.0 GB |

### INT4 (4-bit)

| GPU | VRAM | Best Model | Arena Rank | Params | Arch | Serving VRAM |
|-----|------|------------|------------|--------|------|--------------|
| H100 SXM | 80 GB | qwen3-next-80b-a3b-instruct | #24 | 80B | MoE | 50.0 GB |
| RTX PRO 6000 | 96 GB | qwen3-next-80b-a3b-instruct | #24 | 80B | MoE | 50.0 GB |
| H200 SXM | 141 GB | qwen3-next-80b-a3b-instruct | #24 | 80B | MoE | 50.0 GB |
| B200 SXM | 180 GB | qwen3-235b-a22b-instruct-2507 | #10 | 235B | MoE | 146.9 GB |
| B300 SXM | 288 GB | qwen3.5-397b-a17b | #2 | 397B | MoE | 248.1 GB |

## Full Leaderboard

| Rank | Model | Score | Params (B) | Arch | VRAM BF16 | VRAM FP8 | VRAM INT4 | Fits on |
|------|-------|-------|------------|------|-----------|----------|-----------|---------|
| 1 | glm-5 | 1455 | 744 (40) | MoE | 1860 | 930 | 465 | Multi-GPU |
| 2 | qwen3.5-397b-a17b | 1453 | 397 (17) | MoE | 992.5 | 496.2 | 248.1 | Multi-GPU |
| 3 | kimi-k2.5-thinking | 1451 | 1000 (32) | MoE | 2500 | 1250 | 625 | Multi-GPU |
| 4 | glm-4.7 | 1440 | 357 (32) | MoE | 892.5 | 446.2 | 223.1 | Multi-GPU |
| 5 | kimi-k2.5-instant | 1434 | 1000 (32) | MoE | 2500 | 1250 | 625 | Multi-GPU |
| 6 | kimi-k2-thinking-turbo | 1428 | 1000 (32) | MoE | 2500 | 1250 | 625 | Multi-GPU |
| 7 | glm-4.6 | 1425 | 357 (32) | MoE | 892.5 | 446.2 | 223.1 | Multi-GPU |
| 8 | deepseek-v3.2-exp | 1423 | 685 (37) | MoE | 1712.5 | 856.2 | 428.1 | Multi-GPU |
| 9 | deepseek-v3.2-exp-thinking | 1423 | 685 (37) | MoE | 1712.5 | 856.2 | 428.1 | Multi-GPU |
| 10 | qwen3-235b-a22b-instruct-2507 | 1422 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |
| 11 | deepseek-v3.2-thinking | 1420 | 685 (37) | MoE | 1712.5 | 856.2 | 428.1 | Multi-GPU |
| 12 | deepseek-v3.2 | 1419 | 685 (37) | MoE | 1712.5 | 856.2 | 428.1 | Multi-GPU |
| 13 | deepseek-r1-0528 | 1419 | 685 (37) | MoE | 1712.5 | 856.2 | 428.1 | Multi-GPU |
| 14 | deepseek-v3.1 | 1418 | 685 (37) | MoE | 1712.5 | 856.2 | 428.1 | Multi-GPU |
| 15 | deepseek-v3.1-thinking | 1417 | 685 (37) | MoE | 1712.5 | 856.2 | 428.1 | Multi-GPU |
| 16 | kimi-k2-0905-preview | 1417 | 1000 (32) | MoE | 2500 | 1250 | 625 | Multi-GPU |
| 17 | kimi-k2-0711-preview | 1416 | 1000 (32) | MoE | 2500 | 1250 | 625 | Multi-GPU |
| 18 | deepseek-v3.1-terminus | 1416 | 685 (37) | MoE | 1712.5 | 856.2 | 428.1 | Multi-GPU |
| 19 | deepseek-v3.1-terminus-thinking | 1415 | 685 (37) | MoE | 1712.5 | 856.2 | 428.1 | Multi-GPU |
| 20 | mistral-large-3 | 1414 | 675 (41) | MoE | 1687.5 | 843.8 | 421.9 | Multi-GPU |
| 21 | qwen3-vl-235b-a22b-instruct | 1414 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |
| 22 | glm-4.5 | 1410 | 355 (32) | MoE | 887.5 | 443.8 | 221.9 | Multi-GPU |
| 23 | qwen3-235b-a22b-no-thinking | 1401 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |
| 24 | qwen3-next-80b-a3b-instruct | 1401 | 80 (3) | MoE | 200 | 100 | 50 | H200 SXM (FP8) |
| 25 | minimax-m2.5 | 1401 | 230 (10) | MoE | 575 | 287.5 | 143.8 | B300 SXM (FP8) |
| 26 | longcat-flash-chat | 1399 | 560 (27) | MoE | 1400 | 700 | 350 | Multi-GPU |
| 27 | qwen3-235b-a22b-thinking-2507 | 1398 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |
| 28 | deepseek-r1 | 1397 | 685 (37) | MoE | 1712.5 | 856.2 | 428.1 | Multi-GPU |
| 29 | qwen3-vl-235b-a22b-thinking | 1395 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |
| 30 | deepseek-v3-0324 | 1394 | 671 (37) | MoE | 1677.5 | 838.8 | 419.4 | Multi-GPU |
| 31 | mimo-v2-flash (non-thinking) | 1390 | 309 (15) | MoE | 772.5 | 386.2 | 193.1 | Multi-GPU |
| 32 | step-3.5-flash | 1389 | ? | ? | ? | ? | ? | ? |
| 33 | mimo-v2-flash (thinking) | 1386 | 309 (15) | MoE | 772.5 | 386.2 | 193.1 | Multi-GPU |
| 34 | qwen3-coder-480b-a35b-instruct | 1386 | 480 (35) | MoE | 1200 | 600 | 300 | Multi-GPU |
| 35 | minimax-m2.1-preview | 1385 | 230 (10) | MoE | 575 | 287.5 | 143.8 | B300 SXM (FP8) |
| 36 | qwen3-30b-a3b-instruct-2507 | 1383 | 30.5 (3.3) | MoE | 76.2 | 38.1 | 19.1 | H100 SXM (FP8) |
| 37 | glm-4.6v | 1377 | 108 | Dense | 270 | 135 | 67.5 | H200 SXM (FP8) |
| 38 | trinity-large | 1375 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 39 | qwen3-235b-a22b | 1374 | 235 (22) | MoE | 587.5 | 293.8 | 146.9 | Multi-GPU |
| 40 | glm-4.5-air | 1371 | 106 (12) | MoE | 265 | 132.5 | 66.2 | H200 SXM (FP8) |
| 41 | qwen3-next-80b-a3b-thinking | 1368 | 80 (3) | MoE | 200 | 100 | 50 | H200 SXM (FP8) |
| 42 | minimax-m1 | 1367 | 456 (45.9) | MoE | 1140 | 570 | 285 | Multi-GPU |
| 43 | glm-4.7-flash | 1365 | 31.2 (3) | MoE | 78 | 39 | 19.5 | H100 SXM (FP8) |
| 44 | gemma-3-27b-it | 1365 | 27.4 | Dense | 68.5 | 34.2 | 17.1 | H100 SXM (FP8) |
| 45 | deepseek-v3 | 1358 | 671 (37) | MoE | 1677.5 | 838.8 | 419.4 | Multi-GPU |
| 46 | intellect-3 | 1356 | 107 | Dense | 267.5 | 133.8 | 66.9 | H200 SXM (FP8) |
| 47 | mistral-small-2506 | 1356 | 22 | Dense | 55 | 27.5 | 13.8 | H100 SXM (FP8) |
| 48 | gpt-oss-120b | 1353 | 117 (5.1) | MoE | 292.5 | 146.2 | 73.1 | B200 SXM (FP8) |
| 49 | glm-4.5v | 1353 | 108 (12) | MoE | 270 | 135 | 67.5 | H200 SXM (FP8) |
| 50 | command-a-03-2025 | 1353 | 111 | Dense | 277.5 | 138.8 | 69.4 | H200 SXM (FP8) |

<details><summary>Show remaining 137 models</summary>

| Rank | Model | Score | Params (B) | Arch | VRAM BF16 | VRAM FP8 | VRAM INT4 | Fits on |
|------|-------|-------|------------|------|-----------|----------|-----------|---------|
| 51 | llama-3.1-nemotron-ultra-253b-v1 | 1347 | 253 | Dense | 632.5 | 316.2 | 158.1 | Multi-GPU |
| 52 | qwen3-32b | 1347 | 32.8 | Dense | 82 | 41 | 20.5 | H100 SXM (FP8) |
| 53 | minimax-m2 | 1346 | 230 (10) | MoE | 575 | 287.5 | 143.8 | B300 SXM (FP8) |
| 54 | ling-flash-2.0 | 1346 | 103 (6.1) | MoE | 257.5 | 128.8 | 64.4 | H200 SXM (FP8) |
| 55 | step-3 | 1346 | ? | ? | ? | ? | ? | ? |
| 56 | gemma-3-12b-it | 1341 | 12.2 | Dense | 30.5 | 15.2 | 7.6 | H100 SXM (FP8) |
| 57 | nvidia-llama-3.3-nemotron-super-49b-v1.5 | 1341 | 49 | Dense | 122.5 | 61.2 | 30.6 | H100 SXM (FP8) |
| 58 | qwq-32b | 1335 | 32.8 | Dense | 82 | 41 | 20.5 | H100 SXM (FP8) |
| 59 | llama-3.1-405b-instruct-bf16 | 1335 | 405 | Dense | 1012.5 | 506.2 | 253.1 | Multi-GPU |
| 60 | llama-3.1-405b-instruct-fp8 | 1333 | 405 | Dense | 1012.5 | 506.2 | 253.1 | Multi-GPU |
| 61 | olmo-3.1-32b-instruct | 1330 | 32.2 | Dense | 80.5 | 40.2 | 20.1 | H100 SXM (FP8) |
| 62 | molmo-2-8b | 1329 | 8.7 | Dense | 21.8 | 10.9 | 5.4 | H100 SXM (FP8) |
| 63 | qwen3-30b-a3b | 1328 | 30.5 (3.3) | MoE | 76.2 | 38.1 | 19.1 | H100 SXM (FP8) |
| 64 | llama-4-maverick-17b-128e-instruct | 1327 | 400 (17) | MoE | 1000 | 500 | 250 | Multi-GPU |
| 65 | llama-3.3-nemotron-49b-super-v1 | 1327 | 49 | Dense | 122.5 | 61.2 | 30.6 | H100 SXM (FP8) |
| 66 | deepseek-v2.5-1210 | 1323 | 236 (21) | MoE | 590 | 295 | 147.5 | Multi-GPU |
| 67 | llama-4-scout-17b-16e-instruct | 1322 | 109 (17) | MoE | 272.5 | 136.2 | 68.1 | H200 SXM (FP8) |
| 68 | ring-flash-2.0 | 1320 | 103 (6.1) | MoE | 257.5 | 128.8 | 64.4 | H200 SXM (FP8) |
| 69 | llama-3.3-70b-instruct | 1319 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 70 | gemma-3n-e4b-it | 1319 | 8.4 (4) | MoE | 21 | 10.5 | 5.2 | H100 SXM (FP8) |
| 71 | qwen-max-0919 | 1318 | ? | ? | ? | ? | ? | ? |
| 72 | gpt-oss-20b | 1317 | 21 (3.6) | MoE | 52.5 | 26.2 | 13.1 | H100 SXM (FP8) |
| 73 | nvidia-nemotron-3-nano-30b-a3b-bf16 | 1317 | 31.6 (3.6) | MoE | 79 | 39.5 | 19.8 | H100 SXM (FP8) |
| 74 | athene-v2-chat | 1314 | 72 | Dense | 180 | 90 | 45 | RTX PRO 6000 (FP8) |
| 75 | mistral-large-2407 | 1314 | 123 | Dense | 307.5 | 153.8 | 76.9 | B200 SXM (FP8) |
| 76 | deepseek-v2.5 | 1306 | 236 (21) | MoE | 590 | 295 | 147.5 | Multi-GPU |
| 77 | athene-70b-0725 | 1305 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 78 | olmo-3-32b-think | 1305 | 32.2 | Dense | 80.5 | 40.2 | 20.1 | H100 SXM (FP8) |
| 79 | mistral-large-2411 | 1305 | 123 | Dense | 307.5 | 153.8 | 76.9 | B200 SXM (FP8) |
| 80 | mistral-small-3.1-24b-instruct-2503 | 1304 | 24 | Dense | 60 | 30 | 15 | H100 SXM (FP8) |
| 81 | gemma-3-4b-it | 1303 | 4.3 | Dense | 10.8 | 5.4 | 2.7 | H100 SXM (FP8) |
| 82 | qwen2.5-72b-instruct | 1302 | 72 | Dense | 180 | 90 | 45 | RTX PRO 6000 (FP8) |
| 83 | llama-3.1-nemotron-70b-instruct | 1298 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 84 | llama-3.1-70b-instruct | 1293 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 85 | jamba-1.5-large | 1288 | 398 (94) | MoE | 995 | 497.5 | 248.8 | Multi-GPU |
| 86 | gemma-2-27b-it | 1288 | 27 | Dense | 67.5 | 33.8 | 16.9 | H100 SXM (FP8) |
| 87 | ibm-granite-h-small | 1287 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 88 | llama-3.1-tulu-3-70b | 1286 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 89 | llama-3.1-nemotron-51b-instruct | 1286 | 51 | Dense | 127.5 | 63.8 | 31.9 | H100 SXM (FP8) |
| 90 | olmo-3.1-32b-think | 1285 | 32.2 | Dense | 80.5 | 40.2 | 20.1 | H100 SXM (FP8) |
| 91 | gemma-2-9b-it-simpo | 1279 | 9 | Dense | 22.5 | 11.2 | 5.6 | H100 SXM (FP8) |
| 92 | nemotron-4-340b-instruct | 1277 | 340 | Dense | 850 | 425 | 212.5 | Multi-GPU |
| 93 | command-r-plus-08-2024 | 1276 | 104 | Dense | 260 | 130 | 65 | H200 SXM (FP8) |
| 94 | llama-3-70b-instruct | 1275 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 95 | mistral-small-24b-instruct-2501 | 1273 | 24 | Dense | 60 | 30 | 15 | H100 SXM (FP8) |
| 96 | qwen2.5-coder-32b-instruct | 1270 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 97 | c4ai-aya-expanse-32b | 1267 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 98 | gemma-2-9b-it | 1265 | 9 | Dense | 22.5 | 11.2 | 5.6 | H100 SXM (FP8) |
| 99 | deepseek-coder-v2 | 1264 | 236 (21) | MoE | 590 | 295 | 147.5 | Multi-GPU |
| 100 | command-r-plus | 1261 | 104 | Dense | 260 | 130 | 65 | H200 SXM (FP8) |
| 101 | qwen2-72b-instruct | 1261 | 72 | Dense | 180 | 90 | 45 | RTX PRO 6000 (FP8) |
| 102 | phi-4 | 1256 | 14 | Dense | 35 | 17.5 | 8.8 | H100 SXM (FP8) |
| 103 | olmo-2-0325-32b-instruct | 1251 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 104 | command-r-08-2024 | 1250 | 35 | Dense | 87.5 | 43.8 | 21.9 | H100 SXM (FP8) |
| 105 | jamba-1.5-mini | 1238 | 52 (12) | MoE | 130 | 65 | 32.5 | H100 SXM (FP8) |
| 106 | ministral-8b-2410 | 1237 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 107 | qwen1.5-110b-chat | 1233 | 110 | Dense | 275 | 137.5 | 68.8 | H200 SXM (FP8) |
| 108 | qwen1.5-72b-chat | 1233 | 72 | Dense | 180 | 90 | 45 | RTX PRO 6000 (FP8) |
| 109 | mixtral-8x22b-instruct-v0.1 | 1229 | 141 (39) | MoE | 352.5 | 176.2 | 88.1 | B200 SXM (FP8) |
| 110 | command-r | 1226 | 35 | Dense | 87.5 | 43.8 | 21.9 | H100 SXM (FP8) |
| 111 | llama-3-8b-instruct | 1223 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 112 | c4ai-aya-expanse-8b | 1223 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 113 | llama-3.1-tulu-3-8b | 1220 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 114 | yi-1.5-34b-chat | 1213 | 34 | Dense | 85 | 42.5 | 21.2 | H100 SXM (FP8) |
| 115 | zephyr-orpo-141b-A35b-v0.1 | 1212 | 141 (35) | MoE | 352.5 | 176.2 | 88.1 | B200 SXM (FP8) |
| 116 | llama-3.1-8b-instruct | 1211 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 117 | granite-3.1-8b-instruct | 1208 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 118 | qwen1.5-32b-chat | 1203 | 32 | Dense | 80 | 40 | 20 | H100 SXM (FP8) |
| 119 | gemma-2-2b-it | 1198 | 2 | Dense | 5 | 2.5 | 1.2 | H100 SXM (FP8) |
| 120 | phi-3-medium-4k-instruct | 1197 | 14 | Dense | 35 | 17.5 | 8.8 | H100 SXM (FP8) |
| 121 | mixtral-8x7b-instruct-v0.1 | 1197 | 46.7 (12.9) | MoE | 116.8 | 58.4 | 29.2 | H100 SXM (FP8) |
| 122 | dbrx-instruct-preview | 1194 | 132 (36) | MoE | 330 | 165 | 82.5 | B200 SXM (FP8) |
| 123 | internlm2_5-20b-chat | 1191 | 20 | Dense | 50 | 25 | 12.5 | H100 SXM (FP8) |
| 124 | qwen1.5-14b-chat | 1190 | 14 | Dense | 35 | 17.5 | 8.8 | H100 SXM (FP8) |
| 125 | wizardlm-70b | 1184 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 126 | deepseek-llm-67b-chat | 1184 | 67 | Dense | 167.5 | 83.8 | 41.9 | RTX PRO 6000 (FP8) |
| 127 | yi-34b-chat | 1183 | 34 | Dense | 85 | 42.5 | 21.2 | H100 SXM (FP8) |
| 128 | openchat-3.5-0106 | 1182 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 129 | openchat-3.5 | 1182 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 130 | granite-3.0-8b-instruct | 1181 | 8 | Dense | 20 | 10 | 5 | H100 SXM (FP8) |
| 131 | gemma-1.1-7b-it | 1180 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 132 | snowflake-arctic-instruct | 1179 | 480 (17) | MoE | 1200 | 600 | 300 | Multi-GPU |
| 133 | granite-3.1-2b-instruct | 1179 | 2 | Dense | 5 | 2.5 | 1.2 | H100 SXM (FP8) |
| 134 | tulu-2-dpo-70b | 1178 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 135 | openhermes-2.5-mistral-7b | 1175 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 136 | vicuna-33b | 1172 | 33 | Dense | 82.5 | 41.2 | 20.6 | H100 SXM (FP8) |
| 137 | starling-lm-7b-beta | 1171 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 138 | phi-3-small-8k-instruct | 1171 | 7.4 | Dense | 18.5 | 9.2 | 4.6 | H100 SXM (FP8) |
| 139 | llama-2-70b-chat | 1170 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 140 | starling-lm-7b-alpha | 1167 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 141 | llama-3.2-3b-instruct | 1166 | 3 | Dense | 7.5 | 3.8 | 1.9 | H100 SXM (FP8) |
| 142 | nous-hermes-2-mixtral-8x7b-dpo | 1164 | 44.8 (12.6) | MoE | 112 | 56 | 28 | H100 SXM (FP8) |
| 143 | qwq-32b-preview | 1157 | 32.8 | Dense | 82 | 41 | 20.5 | H100 SXM (FP8) |
| 144 | granite-3.0-2b-instruct | 1155 | 2 | Dense | 5 | 2.5 | 1.2 | H100 SXM (FP8) |
| 145 | llama2-70b-steerlm-chat | 1155 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 146 | solar-10.7b-instruct-v1.0 | 1152 | 10.7 | Dense | 26.8 | 13.4 | 6.7 | H100 SXM (FP8) |
| 147 | dolphin-2.2.1-mistral-7b | 1151 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 148 | mpt-30b-chat | 1150 | 30 | Dense | 75 | 37.5 | 18.8 | H100 SXM (FP8) |
| 149 | mistral-7b-instruct-v0.2 | 1149 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 150 | wizardlm-13b | 1149 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 151 | falcon-180b-chat | 1146 | 180 | Dense | 450 | 225 | 112.5 | B300 SXM (FP8) |
| 152 | qwen1.5-7b-chat | 1143 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 153 | phi-3-mini-4k-instruct-june-2024 | 1142 | 3.8 | Dense | 9.5 | 4.8 | 2.4 | H100 SXM (FP8) |
| 154 | llama-2-13b-chat | 1141 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 155 | vicuna-13b | 1140 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 156 | qwen-14b-chat | 1138 | 14 | Dense | 35 | 17.5 | 8.8 | H100 SXM (FP8) |
| 157 | codellama-34b-instruct | 1136 | 34 | Dense | 85 | 42.5 | 21.2 | H100 SXM (FP8) |
| 158 | gemma-7b-it | 1135 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 159 | zephyr-7b-beta | 1131 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 160 | phi-3-mini-128k-instruct | 1129 | 3.8 | Dense | 9.5 | 4.8 | 2.4 | H100 SXM (FP8) |
| 161 | phi-3-mini-4k-instruct | 1128 | 3.8 | Dense | 9.5 | 4.8 | 2.4 | H100 SXM (FP8) |
| 162 | guanaco-33b | 1127 | 33 | Dense | 82.5 | 41.2 | 20.6 | H100 SXM (FP8) |
| 163 | zephyr-7b-alpha | 1126 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 164 | stripedhyena-nous-7b | 1120 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 165 | codellama-70b-instruct | 1118 | 70 | Dense | 175 | 87.5 | 43.8 | RTX PRO 6000 (FP8) |
| 166 | vicuna-7b | 1114 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 167 | smollm2-1.7b-instruct | 1114 | 1.7 | Dense | 4.2 | 2.1 | 1.1 | H100 SXM (FP8) |
| 168 | gemma-1.1-2b-it | 1114 | 2 | Dense | 5 | 2.5 | 1.2 | H100 SXM (FP8) |
| 169 | llama-3.2-1b-instruct | 1111 | 1 | Dense | 2.5 | 1.2 | 0.6 | H100 SXM (FP8) |
| 170 | mistral-7b-instruct | 1109 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 171 | llama-2-7b-chat | 1108 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 172 | gemma-2b-it | 1091 | 2 | Dense | 5 | 2.5 | 1.2 | H100 SXM (FP8) |
| 173 | qwen1.5-4b-chat | 1090 | 4 | Dense | 10 | 5 | 2.5 | H100 SXM (FP8) |
| 174 | olmo-7b-instruct | 1074 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 175 | koala-13b | 1070 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 176 | alpaca-13b | 1067 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 177 | gpt4all-13b-snoozy | 1065 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 178 | mpt-7b-chat | 1061 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |
| 179 | chatglm3-6b | 1055 | 6 | Dense | 15 | 7.5 | 3.8 | H100 SXM (FP8) |
| 180 | RWKV-4-Raven-14B | 1041 | 14 | Dense | 35 | 17.5 | 8.8 | H100 SXM (FP8) |
| 181 | chatglm2-6b | 1024 | 6 | Dense | 15 | 7.5 | 3.8 | H100 SXM (FP8) |
| 182 | oasst-pythia-12b | 1022 | 12 | Dense | 30 | 15 | 7.5 | H100 SXM (FP8) |
| 183 | chatglm-6b | 995 | 6 | Dense | 15 | 7.5 | 3.8 | H100 SXM (FP8) |
| 184 | fastchat-t5-3b | 991 | 3 | Dense | 7.5 | 3.8 | 1.9 | H100 SXM (FP8) |
| 185 | dolly-v2-12b | 979 | 12 | Dense | 30 | 15 | 7.5 | H100 SXM (FP8) |
| 186 | llama-13b | 972 | 13 | Dense | 32.5 | 16.2 | 8.1 | H100 SXM (FP8) |
| 187 | stablelm-tuned-alpha-7b | 952 | 7 | Dense | 17.5 | 8.8 | 4.4 | H100 SXM (FP8) |

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
