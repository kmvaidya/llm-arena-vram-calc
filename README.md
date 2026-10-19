# LLM Arena VRAM Calculator

Enriches the [Arena.ai](https://arena.ai/leaderboard/text?license=open-source) open-source LLM leaderboard with parameter counts and VRAM estimates and speed estimates for consumer GPUs, Macs and other unified-memory computers, workstations, data-center GPUs and 8-GPU servers.

Most LLM leaderboards rank models by quality but ignore deployment constraints. This tool answers: *"What's the best model I can actually run on my hardware?"* by cross-referencing Arena rankings with VRAM requirements across precisions.

> **Last updated:** 2026-10-19 11:48 UTC | **Arena data:** 2026-10-15 | **Models:** 225 | **Resolved:** 225 (100.0%)

## Best Model Per Hardware

Highest-ranked Arena model that fits each piece of hardware, with its Arena rank, the memory it needs and its estimated generation speed for one request (see [Memory formula](#memory-formula) and [Speed formula](#speed-formula)). *As released* is the model as its makers published it, at full quality; *8-bit* and *4-bit* are quantized to fit more, unless the release is already that small: a model published in 4-bit counts as released in every column. *Memory* is what a model can use: the GPU's memory, or for computers whose GPU shares system RAM, the share the system gives it. A model fits a GPU when it needs at most 95% of its memory.

| Tier | Hardware | Memory | Bandwidth | As released | 8-bit | 4-bit |
|------|----------|--------|-----------|------|------|------|
| Consumer GPU | RTX 5060 Ti 16 GB | 16 GiB | 0.448 TB/s | gemma-3-4b-it (#116, 11 GiB, ~43 tok/s) | molmo-2-8b (#103, 12 GiB, ~37 tok/s) | gemma-3-12b-it (#92, 10 GiB, ~47 tok/s) |
|  | RTX 5080 | 16 GiB | 0.96 TB/s | gemma-3-4b-it (#116, 11 GiB, ~88 tok/s) | molmo-2-8b (#103, 12 GiB, ~77 tok/s) | gemma-3-12b-it (#92, 10 GiB, ~96 tok/s) |
|  | RTX 3090 | 24 GiB | 0.936 TB/s | nvidia-nemotron-3.5-lightning-30b-a3b-nvfp4 (#85, 22 GiB, ~210 tok/s) | nvidia-nemotron-3.5-lightning-30b-a3b-nvfp4 (#85, 22 GiB, ~210 tok/s) | gemma-4-31b (#15, 23 GiB, ~39 tok/s) |
|  | RTX 4090 | 24 GiB | 1.01 TB/s | nvidia-nemotron-3.5-lightning-30b-a3b-nvfp4 (#85, 22 GiB, ~220 tok/s) | nvidia-nemotron-3.5-lightning-30b-a3b-nvfp4 (#85, 22 GiB, ~220 tok/s) | gemma-4-31b (#15, 23 GiB, ~42 tok/s) |
|  | RTX 5090 | 32 GiB | 1.79 TB/s | nvidia-nemotron-3.5-lightning-30b-a3b-nvfp4 (#85, 22 GiB, ~290 tok/s) | gemma-4-26b-a4b (#24, 30 GiB, ~190 tok/s) | gemma-4-31b (#15, 23 GiB, ~72 tok/s) |
| Unified memory | Mac mini / MacBook Pro, M5 Pro 64 GB | 48 of 64 GiB | 0.307 TB/s | nvidia-nemotron-3.5-lightning-30b-a3b-nvfp4 (#85, 22 GiB, ~74 tok/s) | gemma-4-31b (#15, 36 GiB, ~7.5 tok/s) | gemma-4-31b (#15, 23 GiB, ~13 tok/s) |
|  | Ryzen AI Max+ 395, 128 GB | 96 of 128 GiB | 0.256 TB/s | gemma-4-31b (#15, 64 GiB, ~3.3 tok/s) | gemma-4-31b (#15, 36 GiB, ~6.3 tok/s) | gemma-4-31b (#15, 23 GiB, ~11 tok/s) |
|  | MacBook Pro / Mac Studio, M5 Max 128 GB | 96 of 128 GiB | 0.614 TB/s | gemma-4-31b (#15, 64 GiB, ~7.8 tok/s) | gemma-4-31b (#15, 36 GiB, ~15 tok/s) | gemma-4-31b (#15, 23 GiB, ~26 tok/s) |
|  | DGX Spark (GB10), 128 GB | 115 of 128 GiB | 0.273 TB/s | gemma-4-31b (#15, 64 GiB, ~3.5 tok/s) | gemma-4-31b (#15, 36 GiB, ~6.7 tok/s) | gemma-4-31b (#15, 23 GiB, ~12 tok/s) |
|  | Mac Studio, M5 Ultra 256 GB | 192 of 256 GiB | 1.23 TB/s | gemma-4-31b (#15, 64 GiB, ~15 tok/s) | gemma-4-31b (#15, 36 GiB, ~29 tok/s) | glm-5.3-flash (#6, 184 GiB, ~57 tok/s) |
|  | Mac Studio, M5 Ultra 512 GB | 384 of 512 GiB | 1.23 TB/s | glm-5.3-flash (#6, 311 GiB, ~38 tok/s) | glm-5.3-flash (#6, 311 GiB, ~38 tok/s) | glm-5.3-flash (#6, 184 GiB, ~57 tok/s) |
| Workstation | Radeon AI PRO R9700 | 32 GiB | 0.64 TB/s | nvidia-nemotron-3.5-lightning-30b-a3b-nvfp4 (#85, 22 GiB, ~170 tok/s) | gemma-4-26b-a4b (#24, 30 GiB, ~91 tok/s) | gemma-4-31b (#15, 23 GiB, ~27 tok/s) |
|  | RTX PRO 6000 Blackwell | 96 GiB | 1.79 TB/s | gemma-4-31b (#15, 64 GiB, ~22 tok/s) | gemma-4-31b (#15, 36 GiB, ~42 tok/s) | gemma-4-31b (#15, 23 GiB, ~72 tok/s) |
|  | DGX Station (GB300) | 252 GiB | 7.1 TB/s | gemma-4-31b (#15, 64 GiB, ~83 tok/s) | gemma-4-31b (#15, 36 GiB, ~150 tok/s) | glm-5.3-flash (#6, 184 GiB, ~240 tok/s) |
| Data center GPU | L40S | 48 GiB | 0.864 TB/s | nvidia-nemotron-3.5-lightning-30b-a3b-nvfp4 (#85, 22 GiB, ~200 tok/s) | gemma-4-31b (#15, 36 GiB, ~21 tok/s) | gemma-4-31b (#15, 23 GiB, ~36 tok/s) |
|  | H100 SXM | 80 GiB | 3.35 TB/s | gemma-4-31b (#15, 64 GiB, ~41 tok/s) | gemma-4-31b (#15, 36 GiB, ~76 tok/s) | gemma-4-31b (#15, 23 GiB, ~130 tok/s) |
|  | A100 80GB SXM | 80 GiB | 2.04 TB/s | gemma-4-31b (#15, 64 GiB, ~25 tok/s) | gemma-4-31b (#15, 36 GiB, ~48 tok/s) | gemma-4-31b (#15, 23 GiB, ~81 tok/s) |
|  | H100 NVL | 94 GiB | 3.9 TB/s | gemma-4-31b (#15, 64 GiB, ~48 tok/s) | gemma-4-31b (#15, 36 GiB, ~87 tok/s) | gemma-4-31b (#15, 23 GiB, ~140 tok/s) |
|  | Gaudi 3 | 128 GiB | 3.7 TB/s | gemma-4-31b (#15, 64 GiB, ~45 tok/s) | gemma-4-31b (#15, 36 GiB, ~83 tok/s) | gemma-4-31b (#15, 23 GiB, ~140 tok/s) |
|  | H200 SXM / NVL | 141 GiB | 4.8 TB/s | gemma-4-31b (#15, 64 GiB, ~58 tok/s) | gemma-4-31b (#15, 36 GiB, ~110 tok/s) | gemma-4-31b (#15, 23 GiB, ~170 tok/s) |
|  | B200 | 180 of 192 GiB | 8 TB/s | gemma-4-31b (#15, 64 GiB, ~93 tok/s) | gemma-4-31b (#15, 36 GiB, ~160 tok/s) | gemma-4-31b (#15, 23 GiB, ~260 tok/s) |
|  | MI300X | 192 GiB | 5.3 TB/s | gemma-4-31b (#15, 64 GiB, ~63 tok/s) | gemma-4-31b (#15, 36 GiB, ~120 tok/s) | hy3 (#14, 173 GiB, ~180 tok/s) |
|  | Rubin | 250 of 288 GiB | 16.2 TB/s | gemma-4-31b (#15, 64 GiB, ~170 tok/s) | gemma-4-31b (#15, 36 GiB, ~290 tok/s) | glm-5.3-flash (#6, 184 GiB, ~340 tok/s) |
|  | MI325X | 256 GiB | 6 TB/s | gemma-4-31b (#15, 64 GiB, ~71 tok/s) | gemma-4-31b (#15, 36 GiB, ~130 tok/s) | glm-5.3-flash (#6, 184 GiB, ~220 tok/s) |
|  | B300 | 268 of 288 GiB | 8 TB/s | gemma-4-31b (#15, 64 GiB, ~93 tok/s) | gemma-4-31b (#15, 36 GiB, ~160 tok/s) | glm-5.3-flash (#6, 184 GiB, ~250 tok/s) |
|  | MI350X / MI355X | 288 GiB | 8 TB/s | gemma-4-31b (#15, 64 GiB, ~93 tok/s) | gemma-4-31b (#15, 36 GiB, ~160 tok/s) | glm-5.3-flash (#6, 184 GiB, ~250 tok/s) |
| 8-GPU server | 8x L40S | 384 GiB | 8 × 0.864 TB/s | glm-5.3-flash (#6, 322 GiB) | glm-5.3-flash (#6, 322 GiB) | glm-5.3-flash (#6, 195 GiB) |
|  | 8x H100 SXM | 640 GiB | 8 × 3.35 TB/s | mimo-v2.6-pro (#2, 543 GiB, ~110 tok/s) | mimo-v2.6-pro (#2, 543 GiB, ~110 tok/s) | mimo-v2.6-pro (#2, 543 GiB, ~110 tok/s) |
|  | 8x A100 80GB SXM | 640 GiB | 8 × 2.04 TB/s | mimo-v2.6-pro (#2, 543 GiB, ~99 tok/s) | mimo-v2.6-pro (#2, 543 GiB, ~99 tok/s) | mimo-v2.6-pro (#2, 543 GiB, ~99 tok/s) |
|  | 8x H100 NVL | 752 GiB | 8 × 3.9 TB/s | mimo-v2.6-pro (#2, 543 GiB) | mimo-v2.6-pro (#2, 543 GiB) | mimo-v2.6-pro (#2, 543 GiB) |
|  | 8x RTX PRO 6000 Blackwell Server Edition | 768 GiB | 8 × 1.6 TB/s | mimo-v2.6-pro (#2, 543 GiB) | mimo-v2.6-pro (#2, 543 GiB) | mimo-v2.6-pro (#2, 543 GiB) |
|  | 8x Gaudi 3 | 1,024 GiB | 8 × 3.7 TB/s | mimo-v2.6-pro (#2, 543 GiB, ~110 tok/s) | mimo-v2.6-pro (#2, 543 GiB, ~110 tok/s) | mimo-v2.6-pro (#2, 543 GiB, ~110 tok/s) |
|  | 8x H200 SXM | 1,128 GiB | 8 × 4.8 TB/s | mimo-v2.6-pro (#2, 543 GiB, ~110 tok/s) | mimo-v2.6-pro (#2, 543 GiB, ~110 tok/s) | mimo-v2.6-pro (#2, 543 GiB, ~110 tok/s) |
|  | 8x B200 | 1,440 GiB | 8 × 8 TB/s | mimo-v2.6-pro (#2, 543 GiB, ~120 tok/s) | mimo-v2.6-pro (#2, 543 GiB, ~120 tok/s) | mimo-v2.6-pro (#2, 543 GiB, ~120 tok/s) |
|  | 8x MI300X | 1,536 GiB | 8 × 5.3 TB/s | mimo-v2.6-pro (#2, 543 GiB, ~110 tok/s) | mimo-v2.6-pro (#2, 543 GiB, ~110 tok/s) | mimo-v2.6-pro (#2, 543 GiB, ~110 tok/s) |
|  | 8x Rubin | 2,000 GiB | 8 × 16.2 TB/s | Kimi K3 (Max) (#1, 1,470 GiB, ~120 tok/s) | Kimi K3 (Max) (#1, 1,470 GiB, ~120 tok/s) | Kimi K3 (Max) (#1, 1,470 GiB, ~120 tok/s) |
|  | 8x MI325X | 2,048 GiB | 8 × 6 TB/s | Kimi K3 (Max) (#1, 1,470 GiB, ~100 tok/s) | Kimi K3 (Max) (#1, 1,470 GiB, ~100 tok/s) | Kimi K3 (Max) (#1, 1,470 GiB, ~100 tok/s) |
|  | 8x B300 | 2,144 GiB | 8 × 8 TB/s | Kimi K3 (Max) (#1, 1,470 GiB, ~110 tok/s) | Kimi K3 (Max) (#1, 1,470 GiB, ~110 tok/s) | Kimi K3 (Max) (#1, 1,470 GiB, ~110 tok/s) |
|  | 8x MI350X / MI355X | 2,304 GiB | 8 × 8 TB/s | Kimi K3 (Max) (#1, 1,470 GiB, ~110 tok/s) | Kimi K3 (Max) (#1, 1,470 GiB, ~110 tok/s) | Kimi K3 (Max) (#1, 1,470 GiB, ~110 tok/s) |

- **8-bit without FP8 hardware** (RTX 3090, Ryzen AI Max+ 395, A100 80GB SXM, Macs): 8-bit means weights such as GGUF Q8_0 or MLX 8-bit, which need the same memory; a model released in FP8 runs in those formats there.
- **Unified memory**: the GPU shares system RAM, which is several times slower than GPU memory; the speeds show it. macOS gives the GPU 75% of RAM by default, and `sudo sysctl iogpu.wired_limit_mb=<MB>` raises that until the next restart (leave 8–16 GiB for macOS). The Ryzen AI Max+ 395 gives its GPU up to 96 GiB on Windows, and more on Linux, where the kernel's limit can be raised (AMD's guides use about 120 GiB); the DGX Spark about 115 GiB after its operating system.
- **8-GPU servers** list only GPUs sold that way: GeForce cards are left out because NVIDIA's GeForce driver license bars data-center deployment, as are workstation cards, built for desktops of up to four; the RTX PRO 6000 appears as its Server Edition. Speeds are estimated for boards that link every GPU to every other (NVLink, Infinity Fabric, Gaudi's on-board network); servers that connect GPUs over PCIe (L40S, H100 NVL, RTX PRO 6000 Blackwell Server Edition) show none, for lack of measurements. For MoE models they assume a latency-tuned engine: the measurement behind them is TensorRT-LLM's fastest DeepSeek-R1 setup, and default setups have run at about half that speed.
- **Not listed**: rack-scale systems (GB200/GB300 NVL72, AMD Helios with MI455X), which come as 72-GPU racks, and AMD's MI440X 8-GPU platform, whose memory is not yet published. B200, B300 and Rubin show the memory an HGX board gives each GPU; Rubin's figures (2 TB and 130 TB/s per board) are from NVIDIA's US page of 2026-10-15, while its other pages list 2.3 TB and 176 TB/s. The DGX Station counts its 252 GiB of GPU memory; its 496 GiB of CPU memory is about 18× slower.

## Full Leaderboard

*Release* is the precision a model's makers published its weights in, read from its files on Hugging Face (*?* where none were found: 16-bit is assumed). *GiB* columns are the memory each model needs as released and quantized to 8 or 4 bits, never more than as released (see [Memory formula](#memory-formula)). *Fits on* is the smallest of the most widely deployed GPUs (RTX 4090, RTX 5090, L40S, H100 SXM, RTX PRO 6000 Blackwell, H200 SXM / NVL, B200, B300) that holds the model at 8-bit, or else an 8-GPU server of the smallest of them; *Speed* is the estimated generation speed for one request there (see [Speed formula](#speed-formula)).

| Rank | Model | Score | Params (B) | Arch | Release | GiB as released | GiB 8-bit | GiB 4-bit | Fits on | Speed |
|------|-------|-------|------------|------|---------|-----------------|-----------|-----------|---------|-------|
| 1 | Kimi K3 (Max) | 1487 | 2800 (104) | MoE | 4-bit | 1,459 | 1,459 | 1,459 | 8x B300 | ~110 tok/s |
| 2 | mimo-v2.6-pro | 1479 | 1000 (42) | MoE | 4-bit | 533 | 533 | 533 | 8x H100 SXM | ~110 tok/s |
| 3 | glm-5.3-max | 1478 | 753 (40) | MoE | 8-bit | 709 | 709 | 426 | 8x RTX PRO 6000 Blackwell Server Edition | — |
| 4 | glm-5.2-max | 1475 | 753 (40) | MoE | 16-bit | 1,409 | 749 | 426 | 8x H200 SXM | ~100 tok/s |
| 5 | deepseek-v4.1-flash-max | 1474 | 763 (16) | MoE | 4-bit | 481 | 481 | 432 | 8x H100 SXM | ~120 tok/s |
| 6 | glm-5.3-flash | 1473 | 320 (18) | MoE | 8-bit | 311 | 311 | 184 | 8x L40S | — |
| 7 | mimo-v2.5-pro | 1467 | 1000 (42) | MoE | 8-bit | 968 | 968 | 564 | 8x H200 SXM | ~100 tok/s |
| 8 | glm-5.1 | 1465 | 744 (40) | MoE | 16-bit | 1,410 | 740 | 421 | 8x H200 SXM | ~100 tok/s |
| 9 | deepseek-v4-pro-high-20260813 | 1464 | 1600 (49) | MoE | 4-bit | 837 | 837 | 837 | 8x H200 SXM | ~110 tok/s |
| 10 | kimi-k2.6 | 1460 | 1000 (32) | MoE | 4-bit | 560 | 560 | 560 | 8x H100 SXM | ~110 tok/s |
| 11 | deepseek-v4-pro | 1457 | 1600 (49) | MoE | 4-bit | 837 | 837 | 837 | 8x H200 SXM | ~110 tok/s |
| 12 | glm-5 | 1457 | 744 (40) | MoE | 16-bit | 1,410 | 740 | 421 | 8x H200 SXM | ~100 tok/s |
| 13 | deepseek-v4-pro-high-preview | 1454 | 1600 (49) | MoE | 4-bit | 837 | 837 | 837 | 8x H200 SXM | ~110 tok/s |
| 14 | hy3 | 1453 | 299 (21) | MoE | 16-bit | 562 | 301 | 173 | 8x L40S | — |
| 15 | gemma-4-31b | 1452 | 30.7 | Dense | 16-bit | 63.8 | 35.8 | 22.7 | L40S | ~21 tok/s |
| 16 | mimo-v2.6-flash | 1450 | 309 (15) | MoE | 4-bit | 167 | 167 | 167 | B200 | ~280 tok/s |
| 17 | kimi-k2.5-thinking | 1449 | 1000 (32) | MoE | 4-bit | 560 | 560 | 560 | 8x H100 SXM | ~110 tok/s |
| 18 | qwen3.5-397b-a17b | 1441 | 397 (17) | MoE | 16-bit | 757 | 397 | 227 | 8x H100 SXM | ~110 tok/s |
| 19 | glm-4.7 | 1441 | 357 (32) | MoE | 16-bit | 673 | 358 | 205 | 8x H100 SXM | ~100 tok/s |
| 20 | minimax-m3 | 1441 | 428 (23) | MoE | 16-bit | 801 | 428 | 245 | 8x H100 SXM | ~110 tok/s |
| 21 | inkling | 1440 | 975 (41) | MoE | 16-bit | 1,779 | 968 | 550 | 8x H200 SXM | ~100 tok/s |
| 22 | deepseek-v4-flash-high-preview | 1438 | 284 (13) | MoE | 4-bit | 161 | 161 | 161 | B200 | ~300 tok/s |
| 23 | qwen3.8-27b | 1437 | 27 | Dense | 16-bit | 57.2 | 32.2 | 20.6 | L40S | ~24 tok/s |
| 24 | gemma-4-26b-a4b | 1437 | 25.2 (3.8) | MoE | 16-bit | 53.6 | 30.4 | 19.6 | RTX 5090 | ~190 tok/s |
| 25 | deepseek-v4-flash | 1435 | 284 (13) | MoE | 4-bit | 161 | 161 | 161 | B200 | ~300 tok/s |
| 26 | mimo-v2.5 | 1433 | 310.8 (15) | MoE | 8-bit | 299 | 299 | 179 | 8x L40S | — |
| 27 | kimi-k2.5-instant | 1430 | 1000 (32) | MoE | 4-bit | 560 | 560 | 560 | 8x H100 SXM | ~110 tok/s |
| 28 | kimi-k2-thinking-turbo | 1430 | 1000 (32) | MoE | 4-bit | 559 | 559 | 559 | 8x H100 SXM | ~110 tok/s |
| 29 | mistral-medium-3.5 | 1426 | 128 | Dense | 8-bit | 130 | 130 | 77.0 | H200 SXM / NVL | ~28 tok/s |
| 30 | nvidia-nemotron-3-ultra-550b-a55b-nvfp4 | 1426 | 550 (55) | MoE | 4-bit | 313 | 313 | 313 | 8x L40S | — |
| 31 | deepseek-v3.2-exp-thinking | 1425 | 685 (37) | MoE | 8-bit | 648 | 648 | 388 | 8x RTX PRO 6000 Blackwell Server Edition | — |
| 32 | deepseek-v3.2 | 1424 | 685 (37) | MoE | 8-bit | 648 | 648 | 388 | 8x RTX PRO 6000 Blackwell Server Edition | — |
| 33 | muse-glimmer | 1424 | 30 | Dense | 16-bit | 61.0 | 35.1 | 22.3 | L40S | ~21 tok/s |
| 34 | glm-4.6 | 1424 | 357 (32) | MoE | 16-bit | 670 | 358 | 205 | 8x H100 SXM | ~100 tok/s |
| 35 | deepseek-v3.2-thinking | 1422 | 685 (37) | MoE | 8-bit | 648 | 648 | 388 | 8x RTX PRO 6000 Blackwell Server Edition | — |
| 36 | qwen3-235b-a22b-instruct-2507 | 1422 | 235 (22) | MoE | 16-bit | 443 | 237 | 137 | B300 | ~160 tok/s |
| 37 | deepseek-v3.2-exp | 1421 | 685 (37) | MoE | 8-bit | 648 | 648 | 388 | 8x RTX PRO 6000 Blackwell Server Edition | — |
| 38 | deepseek-r1-0528 | 1421 | 684.5 (37) | MoE | 8-bit | 647 | 647 | 388 | 8x RTX PRO 6000 Blackwell Server Edition | — |
| 39 | kimi-k2-0905-preview | 1418 | 1000 (32) | MoE | 8-bit | 964 | 964 | 564 | 8x H200 SXM | ~110 tok/s |
| 40 | kimi-k2-0711-preview | 1417 | 1000 (32) | MoE | 8-bit | 964 | 964 | 564 | 8x H200 SXM | ~110 tok/s |
| 41 | deepseek-v3.1 | 1417 | 685 (37) | MoE | 8-bit | 647 | 647 | 388 | 8x RTX PRO 6000 Blackwell Server Edition | — |
| 42 | qwen3.5-122b-a10b | 1416 | 125 (10) | MoE | 16-bit | 238 | 129 | 75.3 | H200 SXM / NVL | ~190 tok/s |
| 43 | deepseek-v3.1-terminus-thinking | 1416 | 685 (37) | MoE | 8-bit | 647 | 647 | 388 | 8x RTX PRO 6000 Blackwell Server Edition | — |
| 44 | nvidia-nemotron-3.5-super-vl-120b-a12b-nvfp4 | 1415 | 120 (12) | MoE | 4-bit | 72.6 | 72.6 | 72.6 | H100 SXM | ~200 tok/s |
| 45 | deepseek-v3.1-thinking | 1415 | 685 (37) | MoE | 8-bit | 647 | 647 | 388 | 8x RTX PRO 6000 Blackwell Server Edition | — |
| 46 | minimax-m2.7 | 1415 | 230 (10) | MoE | 8-bit | 220 | 220 | 134 | B300 | ~260 tok/s |
| 47 | deepseek-v3.1-terminus | 1414 | 685 (37) | MoE | 8-bit | 647 | 647 | 388 | 8x RTX PRO 6000 Blackwell Server Edition | — |
| 48 | mistral-large-3 | 1414 | 675 (41) | MoE | 8-bit | 640 | 640 | 383 | 8x RTX PRO 6000 Blackwell Server Edition | — |
| 49 | qwen3-vl-235b-a22b-instruct | 1413 | 235 (22) | MoE | 16-bit | 444 | 237 | 137 | B300 | ~160 tok/s |
| 50 | glm-4.5 | 1410 | 355 (32) | MoE | 16-bit | 673 | 356 | 204 | 8x H100 SXM | ~100 tok/s |

<details><summary>Show remaining 175 models</summary>

| Rank | Model | Score | Params (B) | Arch | Release | GiB as released | GiB 8-bit | GiB 4-bit | Fits on | Speed |
|------|-------|-------|------------|------|---------|-----------------|-----------|-----------|---------|-------|
| 51 | hunyuan-hy3-preview | 1409 | 298.8 (21) | MoE | 16-bit | 562 | 300 | 172 | 8x L40S | — |
| 52 | qwen3.5-27b | 1408 | 27 | Dense | 16-bit | 57.2 | 32.2 | 20.6 | L40S | ~24 tok/s |
| 53 | Inkling Small | 1403 | 266 (12) | MoE | 16-bit | 501 | 268 | 154 | 8x L40S | — |
| 54 | qwen3-235b-a22b-no-thinking | 1402 | 235 (22) | MoE | 16-bit | 443 | 237 | 137 | B300 | ~160 tok/s |
| 55 | longcat-flash-chat | 1400 | 560 (27) | MoE | 16-bit | 1,052 | 558 | 318 | 8x H100 SXM | ~100 tok/s |
| 56 | qwen3-235b-a22b-thinking-2507 | 1399 | 235 (22) | MoE | 16-bit | 443 | 237 | 137 | B300 | ~160 tok/s |
| 57 | qwen3-next-80b-a3b-instruct | 1398 | 80 (3) | MoE | 16-bit | 157 | 84.5 | 50.2 | RTX PRO 6000 Blackwell | ~220 tok/s |
| 58 | deepseek-r1 | 1398 | 685 (37) | MoE | 8-bit | 647 | 647 | 388 | 8x RTX PRO 6000 Blackwell Server Edition | — |
| 59 | deepseek-v3-0324 | 1395 | 671 (37) | MoE | 8-bit | 647 | 647 | 380 | 8x RTX PRO 6000 Blackwell Server Edition | — |
| 60 | qwen3-vl-235b-a22b-thinking | 1394 | 235 (22) | MoE | 16-bit | 444 | 237 | 137 | B300 | ~160 tok/s |
| 61 | qwen3.5-35b-a3b | 1394 | 35 (3) | MoE | ? | 70.7 | 40.1 | 25.1 | L40S | ~140 tok/s |
| 62 | step-3.5-flash | 1392 | 199.4 (11) | MoE | 16-bit | 377 | 202 | 117 | B300 | ~240 tok/s |
| 63 | mimo-v2-flash (non-thinking) | 1391 | 309 (15) | MoE | 8-bit | 297 | 297 | 178 | 8x L40S | — |
| 64 | minimax-m2.5 | 1390 | 230 (10) | MoE | 8-bit | 220 | 220 | 134 | B300 | ~260 tok/s |
| 65 | qwen3-coder-480b-a35b-instruct | 1387 | 480 (35) | MoE | ? | 900 | 479 | 274 | 8x H100 SXM | ~100 tok/s |
| 66 | mimo-v2-flash (thinking) | 1386 | 309 (15) | MoE | 8-bit | 297 | 297 | 178 | 8x L40S | — |
| 67 | minimax-m2.1-preview | 1384 | 230 (10) | MoE | 8-bit | 220 | 220 | 134 | B300 | ~260 tok/s |
| 68 | qwen3-30b-a3b-instruct-2507 | 1382 | 30 (3) | MoE | ? | 61.4 | 35.1 | 22.3 | L40S | ~140 tok/s |
| 69 | trinity-large-preview | 1378 | 398.6 (13) | MoE | 16-bit | 748 | 399 | 228 | 8x H100 SXM | ~110 tok/s |
| 70 | glm-4.6v | 1378 | 108 (12) | MoE | 16-bit | 206 | 112 | 65.8 | H200 SXM / NVL | ~170 tok/s |
| 71 | qwen3-235b-a22b | 1374 | 235 (22) | MoE | 16-bit | 443 | 237 | 137 | B300 | ~160 tok/s |
| 72 | glm-4.5-air | 1372 | 106 (12) | MoE | 16-bit | 211 | 110 | 64.7 | H200 SXM / NVL | ~170 tok/s |
| 73 | qwen3-next-80b-a3b-thinking | 1369 | 80 (3) | MoE | ? | 155 | 84.5 | 50.2 | RTX PRO 6000 Blackwell | ~220 tok/s |
| 74 | trinity-large-thinking | 1367 | 398.6 (13) | MoE | 16-bit | 748 | 399 | 228 | 8x H100 SXM | ~110 tok/s |
| 75 | gemma-3-27b-it | 1365 | 27 | Dense | ? | 55.8 | 32.2 | 20.6 | L40S | ~24 tok/s |
| 76 | glm-4.7-flash | 1364 | 31.2 (3) | MoE | 16-bit | 63.7 | 36.3 | 22.9 | L40S | ~140 tok/s |
| 77 | minimax-m1 | 1363 | 456.1 (45.9) | MoE | 16-bit | 855 | 456 | 260 | 8x H100 SXM | ~94 tok/s |
| 78 | nvidia-nemotron-3-super-120b-a12b | 1360 | 120 (12) | MoE | ? | 229 | 124 | 72.6 | H200 SXM / NVL | ~170 tok/s |
| 79 | deepseek-v3 | 1358 | 671 (37) | MoE | 8-bit | 647 | 647 | 380 | 8x RTX PRO 6000 Blackwell Server Edition | — |
| 80 | mistral-small-2506 | 1356 | 24 | Dense | ? | 50.2 | 29.2 | 18.9 | RTX 5090 | ~53 tok/s |
| 81 | intellect-3 | 1355 | 107 (12) | MoE | 16-bit | 205 | 111 | 65.3 | H200 SXM / NVL | ~170 tok/s |
| 82 | command-a-03-2025 | 1353 | 111 | Dense | 16-bit | 212 | 115 | 67.5 | H200 SXM / NVL | ~32 tok/s |
| 83 | glm-4.5v | 1352 | 108 (12) | MoE | 16-bit | 206 | 112 | 65.8 | H200 SXM / NVL | ~170 tok/s |
| 84 | gpt-oss-120b | 1351 | 117 (5.1) | MoE | 4-bit | 66.3 | 66.3 | 66.3 | H100 SXM | ~310 tok/s |
| 85 | nvidia-nemotron-3.5-lightning-30b-a3b-nvfp4 | 1351 | 30 (3) | MoE | 4-bit | 22.3 | 22.3 | 22.3 | RTX 4090 | ~220 tok/s |
| 86 | step-3 | 1348 | 321 (38) | MoE | 16-bit | 603 | 322 | 185 | 8x L40S | — |
| 87 | llama-3.1-nemotron-ultra-253b-v1 | 1347 | 253 | Dense | 16-bit | 477 | 255 | 147 | 8x L40S | — |
| 88 | qwen3-32b | 1346 | 32 | Dense | ? | 65.1 | 37.1 | 23.4 | L40S | ~20 tok/s |
| 89 | minimax-m2 | 1343 | 230 (10) | MoE | 8-bit | 220 | 220 | 134 | B300 | ~260 tok/s |
| 90 | ling-flash-2.0 | 1343 | 103 (6.1) | MoE | 16-bit | 197 | 107 | 63.1 | H200 SXM / NVL | ~250 tok/s |
| 91 | nvidia-llama-3.3-nemotron-super-49b-v1.5 | 1342 | 49 | Dense | ? | 96.8 | 53.9 | 32.9 | H100 SXM | ~49 tok/s |
| 92 | gemma-3-12b-it | 1341 | 12 | Dense | ? | 25.9 | 15.3 | 10.2 | RTX 4090 | ~60 tok/s |
| 93 | granite-4.2-30b | 1340 | 30 | Dense | 16-bit | 60.0 | 35.1 | 22.3 | L40S | ~21 tok/s |
| 94 | qwq-32b | 1335 | 32 | Dense | ? | 65.1 | 37.1 | 23.4 | L40S | ~20 tok/s |
| 95 | llama-3.1-405b-instruct-bf16 | 1335 | 405 | Dense | 16-bit | 760 | 405 | 232 | 8x H100 SXM | ~42 tok/s |
| 96 | llama-3.1-405b-instruct-fp8 | 1333 | 405 | Dense | 8-bit | 405 | 405 | 232 | 8x H100 SXM | ~42 tok/s |
| 97 | olmo-3.1-32b-instruct | 1329 | 32.2 | Dense | 16-bit | 65.5 | 37.3 | 23.5 | L40S | ~20 tok/s |
| 98 | llama-3.3-nemotron-49b-super-v1 | 1327 | 49 | Dense | ? | 96.8 | 53.9 | 32.9 | H100 SXM | ~49 tok/s |
| 99 | llama-4-maverick-17b-128e-instruct | 1326 | 400 (17) | MoE | 16-bit | 754 | 400 | 229 | 8x H100 SXM | ~110 tok/s |
| 100 | qwen3-30b-a3b | 1326 | 30 (3) | MoE | ? | 61.4 | 35.1 | 22.3 | L40S | ~140 tok/s |
| 101 | deepseek-v2.5-1210 | 1323 | 235.7 (21) | MoE | ? | 445 | 238 | 137 | B300 | ~170 tok/s |
| 102 | ring-flash-2.0 | 1322 | 103 (6.1) | MoE | 16-bit | 197 | 107 | 63.1 | H200 SXM / NVL | ~250 tok/s |
| 103 | molmo-2-8b | 1322 | 8.7 | Dense | ? | 19.7 | 12.1 | 8.4 | RTX 4090 | ~80 tok/s |
| 104 | llama-4-scout-17b-16e-instruct | 1321 | 109 (17) | MoE | 16-bit | 208 | 113 | 66.4 | H200 SXM / NVL | ~140 tok/s |
| 105 | llama-3.3-70b-instruct | 1318 | 70 | Dense | ? | 136 | 74.6 | 44.6 | H100 SXM | ~35 tok/s |
| 106 | gemma-3n-e4b-it | 1317 | 7.85 | Dense | ? | 18.1 | 11.2 | 7.9 | RTX 4090 | ~88 tok/s |
| 107 | gpt-oss-20b | 1317 | 21 (3.6) | MoE | 4-bit | 18.3 | 18.3 | 17.2 | RTX 4090 | ~190 tok/s |
| 108 | mistral-large-2407 | 1314 | 123 | Dense | 16-bit | 234 | 127 | 74.2 | H200 SXM / NVL | ~29 tok/s |
| 109 | athene-v2-chat | 1314 | 72 | Dense | ? | 140 | 76.6 | 45.7 | RTX PRO 6000 Blackwell | ~18 tok/s |
| 110 | nvidia-nemotron-3-nano-30b-a3b-bf16 | 1313 | 31.6 (3.6) | MoE | 16-bit | 64.4 | 36.7 | 23.2 | L40S | ~120 tok/s |
| 111 | deepseek-v2.5 | 1307 | 236 (21) | MoE | 16-bit | 445 | 238 | 137 | B300 | ~170 tok/s |
| 112 | athene-70b-0725 | 1306 | 70 | Dense | ? | 136 | 74.6 | 44.6 | H100 SXM | ~35 tok/s |
| 113 | olmo-3-32b-think | 1306 | 32 | Dense | ? | 65.1 | 37.1 | 23.4 | L40S | ~20 tok/s |
| 114 | mistral-large-2411 | 1305 | 122.6 | Dense | ? | 234 | 127 | 74.0 | H200 SXM / NVL | ~29 tok/s |
| 115 | granite-4.1-8b | 1304 | 8 | Dense | ? | 18.4 | 11.4 | 8.0 | RTX 4090 | ~87 tok/s |
| 116 | gemma-3-4b-it | 1303 | 4 | Dense | ? | 11.0 | 7.4 | 5.7 | RTX 4090 | ~160 tok/s |
| 117 | qwen2.5-72b-instruct | 1302 | 72 | Dense | ? | 140 | 76.6 | 45.7 | RTX PRO 6000 Blackwell | ~18 tok/s |
| 118 | mistral-small-3.1-24b-instruct-2503 | 1302 | 24 | Dense | ? | 50.2 | 29.2 | 18.9 | RTX 5090 | ~53 tok/s |
| 119 | llama-3.1-nemotron-70b-instruct | 1298 | 70 | Dense | ? | 136 | 74.6 | 44.6 | H100 SXM | ~35 tok/s |
| 120 | llama-3.1-70b-instruct | 1293 | 70 | Dense | ? | 136 | 74.6 | 44.6 | H100 SXM | ~35 tok/s |
| 121 | granite-4.2-8b | 1290 | 8 | Dense | 16-bit | 19.9 | 11.4 | 8.0 | RTX 4090 | ~87 tok/s |
| 122 | granite-4.2-3b | 1289 | 3 | Dense | ? | 9.1 | 6.5 | 5.2 | RTX 4090 | ~200 tok/s |
| 123 | gemma-2-27b-it | 1289 | 27 | Dense | ? | 55.8 | 32.2 | 20.6 | L40S | ~24 tok/s |
| 124 | jamba-1.5-large | 1289 | 398 (94) | MoE | 16-bit | 748 | 398 | 228 | 8x H100 SXM | ~75 tok/s |
| 125 | ibm-granite-h-small | 1287 | 32.2 (9) | MoE | ? | 65.5 | 37.3 | 23.5 | L40S | ~56 tok/s |
| 126 | llama-3.1-nemotron-51b-instruct | 1286 | 51 | Dense | ? | 100 | 55.8 | 34.0 | H100 SXM | ~47 tok/s |
| 127 | olmo-3.1-32b-think | 1286 | 32.2 | Dense | 16-bit | 65.5 | 37.3 | 23.5 | L40S | ~20 tok/s |
| 128 | llama-3.1-tulu-3-70b | 1286 | 70 | Dense | ? | 136 | 74.6 | 44.6 | H100 SXM | ~35 tok/s |
| 129 | gemma-2-9b-it-simpo | 1280 | 9 | Dense | ? | 20.3 | 12.4 | 8.5 | RTX 4090 | ~78 tok/s |
| 130 | nemotron-4-340b-instruct | 1277 | 340 | Dense | ? | 639 | 341 | 195 | 8x L40S | — |
| 131 | llama-3-70b-instruct | 1276 | 70 | Dense | ? | 136 | 74.6 | 44.6 | H100 SXM | ~35 tok/s |
| 132 | command-r-plus-08-2024 | 1276 | 104 | Dense | ? | 199 | 108 | 63.6 | H200 SXM / NVL | ~34 tok/s |
| 133 | mistral-small-24b-instruct-2501 | 1274 | 24 | Dense | ? | 50.2 | 29.2 | 18.9 | RTX 5090 | ~53 tok/s |
| 134 | qwen2.5-coder-32b-instruct | 1270 | 32 | Dense | ? | 65.1 | 37.1 | 23.4 | L40S | ~20 tok/s |
| 135 | c4ai-aya-expanse-32b | 1267 | 32 | Dense | ? | 65.1 | 37.1 | 23.4 | L40S | ~20 tok/s |
| 136 | gemma-2-9b-it | 1267 | 9 | Dense | ? | 20.3 | 12.4 | 8.5 | RTX 4090 | ~78 tok/s |
| 137 | deepseek-coder-v2 | 1265 | 236 (21) | MoE | 16-bit | 445 | 238 | 137 | B300 | ~170 tok/s |
| 138 | qwen2-72b-instruct | 1261 | 72 | Dense | ? | 140 | 76.6 | 45.7 | RTX PRO 6000 Blackwell | ~18 tok/s |
| 139 | command-r-plus | 1261 | 104 | Dense | ? | 199 | 108 | 63.6 | H200 SXM / NVL | ~34 tok/s |
| 140 | phi-4 | 1256 | 14 | Dense | 16-bit | 30.8 | 17.3 | 11.3 | RTX 4090 | ~52 tok/s |
| 141 | olmo-2-0325-32b-instruct | 1251 | 32 | Dense | ? | 65.1 | 37.1 | 23.4 | L40S | ~20 tok/s |
| 142 | command-r-08-2024 | 1250 | 32.3 | Dense | ? | 65.7 | 37.4 | 23.5 | L40S | ~20 tok/s |
| 143 | jamba-1.5-mini | 1239 | 52 (12) | MoE | 16-bit | 102 | 56.8 | 34.6 | H100 SXM | ~130 tok/s |
| 144 | ministral-8b-2410 | 1237 | 8 | Dense | ? | 18.4 | 11.4 | 8.0 | RTX 4090 | ~87 tok/s |
| 145 | qwen1.5-110b-chat | 1234 | 110 | Dense | ? | 210 | 114 | 67.0 | H200 SXM / NVL | ~32 tok/s |
| 146 | qwen1.5-72b-chat | 1233 | 72 | Dense | ? | 140 | 76.6 | 45.7 | RTX PRO 6000 Blackwell | ~18 tok/s |
| 147 | mixtral-8x22b-instruct-v0.1 | 1229 | 140.8 (39.6) | MoE | ? | 268 | 144 | 84.2 | B200 | ~100 tok/s |
| 148 | command-r | 1226 | 35 | Dense | ? | 70.7 | 40.1 | 25.1 | L40S | ~18 tok/s |
| 149 | llama-3-8b-instruct | 1223 | 8 | Dense | ? | 18.4 | 11.4 | 8.0 | RTX 4090 | ~87 tok/s |
| 150 | c4ai-aya-expanse-8b | 1223 | 8 | Dense | ? | 18.4 | 11.4 | 8.0 | RTX 4090 | ~87 tok/s |
| 151 | llama-3.1-tulu-3-8b | 1220 | 8 | Dense | ? | 18.4 | 11.4 | 8.0 | RTX 4090 | ~87 tok/s |
| 152 | zephyr-orpo-141b-A35b-v0.1 | 1213 | 141 (35) | MoE | ? | 268 | 145 | 84.3 | B200 | ~120 tok/s |
| 153 | yi-1.5-34b-chat | 1212 | 34 | Dense | ? | 68.8 | 39.1 | 24.5 | L40S | ~19 tok/s |
| 154 | llama-3.1-8b-instruct | 1211 | 8 | Dense | ? | 18.4 | 11.4 | 8.0 | RTX 4090 | ~87 tok/s |
| 155 | granite-3.1-8b-instruct | 1208 | 8 | Dense | ? | 18.4 | 11.4 | 8.0 | RTX 4090 | ~87 tok/s |
| 156 | qwen1.5-32b-chat | 1204 | 32 | Dense | ? | 65.1 | 37.1 | 23.4 | L40S | ~20 tok/s |
| 157 | gemma-2-2b-it | 1200 | 2 | Dense | ? | 7.2 | 5.5 | 4.6 | RTX 4090 | ~280 tok/s |
| 158 | phi-3-medium-4k-instruct | 1198 | 14 | Dense | ? | 29.6 | 17.3 | 11.3 | RTX 4090 | ~52 tok/s |
| 159 | mixtral-8x7b-instruct-v0.1 | 1197 | 44.8 (12.6) | MoE | ? | 88.9 | 49.7 | 30.5 | H100 SXM | ~130 tok/s |
| 160 | dbrx-instruct-preview | 1195 | 132 (36) | MoE | ? | 251 | 136 | 79.3 | B200 | ~110 tok/s |
| 161 | qwen1.5-14b-chat | 1191 | 14 | Dense | ? | 29.6 | 17.3 | 11.3 | RTX 4090 | ~52 tok/s |
| 162 | internlm2_5-20b-chat | 1190 | 20 | Dense | ? | 42.8 | 25.2 | 16.7 | RTX 5090 | ~63 tok/s |
| 163 | deepseek-llm-67b-chat | 1185 | 67 | Dense | ? | 130 | 71.6 | 42.9 | H100 SXM | ~36 tok/s |
| 164 | wizardlm-70b | 1184 | 70 | Dense | ? | 136 | 74.6 | 44.6 | H100 SXM | ~35 tok/s |
| 165 | yi-34b-chat | 1183 | 34 | Dense | ? | 68.8 | 39.1 | 24.5 | L40S | ~19 tok/s |
| 166 | granite-3.0-8b-instruct | 1183 | 8 | Dense | ? | 18.4 | 11.4 | 8.0 | RTX 4090 | ~87 tok/s |
| 167 | openchat-3.5 | 1183 | 7.24 | Dense | ? | 17.0 | 10.6 | 7.5 | RTX 4090 | ~95 tok/s |
| 168 | openchat-3.5-0106 | 1182 | 7.24 | Dense | ? | 17.0 | 10.6 | 7.5 | RTX 4090 | ~95 tok/s |
| 169 | gemma-1.1-7b-it | 1182 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 170 | snowflake-arctic-instruct | 1180 | 478.6 (17) | MoE | ? | 897 | 478 | 273 | 8x H100 SXM | ~110 tok/s |
| 171 | granite-3.1-2b-instruct | 1178 | 2 | Dense | ? | 7.2 | 5.5 | 4.6 | RTX 4090 | ~280 tok/s |
| 172 | tulu-2-dpo-70b | 1177 | 70 | Dense | ? | 136 | 74.6 | 44.6 | H100 SXM | ~35 tok/s |
| 173 | openhermes-2.5-mistral-7b | 1176 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 174 | vicuna-33b | 1173 | 33 | Dense | ? | 67.0 | 38.1 | 23.9 | L40S | ~19 tok/s |
| 175 | phi-3-small-8k-instruct | 1171 | 7.4 | Dense | ? | 17.3 | 10.8 | 7.6 | RTX 4090 | ~93 tok/s |
| 176 | starling-lm-7b-beta | 1171 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 177 | llama-2-70b-chat | 1170 | 70 | Dense | ? | 136 | 74.6 | 44.6 | H100 SXM | ~35 tok/s |
| 178 | starling-lm-7b-alpha | 1167 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 179 | llama-3.2-3b-instruct | 1166 | 3 | Dense | ? | 9.1 | 6.5 | 5.2 | RTX 4090 | ~200 tok/s |
| 180 | nous-hermes-2-mixtral-8x7b-dpo | 1164 | 44.8 (12.6) | MoE | ? | 88.9 | 49.7 | 30.5 | H100 SXM | ~130 tok/s |
| 181 | granite-3.0-2b-instruct | 1156 | 2 | Dense | ? | 7.2 | 5.5 | 4.6 | RTX 4090 | ~280 tok/s |
| 182 | llama2-70b-steerlm-chat | 1154 | 70 | Dense | ? | 136 | 74.6 | 44.6 | H100 SXM | ~35 tok/s |
| 183 | qwq-32b-preview | 1154 | 32 | Dense | ? | 65.1 | 37.1 | 23.4 | L40S | ~20 tok/s |
| 184 | solar-10.7b-instruct-v1.0 | 1152 | 10.7 | Dense | ? | 23.4 | 14.1 | 9.5 | RTX 4090 | ~66 tok/s |
| 185 | dolphin-2.2.1-mistral-7b | 1152 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 186 | mpt-30b-chat | 1150 | 30 | Dense | ? | 61.4 | 35.1 | 22.3 | L40S | ~21 tok/s |
| 187 | wizardlm-13b | 1149 | 13 | Dense | ? | 27.7 | 16.3 | 10.8 | RTX 4090 | ~55 tok/s |
| 188 | mistral-7b-instruct-v0.2 | 1149 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 189 | falcon-180b-chat | 1148 | 180 | Dense | ? | 341 | 183 | 106 | B300 | ~32 tok/s |
| 190 | qwen1.5-7b-chat | 1143 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 191 | phi-3-mini-4k-instruct-june-2024 | 1143 | 3.8 | Dense | ? | 10.6 | 7.3 | 5.6 | RTX 4090 | ~170 tok/s |
| 192 | vicuna-13b | 1141 | 13 | Dense | ? | 27.7 | 16.3 | 10.8 | RTX 4090 | ~55 tok/s |
| 193 | llama-2-13b-chat | 1141 | 13 | Dense | ? | 27.7 | 16.3 | 10.8 | RTX 4090 | ~55 tok/s |
| 194 | qwen-14b-chat | 1139 | 14 | Dense | ? | 29.6 | 17.3 | 11.3 | RTX 4090 | ~52 tok/s |
| 195 | gemma-7b-it | 1137 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 196 | codellama-34b-instruct | 1136 | 34 | Dense | ? | 68.8 | 39.1 | 24.5 | L40S | ~19 tok/s |
| 197 | zephyr-7b-beta | 1130 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 198 | phi-3-mini-128k-instruct | 1129 | 3.8 | Dense | ? | 10.6 | 7.3 | 5.6 | RTX 4090 | ~170 tok/s |
| 199 | phi-3-mini-4k-instruct | 1128 | 3.8 | Dense | ? | 10.6 | 7.3 | 5.6 | RTX 4090 | ~170 tok/s |
| 200 | guanaco-33b | 1127 | 33 | Dense | ? | 67.0 | 38.1 | 23.9 | L40S | ~19 tok/s |
| 201 | zephyr-7b-alpha | 1127 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 202 | stripedhyena-nous-7b | 1121 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 203 | codellama-70b-instruct | 1119 | 70 | Dense | ? | 136 | 74.6 | 44.6 | H100 SXM | ~35 tok/s |
| 204 | gemma-1.1-2b-it | 1116 | 2 | Dense | ? | 7.2 | 5.5 | 4.6 | RTX 4090 | ~280 tok/s |
| 205 | vicuna-7b | 1115 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 206 | smollm2-1.7b-instruct | 1114 | 1.7 | Dense | ? | 6.7 | 5.2 | 4.4 | RTX 4090 | ~310 tok/s |
| 207 | llama-3.2-1b-instruct | 1111 | 1 | Dense | ? | 5.4 | 4.5 | 4.1 | RTX 4090 | ~430 tok/s |
| 208 | mistral-7b-instruct | 1110 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 209 | llama-2-7b-chat | 1108 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 210 | gemma-2b-it | 1093 | 2 | Dense | ? | 7.2 | 5.5 | 4.6 | RTX 4090 | ~280 tok/s |
| 211 | qwen1.5-4b-chat | 1091 | 4 | Dense | ? | 11.0 | 7.4 | 5.7 | RTX 4090 | ~160 tok/s |
| 212 | olmo-7b-instruct | 1073 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 213 | koala-13b | 1071 | 13 | Dense | ? | 27.7 | 16.3 | 10.8 | RTX 4090 | ~55 tok/s |
| 214 | alpaca-13b | 1070 | 13 | Dense | ? | 27.7 | 16.3 | 10.8 | RTX 4090 | ~55 tok/s |
| 215 | gpt4all-13b-snoozy | 1067 | 13 | Dense | ? | 27.7 | 16.3 | 10.8 | RTX 4090 | ~55 tok/s |
| 216 | mpt-7b-chat | 1063 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 217 | chatglm3-6b | 1056 | 6 | Dense | ? | 14.7 | 9.4 | 6.9 | RTX 4090 | ~110 tok/s |
| 218 | RWKV-4-Raven-14B | 1042 | 14 | Dense | ? | 29.6 | 17.3 | 11.3 | RTX 4090 | ~52 tok/s |
| 219 | chatglm2-6b | 1024 | 6 | Dense | ? | 14.7 | 9.4 | 6.9 | RTX 4090 | ~110 tok/s |
| 220 | oasst-pythia-12b | 1023 | 12 | Dense | ? | 25.9 | 15.3 | 10.2 | RTX 4090 | ~60 tok/s |
| 221 | chatglm-6b | 995 | 6 | Dense | ? | 14.7 | 9.4 | 6.9 | RTX 4090 | ~110 tok/s |
| 222 | fastchat-t5-3b | 992 | 3 | Dense | ? | 9.1 | 6.5 | 5.2 | RTX 4090 | ~200 tok/s |
| 223 | dolly-v2-12b | 982 | 12 | Dense | ? | 25.9 | 15.3 | 10.2 | RTX 4090 | ~60 tok/s |
| 224 | llama-13b | 975 | 13 | Dense | ? | 27.7 | 16.3 | 10.8 | RTX 4090 | ~55 tok/s |
| 225 | stablelm-tuned-alpha-7b | 953 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |

</details>

## Architecture

**Data flow:** Arena.ai leaderboard → parameter resolution (4-strategy fallback) → release lookup (Hugging Face) → memory and speed estimates → hardware table

The parameter resolution chain prioritizes accuracy: manual overrides catch known errors, [Artificial Analysis](https://artificialanalysis.ai) supplies sizes from its model pages (its model list maps Arena names to pages, and page results are cached in the repository so each run only reads new models), name parsing extracts `{N}B` patterns as a fallback, and guessed page names handle the long tail. Results where active parameters exceed the total are rejected.

### Memory formula

```
Memory (GiB) = weights
             + KV cache: 4 GiB (16,384 tokens × 256 KiB; 2 GiB under 15B parameters)
             + 1.5 GiB per GPU
```

| Column | Weights | GiB per billion parameters | Files it covers |
|--------|---------|----------------------------|-----------------|
| As released | the published weight files on Hugging Face (2 bytes per parameter when none are found) | 1.86 for 16-bit, about 0.93 for FP8, 0.52–0.60 for 4-bit | the makers' own release |
| 8-bit | 1.06 bytes per parameter, or the release if smaller | 0.987 | FP8, GGUF Q8_0, MLX 8-bit |
| 4-bit | 0.6 bytes per parameter, or the release if smaller | 0.559 | GGUF Q4_K_M, AWQ, GPTQ, NVFP4, MLX 4-bit |

- **GiB** (1,073,741,824 bytes) is the unit GPU and Mac memory is sold in, though it is labelled GB: an "80 GB" H100 holds 80 GiB. A billion parameters at 2 bytes each is 2 billion bytes, or 1.86 GiB.
- **Bytes per parameter** are those of real files: quantized checkpoints also store scale factors and usually keep the embeddings and output layer in 16-bit, so 4-bit files of 30B-and-up models come to 0.56–0.64 bytes per parameter rather than 0.5, and 8-bit files to 1.00–1.07.
- **As released**: many current models are published only in 8-bit or 4-bit (DeepSeek, GLM-5, Kimi, gpt-oss), so their full-quality size is that of their own files, which the tool sums from the model's Hugging Face repository (the one its Artificial Analysis page links). Quantizing such a model further is possible; making it bigger gains nothing, so no column exceeds the release.
- **MoE models** count all their parameters: every expert must sit in memory, even though each token uses only a few.
- **KV cache**: the attention keys and values kept for every token in context, reserved for one 16,384-token conversation. It does not shrink when the weights are quantized, and depends on the attention design: 20–100 KiB per token for most current MoE models (DeepSeek's and Kimi's MLA, hybrid and sliding-window attention), 190–370 KiB for models such as Qwen3-235B, GLM-4.6 and Llama 3.3 70B, and 500 KiB for Llama 3.1 405B. This tool does not know it per model; 256 KiB is a middle value, generous for the newest models and short for the oldest large ones. Longer contexts and several users at once need more.
- **1.5 GiB per GPU** covers the CUDA or Metal context, the framework and activations. Split across 8 GPUs, a model needs it on each.
- **95%**: a model fits a GPU when it needs at most 95% of its memory. GPUs report a little less than their label (an L40S with ECC on shows 45 of 48 GiB), and engines want headroom: vLLM uses 90% unless told otherwise (`--gpu-memory-utilization 0.95`). On computers whose GPU shares system RAM the whole share the system gives the GPU counts, as it already leaves the system its own.
- Example: a 70B dense model at 8-bit needs 70 × 0.987 + 4 + 1.5 = 74.6 GiB, so it fits an H100 (76 of 80 GiB).

### Speed formula

Generating each token reads every active weight from memory once, so with one request at a time memory bandwidth, not compute, sets the speed:

```
Time per token (ms) = bytes per token ÷ (0.8 × bandwidth × GPUs) + fixed cost
Bytes per token     = active parameters × bytes per parameter (× 1.15 for MoE)
Speed (tokens/s)    = 1000 ÷ time per token
```

- **Bytes per parameter** are the column's: the release's own files as released, 1.06 at 8-bit and 0.6 at 4-bit unless the release is smaller.
- **Active parameters** are all the parameters of a dense model, and only those used per token for an MoE model (37B of DeepSeek V3's 671B). MoE models read somewhat more per token than that (the router, and shared layers that quantized files keep at higher precision); ×1.15 is fitted on the measurements.
- **0.8**: software reads weights at about 80% of the memory's peak bandwidth.
- **Fixed cost per token** (starting the work for every layer, synchronising, sampling): 1 ms for dense and 2 ms for MoE models on NVIDIA, AMD and Intel hardware, 1 and 5 ms on Apple Silicon. When 8 GPUs split one request they also exchange results at every layer: 3 ms more for dense and 6 ms for MoE models. This is why small and MoE models fall short of their bandwidth limit on fast GPUs, and why 8 GPUs do not make one request 8× faster.
- Example: a 30B dense model at 8-bit on an H100 SXM reads 31.8 GB per token: 31.8 ÷ (0.8 × 3,350 GB/s) = 11.9 ms, plus 1 ms, so about 78 tokens/s.
- **Assumes** one request, a short context (a long context's KV cache is read for every token too: a small MoE model runs about 20% slower at 8K tokens and half as fast at 32K), no speculative decoding or multi-token prediction (typically 1.5–2.5× faster), and an engine suited to the hardware: vLLM, SGLang or TensorRT-LLM on data-center GPUs, where llama.cpp runs 30–55% below these estimates for small models.
- **Reading the prompt** is a separate cost the tables do not estimate, limited by compute rather than memory (about 2 × active parameters floating-point operations per prompt token). It is where computers whose GPU shares system RAM fall furthest behind: llama.cpp reads a 7B model's prompt at about 1,700 tokens/s on a Ryzen AI Max+ 395, 3,200 on an M5 Max and 3,300 on a DGX Spark, against 14,800 on an RTX 4090 ([Apple](https://github.com/ggml-org/llama.cpp/discussions/4167), [Vulkan](https://github.com/ggml-org/llama.cpp/discussions/10879), [CUDA](https://github.com/ggml-org/llama.cpp/discussions/15013)). A 10,000-token prompt to a 70B model takes about half a minute on an M5 Max or a DGX Spark and a minute on the Ryzen, against a few seconds on data-center GPUs.

#### How close are the estimates?

Checked against 40 published single-request speeds (`arena_enrichment/measurements.py`): the median estimate is within 11% of the measurement, and 82% are within ±20%. The constants were chosen with these same measurements, so this shows the formula fits them rather than proving it on new hardware. Most are of small models: large models have few measurements, and large dense models on a Mac none from llama-bench (LM Studio users report speeds 25–50% below these estimates). A selection:

| Hardware | Model | Engine | Measured | Estimate |
|----------|-------|--------|----------|----------|
| RTX 4090 | [Llama-2-7B Q4_0](https://github.com/ggml-org/llama.cpp/discussions/15013) | llama.cpp | 189 tok/s | 170 tok/s |
| H100 SXM | [Llama-2-7B Q4_0](https://github.com/ggml-org/llama.cpp/discussions/15013) | llama.cpp | 281 tok/s | 400 tok/s |
| H200 | [Llama-3.3-70B FP8](https://developer.nvidia.com/blog/boost-llama-3-3-70b-inference-throughput-3x-with-nvidia-tensorrt-llm-speculative-decoding) | TensorRT-LLM | 51.1 tok/s | 49 tok/s |
| A100 80GB | [Qwen2.5-32B BF16](https://qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html) | vLLM | 22.1 tok/s | 24 tok/s |
| M5 Max | [Llama-2-7B Q8_0](https://github.com/ggml-org/llama.cpp/discussions/4167) | llama.cpp | 72.4 tok/s | 64 tok/s |
| M5 Max | [Llama-2-7B Q4_0](https://github.com/ggml-org/llama.cpp/discussions/4167) | llama.cpp | 120 tok/s | 110 tok/s |
| M5 Ultra | [Llama-2-7B F16](https://github.com/ggml-org/llama.cpp/discussions/4167) | llama.cpp | 73.6 tok/s | 68 tok/s |
| DGX Spark | [Llama-3.1-70B FP8](https://lmsys.org/blog/2025-10-13-nvidia-dgx-spark/) | SGLang | 2.7 tok/s | 2.9 tok/s |
| Ryzen AI Max+ 395 | [70B Q4_K_M](https://community.frame.work/t/amd-strix-halo-ryzen-ai-max-395-gpu-llm-performance-tests/72521) | llama.cpp (Vulkan) | 5 tok/s | 4.8 tok/s |
| RTX 5090 | [gpt-oss-20b MXFP4](https://github.com/ggml-org/llama.cpp/discussions/15396) | llama.cpp | 283 tok/s | 270 tok/s |
| M3 Ultra | [gpt-oss-20b MXFP4](https://github.com/ggml-org/llama.cpp/discussions/15396) | llama.cpp | 116 tok/s | 110 tok/s |
| DGX Spark | [gpt-oss-120b MXFP4](https://github.com/ggml-org/llama.cpp/blob/master/benches/dgx-spark/dgx-spark.md) | llama.cpp | 58.7 tok/s | 55 tok/s |
| Ryzen AI Max+ 395 | [gpt-oss-120b MXFP4](https://github.com/lhl/strix-halo-testing) | llama.cpp (Vulkan) | 50 tok/s | 52 tok/s |
| 8x H200 | [Llama-3.1-405B FP8](https://developer.nvidia.com/blog/low-latency-inference-chapter-1-up-to-1-9x-higher-llama-3-1-performance-with-medusa-on-nvidia-hgx-h200-with-nvlink-switch/) | TensorRT-LLM | 56 tok/s | 56 tok/s |
| 8x B200 | [DeepSeek-R1 FP4](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog01_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.html) | TensorRT-LLM, no MTP | 111 tok/s | 120 tok/s |
| 8x H200 | [DeepSeek-R1 FP8](https://verda.com/blog/deploy-deepseek-r1-on-8x-nvidia-h200) | SGLang (Jan 2025) | 47.7 tok/s | 110 tok/s |

The largest misses are llama.cpp on data-center GPUs (the A100, H100 and MI300X run 30–55% below the estimate with a 7B model) and an early-2025 SGLang run of DeepSeek-R1 on 8× H200 at less than half the estimate; reported DeepSeek speeds have varied widely between engine versions and settings.

## Usage

```bash
# Install dependencies
uv sync

# Full pipeline (downloads the latest leaderboard, updates README)
uv run python arena_enrichment/enrich_arena.py --update-readme

# Use a pre-downloaded CSV
uv run python arena_enrichment/enrich_arena.py --input data.csv --update-readme

# Skip network resolution (overrides + name parsing only)
uv run python arena_enrichment/enrich_arena.py --no-network --update-readme
```
