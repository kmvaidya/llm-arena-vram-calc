# LLM Arena VRAM Calculator

Enriches the [Arena.ai](https://arena.ai/leaderboard/text?license=open-source) open-source LLM leaderboard with parameter counts and VRAM estimates and speed estimates for consumer GPUs, Macs and other unified-memory computers, workstations, data-center GPUs and 8-GPU servers.

Most LLM leaderboards rank models by quality but ignore deployment constraints. This tool answers: *"What's the best model I can actually run on my hardware?"* by cross-referencing Arena rankings with VRAM requirements across precisions.

> **Last updated:** 2026-10-23 10:50 UTC | **Arena data:** 2026-10-22 | **Models:** 227 | **Resolved:** 212 (93.4%)

## Best Model Per Hardware

Highest-ranked Arena model that fits each piece of hardware, with its Arena rank, the memory it needs and its estimated generation speed for one request (see [Memory formula](#memory-formula) and [Speed formula](#speed-formula)). *As released* is the model as its makers published it, at full quality; *8-bit* and *4-bit* are quantized to fit more, unless the release is already that small: a model published in 4-bit counts as released in every column. *Memory* is what a model can use: the GPU's memory, or for computers whose GPU shares system RAM, the share the system gives it. A model fits a GPU when it needs at most 95% of its memory.

| Tier | Hardware | Memory | Bandwidth | As released | 8-bit | 4-bit |
|------|----------|--------|-----------|------|------|------|
| Consumer GPU | RTX 5060 Ti 16 GB | 16 GiB | 0.448 TB/s | Gemma 3 4B IT (#118, 11 GiB, ~43 tok/s) | Molmo 2 8B (#104, 12 GiB, ~37 tok/s) | Gemma 3 12B IT (#93, 10 GiB, ~47 tok/s) |
|  | RTX 5080 | 16 GiB | 0.96 TB/s | Gemma 3 4B IT (#118, 11 GiB, ~88 tok/s) | Molmo 2 8B (#104, 12 GiB, ~77 tok/s) | Gemma 3 12B IT (#93, 10 GiB, ~96 tok/s) |
|  | RTX 3090 | 24 GiB | 0.936 TB/s | Molmo 2 8B (#104, 20 GiB, ~41 tok/s) | Gemma 3 12B IT (#93, 15 GiB, ~56 tok/s) | Gemma 4 31B (#15, 23 GiB, ~39 tok/s) |
|  | RTX 4090 | 24 GiB | 1.01 TB/s | Molmo 2 8B (#104, 20 GiB, ~44 tok/s) | Gemma 3 12B IT (#93, 15 GiB, ~60 tok/s) | Gemma 4 31B (#15, 23 GiB, ~42 tok/s) |
|  | RTX 5090 | 32 GiB | 1.79 TB/s | Gemma 3 12B IT (#93, 26 GiB, ~56 tok/s) | Gemma 4 26B A4B (#23, 30 GiB, ~190 tok/s) | Gemma 4 31B (#15, 23 GiB, ~72 tok/s) |
| Unified memory | Mac mini / MacBook Pro, M5 Pro 64 GB | 48 of 64 GiB | 0.307 TB/s | Gemma 3 12B IT (#93, 26 GiB, ~10 tok/s) | Gemma 4 31B (#15, 36 GiB, ~7.5 tok/s) | Gemma 4 31B (#15, 23 GiB, ~13 tok/s) |
|  | Ryzen AI Max+ 395, 128 GB | 96 of 128 GiB | 0.256 TB/s | Gemma 4 31B (#15, 64 GiB, ~3.3 tok/s) | Gemma 4 31B (#15, 36 GiB, ~6.3 tok/s) | Gemma 4 31B (#15, 23 GiB, ~11 tok/s) |
|  | MacBook Pro / Mac Studio, M5 Max 128 GB | 96 of 128 GiB | 0.614 TB/s | Gemma 4 31B (#15, 64 GiB, ~7.8 tok/s) | Gemma 4 31B (#15, 36 GiB, ~15 tok/s) | Gemma 4 31B (#15, 23 GiB, ~26 tok/s) |
|  | DGX Spark (GB10), 128 GB | 115 of 128 GiB | 0.273 TB/s | Gemma 4 31B (#15, 64 GiB, ~3.5 tok/s) | Gemma 4 31B (#15, 36 GiB, ~6.7 tok/s) | Gemma 4 31B (#15, 23 GiB, ~12 tok/s) |
|  | Mac Studio, M5 Ultra 256 GB | 192 of 256 GiB | 1.23 TB/s | Gemma 4 31B (#15, 64 GiB, ~15 tok/s) | Gemma 4 31B (#15, 36 GiB, ~29 tok/s) | GLM-5.3-Flash (#5, 184 GiB, ~57 tok/s) |
|  | Mac Studio, M5 Ultra 512 GB | 384 of 512 GiB | 1.23 TB/s | GLM-5.3-Flash (#5, 311 GiB, ~38 tok/s) | GLM-5.3-Flash (#5, 311 GiB, ~38 tok/s) | GLM-5.3-Flash (#5, 184 GiB, ~57 tok/s) |
| Workstation | Radeon AI PRO R9700 | 32 GiB | 0.64 TB/s | Gemma 3 12B IT (#93, 26 GiB, ~21 tok/s) | Gemma 4 26B A4B (#23, 30 GiB, ~91 tok/s) | Gemma 4 31B (#15, 23 GiB, ~27 tok/s) |
|  | RTX PRO 6000 Blackwell | 96 GiB | 1.79 TB/s | Gemma 4 31B (#15, 64 GiB, ~22 tok/s) | Gemma 4 31B (#15, 36 GiB, ~42 tok/s) | Gemma 4 31B (#15, 23 GiB, ~72 tok/s) |
|  | DGX Station (GB300) | 252 GiB | 7.1 TB/s | Gemma 4 31B (#15, 64 GiB, ~83 tok/s) | Gemma 4 31B (#15, 36 GiB, ~150 tok/s) | GLM-5.3-Flash (#5, 184 GiB, ~240 tok/s) |
| Data center GPU | L40S | 48 GiB | 0.864 TB/s | Gemma 3 12B IT (#93, 26 GiB, ~28 tok/s) | Gemma 4 31B (#15, 36 GiB, ~21 tok/s) | Gemma 4 31B (#15, 23 GiB, ~36 tok/s) |
|  | H100 SXM | 80 GiB | 3.35 TB/s | Gemma 4 31B (#15, 64 GiB, ~41 tok/s) | Gemma 4 31B (#15, 36 GiB, ~76 tok/s) | Gemma 4 31B (#15, 23 GiB, ~130 tok/s) |
|  | A100 80GB SXM | 80 GiB | 2.04 TB/s | Gemma 4 31B (#15, 64 GiB, ~25 tok/s) | Gemma 4 31B (#15, 36 GiB, ~48 tok/s) | Gemma 4 31B (#15, 23 GiB, ~81 tok/s) |
|  | H100 NVL | 94 GiB | 3.9 TB/s | Gemma 4 31B (#15, 64 GiB, ~48 tok/s) | Gemma 4 31B (#15, 36 GiB, ~87 tok/s) | Gemma 4 31B (#15, 23 GiB, ~140 tok/s) |
|  | Gaudi 3 | 128 GiB | 3.7 TB/s | Gemma 4 31B (#15, 64 GiB, ~45 tok/s) | Gemma 4 31B (#15, 36 GiB, ~83 tok/s) | Gemma 4 31B (#15, 23 GiB, ~140 tok/s) |
|  | H200 SXM / NVL | 141 GiB | 4.8 TB/s | Gemma 4 31B (#15, 64 GiB, ~58 tok/s) | Gemma 4 31B (#15, 36 GiB, ~110 tok/s) | Gemma 4 31B (#15, 23 GiB, ~170 tok/s) |
|  | B200 | 180 of 192 GiB | 8 TB/s | Gemma 4 31B (#15, 64 GiB, ~93 tok/s) | Gemma 4 31B (#15, 36 GiB, ~160 tok/s) | Gemma 4 31B (#15, 23 GiB, ~260 tok/s) |
|  | MI300X | 192 GiB | 5.3 TB/s | Gemma 4 31B (#15, 64 GiB, ~63 tok/s) | Gemma 4 31B (#15, 36 GiB, ~120 tok/s) | Hy3 (#14, 173 GiB, ~180 tok/s) |
|  | Rubin | 250 of 288 GiB | 16.2 TB/s | Gemma 4 31B (#15, 64 GiB, ~170 tok/s) | Gemma 4 31B (#15, 36 GiB, ~290 tok/s) | GLM-5.3-Flash (#5, 184 GiB, ~340 tok/s) |
|  | MI325X | 256 GiB | 6 TB/s | Gemma 4 31B (#15, 64 GiB, ~71 tok/s) | Gemma 4 31B (#15, 36 GiB, ~130 tok/s) | GLM-5.3-Flash (#5, 184 GiB, ~220 tok/s) |
|  | B300 | 268 of 288 GiB | 8 TB/s | Gemma 4 31B (#15, 64 GiB, ~93 tok/s) | Gemma 4 31B (#15, 36 GiB, ~160 tok/s) | GLM-5.3-Flash (#5, 184 GiB, ~250 tok/s) |
|  | MI350X / MI355X | 288 GiB | 8 TB/s | Gemma 4 31B (#15, 64 GiB, ~93 tok/s) | Gemma 4 31B (#15, 36 GiB, ~160 tok/s) | GLM-5.3-Flash (#5, 184 GiB, ~250 tok/s) |
| 8-GPU server | 8x L40S | 384 GiB | 8 × 0.864 TB/s | GLM-5.3-Flash (#5, 322 GiB) | GLM-5.3-Flash (#5, 322 GiB) | GLM-5.3-Flash (#5, 195 GiB) |
|  | 8x H100 SXM | 640 GiB | 8 × 3.35 TB/s | MiMo-V2.6-Pro (#3, 543 GiB, ~110 tok/s) | MiMo-V2.6-Pro (#3, 543 GiB, ~110 tok/s) | GLM-5.3 (Max) (#2, 437 GiB, ~110 tok/s) |
|  | 8x A100 80GB SXM | 640 GiB | 8 × 2.04 TB/s | MiMo-V2.6-Pro (#3, 543 GiB, ~99 tok/s) | MiMo-V2.6-Pro (#3, 543 GiB, ~99 tok/s) | GLM-5.3 (Max) (#2, 437 GiB, ~99 tok/s) |
|  | 8x H100 NVL | 752 GiB | 8 × 3.9 TB/s | MiMo-V2.6-Pro (#3, 543 GiB) | MiMo-V2.6-Pro (#3, 543 GiB) | GLM-5.3 (Max) (#2, 437 GiB) |
|  | 8x RTX PRO 6000 Blackwell Server Edition | 768 GiB | 8 × 1.6 TB/s | GLM-5.3 (Max) (#2, 720 GiB) | GLM-5.3 (Max) (#2, 720 GiB) | GLM-5.3 (Max) (#2, 437 GiB) |
|  | 8x Gaudi 3 | 1,024 GiB | 8 × 3.7 TB/s | GLM-5.3 (Max) (#2, 720 GiB, ~100 tok/s) | GLM-5.3 (Max) (#2, 720 GiB, ~100 tok/s) | GLM-5.3 (Max) (#2, 437 GiB, ~110 tok/s) |
|  | 8x H200 SXM | 1,128 GiB | 8 × 4.8 TB/s | GLM-5.3 (Max) (#2, 720 GiB, ~110 tok/s) | GLM-5.3 (Max) (#2, 720 GiB, ~110 tok/s) | GLM-5.3 (Max) (#2, 437 GiB, ~110 tok/s) |
|  | 8x B200 | 1,440 GiB | 8 × 8 TB/s | GLM-5.3 (Max) (#2, 720 GiB, ~110 tok/s) | GLM-5.3 (Max) (#2, 720 GiB, ~110 tok/s) | GLM-5.3 (Max) (#2, 437 GiB, ~120 tok/s) |
|  | 8x MI300X | 1,536 GiB | 8 × 5.3 TB/s | GLM-5.3 (Max) (#2, 720 GiB, ~110 tok/s) | GLM-5.3 (Max) (#2, 720 GiB, ~110 tok/s) | GLM-5.3 (Max) (#2, 437 GiB, ~110 tok/s) |
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
| 2 | GLM-5.3 (Max) | 1480 | 753 (40) | MoE | 8-bit | 709 | 709 | 426 | 8x RTX PRO 6000 Blackwell Server Edition | — |
| 3 | MiMo-V2.6-Pro | 1479 | 1000 (42) | MoE | 4-bit | 533 | 533 | 533 | 8x H100 SXM | ~110 tok/s |
| 4 | DeepSeek-V4.1-Flash-20260910 (Max) | 1476 | ? | ? | ? | ? | ? | ? | ? | ? |
| 5 | GLM-5.3-Flash | 1475 | 320 (18) | MoE | 8-bit | 311 | 311 | 184 | 8x L40S | — |
| 6 | GLM-5.2 (Max) | 1475 | 753 (40) | MoE | 16-bit | 1,409 | 749 | 426 | 8x H200 SXM | ~100 tok/s |
| 7 | MiMo-V2.5-Pro | 1466 | 1000 (42) | MoE | 8-bit | 968 | 968 | 564 | 8x H200 SXM | ~100 tok/s |
| 8 | GLM-5.1 | 1465 | 744 (40) | MoE | 16-bit | 1,410 | 740 | 421 | 8x H200 SXM | ~100 tok/s |
| 9 | DeepSeek-V4-Pro (High, 20260813) | 1464 | ? | ? | ? | ? | ? | ? | ? | ? |
| 10 | Kimi K2.6 | 1460 | 1000 (32) | MoE | 4-bit | 560 | 560 | 560 | 8x H100 SXM | ~110 tok/s |
| 11 | DeepSeek-V4-Pro | 1457 | 1600 (49) | MoE | 4-bit | 837 | 837 | 837 | 8x H200 SXM | ~110 tok/s |
| 12 | GLM-5 | 1457 | 744 (40) | MoE | 16-bit | 1,410 | 740 | 421 | 8x H200 SXM | ~100 tok/s |
| 13 | DeepSeek-V4-Pro Preview (High) | 1454 | 1600 (49) | MoE | 4-bit | 837 | 837 | 837 | 8x H200 SXM | ~110 tok/s |
| 14 | Hy3 | 1453 | 299 (21) | MoE | 16-bit | 562 | 301 | 173 | 8x L40S | — |
| 15 | Gemma 4 31B | 1452 | 30.7 | Dense | 16-bit | 63.8 | 35.8 | 22.7 | L40S | ~21 tok/s |
| 16 | Kimi K2.5 (Thinking) | 1449 | 1000 (32) | MoE | 4-bit | 560 | 560 | 560 | 8x H100 SXM | ~110 tok/s |
| 17 | MiMo-V2.6-Flash | 1448 | 309 (15) | MoE | 4-bit | 167 | 167 | 167 | B200 | ~280 tok/s |
| 18 | Qwen3.5-397B-A17B | 1441 | 397 (17) | MoE | 16-bit | 757 | 397 | 227 | 8x H100 SXM | ~110 tok/s |
| 19 | Inkling | 1441 | 975 (41) | MoE | 16-bit | 1,779 | 968 | 550 | 8x H200 SXM | ~100 tok/s |
| 20 | GLM-4.7 | 1441 | 357 (32) | MoE | 16-bit | 673 | 358 | 205 | 8x H100 SXM | ~100 tok/s |
| 21 | MiniMax-M3 | 1439 | 428 (23) | MoE | 16-bit | 801 | 428 | 245 | 8x H100 SXM | ~110 tok/s |
| 22 | DeepSeek-V4-Flash Preview (High) | 1438 | 284 (13) | MoE | 4-bit | 161 | 161 | 161 | B200 | ~300 tok/s |
| 23 | Gemma 4 26B A4B | 1437 | 25.2 (3.8) | MoE | 16-bit | 53.6 | 30.4 | 19.6 | RTX 5090 | ~190 tok/s |
| 24 | Qwen3.8-27B | 1437 | 27 | Dense | 16-bit | 57.2 | 32.2 | 20.6 | L40S | ~24 tok/s |
| 25 | Beam 1.0 501B A23B | 1435 | 501 (23) | MoE | ? | 939 | 500 | 285 | 8x H100 SXM | ~110 tok/s |
| 26 | DeepSeek-V4-Flash | 1435 | 284 (13) | MoE | 4-bit | 161 | 161 | 161 | B200 | ~300 tok/s |
| 27 | MiMo-V2.5 | 1433 | 310.8 (15) | MoE | 8-bit | 299 | 299 | 179 | 8x L40S | — |
| 28 | Kimi K2.5 (Instant) | 1430 | ? | ? | ? | ? | ? | ? | ? | ? |
| 29 | Kimi K2 Thinking Turbo | 1429 | 1000 (32) | MoE | 4-bit | 559 | 559 | 559 | 8x H100 SXM | ~110 tok/s |
| 30 | Mistral Medium 3.5 | 1426 | 128 | Dense | 8-bit | 130 | 130 | 77.0 | H200 SXM / NVL | ~28 tok/s |
| 31 | NVIDIA-Nemotron-3-Ultra-550B-A55B (NVFP4) | 1426 | 550 (55) | MoE | 16-bit | 1,050 | 548 | 313 | 8x H100 SXM | ~90 tok/s |
| 32 | DeepSeek-V3.2-Exp (Thinking) | 1425 | 685 (37) | MoE | 8-bit | 648 | 648 | 388 | 8x RTX PRO 6000 Blackwell Server Edition | — |
| 33 | DeepSeek-V3.2 | 1424 | 685 (37) | MoE | 8-bit | 648 | 648 | 388 | 8x RTX PRO 6000 Blackwell Server Edition | — |
| 34 | Muse Glimmer | 1424 | 30 | Dense | 16-bit | 61.0 | 35.1 | 22.3 | L40S | ~21 tok/s |
| 35 | GLM-4.6 | 1424 | 357 (32) | MoE | 16-bit | 670 | 358 | 205 | 8x H100 SXM | ~100 tok/s |
| 36 | DeepSeek-V3.2 (Thinking) | 1422 | 685 (37) | MoE | 8-bit | 648 | 648 | 388 | 8x RTX PRO 6000 Blackwell Server Edition | — |
| 37 | Qwen3-235B-A22B-Instruct (2507) | 1422 | 235 (22) | MoE | ? | 443 | 237 | 137 | B300 | ~160 tok/s |
| 38 | DeepSeek-V3.2-Exp | 1421 | 685 (37) | MoE | 8-bit | 648 | 648 | 388 | 8x RTX PRO 6000 Blackwell Server Edition | — |
| 39 | DeepSeek-R1 (0528) | 1421 | ? | ? | ? | ? | ? | ? | ? | ? |
| 40 | Kimi K2 Preview (0905) | 1418 | ? | ? | ? | ? | ? | ? | ? | ? |
| 41 | Kimi K2 Preview (0711) | 1417 | ? | ? | ? | ? | ? | ? | ? | ? |
| 42 | DeepSeek-V3.1 | 1417 | 685 (37) | MoE | 8-bit | 647 | 647 | 388 | 8x RTX PRO 6000 Blackwell Server Edition | — |
| 43 | Qwen3.5-122B-A10B | 1416 | 125 (10) | MoE | 16-bit | 238 | 129 | 75.3 | H200 SXM / NVL | ~190 tok/s |
| 44 | DeepSeek-V3.1-Terminus (Thinking) | 1416 | 685 (37) | MoE | 8-bit | 647 | 647 | 388 | 8x RTX PRO 6000 Blackwell Server Edition | — |
| 45 | DeepSeek-V3.1 (Thinking) | 1415 | 685 (37) | MoE | 8-bit | 647 | 647 | 388 | 8x RTX PRO 6000 Blackwell Server Edition | — |
| 46 | MiniMax-M2.7 | 1414 | 230 (10) | MoE | 8-bit | 220 | 220 | 134 | B300 | ~260 tok/s |
| 47 | DeepSeek-V3.1-Terminus | 1414 | 685 (37) | MoE | 8-bit | 647 | 647 | 388 | 8x RTX PRO 6000 Blackwell Server Edition | — |
| 48 | Mistral Large 3 | 1414 | 675 (41) | MoE | 8-bit | 640 | 640 | 383 | 8x RTX PRO 6000 Blackwell Server Edition | — |
| 49 | Qwen3-VL-235B-A22B-Instruct | 1413 | 235 (22) | MoE | 16-bit | 444 | 237 | 137 | B300 | ~160 tok/s |
| 50 | NVIDIA-Nemotron-3.5-Super-VL-120B-A12B (NVFP4) | 1412 | 120 (12) | MoE | ? | 229 | 124 | 72.6 | H200 SXM / NVL | ~170 tok/s |

<details><summary>Show remaining 177 models</summary>

| Rank | Model | Score | Params (B) | Arch | Release | GiB as released | GiB 8-bit | GiB 4-bit | Fits on | Speed |
|------|-------|-------|------------|------|---------|-----------------|-----------|-----------|---------|-------|
| 51 | GLM-4.5 | 1410 | 355 (32) | MoE | 16-bit | 673 | 356 | 204 | 8x H100 SXM | ~100 tok/s |
| 52 | Hy3 preview | 1409 | 295 (21) | MoE | 16-bit | 562 | 297 | 170 | 8x L40S | — |
| 53 | Qwen3.5-27B | 1408 | 27 | Dense | 16-bit | 57.2 | 32.2 | 20.6 | L40S | ~24 tok/s |
| 54 | Inkling-Small | 1404 | 266 (12) | MoE | 16-bit | 501 | 268 | 154 | 8x L40S | — |
| 55 | Qwen3-235B-A22B (Non-Thinking) | 1402 | 235 (22) | MoE | ? | 443 | 237 | 137 | B300 | ~160 tok/s |
| 56 | LongCat-Flash-Chat | 1400 | 560 (27) | MoE | 16-bit | 1,052 | 558 | 318 | 8x H100 SXM | ~100 tok/s |
| 57 | Qwen3-235B-A22B-Thinking (2507) | 1399 | 235 (22) | MoE | ? | 443 | 237 | 137 | B300 | ~160 tok/s |
| 58 | Qwen3-Next-80B-A3B-Instruct | 1398 | 80 (3) | MoE | 16-bit | 157 | 84.5 | 50.2 | RTX PRO 6000 Blackwell | ~220 tok/s |
| 59 | DeepSeek-R1 | 1398 | 685 (37) | MoE | 8-bit | 647 | 647 | 388 | 8x RTX PRO 6000 Blackwell Server Edition | — |
| 60 | DeepSeek-V3 (0324) | 1395 | 671 (37) | MoE | 8-bit | 647 | 647 | 380 | 8x RTX PRO 6000 Blackwell Server Edition | — |
| 61 | Qwen3-VL-235B-A22B-Thinking | 1394 | 235 (22) | MoE | 16-bit | 444 | 237 | 137 | B300 | ~160 tok/s |
| 62 | Qwen3.5-35B-A3B | 1393 | 35 (3) | MoE | ? | 70.7 | 40.1 | 25.1 | L40S | ~140 tok/s |
| 63 | Step-3.5-Flash | 1392 | 199.4 (11) | MoE | 16-bit | 377 | 202 | 117 | B300 | ~240 tok/s |
| 64 | MiMo-V2-Flash (Non-Thinking) | 1391 | 309 (15) | MoE | 8-bit | 297 | 297 | 178 | 8x L40S | — |
| 65 | MiniMax-M2.5 | 1390 | 230 (10) | MoE | 8-bit | 220 | 220 | 134 | B300 | ~260 tok/s |
| 66 | Qwen3-Coder-480B-A35B-Instruct | 1387 | 480 (35) | MoE | ? | 900 | 479 | 274 | 8x H100 SXM | ~100 tok/s |
| 67 | MiMo-V2-Flash (Thinking) | 1386 | 309 (15) | MoE | 8-bit | 297 | 297 | 178 | 8x L40S | — |
| 68 | MiniMax-M2.1 Preview | 1384 | 230 (10) | MoE | 8-bit | 220 | 220 | 134 | B300 | ~260 tok/s |
| 69 | Qwen3-30B-A3B-Instruct (2507) | 1382 | 30 (3) | MoE | ? | 61.4 | 35.1 | 22.3 | L40S | ~140 tok/s |
| 70 | Trinity Large Preview | 1378 | 398.6 (13) | MoE | 16-bit | 748 | 399 | 228 | 8x H100 SXM | ~110 tok/s |
| 71 | GLM-4.6V | 1378 | 108 (12) | MoE | 16-bit | 206 | 112 | 65.8 | H200 SXM / NVL | ~170 tok/s |
| 72 | Qwen3-235B-A22B | 1374 | 235 (22) | MoE | 16-bit | 443 | 237 | 137 | B300 | ~160 tok/s |
| 73 | GLM-4.5-Air | 1373 | 106 (12) | MoE | 16-bit | 211 | 110 | 64.7 | H200 SXM / NVL | ~170 tok/s |
| 74 | Qwen3-Next-80B-A3B-Thinking | 1369 | 80 (3) | MoE | ? | 155 | 84.5 | 50.2 | RTX PRO 6000 Blackwell | ~220 tok/s |
| 75 | Trinity Large Thinking | 1367 | 398.6 (13) | MoE | 16-bit | 748 | 399 | 228 | 8x H100 SXM | ~110 tok/s |
| 76 | Gemma 3 27B IT | 1365 | 27 | Dense | ? | 55.8 | 32.2 | 20.6 | L40S | ~24 tok/s |
| 77 | GLM-4.7-Flash | 1364 | 31.2 (3) | MoE | 16-bit | 63.7 | 36.3 | 22.9 | L40S | ~140 tok/s |
| 78 | MiniMax-M1 | 1363 | 456.1 (45.9) | MoE | 16-bit | 855 | 456 | 260 | 8x H100 SXM | ~94 tok/s |
| 79 | NVIDIA-Nemotron-3-Super-120B-A12B | 1360 | 120 (12) | MoE | ? | 229 | 124 | 72.6 | H200 SXM / NVL | ~170 tok/s |
| 80 | DeepSeek-V3 | 1358 | 671 (37) | MoE | 8-bit | 647 | 647 | 380 | 8x RTX PRO 6000 Blackwell Server Edition | — |
| 81 | Mistral Small 3.2 (2506) | 1356 | ? | ? | ? | ? | ? | ? | ? | ? |
| 82 | INTELLECT-3 | 1355 | 107 (12) | MoE | 16-bit | 205 | 111 | 65.3 | H200 SXM / NVL | ~170 tok/s |
| 83 | Command A (03-2025) | 1353 | 111 | Dense | 16-bit | 212 | 115 | 67.5 | H200 SXM / NVL | ~32 tok/s |
| 84 | NVIDIA-Nemotron-3.5-Lightning-30B-A3B (NVFP4) | 1353 | 30 (3) | MoE | ? | 61.4 | 35.1 | 22.3 | L40S | ~140 tok/s |
| 85 | GLM-4.5V | 1352 | 108 (12) | MoE | 16-bit | 206 | 112 | 65.8 | H200 SXM / NVL | ~170 tok/s |
| 86 | gpt-oss-120b | 1351 | 117 (5.1) | MoE | 4-bit | 66.3 | 66.3 | 66.3 | H100 SXM | ~310 tok/s |
| 87 | Step-3 | 1349 | 321 (38) | MoE | 16-bit | 603 | 322 | 185 | 8x L40S | — |
| 88 | Llama 3.1 Nemotron Ultra 253B v1 | 1347 | 253 | Dense | 16-bit | 477 | 255 | 147 | 8x L40S | — |
| 89 | Qwen3-32B | 1346 | 32 | Dense | ? | 65.1 | 37.1 | 23.4 | L40S | ~20 tok/s |
| 90 | MiniMax-M2 | 1343 | 230 (10) | MoE | 8-bit | 220 | 220 | 134 | B300 | ~260 tok/s |
| 91 | Ling-flash-2.0 | 1343 | 103 (6.1) | MoE | 16-bit | 197 | 107 | 63.1 | H200 SXM / NVL | ~250 tok/s |
| 92 | Llama 3.3 Nemotron Super 49B v1.5 | 1342 | 49 | Dense | ? | 96.8 | 53.9 | 32.9 | H100 SXM | ~49 tok/s |
| 93 | Gemma 3 12B IT | 1341 | 12 | Dense | ? | 25.9 | 15.3 | 10.2 | RTX 4090 | ~60 tok/s |
| 94 | Granite 4.2 30B | 1339 | 30 | Dense | 16-bit | 60.0 | 35.1 | 22.3 | L40S | ~21 tok/s |
| 95 | QwQ-32B | 1335 | 32 | Dense | ? | 65.1 | 37.1 | 23.4 | L40S | ~20 tok/s |
| 96 | Llama 3.1 405B Instruct (BF16) | 1335 | 405 | Dense | ? | 760 | 405 | 232 | 8x H100 SXM | ~42 tok/s |
| 97 | Llama 3.1 405B Instruct (FP8) | 1333 | 405 | Dense | ? | 760 | 405 | 232 | 8x H100 SXM | ~42 tok/s |
| 98 | Olmo 3.1 32B Instruct | 1329 | 32.2 | Dense | 16-bit | 65.5 | 37.3 | 23.5 | L40S | ~20 tok/s |
| 99 | Llama 3.3 Nemotron Super 49B v1 | 1327 | 49 | Dense | ? | 96.8 | 53.9 | 32.9 | H100 SXM | ~49 tok/s |
| 100 | Llama 4 Maverick 17B 128E Instruct | 1326 | 400 (17) | MoE | 16-bit | 754 | 400 | 229 | 8x H100 SXM | ~110 tok/s |
| 101 | Qwen3-30B-A3B | 1326 | 30 (3) | MoE | ? | 61.4 | 35.1 | 22.3 | L40S | ~140 tok/s |
| 102 | DeepSeek-V2.5 (1210) | 1323 | ? | ? | ? | ? | ? | ? | ? | ? |
| 103 | Ring-flash-2.0 | 1322 | 103 (6.1) | MoE | 16-bit | 197 | 107 | 63.1 | H200 SXM / NVL | ~250 tok/s |
| 104 | Molmo 2 8B | 1321 | 8.7 | Dense | ? | 19.7 | 12.1 | 8.4 | RTX 4090 | ~80 tok/s |
| 105 | Llama 4 Scout 17B 16E Instruct | 1321 | 109 (17) | MoE | 16-bit | 208 | 113 | 66.4 | H200 SXM / NVL | ~140 tok/s |
| 106 | Qwen-Max (0919) | 1318 | ? | ? | ? | ? | ? | ? | ? | ? |
| 107 | Llama 3.3 70B Instruct | 1318 | 70 | Dense | ? | 136 | 74.6 | 44.6 | H100 SXM | ~35 tok/s |
| 108 | Gemma 3n E4B IT | 1317 | 7.85 | Dense | ? | 18.1 | 11.2 | 7.9 | RTX 4090 | ~88 tok/s |
| 109 | gpt-oss-20b | 1317 | 21 (3.6) | MoE | 4-bit | 18.3 | 18.3 | 17.2 | RTX 4090 | ~190 tok/s |
| 110 | Mistral Large 2 (2407) | 1314 | ? | ? | ? | ? | ? | ? | ? | ? |
| 111 | Athene V2 Chat | 1314 | 72 | Dense | ? | 140 | 76.6 | 45.7 | RTX PRO 6000 Blackwell | ~18 tok/s |
| 112 | NVIDIA-Nemotron-3-Nano-30B-A3B (BF16) | 1313 | 31.6 (3.6) | MoE | 16-bit | 64.3 | 36.7 | 23.2 | L40S | ~120 tok/s |
| 113 | DeepSeek-V2.5 | 1307 | 236 (21) | MoE | 16-bit | 445 | 238 | 137 | B300 | ~170 tok/s |
| 114 | Athene 70B (0725) | 1306 | 70 | Dense | ? | 136 | 74.6 | 44.6 | H100 SXM | ~35 tok/s |
| 115 | Olmo 3 32B Think | 1306 | 32 | Dense | ? | 65.1 | 37.1 | 23.4 | L40S | ~20 tok/s |
| 116 | Mistral Large (2411) | 1305 | ? | ? | ? | ? | ? | ? | ? | ? |
| 117 | Granite 4.1 8B | 1304 | 8 | Dense | ? | 18.4 | 11.4 | 8.0 | RTX 4090 | ~87 tok/s |
| 118 | Gemma 3 4B IT | 1303 | 4 | Dense | ? | 11.0 | 7.4 | 5.7 | RTX 4090 | ~160 tok/s |
| 119 | Qwen2.5-72B-Instruct | 1302 | 72 | Dense | ? | 140 | 76.6 | 45.7 | RTX PRO 6000 Blackwell | ~18 tok/s |
| 120 | Mistral Small 3.1 24B Instruct (2503) | 1302 | 24 | Dense | ? | 50.2 | 29.2 | 18.9 | RTX 5090 | ~53 tok/s |
| 121 | Llama 3.1 Nemotron 70B Instruct | 1298 | 70 | Dense | ? | 136 | 74.6 | 44.6 | H100 SXM | ~35 tok/s |
| 122 | Llama 3.1 70B Instruct | 1293 | 70 | Dense | ? | 136 | 74.6 | 44.6 | H100 SXM | ~35 tok/s |
| 123 | Granite 4.2 3B | 1290 | 3 | Dense | ? | 9.1 | 6.5 | 5.2 | RTX 4090 | ~200 tok/s |
| 124 | Granite 4.2 8B | 1289 | 8 | Dense | 16-bit | 19.9 | 11.4 | 8.0 | RTX 4090 | ~87 tok/s |
| 125 | Gemma 2 27B IT | 1289 | 27 | Dense | ? | 55.8 | 32.2 | 20.6 | L40S | ~24 tok/s |
| 126 | Jamba 1.5 Large | 1289 | 398 (94) | MoE | 16-bit | 748 | 398 | 228 | 8x H100 SXM | ~75 tok/s |
| 127 | Granite-4.0-H-Small | 1287 | 32 (9) | MoE | 16-bit | 65.5 | 37.1 | 23.4 | L40S | ~56 tok/s |
| 128 | Llama 3.1 Nemotron 51B Instruct | 1286 | 51 | Dense | ? | 100 | 55.8 | 34.0 | H100 SXM | ~47 tok/s |
| 129 | Olmo 3.1 32B Think | 1286 | 32.2 | Dense | 16-bit | 65.5 | 37.3 | 23.5 | L40S | ~20 tok/s |
| 130 | Llama 3.1 Tulu 3 70B | 1286 | 70 | Dense | ? | 136 | 74.6 | 44.6 | H100 SXM | ~35 tok/s |
| 131 | Gemma 2 9B IT SimPO | 1280 | 9 | Dense | ? | 20.3 | 12.4 | 8.5 | RTX 4090 | ~78 tok/s |
| 132 | Nemotron-4 340B Instruct | 1277 | 340 | Dense | ? | 639 | 341 | 195 | 8x L40S | — |
| 133 | Llama 3 70B Instruct | 1276 | 70 | Dense | ? | 136 | 74.6 | 44.6 | H100 SXM | ~35 tok/s |
| 134 | Command R+ (08-2024) | 1276 | ? | ? | ? | ? | ? | ? | ? | ? |
| 135 | Mistral Small 24B Instruct (2501) | 1274 | 24 | Dense | ? | 50.2 | 29.2 | 18.9 | RTX 5090 | ~53 tok/s |
| 136 | Qwen2.5-Coder-32B-Instruct | 1270 | 32 | Dense | ? | 65.1 | 37.1 | 23.4 | L40S | ~20 tok/s |
| 137 | Aya Expanse 32B | 1267 | 32 | Dense | ? | 65.1 | 37.1 | 23.4 | L40S | ~20 tok/s |
| 138 | Gemma 2 9B IT | 1267 | 9 | Dense | ? | 20.3 | 12.4 | 8.5 | RTX 4090 | ~78 tok/s |
| 139 | DeepSeek-Coder-V2 | 1265 | 236 (21) | MoE | 16-bit | 445 | 238 | 137 | B300 | ~170 tok/s |
| 140 | Qwen2-72B-Instruct | 1261 | 72 | Dense | ? | 140 | 76.6 | 45.7 | RTX PRO 6000 Blackwell | ~18 tok/s |
| 141 | Command R+ | 1261 | 104 | Dense | ? | 199 | 108 | 63.6 | H200 SXM / NVL | ~34 tok/s |
| 142 | Phi-4 | 1256 | 14 | Dense | 16-bit | 30.8 | 17.3 | 11.3 | RTX 4090 | ~52 tok/s |
| 143 | OLMo 2 32B Instruct (0325) | 1251 | 32 | Dense | ? | 65.1 | 37.1 | 23.4 | L40S | ~20 tok/s |
| 144 | Command R (08-2024) | 1250 | ? | ? | ? | ? | ? | ? | ? | ? |
| 145 | Jamba 1.5 Mini | 1239 | 52 (12) | MoE | 16-bit | 102 | 56.8 | 34.6 | H100 SXM | ~130 tok/s |
| 146 | Ministral 8B (2410) | 1237 | 8 | Dense | ? | 18.4 | 11.4 | 8.0 | RTX 4090 | ~87 tok/s |
| 147 | Qwen1.5-110B-Chat | 1234 | 110 | Dense | ? | 210 | 114 | 67.0 | H200 SXM / NVL | ~32 tok/s |
| 148 | Qwen1.5-72B-Chat | 1233 | 72 | Dense | ? | 140 | 76.6 | 45.7 | RTX PRO 6000 Blackwell | ~18 tok/s |
| 149 | Mixtral 8x22B Instruct v0.1 | 1229 | 140.8 (39.6) | MoE | ? | 268 | 144 | 84.2 | B200 | ~100 tok/s |
| 150 | Command R | 1226 | 35 | Dense | ? | 70.7 | 40.1 | 25.1 | L40S | ~18 tok/s |
| 151 | Llama 3 8B Instruct | 1223 | 8 | Dense | ? | 18.4 | 11.4 | 8.0 | RTX 4090 | ~87 tok/s |
| 152 | Aya Expanse 8B | 1223 | 8 | Dense | ? | 18.4 | 11.4 | 8.0 | RTX 4090 | ~87 tok/s |
| 153 | Llama 3.1 Tulu 3 8B | 1220 | 8 | Dense | ? | 18.4 | 11.4 | 8.0 | RTX 4090 | ~87 tok/s |
| 154 | Zephyr ORPO 141B A35B v0.1 | 1213 | 141 (35) | MoE | ? | 268 | 145 | 84.3 | B200 | ~120 tok/s |
| 155 | Yi 1.5 34B Chat | 1212 | 34 | Dense | ? | 68.8 | 39.1 | 24.5 | L40S | ~19 tok/s |
| 156 | Llama 3.1 8B Instruct | 1211 | 8 | Dense | ? | 18.4 | 11.4 | 8.0 | RTX 4090 | ~87 tok/s |
| 157 | Granite 3.1 8B Instruct | 1208 | 8 | Dense | ? | 18.4 | 11.4 | 8.0 | RTX 4090 | ~87 tok/s |
| 158 | Qwen1.5-32B-Chat | 1204 | 32 | Dense | ? | 65.1 | 37.1 | 23.4 | L40S | ~20 tok/s |
| 159 | Gemma 2 2B IT | 1200 | 2 | Dense | ? | 7.2 | 5.5 | 4.6 | RTX 4090 | ~280 tok/s |
| 160 | Phi-3 Medium 4K Instruct | 1197 | 14 | Dense | ? | 29.6 | 17.3 | 11.3 | RTX 4090 | ~52 tok/s |
| 161 | Mixtral 8x7B Instruct v0.1 | 1197 | 44.8 (12.6) | MoE | ? | 88.9 | 49.7 | 30.5 | H100 SXM | ~130 tok/s |
| 162 | DBRX Instruct Preview | 1195 | 132 (36) | MoE | ? | 251 | 136 | 79.3 | B200 | ~110 tok/s |
| 163 | Qwen1.5-14B-Chat | 1191 | 14 | Dense | ? | 29.6 | 17.3 | 11.3 | RTX 4090 | ~52 tok/s |
| 164 | InternLM2.5 20B Chat | 1190 | 20 | Dense | ? | 42.8 | 25.2 | 16.7 | RTX 5090 | ~63 tok/s |
| 165 | DeepSeek LLM 67B Chat | 1185 | 67 | Dense | ? | 130 | 71.6 | 42.9 | H100 SXM | ~36 tok/s |
| 166 | WizardLM 70B | 1184 | 70 | Dense | ? | 136 | 74.6 | 44.6 | H100 SXM | ~35 tok/s |
| 167 | Yi 34B Chat | 1183 | 34 | Dense | ? | 68.8 | 39.1 | 24.5 | L40S | ~19 tok/s |
| 168 | Granite 3.0 8B Instruct | 1183 | 8 | Dense | ? | 18.4 | 11.4 | 8.0 | RTX 4090 | ~87 tok/s |
| 169 | OpenChat 3.5 | 1183 | 7.24 | Dense | ? | 17.0 | 10.6 | 7.5 | RTX 4090 | ~95 tok/s |
| 170 | OpenChat 3.5 (0106) | 1182 | ? | ? | ? | ? | ? | ? | ? | ? |
| 171 | Gemma 1.1 7B IT | 1182 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 172 | Snowflake Arctic Instruct | 1180 | 478.6 (17) | MoE | ? | 897 | 478 | 273 | 8x H100 SXM | ~110 tok/s |
| 173 | Granite 3.1 2B Instruct | 1178 | 2 | Dense | ? | 7.2 | 5.5 | 4.6 | RTX 4090 | ~280 tok/s |
| 174 | Tulu 2 DPO 70B | 1177 | 70 | Dense | ? | 136 | 74.6 | 44.6 | H100 SXM | ~35 tok/s |
| 175 | OpenHermes 2.5 Mistral 7B | 1176 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 176 | Vicuna 33B | 1172 | 33 | Dense | ? | 67.0 | 38.1 | 23.9 | L40S | ~19 tok/s |
| 177 | Phi-3 Small 8K Instruct | 1171 | 7.4 | Dense | ? | 17.3 | 10.8 | 7.6 | RTX 4090 | ~93 tok/s |
| 178 | Starling LM 7B Beta | 1171 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 179 | Llama 2 70B Chat | 1170 | 70 | Dense | ? | 136 | 74.6 | 44.6 | H100 SXM | ~35 tok/s |
| 180 | Starling LM 7B Alpha | 1167 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 181 | Llama 3.2 3B Instruct | 1166 | 3 | Dense | ? | 9.1 | 6.5 | 5.2 | RTX 4090 | ~200 tok/s |
| 182 | Nous Hermes 2 Mixtral 8x7B DPO | 1164 | 44.8 (12.6) | MoE | ? | 88.9 | 49.7 | 30.5 | H100 SXM | ~130 tok/s |
| 183 | Granite 3.0 2B Instruct | 1156 | 2 | Dense | ? | 7.2 | 5.5 | 4.6 | RTX 4090 | ~280 tok/s |
| 184 | Llama 2 70B SteerLM Chat | 1154 | 70 | Dense | ? | 136 | 74.6 | 44.6 | H100 SXM | ~35 tok/s |
| 185 | QwQ-32B-Preview | 1154 | 32 | Dense | ? | 65.1 | 37.1 | 23.4 | L40S | ~20 tok/s |
| 186 | SOLAR 10.7B Instruct v1.0 | 1152 | 10.7 | Dense | ? | 23.4 | 14.1 | 9.5 | RTX 4090 | ~66 tok/s |
| 187 | Dolphin 2.2.1 Mistral 7B | 1152 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 188 | MPT 30B Chat | 1150 | 30 | Dense | ? | 61.4 | 35.1 | 22.3 | L40S | ~21 tok/s |
| 189 | WizardLM 13B | 1149 | 13 | Dense | ? | 27.7 | 16.3 | 10.8 | RTX 4090 | ~55 tok/s |
| 190 | Mistral 7B Instruct v0.2 | 1149 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 191 | Falcon 180B Chat | 1148 | 180 | Dense | ? | 341 | 183 | 106 | B300 | ~32 tok/s |
| 192 | Qwen1.5-7B-Chat | 1143 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 193 | Phi-3 Mini 4K Instruct (June 2024) | 1143 | ? | ? | ? | ? | ? | ? | ? | ? |
| 194 | Vicuna 13B | 1141 | 13 | Dense | ? | 27.7 | 16.3 | 10.8 | RTX 4090 | ~55 tok/s |
| 195 | Llama 2 13B Chat | 1141 | 13 | Dense | ? | 27.7 | 16.3 | 10.8 | RTX 4090 | ~55 tok/s |
| 196 | Qwen-14B-Chat | 1139 | 14 | Dense | ? | 29.6 | 17.3 | 11.3 | RTX 4090 | ~52 tok/s |
| 197 | Gemma 7B IT | 1137 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 198 | Code Llama 34B Instruct | 1136 | 34 | Dense | ? | 68.8 | 39.1 | 24.5 | L40S | ~19 tok/s |
| 199 | Zephyr 7B Beta | 1130 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 200 | Phi-3 Mini 128K Instruct | 1129 | 3.8 | Dense | ? | 10.6 | 7.3 | 5.6 | RTX 4090 | ~170 tok/s |
| 201 | Phi-3 Mini 4K Instruct | 1128 | 3.8 | Dense | ? | 10.6 | 7.3 | 5.6 | RTX 4090 | ~170 tok/s |
| 202 | Guanaco 33B | 1127 | 33 | Dense | ? | 67.0 | 38.1 | 23.9 | L40S | ~19 tok/s |
| 203 | Zephyr 7B Alpha | 1127 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 204 | StripedHyena Nous 7B | 1121 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 205 | Code Llama 70B Instruct | 1119 | 70 | Dense | ? | 136 | 74.6 | 44.6 | H100 SXM | ~35 tok/s |
| 206 | Gemma 1.1 2B IT | 1116 | 2 | Dense | ? | 7.2 | 5.5 | 4.6 | RTX 4090 | ~280 tok/s |
| 207 | Vicuna 7B | 1115 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 208 | SmolLM2 1.7B Instruct | 1114 | 1.7 | Dense | ? | 6.7 | 5.2 | 4.4 | RTX 4090 | ~310 tok/s |
| 209 | Llama 3.2 1B Instruct | 1111 | 1 | Dense | ? | 5.4 | 4.5 | 4.1 | RTX 4090 | ~430 tok/s |
| 210 | Mistral 7B Instruct | 1110 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 211 | Llama 2 7B Chat | 1107 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 212 | Gemma 2B IT | 1093 | 2 | Dense | ? | 7.2 | 5.5 | 4.6 | RTX 4090 | ~280 tok/s |
| 213 | Qwen1.5-4B-Chat | 1091 | 4 | Dense | ? | 11.0 | 7.4 | 5.7 | RTX 4090 | ~160 tok/s |
| 214 | OLMo 7B Instruct | 1073 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 215 | Koala 13B | 1070 | 13 | Dense | ? | 27.7 | 16.3 | 10.8 | RTX 4090 | ~55 tok/s |
| 216 | Alpaca 13B | 1070 | 13 | Dense | ? | 27.7 | 16.3 | 10.8 | RTX 4090 | ~55 tok/s |
| 217 | GPT4All 13B Snoozy | 1067 | 13 | Dense | ? | 27.7 | 16.3 | 10.8 | RTX 4090 | ~55 tok/s |
| 218 | MPT 7B Chat | 1063 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |
| 219 | ChatGLM3 6B | 1056 | 6 | Dense | ? | 14.7 | 9.4 | 6.9 | RTX 4090 | ~110 tok/s |
| 220 | RWKV-4 Raven 14B | 1042 | 14 | Dense | ? | 29.6 | 17.3 | 11.3 | RTX 4090 | ~52 tok/s |
| 221 | ChatGLM2 6B | 1024 | 6 | Dense | ? | 14.7 | 9.4 | 6.9 | RTX 4090 | ~110 tok/s |
| 222 | OpenAssistant Pythia 12B | 1023 | 12 | Dense | ? | 25.9 | 15.3 | 10.2 | RTX 4090 | ~60 tok/s |
| 223 | ChatGLM 6B | 995 | 6 | Dense | ? | 14.7 | 9.4 | 6.9 | RTX 4090 | ~110 tok/s |
| 224 | FastChat-T5 3B | 992 | 3 | Dense | ? | 9.1 | 6.5 | 5.2 | RTX 4090 | ~200 tok/s |
| 225 | Dolly 2.0 12B | 982 | 12 | Dense | ? | 25.9 | 15.3 | 10.2 | RTX 4090 | ~60 tok/s |
| 226 | LLaMA 13B | 975 | 13 | Dense | ? | 27.7 | 16.3 | 10.8 | RTX 4090 | ~55 tok/s |
| 227 | StableLM Tuned Alpha 7B | 953 | 7 | Dense | ? | 16.5 | 10.4 | 7.4 | RTX 4090 | ~98 tok/s |

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
