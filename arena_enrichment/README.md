# Arena.ai Leaderboard Enrichment

Enriches the [Arena.ai](https://arena.ai/leaderboard/text?license=open-source) open-source LLM leaderboard with parameter counts, memory estimates and generation-speed estimates, to answer which model is the best one a given GPU, Mac or server can run.

## Features

- **Load** the Arena.ai open-source text leaderboard from Arena's [Hugging Face dataset](https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset) (or a CSV export)
- **Resolve** total and active parameter counts via multiple strategies:
  - Small overrides dict (`known_models.py`) for models with misleading names or no public metadata
  - Artificial Analysis: its model list maps Arena names to model pages, whose sizes are cached in `data/aa_params.json`
  - Regex name parsing (e.g., `Qwen3-30B-A3B` → 30B total, 3B active, MoE)
  - Artificial Analysis pages found by guessing the page name, for models missing from the list
  - Results with more active than total parameters are rejected
- **Compute** the memory each model needs at BF16, FP8 and INT4, in GiB (weights, a 16K-token KV cache and the runtime), and its single-request generation speed from the hardware's memory bandwidth, on one device or an 8-GPU server. The formulas are in the root README; the speed formula is checked against published measurements in `measurements.py`
- **Flag** which models fit on each device: consumer GPUs (RTX 4090, RTX 5090), computers whose GPU shares system RAM (Macs, DGX Spark, Ryzen AI Max+ 395, counting the share the system gives the GPU), workstations (Radeon AI PRO R9700, RTX PRO 6000, DGX Station) and data-center GPUs (NVIDIA A100 to Rubin, AMD MI300X to MI355X, Intel Gaudi 3), and on 8-GPU servers of the GPUs sold that way (the list is `HARDWARE` in `vram_calculator.py`)
- **Output** enriched CSV, formatted XLSX, and auto-updated README.md tables

## Installation

```bash
uv sync
```

## Usage

```bash
# Full pipeline (downloads the latest leaderboard)
uv run python arena_enrichment/enrich_arena.py --update-readme

# Use a pre-downloaded CSV
uv run python arena_enrichment/enrich_arena.py --input arena_data.csv --update-readme

# Skip network resolution (overrides + name parsing only)
uv run python arena_enrichment/enrich_arena.py --input arena_data.csv --no-network --update-readme
```

## Automation

A GitHub Actions workflow runs daily at 06:00 UTC to fetch the latest leaderboard and update the root README.md with fresh tables. It commits the outputs and the Artificial Analysis page cache, so each run only reads pages for new models. See `.github/workflows/update-leaderboard.yml`.
