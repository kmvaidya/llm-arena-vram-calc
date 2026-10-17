# Arena.ai Leaderboard Enrichment

Enriches the [Arena.ai](https://arena.ai/leaderboard/text?license=open-source) open-source LLM leaderboard with parameter counts and VRAM estimates for GPU deployment feasibility.

## Features

- **Load** the Arena.ai open-source text leaderboard from Arena's [Hugging Face dataset](https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset) (or a CSV export)
- **Resolve** total and active parameter counts via multiple strategies:
  - Small overrides dict (`known_models.py`) for models with misleading names or no public metadata
  - Artificial Analysis: its model list maps Arena names to model pages, whose sizes are cached in `data/aa_params.json`
  - Regex name parsing (e.g., `Qwen3-30B-A3B` → 30B total, 3B active, MoE)
  - Artificial Analysis pages found by guessing the page name, for models missing from the list
  - Results with more active than total parameters are rejected
- **Compute** VRAM estimates at BF16, FP8, and INT4 precisions, and single-request decode speed on one GPU from its memory bandwidth (formulas in the root README)
- **Flag** which models fit on a single GPU, from consumer cards (RTX 4090, RTX 5090) through workstation (RTX PRO 6000) to data center (NVIDIA A100 to Rubin, AMD MI300X to MI355X, Intel Gaudi 3), or on an 8-GPU server of the GPUs sold that way (the list is `GPUS` in `vram_calculator.py`)
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
