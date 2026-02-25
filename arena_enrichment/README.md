# Arena.ai Leaderboard Enrichment

Enriches the [Arena.ai](https://arena.ai/leaderboard/text?license=open-source) open-source LLM leaderboard with parameter counts and VRAM estimates for GPU deployment feasibility.

## Features

- **Scrape** the Arena.ai open-source text leaderboard (API, HF Space, Playwright, or CSV input)
- **Resolve** total and active parameter counts via multiple strategies:
  - Small overrides dict for models with misleading names or no public metadata
  - Regex name parsing (e.g., `Qwen3-30B-A3B` → 30B total, 3B active, MoE)
  - HuggingFace API (safetensors metadata + config.json)
  - Artificial Analysis scraping
- **Compute** VRAM estimates at BF16, FP8, and INT4 precisions
- **Flag** which models fit on single-GPU configurations (H100, RTX PRO 6000, H200, B200, B300)
- **Output** enriched CSV, formatted XLSX, and auto-updated README.md tables

## Installation

```bash
pip install -r requirements.txt

# Only if using browser-based scraping:
playwright install chromium
```

## Usage

```bash
# Full pipeline (scrapes arena.ai live)
python enrich_arena.py --update-readme

# Use a pre-downloaded CSV
python enrich_arena.py --input arena_data.csv --update-readme

# Skip network resolution (overrides + name parsing only)
python enrich_arena.py --input arena_data.csv --no-network --update-readme
```

## Automation

A GitHub Actions workflow runs daily at 06:00 UTC to scrape the latest leaderboard and update the root README.md with fresh tables. See `.github/workflows/update-leaderboard.yml`.
