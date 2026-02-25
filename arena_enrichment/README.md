# Arena.ai Leaderboard Enrichment

Enriches the [Arena.ai](https://arena.ai/leaderboard/text?license=open-source) open-source LLM leaderboard with parameter counts and VRAM estimates for GPU deployment feasibility.

## Features

- **Scrape** the Arena.ai open-source text leaderboard (API, Playwright, or CSV input)
- **Resolve** total and active parameter counts via multiple strategies:
  - Hardcoded known values for major models
  - Regex name parsing (e.g., `Qwen3-30B-A3B` → 30B total, 3B active, MoE)
  - Artificial Analysis scraping
  - HuggingFace API lookup
- **Compute** VRAM estimates at BF16, FP8, and INT4 precisions
- **Flag** which models fit on single-GPU configurations (RTX PRO 6000, H100, H200, B200, B300)
- **Output** enriched CSV and formatted XLSX with conditional formatting

## Installation

```bash
pip install -r requirements.txt

# Only if using browser-based scraping:
playwright install chromium
```

## Usage

```bash
# Full pipeline (scrapes arena.ai live)
python enrich_arena.py

# Use a pre-downloaded CSV
python enrich_arena.py --input arena_data.csv

# Skip network resolution (hardcoded + name parsing only)
python enrich_arena.py --input arena_data.csv --no-network

# Custom output directory
python enrich_arena.py -o /path/to/output
```

## Output

Files are saved to `output/`:

- `arena_leaderboard_enriched.csv` — Full enriched table
- `arena_leaderboard_enriched.xlsx` — Formatted workbook with 3 sheets:
  - **Full Table** — All data with green/red GPU fit indicators
  - **GPU Summary** — Best model per GPU at each precision
  - **Resolution Log** — How each model's parameters were resolved

## GPU Configurations

| GPU | VRAM | Architecture | Native FP8 |
|-----|------|-------------|------------|
| RTX PRO 6000 | 96 GB | Ada Lovelace | No (software emulation) |
| H100 SXM | 80 GB | Hopper | Yes |
| H200 SXM | 141 GB | Hopper | Yes |
| B200 SXM | 180 GB | Blackwell | Yes |
| B300 SXM | 288 GB | Blackwell Ultra | Yes |

## VRAM Calculation

- **Weight VRAM** = total_params × bytes_per_param (BF16: 2B, FP8: 1B, INT4: 0.5B)
- **Serving VRAM** = weight VRAM × 1.25 (25% overhead for KV cache, activations, framework)
- For MoE models, VRAM is based on **total** parameters (all experts must be loaded)
