"""
Load the Arena.ai open-source text leaderboard.

Since 2026-10-14 arena.ai answers scripts with a Cloudflare bot challenge, so
the page itself can no longer be read. Arena publishes the same leaderboard
to the Hugging Face dataset lmarena-ai/leaderboard-dataset: the
text_style_control config is the ranking arena.ai shows by default, and its
"latest" split holds the most recent publication. The data includes all
models (proprietary + open-source) and every category; we keep the overall
ranking and filter client-side to match the ?license=open-source URL filter.

Approaches in order:
1. Load from local CSV if provided
2. Download the latest publication from the Hugging Face dataset
3. Raise error with instructions
"""

from __future__ import annotations

import io
import logging
import re
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ARENA_URL = "https://arena.ai/leaderboard/text?license=open-source"

HF_DATASET_URL = (
    "https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset/resolve/main/"
    "text_style_control/latest-00000-of-00001.parquet"
)

# Dataset columns this module reads
HF_COLUMNS = [
    "model_name", "organization", "license", "rating", "rating_lower",
    "rating_upper", "vote_count", "category",
]

EXPECTED_COLUMNS = [
    "rank", "model_name", "arena_score", "ci_lower", "ci_upper",
    "votes", "organization", "license",
]

# Licenses that indicate a proprietary / closed-source model
PROPRIETARY_LICENSES = {"Proprietary"}


def scrape_arena_leaderboard(input_csv: str | None = None) -> pd.DataFrame:
    """
    Download or load the Arena.ai open-source leaderboard.

    Args:
        input_csv: Optional path to a pre-downloaded CSV file.

    Returns:
        pd.DataFrame with columns: rank, model_name, arena_score,
        ci_lower, ci_upper, votes, organization, license
    """
    if input_csv:
        return _load_from_csv(input_csv)

    df = _fetch_hf_leaderboard()
    if df is not None:
        return df

    raise RuntimeError(
        "Could not download the Arena.ai leaderboard from Hugging Face.\n"
        "Please download the leaderboard as CSV and use: --input arena_data.csv\n"
        f"\nExport the table from {ARENA_URL} as CSV.\n"
    )


def _fetch_hf_leaderboard() -> pd.DataFrame | None:
    """
    Download the latest leaderboard publication from Hugging Face and reduce
    it to the open-source models of the overall text ranking.

    The dataset ranks all models together, so we re-rank the open-source ones
    sequentially, best rating first, as arena.ai does for its license filter.
    """
    try:
        logger.info(f"Fetching {HF_DATASET_URL}")
        resp = requests.get(HF_DATASET_URL, timeout=60)
        if resp.status_code != 200:
            logger.warning(f"Hugging Face returned status {resp.status_code}")
            return None
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch the leaderboard dataset: {e}")
        return None

    try:
        data = pd.read_parquet(io.BytesIO(resp.content))
    except (ValueError, OSError) as e:
        logger.warning(f"Failed to read the leaderboard dataset: {e}")
        return None

    missing = [c for c in HF_COLUMNS if c not in data.columns]
    if missing:
        logger.warning(f"Leaderboard dataset is missing columns: {missing}")
        return None

    overall = data[data["category"] == "overall"]
    if "leaderboard_publish_date" in overall.columns:
        logger.info(f"Leaderboard published {overall['leaderboard_publish_date'].max()}")
    logger.info(f"Loaded {len(overall)} models from the overall ranking")

    # A model without a license is treated as proprietary
    licenses = overall["license"].fillna("Proprietary")
    open_source = overall[~licenses.isin(PROPRIETARY_LICENSES)]
    logger.info(f"Filtered to {len(open_source)} open-source models")

    if open_source.empty:
        return None

    df = open_source.sort_values("rating", ascending=False, kind="stable").rename(columns={
        "rating": "arena_score",
        "rating_lower": "ci_lower",
        "rating_upper": "ci_upper",
        "vote_count": "votes",
    })
    df["rank"] = range(1, len(df) + 1)
    df["votes"] = df["votes"].round().astype("Int64")
    df = df[EXPECTED_COLUMNS].reset_index(drop=True)
    logger.info(f"Built DataFrame with {len(df)} open-source models")
    return df


# ---------------------------------------------------------------------------
# CSV loading (unchanged)
# ---------------------------------------------------------------------------

def _load_from_csv(path: str) -> pd.DataFrame:
    """Load leaderboard from a local CSV file."""
    logger.info(f"Loading leaderboard from CSV: {path}")
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Try to map common alternative column names
    rename_map: dict[str, str] = {}
    for col in df.columns:
        if "rank" in col and "rank" not in rename_map.values():
            rename_map[col] = "rank"
        elif (("model" in col and "name" in col) or col == "model") and "model_name" not in rename_map.values():
            rename_map[col] = "model_name"
        elif "score" in col and "arena_score" not in rename_map.values():
            rename_map[col] = "arena_score"
        elif "vote" in col and "votes" not in rename_map.values():
            rename_map[col] = "votes"
        elif "org" in col and "organization" not in rename_map.values():
            rename_map[col] = "organization"
        elif "license" in col and "license" not in rename_map.values():
            rename_map[col] = "license"

    if rename_map:
        df = df.rename(columns=rename_map)

    # Handle CI column — might be a single "95%_ci" or "ci" column like "+5/-3"
    if "ci_lower" not in df.columns or "ci_upper" not in df.columns:
        ci_col = None
        for col in df.columns:
            if "ci" in col.lower() or "confidence" in col.lower():
                ci_col = col
                break
        if ci_col:
            df["ci_lower"], df["ci_upper"] = zip(
                *df[ci_col].apply(_parse_ci_string), strict=False
            )
            if ci_col not in ("ci_lower", "ci_upper"):
                df = df.drop(columns=[ci_col])
        else:
            df["ci_lower"] = None
            df["ci_upper"] = None

    # Ensure model_name exists
    if "model_name" not in df.columns:
        # Try first string-like column
        for col in df.columns:
            if df[col].dtype == object and col not in ("license", "organization"):
                df = df.rename(columns={col: "model_name"})
                break

    # Ensure rank column
    if "rank" not in df.columns:
        df["rank"] = range(1, len(df) + 1)

    # Fill missing columns
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    return df[EXPECTED_COLUMNS].copy()


def _parse_ci_string(ci_str: Any) -> tuple[float | None, float | None]:
    """Parse CI string like '+5/-3' or '(1450, 1460)' into (lower, upper) offsets."""
    if pd.isna(ci_str):
        return (None, None)
    ci_str = str(ci_str).strip()

    # Pattern: +X/-Y
    m = re.match(r'[+]?(\d+(?:\.\d+)?)\s*/\s*[-]?(\d+(?:\.\d+)?)', ci_str)
    if m:
        return (-float(m.group(2)), float(m.group(1)))

    # Pattern: (lower, upper)
    m = re.match(r'\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?', ci_str)
    if m:
        return (float(m.group(1)), float(m.group(2)))

    # Pattern: lower-upper or lower–upper
    m = re.match(r'(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)', ci_str)
    if m:
        return (float(m.group(1)), float(m.group(2)))

    return (None, None)
