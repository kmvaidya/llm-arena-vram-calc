"""
Scrape the Arena.ai open-source text leaderboard.

The leaderboard at arena.ai is a Next.js app that embeds all model data as
double-escaped JSON inside self.__next_f.push() streaming calls. The data
includes all models (proprietary + open-source); we filter client-side to
match the ?license=open-source URL filter.

Approaches in order:
1. Load from local CSV if provided
2. Extract embedded Next.js data from page HTML (no JS rendering needed)
3. Raise error with instructions
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ARENA_URL = "https://arena.ai/leaderboard/text?license=open-source"

EXPECTED_COLUMNS = [
    "rank", "model_name", "arena_score", "ci_lower", "ci_upper",
    "votes", "organization", "license",
]

# Licenses that indicate a proprietary / closed-source model
PROPRIETARY_LICENSES = {"Proprietary"}


def scrape_arena_leaderboard(input_csv: str | None = None) -> pd.DataFrame:
    """
    Scrape or load the Arena.ai open-source leaderboard.

    Args:
        input_csv: Optional path to a pre-downloaded CSV file.

    Returns:
        pd.DataFrame with columns: rank, model_name, arena_score,
        ci_lower, ci_upper, votes, organization, license
    """
    if input_csv:
        return _load_from_csv(input_csv)

    df = _scrape_arena_html()
    if df is not None:
        return df

    raise RuntimeError(
        "Could not scrape arena.ai leaderboard.\n"
        "Please download the leaderboard as CSV and use: --input arena_data.csv\n"
        "\nExport the table from https://arena.ai/leaderboard/text"
        "?license=open-source as CSV.\n"
    )


def _scrape_arena_html() -> pd.DataFrame | None:
    """
    Fetch the arena.ai leaderboard page and extract model data from the
    embedded Next.js payload.

    The page embeds all leaderboard entries as double-escaped JSON inside
    self.__next_f.push() calls. The entries are in a single array containing
    all models (proprietary + open-source). We parse this array and filter
    to open-source licenses only.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
    }

    try:
        logger.info(f"Fetching {ARENA_URL}")
        resp = requests.get(ARENA_URL, headers=headers, timeout=20)
        if resp.status_code != 200:
            logger.warning(f"arena.ai returned status {resp.status_code}")
            return None
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch arena.ai: {e}")
        return None

    html = resp.text

    # The leaderboard data is embedded as double-escaped JSON:
    #   \"entries\":[{\"rank\":1,\"modelDisplayName\":\"...\", ...}]
    # Find the entries array and unescape it.
    entries = _extract_entries_from_nextjs(html)
    if not entries:
        logger.warning("Could not find leaderboard entries in page HTML")
        return None

    logger.info(f"Extracted {len(entries)} total models from arena.ai")

    # Filter to open-source only
    open_source = [
        e for e in entries
        if e.get("license", "Proprietary") not in PROPRIETARY_LICENSES
    ]
    logger.info(f"Filtered to {len(open_source)} open-source models")

    if not open_source:
        return None

    # Convert to DataFrame with sequential open-source ranks
    rows = []
    for i, entry in enumerate(open_source, start=1):
        rows.append({
            "rank": i,
            "model_name": entry.get("modelDisplayName", ""),
            "arena_score": entry.get("rating"),
            "ci_lower": entry.get("ratingLower"),
            "ci_upper": entry.get("ratingUpper"),
            "votes": entry.get("votes"),
            "organization": entry.get("modelOrganization", ""),
            "license": entry.get("license", ""),
        })

    df = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
    logger.info(f"Built DataFrame with {len(df)} open-source models")
    return df


def _extract_entries_from_nextjs(html: str) -> list[dict[str, Any]] | None:
    """
    Extract the leaderboard entries array from Next.js embedded data.

    Arena.ai uses Next.js streaming which embeds data as double-escaped JSON
    inside self.__next_f.push() calls. The entries are in a pattern like:
        \\"entries\\":[{\\"rank\\":1,...},...]
    """
    # Find the start of the entries array
    marker = '\\"entries\\":['
    idx = html.find(marker)
    if idx < 0:
        return None

    # Start after the marker, at the opening [
    arr_start = idx + len(marker) - 1  # points to '['

    # Extract a generous chunk and unescape the double-escaped JSON
    chunk = html[arr_start:arr_start + 500000]
    unescaped = chunk.replace('\\"', '"')

    # Find the matching closing bracket
    depth = 0
    end = -1
    for i, c in enumerate(unescaped):
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end < 0:
        logger.warning("Could not find end of entries array")
        return None

    try:
        entries = json.loads(unescaped[:end])
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse entries JSON: {e}")
        return None

    # Validate that it looks like leaderboard data
    if not entries or not isinstance(entries, list):
        return None
    sample = entries[0]
    if "modelDisplayName" not in sample or "rating" not in sample:
        logger.warning("Entries don't look like leaderboard data")
        return None

    return entries


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
