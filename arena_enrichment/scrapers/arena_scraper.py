"""
Scrape the Arena.ai open-source text leaderboard.

Tries multiple approaches in order:
1. Load from local CSV if provided
2. Fetch from arena.ai API endpoint
3. Use Playwright to render the JS page
4. Raise error with instructions
"""

import json
import logging
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

ARENA_URL = "https://arena.ai/leaderboard/text?license=open-source"

# Known API patterns for arena.ai / lmsys
API_ENDPOINTS = [
    "https://arena.ai/api/leaderboard?category=text&license=open-source",
    "https://arena.ai/api/v1/leaderboard?category=text&license=open-source",
    "https://arena.ai/api/leaderboard/text?license=open-source",
]

EXPECTED_COLUMNS = [
    "rank", "model_name", "arena_score", "ci_lower", "ci_upper",
    "votes", "organization", "license",
]


def scrape_arena_leaderboard(input_csv=None):
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

    # Try API endpoints first
    df = _try_api_endpoints()
    if df is not None:
        return df

    # Try HuggingFace Space (lmsys chatbot arena hosts data there)
    df = _try_huggingface_space()
    if df is not None:
        return df

    # Try Playwright browser rendering
    df = _try_playwright()
    if df is not None:
        return df

    raise RuntimeError(
        "Could not scrape arena.ai leaderboard. The page requires JavaScript "
        "rendering. Please either:\n"
        "  1. Install playwright: pip install playwright && playwright install chromium\n"
        "  2. Download the leaderboard as CSV and use: --input arena_data.csv\n"
        "\nAlternatively, export the table from https://arena.ai/leaderboard/text"
        "?license=open-source as CSV.\n"
    )


def _load_from_csv(path):
    """Load leaderboard from a local CSV file."""
    logger.info(f"Loading leaderboard from CSV: {path}")
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Try to map common alternative column names
    rename_map = {}
    for col in df.columns:
        if "rank" in col and "rank" not in rename_map.values():
            rename_map[col] = "rank"
        elif "model" in col and "name" in col and "model_name" not in rename_map.values():
            rename_map[col] = "model_name"
        elif col == "model" and "model_name" not in rename_map.values():
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
                *df[ci_col].apply(_parse_ci_string)
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


def _parse_ci_string(ci_str):
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


def _try_api_endpoints():
    """Try known API endpoint patterns."""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ArenaEnrichment/1.0)",
        "Accept": "application/json",
    }

    for url in API_ENDPOINTS:
        try:
            logger.info(f"Trying API endpoint: {url}")
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                df = _parse_api_response(data)
                if df is not None and len(df) > 10:
                    logger.info(f"Successfully loaded {len(df)} models from API")
                    return df
        except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
            logger.debug(f"API endpoint {url} failed: {e}")
            continue

    # Also try fetching the page HTML and looking for embedded JSON data
    try:
        logger.info("Trying to find embedded JSON in page HTML...")
        resp = requests.get(ARENA_URL, headers=headers, timeout=15)
        if resp.status_code == 200:
            df = _extract_json_from_html(resp.text)
            if df is not None and len(df) > 10:
                logger.info(f"Extracted {len(df)} models from embedded HTML data")
                return df
    except requests.RequestException as e:
        logger.debug(f"HTML fetch failed: {e}")

    return None


def _parse_api_response(data):
    """Parse various API response formats into a DataFrame."""
    # Handle direct list of models
    if isinstance(data, list) and len(data) > 0:
        df = pd.json_normalize(data)
        return _normalize_api_df(df)

    # Handle wrapped response
    if isinstance(data, dict):
        for key in ("data", "results", "models", "leaderboard", "rows"):
            if key in data and isinstance(data[key], list):
                df = pd.json_normalize(data[key])
                return _normalize_api_df(df)

    return None


def _normalize_api_df(df):
    """Normalize API DataFrame columns to expected format."""
    if df.empty:
        return None

    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl in ("rank", "#"):
            col_map[col] = "rank"
        elif "model" in cl and ("name" in cl or cl == "model"):
            col_map[col] = "model_name"
        elif "score" in cl or cl == "rating" or cl == "elo":
            col_map[col] = "arena_score"
        elif "vote" in cl or "num_battles" in cl:
            col_map[col] = "votes"
        elif "org" in cl or "provider" in cl:
            col_map[col] = "organization"
        elif "license" in cl:
            col_map[col] = "license"

    df = df.rename(columns=col_map)

    if "model_name" not in df.columns:
        return None

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    if "rank" not in df.columns or df["rank"].isna().all():
        df["rank"] = range(1, len(df) + 1)

    return df[EXPECTED_COLUMNS].copy()


def _extract_json_from_html(html):
    """Try to extract leaderboard data embedded in HTML page."""
    soup = BeautifulSoup(html, "lxml")

    # Look for JSON-LD
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
            if isinstance(data, dict) and data.get("@type") == "Dataset":
                logger.info("Found JSON-LD Dataset in page")
        except (json.JSONDecodeError, TypeError):
            continue

    # Look for __NEXT_DATA__ or similar embedded JSON
    for script in soup.find_all("script"):
        if not script.string:
            continue
        text = script.string.strip()

        # Next.js data
        if "__NEXT_DATA__" in text:
            m = re.search(r'__NEXT_DATA__\s*=\s*({.*?})\s*;?\s*$', text, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(1))
                    df = _search_nested_for_leaderboard(data)
                    if df is not None:
                        return df
                except json.JSONDecodeError:
                    continue

        # Generic JSON blob with model data
        if '"arena_score"' in text or '"elo_rating"' in text or '"rating"' in text:
            try:
                data = json.loads(text)
                df = _parse_api_response(data)
                if df is not None:
                    return df
            except json.JSONDecodeError:
                continue

    return None


def _search_nested_for_leaderboard(data, depth=0):
    """Recursively search nested JSON for leaderboard data."""
    if depth > 5:
        return None

    if isinstance(data, list) and len(data) > 10:
        if all(isinstance(item, dict) for item in data[:5]):
            sample = data[0]
            if any(k in str(sample.keys()).lower() for k in ("model", "score", "rating", "elo")):
                df = _parse_api_response(data)
                if df is not None:
                    return df

    if isinstance(data, dict):
        for value in data.values():
            result = _search_nested_for_leaderboard(value, depth + 1)
            if result is not None:
                return result

    return None


def _try_huggingface_space():
    """
    Try to fetch leaderboard data from the lmsys HuggingFace Space.

    The Chatbot Arena leaderboard data is hosted on HuggingFace Spaces.
    We try to find and download the data files.
    """
    HF_SPACE_API = "https://huggingface.co/api/spaces/lmarena-ai/chatbot-arena-leaderboard/tree/main"

    try:
        logger.info("Trying HuggingFace Space for arena data...")
        resp = requests.get(HF_SPACE_API, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (compatible; ArenaEnrichment/1.0)",
        })
        if resp.status_code != 200:
            logger.debug(f"HF Space API returned {resp.status_code}")
            return None

        files = resp.json()
        if not isinstance(files, list):
            return None

        # Look for CSV or JSON data files
        data_files = [
            f for f in files
            if isinstance(f, dict) and f.get("path", "").endswith((".csv", ".json", ".jsonl"))
            and "leaderboard" in f.get("path", "").lower()
        ]

        for file_info in data_files:
            path = file_info["path"]
            url = f"https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard/resolve/main/{path}"
            logger.info(f"Downloading HF Space file: {path}")

            file_resp = requests.get(url, timeout=30)
            if file_resp.status_code != 200:
                continue

            if path.endswith(".csv"):
                from io import StringIO
                df = pd.read_csv(StringIO(file_resp.text))
                df = _normalize_api_df(df)
                if df is not None and len(df) > 10:
                    logger.info(f"Loaded {len(df)} models from HF Space CSV")
                    return df

            elif path.endswith(".json"):
                data = file_resp.json()
                df = _parse_api_response(data)
                if df is not None and len(df) > 10:
                    logger.info(f"Loaded {len(df)} models from HF Space JSON")
                    return df

    except Exception as e:
        logger.debug(f"HuggingFace Space approach failed: {e}")

    return None


def _try_playwright():
    """Use Playwright to render the JS page and scrape the table."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.warning(
            "Playwright not installed. Install with: "
            "pip install playwright && playwright install chromium"
        )
        return None

    logger.info("Attempting Playwright browser scraping...")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(ARENA_URL, timeout=30000)

            # Wait for the table to render
            page.wait_for_selector("table", timeout=15000)

            # Give extra time for data to populate
            page.wait_for_timeout(3000)

            html = page.content()
            browser.close()

        return _parse_html_table(html)
    except Exception as e:
        logger.warning(f"Playwright scraping failed: {e}")
        return None


def _parse_html_table(html):
    """Parse leaderboard table from rendered HTML."""
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")

    if not tables:
        logger.warning("No tables found in rendered HTML")
        return None

    # Find the largest table (likely the leaderboard)
    best_table = max(tables, key=lambda t: len(t.find_all("tr")))
    rows = best_table.find_all("tr")

    if len(rows) < 5:
        logger.warning(f"Table too small: {len(rows)} rows")
        return None

    # Extract headers
    header_row = rows[0]
    headers = [th.get_text(strip=True).lower() for th in header_row.find_all(["th", "td"])]

    # Extract data rows
    data = []
    for row in rows[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        if cells:
            data.append(cells)

    if not data:
        return None

    df = pd.DataFrame(data, columns=headers[:len(data[0])] if headers else None)
    return _normalize_api_df(df)
