"""
Resolve model parameter counts from Artificial Analysis.

Fetches model data from artificialanalysis.ai via the React Server Component
stream (one HTTP request returns all ~400 models with full metadata).  Results
are cached locally as JSON so repeated runs avoid hitting the site.

Per-model HTML scraping is available as a targeted fallback for models that
are missing from the bulk data.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, TypedDict

import requests
from bs4 import BeautifulSoup


class AAEntry(TypedDict):
    """Processed model entry in the AA lookup cache."""

    name: str
    slug: str
    is_open_weights: bool
    context_window: int | None
    total_params_b: float | None
    active_params_b: float | None
    architecture: str | None


class ModelParams(TypedDict):
    """Resolved model parameter data returned by resolution functions."""

    total_params_b: float
    active_params_b: float
    architecture: str

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "aa_models.json")
DEFAULT_MAX_AGE_HOURS = 24

AA_LEADERBOARD_URL = "https://artificialanalysis.ai/leaderboards/models"
AA_MODEL_BASE_URL = "https://artificialanalysis.ai/models"

# Rate limiting for per-model fallback requests
_last_request_time: float = 0
RATE_LIMIT_SECONDS = 1.5


def _rate_limit() -> None:
    """Enforce rate limiting between requests."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < RATE_LIMIT_SECONDS:
        time.sleep(RATE_LIMIT_SECONDS - elapsed)
    _last_request_time = time.time()


# ---------------------------------------------------------------------------
# Bulk fetch via RSC stream
# ---------------------------------------------------------------------------

def _fetch_all_via_rsc() -> list[dict[str, Any]]:
    """
    Fetch all models from Artificial Analysis in a single HTTP request.

    Uses the Next.js React Server Component streaming protocol to get the
    same JSON payload the browser receives.  Returns a list of raw model
    dicts (100+ fields each) or an empty list on failure.
    """
    logger.info("Fetching all models from Artificial Analysis (RSC stream)...")
    try:
        resp = requests.get(
            AA_LEADERBOARD_URL,
            headers={
                "RSC": "1",
                "Next-Router-State-Tree": "%5B%22%22%5D",
                "Next-Url": "/leaderboards/models",
                "User-Agent": "Mozilla/5.0 (compatible; ArenaEnrichment/2.0)",
            },
            timeout=30,
        )
        if resp.status_code != 200:
            logger.warning(f"RSC stream returned {resp.status_code}")
            return []
    except requests.RequestException as e:
        logger.warning(f"RSC stream request failed: {e}")
        return []

    text = resp.text
    models = _extract_models_from_rsc(text)
    logger.info(f"RSC stream: extracted {len(models)} models")
    return models


def _extract_models_from_rsc(text: str) -> list[dict[str, Any]]:
    """Parse the model array out of the RSC payload."""
    # The model array contains objects with 'inference_parameters_active_billions'.
    # Find this marker, then scan backwards to the array start '[{'.
    marker = '"inference_parameters_active_billions"'
    idx = text.find(marker)
    if idx < 0:
        logger.warning("RSC payload: marker field not found")
        return []

    # Scan backwards for the opening '[{' of the array
    search_start = max(0, idx - 50000)
    arr_start = text.rfind("[{", search_start, idx)
    if arr_start < 0:
        logger.warning("RSC payload: could not locate array start")
        return []

    # Find matching closing bracket
    depth = 0
    end = -1
    for i in range(arr_start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end < 0:
        logger.warning("RSC payload: could not locate array end")
        return []

    try:
        models = json.loads(text[arr_start:end])
    except json.JSONDecodeError as e:
        logger.warning(f"RSC payload: JSON parse error: {e}")
        return []

    if not models or not isinstance(models[0], dict):
        return []

    return models


# ---------------------------------------------------------------------------
# Per-model HTML fallback (Method 1: HTML table scraping)
# ---------------------------------------------------------------------------

def _fetch_single_model(slug: str) -> ModelParams | None:
    """
    Scrape parameter data from an individual model page as a fallback.

    Uses the <table class="text-sm"> Technical Specifications table.
    Returns a dict with total_params_b, active_params_b, architecture or None.
    """
    url = f"{AA_MODEL_BASE_URL}/{slug}"
    logger.info(f"Fetching single model from AA: {url}")
    _rate_limit()

    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (compatible; ArenaEnrichment/2.0)",
        })
        if resp.status_code != 200:
            return None
    except requests.RequestException:
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.select_one("table.text-sm")
    if not table:
        return None

    total_b = None
    active_b = None

    for row in table.select("tr"):
        label_span = row.select_one("th span.align-middle")
        td = row.select_one("td")
        if not label_span or not td:
            continue

        label = label_span.get_text(strip=True)
        if label == "Total parameters":
            total_b = _parse_param_value(td.get_text(strip=True))
        elif label == "Active parameters":
            val_span = td.select_one("span.align-middle")
            text = val_span.get_text(strip=True) if val_span else td.get_text(strip=True)
            active_b = _parse_param_value(text)

    if total_b is None:
        return None

    if active_b is None:
        active_b = total_b

    arch = "moe" if active_b < total_b * 0.9 else "dense"
    return {
        "total_params_b": round(total_b, 1),
        "active_params_b": round(active_b, 1),
        "architecture": arch,
    }


def _parse_param_value(text: str) -> float | None:
    """Parse a parameter value like '30.5B' or '671B' into a float in billions."""
    m = re.search(r"([\d.]+)\s*[Bb]", text)
    if m:
        return float(m.group(1))
    # Try plain number (already in billions from AA)
    m = re.search(r"([\d.]+)", text)
    if m:
        val = float(m.group(1))
        # If the value is very large, it's in raw params not billions
        if val > 10000:
            return val / 1e9
        return val
    return None


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def _load_cache() -> tuple[dict[str, AAEntry] | None, str | None]:
    """Load cached AA model data from disk.  Returns (models_dict, fetched_at) or (None, None)."""
    if not os.path.exists(CACHE_FILE):
        return None, None
    try:
        with open(CACHE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("models", {}), data.get("fetched_at")
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Cache load failed: {e}")
        return None, None


def _save_cache(models_dict: dict[str, AAEntry]) -> None:
    """Persist the processed model lookup dict to disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    data = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "model_count": len(models_dict),
        "models": models_dict,
    }
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Cache saved: {len(models_dict)} models → {CACHE_FILE}")


def _cache_is_fresh(fetched_at: str | None, max_age_hours: float) -> bool:
    """Check whether the cache timestamp is within the allowed age."""
    if not fetched_at:
        return False
    try:
        fetched = datetime.fromisoformat(fetched_at)
        age_hours = (datetime.now(timezone.utc) - fetched).total_seconds() / 3600
        return age_hours < max_age_hours
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Build the lookup index from raw AA model list
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    """Normalize a model name for fuzzy matching: lowercase, strip separators and punctuation."""
    # Strip spaces, hyphens, underscores, dots, parentheses
    return re.sub(r"[\s\-_.()]+", "", name.lower())


def _build_lookup(raw_models: list[dict[str, Any]]) -> dict[str, AAEntry]:
    """
    Convert the raw list of AA model dicts into a lookup dict keyed by
    multiple normalized name variants for flexible matching.

    Each entry contains: name, slug, total_params_b, active_params_b, architecture.
    """
    lookup: dict[str, Any] = {}

    for m in raw_models:
        name = m.get("name") or ""
        slug = m.get("slug") or ""
        total = m.get("parameters")           # Already in billions
        active = m.get("inference_parameters_active_billions") or m.get("activeParams")

        if not name:
            continue

        # Build the resolved entry
        entry = {
            "name": name,
            "slug": slug,
            "is_open_weights": m.get("is_open_weights", False),
            "context_window": m.get("context_window_tokens"),
        }

        if total is not None:
            total_b = float(total)
            active_b = float(active) if active is not None else total_b
            arch = "moe" if active_b < total_b * 0.9 else "dense"
            entry["total_params_b"] = round(total_b, 1)
            entry["active_params_b"] = round(active_b, 1)
            entry["architecture"] = arch
        else:
            entry["total_params_b"] = None
            entry["active_params_b"] = None
            entry["architecture"] = None

        # Index under multiple keys for flexible matching
        keys = set()
        keys.add(_normalize(name))
        if slug:
            keys.add(_normalize(slug))

        for key in keys:
            # Prefer entries that have parameter data
            if (
                key in lookup
                and lookup[key].get("total_params_b") is not None
                and entry.get("total_params_b") is None
            ):
                continue
            lookup[key] = entry

    return lookup


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_aa_models(max_age_hours: float = DEFAULT_MAX_AGE_HOURS, force_refresh: bool = False) -> tuple[dict[str, AAEntry], bool]:
    """
    Get the full AA model lookup, using cache when fresh.

    Returns:
        tuple of (lookup_dict, is_stale) where is_stale indicates the data
        came from a stale cache fallback (RSC fetch failed).
    """
    if not force_refresh:
        cached, fetched_at = _load_cache()
        if cached and _cache_is_fresh(fetched_at, max_age_hours):
            logger.info(f"Using cached AA data ({len(cached)} entries, fetched {fetched_at})")
            return cached, False

    # Fetch fresh data
    raw_models = _fetch_all_via_rsc()
    if not raw_models:
        # Fall back to stale cache if available
        cached, fetched_at = _load_cache()
        if cached:
            logger.warning(f"RSC fetch failed, using stale cache ({fetched_at})")
            return cached, True
        logger.warning("No AA data available (fetch failed, no cache)")
        return {}, True

    lookup = _build_lookup(raw_models)
    _save_cache(lookup)
    return lookup, False


def resolve_from_aa(model_name: str, aa_lookup: dict[str, AAEntry] | None = None) -> ModelParams | None:
    """
    Resolve model parameters from the Artificial Analysis dataset.

    Args:
        model_name: Arena model name (e.g., "deepseek-v3", "qwen3-30b-a3b")
        aa_lookup: Pre-loaded lookup dict (if None, loads/fetches automatically)

    Returns:
        dict with {total_params_b, active_params_b, architecture} or None
    """
    if aa_lookup is None:
        aa_lookup, _ = get_aa_models()

    if not aa_lookup:
        return None

    name_norm = _normalize(model_name)

    # 1. Exact match on normalized name/slug
    entry = aa_lookup.get(name_norm)

    # 2. Try with common suffixes/noise stripped.
    #    Arena names often include context window ("4k", "128k"), dates ("082024",
    #    "june2024"), and role tags ("instruct", "chat") that AA names omit.
    if not entry:
        # Strip trailing noise progressively
        stripped_variants = set()
        stripped_variants.add(re.sub(r"(instruct|chat|it|hf)$", "", name_norm))
        stripped_variants.add(re.sub(r"\d{6,8}$", "", name_norm))       # trailing dates
        stripped_variants.add(re.sub(r"[a-z]*\d{4}$", "", name_norm))   # "june2024"
        stripped_variants.add(re.sub(r"\d+k", "", name_norm))           # context "4k","128k"
        # Combine: strip context + suffix + date
        base = re.sub(r"\d+k", "", name_norm)
        base = re.sub(r"(instruct|chat|it|hf)$", "", base)
        base = re.sub(r"\d{6,8}$", "", base)
        base = re.sub(r"[a-z]*\d{4}$", "", base)
        stripped_variants.add(base)
        stripped_variants.discard(name_norm)  # don't re-check original
        stripped_variants.discard("")

        for variant in stripped_variants:
            if variant in aa_lookup:
                entry = aa_lookup[variant]
                break

    # 3. Substring matching — find the longest AA key contained in the arena name.
    #    Require the matched key to be at least 50% of the arena name length
    #    to avoid false positives like "qwen3" matching "qwen35397ba17b".
    if not entry:
        best_key = None
        best_len = 0
        min_key_len = max(6, len(name_norm) // 2)
        for key in aa_lookup:
            if len(key) >= min_key_len and len(key) > best_len and key in name_norm:
                best_key = key
                best_len = len(key)
        # Also try the reverse: arena name contained in an AA key
        if not best_key:
            for key in aa_lookup:
                if (
                    len(name_norm) >= min_key_len
                    and name_norm in key
                    and (not best_key or len(key) < len(best_key))
                ):
                    best_key = key
        if best_key:
            entry = aa_lookup[best_key]

    if not entry:
        return None

    total = entry.get("total_params_b")
    if total is None:
        return None

    active = entry.get("active_params_b")
    arch = entry.get("architecture")
    return {
        "total_params_b": total,
        "active_params_b": active if active is not None else total,
        "architecture": arch if arch is not None else "dense",
    }


def resolve_single_from_aa(model_name: str) -> ModelParams | None:
    """
    Fallback: scrape a single model page from AA by trying slug variants.

    This is expensive (one HTTP request per model) so should only be used
    for models not found in the bulk data.
    """
    # Generate slug candidates from the model name
    slug_base = re.sub(r"[\s_]+", "-", model_name.lower()).strip("-")
    slugs_to_try = [slug_base]

    # Also try without common suffixes
    for suffix in ["-instruct", "-chat", "-it", "-hf"]:
        if slug_base.endswith(suffix):
            slugs_to_try.append(slug_base[:-len(suffix)])

    for slug in slugs_to_try:
        result = _fetch_single_model(slug)
        if result:
            return result

    return None
