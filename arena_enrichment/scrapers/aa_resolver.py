"""
Resolve model parameter counts from Artificial Analysis.

Scrapes public pages from artificialanalysis.ai to get model parameter
information. No API key required — uses public HTML scraping.
"""

import json
import logging
import re
import time

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

AA_BASE = "https://artificialanalysis.ai"
AA_MODELS_PAGE = f"{AA_BASE}/models/open-source"

# Rate limiting
_last_request_time = 0
RATE_LIMIT_SECONDS = 1.5


def _rate_limit():
    """Enforce rate limiting between requests."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < RATE_LIMIT_SECONDS:
        time.sleep(RATE_LIMIT_SECONDS - elapsed)
    _last_request_time = time.time()


def _safe_get(url, timeout=15):
    """Make a rate-limited GET request returning raw text."""
    _rate_limit()
    try:
        resp = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (compatible; ArenaEnrichment/1.0)",
            "Accept": "text/html,application/xhtml+xml,application/json",
        })
        if resp.status_code == 200:
            return resp.text
    except requests.RequestException as e:
        logger.debug(f"Request failed for {url}: {e}")
    return None


# Cache for the model index (scraped once, reused for all lookups)
_model_index = None


def _build_model_index():
    """
    Scrape the Artificial Analysis open-source models page to build
    a lookup index of model names -> detail page URLs.
    """
    global _model_index
    if _model_index is not None:
        return _model_index

    _model_index = {}
    logger.info("Building Artificial Analysis model index...")

    html = _safe_get(AA_MODELS_PAGE)
    if not html:
        logger.warning("Could not fetch Artificial Analysis models page")
        return _model_index

    soup = BeautifulSoup(html, "lxml")

    # Look for JSON-LD dataset
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
            if isinstance(data, dict) and data.get("@type") == "Dataset":
                logger.info("Found JSON-LD Dataset on AA page")
                # Extract model entries if available
                _parse_jsonld_dataset(data)
        except (json.JSONDecodeError, TypeError):
            continue

    # Look for embedded JSON data (Next.js __NEXT_DATA__ or similar)
    for script in soup.find_all("script"):
        if not script.string:
            continue
        if "__NEXT_DATA__" in script.string:
            try:
                m = re.search(r'__NEXT_DATA__\s*=\s*({.*?})\s*;?\s*$',
                              script.string, re.DOTALL)
                if m:
                    data = json.loads(m.group(1))
                    _extract_models_from_next_data(data)
            except (json.JSONDecodeError, TypeError):
                continue

    # Also look for model links in the HTML
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/models/" in href and href != "/models/open-source":
            name = link.get_text(strip=True)
            if name and len(name) > 2:
                slug = href.rstrip("/").split("/")[-1]
                _model_index[_normalize(name)] = {
                    "name": name,
                    "slug": slug,
                    "url": f"{AA_BASE}{href}" if href.startswith("/") else href,
                }

    logger.info(f"Built AA index with {len(_model_index)} models")
    return _model_index


def _parse_jsonld_dataset(data):
    """Extract model data from JSON-LD Dataset."""
    global _model_index
    # JSON-LD datasets might have distribution or hasPart with model refs
    for key in ("distribution", "hasPart", "dataset"):
        items = data.get(key, [])
        if isinstance(items, list):
            for item in items:
                name = item.get("name", "")
                url = item.get("url", "")
                if name:
                    _model_index[_normalize(name)] = {
                        "name": name,
                        "slug": url.rstrip("/").split("/")[-1] if url else "",
                        "url": url,
                        "data": item,
                    }


def _extract_models_from_next_data(data, depth=0):
    """Recursively extract model data from Next.js page data."""
    global _model_index
    if depth > 6:
        return

    if isinstance(data, dict):
        # Check if this dict represents a model
        name = data.get("model_name") or data.get("name") or data.get("modelName")
        params = data.get("total_params") or data.get("parameters") or data.get("param_count")
        if name and params:
            _model_index[_normalize(str(name))] = {
                "name": str(name),
                "data": data,
            }
            return

        for value in data.values():
            _extract_models_from_next_data(value, depth + 1)

    elif isinstance(data, list):
        for item in data:
            _extract_models_from_next_data(item, depth + 1)


def _normalize(name):
    """Normalize model name for fuzzy matching."""
    return re.sub(r'[\s\-_]+', '', name.lower())


def resolve_from_artificial_analysis(model_name):
    """
    Try to resolve model parameters from Artificial Analysis.

    Args:
        model_name: Arena model name

    Returns:
        dict with {total_params_b, active_params_b, architecture} or None
    """
    index = _build_model_index()
    if not index:
        return None

    norm_name = _normalize(model_name)

    # Try exact match
    entry = index.get(norm_name)

    # Try substring matching
    if not entry:
        for key, val in index.items():
            if norm_name in key or key in norm_name:
                entry = val
                break

    if not entry:
        return None

    # If the entry already has param data from embedded JSON
    data = entry.get("data", {})
    total = data.get("total_params") or data.get("parameters")
    active = data.get("active_params") or data.get("activated_params")
    if total:
        total_b = float(total) / 1e9 if float(total) > 1000 else float(total)
        active_b = float(active) / 1e9 if active and float(active) > 1000 else (float(active) if active else total_b)
        arch = "moe" if (active_b < total_b * 0.9) else "dense"
        return {
            "total_params_b": round(total_b, 1),
            "active_params_b": round(active_b, 1),
            "architecture": arch,
        }

    # Try fetching the model detail page
    slug = entry.get("slug", "")
    if slug:
        return _scrape_model_detail(slug)

    return None


def _scrape_model_detail(slug):
    """Scrape parameter info from an individual model detail page."""
    url = f"{AA_BASE}/models/{slug}"
    logger.info(f"Scraping AA model detail: {url}")

    html = _safe_get(url)
    if not html:
        return None

    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)

    # Look for parameter information in the page text
    # Common patterns: "Parameters: 70B", "70 billion parameters"
    total_b = None
    active_b = None

    # Pattern: "X billion parameters" or "XB parameters"
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:billion|B)\s*(?:total\s*)?param', text, re.IGNORECASE)
    if m:
        total_b = float(m.group(1))

    # Pattern: "Active parameters: XB"
    m = re.search(r'active\s*param[^:]*:\s*(\d+(?:\.\d+)?)\s*(?:billion|B)', text, re.IGNORECASE)
    if m:
        active_b = float(m.group(1))

    # Also check embedded JSON in the page
    for script in soup.find_all("script"):
        if not script.string:
            continue
        try:
            data = json.loads(script.string)
            if isinstance(data, dict):
                t = data.get("total_params") or data.get("parameters")
                a = data.get("active_params")
                if t:
                    total_b = float(t) / 1e9 if float(t) > 1000 else float(t)
                if a:
                    active_b = float(a) / 1e9 if float(a) > 1000 else float(a)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    if total_b:
        if not active_b:
            active_b = total_b
        arch = "moe" if (active_b < total_b * 0.9) else "dense"
        return {
            "total_params_b": round(total_b, 1),
            "active_params_b": round(active_b, 1),
            "architecture": arch,
        }

    return None
