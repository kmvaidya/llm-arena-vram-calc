"""
Sizes of published checkpoints, from the Hugging Face API.

A model's release (16-bit, 8-bit or 4-bit, and the memory its weights need)
is the size of the weight files in its official repository: the top-level
.safetensors files.  Subfolders hold copies in other formats (gpt-oss keeps
its original weights in original/), and the per-dtype tensor counts the API
also reports undercount packed 4-bit formats, so neither is used.

Results are kept in a cache committed to the repository
(data/hf_checkpoints.json), so a daily run only asks about new repositories.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

import requests

from scrapers.aa_resolver import DATA_DIR, age_days

logger = logging.getLogger(__name__)

HF_MODEL_API = "https://huggingface.co/api/models/{repo}"
CACHE_FILE = os.path.join(DATA_DIR, "hf_checkpoints.json")
# Days before a repository is read again: published weights rarely change,
# while a repository without them may still be filling up
MAX_AGE_DAYS = 30
MISSING_MAX_AGE_DAYS = 7


class HFUnavailable(Exception):
    """The Hugging Face API could not be read (network error, rate limit or server error)."""


def _fetch_checkpoint_bytes(repo: str) -> int | None:
    """
    Bytes of a repository's top-level .safetensors files, or None when the
    repository does not exist or holds none.  Raises HFUnavailable when the
    API cannot be read, so that an outage is not remembered as a missing
    repository.
    """
    url = HF_MODEL_API.format(repo=repo)
    try:
        resp = requests.get(url, params={"blobs": "true"}, timeout=30)
    except requests.RequestException as e:
        raise HFUnavailable(f"{url}: {e}") from e
    if resp.status_code == 429 or resp.status_code >= 500:
        raise HFUnavailable(f"{url}: status {resp.status_code}")
    if resp.status_code != 200:
        # Hugging Face answers 401 for a repository that does not exist
        return None
    siblings = resp.json().get("siblings") or []
    total = sum(
        f.get("size") or 0 for f in siblings
        if f.get("rfilename", "").endswith(".safetensors") and "/" not in f["rfilename"]
    )
    return total or None


def _load_cache() -> dict[str, dict[str, Any]]:
    """The cached results, keyed by repository; an unreadable file raises."""
    if not os.path.exists(CACHE_FILE):
        return {}
    with open(CACHE_FILE, encoding="utf-8") as f:
        data: dict[str, dict[str, Any]] = json.load(f)
    return data


def _save_cache(cache: dict[str, dict[str, Any]]) -> None:
    """Write the cache sorted, through a copy swapped in, so an interrupted run leaves the old file."""
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    partial = CACHE_FILE + ".partial"
    with open(partial, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(partial, CACHE_FILE)


def validate_cache() -> None:
    """Raise if the committed cache cannot be read, before a run relies on it."""
    _load_cache()


def checkpoint_bytes(repo: str, fetch: bool = True) -> int | None:
    """
    Bytes of a repository's published weights, through the committed cache.

    The API is asked only when the cached result is older than MAX_AGE_DAYS
    (MISSING_MAX_AGE_DAYS when the repository had no weights) and fetch is
    True.  If it cannot be read, an expired size is still returned; without
    one, HFUnavailable is raised.  A size that disappears is kept too: a
    repository does not lose its weights, so it more likely moved.
    """
    cache = _load_cache()
    hit = cache.get(repo)
    if hit is not None:
        max_age = MAX_AGE_DAYS if hit.get("bytes") else MISSING_MAX_AGE_DAYS
        if age_days(hit.get("checked")) < max_age or not fetch:
            return hit.get("bytes")
    if not fetch:
        return None

    try:
        size = _fetch_checkpoint_bytes(repo)
    except HFUnavailable:
        if hit is not None and hit.get("bytes"):
            return int(hit["bytes"])
        raise
    if size is None and hit is not None and hit.get("bytes"):
        logger.warning(f"Hugging Face repository {repo} no longer lists weights; keeping the cached size")
        return int(hit["bytes"])

    cache[repo] = {"bytes": size, "checked": datetime.now(UTC).strftime("%Y-%m-%d")}
    _save_cache(cache)
    return size
