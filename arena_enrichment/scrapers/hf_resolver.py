"""
Resolve model parameter counts from HuggingFace API.

Uses the HuggingFace model info API (unauthenticated) to:
1. Look up models by known repo ID (ARENA_TO_HF mapping)
2. Search for models by name with fuzzy matching
3. Extract param counts from safetensors metadata or config.json
"""

import logging
import re
import time

import requests

logger = logging.getLogger(__name__)

HF_API_BASE = "https://huggingface.co/api/models"
HF_RAW_BASE = "https://huggingface.co"

# Rate limiting
_last_request_time = 0
RATE_LIMIT_SECONDS = 1.0


def _rate_limit():
    """Enforce rate limiting between HuggingFace API calls."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < RATE_LIMIT_SECONDS:
        time.sleep(RATE_LIMIT_SECONDS - elapsed)
    _last_request_time = time.time()


def _safe_get(url, timeout=10):
    """Make a rate-limited GET request."""
    _rate_limit()
    try:
        resp = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (compatible; ArenaEnrichment/1.0)",
        })
        if resp.status_code == 200:
            return resp.json()
    except (requests.RequestException, ValueError) as e:
        logger.debug(f"Request failed for {url}: {e}")
    return None


def resolve_from_huggingface(model_name, hf_id=None):
    """
    Try to resolve model parameters from HuggingFace.

    Args:
        model_name: Arena model name (e.g., "Gemma 3 27B IT")
        hf_id: Optional known HuggingFace repo ID

    Returns:
        dict with {total_params_b, active_params_b, architecture} or None
    """
    if hf_id:
        result = _resolve_by_repo_id(hf_id)
        if result:
            return result

    result = _search_and_resolve(model_name)
    if result:
        return result

    return None


def _resolve_by_repo_id(repo_id):
    """Resolve params from a specific HuggingFace repo."""
    logger.info(f"Resolving HF repo: {repo_id}")
    data = _safe_get(f"{HF_API_BASE}/{repo_id}")
    if not data:
        return None
    return _extract_params_from_model_info(data, repo_id)


def _search_and_resolve(model_name):
    """Search HuggingFace for the model and try to resolve params."""
    search_terms = _generate_search_terms(model_name)

    for search_term in search_terms:
        logger.info(f"Searching HuggingFace for: {search_term}")
        data = _safe_get(
            f"{HF_API_BASE}?search={search_term}&limit=5&sort=downloads&direction=-1"
        )
        if not data or not isinstance(data, list):
            continue

        for model_info in data:
            model_id = model_info.get("modelId", "")
            if _is_good_match(model_name, model_id):
                result = _extract_params_from_model_info(model_info, model_id)
                if result:
                    logger.info(f"Matched '{model_name}' to HF repo '{model_id}'")
                    return result

    return None


def _generate_search_terms(name):
    """
    Generate multiple search terms from an arena model name.

    Arena names can differ significantly from HF repo names:
      "Gemma 3 27B IT" → ["gemma-3-27b", "gemma-3-27b-it"]
      "Command R+" → ["command-r-plus", "c4ai-command-r"]
      "Mistral-Small-2506-24B" → ["mistral-small-24b", "mistral-small-2506"]
    """
    terms = []

    # Basic: replace spaces with hyphens
    base = name.replace(" ", "-")
    terms.append(base)

    # Remove common suffixes (Instruct, IT, Chat, Preview)
    cleaned = re.sub(
        r'[\s\-]*(Instruct|IT|Chat|Preview|Online)\s*$', '', name, flags=re.IGNORECASE
    )
    if cleaned != name:
        terms.append(cleaned.replace(" ", "-"))

    # Remove date-like version numbers (e.g., 2506, 2507)
    no_date = re.sub(r'[\s\-]?\d{4}(?=[\s\-]|$)', '', cleaned)
    if no_date != cleaned:
        terms.append(no_date.replace(" ", "-"))

    # Handle special characters
    special = name.replace("+", "-plus").replace(" ", "-")
    if special != base:
        terms.append(special)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in terms:
        t_lower = t.lower().strip("-")
        if t_lower and t_lower not in seen:
            seen.add(t_lower)
            unique.append(t_lower)

    return unique


def _normalize(name):
    """Normalize a name for comparison: lowercase, strip separators."""
    return re.sub(r'[\s\-_]+', '', name.lower())


def _is_good_match(arena_name, hf_id):
    """Check if a HuggingFace model ID is a good match for an arena name."""
    arena_norm = _normalize(arena_name)
    hf_repo_name = hf_id.split("/")[-1] if "/" in hf_id else hf_id
    hf_norm = _normalize(hf_repo_name)

    # Exact normalized match
    if arena_norm == hf_norm:
        return True

    # Arena name contained in HF ID (or vice versa)
    if arena_norm in hf_norm or hf_norm in arena_norm:
        return True

    # Strip common suffixes and compare
    suffixes = ["-instruct", "-it", "-chat", "-preview", "-hf", "-online"]
    arena_stripped = arena_norm
    hf_stripped = hf_norm
    for suffix in suffixes:
        s = suffix.replace("-", "")
        if arena_stripped.endswith(s):
            arena_stripped = arena_stripped[:-len(s)]
        if hf_stripped.endswith(s):
            hf_stripped = hf_stripped[:-len(s)]

    if arena_stripped == hf_stripped:
        return True

    # Remove version dates and compare
    arena_no_date = re.sub(r'\d{4}', '', arena_stripped)
    hf_no_date = re.sub(r'\d{4}', '', hf_stripped)
    if arena_no_date and arena_no_date == hf_no_date:
        return True

    return False


def _extract_params_from_model_info(data, repo_id):
    """Extract parameter count from HuggingFace model info."""
    # Method 1: safetensors metadata (most reliable)
    safetensors = data.get("safetensors")
    if safetensors:
        total_params = safetensors.get("total")
        if total_params and total_params > 0:
            total_b = total_params / 1e9
            arch_info = _get_architecture_info(repo_id, total_b)
            return {
                "total_params_b": round(total_b, 1),
                "active_params_b": round(arch_info.get("active_b", total_b), 1),
                "architecture": arch_info.get("architecture", "dense"),
            }

    # Method 2: Try config.json for param estimation
    return _resolve_from_config(repo_id)


def _get_architecture_info(repo_id, total_b):
    """Check config.json for MoE architecture details."""
    data = _safe_get(f"{HF_RAW_BASE}/{repo_id}/raw/main/config.json")
    if not data:
        return {"active_b": total_b, "architecture": "dense"}

    num_experts = data.get("num_local_experts") or data.get("num_experts", 0)
    num_active = data.get("num_experts_per_tok") or data.get("num_selected_experts", 0)

    if num_experts > 1 and num_active > 0:
        expert_fraction = 0.6
        non_expert_fraction = 0.4
        active_ratio = non_expert_fraction + expert_fraction * (num_active / num_experts)
        active_b = total_b * active_ratio
        return {"active_b": round(active_b, 1), "architecture": "moe"}

    return {"active_b": total_b, "architecture": "dense"}


def _resolve_from_config(repo_id):
    """Try to estimate params from model config.json."""
    data = _safe_get(f"{HF_RAW_BASE}/{repo_id}/raw/main/config.json")
    if not data:
        return None

    hidden_size = data.get("hidden_size", 0)
    num_layers = data.get("num_hidden_layers", 0)
    vocab_size = data.get("vocab_size", 0)
    intermediate_size = data.get("intermediate_size", 0)
    num_kv_heads = data.get("num_key_value_heads", data.get("num_attention_heads", 0))
    num_heads = data.get("num_attention_heads", 0)

    if not all([hidden_size, num_layers, vocab_size]):
        return None

    embed_params = vocab_size * hidden_size * 2

    head_dim = hidden_size // num_heads if num_heads > 0 else 128
    attn_params = (
        hidden_size * num_heads * head_dim +
        hidden_size * num_kv_heads * head_dim +
        hidden_size * num_kv_heads * head_dim +
        num_heads * head_dim * hidden_size
    ) if num_heads > 0 else 4 * hidden_size * hidden_size

    ffn_params = (2 * hidden_size * intermediate_size
                  if intermediate_size
                  else 8 * hidden_size * hidden_size // 3)

    num_experts = data.get("num_local_experts") or data.get("num_experts", 0)
    num_active = data.get("num_experts_per_tok") or data.get("num_selected_experts", 0)

    if num_experts > 1:
        layer_params = attn_params + ffn_params * num_experts
        total = embed_params + layer_params * num_layers
        active_layer = attn_params + ffn_params * max(num_active, 1)
        active_total = embed_params + active_layer * num_layers
        total_b = total / 1e9
        active_b = active_total / 1e9
        return {
            "total_params_b": round(total_b, 1),
            "active_params_b": round(active_b, 1),
            "architecture": "moe",
        }
    else:
        layer_params = attn_params + ffn_params
        total = embed_params + layer_params * num_layers
        total_b = total / 1e9
        return {
            "total_params_b": round(total_b, 1),
            "active_params_b": round(total_b, 1),
            "architecture": "dense",
        }
