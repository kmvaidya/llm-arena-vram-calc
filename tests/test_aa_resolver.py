
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
import requests
from scrapers import aa_resolver
from scrapers.aa_resolver import (
    AAUnavailable,
    _build_lookup,
    _extract_models_from_rsc,
    _fetch_single_model,
    _normalize,
    _parse_param_value,
    get_model_params,
    resolve_from_aa,
    resolve_single_from_aa,
)

# ── _normalize ────────────────────────────────────────────────────────────

class TestNormalize:
    def test_lowercase(self):
        assert _normalize("DeepSeek-V3") == "deepseekv3"

    def test_strip_hyphens(self):
        assert _normalize("llama-3-70b") == "llama370b"

    def test_strip_underscores(self):
        assert _normalize("llama_3_70b") == "llama370b"

    def test_strip_spaces(self):
        assert _normalize("Gemma 3 27B") == "gemma327b"

    def test_strip_dots(self):
        assert _normalize("Llama-3.3-70B") == "llama3370b"

    def test_strip_parentheses(self):
        assert _normalize("GPT-OSS-120B (high)") == "gptoss120bhigh"

    def test_empty_string(self):
        assert _normalize("") == ""

    def test_already_normalized(self):
        assert _normalize("abc123") == "abc123"

    def test_mixed_separators(self):
        assert _normalize("A-B_C.D (E)") == "abcde"


# ── _parse_param_value ────────────────────────────────────────────────────

class TestParseParamValue:
    def test_billions_with_uppercase_b(self):
        assert _parse_param_value("70.6B") == 70.6

    def test_billions_with_lowercase_b(self):
        assert _parse_param_value("70.6b") == 70.6

    def test_integer_billions(self):
        assert _parse_param_value("671B") == 671.0

    def test_plain_number_small(self):
        assert _parse_param_value("70.6") == 70.6

    def test_very_large_plain_number(self):
        # > 10000 means raw param count, not billions
        assert _parse_param_value("671000000000") == pytest.approx(671.0)

    def test_text_with_b(self):
        assert _parse_param_value("Parameters: 30.5 B") == 30.5

    def test_trillions(self):
        # AA lists DeepSeek V4 Pro as "1.6T"; reading it as 1.6B put a 1.6T
        # model at the top of every single-GPU table
        assert _parse_param_value("1.6T") == pytest.approx(1600.0)

    def test_millions(self):
        assert _parse_param_value("500M") == pytest.approx(0.5)

    def test_unit_words(self):
        assert _parse_param_value("1.2 trillion") == pytest.approx(1200.0)
        assert _parse_param_value("30 billion") == 30.0

    def test_unit_followed_by_description(self):
        # The active-parameters cell carries a description after the value
        assert _parse_param_value("49B Number of parameters active per token") == 49.0

    def test_word_starting_with_unit_letter_is_not_a_unit(self):
        assert _parse_param_value("8 total") == 8.0

    def test_no_number(self):
        assert _parse_param_value("N/A") is None

    def test_empty_string(self):
        assert _parse_param_value("") is None


# ── _build_lookup ─────────────────────────────────────────────────────────

class TestBuildLookup:
    def _make_raw_model(self, name, slug="", **extra):
        m = {"name": name, "slug": slug, "isOpenWeights": True,
             "contextWindowTokens": 4096, "paramClass": "large"}
        m.update(extra)
        return m

    def test_basic_indexing(self):
        raw = [self._make_raw_model("Llama 3 70B", slug="llama-3-70b")]
        lookup = _build_lookup(raw)
        assert "llama370b" in lookup
        assert lookup["llama370b"]["slug"] == "llama-3-70b"

    def test_slug_also_indexed(self):
        raw = [self._make_raw_model("GLM-5.3 (Max)", slug="glm-5-3")]
        lookup = _build_lookup(raw)
        assert lookup["glm53"] is lookup["glm53max"]

    def test_fields_mapped(self):
        raw = [self._make_raw_model("X", isOpenWeights=False, contextWindowTokens=8192)]
        entry = _build_lookup(raw)["x"]
        assert entry["is_open_weights"] is False
        assert entry["context_window"] == 8192

    def test_params_left_for_model_page(self):
        raw = [self._make_raw_model("NoParams")]
        lookup = _build_lookup(raw)
        assert lookup["noparams"]["total_params_b"] is None
        assert lookup["noparams"]["architecture"] is None

    def test_nameless_entry_skipped(self):
        assert _build_lookup([self._make_raw_model("", slug="ghost")]) == {}

    def test_no_family_slug_indexing(self):
        """Family slug must NOT be used as a lookup key (caused false matches)."""
        raw = [self._make_raw_model(
            "Qwen1.5 110B", slug="qwen-1.5-110b", model_family_slug="qwen15",
        )]
        lookup = _build_lookup(raw)
        # The family slug key should NOT exist
        assert "qwen15" not in lookup
        # But name and slug keys should
        assert "qwen15110b" in lookup


# ── resolve_from_aa ───────────────────────────────────────────────────────

class TestResolveFromAa:
    def test_exact_match(self, aa_lookup):
        result = resolve_from_aa("deepseek-v3", aa_lookup=aa_lookup)
        assert result is not None
        assert result["total_params_b"] == 671.0
        assert result["architecture"] == "moe"

    def test_suffix_stripping_instruct(self, aa_lookup):
        result = resolve_from_aa("llama-3-70b-instruct", aa_lookup=aa_lookup)
        assert result is not None
        assert result["total_params_b"] == 70.6

    def test_suffix_stripping_chat(self, aa_lookup):
        # "chat" suffix should be stripped
        result = resolve_from_aa("gemma-3-27b-chat", aa_lookup=aa_lookup)
        assert result is not None
        assert result["total_params_b"] == 27.2

    def test_context_window_stripping(self, aa_lookup):
        # "128k" should be stripped from the name
        result = resolve_from_aa("llama-3-70b-128k", aa_lookup=aa_lookup)
        assert result is not None
        assert result["total_params_b"] == 70.6

    def test_substring_matching(self, aa_lookup):
        # Longer arena name contains an AA key as substring.
        # Key "commandrplus" (12 chars) inside "commandrplus082024" (18 chars).
        # min_key_len = max(6, 18//2) = 9, so 12 >= 9 passes.
        result = resolve_from_aa("command-r-plus-08-2024", aa_lookup=aa_lookup)
        assert result is not None
        assert result["total_params_b"] == 104.0

    def test_substring_min_length_guard(self, aa_lookup):
        """Short AA keys (like 'xy', 2 chars) must NOT match long arena names."""
        result = resolve_from_aa("some-very-long-model-name-xyz-instruct", aa_lookup=aa_lookup)
        # 'xy' is only 2 chars, min_key_len = max(6, 19) = 19, so it shouldn't match
        assert result is None

    def test_null_params_returns_none(self, aa_lookup):
        result = resolve_from_aa("null-params", aa_lookup=aa_lookup)
        assert result is None

    def test_empty_lookup_returns_none(self):
        result = resolve_from_aa("deepseek-v3", aa_lookup={})
        assert result is None

    def test_no_match_returns_none(self, aa_lookup):
        result = resolve_from_aa("completely-unknown-model-xyz", aa_lookup=aa_lookup)
        assert result is None

    def test_returns_correct_keys(self, aa_lookup):
        result = resolve_from_aa("deepseek-v3", aa_lookup=aa_lookup)
        assert set(result.keys()) == {"total_params_b", "active_params_b", "architecture"}


# ── _extract_models_from_rsc ──────────────────────────────────────────────

class TestExtractModelsFromRsc:
    def test_valid_payload(self, sample_rsc_payload):
        models = _extract_models_from_rsc(sample_rsc_payload)
        assert len(models) == 2
        assert models[0]["name"] == "Test Model 70B"
        assert models[1]["slug"] == "test-moe"

    def test_no_marker(self):
        assert _extract_models_from_rsc("no relevant data here") == []

    def test_malformed_json(self):
        text = 'prefix[{"isOpenWeights": broken'
        assert _extract_models_from_rsc(text) == []

    def test_no_array_start(self):
        # Marker exists but no '[{' before it
        text = '"isOpenWeights": true'
        assert _extract_models_from_rsc(text) == []


# ── model pages ───────────────────────────────────────────────────────────

def _spec_page(total=None, active=None):
    """An AA model page with the Technical Specifications rows we read."""
    rows = ""
    if total is not None:
        rows += (
            '<tr><th scope="row" class="text-left"><svg></svg>'
            '<span class="align-middle ml-1">Total parameters</span></th>'
            f'<td class="py-2 text-right font-medium">{total}</td></tr>'
        )
    if active is not None:
        rows += (
            '<tr><th scope="row" class="text-left"><svg></svg>'
            '<span class="align-middle ml-1">Active parameters</span></th>'
            f'<td><span class="align-middle mr-1">{active}</span><button></button>'
            '<div>Number of parameters active per token during inference</div></td></tr>'
        )
    return f'<html><body><table class="text-sm w-full">{rows}</table></body></html>'


@pytest.fixture
def no_rate_limit(monkeypatch):
    monkeypatch.setattr(aa_resolver, "_rate_limit", lambda: None)


@pytest.fixture
def params_cache(monkeypatch, tmp_path):
    """Point the committed page cache at a temporary file."""
    path = tmp_path / "aa_params.json"
    monkeypatch.setattr(aa_resolver, "PARAMS_CACHE_FILE", str(path))
    return path


def _days_ago(days):
    return (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%d")


@pytest.mark.usefixtures("no_rate_limit")
class TestFetchSingleModel:
    @patch("scrapers.aa_resolver.requests.get")
    def test_moe_in_trillions(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, text=_spec_page("1.6T", "49B"))
        assert _fetch_single_model("deepseek-v4-pro") == {
            "total_params_b": 1600.0, "active_params_b": 49.0, "architecture": "moe",
        }

    @patch("scrapers.aa_resolver.requests.get")
    def test_dense_without_active_row(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, text=_spec_page("70.64B"))
        assert _fetch_single_model("x") == {
            "total_params_b": 70.6, "active_params_b": 70.6, "architecture": "dense",
        }

    @patch("scrapers.aa_resolver.requests.get")
    def test_page_without_parameters(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, text=_spec_page())
        assert _fetch_single_model("x") is None

    @patch("scrapers.aa_resolver.requests.get")
    def test_missing_page(self, mock_get):
        mock_get.return_value = MagicMock(status_code=404, text="")
        assert _fetch_single_model("no-such-model") is None

    @pytest.mark.parametrize("status", [429, 500, 503])
    @patch("scrapers.aa_resolver.requests.get")
    def test_server_trouble_raises(self, mock_get, status):
        mock_get.return_value = MagicMock(status_code=status, text="")
        with pytest.raises(AAUnavailable):
            _fetch_single_model("x")

    @patch("scrapers.aa_resolver.requests.get", side_effect=requests.Timeout("slow"))
    def test_network_error_raises(self, mock_get):
        with pytest.raises(AAUnavailable):
            _fetch_single_model("x")


class TestGetModelParams:
    PARAMS = {"total_params_b": 1600.0, "active_params_b": 49.0, "architecture": "moe"}

    def _write(self, path, slug, checked, params=None):
        entry = dict(params or {"total_params_b": None, "active_params_b": None,
                                "architecture": None})
        entry["checked"] = checked
        path.write_text(json.dumps({slug: entry}))

    @patch("scrapers.aa_resolver._fetch_single_model")
    def test_fetches_and_caches(self, mock_fetch, params_cache):
        mock_fetch.return_value = self.PARAMS
        assert get_model_params("deepseek-v4-pro") == self.PARAMS
        saved = json.loads(params_cache.read_text())["deepseek-v4-pro"]
        assert saved["total_params_b"] == 1600.0
        assert saved["checked"] == _days_ago(0)

    @patch("scrapers.aa_resolver._fetch_single_model")
    def test_fresh_hit_skips_fetch(self, mock_fetch, params_cache):
        self._write(params_cache, "m", _days_ago(29), self.PARAMS)
        assert get_model_params("m") == self.PARAMS
        mock_fetch.assert_not_called()

    @patch("scrapers.aa_resolver._fetch_single_model")
    def test_expired_hit_refetched(self, mock_fetch, params_cache):
        self._write(params_cache, "m", _days_ago(31), self.PARAMS)
        mock_fetch.return_value = {"total_params_b": 1.0, "active_params_b": 1.0,
                                   "architecture": "dense"}
        assert get_model_params("m")["total_params_b"] == 1.0

    @patch("scrapers.aa_resolver._fetch_single_model")
    def test_missing_params_remembered_for_a_week(self, mock_fetch, params_cache):
        self._write(params_cache, "m", _days_ago(6))
        assert get_model_params("m") is None
        mock_fetch.assert_not_called()

        self._write(params_cache, "m", _days_ago(8))
        mock_fetch.return_value = None
        assert get_model_params("m") is None
        mock_fetch.assert_called_once_with("m")

    @patch("scrapers.aa_resolver._fetch_single_model", return_value=None)
    def test_page_that_lost_its_sizes_keeps_them(self, mock_fetch, params_cache, monkeypatch):
        monkeypatch.setattr(aa_resolver, "LOST_SIZES", set())
        self._write(params_cache, "m", _days_ago(31), self.PARAMS)
        assert get_model_params("m") == self.PARAMS
        assert {"m"} == aa_resolver.LOST_SIZES
        # Left expired, so the next run reads the page again
        assert json.loads(params_cache.read_text())["m"]["checked"] == _days_ago(31)

    @patch("scrapers.aa_resolver._fetch_single_model", side_effect=AAUnavailable("down"))
    def test_outage_keeps_expired_params(self, mock_fetch, params_cache):
        self._write(params_cache, "m", _days_ago(90), self.PARAMS)
        assert get_model_params("m") == self.PARAMS

    @patch("scrapers.aa_resolver._fetch_single_model", side_effect=AAUnavailable("down"))
    def test_outage_without_params_raises_and_caches_nothing(self, mock_fetch, params_cache):
        with pytest.raises(AAUnavailable):
            get_model_params("m")
        assert not params_cache.exists()


class TestResolveFromAaWithPages:
    def _lookup(self):
        return _build_lookup([{"name": "GLM-5.3 (Max)", "slug": "glm-5-3", "isOpenWeights": True}])

    @patch("scrapers.aa_resolver.get_model_params")
    def test_reads_matched_models_page(self, mock_params):
        mock_params.return_value = {"total_params_b": 754.0, "active_params_b": 40.0,
                                    "architecture": "moe"}
        lookup = self._lookup()
        result = resolve_from_aa("glm-5.3-max", aa_lookup=lookup, fetch_params=True)
        assert result["total_params_b"] == 754.0
        mock_params.assert_called_once_with("glm-5-3")
        # The entry keeps the counts for later names matching the same model
        assert lookup["glm53"]["total_params_b"] == 754.0

    @patch("scrapers.aa_resolver.get_model_params")
    def test_no_page_reads_without_fetch_params(self, mock_params):
        assert resolve_from_aa("glm-5.3-max", aa_lookup=self._lookup()) is None
        mock_params.assert_not_called()


class TestResolveSingleFromAa:
    @patch("scrapers.aa_resolver.get_model_params", return_value=None)
    def test_slug_candidates(self, mock_params):
        assert resolve_single_from_aa("GLM-4.6-Instruct") is None
        tried = [c.args[0] for c in mock_params.call_args_list]
        assert tried == ["glm-4.6-instruct", "glm-4.6", "glm-4-6-instruct", "glm-4-6"]

    @patch("scrapers.aa_resolver.get_model_params", return_value=None)
    def test_brackets_dropped_in_hyphenated_slug(self, mock_params):
        resolve_single_from_aa("mimo-v2-flash (thinking)")
        tried = [c.args[0] for c in mock_params.call_args_list]
        assert tried == ["mimo-v2-flash-(thinking)", "mimo-v2-flash-thinking", "mimo-v2-flash"]

    @patch("scrapers.aa_resolver.get_model_params", return_value=None)
    def test_variant_tags_stripped_one_at_a_time(self, mock_params):
        resolve_single_from_aa("deepseek-v3.2-exp-thinking")
        tried = [c.args[0] for c in mock_params.call_args_list]
        assert tried == [
            "deepseek-v3.2-exp-thinking", "deepseek-v3.2-exp", "deepseek-v3.2",
            "deepseek-v3-2-exp-thinking", "deepseek-v3-2-exp", "deepseek-v3-2",
        ]

    @patch("scrapers.aa_resolver.get_model_params", return_value=None)
    def test_non_thinking_stripped_whole(self, mock_params):
        resolve_single_from_aa("mimo-v2-flash (non-thinking)")
        tried = [c.args[0] for c in mock_params.call_args_list]
        assert tried == [
            "mimo-v2-flash-(non-thinking)", "mimo-v2-flash-non-thinking", "mimo-v2-flash",
        ]

    @patch("scrapers.aa_resolver.get_model_params", return_value=None)
    def test_reasoning_effort_stripped(self, mock_params):
        resolve_single_from_aa("deepseek-v4-flash-high-preview")
        tried = [c.args[0] for c in mock_params.call_args_list]
        assert tried == [
            "deepseek-v4-flash-high-preview", "deepseek-v4-flash-high", "deepseek-v4-flash",
        ]

    @patch("scrapers.aa_resolver.get_model_params", return_value=None)
    def test_dates_and_sizes_kept(self, mock_params):
        # A date or a "flash" names a different model, so it is never stripped
        resolve_single_from_aa("mistral-small-2506")
        resolve_single_from_aa("glm-4.7-flash")
        tried = [c.args[0] for c in mock_params.call_args_list]
        assert tried == ["mistral-small-2506", "glm-4.7-flash", "glm-4-7-flash"]

    @patch("scrapers.aa_resolver.get_model_params")
    def test_stops_at_first_hit(self, mock_params):
        params = {"total_params_b": 355.0, "active_params_b": 32.0, "architecture": "moe"}
        mock_params.side_effect = [None, params]
        assert resolve_single_from_aa("glm-4.6") == params
        assert mock_params.call_count == 2


class TestReverseMatching:
    def _lookup(self):
        return {
            "intellect3basedonglm45air": {"name": "INTELLECT-3 (Based on GLM-4.5-Air)",
                                          "slug": "intellect-3", "total_params_b": 107.0,
                                          "active_params_b": 12.0, "architecture": "moe"},
            "deepseekv30324": {"name": "DeepSeek V3 0324", "slug": "deepseek-v3-0324",
                               "total_params_b": 671.0, "active_params_b": 37.0,
                               "architecture": "moe"},
        }

    def test_key_starting_with_arena_name_matches(self):
        assert resolve_from_aa("deepseek-v3", aa_lookup=self._lookup())["total_params_b"] == 671.0

    def test_key_merely_containing_arena_name_does_not(self):
        assert resolve_from_aa("glm-4.5-air", aa_lookup=self._lookup()) is None


class TestModelListCache:
    @pytest.fixture
    def list_cache(self, monkeypatch, tmp_path):
        path = tmp_path / "aa_models.json"
        monkeypatch.setattr(aa_resolver, "CACHE_FILE", str(path))
        return path

    def test_saved_sorted_with_age(self, list_cache):
        lookup = _build_lookup([{"name": "Zeta 1", "slug": "zeta-1"},
                                {"name": "Alpha 2", "slug": "alpha-2"}])
        aa_resolver._save_cache(lookup)
        keys = list(json.loads(list_cache.read_text())["models"])
        assert keys == sorted(keys)
        assert aa_resolver.model_list_age_days() < 0.01

    def test_age_without_list(self, list_cache):
        assert aa_resolver.model_list_age_days() is None

    @patch("scrapers.aa_resolver._fetch_all_via_rsc", return_value=[])
    def test_failed_fetch_falls_back_to_saved_list(self, mock_fetch, list_cache):
        aa_resolver._save_cache(_build_lookup([{"name": "GLM-5.3", "slug": "glm-5-3"}]))
        lookup, stale = aa_resolver.get_aa_models(force_refresh=True)
        assert stale
        assert lookup["glm53"]["slug"] == "glm-5-3"
