
import pytest
from scrapers.aa_resolver import (
    _build_lookup,
    _extract_models_from_rsc,
    _normalize,
    _parse_param_value,
    resolve_from_aa,
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

    def test_no_number(self):
        assert _parse_param_value("N/A") is None

    def test_empty_string(self):
        assert _parse_param_value("") is None


# ── _build_lookup ─────────────────────────────────────────────────────────

class TestBuildLookup:
    def _make_raw_model(self, name, slug="", params=None, active=None, **extra):
        m = {"name": name, "slug": slug, "is_open_weights": True,
             "context_window_tokens": 4096}
        if params is not None:
            m["parameters"] = params
        if active is not None:
            m["inference_parameters_active_billions"] = active
        m.update(extra)
        return m

    def test_basic_indexing(self):
        raw = [self._make_raw_model("Llama 3 70B", params=70.6, active=70.6)]
        lookup = _build_lookup(raw)
        assert "llama370b" in lookup
        assert lookup["llama370b"]["total_params_b"] == 70.6

    def test_slug_also_indexed(self):
        raw = [self._make_raw_model("X", slug="llama-3-70b", params=70.6, active=70.6)]
        lookup = _build_lookup(raw)
        assert "llama370b" in lookup

    def test_moe_detection(self):
        raw = [self._make_raw_model("MoE", params=600, active=30)]
        lookup = _build_lookup(raw)
        assert lookup["moe"]["architecture"] == "moe"

    def test_dense_detection(self):
        raw = [self._make_raw_model("Dense", params=70, active=70)]
        lookup = _build_lookup(raw)
        assert lookup["dense"]["architecture"] == "dense"

    def test_null_params(self):
        raw = [self._make_raw_model("NoParams")]
        lookup = _build_lookup(raw)
        assert lookup["noparams"]["total_params_b"] is None

    def test_prefers_entry_with_params(self):
        # Same normalized key, entry with params should win
        raw = [
            self._make_raw_model("Model"),
            self._make_raw_model("Model", params=10, active=10),
        ]
        lookup = _build_lookup(raw)
        assert lookup["model"]["total_params_b"] == 10.0

    def test_no_family_slug_indexing(self):
        """Family slug must NOT be used as a lookup key (caused false matches)."""
        raw = [self._make_raw_model(
            "Qwen1.5 110B", slug="qwen-1.5-110b", params=110, active=110,
            model_family_slug="qwen15",
        )]
        lookup = _build_lookup(raw)
        # The family slug key should NOT exist
        assert "qwen15" not in lookup
        # But name and slug keys should
        assert "qwen15110b" in lookup

    def test_rounding(self):
        raw = [self._make_raw_model("X", params=70.64, active=70.64)]
        lookup = _build_lookup(raw)
        assert lookup["x"]["total_params_b"] == 70.6


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
        assert models[1]["parameters"] == 600.0

    def test_no_marker(self):
        assert _extract_models_from_rsc("no relevant data here") == []

    def test_malformed_json(self):
        text = 'prefix[{"inference_parameters_active_billions": broken'
        assert _extract_models_from_rsc(text) == []

    def test_no_array_start(self):
        # Marker exists but no '[{' before it
        text = '"inference_parameters_active_billions": 70.0'
        assert _extract_models_from_rsc(text) == []
