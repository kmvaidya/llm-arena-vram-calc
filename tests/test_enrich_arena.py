from unittest.mock import patch

import pytest

from enrich_arena import resolve_from_overrides, resolve_from_name, resolve_model_params


# ── resolve_from_overrides ────────────────────────────────────────────────

class TestResolveFromOverrides:
    def test_exact_match(self):
        result = resolve_from_overrides("GPT-OSS-120B")
        assert result is not None
        assert result["total_params_b"] == 117
        assert result["architecture"] == "moe"

    def test_case_insensitive_exact(self):
        result = resolve_from_overrides("gpt-oss-120b")
        assert result is not None
        assert result["total_params_b"] == 117

    def test_exact_match_case_variant(self):
        """llama-4-maverick-17b-128e-instruct is an exact key in overrides."""
        result = resolve_from_overrides("Llama-4-Maverick-17B-128E-Instruct")
        assert result is not None
        assert result["total_params_b"] == 400

    def test_llama4_exact_match(self):
        result = resolve_from_overrides("llama-4-maverick-17b-128e-instruct")
        assert result is not None
        assert result["total_params_b"] == 400

    def test_distilled_model_skips_substring(self):
        """Models with size in name should not match parent overrides via substring."""
        result = resolve_from_overrides("Kimi-K2-8B-distilled")
        # Has "8B" in name -> has_size_in_name is True -> returns None
        assert result is None

    def test_no_match_returns_none(self):
        assert resolve_from_overrides("some-random-model") is None

    def test_returns_correct_keys(self):
        result = resolve_from_overrides("GPT-OSS-120B")
        assert set(result.keys()) == {"total_params_b", "active_params_b", "architecture"}

    def test_kimi_substring_match(self):
        """Kimi-K2.5 should match via substring for names without size indicators."""
        result = resolve_from_overrides("Kimi-K2.5-thinking")
        assert result is not None
        assert result["total_params_b"] == 1000

    def test_phi4_case_insensitive(self):
        result = resolve_from_overrides("phi-4")
        assert result is not None
        assert result["total_params_b"] == 14


# ── resolve_from_name ─────────────────────────────────────────────────────

class TestResolveFromName:
    def test_simple_dense_model(self):
        result = resolve_from_name("Llama-3.3-70B-Instruct")
        assert result is not None
        assert result["total_params_b"] == 70
        assert result["active_params_b"] == 70
        assert result["architecture"] == "dense"

    def test_moe_with_active_params(self):
        result = resolve_from_name("Qwen3-30B-A3B")
        assert result is not None
        assert result["total_params_b"] == 30
        assert result["active_params_b"] == 3
        assert result["architecture"] == "moe"

    def test_mixtral_8x7b(self):
        result = resolve_from_name("Mixtral-8x7B")
        assert result is not None
        assert result["total_params_b"] == pytest.approx(44.8, abs=0.1)
        assert result["architecture"] == "moe"

    def test_mixtral_8x22b(self):
        result = resolve_from_name("Mixtral-8x22B")
        assert result is not None
        assert result["total_params_b"] == pytest.approx(140.8, abs=0.1)

    def test_decimal_size(self):
        result = resolve_from_name("Model-3.5B-Chat")
        assert result is not None
        assert result["total_params_b"] == 3.5

    def test_no_size_in_name(self):
        assert resolve_from_name("GPT-4o") is None

    def test_model_with_trailing_suffix(self):
        result = resolve_from_name("Mistral-Small-2506-24B")
        assert result is not None
        assert result["total_params_b"] == 24

    def test_lowercase_b(self):
        result = resolve_from_name("model-7b-instruct")
        assert result is not None
        assert result["total_params_b"] == 7

    def test_no_separator_before_number(self):
        # "abc123B" should NOT match — regex requires separator or start
        assert resolve_from_name("abc123B") is None

    def test_single_digit(self):
        result = resolve_from_name("Phi-4-3B")
        assert result is not None
        assert result["total_params_b"] == 3


# ── resolve_model_params ──────────────────────────────────────────────────

class TestResolveModelParams:
    def test_override_takes_priority(self, aa_lookup):
        """Phi-4 is in both overrides (14B) and AA fixture (5.6B). Override wins."""
        result, source = resolve_model_params("Phi-4", use_network=False, aa_lookup=aa_lookup)
        assert source == "override"
        assert result["total_params_b"] == 14

    def test_aa_takes_priority_over_name(self, aa_lookup):
        result, source = resolve_model_params("deepseek-v3", use_network=False, aa_lookup=aa_lookup)
        assert source == "artificial_analysis"
        assert result["total_params_b"] == 671.0

    def test_fallback_to_name_parsing(self):
        result, source = resolve_model_params(
            "NewModel-50B-Instruct", use_network=False, aa_lookup={}
        )
        assert source == "name_parsing"
        assert result["total_params_b"] == 50

    def test_unknown_when_nothing_matches(self):
        result, source = resolve_model_params(
            "mystery-model", use_network=False, aa_lookup={}
        )
        assert source == "UNKNOWN"
        assert result is None

    def test_source_labels(self, aa_lookup):
        _, src1 = resolve_model_params("GPT-OSS-120B", use_network=False, aa_lookup=aa_lookup)
        assert src1 == "override"

        _, src2 = resolve_model_params("deepseek-v3", use_network=False, aa_lookup=aa_lookup)
        assert src2 == "artificial_analysis"

        _, src3 = resolve_model_params("Random-50B", use_network=False, aa_lookup={})
        assert src3 == "name_parsing"

        _, src4 = resolve_model_params("mystery", use_network=False, aa_lookup={})
        assert src4 == "UNKNOWN"

    @patch("enrich_arena.resolve_single_from_aa")
    def test_network_disabled_skips_scrape(self, mock_scrape, aa_lookup):
        resolve_model_params("mystery-model", use_network=False, aa_lookup={})
        mock_scrape.assert_not_called()

    @patch("enrich_arena.resolve_from_aa", side_effect=Exception("test error"))
    def test_aa_exception_falls_through(self, mock_aa, aa_lookup):
        """If AA resolution raises, it should fall through to name parsing."""
        result, source = resolve_model_params(
            "NewModel-50B", use_network=False, aa_lookup=aa_lookup
        )
        assert source == "name_parsing"
        assert result["total_params_b"] == 50
