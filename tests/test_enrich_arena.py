from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from enrich_arena import (
    check_health,
    resolve_from_name,
    resolve_from_overrides,
    resolve_model_params,
)

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

    @patch("enrich_arena.resolve_from_aa")
    def test_implausible_result_falls_through(self, mock_aa):
        """A 1.6T model read as 1.6B has more active than total parameters."""
        mock_aa.return_value = {"total_params_b": 1.6, "active_params_b": 49.0,
                                "architecture": "dense"}
        result, source = resolve_model_params(
            "NewModel-50B", use_network=False, aa_lookup={"x": {}}
        )
        assert source == "name_parsing"
        assert result["total_params_b"] == 50

    @patch("enrich_arena.resolve_from_aa")
    def test_implausible_result_with_no_fallback_is_unknown(self, mock_aa):
        mock_aa.return_value = {"total_params_b": 0, "active_params_b": 0,
                                "architecture": "dense"}
        result, source = resolve_model_params(
            "mystery-model", use_network=False, aa_lookup={"x": {}}
        )
        assert (result, source) == (None, "UNKNOWN")

    @patch("enrich_arena.resolve_from_aa")
    def test_aa_reads_pages_only_with_network(self, mock_aa):
        mock_aa.return_value = None
        resolve_model_params("m", use_network=False, aa_lookup={"x": {}})
        assert mock_aa.call_args.kwargs["fetch_params"] is False
        with patch("enrich_arena.resolve_single_from_aa", return_value=None):
            resolve_model_params("m", use_network=True, aa_lookup={"x": {}})
        assert mock_aa.call_args.kwargs["fetch_params"] is True


# ── check_health ──────────────────────────────────────────────────────────

def _published(days_ago):
    return str((datetime.now(UTC) - timedelta(days=days_ago)).date())


class TestCheckHealth:
    def test_healthy_run(self):
        assert check_health(226, 211, _published(2)) == []

    def test_stalled_leaderboard(self):
        assert check_health(226, 211, _published(19)) == []
        [problem] = check_health(226, 211, _published(22))
        assert "last published 22 days ago" in problem

    def test_too_few_sizes(self):
        assert check_health(100, 85, None) == []
        [problem] = check_health(100, 84, None)
        assert "Only 84 of 100 models (84%)" in problem

    def test_model_list_unreadable(self):
        [old_copy] = check_health(226, 211, None, aa_list_problem=True, aa_list_age_days=4.2)
        assert "the last good copy is 4 days old" in old_copy
        [no_copy] = check_health(226, 211, None, aa_list_problem=True)
        assert "there is no saved copy" in no_copy

    def test_pages_lost_sizes(self):
        [problem] = check_health(226, 211, None, lost_sizes=["glm-5-3", "deepseek-v4-pro"])
        assert "2 Artificial Analysis page(s)" in problem
        assert "(deepseek-v4-pro, glm-5-3)" in problem


@patch("enrich_arena.resolve_from_name", side_effect=KeyError("bug"))
def test_crashing_strategy_logged_and_skipped(mock_name, caplog):
    params, source = resolve_model_params("some-model-7b", use_network=False)
    assert (params, source) == (None, "UNKNOWN")
    assert "name_parsing resolution crashed for some-model-7b" in caplog.text
