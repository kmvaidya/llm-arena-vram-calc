import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
import requests
from scrapers import hf_checkpoints
from scrapers.hf_checkpoints import HFUnavailable, _fetch_checkpoint_bytes, checkpoint_bytes


def _model_info(*files):
    """A Hugging Face model-info response listing (filename, size) files."""
    return MagicMock(status_code=200, json=lambda: {
        "siblings": [{"rfilename": name, "size": size} for name, size in files]})


def _days_ago(days):
    return (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%d")


@pytest.fixture
def cache(monkeypatch, tmp_path):
    """Point the committed cache at a temporary file."""
    path = tmp_path / "hf_checkpoints.json"
    monkeypatch.setattr(hf_checkpoints, "CACHE_FILE", str(path))
    return path


class TestFetchCheckpointBytes:
    @patch("scrapers.hf_checkpoints.requests.get")
    def test_sums_top_level_safetensors(self, mock_get):
        # gpt-oss-120b keeps a second copy of its weights in original/
        mock_get.return_value = _model_info(
            ("model-00001-of-00002.safetensors", 40_000_000_000),
            ("model-00002-of-00002.safetensors", 25_200_000_000),
            ("original/model--00001-of-00007.safetensors", 60_000_000_000),
            ("config.json", 1_000),
        )
        assert _fetch_checkpoint_bytes("openai/gpt-oss-120b") == 65_200_000_000
        assert mock_get.call_args.kwargs["params"] == {"blobs": "true"}

    @patch("scrapers.hf_checkpoints.requests.get")
    def test_mistral_layout_counted_once(self, mock_get):
        # The same weights for mistral-inference and for transformers
        mock_get.return_value = _model_info(
            ("consolidated-00001-of-00002.safetensors", 120_000_000_000),
            ("consolidated-00002-of-00002.safetensors", 125_000_000_000),
            ("model-00001-of-00003.safetensors", 100_000_000_000),
            ("model-00002-of-00003.safetensors", 100_000_000_000),
            ("model-00003-of-00003.safetensors", 45_000_000_000),
        )
        assert _fetch_checkpoint_bytes("mistralai/Mistral-Large-Instruct-2407") == 245_000_000_000

    @patch("scrapers.hf_checkpoints.requests.get")
    def test_unreadable_answer_is_unavailable(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, json=MagicMock(side_effect=ValueError("html")))
        with pytest.raises(HFUnavailable):
            _fetch_checkpoint_bytes("org/model")

    @patch("scrapers.hf_checkpoints.requests.get", side_effect=requests.Timeout("slow"))
    def test_stops_asking_after_failures_in_a_row(self, mock_get):
        for _ in range(hf_checkpoints.MAX_FAILURES_IN_A_ROW + 3):
            with pytest.raises(HFUnavailable):
                _fetch_checkpoint_bytes("org/model")
        assert mock_get.call_count == hf_checkpoints.MAX_FAILURES_IN_A_ROW

    @patch("scrapers.hf_checkpoints.requests.get")
    def test_repository_without_safetensors(self, mock_get):
        mock_get.return_value = _model_info(("pytorch_model.bin", 14_000_000_000))
        assert _fetch_checkpoint_bytes("old/model") is None

    @pytest.mark.parametrize("status", [401, 404])
    @patch("scrapers.hf_checkpoints.requests.get")
    def test_missing_repository(self, mock_get, status):
        # Hugging Face answers 401 for a repository that does not exist
        mock_get.return_value = MagicMock(status_code=status)
        assert _fetch_checkpoint_bytes("no/such-model") is None

    @pytest.mark.parametrize("status", [429, 500, 503])
    @patch("scrapers.hf_checkpoints.requests.get")
    def test_server_trouble_raises(self, mock_get, status):
        mock_get.return_value = MagicMock(status_code=status)
        with pytest.raises(HFUnavailable):
            _fetch_checkpoint_bytes("org/model")

    @patch("scrapers.hf_checkpoints.requests.get", side_effect=requests.ConnectionError("reset"))
    def test_network_error_raises(self, mock_get):
        with pytest.raises(HFUnavailable):
            _fetch_checkpoint_bytes("org/model")


class TestCheckpointBytes:
    def _write(self, path, repo, checked, size):
        path.write_text(json.dumps({repo: {"bytes": size, "checked": checked}}))

    @patch("scrapers.hf_checkpoints._fetch_checkpoint_bytes", return_value=595_200_000_000)
    def test_fetches_and_caches(self, mock_fetch, cache):
        assert checkpoint_bytes("moonshotai/Kimi-K2.6") == 595_200_000_000
        saved = json.loads(cache.read_text())["moonshotai/Kimi-K2.6"]
        assert saved == {"bytes": 595_200_000_000, "checked": _days_ago(0)}

    @patch("scrapers.hf_checkpoints._fetch_checkpoint_bytes")
    def test_fresh_hit_skips_fetch(self, mock_fetch, cache):
        self._write(cache, "r", _days_ago(29), 10)
        assert checkpoint_bytes("r") == 10
        mock_fetch.assert_not_called()

    @patch("scrapers.hf_checkpoints._fetch_checkpoint_bytes", return_value=20)
    def test_expired_hit_refetched(self, mock_fetch, cache):
        self._write(cache, "r", _days_ago(41), 10)
        assert checkpoint_bytes("r") == 20

    @patch("scrapers.hf_checkpoints._fetch_checkpoint_bytes")
    def test_no_fetch_reads_only_the_cache(self, mock_fetch, cache):
        self._write(cache, "r", _days_ago(90), 10)
        assert checkpoint_bytes("r", fetch=False) == 10
        assert checkpoint_bytes("other", fetch=False) is None
        mock_fetch.assert_not_called()

    @patch("scrapers.hf_checkpoints._fetch_checkpoint_bytes", side_effect=HFUnavailable("down"))
    def test_outage_keeps_expired_size(self, mock_fetch, cache):
        self._write(cache, "r", _days_ago(90), 10)
        assert checkpoint_bytes("r") == 10

    @patch("scrapers.hf_checkpoints._fetch_checkpoint_bytes", side_effect=HFUnavailable("down"))
    def test_outage_without_size_raises_and_caches_nothing(self, mock_fetch, cache):
        with pytest.raises(HFUnavailable):
            checkpoint_bytes("r")
        assert not cache.exists()

    @patch("scrapers.hf_checkpoints._fetch_checkpoint_bytes", return_value=None)
    def test_size_that_disappears_is_kept(self, mock_fetch, cache):
        self._write(cache, "r", _days_ago(41), 10)
        assert checkpoint_bytes("r") == 10

    def test_corrupt_cache_raises(self, cache):
        cache.write_text('{"r": {"bytes": 1')
        with pytest.raises(json.JSONDecodeError):
            hf_checkpoints.validate_cache()
