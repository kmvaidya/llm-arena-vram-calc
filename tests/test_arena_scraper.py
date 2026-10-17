import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests
from scrapers.arena_scraper import (
    EXPECTED_COLUMNS,
    _fetch_hf_leaderboard,
    scrape_arena_leaderboard,
)


def _row(name, rating, license="MIT", category="overall", votes=1000.0):
    """One row shaped like lmarena-ai/leaderboard-dataset."""
    return {
        "model_name": name, "organization": "acme", "license": license,
        "rating": rating, "rating_lower": rating - 5, "rating_upper": rating + 5,
        "variance": 4.0, "vote_count": votes, "rank": 1, "category": category,
        "leaderboard_publish_date": "2026-10-15",
    }


def _parquet(rows):
    buf = io.BytesIO()
    pd.DataFrame(rows).to_parquet(buf, index=False)
    return buf.getvalue()


def _response(content=b"", status_code=200):
    return MagicMock(status_code=status_code, content=content)


SAMPLE_ROWS = [
    _row("closed-model", 1500, license="Proprietary"),
    _row("open-second", 1400, votes=2000.0),
    _row("open-first", 1450, license="Apache 2.0"),
    _row("open-first", 1600, category="coding"),
    _row("unlicensed-model", 1550, license=None),
]


# ── _fetch_hf_leaderboard ─────────────────────────────────────────────────

class TestFetchHfLeaderboard:
    @patch("scrapers.arena_scraper.requests.get")
    def test_keeps_open_source_overall_rows(self, mock_get):
        mock_get.return_value = _response(_parquet(SAMPLE_ROWS))
        df = _fetch_hf_leaderboard()
        assert df is not None
        assert list(df["model_name"]) == ["open-first", "open-second"]

    @patch("scrapers.arena_scraper.requests.get")
    def test_ranks_sequentially_by_rating(self, mock_get):
        mock_get.return_value = _response(_parquet(SAMPLE_ROWS))
        df = _fetch_hf_leaderboard()
        assert list(df["rank"]) == [1, 2]
        assert list(df["arena_score"]) == [1450, 1400]

    @patch("scrapers.arena_scraper.requests.get")
    def test_maps_columns(self, mock_get):
        mock_get.return_value = _response(_parquet(SAMPLE_ROWS))
        df = _fetch_hf_leaderboard()
        assert list(df.columns) == EXPECTED_COLUMNS
        second = df.iloc[1]
        assert second["ci_lower"] == 1395
        assert second["ci_upper"] == 1405
        assert second["votes"] == 2000
        assert str(df["votes"].dtype) == "Int64"
        assert df.iloc[0]["license"] == "Apache 2.0"
        assert df.iloc[0]["organization"] == "acme"

    @patch("scrapers.arena_scraper.requests.get")
    def test_records_publication_date(self, mock_get):
        mock_get.return_value = _response(_parquet(SAMPLE_ROWS))
        df = _fetch_hf_leaderboard()
        assert df.attrs["leaderboard_date"] == "2026-10-15"

    @patch("scrapers.arena_scraper.requests.get")
    def test_http_error_returns_none(self, mock_get):
        mock_get.return_value = _response(status_code=403)
        assert _fetch_hf_leaderboard() is None

    @patch("scrapers.arena_scraper.requests.get",
           side_effect=requests.ConnectionError("network down"))
    def test_request_exception_returns_none(self, mock_get):
        assert _fetch_hf_leaderboard() is None

    @patch("scrapers.arena_scraper.requests.get")
    def test_unreadable_file_returns_none(self, mock_get):
        mock_get.return_value = _response(b"<html>not a parquet file</html>")
        assert _fetch_hf_leaderboard() is None

    @patch("scrapers.arena_scraper.requests.get")
    def test_missing_column_returns_none(self, mock_get):
        rows = [{k: v for k, v in r.items() if k != "rating"} for r in SAMPLE_ROWS]
        mock_get.return_value = _response(_parquet(rows))
        assert _fetch_hf_leaderboard() is None

    @patch("scrapers.arena_scraper.requests.get")
    def test_no_open_source_returns_none(self, mock_get):
        mock_get.return_value = _response(_parquet([_row("closed", 1500, license="Proprietary")]))
        assert _fetch_hf_leaderboard() is None


# ── scrape_arena_leaderboard ──────────────────────────────────────────────

class TestScrapeArenaLeaderboard:
    @patch("scrapers.arena_scraper.requests.get")
    def test_raises_when_download_fails(self, mock_get):
        mock_get.return_value = _response(status_code=503)
        with pytest.raises(RuntimeError, match="--input"):
            scrape_arena_leaderboard()

    @patch("scrapers.arena_scraper.requests.get")
    def test_csv_input_skips_download(self, mock_get, tmp_path):
        csv = tmp_path / "arena.csv"
        csv.write_text("Rank,Model,Score,Votes,Organization,License\n1,m-7b,1300,500,Org,MIT\n")
        df = scrape_arena_leaderboard(input_csv=str(csv))
        mock_get.assert_not_called()
        assert list(df["model_name"]) == ["m-7b"]
