"""Tests for the Jina Reader scraper, with the network stubbed out."""

from unittest.mock import patch

import pytest

from src.scraper import fetch_markdown, scrape_all


class TestFetchMarkdown:
    def test_builds_jina_reader_url(self):
        with patch("src.scraper.urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = b"# md"
            fetch_markdown("https://sxodim.com/almaty")

        request = mock_open.call_args[0][0]
        assert request.full_url == "https://r.jina.ai/https://sxodim.com/almaty"

    def test_decodes_utf8(self):
        with patch("src.scraper.urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                "Афиша Алматы".encode("utf-8")
            )
            assert fetch_markdown("https://x") == "Афиша Алматы"


class TestScrapeAll:
    def test_writes_one_file_per_page(self, tmp_path):
        with patch("src.scraper.fetch_markdown", return_value="# content"):
            written = scrape_all({"a": "https://x/a", "b": "https://x/b"}, out_dir=tmp_path)

        assert set(written) == {"a", "b"}
        assert (tmp_path / "a.md").read_text(encoding="utf-8") == "# content"

    def test_one_failure_does_not_abort_the_rest(self, tmp_path):
        def flaky(url, **kwargs):
            if "bad" in url:
                raise RuntimeError("boom")
            return "# ok"

        with patch("src.scraper.fetch_markdown", side_effect=flaky):
            written = scrape_all({"good": "https://x/ok", "bad": "https://x/bad"}, out_dir=tmp_path)

        assert set(written) == {"good"}

    def test_raises_when_everything_fails(self, tmp_path):
        with patch("src.scraper.fetch_markdown", side_effect=RuntimeError("down")):
            with pytest.raises(RuntimeError, match="every page failed"):
                scrape_all({"a": "https://x/a"}, out_dir=tmp_path)

    def test_creates_output_dir(self, tmp_path):
        nested = tmp_path / "does" / "not" / "exist"
        with patch("src.scraper.fetch_markdown", return_value="# c"):
            scrape_all({"a": "https://x/a"}, out_dir=nested)
        assert nested.exists()
