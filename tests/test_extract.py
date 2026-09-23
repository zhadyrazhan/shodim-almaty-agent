"""Tests for chunking and deduplication — the pure logic, no API calls."""

import pytest

from src.extract import Item, chunk_markdown, dedupe


def make_item(title: str, url: str = "", kind: str = "event") -> Item:
    return Item(title=title, kind=kind, description="d", category="c", url=url)


class TestChunkMarkdown:
    def test_short_text_is_one_chunk(self):
        assert len(chunk_markdown("hello world", size=1000)) == 1

    def test_splits_on_blank_lines_when_over_size(self):
        text = "\n\n".join(["x" * 400 for _ in range(5)])
        chunks = chunk_markdown(text, size=1000)
        assert len(chunks) > 1

    def test_no_content_is_lost(self):
        blocks = [f"block{i} " + "y" * 300 for i in range(8)]
        chunks = chunk_markdown("\n\n".join(blocks), size=700)
        joined = " ".join(chunks)
        for i in range(8):
            assert f"block{i}" in joined

    def test_empty_input_yields_no_chunks(self):
        assert chunk_markdown("") == []
        assert chunk_markdown("   \n\n  ") == []

    def test_single_block_larger_than_size_is_kept_whole(self):
        # We split on paragraph boundaries only, so an oversized paragraph
        # stays intact rather than being cut mid-sentence.
        big = "z" * 5000
        assert chunk_markdown(big, size=1000) == [big]


class TestDedupe:
    def test_drops_duplicate_urls(self):
        items = [
            make_item("Concert", "https://sxodim.com/a"),
            make_item("Concert (repeat listing)", "https://sxodim.com/a"),
        ]
        assert len(dedupe(items)) == 1

    def test_falls_back_to_title_when_url_missing(self):
        items = [make_item("Same Event"), make_item("Same Event")]
        assert len(dedupe(items)) == 1

    def test_keeps_distinct_items(self):
        items = [
            make_item("A", "https://sxodim.com/a"),
            make_item("B", "https://sxodim.com/b"),
        ]
        assert len(dedupe(items)) == 2

    def test_is_case_insensitive(self):
        items = [make_item("Jazz Night"), make_item("JAZZ NIGHT")]
        assert len(dedupe(items)) == 1

    def test_preserves_first_occurrence_order(self):
        items = [make_item("First"), make_item("Second"), make_item("First")]
        assert [i.title for i in dedupe(items)] == ["First", "Second"]

    def test_empty_list(self):
        assert dedupe([]) == []


class TestItemSchema:
    def test_optional_fields_default_to_empty(self):
        item = Item(title="t", kind="place", description="d", category="c")
        assert item.url == ""
        assert item.good_for == []

    def test_kind_is_constrained(self):
        with pytest.raises(Exception):
            Item(title="t", kind="not_a_kind", description="d", category="c")
