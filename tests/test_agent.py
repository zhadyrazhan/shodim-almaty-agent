"""Tests for document construction and data loading.

These avoid building a real index (that needs embeddings and an API key) and
instead cover the parts that shape what the retriever actually sees.
"""

import json

import pytest

from src.agent import load_records, record_to_document


class TestRecordToDocument:
    def test_core_fields_are_searchable(self):
        doc = record_to_document(
            {
                "title": "Джазовый вечер",
                "kind": "event",
                "description": "Живая музыка в центре",
                "category": "концерт",
            }
        )
        assert "Джазовый вечер" in doc.text
        assert "Живая музыка" in doc.text
        assert "концерт" in doc.text

    def test_good_for_is_included(self):
        # This is what makes "место для свидания" retrieve different results
        # than "куда сводить ребенка" — it must reach the embedded text.
        doc = record_to_document(
            {
                "title": "Ресторан",
                "kind": "restaurant",
                "description": "d",
                "category": "кафе",
                "good_for": ["свидание", "семья"],
            }
        )
        assert "свидание" in doc.text
        assert "семья" in doc.text

    def test_optional_fields_are_labelled(self):
        doc = record_to_document(
            {
                "title": "T",
                "kind": "event",
                "description": "d",
                "category": "c",
                "date": "26 сентября",
                "price": "5000 тг",
                "address": "Абая 44",
            }
        )
        assert "Когда: 26 сентября" in doc.text
        assert "Цена: 5000 тг" in doc.text
        assert "Адрес: Абая 44" in doc.text

    def test_empty_fields_are_omitted(self):
        doc = record_to_document(
            {"title": "T", "kind": "event", "description": "d", "category": "c",
             "date": "", "price": "", "address": ""}
        )
        assert "Когда:" not in doc.text
        assert "Цена:" not in doc.text

    def test_metadata_carries_url_for_citation(self):
        doc = record_to_document(
            {"title": "T", "kind": "place", "description": "d", "category": "c",
             "url": "https://sxodim.com/x"}
        )
        assert doc.metadata["url"] == "https://sxodim.com/x"
        assert doc.metadata["kind"] == "place"

    def test_missing_keys_do_not_crash(self):
        doc = record_to_document({"title": "Only a title"})
        assert "Only a title" in doc.text


class TestLoadRecords:
    def test_reads_json(self, tmp_path):
        path = tmp_path / "d.json"
        path.write_text(json.dumps([{"title": "A"}]), encoding="utf-8")
        assert load_records(path) == [{"title": "A"}]

    def test_missing_file_explains_the_fix(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="scraper"):
            load_records(tmp_path / "nope.json")
