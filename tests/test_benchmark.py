"""Tests for KG client, config, and i18n utilities."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Settings
from kg.client import KGFact, sanitize_lucene, split_episodes
from ui.i18n import TRANSLATIONS, get_translator


# --- Config tests ---

class TestConfig:
    def test_default_settings(self):
        s = Settings(openai_api_key="test-key")
        assert s.embedding_model == "text-embedding-3-small"
        assert s.llm_model == "gpt-4o-mini"
        assert s.neo4j_uri == "bolt://localhost:7687"
        assert s.kg_enabled is True
        assert s.kg_max_episode_chars == 8000
        assert s.chunk_size == 800

    def test_custom_settings(self):
        s = Settings(
            openai_api_key="test",
            chunk_size=500,
            kg_enabled=False,
            kg_max_episode_chars=5000,
        )
        assert s.chunk_size == 500
        assert s.kg_enabled is False
        assert s.kg_max_episode_chars == 5000


# --- KG Client utility tests ---

class TestSanitizeLucene:
    def test_special_chars_removed(self):
        result = sanitize_lucene("test+query*with:special?chars")
        assert "+" not in result
        assert "*" not in result
        assert ":" not in result
        assert "?" not in result

    def test_plain_text_unchanged(self):
        assert sanitize_lucene("hello world") == "hello world"

    def test_brackets_removed(self):
        result = sanitize_lucene("test[0](1){2}")
        assert "[" not in result
        assert "]" not in result

    def test_empty_string(self):
        assert sanitize_lucene("") == ""


class TestSplitEpisodes:
    def test_short_text_single_episode(self):
        text = "Short text"
        episodes = split_episodes(text, max_chars=100)
        assert len(episodes) == 1
        assert episodes[0] == text

    def test_long_text_splits(self):
        text = "\n\n".join([f"Paragraph {i}" * 20 for i in range(20)])
        episodes = split_episodes(text, max_chars=500, overlap=50)
        assert len(episodes) > 1

    def test_respects_max_chars(self):
        text = "\n\n".join(["x" * 100 for _ in range(10)])
        episodes = split_episodes(text, max_chars=300, overlap=0)
        for ep in episodes:
            assert len(ep) <= 400  # Some tolerance for paragraph boundaries

    def test_single_huge_paragraph(self):
        text = "x" * 2000
        episodes = split_episodes(text, max_chars=500, overlap=0)
        assert len(episodes) >= 4

    def test_overlap(self):
        text = "A" * 500 + "\n\n" + "B" * 500
        episodes = split_episodes(text, max_chars=600, overlap=100)
        assert len(episodes) == 2
        # Second episode should start with overlap from first
        assert episodes[1].startswith("A" * 100)


class TestKGFact:
    def test_default_values(self):
        f = KGFact(content="test fact")
        assert f.content == "test fact"
        assert f.source == "kg"
        assert f.score == 0.8
        assert f.valid_at == ""
        assert f.metadata == {}

    def test_custom_values(self):
        f = KGFact(
            content="fact",
            valid_at="2024-01",
            entity_name="LMCache",
            score=0.9,
            metadata={"type": "entity"},
        )
        assert f.valid_at == "2024-01"
        assert f.entity_name == "LMCache"


# --- i18n tests ---

class TestI18n:
    def test_english_translation(self):
        t = get_translator("en")
        assert t("app_title") == "RAG + Temporal Knowledge Base"

    def test_russian_translation(self):
        t = get_translator("ru")
        assert t("app_title") == "RAG + Темпоральная база знаний"

    def test_missing_key_returns_key(self):
        t = get_translator("en")
        assert t("nonexistent_key_xyz") == "nonexistent_key_xyz"

    def test_all_keys_have_both_languages(self):
        for key, translations in TRANSLATIONS.items():
            assert "en" in translations, f"Key '{key}' missing English translation"
            assert "ru" in translations, f"Key '{key}' missing Russian translation"

    def test_translations_not_empty(self):
        for key, translations in TRANSLATIONS.items():
            assert translations["en"].strip(), f"Key '{key}' has empty English translation"
            assert translations["ru"].strip(), f"Key '{key}' has empty Russian translation"

    def test_all_tab_keys_exist(self):
        tab_keys = ["tab_ingest", "tab_search", "tab_kg", "tab_benchmark", "tab_settings"]
        for key in tab_keys:
            assert key in TRANSLATIONS, f"Tab key '{key}' not found"

    def test_format_kwargs(self):
        """Test that format kwargs work (even if not currently used)."""
        t = get_translator("en")
        # Just ensure it doesn't crash with unknown kwargs
        result = t("app_title", extra="value")
        assert isinstance(result, str)
