"""Tests for hybrid retriever."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.hybrid_retriever import HybridResult, HybridRetriever


class TestHybridResult:
    def test_default_values(self):
        r = HybridResult(content="test")
        assert r.content == "test"
        assert r.source == "vector"
        assert r.score == 0.0
        assert r.metadata == {}

    def test_kg_result(self):
        r = HybridResult(content="fact", source="kg", score=0.8, metadata={"valid_at": "2024"})
        assert r.source == "kg"
        assert r.score == 0.8


class TestHybridRetriever:
    def _make_retriever(self, vector_results=None, kg_facts=None):
        """Create HybridRetriever with mocked dependencies."""
        mock_rag_retriever = MagicMock()
        mock_search_result = MagicMock()
        mock_search_result.chunk.enriched_content = "vector content"
        mock_search_result.chunk.id = "c1"
        mock_search_result.score = 0.9
        mock_search_result.rank = 1

        if vector_results is None:
            vector_results = [mock_search_result]
        mock_rag_retriever.retrieve = MagicMock(return_value=vector_results)

        mock_kg = None
        if kg_facts is not None:
            mock_kg = MagicMock()
            mock_kg.search = AsyncMock(return_value=kg_facts)

        hr = HybridRetriever(retriever=mock_rag_retriever, kg_client=mock_kg)
        return hr

    @patch("retrieval.hybrid_retriever.settings")
    def test_vector_only_mode(self, mock_settings):
        mock_settings.kg_enabled = True
        hr = self._make_retriever()
        results = hr.retrieve("test query", mode="vector", top_k=5)

        assert len(results) > 0
        assert all(r.source == "vector" for r in results)

    @patch("retrieval.hybrid_retriever.settings")
    def test_empty_results(self, mock_settings):
        mock_settings.kg_enabled = True
        hr = self._make_retriever(vector_results=[], kg_facts=None)
        results = hr.retrieve("test query", mode="vector")
        assert results == []


class TestMergeStrategy:
    def test_interleave(self):
        """Test that merge interleaves vector and KG results."""
        hr = HybridRetriever.__new__(HybridRetriever)
        hr._retriever = MagicMock()
        hr._kg_client = None

        v = [HybridResult(content=f"v{i}", source="vector", score=0.9 - i * 0.1) for i in range(3)]
        k = [HybridResult(content=f"k{i}", source="kg", score=0.8) for i in range(3)]

        merged = hr._merge(v, k, top_k=10)

        assert len(merged) == 6
        # Interleaved: v0, k0, v1, k1, v2, k2
        assert merged[0].source == "vector"
        assert merged[1].source == "kg"

    def test_deduplication(self):
        """Test that duplicate content is removed."""
        hr = HybridRetriever.__new__(HybridRetriever)
        hr._retriever = MagicMock()
        hr._kg_client = None

        v = [HybridResult(content="same content here", source="vector", score=0.9)]
        k = [HybridResult(content="same content here", source="kg", score=0.8)]

        merged = hr._merge(v, k, top_k=10)
        assert len(merged) == 1  # Deduped

    def test_top_k_limit(self):
        """Test that merge respects top_k."""
        hr = HybridRetriever.__new__(HybridRetriever)
        hr._retriever = MagicMock()
        hr._kg_client = None

        v = [HybridResult(content=f"v{i}", source="vector", score=0.9) for i in range(10)]
        k = [HybridResult(content=f"k{i}", source="kg", score=0.8) for i in range(10)]

        merged = hr._merge(v, k, top_k=5)
        assert len(merged) == 5

    def test_score_normalization(self):
        """Test vector score normalization to [0, 1]."""
        hr = HybridRetriever.__new__(HybridRetriever)
        hr._retriever = MagicMock()
        hr._kg_client = None

        v = [
            HybridResult(content="v1", source="vector", score=0.5),
            HybridResult(content="v2", source="vector", score=1.0),
        ]
        k = []

        merged = hr._merge(v, k, top_k=10)
        assert merged[0].score == 0.5  # 0.5 / 1.0
        assert merged[1].score == 1.0  # 1.0 / 1.0
