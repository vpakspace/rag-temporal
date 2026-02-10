"""Tests for dual-track ingestion pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.dual_pipeline import DualPipeline, IngestionResult


class TestIngestionResult:
    def test_default_values(self):
        r = IngestionResult()
        assert r.rag_chunks == 0
        assert r.kg_episodes == 0
        assert r.errors == []

    def test_custom_values(self):
        r = IngestionResult(rag_chunks=5, kg_episodes=2, errors=["test"])
        assert r.rag_chunks == 5
        assert r.kg_episodes == 2
        assert r.errors == ["test"]

    def test_independent_error_lists(self):
        r1 = IngestionResult()
        r2 = IngestionResult()
        r1.errors.append("err1")
        assert r2.errors == []  # Should not share reference


class TestDualPipelineRAGOnly:
    """Test RAG-only track with skip_kg=True."""

    def test_ingest_empty_text(self):
        mock_store = MagicMock()
        mock_store.init_index = MagicMock()

        with patch("storage.vector_store.VectorStore", return_value=mock_store):
            pipeline = DualPipeline(vector_store=mock_store)
            result = pipeline.ingest_text("")
            assert result.rag_chunks == 0
            assert "Empty text" in result.errors

    def test_ingest_whitespace_text(self):
        mock_store = MagicMock()
        mock_store.init_index = MagicMock()

        with patch("storage.vector_store.VectorStore", return_value=mock_store):
            pipeline = DualPipeline(vector_store=mock_store)
            result = pipeline.ingest_text("   \n  ")
            assert result.rag_chunks == 0
            assert "Empty text" in result.errors

    def test_ingest_file_not_found(self):
        mock_store = MagicMock()
        mock_store.init_index = MagicMock()
        mock_loader = MagicMock()
        mock_loader.load_file.side_effect = FileNotFoundError("No such file")

        with patch("storage.vector_store.VectorStore", return_value=mock_store), \
             patch("ingestion.dual_pipeline._import_rag", return_value=mock_loader):
            pipeline = DualPipeline(vector_store=mock_store)
            result = pipeline.ingest_file("/nonexistent/file.txt")
            assert result.rag_chunks == 0
            assert len(result.errors) > 0
            assert "Load error" in result.errors[0]


class TestDualPipelineProperties:
    def test_vector_store_property(self):
        mock_store = MagicMock()
        mock_store.init_index = MagicMock()

        with patch("storage.vector_store.VectorStore", return_value=mock_store):
            pipeline = DualPipeline(vector_store=mock_store)
            assert pipeline.vector_store is mock_store

    def test_kg_client_property_none(self):
        mock_store = MagicMock()
        mock_store.init_index = MagicMock()

        with patch("storage.vector_store.VectorStore", return_value=mock_store):
            pipeline = DualPipeline(vector_store=mock_store)
            assert pipeline.kg_client is None

    def test_kg_client_property_set(self):
        mock_store = MagicMock()
        mock_store.init_index = MagicMock()
        mock_kg = MagicMock()

        with patch("storage.vector_store.VectorStore", return_value=mock_store):
            pipeline = DualPipeline(vector_store=mock_store, kg_client=mock_kg)
            assert pipeline.kg_client is mock_kg

    def test_close(self):
        mock_store = MagicMock()
        mock_store.init_index = MagicMock()

        with patch("storage.vector_store.VectorStore", return_value=mock_store):
            pipeline = DualPipeline(vector_store=mock_store)
            pipeline.close()
            mock_store.close.assert_called_once()
