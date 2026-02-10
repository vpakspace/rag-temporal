"""Dual-track ingestion: RAG vector pipeline + Knowledge Graph."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from config import RAG_ROOT, settings

logger = logging.getLogger(__name__)


def _import_rag(module_path: str):
    """Import a module from RAG 2.0 by dotted path."""
    import importlib.util
    parts = module_path.split(".")
    file_path = RAG_ROOT / "/".join(parts[:-1]) / f"{parts[-1]}.py" if len(parts) > 1 else RAG_ROOT / f"{parts[0]}.py"
    # For nested modules like ingestion.loader
    if len(parts) >= 2:
        file_path = RAG_ROOT / parts[0] / f"{parts[1]}.py"
    spec = importlib.util.spec_from_file_location(f"rag2_{module_path}", file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class IngestionResult:
    """Result of dual-track ingestion."""
    rag_chunks: int = 0
    kg_episodes: int = 0
    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class DualPipeline:
    """Dual-track ingestion: RAG chunks + KG episodes from one document."""

    def __init__(self, vector_store=None, kg_client=None):
        """Initialize with optional vector store and KG client.

        Args:
            vector_store: RAG 2.0 VectorStore (created from settings if None)
            kg_client: KGClient instance (created if None and kg_enabled)
        """
        from storage.vector_store import VectorStore

        self._vector_store = vector_store or VectorStore()
        self._vector_store.init_index()

        self._kg_client = kg_client
        self._kg_connected = False

    async def _ensure_kg(self) -> None:
        """Lazy-init KG client if needed."""
        if self._kg_client is not None and self._kg_connected:
            return
        if self._kg_client is None:
            from kg.client import KGClient
            self._kg_client = KGClient()
        if not self._kg_connected:
            await self._kg_client.connect()
            self._kg_connected = True

    def ingest_file(
        self,
        file_path: str,
        use_gpu: bool = False,
        skip_enrichment: bool = False,
        skip_kg: bool = False,
    ) -> IngestionResult:
        """Ingest a file through both RAG and KG tracks."""
        result = IngestionResult()

        rag_loader = _import_rag("ingestion.loader")
        try:
            text = rag_loader.load_file(file_path, use_gpu=use_gpu)
        except Exception as e:
            result.errors.append(f"Load error: {e}")
            return result

        if not text.strip():
            result.errors.append("Empty document")
            return result

        result.rag_chunks = self._run_rag_track(text, skip_enrichment)

        if not skip_kg and settings.kg_enabled:
            result.kg_episodes = self._run_kg_track(text, file_path)

        return result

    def ingest_text(
        self,
        text: str,
        source: str = "text",
        skip_enrichment: bool = False,
        skip_kg: bool = False,
    ) -> IngestionResult:
        """Ingest raw text through both tracks."""
        result = IngestionResult()

        if not text.strip():
            result.errors.append("Empty text")
            return result

        result.rag_chunks = self._run_rag_track(text, skip_enrichment)

        if not skip_kg and settings.kg_enabled:
            result.kg_episodes = self._run_kg_track(text, source)

        return result

    def _run_rag_track(self, text: str, skip_enrichment: bool = False) -> int:
        """RAG track: chunk -> enrich -> embed -> store."""
        try:
            rag_chunker = _import_rag("ingestion.chunker")
            rag_enricher = _import_rag("ingestion.enricher")

            chunks = rag_chunker.chunk_text(text)
            logger.info("RAG track: %d chunks created", len(chunks))

            if not skip_enrichment:
                chunks = rag_enricher.enrich_chunks(chunks)

            chunks = rag_enricher.embed_chunks(chunks)
            count = self._vector_store.add_chunks(chunks)
            logger.info("RAG track: %d chunks stored", count)
            return count
        except Exception as e:
            logger.error("RAG track failed: %s", e)
            return 0

    def _run_kg_track(self, text: str, source: str) -> int:
        """KG track: ingest via Graphiti (async)."""
        from utils.async_helpers import run_async

        return run_async(self._kg_ingest(text, source))

    async def _kg_ingest(self, text: str, source: str) -> int:
        """Async KG ingestion."""
        await self._ensure_kg()
        return await self._kg_client.ingest_text(text, source=source)

    @property
    def vector_store(self):
        """Access underlying vector store."""
        return self._vector_store

    @property
    def kg_client(self):
        """Access underlying KG client."""
        return self._kg_client

    def close(self) -> None:
        """Close all connections."""
        self._vector_store.close()
        if self._kg_client and self._kg_connected:
            try:
                asyncio.run(self._kg_client.close())
            except Exception as e:
                logger.debug("KG client close error: %s", e)
