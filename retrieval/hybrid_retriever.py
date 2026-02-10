"""Hybrid retriever: vector search + knowledge graph search with merge."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """A single result from hybrid retrieval."""
    content: str
    source: Literal["vector", "kg"] = "vector"
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


class HybridRetriever:
    """Combines vector search (RAG 2.0) and knowledge graph search (Graphiti)."""

    def __init__(self, retriever=None, kg_client=None, vector_store=None):
        """Initialize with RAG retriever and KG client.

        Args:
            retriever: RAG 2.0 Retriever instance
            kg_client: KGClient instance
            vector_store: VectorStore (used if retriever is None)
        """
        if retriever is None:
            from retrieval.retriever import Retriever
            from storage.vector_store import VectorStore

            store = vector_store or VectorStore()
            retriever = Retriever(store)

        self._retriever = retriever
        self._kg_client = kg_client

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 10,
    ) -> list[HybridResult]:
        """Retrieve results from vector store, KG, or both.

        Args:
            query: Search query
            mode: "hybrid" | "vector" | "kg"
            top_k: Total results to return

        Returns:
            Merged and sorted HybridResult list
        """
        vector_results: list[HybridResult] = []
        kg_results: list[HybridResult] = []

        half_k = max(top_k // 2, 3)

        if mode in ("hybrid", "vector"):
            vector_results = self._vector_search(query, top_k=half_k if mode == "hybrid" else top_k)

        if mode in ("hybrid", "kg") and self._kg_client and settings.kg_enabled:
            kg_results = self._kg_search(query, num_results=half_k if mode == "hybrid" else top_k)

        if mode == "hybrid":
            merged = self._merge(vector_results, kg_results, top_k)
        elif mode == "vector":
            merged = vector_results[:top_k]
        else:
            merged = kg_results[:top_k]

        logger.info(
            "Hybrid retrieve (%s): %d vector + %d kg -> %d merged",
            mode, len(vector_results), len(kg_results), len(merged),
        )
        return merged

    def _vector_search(self, query: str, top_k: int = 5) -> list[HybridResult]:
        """RAG 2.0 vector search."""
        try:
            results = self._retriever.retrieve(query, top_k=top_k)
            return [
                HybridResult(
                    content=r.chunk.enriched_content,
                    source="vector",
                    score=r.score,
                    metadata={
                        "chunk_id": r.chunk.id,
                        "rank": r.rank,
                    },
                )
                for r in results
            ]
        except Exception as e:
            logger.error("Vector search failed: %s", e)
            return []

    def _kg_search(self, query: str, num_results: int = 5) -> list[HybridResult]:
        """Knowledge graph search via Graphiti."""
        import asyncio

        try:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, self._kg_client.search(query, num_results))
                        facts = future.result()
                else:
                    facts = asyncio.run(self._kg_client.search(query, num_results))
            except RuntimeError:
                facts = asyncio.run(self._kg_client.search(query, num_results))

            return [
                HybridResult(
                    content=f.content,
                    source="kg",
                    score=f.score,
                    metadata={
                        "valid_at": f.valid_at,
                        "entity_name": f.entity_name,
                    },
                )
                for f in facts
                if f.content.strip()
            ]
        except Exception as e:
            logger.error("KG search failed: %s", e)
            return []

    def _merge(
        self,
        vector_results: list[HybridResult],
        kg_results: list[HybridResult],
        top_k: int,
    ) -> list[HybridResult]:
        """Merge and interleave vector + KG results.

        Strategy:
        - Normalize vector scores to [0, 1]
        - KG facts use fixed score (0.8)
        - Interleave: alternate vector and KG results
        - Deduplicate by content similarity (exact substring match)
        - Return top_k
        """
        # Normalize vector scores
        if vector_results:
            max_score = max(r.score for r in vector_results) or 1.0
            for r in vector_results:
                r.score = r.score / max_score

        # Interleave
        merged: list[HybridResult] = []
        vi, ki = 0, 0
        while vi < len(vector_results) or ki < len(kg_results):
            if vi < len(vector_results):
                merged.append(vector_results[vi])
                vi += 1
            if ki < len(kg_results):
                merged.append(kg_results[ki])
                ki += 1

        # Deduplicate by content (remove near-duplicates)
        seen_contents: list[str] = []
        deduped: list[HybridResult] = []
        for r in merged:
            content_lower = r.content.lower().strip()[:200]
            is_dup = False
            for seen in seen_contents:
                if content_lower in seen or seen in content_lower:
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(r)
                seen_contents.append(content_lower)

        return deduped[:top_k]

    @property
    def retriever(self):
        """Access underlying RAG retriever."""
        return self._retriever

    @property
    def kg_client(self):
        """Access underlying KG client."""
        return self._kg_client
