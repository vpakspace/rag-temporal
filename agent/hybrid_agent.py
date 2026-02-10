"""Hybrid agent with 5 tools: 3 RAG + 2 KG."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from config import settings
from core.models import QAResult, SearchResult

if TYPE_CHECKING:
    from openai import OpenAI

    from kg.client import KGClient
    from retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """State passed through agent nodes."""
    query: str = ""
    query_type: str = ""  # factual, overview, comparison, temporal, entity
    tool: str = ""
    results: list[dict] = field(default_factory=list)
    search_results: list[SearchResult] = field(default_factory=list)
    relevance_score: float = 0.0
    retries: int = 0
    answer: str = ""
    qa_result: QAResult | None = None


class HybridAgent:
    """Agent with 5 tools routing queries to optimal search strategy.

    Tools:
    1. vector_search — quick semantic search (RAG 2.0)
    2. focused_search — deep search with expansion + reranking (RAG 2.0)
    3. full_document_read — all chunks concatenated (RAG 2.0)
    4. knowledge_search — entity/fact search (Graphiti KG)
    5. temporal_query — bi-temporal Cypher queries (Neo4j)
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        kg_client: KGClient | None = None,
        openai_client: OpenAI | None = None,
    ):
        self._retriever = hybrid_retriever
        self._kg_client = kg_client or hybrid_retriever.kg_client

        if openai_client is None:
            from openai import OpenAI
            self._openai = OpenAI(api_key=settings.openai_api_key)
        else:
            self._openai = openai_client

    def classify_query(self, state: AgentState) -> AgentState:
        """Classify query type to determine search strategy."""
        response = self._openai.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the query type. Respond with ONLY one word:\n"
                        "- 'factual' for specific fact questions\n"
                        "- 'overview' for broad/summary questions\n"
                        "- 'comparison' for comparing things\n"
                        "- 'temporal' for time-based questions (when, timeline, changes over time)\n"
                        "- 'entity' for questions about specific entities, people, or concepts"
                    ),
                },
                {"role": "user", "content": state.query},
            ],
            temperature=0.0,
        )
        query_type = (response.choices[0].message.content or "factual").strip().lower()
        if query_type not in ("factual", "overview", "comparison", "temporal", "entity"):
            query_type = "factual"
        state.query_type = query_type
        logger.info("Query classified as: %s", query_type)
        return state

    def select_tool(self, state: AgentState) -> AgentState:
        """Select tool based on query type."""
        tool_map = {
            "factual": "focused_search",
            "overview": "full_document_read",
            "comparison": "focused_search",
            "temporal": "temporal_query",
            "entity": "knowledge_search",
        }
        state.tool = tool_map.get(state.query_type, "focused_search")
        logger.info("Selected tool: %s", state.tool)
        return state

    def execute_tool(self, state: AgentState) -> AgentState:
        """Execute selected tool."""
        from agent.tools import focused_search, full_document_read, vector_search
        from core.models import Chunk

        if state.tool == "vector_search":
            state.results = vector_search(
                state.query, self._retriever.retriever, top_k=settings.top_k_rerank
            )

        elif state.tool == "full_document_read":
            doc_text = full_document_read(self._retriever.retriever.vector_store)
            state.results = [{"id": "full_doc", "content": doc_text, "score": 1.0}]

        elif state.tool == "focused_search":
            state.results = focused_search(
                state.query, self._retriever.retriever, top_k=settings.top_k_rerank
            )

        elif state.tool == "knowledge_search":
            state.results = self._kg_search(state.query)

        elif state.tool == "temporal_query":
            state.results = self._temporal_search(state.query)

        # Convert to SearchResult
        state.search_results = [
            SearchResult(
                chunk=Chunk(id=r.get("id", ""), content=r.get("content", "")),
                score=r.get("score", 0.0),
                rank=r.get("rank", i + 1),
            )
            for i, r in enumerate(state.results)
        ]

        logger.info("Tool '%s' returned %d results", state.tool, len(state.results))
        return state

    def _kg_search(self, query: str) -> list[dict]:
        """Search knowledge graph for entities/facts."""
        if not self._kg_client:
            return []
        try:
            from utils.async_helpers import run_async

            facts = run_async(self._kg_client.search(query))

            return [
                {
                    "id": f"kg_{i}",
                    "content": f.content,
                    "score": f.score,
                    "source": "kg",
                    "valid_at": f.valid_at,
                }
                for i, f in enumerate(facts)
                if f.content.strip()
            ]
        except Exception as e:
            logger.error("KG search in agent failed: %s", e)
            return []

    def _temporal_search(self, query: str) -> list[dict]:
        """Temporal Cypher query for time-based questions."""
        if not self._kg_client:
            return []
        try:
            facts = self._kg_client.temporal_query(entity_name=_extract_entity(query))
            return [
                {
                    "id": f"temporal_{i}",
                    "content": f"{f.content} (valid: {f.valid_at})" if f.valid_at else f.content,
                    "score": f.score,
                    "source": "kg",
                }
                for i, f in enumerate(facts)
            ]
        except Exception as e:
            logger.error("Temporal search failed: %s", e)
            return []

    def evaluate(self, state: AgentState) -> AgentState:
        """Evaluate result relevance."""
        from generation.reflector import evaluate_relevance

        if not state.search_results:
            state.relevance_score = 0.0
            return state

        state.relevance_score = evaluate_relevance(
            state.query, state.search_results, self._openai
        )
        logger.info("Relevance score: %.2f", state.relevance_score)
        return state

    def should_retry(self, state: AgentState) -> bool:
        """Decide if retry needed."""
        return (
            state.relevance_score < settings.relevance_threshold
            and state.retries < settings.max_retries
            and state.tool not in ("focused_search", "knowledge_search")
        )

    def retry_with_fallback(self, state: AgentState) -> AgentState:
        """Retry with focused_search as fallback."""
        state.tool = "focused_search"
        state.retries += 1
        logger.info("Retrying with focused_search (attempt %d)", state.retries)
        return self.execute_tool(state)

    def generate(self, state: AgentState) -> AgentState:
        """Generate final answer."""
        from generation.generator import generate_answer

        qa_result = generate_answer(state.query, state.search_results, self._openai)
        qa_result.retries = state.retries
        state.qa_result = qa_result
        state.answer = qa_result.answer
        return state

    def run(self, query: str) -> QAResult:
        """Execute full agent pipeline.

        Flow: classify → select_tool → execute → evaluate → [retry?] → generate
        """
        if not query or not query.strip():
            return QAResult(
                answer="Empty query.",
                sources=[], confidence=0.0, query=query, retries=0,
            )

        state = AgentState(query=query)

        state = self.classify_query(state)
        state = self.select_tool(state)
        state = self.execute_tool(state)
        state = self.evaluate(state)

        if self.should_retry(state):
            state = self.retry_with_fallback(state)
            state = self.evaluate(state)

        state = self.generate(state)
        return state.qa_result


def _extract_entity(query: str) -> str:
    """Extract likely entity name from query for temporal search."""
    # Simple heuristic: use the longest capitalized word or quoted phrase
    import re
    # Try quoted phrases first
    quoted = re.findall(r'"([^"]+)"', query)
    if quoted:
        return quoted[0]
    # Fall back to query itself (temporal_query will use CONTAINS)
    return query
