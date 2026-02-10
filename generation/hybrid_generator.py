"""Enhanced generation with temporal KG context + reflect loop."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from config import settings
from core.models import QAResult, SearchResult

if TYPE_CHECKING:
    from openai import OpenAI

    from retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


def hybrid_answer(
    query: str,
    hybrid_retriever: HybridRetriever,
    openai_client: OpenAI | None = None,
    mode: str = "hybrid",
) -> QAResult:
    """Generate answer using hybrid retrieval + reflect loop.

    Pipeline:
    1. Hybrid retrieve (vector + KG)
    2. Build enriched context with KG temporal facts
    3. Evaluate relevance
    4. If low → retry with KG-expanded query (max 3)
    5. Generate final answer

    Args:
        query: User query
        hybrid_retriever: HybridRetriever instance
        openai_client: Optional OpenAI client
        mode: "hybrid" | "vector" | "kg"
    """
    if openai_client is None:
        from openai import OpenAI
        openai_client = OpenAI(api_key=settings.openai_api_key)

    best_context = ""
    best_score = 0.0
    current_query = query
    retries_used = 0

    for attempt in range(settings.max_retries + 1):
        # Retrieve
        results = hybrid_retriever.retrieve(current_query, mode=mode, top_k=10)

        if not results:
            logger.warning("No results on attempt %d", attempt + 1)
            if attempt < settings.max_retries:
                current_query = _expand_with_kg(query, openai_client)
                retries_used += 1
                continue
            break

        # Build context
        context = _build_context(results)

        # Evaluate relevance
        score = _evaluate_relevance(query, context, openai_client)

        if score > best_score:
            best_context = context
            best_score = score

        if score >= settings.relevance_threshold:
            logger.info("Relevance OK (%.2f >= %.2f) on attempt %d", score, settings.relevance_threshold, attempt + 1)
            break

        if attempt < settings.max_retries:
            logger.info("Low relevance (%.2f), retrying with expanded query", score)
            current_query = _expand_with_kg(query, openai_client)
            retries_used += 1

    # Generate answer
    if not best_context:
        return QAResult(
            answer="I don't have enough context to answer this question.",
            sources=[],
            confidence=0.0,
            query=query,
            retries=retries_used,
        )

    answer = _generate(query, best_context, openai_client)

    return QAResult(
        answer=answer,
        sources=[],
        confidence=best_score / 5.0,
        query=query,
        expanded_query=current_query if retries_used > 0 else "",
        retries=retries_used,
    )


def _build_context(results) -> str:
    """Build context string from HybridResults, marking KG facts with temporal info."""
    parts: list[str] = []
    for i, r in enumerate(results, 1):
        if r.source == "kg":
            valid_at = r.metadata.get("valid_at", "")
            temporal_tag = f" (valid: {valid_at})" if valid_at else ""
            parts.append(f"[Knowledge fact{temporal_tag}]\n{r.content}")
        else:
            parts.append(f"[Chunk {i}]\n{r.content}")
    return "\n\n".join(parts)


def _evaluate_relevance(query: str, context: str, openai_client) -> float:
    """Quick relevance evaluation (1-5 scale)."""
    prompt = f"""Rate how relevant this context is to the query (1-5).
Return ONLY a number.

Query: {query}

Context (first 1500 chars):
{context[:1500]}

Score:"""

    try:
        response = openai_client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        text = response.choices[0].message.content or "2.5"
        return float(text.strip().split()[0])
    except (ValueError, IndexError):
        return 2.5
    except Exception as e:
        logger.error("Relevance evaluation failed: %s", e)
        return 2.5


def _expand_with_kg(query: str, openai_client) -> str:
    """Expand query to include knowledge-graph oriented terms."""
    prompt = f"""Rewrite this query to also search for related entities, temporal facts, and relationships.
Keep it concise (1-2 sentences).

Original: {query}

Expanded:"""

    try:
        response = openai_client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        return response.choices[0].message.content or query
    except Exception:
        return query


def _generate(query: str, context: str, openai_client) -> str:
    """Generate final answer from context."""
    system_prompt = (
        "You are a precise Q&A assistant. Answer ONLY based on the provided context. "
        "The context includes both document chunks and knowledge graph facts (with temporal validity). "
        "If the context doesn't contain enough information, say so. "
        "Cite the source type (chunk or knowledge fact) when possible."
    )

    user_prompt = f"""Query: {query}

Context:
{context}

Please provide an answer based on the above context."""

    try:
        response = openai_client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error("Answer generation failed: %s", e)
        return f"Error generating answer: {e}"
