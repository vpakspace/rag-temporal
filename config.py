"""Unified RAG + Temporal Knowledge Base configuration."""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic_settings import BaseSettings

# Append RAG 2.0 root to sys.path for fallback imports
# Use append (not insert) so our project modules take priority
RAG_ROOT = Path.home() / "rag-2.0"
if str(RAG_ROOT) not in sys.path:
    sys.path.append(str(RAG_ROOT))


class Settings(BaseSettings):
    """Unified settings for RAG + KG pipeline."""

    # OpenAI
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    embedding_dimensions: int = 1536

    # Neo4j (shared container: temporal-kb-neo4j)
    # Credentials loaded from .env — no hardcoded defaults for password
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 100

    # Retrieval
    top_k_retrieval: int = 20
    top_k_rerank: int = 5
    rerank_method: str = "cosine"

    # Reflection
    max_retries: int = 2
    relevance_threshold: float = 3.0

    # Knowledge Graph
    kg_enabled: bool = True
    kg_max_episode_chars: int = 8000
    kg_episode_overlap: int = 200
    kg_search_results: int = 10
    kg_fact_score: float = 0.8  # Fixed score for KG facts in hybrid merge

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
