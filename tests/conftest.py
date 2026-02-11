"""Shared test configuration and stubs for CI environment.

RAG-Temporal imports core.models (Chunk, SearchResult, QAResult) from
the sibling project ~/rag-2.0.  In CI this directory does not exist,
so we provide lightweight Pydantic stubs in sys.modules *before*
pytest collects test files.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

# ---------------------------------------------------------------------------
# 1. Ensure the project root is on sys.path (so `from config import ...` works)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# 2. Stub RAG 2.0 core.models when ~/rag-2.0 is not available (CI)
# ---------------------------------------------------------------------------
RAG_ROOT = Path.home() / "rag-2.0"

if not (RAG_ROOT / "core" / "models.py").exists():
    from pydantic import BaseModel, Field

    class Chunk(BaseModel):
        id: str = ""
        content: str = ""
        context: str = ""
        embedding: list[float] = Field(default_factory=list)
        metadata: dict = Field(default_factory=dict)

        @property
        def enriched_content(self) -> str:
            if self.context:
                return f"{self.context}\n\n{self.content}"
            return self.content

    class SearchResult(BaseModel):
        chunk: Chunk = Field(default_factory=Chunk)
        score: float = 0.0
        rank: int = 0

    class QAResult(BaseModel):
        answer: str = ""
        sources: list[SearchResult] = Field(default_factory=list)
        confidence: float = 0.0
        query: str = ""
        expanded_query: str = ""
        retries: int = 0

    # Register stub modules so `from core.models import ...` resolves
    core_module = ModuleType("core")
    core_module.__path__ = []  # mark as package

    models_module = ModuleType("core.models")
    models_module.Chunk = Chunk
    models_module.SearchResult = SearchResult
    models_module.QAResult = QAResult

    sys.modules["core"] = core_module
    sys.modules["core.models"] = models_module

    # Stub storage.vector_store (used in patch() targets by tests)
    storage_module = ModuleType("storage")
    storage_module.__path__ = []

    vector_store_module = ModuleType("storage.vector_store")
    vector_store_module.VectorStore = type("VectorStore", (), {})

    storage_module.vector_store = vector_store_module

    sys.modules["storage"] = storage_module
    sys.modules["storage.vector_store"] = vector_store_module
