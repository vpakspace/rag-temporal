"""Re-export RAG 2.0 Retriever (avoids sys.path shadowing).

Local retrieval/ package shadows rag-2.0/retrieval/,
so we temporarily swap sys.path and sys.modules to load RAG 2.0's
retrieval submodules (retriever, query_expander, reranker).
"""

import sys
from pathlib import Path

_rag_root = str(Path.home() / "rag-2.0")


def _load_rag20_retriever():
    """Load Retriever from RAG 2.0 with full transitive import support."""
    # Save state
    orig_path = sys.path[:]

    # Save and remove cached local 'retrieval' modules
    saved_modules = {}
    for key in list(sys.modules.keys()):
        if key == "retrieval" or key.startswith("retrieval."):
            saved_modules[key] = sys.modules.pop(key)

    # Prioritize rag-2.0 so retrieval.* resolves to rag-2.0/retrieval/
    sys.path.insert(0, _rag_root)

    try:
        import importlib

        mod = importlib.import_module("retrieval.retriever")
        return mod.Retriever
    finally:
        # Remove rag-2.0's retrieval modules from cache
        for key in list(sys.modules.keys()):
            if key == "retrieval" or key.startswith("retrieval."):
                sys.modules.pop(key, None)

        # Restore local retrieval modules
        sys.modules.update(saved_modules)

        # Restore path
        sys.path[:] = orig_path


try:
    Retriever = _load_rag20_retriever()
except (ImportError, FileNotFoundError) as e:
    raise ImportError(
        f"Cannot load RAG 2.0 Retriever from {_rag_root}. "
        f"Ensure ~/rag-2.0 exists with retrieval/retriever.py: {e}"
    ) from e
