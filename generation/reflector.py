"""Re-export RAG 2.0 evaluate_relevance (avoids sys.path shadowing).

Local generation/ package shadows rag-2.0/generation/,
so we temporarily swap sys.path and sys.modules to load RAG 2.0's
generation.reflector.evaluate_relevance.
"""

import sys
from pathlib import Path

_rag_root = str(Path.home() / "rag-2.0")


def _load_evaluate_relevance():
    """Load evaluate_relevance from RAG 2.0."""
    orig_path = sys.path[:]

    saved_modules = {}
    for key in list(sys.modules.keys()):
        if key == "generation" or key.startswith("generation."):
            saved_modules[key] = sys.modules.pop(key)

    sys.path.insert(0, _rag_root)

    try:
        import importlib

        mod = importlib.import_module("generation.reflector")
        return mod.evaluate_relevance
    finally:
        for key in list(sys.modules.keys()):
            if key == "generation" or key.startswith("generation."):
                sys.modules.pop(key, None)

        sys.modules.update(saved_modules)
        sys.path[:] = orig_path


try:
    evaluate_relevance = _load_evaluate_relevance()
except (ImportError, FileNotFoundError) as e:
    raise ImportError(
        f"Cannot load RAG 2.0 reflector from {_rag_root}. "
        f"Ensure ~/rag-2.0 exists with generation/reflector.py: {e}"
    ) from e
