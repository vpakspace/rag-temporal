"""Re-export RAG 2.0 agent tools (avoids sys.path shadowing).

Local agent/ package shadows rag-2.0/agent/,
so we temporarily swap sys.path and sys.modules to load RAG 2.0's
agent.tools module.
"""

import sys
from pathlib import Path

_rag_root = str(Path.home() / "rag-2.0")


def _load_rag20_tools():
    """Load tools from RAG 2.0 with full transitive import support."""
    orig_path = sys.path[:]

    saved_modules = {}
    for key in list(sys.modules.keys()):
        if key == "agent" or key.startswith("agent."):
            saved_modules[key] = sys.modules.pop(key)

    sys.path.insert(0, _rag_root)

    try:
        import importlib

        mod = importlib.import_module("agent.tools")
        return mod.vector_search, mod.focused_search, mod.full_document_read
    finally:
        for key in list(sys.modules.keys()):
            if key == "agent" or key.startswith("agent."):
                sys.modules.pop(key, None)

        sys.modules.update(saved_modules)
        sys.path[:] = orig_path


try:
    vector_search, focused_search, full_document_read = _load_rag20_tools()
except (ImportError, FileNotFoundError) as e:
    raise ImportError(
        f"Cannot load RAG 2.0 tools from {_rag_root}. "
        f"Ensure ~/rag-2.0 exists with agent/tools.py: {e}"
    ) from e
