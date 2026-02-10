# RAG-Temporal: Unified RAG + Knowledge Graph System

Hybrid information retrieval system that combines **vector search** (RAG 2.0) with **temporal knowledge graph** (Graphiti/Neo4j) for comprehensive document understanding.

One document produces **two representations simultaneously**: dense vector chunks for semantic search and structured knowledge graph episodes for entity/relationship/temporal queries. At query time, results from both sources are merged, deduplicated, and interleaved to provide richer answers than either approach alone.

## Key Results

| Retrieval Mode | Description | Strength |
|---|---|---|
| **Hybrid (Vector + KG)** | Combines both sources | Best for factual questions with temporal context |
| **Vector only** | Semantic similarity search | Best for broad topic questions |
| **KG only** | Entity/relationship traversal | Best for "who/what/when" questions |
| **Agent (auto-route)** | Classifies query, picks optimal tool | Best general-purpose mode |

## Architecture

```
                    ┌─────────────────┐
                    │   Document      │
                    │  (PDF/DOCX/TXT) │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  DualPipeline   │
                    └──┬──────────┬───┘
                       │          │
           ┌───────────▼──┐  ┌───▼───────────┐
           │  RAG Track   │  │   KG Track    │
           │              │  │               │
           │ chunk →      │  │ split_episodes│
           │ enrich →     │  │ → Graphiti    │
           │ embed →      │  │   add_episode │
           │ store        │  │               │
           └───────┬──────┘  └───────┬───────┘
                   │                 │
           ┌───────▼──────┐  ┌───────▼───────┐
           │  Neo4j       │  │   Graphiti    │
           │  Vector Index│  │   Edges/Nodes │
           └───────┬──────┘  └───────┬───────┘
                   │                 │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │ HybridRetriever │
                   │ merge/dedup/    │
                   │ interleave      │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │HybridGenerator  │
                   │ reflect loop +  │
                   │ KG-expanded     │
                   │ retry           │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │    Answer +     │
                   │   Confidence    │
                   └─────────────────┘
```

## Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Language | Python 3.12 | Core runtime |
| Vector Store | Neo4j Vector Index | Cosine similarity search over embeddings |
| Knowledge Graph | Graphiti (graphiti-core 0.26.3) | Temporal knowledge graph with bi-temporal edges |
| Graph Database | Neo4j 5.x (Docker) | Shared backend for vector index + KG |
| LLM | OpenAI gpt-4o-mini | Query expansion, answer generation, relevance evaluation |
| Embeddings | OpenAI text-embedding-3-small (1536 dim) | Document chunk embeddings |
| Document Processing | IBM Docling | PDF/DOCX/TXT parsing with GPU acceleration |
| UI | Streamlit (port 8503) | 5-tab interface with i18n (EN/RU) |
| Configuration | Pydantic Settings | Type-safe config from `.env` |
| CI/CD | GitHub Actions | pytest + ruff on Python 3.11/3.12 |
| RAG Pipeline | [RAG 2.0](https://github.com/vpakspace/rag-2.0) | Reused via `sys.path` — loader, chunker, enricher, vector_store, retriever |

## Prerequisites

- **Python 3.12+**
- **Neo4j 5.x** (Docker recommended)
- **OpenAI API key** (for embeddings + LLM)
- **RAG 2.0** cloned at `~/rag-2.0/` ([vpakspace/rag-2.0](https://github.com/vpakspace/rag-2.0))

## Quick Start

```bash
# 1. Clone
git clone https://github.com/vpakspace/rag-temporal.git
cd rag-temporal

# 2. Clone RAG 2.0 (sibling dependency)
git clone https://github.com/vpakspace/rag-2.0.git ~/rag-2.0

# 3. Start Neo4j
docker run -d \
  --name temporal-kb-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/temporal_kb_2026 \
  -e NEO4J_PLUGINS='["apoc"]' \
  -v neo4j_data:/data \
  neo4j:5

# 4. Install dependencies
pip install -r requirements.txt
# Also install RAG 2.0 dependencies:
pip install -r ~/rag-2.0/requirements.txt

# 5. Configure
cp .env.example .env
# Edit .env — set OPENAI_API_KEY

# 6. Run UI
./run_streamlit.sh
# Opens at http://localhost:8503

# 7. Run tests
pytest tests/ -x -q
```

## Configuration

All settings are managed via `.env` file (Pydantic Settings):

```bash
# OpenAI (required)
OPENAI_API_KEY=sk-...

# Neo4j (shared with temporal-knowledge-base)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=temporal_kb_2026

# LLM Settings
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
EMBEDDING_DIMENSIONS=1536

# Chunking
CHUNK_SIZE=800
CHUNK_OVERLAP=100

# Knowledge Graph
KG_ENABLED=true
KG_MAX_EPISODE_CHARS=8000
```

### Key Configuration Notes

- **Neo4j container** is shared with [temporal-knowledge-base](https://github.com/vpakspace/temporal-knowledge-base) project (`temporal-kb-neo4j`)
- **RAG 2.0 path**: `config.py` sets `RAG_ROOT = Path.home() / "rag-2.0"` and appends it to `sys.path`
- **KG_MAX_EPISODE_CHARS**: documents larger than this are split into overlapping episodes (200 char overlap)

## Project Structure

```
rag-temporal/
├── agent/
│   ├── hybrid_agent.py          # HybridAgent: 5-tool agent with classify → route → execute
│   └── tools.py                 # Bridge: loads RAG 2.0 agent tools
├── config.py                    # Pydantic Settings + sys.path setup
├── generation/
│   ├── hybrid_generator.py      # hybrid_answer(): reflect loop + KG-expanded retry
│   ├── generator.py             # Bridge: loads RAG 2.0 generate_answer
│   └── reflector.py             # Bridge: loads RAG 2.0 evaluate_relevance
├── ingestion/
│   └── dual_pipeline.py         # DualPipeline: RAG chunks + KG episodes simultaneously
├── kg/
│   └── client.py                # KGClient: Graphiti wrapper + direct Cypher queries
├── retrieval/
│   ├── hybrid_retriever.py      # HybridRetriever: vector + KG → merge/dedup/interleave
│   └── retriever.py             # Bridge: loads RAG 2.0 Retriever
├── ui/
│   └── i18n.py                  # 106 translation keys (EN/RU)
├── utils/
│   └── async_helpers.py         # run_async(): sync → async helper for Streamlit
├── tests/
│   ├── conftest.py              # CI stubs for RAG 2.0 modules
│   ├── test_benchmark.py
│   ├── test_dual_ingestion.py
│   ├── test_hybrid_agent.py
│   └── test_hybrid_retrieval.py
├── streamlit_app.py             # 5-tab UI (Ingest, Search, KG, Benchmark, Settings)
├── .github/workflows/ci.yml    # GitHub Actions: pytest + ruff
├── requirements.txt             # 7 dependencies
├── .env.example                 # Environment template
└── run_streamlit.sh             # Launcher (port 8503)
```

## Components in Detail

### Dual-Track Ingestion (`ingestion/dual_pipeline.py`)

A single document goes through two parallel tracks:

**RAG Track** (from RAG 2.0):
1. `load_file()` — Docling parses PDF/DOCX/TXT (optional GPU acceleration)
2. `chunk_text()` — Semantic chunking (800 chars, 100 overlap)
3. `enrich_chunks()` — Contextual enrichment via OpenAI (adds document-level context to each chunk)
4. `embed_chunks()` — Generates embeddings (text-embedding-3-small, 1536 dim)
5. `store.add_chunks()` — Stores in Neo4j Vector Index

**KG Track**:
1. `KGClient.ingest_text()` — Splits large documents into episodes (>8K chars → overlapping segments)
2. `Graphiti.add_episode()` — Extracts entities, relationships, temporal facts into knowledge graph

Result: `IngestionResult(rag_chunks=N, kg_episodes=M, errors=[])`

### Knowledge Graph Client (`kg/client.py`)

Wraps Graphiti + direct Cypher queries with several critical adaptations:

- **Lucene sanitization**: `sanitize_lucene()` escapes special characters (`+ - && || ! ( ) { } [ ] ^ " ~ * ? : \ /`) before Graphiti search
- **Episode splitting**: `split_episodes()` breaks text >8K chars on paragraph boundaries (`\n\n`) with configurable overlap
- **COALESCE queries**: Graphiti uses `uuid`/`labels` while our layer uses `id`/`entity_type` — all Cypher queries use `COALESCE(e.id, e.uuid)` and `COALESCE(e.entity_type, head(e.labels))` to handle both schemas
- **KGFact dataclass**: Unified representation (content, source, valid_at, entity_name, score, metadata)

Methods:
- `search(query, limit)` — Graphiti edge-based search, returns `list[KGFact]`
- `temporal_query(query, limit)` — Direct Cypher for temporal facts
- `get_entities(limit)` — Entity list with COALESCE
- `get_relationships(limit)` — Relationship list with COALESCE
- `ingest_text(text, source)` — Episode splitting + Graphiti ingestion

### Hybrid Retriever (`retrieval/hybrid_retriever.py`)

Combines vector and KG search with configurable mode:

```python
retriever = HybridRetriever(vector_store, kg_client, retriever)
results = retriever.retrieve(query, mode="hybrid", top_k=10)
# mode: "hybrid" | "vector" | "kg"
```

Merge strategy:
1. **Normalize** vector scores to 0-1 range (min-max normalization)
2. **Assign** KG facts a fixed score (0.8) — configurable via `settings.kg_fact_score`
3. **Interleave** vector and KG results alternately
4. **Dedup** by content substring matching (removes near-duplicates)
5. Return top-K `HybridResult(content, source, score, metadata)`

### Hybrid Generator (`generation/hybrid_generator.py`)

Self-reflective generation with KG-expanded retry:

```python
answer, confidence = hybrid_answer(query, hybrid_retriever, kg_client)
```

Algorithm:
1. Retrieve context (hybrid mode)
2. Build context string — KG facts are tagged with `[Knowledge Graph]` and temporal markers
3. Evaluate relevance (1-5 scale via LLM)
4. If relevance < 3.0: **expand query with KG terms** → re-retrieve → re-evaluate
5. Generate final answer with confidence score

Named constants: `_MAX_RELEVANCE_SCORE=5.0`, `_FALLBACK_RELEVANCE=2.5`, `_CONTEXT_PREVIEW_CHARS=1500`

### Hybrid Agent (`agent/hybrid_agent.py`)

LangGraph-style agent with 5 tools:

| Tool | Source | Description |
|---|---|---|
| `vector_search` | RAG 2.0 | Semantic similarity search |
| `focused_search` | RAG 2.0 | Targeted search with query rewriting |
| `full_document_read` | RAG 2.0 | Retrieve full document content |
| `knowledge_search` | KG | Graphiti entity/relationship search |
| `temporal_query` | KG | Direct Cypher for temporal facts |

Pipeline:
1. **Classify** query type (factual / overview / comparison / temporal / entity)
2. **Route** to optimal tool:
   - factual → `focused_search`
   - overview → `full_document_read`
   - comparison → `focused_search`
   - temporal → `temporal_query`
   - entity → `knowledge_search`
3. **Execute** tool
4. **Evaluate** relevance (1-5)
5. If relevance < 3.0 and retries < max: switch to next tool → retry
6. **Generate** answer

### Bridge Modules

RAG-Temporal imports modules from [RAG 2.0](https://github.com/vpakspace/rag-2.0) (`~/rag-2.0/`) via `sys.path.append`. Since both projects have packages with the same names (e.g., `retrieval/`, `generation/`, `agent/`), Python finds the local one first — **sys.path shadowing**.

**Solution**: 4 bridge modules that temporarily swap `sys.path` AND `sys.modules`:

| Bridge | Location | Loads from RAG 2.0 |
|---|---|---|
| `retrieval/retriever.py` | `retrieval/` | `Retriever` class |
| `agent/tools.py` | `agent/` | Agent tool functions |
| `generation/generator.py` | `generation/` | `generate_answer()` |
| `generation/reflector.py` | `generation/` | `evaluate_relevance()` |

Each bridge follows the same pattern:
```python
def _load_external():
    orig_path = sys.path[:]
    saved = {k: sys.modules.pop(k) for k in list(sys.modules)
             if k == "pkg" or k.startswith("pkg.")}
    sys.path.insert(0, external_root)
    try:
        return importlib.import_module("pkg.module").ClassName
    finally:
        for k in list(sys.modules):
            if k == "pkg" or k.startswith("pkg."):
                sys.modules.pop(k, None)
        sys.modules.update(saved)
        sys.path[:] = orig_path
```

**Key insight**: Cleaning `sys.modules` is essential — just changing `sys.path` isn't enough for transitive imports.

### Async Helper (`utils/async_helpers.py`)

Streamlit runs its own event loop, so calling `asyncio.run()` from sync context fails. `run_async()` detects this and uses `ThreadPoolExecutor` as a workaround:

```python
def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Loop already running (Streamlit) — run in thread
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
```

### Internationalization (`ui/i18n.py`)

106 translation keys for English and Russian:

```python
t = get_translator("ru")  # or "en"
t("search_title")         # "Поиск и вопросы"
t("ingest_success", count=5)  # "Загружено 5 чанков"
```

Keys cover: app title, sidebar, 5 tabs (ingest, search, kg, benchmark, settings), and common UI elements.

## Streamlit UI

**5 tabs** on port 8503:

| Tab | Features |
|---|---|
| **Ingest** | Drag & drop files, dual-track progress (RAG chunks + KG episodes) |
| **Search & Q&A** | Mode selector (Hybrid/Vector/KG/Agent), answer + confidence, sources list |
| **Knowledge Graph** | Entity table, relationship table, temporal facts |
| **Benchmark** | Run evaluation suite, PASS/FAIL per question |
| **Settings** | Config viewer, chunk count, clear database |

Sidebar: Language selector (EN/RU), current configuration display.

Cached resources (`@st.cache_resource`): VectorStore, OpenAI client, Retriever, KGClient.

## Testing

```bash
# Run all tests
pytest tests/ -x -q

# 49 tests, ~0.11s
```

**4 test files**:
- `test_benchmark.py` — Benchmark evaluation logic
- `test_dual_ingestion.py` — DualPipeline (RAG + KG tracks)
- `test_hybrid_agent.py` — HybridAgent tool routing and execution
- `test_hybrid_retrieval.py` — HybridRetriever merge/dedup/interleave

### CI/CD

GitHub Actions runs on every push:
- **Matrix**: Python 3.11, 3.12
- **Steps**: Install dependencies → pytest → ruff lint
- **CI stubs**: `tests/conftest.py` creates Pydantic stub models for RAG 2.0 modules (not available in CI environment)

## Known Limitations and Gotchas

### Graphiti Property Mismatch

Graphiti Entity nodes use different property names than our dual-track layer:

| Property | Graphiti | Our Layer |
|---|---|---|
| ID | `uuid` | `id` |
| Type | `labels` (array) | `entity_type` (string) |

All Cypher queries must use `COALESCE()`:
```cypher
RETURN COALESCE(e.id, e.uuid) AS id,
       COALESCE(e.entity_type, head(e.labels)) AS entity_type
```

### WSL2 + Docker Desktop

Neo4j may fail with `UnknownHostException` when running via Docker Compose on WSL2 + Docker Desktop. **Workaround**: run Neo4j standalone with `docker run` (see Quick Start).

### RAG 2.0 Dependency

This project imports modules from `~/rag-2.0/` at runtime. If RAG 2.0 is not cloned at that path, the application will fail with `ImportError`. The path is configured in `config.py:RAG_ROOT`.

### Async in Streamlit

Graphiti operations are async. Streamlit runs its own event loop, so direct `asyncio.run()` fails. All async calls go through `utils/async_helpers.py:run_async()`.

## Dependencies

```
graphiti-core==0.26.3    # Temporal knowledge graph
neo4j>=5.0,<6.0          # Graph database driver
openai>=1.0,<2.0         # LLM + embeddings
streamlit>=1.30.0,<2.0   # Web UI
pydantic-settings>=2.0,<3.0  # Configuration
python-dotenv>=1.0,<2.0  # .env loading
docling>=2.0,<3.0        # Document processing (PDF/DOCX)
```

Plus RAG 2.0 dependencies (see `~/rag-2.0/requirements.txt`).

## Related Projects

| Project | Description | Port |
|---|---|---|
| [RAG 2.0](https://github.com/vpakspace/rag-2.0) | Advanced RAG pipeline (Contextual + Self-Reflective + Agentic) | 8502 |
| [Temporal Knowledge Base](https://github.com/vpakspace/temporal-knowledge-base) | Bi-temporal KG with Graphiti, FastAPI, MCP | 8000/8501 |
| **RAG-Temporal** (this) | Unified RAG + KG hybrid system | 8503 |

## License

MIT
