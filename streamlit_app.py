"""Streamlit UI for unified RAG + Temporal Knowledge Base system."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402 — also sets up RAG 2.0 sys.path
from ui.i18n import get_translator  # noqa: E402

# Page config
st.set_page_config(page_title="RAG + Temporal KB", page_icon="🔗", layout="wide")


# --- Cached resources ---

@st.cache_resource
def get_vector_store():
    from storage.vector_store import VectorStore
    store = VectorStore()
    store.init_index()
    return store


@st.cache_resource
def get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=settings.openai_api_key)


@st.cache_resource
def get_retriever(_vector_store):
    from retrieval.retriever import Retriever
    return Retriever(_vector_store, get_openai_client())


@st.cache_resource
def get_kg_client():
    import asyncio
    from kg.client import KGClient
    client = KGClient()
    asyncio.run(client.connect())
    return client


@st.cache_resource
def get_hybrid_retriever(_retriever, _kg_client):
    from retrieval.hybrid_retriever import HybridRetriever
    return HybridRetriever(retriever=_retriever, kg_client=_kg_client)


@st.cache_resource
def get_dual_pipeline(_vector_store, _kg_client):
    from ingestion.dual_pipeline import DualPipeline
    return DualPipeline(vector_store=_vector_store, kg_client=_kg_client)


@st.cache_resource
def get_hybrid_agent(_hybrid_retriever, _kg_client):
    from agent.hybrid_agent import HybridAgent
    return HybridAgent(_hybrid_retriever, kg_client=_kg_client, openai_client=get_openai_client())


# --- Sidebar ---

def render_sidebar(t):
    with st.sidebar:
        st.title(t("app_title"))
        st.caption(t("app_subtitle"))

        st.divider()

        # Language selector
        lang_options = {"English": "en", "Русский": "ru"}
        selected_lang_label = st.selectbox(
            t("language"),
            options=list(lang_options.keys()),
            index=0 if st.session_state.get("lang", "en") == "en" else 1,
        )
        new_lang = lang_options[selected_lang_label]
        if new_lang != st.session_state.get("lang", "en"):
            st.session_state.lang = new_lang
            st.rerun()

        # GPU toggle
        st.session_state.use_gpu = st.toggle(
            t("gpu_acceleration"),
            value=st.session_state.get("use_gpu", False),
        )

        # KG toggle
        st.session_state.kg_enabled = st.toggle(
            t("kg_enabled"),
            value=st.session_state.get("kg_enabled", True),
        )

        st.divider()

        # Stats
        st.subheader(t("stats_title"))
        try:
            store = get_vector_store()
            st.metric(t("vector_chunks"), store.count())
        except Exception as e:
            logging.getLogger(__name__).debug("Vector store stats unavailable: %s", e)
            st.metric(t("vector_chunks"), "N/A")

        if st.session_state.get("kg_enabled", True):
            try:
                kg = get_kg_client()
                st.metric(t("kg_entities"), kg.entity_count())
            except Exception as e:
                logging.getLogger(__name__).debug("KG stats unavailable: %s", e)
                st.metric(t("kg_entities"), "N/A")


# --- Tab: Ingest ---

def tab_ingest(t):
    st.header(t("ingest_title"))
    st.caption(t("ingest_supported"))

    col1, col2 = st.columns(2)

    with col1:
        uploaded = st.file_uploader(t("ingest_upload"), type=["txt", "pdf", "docx", "pptx", "xlsx", "html"])
    with col2:
        file_path_input = st.text_input(t("ingest_file_path"), placeholder="/path/to/document.pdf")

    col_a, col_b = st.columns(2)
    with col_a:
        skip_enrichment = st.checkbox(t("ingest_skip_enrichment"), value=False)
    with col_b:
        skip_kg = st.checkbox(t("ingest_skip_kg"), value=not st.session_state.get("kg_enabled", True))

    if st.button(t("ingest_button"), type="primary", width="stretch"):
        file_path = None

        # Determine source
        if uploaded is not None:
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                file_path = tmp.name
        elif file_path_input.strip():
            file_path = file_path_input.strip()

        if not file_path:
            st.warning(t("warning") + ": No file selected")
            return

        tmp_file_created = uploaded is not None  # track if we created a temp file

        with st.spinner(t("ingest_processing")):
            try:
                store = get_vector_store()
                kg = get_kg_client() if st.session_state.get("kg_enabled", True) and not skip_kg else None
                pipeline = get_dual_pipeline(store, kg)
                result = pipeline.ingest_file(
                    file_path,
                    use_gpu=st.session_state.get("use_gpu", False),
                    skip_enrichment=skip_enrichment,
                    skip_kg=skip_kg,
                )

                # Show results
                col_r, col_k = st.columns(2)
                with col_r:
                    st.metric(t("ingest_rag_track"), f"{result.rag_chunks} {t('ingest_chunks_stored')}")
                with col_k:
                    st.metric(t("ingest_kg_track"), f"{result.kg_episodes} {t('ingest_episodes_ingested')}")

                if result.errors:
                    for err in result.errors:
                        st.error(f"{t('ingest_error')}: {err}")
                else:
                    st.success(t("ingest_success"))

            except Exception as e:
                st.error(f"{t('error')}: {e}")
            finally:
                if tmp_file_created and file_path and os.path.exists(file_path):
                    os.unlink(file_path)


# --- Tab: Search & Q&A ---

def tab_search(t):
    st.header(t("search_title"))

    query = st.text_input(t("search_query"), placeholder="What is LMCache?")

    mode_labels = {
        t("search_mode_hybrid"): "hybrid",
        t("search_mode_vector"): "vector",
        t("search_mode_kg"): "kg",
        t("search_mode_agent"): "agent",
    }
    selected_mode_label = st.radio(
        t("search_mode"),
        options=list(mode_labels.keys()),
        horizontal=True,
    )
    mode = mode_labels[selected_mode_label]

    if st.button(t("search_button"), type="primary") and query.strip():
        with st.spinner(t("search_thinking")):
            try:
                store = get_vector_store()
                client = get_openai_client()
                retriever = get_retriever(store)
                kg = get_kg_client() if st.session_state.get("kg_enabled", True) else None
                hybrid_ret = get_hybrid_retriever(retriever, kg)

                if mode == "agent":
                    agent = get_hybrid_agent(hybrid_ret, kg)
                    qa_result = agent.run(query)
                else:
                    from generation.hybrid_generator import hybrid_answer
                    qa_result = hybrid_answer(query, hybrid_ret, client, mode=mode)

                # Display answer
                st.subheader(t("search_answer"))
                st.write(qa_result.answer)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(t("search_confidence"), f"{qa_result.confidence:.0%}")
                with col2:
                    st.metric(t("search_retries"), qa_result.retries)

                # Sources
                if qa_result.sources:
                    with st.expander(t("search_sources")):
                        for i, src in enumerate(qa_result.sources, 1):
                            st.markdown(f"**Source {i}** (score: {src.score:.3f})")
                            st.text(src.chunk.content[:300])
                            st.divider()

            except Exception as e:
                st.error(f"{t('error')}: {e}")


# --- Tab: Knowledge Graph ---

def tab_kg(t):
    st.header(t("kg_title"))

    if not st.session_state.get("kg_enabled", True):
        st.info(t("kg_no_data"))
        return

    try:
        kg = get_kg_client()
    except Exception as e:
        st.error(f"{t('error')}: {e}")
        return

    tab_entities, tab_rels, tab_temporal = st.tabs([
        t("kg_entities_title"),
        t("kg_relationships_title"),
        t("kg_temporal_title"),
    ])

    with tab_entities:
        entities = kg.get_entities(limit=100)
        if entities:
            st.dataframe(
                [
                    {
                        t("kg_entity_name"): e.get("name") or e.get("id", "?"),
                        t("kg_entity_type"): e.get("entity_type") or "unknown",
                        t("kg_entity_summary"): (e.get("summary") or "")[:100],
                    }
                    for e in entities
                ],
                width="stretch",
            )
        else:
            st.info(t("kg_no_data"))

    with tab_rels:
        rels = kg.get_relationships(limit=100)
        if rels:
            st.dataframe(
                [
                    {
                        t("kg_source"): r.get("source") or "?",
                        "→": r.get("rel_type") or "",
                        t("kg_target"): r.get("target") or "?",
                        t("kg_fact"): (r.get("fact") or "")[:80],
                        t("kg_valid_at"): r.get("valid_at") or "",
                    }
                    for r in rels
                ],
                width="stretch",
            )
        else:
            st.info(t("kg_no_data"))

    with tab_temporal:
        col1, col2 = st.columns(2)
        with col1:
            date_from = st.text_input(t("kg_date_from"), placeholder="2024-01-01")
        with col2:
            date_to = st.text_input(t("kg_date_to"), placeholder="2025-12-31")

        entity_filter = st.text_input(t("kg_entity_name"), key="temporal_entity")

        if st.button(t("kg_filter")):
            facts = kg.temporal_query(
                date_from=date_from,
                date_to=date_to,
                entity_name=entity_filter,
            )
            if facts:
                st.dataframe(
                    [
                        {
                            t("kg_entity_name"): f.entity_name or "?",
                            t("kg_fact"): f.content[:100],
                            t("kg_valid_at"): f.valid_at or "",
                        }
                        for f in facts
                    ],
                    width="stretch",
                )
            else:
                st.info(t("search_no_results"))


# --- Tab: Benchmark ---

def tab_benchmark(t):
    st.header(t("bench_title"))

    if st.button(t("bench_run"), type="primary"):
        from evaluation.benchmark import load_questions, run_benchmark

        store = get_vector_store()
        client = get_openai_client()
        retriever = get_retriever(store)
        kg = get_kg_client() if st.session_state.get("kg_enabled", True) else None
        hybrid_ret = get_hybrid_retriever(retriever, kg)

        questions = load_questions()

        results_all = {}

        # Run in 3 modes
        for mode_key, mode_label in [("hybrid", t("bench_mode_hybrid")), ("vector", t("bench_mode_vector")), ("agent", t("bench_mode_agent"))]:
            with st.spinner(f"{t('bench_running')} ({mode_label})"):
                if mode_key == "agent":
                    agent = get_hybrid_agent(hybrid_ret, kg)
                    ask_fn = agent.run
                else:
                    from generation.hybrid_generator import hybrid_answer
                    from functools import partial
                    ask_fn = partial(hybrid_answer, hybrid_retriever=hybrid_ret, openai_client=client, mode=mode_key)

                bench = run_benchmark(ask_fn, client, questions)
                results_all[mode_label] = bench

        # Summary table
        st.subheader(t("bench_accuracy"))
        cols = st.columns(len(results_all))
        for col, (label, bench) in zip(cols, results_all.items()):
            with col:
                st.metric(label, f"{bench['correct']}/{bench['total']} ({bench['accuracy']:.0%})")

        # Detailed per-question table
        for label, bench in results_all.items():
            with st.expander(label):
                st.dataframe(
                    [
                        {
                            "#": r["id"],
                            t("bench_question"): r["question"][:60],
                            t("bench_result"): t("bench_pass") if r["correct"] else t("bench_fail"),
                            t("search_confidence"): f"{r['confidence']:.2f}",
                            t("search_retries"): r["retries"],
                        }
                        for r in bench["results"]
                    ],
                    width="stretch",
                )


# --- Tab: Settings ---

def tab_settings(t):
    st.header(t("settings_title"))

    # Show config
    with st.expander(t("settings_config"), expanded=True):
        st.json({
            "openai_api_key": "***" + (settings.openai_api_key[-4:] if settings.openai_api_key else ""),
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
            "neo4j_uri": settings.neo4j_uri,
            "chunk_size": settings.chunk_size,
            "kg_enabled": settings.kg_enabled,
            "top_k_retrieval": settings.top_k_retrieval,
            "top_k_rerank": settings.top_k_rerank,
            "relevance_threshold": settings.relevance_threshold,
            "max_retries": settings.max_retries,
        })

    st.divider()

    # Re-init index
    if st.button(t("settings_reinit")):
        try:
            store = get_vector_store()
            store.init_index()
            st.success(t("settings_reinit_done"))
        except Exception as e:
            st.error(f"{t('error')}: {e}")

    st.divider()

    # Clear vector store
    st.subheader(t("settings_clear_vector"))
    confirm_vec = st.text_input(t("settings_confirm_delete"), key="confirm_vec")
    if st.button(t("settings_clear_vector"), key="btn_clear_vec") and confirm_vec == "DELETE":
        try:
            store = get_vector_store()
            count = store.delete_all()
            st.success(f"{t('settings_cleared')}: {count} chunks")
        except Exception as e:
            st.error(f"{t('error')}: {e}")


# --- Main ---

def main():
    # Init session state
    if "lang" not in st.session_state:
        st.session_state.lang = "en"

    t = get_translator(st.session_state.lang)
    render_sidebar(t)

    # Tabs
    tabs = st.tabs([
        t("tab_ingest"),
        t("tab_search"),
        t("tab_kg"),
        t("tab_benchmark"),
        t("tab_settings"),
    ])

    with tabs[0]:
        tab_ingest(t)
    with tabs[1]:
        tab_search(t)
    with tabs[2]:
        tab_kg(t)
    with tabs[3]:
        tab_benchmark(t)
    with tabs[4]:
        tab_settings(t)


if __name__ == "__main__":
    main()
