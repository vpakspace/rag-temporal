"""Internationalization for RAG + Temporal unified UI (EN/RU)."""

from __future__ import annotations

from typing import Callable

TRANSLATIONS: dict[str, dict[str, str]] = {
    # App
    "app_title": {"en": "RAG + Temporal Knowledge Base", "ru": "RAG + Темпоральная база знаний"},
    "app_subtitle": {"en": "Hybrid Vector + Knowledge Graph System", "ru": "Гибридная система Vector + Knowledge Graph"},

    # Sidebar
    "language": {"en": "Language", "ru": "Язык"},
    "gpu_acceleration": {"en": "GPU Acceleration (Docling)", "ru": "GPU ускорение (Docling)"},
    "kg_enabled": {"en": "Knowledge Graph enabled", "ru": "Knowledge Graph включён"},
    "stats_title": {"en": "Statistics", "ru": "Статистика"},
    "vector_chunks": {"en": "Vector chunks", "ru": "Векторных чанков"},
    "kg_entities": {"en": "KG entities", "ru": "Сущностей в KG"},

    # Tabs
    "tab_ingest": {"en": "Ingest", "ru": "Загрузка"},
    "tab_search": {"en": "Search & Q&A", "ru": "Поиск и Q&A"},
    "tab_kg": {"en": "Knowledge Graph", "ru": "Knowledge Graph"},
    "tab_benchmark": {"en": "Benchmark", "ru": "Бенчмарк"},
    "tab_settings": {"en": "Settings", "ru": "Настройки"},

    # Ingest tab
    "ingest_title": {"en": "Document Ingestion", "ru": "Загрузка документов"},
    "ingest_upload": {"en": "Upload file", "ru": "Загрузить файл"},
    "ingest_file_path": {"en": "Or enter file path", "ru": "Или введите путь к файлу"},
    "ingest_skip_enrichment": {"en": "Skip enrichment (faster)", "ru": "Пропустить обогащение (быстрее)"},
    "ingest_skip_kg": {"en": "Skip Knowledge Graph", "ru": "Пропустить Knowledge Graph"},
    "ingest_button": {"en": "Ingest", "ru": "Загрузить"},
    "ingest_processing": {"en": "Processing document...", "ru": "Обработка документа..."},
    "ingest_rag_track": {"en": "RAG Track", "ru": "RAG трек"},
    "ingest_kg_track": {"en": "KG Track", "ru": "KG трек"},
    "ingest_chunks_stored": {"en": "chunks stored", "ru": "чанков сохранено"},
    "ingest_episodes_ingested": {"en": "episodes ingested", "ru": "эпизодов загружено"},
    "ingest_success": {"en": "Ingestion complete!", "ru": "Загрузка завершена!"},
    "ingest_error": {"en": "Ingestion error", "ru": "Ошибка загрузки"},
    "ingest_supported": {"en": "Supported: TXT, PDF, DOCX, PPTX, XLSX, HTML", "ru": "Поддерживается: TXT, PDF, DOCX, PPTX, XLSX, HTML"},

    # Search tab
    "search_title": {"en": "Search & Question Answering", "ru": "Поиск и ответы на вопросы"},
    "search_query": {"en": "Enter your question", "ru": "Введите ваш вопрос"},
    "search_mode": {"en": "Search mode", "ru": "Режим поиска"},
    "search_mode_hybrid": {"en": "Hybrid (Vector + KG)", "ru": "Гибридный (Vector + KG)"},
    "search_mode_vector": {"en": "Vector only", "ru": "Только Vector"},
    "search_mode_kg": {"en": "KG only", "ru": "Только KG"},
    "search_mode_agent": {"en": "Agent (auto-route)", "ru": "Агент (авто-маршрутизация)"},
    "search_button": {"en": "Search", "ru": "Искать"},
    "search_thinking": {"en": "Thinking...", "ru": "Думаю..."},
    "search_answer": {"en": "Answer", "ru": "Ответ"},
    "search_confidence": {"en": "Confidence", "ru": "Уверенность"},
    "search_retries": {"en": "Retries", "ru": "Повторных попыток"},
    "search_sources": {"en": "Sources", "ru": "Источники"},
    "search_no_results": {"en": "No results found", "ru": "Результатов не найдено"},

    # KG tab
    "kg_title": {"en": "Knowledge Graph Explorer", "ru": "Обозреватель Knowledge Graph"},
    "kg_entities_title": {"en": "Entities", "ru": "Сущности"},
    "kg_relationships_title": {"en": "Relationships", "ru": "Связи"},
    "kg_temporal_title": {"en": "Temporal Facts", "ru": "Темпоральные факты"},
    "kg_no_data": {"en": "No knowledge graph data yet. Ingest documents first.", "ru": "Данных в KG пока нет. Сначала загрузите документы."},
    "kg_entity_name": {"en": "Entity name", "ru": "Имя сущности"},
    "kg_entity_type": {"en": "Type", "ru": "Тип"},
    "kg_entity_summary": {"en": "Summary", "ru": "Описание"},
    "kg_source": {"en": "Source", "ru": "Источник"},
    "kg_target": {"en": "Target", "ru": "Цель"},
    "kg_fact": {"en": "Fact", "ru": "Факт"},
    "kg_valid_at": {"en": "Valid at", "ru": "Актуально на"},
    "kg_date_from": {"en": "From date", "ru": "С даты"},
    "kg_date_to": {"en": "To date", "ru": "По дату"},
    "kg_filter": {"en": "Filter", "ru": "Фильтровать"},

    # Benchmark tab
    "bench_title": {"en": "Benchmark: Hybrid vs Vector-only", "ru": "Бенчмарк: Гибрид vs Только Vector"},
    "bench_run": {"en": "Run Benchmark", "ru": "Запустить бенчмарк"},
    "bench_running": {"en": "Running benchmark...", "ru": "Выполняю бенчмарк..."},
    "bench_mode_hybrid": {"en": "Hybrid", "ru": "Гибрид"},
    "bench_mode_vector": {"en": "Vector-only", "ru": "Только Vector"},
    "bench_mode_agent": {"en": "Agent", "ru": "Агент"},
    "bench_accuracy": {"en": "Accuracy", "ru": "Точность"},
    "bench_question": {"en": "Question", "ru": "Вопрос"},
    "bench_expected": {"en": "Expected", "ru": "Ожидалось"},
    "bench_actual": {"en": "Actual", "ru": "Получено"},
    "bench_result": {"en": "Result", "ru": "Результат"},
    "bench_pass": {"en": "PASS", "ru": "ВЕРНО"},
    "bench_fail": {"en": "FAIL", "ru": "НЕВЕРНО"},

    # Settings tab
    "settings_title": {"en": "Settings & Database", "ru": "Настройки и база данных"},
    "settings_config": {"en": "Current Configuration", "ru": "Текущая конфигурация"},
    "settings_clear_vector": {"en": "Clear Vector Store", "ru": "Очистить Vector Store"},
    "settings_clear_all": {"en": "Clear All Data", "ru": "Очистить все данные"},
    "settings_confirm_delete": {"en": "Type DELETE to confirm", "ru": "Введите DELETE для подтверждения"},
    "settings_cleared": {"en": "Data cleared successfully", "ru": "Данные очищены"},
    "settings_reinit": {"en": "Re-initialize Index", "ru": "Переинициализировать индекс"},
    "settings_reinit_done": {"en": "Index re-initialized", "ru": "Индекс переинициализирован"},

    # Common
    "loading": {"en": "Loading...", "ru": "Загрузка..."},
    "error": {"en": "Error", "ru": "Ошибка"},
    "success": {"en": "Success", "ru": "Успешно"},
    "warning": {"en": "Warning", "ru": "Внимание"},
}


def get_translator(lang: str = "en") -> Callable[..., str]:
    """Return a translator function for the given language.

    Usage:
        t = get_translator("ru")
        t("app_title")  # "RAG + Темпоральная база знаний"
        t("missing_key")  # "missing_key" (fallback to key name)
    """

    def t(key: str, **kwargs) -> str:
        entry = TRANSLATIONS.get(key)
        if entry is None:
            return key
        text = entry.get(lang, entry.get("en", key))
        if kwargs:
            try:
                return text.format(**kwargs)
            except (KeyError, IndexError):
                return text
        return text

    return t
