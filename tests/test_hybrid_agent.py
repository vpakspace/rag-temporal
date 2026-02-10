"""Tests for hybrid agent with 5 tools."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.hybrid_agent import AgentState, HybridAgent, _extract_entity


class TestAgentState:
    def test_default_state(self):
        state = AgentState()
        assert state.query == ""
        assert state.query_type == ""
        assert state.tool == ""
        assert state.results == []
        assert state.retries == 0

    def test_custom_state(self):
        state = AgentState(query="test", query_type="factual", tool="vector_search")
        assert state.query == "test"
        assert state.query_type == "factual"


class TestExtractEntity:
    def test_quoted_entity(self):
        assert _extract_entity('What happened to "LMCache"?') == "LMCache"

    def test_unquoted_query(self):
        result = _extract_entity("How does caching work?")
        assert result == "How does caching work?"


class TestHybridAgent:
    def _make_agent(self):
        """Create agent with mocked dependencies."""
        mock_retriever = MagicMock()
        mock_retriever.retriever = MagicMock()
        mock_retriever.retriever.vector_store = MagicMock()
        mock_retriever.kg_client = MagicMock()

        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "factual"
        mock_openai.chat.completions.create = MagicMock(return_value=mock_response)

        agent = HybridAgent(
            hybrid_retriever=mock_retriever,
            kg_client=mock_retriever.kg_client,
            openai_client=mock_openai,
        )
        return agent

    def test_classify_factual(self):
        agent = self._make_agent()
        state = AgentState(query="What is LMCache?")
        state = agent.classify_query(state)
        assert state.query_type == "factual"

    def test_classify_temporal(self):
        agent = self._make_agent()
        agent._openai.chat.completions.create.return_value.choices[0].message.content = "temporal"
        state = AgentState(query="When was LMCache created?")
        state = agent.classify_query(state)
        assert state.query_type == "temporal"

    def test_classify_entity(self):
        agent = self._make_agent()
        agent._openai.chat.completions.create.return_value.choices[0].message.content = "entity"
        state = AgentState(query="Tell me about LMCache")
        state = agent.classify_query(state)
        assert state.query_type == "entity"

    def test_classify_invalid_defaults_to_factual(self):
        agent = self._make_agent()
        agent._openai.chat.completions.create.return_value.choices[0].message.content = "unknown_type"
        state = AgentState(query="Test")
        state = agent.classify_query(state)
        assert state.query_type == "factual"

    def test_select_tool_mapping(self):
        agent = self._make_agent()

        mappings = {
            "factual": "focused_search",
            "overview": "full_document_read",
            "comparison": "focused_search",
            "temporal": "temporal_query",
            "entity": "knowledge_search",
        }
        for qtype, expected_tool in mappings.items():
            state = AgentState(query_type=qtype)
            state = agent.select_tool(state)
            assert state.tool == expected_tool, f"Expected {expected_tool} for {qtype}"

    def test_should_retry_false_for_focused(self):
        agent = self._make_agent()
        state = AgentState(
            relevance_score=1.0,
            retries=0,
            tool="focused_search",
        )
        assert not agent.should_retry(state)

    @patch("agent.hybrid_agent.settings")
    def test_should_retry_on_low_relevance(self, mock_settings):
        mock_settings.relevance_threshold = 3.0
        mock_settings.max_retries = 2

        agent = self._make_agent()
        state = AgentState(
            relevance_score=1.5,
            retries=0,
            tool="full_document_read",
        )
        assert agent.should_retry(state)
