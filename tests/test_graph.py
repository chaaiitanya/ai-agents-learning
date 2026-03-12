import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.messages import AIMessage

from agents.state import AgentState
from agents.reviewer import _parse_decision


def test_parse_decision_approve():
    text = "DECISION: approve\n\nREASONING:\nCode is correct."
    assert _parse_decision(text) == "approve"


def test_parse_decision_revise():
    text = "DECISION: revise\n\nFEEDBACK:\nMissing error handling."
    assert _parse_decision(text) == "revise"


def test_parse_decision_reject():
    text = "DECISION: reject\n\nREASONING:\nFundamentally wrong approach."
    assert _parse_decision(text) == "reject"


def test_parse_decision_fallback():
    text = "No clear decision in this text."
    assert _parse_decision(text) == "revise"


@pytest.mark.asyncio
async def test_researcher_node():
    mock_response = MagicMock()
    mock_response.content = "Research findings: ..."

    with patch("agents.researcher.get_llm") as mock_llm, \
         patch("agents.researcher.VectorStore") as mock_vs, \
         patch("agents.researcher.RAGRetriever") as mock_rag:

        mock_llm.return_value.invoke = MagicMock(return_value=mock_response)
        mock_rag.return_value.retrieve_as_text = MagicMock(return_value="context")

        from agents.researcher import researcher_node

        state: AgentState = {
            "task": "Build a REST API",
            "session_id": "test-123",
            "messages": [],
            "research_output": "",
            "code_output": "",
            "review_output": "",
            "current_agent": "researcher",
            "review_cycle": 0,
            "max_review_cycles": 3,
            "review_decision": "",
            "awaiting_human": False,
            "human_feedback": "",
            "final_output": "",
            "error": "",
        }

        result = researcher_node(state)
        assert result["research_output"] == "Research findings: ..."
        assert result["current_agent"] == "coder"
