"""Tests for the planner module — intent detection and tool routing."""

from unittest.mock import MagicMock

from planner import Planner, _FIX_KEYWORDS


def _make_planner(laya_intent_return=None, laya_tool_return=None):
    """Build a Planner with mocked LLM, Laya, retriever, and tool registry."""
    llm = MagicMock()
    llm.generate_response.return_value = "placeholder"

    laya = MagicMock()
    laya.classify_intent.return_value = laya_intent_return
    laya.select_tool.return_value = laya_tool_return

    retriever = MagicMock()
    retriever.retrieve.return_value = "retrieved docs"
    retriever.retrieve_issue.return_value = "retrieved docs"

    tools = MagicMock()
    tools.tools = {"validate": MagicMock(), "repair_metadata": MagicMock(), "rename_subject": MagicMock()}
    tools.get_tool_schemas.return_value = []
    tools.execute.return_value = "tool ran"

    planner = Planner(llm=llm, retriever=retriever, tools=tools)
    planner.laya = laya  # Replace the real Laya engine with a mock
    return planner, llm, laya


# ------------------------------------------------------------------
# Intent classification
# ------------------------------------------------------------------


def test_laya_fix_intent():
    p, llm, laya = _make_planner(laya_intent_return="fix")
    assert p._decided_intent("rename the subject", "ctx") == "fix"
    laya.classify_intent.assert_called_once()


def test_laya_explain_intent():
    p, llm, laya = _make_planner(laya_intent_return="explain")
    assert p._decided_intent("what does this error mean", "ctx") == "explain"


def test_keyword_fallback_when_laya_unavailable():
    p, llm, laya = _make_planner(laya_intent_return=None)
    assert p._decided_intent("fix the file", "ctx") == "fix"
    assert p._decided_intent("resolve this issue", "ctx") == "fix"
    assert p._decided_intent("what is this", "ctx") == "explain"


def test_keyword_fix_keywords_are_complete():
    """Ensure all expected fix-triggering words are in _FIX_KEYWORDS."""
    expected = {"fix", "repair", "correct", "resolve", "clean", "rename"}
    assert expected == _FIX_KEYWORDS


# ------------------------------------------------------------------
# Tool routing
# ------------------------------------------------------------------


def test_call_tool_exact_match():
    p, llm, laya = _make_planner(laya_tool_return="repair_metadata")
    result = p._call_tool("repair_metadata", "ctx")
    assert "Successfully executed tool" in result
    p.tools.execute.assert_called_with("repair_metadata", context="ctx")


def test_call_tool_substring_fallback():
    p, llm, laya = _make_planner(laya_tool_return="repair_metadata")
    result = p._call_tool("I think you should repair_metadata now", "ctx")
    assert "Successfully executed tool" in result


def test_call_tool_no_match():
    p, llm, laya = _make_planner(laya_tool_return=None)
    result = p._call_tool(None, "ctx")
    assert "No matching tool" in result
    p.tools.execute.assert_not_called()


# ------------------------------------------------------------------
# execute() wiring
# ------------------------------------------------------------------


def test_execute_routes_to_explain():
    p, llm, laya = _make_planner(laya_intent_return="explain")
    result = p.execute("system", "what is this", {"path": "T1.json"})
    assert result == "placeholder"
    llm.generate_response.assert_called()


def test_execute_routes_to_fix():
    p, llm, laya = _make_planner(laya_intent_return="fix", laya_tool_return="validate")
    result = p.execute("system", "fix this", {"path": "T1.json"})
    assert result == "placeholder"
    laya.select_tool.assert_called()
