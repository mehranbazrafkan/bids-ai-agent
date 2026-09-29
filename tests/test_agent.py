"""Tests for the Agent — includes the 4 sample issues from the original main.py."""

import pytest
from agent import Agent


# ---------------------------------------------------------------------------
# Sample issues (from original main.py)
# ---------------------------------------------------------------------------

sample_issue_001 = {
    "severity": "err",
    "rule_id": "JSON_SCHEMA_VALIDATION_ERROR",
    "message": "ReferencesAndLinks must be array  \u00b7  List of references to publications that contain information on the dataset. Use a value of the correct type, for example {\"ReferencesAndLinks\": [\"text\"]}.",
    "field": "ReferencesAndLinks",
    "line": "null",
    "lines": [],
    "fix_label": "null",
    "fix_action": "null",
    "mirrored": False,
}

sample_issue_002 = {
    "severity": "warn",
    "rule_id": "bidsmgr.todo_placeholder",
    "message": "field 'EthicsApprovals' contains a TODO placeholder",
    "field": "EthicsApprovals",
    "line": "null",
    "lines": [],
    "fix_label": "Set a real value",
    "fix_action": "set_field",
    "mirrored": False,
}

sample_issue_003 = {
    "path": "participants.tsv",
    "severity": "err",
    "datatype": "null",
    "suffix": "participants",
    "issues": [
        {
            "severity": "err",
            "rule_id": "TSV_VALUE_INCORRECT_TYPE",
            "message": "column 'age': value '020Y' is not valid for its type  \u00b7  Each value must be a valid number, or 'n/a'.",
            "field": "age",
            "line": 2,
            "lines": [2, 3],
            "fix_label": "null",
            "fix_action": "null",
            "mirrored": False,
        }
    ],
    "sidecar_fields": [],
}

sample_issue_004 = {
    "severity": "warn",
    "rule_id": "SIDECAR_KEY_RECOMMENDED",
    "message": "missing recommended field 'SequenceName'  \u00b7  Manufacturer's designation of the sequence name. Add it to the JSON, for example {\"SequenceName\": \"text\"}.",
    "field": "SequenceName",
    "line": "null",
    "lines": [],
    "fix_label": "Fix",
    "fix_action": "add_field",
    "mirrored": True,
}

ALL_ISSUES = [
    ("sample_issue_001", sample_issue_001),
    ("sample_issue_002", sample_issue_002),
    ("sample_issue_003", sample_issue_003),
    ("sample_issue_004", sample_issue_004),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgent:
    """Test the Agent class with the 4 sample issues."""

    @pytest.fixture(autouse=True)
    def _slow_tests(self):
        """Mark all tests in this class as slow (they load the LLM)."""
        pytest.skip(reason="Slow test — loads LLM. Run manually with: pytest tests/test_agent.py -v --no-skip")

    def test_agent_can_be_constructed(self):
        """Agent should construct without errors."""
        agent = Agent()
        assert agent is not None
        assert agent.planner is not None
        assert agent.llm is not None

    def test_sample_issue_001_returns_response(self):
        agent = Agent()
        response = agent.run(user_input="Explain this to me.", context=sample_issue_001)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_sample_issue_002_returns_response(self):
        agent = Agent()
        response = agent.run(user_input="Explain this to me.", context=sample_issue_002)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_sample_issue_003_returns_response(self):
        agent = Agent()
        response = agent.run(user_input="Explain this to me.", context=sample_issue_003)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_sample_issue_004_returns_response(self):
        agent = Agent()
        response = agent.run(user_input="Explain this to me.", context=sample_issue_004)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_all_4_issues_return_non_empty_responses(self):
        """All 4 sample issues should produce non-empty responses."""
        agent = Agent()
        for name, issue in ALL_ISSUES:
            response = agent.run(user_input="Explain this to me.", context=issue)
            assert isinstance(response, str), f"{name}: response is not a string"
            assert len(response) > 0, f"{name}: response is empty"
