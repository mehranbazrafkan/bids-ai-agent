"""Slow tests for the Agent — the 4 sample issues from the original main.py.

Marked ``slow`` (excluded by default, see pytest.ini): they build the real
agent, which loads Laya and the LLM. Run explicitly with::

    pytest tests/test_agent.py -m slow -v
"""

import pytest

from agent import Agent
from planner import PlannerResult

pytestmark = pytest.mark.slow


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

    def test_agent_can_be_constructed(self):
        """Agent should construct without errors (no model load yet)."""
        agent = Agent()
        assert agent is not None
        assert agent.planner is not None
        assert agent.llm is not None

    def test_sample_issue_returns_result(self):
        agent = Agent()
        for name, issue in ALL_ISSUES:
            result = agent.run(user_input="Explain this to me.", context=issue)
            assert isinstance(result, PlannerResult), name
            assert isinstance(result.response, str) and result.response, name
            # explanation-only contract, even end-to-end
            assert result.tool_used is None, name
            assert result.intent in ("explain", "fix", "unclear"), name

    def test_action_request_never_claims_execution(self):
        agent = Agent()
        result = agent.run(
            user_input="Fix this error in my dataset.",
            context=sample_issue_001,
        )
        assert result.tool_used is None
        assert result.execution_status == "not_available"
        lowered = result.response.lower()
        for false_claim in ("i fixed", "i repaired", "i renamed",
                            "validation completed successfully",
                            "successfully repaired", "i ran the validator"):
            assert false_claim not in lowered
