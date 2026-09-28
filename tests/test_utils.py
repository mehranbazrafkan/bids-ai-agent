"""Tests for the small prompt-formatting utilities."""

from utils import compact_json, json_to_llm_text


def test_compact_json_recurses_dicts_and_lists():
    obj = {
        "severity": "err",
        "field": "age",
        "lines": [2, 3],
        "nested": {"a": "b", "ok": True},
    }
    out = compact_json(obj)
    assert "severity:err" in out
    assert "field:age" in out
    assert "lines:[2,3]" in out
    assert "nested:a:b" in out


def test_compact_json_handles_scalars():
    assert compact_json("txt") == "txt"
    assert compact_json(42) == "42"
    assert compact_json(["x", 1]) == "[x,1]"


def test_json_to_llm_text_indents_nested_structures():
    obj = {"a": {"b": 1}, "c": ["x", "y"]}
    out = json_to_llm_text(obj)
    assert out.startswith("a:")
    assert " b=1" in out
    assert "c:" in out
    assert "[x,y]" in out