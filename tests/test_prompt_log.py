"""Tests for the temporary prompt dump: the latest request and answer.

The file is an inspection aid, not a conversation record. Each request
overwrites it with three labelled parts -- the system prompt, the user
prompt and the agent output -- but never the chat history; it must be
switch-off-able, and it must never be able to cost a user their answer.
The planner's repair pass is a second call for the same request, so it must
not replace the record of the request the user actually made. When the
quality gate changed the answer, the text that was actually returned is
added as a fourth part, so a repair can be read instead of inferred.
"""

from pathlib import Path

import pytest

import prompt_log
from config import load_config

SYSTEM_PROMPT = """System: You are a BIDS assistant.

[Issue / Application Context]
rule=entity.required field=RepetitionTime severity=err

[Dataset Structure — condensed tree]
dataset/
 sub-01/
  func/
   sub-01_task-rest_bold.json  # sidecar
"""

USER_PROMPT = "Why is RepetitionTime missing?"
AGENT_OUTPUT = "The bold sidecar is missing the required RepetitionTime field."


def _rendered(system: str, user: str, output: str) -> str:
    return (
        f"[System prompt]\n{system}\n"
        "\n"
        f"[User prompt]\n{user}\n"
        "\n"
        f"[Agent output]\n{output}\n"
    )


def test_it_writes_all_three_parts(tmp_path):
    target = tmp_path / "prompt_log.txt"
    written = prompt_log.save(
        SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
        output=AGENT_OUTPUT,
        path=target,
    )
    assert written == target
    # Byte for byte: the three labelled parts, in that order, nothing else.
    text = target.read_text(encoding="utf-8")
    assert text == _rendered(SYSTEM_PROMPT, USER_PROMPT, AGENT_OUTPUT)
    # The user's question and the model's answer each appear verbatim.
    assert USER_PROMPT in text
    assert AGENT_OUTPUT in text


def test_a_new_request_replaces_the_old_one(tmp_path):
    target = tmp_path / "prompt_log.txt"
    prompt_log.save(
        "the first request's system prompt",
        user_prompt="first question",
        output="first answer",
        path=target,
    )
    prompt_log.save(
        "the second request's system prompt",
        user_prompt="second question",
        output="second answer",
        path=target,
    )
    text = target.read_text(encoding="utf-8")
    assert text == _rendered(
        "the second request's system prompt",
        "second question",
        "second answer",
    )
    assert "first" not in text


def test_the_file_sits_next_to_main_py_whatever_the_cwd(tmp_path, monkeypatch):
    # The service spawns the agent with ai_agent/ as its cwd, but running
    # ``python main.py`` from anywhere must land in the same place, so the
    # path is anchored on the module rather than on the working directory.
    monkeypatch.chdir(tmp_path)
    assert prompt_log.DEFAULT_PATH.name == "prompt_log.txt"
    assert prompt_log.DEFAULT_PATH.parent == Path(
        prompt_log.__file__
    ).resolve().parent


def test_the_switch_defaults_on_and_the_env_var_turns_it_off(monkeypatch):
    defaults = load_config(config_file="__no_such_config__.json")
    assert defaults.llm.save_prompts is True

    monkeypatch.setenv("BIDS_AGENT_LLM_SAVE_PROMPTS", "false")
    off = load_config(config_file="__no_such_config__.json")
    assert off.llm.save_prompts is False


def test_the_switch_turns_writing_off(tmp_path, monkeypatch):
    monkeypatch.setattr(prompt_log.CONFIG.llm, "save_prompts", False)
    target = tmp_path / "prompt_log.txt"
    assert (
        prompt_log.save(
            SYSTEM_PROMPT,
            user_prompt=USER_PROMPT,
            output=AGENT_OUTPUT,
            path=target,
        )
        is None
    )
    assert not target.exists()


def test_a_path_that_cannot_be_written_costs_nothing(tmp_path):
    # ``tmp_path`` is a directory, so opening it for writing fails -- and
    # the answer must still come back.
    assert (
        prompt_log.save(
            SYSTEM_PROMPT,
            user_prompt=USER_PROMPT,
            output=AGENT_OUTPUT,
            path=tmp_path,
        )
        is None
    )


# ---------------------------------------------------------------------------
# The repair pass must not replace the request the user made
# ---------------------------------------------------------------------------


def test_the_repair_pass_does_not_replace_the_request(tmp_path):
    target = tmp_path / "prompt_log.txt"
    prompt_log.save(
        "the real system prompt",
        user_prompt="the real question",
        output="the model's first answer",
        path=target,
    )

    with prompt_log.suspended():
        assert prompt_log.save(
            "[Answer Repair] rewrite the answer below",
            user_prompt="Original question ... Answer to repair ...",
            output="the repair attempt",
            path=target,
        ) is None

    text = target.read_text(encoding="utf-8")
    assert text == _rendered(
        "the real system prompt",
        "the real question",
        "the model's first answer",
    )
    assert "[Answer Repair]" not in text
    assert "the repair attempt" not in text


def test_suspension_ends_with_the_context(tmp_path):
    target = tmp_path / "prompt_log.txt"

    with prompt_log.suspended():
        prompt_log.save("suppressed", user_prompt="q", output="a", path=target)
    prompt_log.save("after", user_prompt="q", output="a", path=target)

    text = target.read_text(encoding="utf-8")
    assert "suppressed" not in text
    assert "after" in text


# ---------------------------------------------------------------------------
# The answer the caller actually received (after the quality gate)
# ---------------------------------------------------------------------------

FINAL = "**Problem**\nThe field is missing.\n\nThe knowledge base does not document this rule."
GATE_NOTE = "quality gate: repaired (example_not_allowed)"


@pytest.fixture(autouse=True)
def _forget_the_previous_record(monkeypatch):
    monkeypatch.setattr(prompt_log, "_LAST", None)


def test_the_final_answer_is_added_after_the_agent_output(tmp_path):
    target = tmp_path / "prompt_log.txt"
    prompt_log.save(
        SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
        output=AGENT_OUTPUT,
        path=target,
    )
    written = prompt_log.annotate_final(FINAL, note=GATE_NOTE)
    assert written == target

    text = target.read_text(encoding="utf-8")
    # The model's own text is still there, unchanged, and the returned text
    # follows it under its own heading -- in that order.
    assert text == _rendered(SYSTEM_PROMPT, USER_PROMPT, AGENT_OUTPUT) + (
        f"\n[Final response]\n{GATE_NOTE}\n{FINAL}\n"
    )
    assert text.index("[Agent output]") < text.index("[Final response]")


def test_the_note_is_optional(tmp_path):
    target = tmp_path / "prompt_log.txt"
    prompt_log.save(
        SYSTEM_PROMPT, user_prompt=USER_PROMPT, output=AGENT_OUTPUT, path=target
    )
    prompt_log.annotate_final(FINAL)
    text = target.read_text(encoding="utf-8")
    assert text.endswith(f"\n[Final response]\n{FINAL}\n")


def test_a_failed_write_leaves_nothing_to_extend(tmp_path):
    target = tmp_path / "prompt_log.txt"
    prompt_log.save(
        SYSTEM_PROMPT, user_prompt=USER_PROMPT, output=AGENT_OUTPUT, path=target
    )
    before = target.read_text(encoding="utf-8")
    # The directory cannot be written: this request has no record, so the
    # previous request's record must not collect this request's answer.
    assert prompt_log.save(
        "another system prompt",
        user_prompt="another question",
        output="another answer",
        path=tmp_path,
    ) is None
    assert prompt_log.annotate_final(FINAL) is None
    assert target.read_text(encoding="utf-8") == before


def test_an_unchanged_answer_is_not_written_twice(tmp_path):
    target = tmp_path / "prompt_log.txt"
    prompt_log.save(
        SYSTEM_PROMPT, user_prompt=USER_PROMPT, output=AGENT_OUTPUT, path=target
    )
    assert prompt_log.annotate_final(AGENT_OUTPUT) is None
    # Three parts: the gate changed nothing, so there is nothing to compare.
    assert target.read_text(encoding="utf-8") == _rendered(
        SYSTEM_PROMPT, USER_PROMPT, AGENT_OUTPUT
    )


def test_annotating_without_a_record_does_nothing(tmp_path, monkeypatch):
    target = tmp_path / "prompt_log.txt"
    monkeypatch.setattr(prompt_log, "DEFAULT_PATH", target)
    assert prompt_log.annotate_final(FINAL) is None
    assert not target.exists()


def test_a_new_request_drops_the_previous_final_answer(tmp_path):
    target = tmp_path / "prompt_log.txt"
    prompt_log.save(
        SYSTEM_PROMPT, user_prompt=USER_PROMPT, output=AGENT_OUTPUT, path=target
    )
    prompt_log.annotate_final(FINAL)
    # A new request's save() rewrites the whole file: the old final answer
    # must not survive next to the new model output.
    prompt_log.save(
        "the next system prompt",
        user_prompt="the next question",
        output="the next answer",
        path=target,
    )
    text = target.read_text(encoding="utf-8")
    assert "[Final response]" not in text
    assert FINAL not in text


def test_the_final_answer_respects_the_switch(tmp_path, monkeypatch):
    target = tmp_path / "prompt_log.txt"
    prompt_log.save(
        SYSTEM_PROMPT, user_prompt=USER_PROMPT, output=AGENT_OUTPUT, path=target
    )
    monkeypatch.setattr(prompt_log.CONFIG.llm, "save_prompts", False)
    assert prompt_log.annotate_final(FINAL) is None
    assert target.read_text(encoding="utf-8") == _rendered(
        SYSTEM_PROMPT, USER_PROMPT, AGENT_OUTPUT
    )
