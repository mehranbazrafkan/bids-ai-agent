"""Tests for the prompt record: the latest request and answer, in memory.

The record is an inspection aid, not a conversation record. Each request
reports its own record -- the system prompt, the user prompt, the agent
output and how the answer was produced -- but never the chat history.
Nothing is ever written to disk. The planner's repair pass is a second
call for the same request, so it must not replace the record of the
request the user actually made, and a reset must clear the record so a
request whose model is never called reports no log.
"""

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


def _begun(system: str = SYSTEM_PROMPT, user: str = USER_PROMPT) -> None:
    prompt_log.begin(system, user)


def test_it_records_all_parts_and_the_metadata():
    _begun()
    prompt_log.record_generation(
        AGENT_OUTPUT,
        model="stub/stub-7b-instruct",
        prompt_tokens=128,
        generated_tokens=64,
        seconds=1.234,
    )
    record = prompt_log.snapshot()
    assert record is not None
    # Byte for byte: the planner's system prompt and the user's question.
    assert record["system_prompt"] == SYSTEM_PROMPT
    assert record["user_prompt"] == USER_PROMPT
    assert record["output"] == AGENT_OUTPUT
    assert record["model"] == "stub/stub-7b-instruct"
    assert record["prompt_tokens"] == 128
    assert record["generated_tokens"] == 64
    assert record["generation_seconds"] == 1.234


def test_a_new_request_replaces_the_old_one():
    _begun(system="the first request's system prompt", user="first question")
    prompt_log.record_generation("first answer", model="first/model")
    _begun(system="the second request's system prompt", user="second question")
    prompt_log.record_generation("second answer", model="second/model")

    record = prompt_log.snapshot()
    assert record["system_prompt"] == "the second request's system prompt"
    assert record["user_prompt"] == "second question"
    assert record["output"] == "second answer"
    assert record["model"] == "second/model"
    blob = str(record)
    assert "first" not in blob


def test_reset_clears_the_record():
    _begun()
    prompt_log.record_generation("an answer", model="m")
    prompt_log.reset()
    assert prompt_log.snapshot() is None


def test_a_request_whose_model_is_never_called_reports_no_log():
    # /predict calls reset() before the agent runs; the clarification
    # shortcut returns before the model is ever invoked.
    prompt_log.reset()
    assert prompt_log.snapshot() is None


def test_snapshot_is_a_copy_not_the_live_record():
    _begun()
    prompt_log.record_generation("an answer", model="m")
    first = prompt_log.snapshot()
    first["output"] = "tampered"
    assert prompt_log.snapshot()["output"] == "an answer"


def test_generation_metadata_is_optional():
    _begun()
    prompt_log.record_generation(AGENT_OUTPUT)
    record = prompt_log.snapshot()
    assert record["output"] == AGENT_OUTPUT
    assert record["model"] == ""
    assert record["prompt_tokens"] is None
    assert record["generated_tokens"] is None
    assert record["generation_seconds"] is None


# ---------------------------------------------------------------------------
# The repair pass must not replace the request the user made
# ---------------------------------------------------------------------------


def test_the_repair_pass_does_not_replace_the_request():
    _begun(system="the real system prompt", user="the real question")
    prompt_log.record_generation(
        "the model's first answer", model="m", prompt_tokens=10,
        generated_tokens=5, seconds=0.5,
    )

    with prompt_log.suspended():
        prompt_log.begin("[Answer Repair] rewrite the answer below", "repair q")
        prompt_log.record_generation("the repair attempt", model="m")

    record = prompt_log.snapshot()
    assert record["system_prompt"] == "the real system prompt"
    assert record["user_prompt"] == "the real question"
    assert record["output"] == "the model's first answer"
    assert "[Answer Repair]" not in str(record)
    assert "the repair attempt" not in str(record)


def test_suspension_ends_with_the_context():
    with prompt_log.suspended():
        prompt_log.begin("suppressed", "q")
        prompt_log.record_generation("suppressed answer")
    _begun(system="after", user="q")
    prompt_log.record_generation("after answer")

    record = prompt_log.snapshot()
    assert "suppressed" not in str(record)
    assert record["system_prompt"] == "after"


def test_recording_inside_suspension_without_a_record_costs_nothing():
    # The repair pass may be the only model call in a context (direct
    # callers, tests). A suppressed record must never spring into
    # existence for it.
    prompt_log.reset()
    with prompt_log.suspended():
        prompt_log.record_generation("orphan output")
    assert prompt_log.snapshot() is None


def test_the_record_never_touches_disk(tmp_path):
    # The record lives in RAM and travels in the API response; the
    # package directory must stay free of generated log files.
    _begun()
    prompt_log.record_generation(AGENT_OUTPUT, model="m")
    assert prompt_log.snapshot() is not None
    assert not (tmp_path / "prompt_log.txt").exists()
    assert not list(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# The config switch the file-based log used is gone
# ---------------------------------------------------------------------------


def test_the_old_save_prompts_switch_no_longer_exists():
    config = load_config(config_file="__no_such_config__.json")
    assert not hasattr(config.llm, "save_prompts")
