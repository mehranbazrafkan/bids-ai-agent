"""Tests for the temporary prompt dump: the latest system prompt only.

The file is an inspection aid, not a conversation record. Each request
overwrites it, only the system prompt is written (no chat turns, no
question, no rendered text), it must be switch-off-able, and it must
never be able to cost a user their answer.
"""

from pathlib import Path

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


def test_it_writes_exactly_the_system_prompt(tmp_path):
    target = tmp_path / "prompt_log.txt"
    written = prompt_log.save(SYSTEM_PROMPT, path=target)
    assert written == target
    # Byte for byte: nothing prepended, nothing appended, no header.
    assert target.read_text(encoding="utf-8") == SYSTEM_PROMPT


def test_a_new_request_replaces_the_old_one(tmp_path):
    target = tmp_path / "prompt_log.txt"
    prompt_log.save("the first request's system prompt", path=target)
    prompt_log.save("the second request's system prompt", path=target)
    text = target.read_text(encoding="utf-8")
    assert text == "the second request's system prompt"
    assert "first request" not in text


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
    assert prompt_log.save(SYSTEM_PROMPT, path=target) is None
    assert not target.exists()


def test_a_path_that_cannot_be_written_costs_nothing(tmp_path):
    # ``tmp_path`` is a directory, so opening it for writing fails -- and
    # the answer must still come back.
    assert prompt_log.save(SYSTEM_PROMPT, path=tmp_path) is None
