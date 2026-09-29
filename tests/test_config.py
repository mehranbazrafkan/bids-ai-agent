"""Tests for the configuration module."""

import json
import os

import pytest

import config as config_module
from config import AgentConfig, RetrievalConfig, load_config


def test_defaults():
    cfg = load_config(config_file="__no_such_config__.json")  # force defaults
    assert cfg.llm.model_name == "Qwen/Qwen3-1.7B"
    assert cfg.llm.max_new_tokens == 250
    assert cfg.retrieval.knowledge_base_dir == "./knowledge-base"
    assert cfg.retrieval.min_best_score == 3.0
    assert cfg.retrieval.lazy_raw_content is True


def test_json_file_overrides_defaults(tmp_path):
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "llm": {"model_name": "Qwen/Qwen3-4B", "max_new_tokens": 512},
        "retrieval": {"min_best_score": 5.0, "default_top_k": 5},
    }), encoding="utf-8")
    cfg = load_config(config_file=cfg_file)
    assert cfg.llm.model_name == "Qwen/Qwen3-4B"
    assert cfg.llm.max_new_tokens == 512
    assert cfg.retrieval.min_best_score == 5.0
    assert cfg.retrieval.default_top_k == 5
    # untouched defaults survive
    assert cfg.llm.temperature == 0.75
    assert cfg.retrieval.knowledge_base_dir == "./knowledge-base"


def test_env_overrides_file_and_defaults(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"llm": {"max_new_tokens": 256}}), encoding="utf-8")

    monkeypatch.setenv("BIDS_AGENT_LLM_MAX_NEW_TOKENS", "900")
    monkeypatch.setenv("BIDS_AGENT_RETRIEVER_KNOWLEDGE_BASE_DIR", "./my-kb")
    monkeypatch.setenv("BIDS_AGENT_LLM_ENABLE_THINKING", "false")

    cfg = load_config(config_file=cfg_file)
    assert cfg.llm.max_new_tokens == 900       # env beats file
    assert cfg.retrieval.knowledge_base_dir == "./my-kb"
    assert cfg.llm.enable_thinking is False
    assert cfg.llm.model_name == "Qwen/Qwen3-1.7B"  # unaffected default


def test_unknown_keys_ignored(tmp_path):
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"llm": {"bogus_key": 1}}), encoding="utf-8")
    cfg = load_config(config_file=cfg_file)
    assert not hasattr(cfg.llm, "bogus_key")


def test_missing_config_file_falls_back_to_defaults(tmp_path):
    cfg = load_config(config_file=tmp_path / "nope.json")
    assert cfg.llm.model_name == "Qwen/Qwen3-1.7B"


def test_module_singleton_is_agent_config():
    assert isinstance(config_module.CONFIG, AgentConfig)
    assert isinstance(config_module.CONFIG.retrieval, RetrievalConfig)


def test_load_in_4bit_default():
    cfg = load_config(config_file="__no_such_config__.json")
    assert cfg.llm.load_in_4bit is False


def test_load_in_4bit_from_env(monkeypatch):
    monkeypatch.setenv("BIDS_AGENT_LLM_LOAD_IN_4BIT", "true")
    cfg = load_config(config_file="__no_such_config__.json")
    assert cfg.llm.load_in_4bit is True


def test_load_in_4bit_from_json(tmp_path):
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"llm": {"load_in_4bit": True}}), encoding="utf-8")
    cfg = load_config(config_file=cfg_file)
    assert cfg.llm.load_in_4bit is True