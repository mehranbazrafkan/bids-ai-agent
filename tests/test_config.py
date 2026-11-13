"""Tests for the configuration module."""

import json
import os

import pytest

import config as config_module
from config import AgentConfig, RetrievalConfig, load_config, save_config


def test_defaults():
    cfg = load_config(config_file="__no_such_config__.json")  # force defaults
    assert cfg.llm.model_name == "Qwen/Qwen3-1.7B"
    assert cfg.llm.max_new_tokens == 512
    assert cfg.retrieval.knowledge_base_dir == "./knowledge-base"
    assert cfg.retrieval.min_best_score == 3.0
    assert cfg.retrieval.lazy_raw_content is True
    assert cfg.retrieval.retriever_type == "lexical"
    assert cfg.retrieval.embedding_model == "BAAI/bge-small-en-v1.5"
    assert cfg.intent.mode == "auto"
    assert cfg.intent.classifier == "llm"


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
    assert cfg.llm.temperature == 0.2
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


def test_save_config_round_trips(tmp_path):
    cfg = load_config(config_file=tmp_path / "missing.json")
    cfg.llm.quantization = "4bit"
    cfg.llm.load_in_4bit = True
    cfg.llm.max_new_tokens = 321
    cfg.retrieval.default_top_k = 2
    cfg.retrieval.retriever_type = "embedding"
    cfg.retrieval.embedding_model = "test/embedder"
    cfg.intent.mode = "fix"
    cfg.intent.classifier = "nli"
    cfg.quality.enabled = False

    path = save_config(cfg, config_file=tmp_path / "config.json")

    assert path.exists()
    assert not (tmp_path / "config.json.tmp").exists()  # atomic write cleaned up
    reloaded = load_config(config_file=path)
    assert reloaded.llm.quantization == "4bit"
    assert reloaded.llm.load_in_4bit is True
    assert reloaded.llm.max_new_tokens == 321
    assert reloaded.retrieval.default_top_k == 2
    assert reloaded.retrieval.retriever_type == "embedding"
    assert reloaded.retrieval.embedding_model == "test/embedder"
    assert reloaded.intent.mode == "fix"
    assert reloaded.intent.classifier == "nli"
    assert reloaded.quality.enabled is False


def test_save_config_honours_the_config_file_env(tmp_path, monkeypatch):
    target = tmp_path / "custom.json"
    monkeypatch.setenv("BIDS_AGENT_CONFIG_FILE", str(target))

    cfg = load_config(config_file="__no_such_config__.json")
    assert save_config(cfg) == target
    assert target.exists()