"""
Central configuration for bids-ai-agent.

Every value that previously lived as a hardcoded assumption (the LLM model
name and sampling parameters, retrieval thresholds, knowledge-base file
names and paths) is defined here and can be changed without editing code.

Override sources, lowest to highest precedence:

1. Dataclass defaults in this module.
2. An optional JSON config file (default ``./config.json``; the path can
   be overridden with the env var ``BIDS_AGENT_CONFIG_FILE``). Example::

       {
         "llm": {"model_name": "Qwen/Qwen3-4B"},
         "retrieval": {"knowledge_base_dir": "/data/kb", "min_best_score": 5.0}
       }

3. Environment variables grouped by section and prefixed with
   ``BIDS_AGENT_``, e.g. ``BIDS_AGENT_LLM_MAX_NEW_TOKENS=512`` or
   ``BIDS_AGENT_RETRIEVER_KNOWLEDGE_BASE_DIR=./kb``.

The loaded configuration is exposed as the module-level ``CONFIG`` object,
which the rest of the code imports.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

_ENV_PREFIX = "BIDS_AGENT_"
_CONFIG_FILE_ENV = "BIDS_AGENT_CONFIG_FILE"
_DEFAULT_CONFIG_FILE = Path("config.json")


# ---------------------------------------------------------------------------
# Configuration sections
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    """Model and generation settings for the local LLM."""

    # model_name = "google/gemma-3-1b-it"
    # model_name = "google/gemma-3-270m"
    # model_name = "google/gemma-3-4b-it" # Is not downloaded (8GB)
    # model_name = "Qwen/Qwen3-4B" # Is not downloaded (8GB)
    model_name: str = "Qwen/Qwen3-1.7B"
    # model_name: str = "RedHatAI/Qwen3-1.7B-quantized.w4a16"
    # model_name: str = "RedHatAI/Qwen3-1.7B-quantized.w4a16"
    max_new_tokens: int = 1000
    temperature: float = 0.75
    top_p: float = 0.95
    top_k: int = 20
    repetition_penalty: float = 1.2
    torch_dtype: str = "auto"
    device_map: str = "auto"
    enable_thinking: bool = True
    # 4-bit quantization (requires bitsandbytes: pip install bitsandbytes).
    # Reduces VRAM from ~3.4 GB to ~0.9 GB for 1.7B, or ~2.1 GB for 4B.
    load_in_4bit: bool = False


@dataclass
class RetrievalConfig:
    """Knowledge-base location and retrieval thresholds."""

    knowledge_base_dir: str = "./knowledge-base"
    knowledge_file: str = "enriched_knowledge.jsonl"
    relationships_file: str = "relationships.jsonl"
    sources_file: str = "sources.jsonl"
    # Minimum score a record must reach to be considered a real match. For a
    # large KB this floor rejects boilerplate-driven noise (e.g. generic
    # tokens like "Funding"); for a small KB it is scaled down automatically
    # (see min_best_scale) so real matches still pass.
    min_best_score: float = 3.0
    # Adaptive floor: for small knowledge bases the floor is reduced to
    # max(0.5, n * min_best_scale), never above min_best_score.
    min_best_scale: float = 0.005
    default_top_k: int = 1
    # How many top results contribute relationship-expanded neighbours.
    max_related: int = 3
    # Keep raw_content out of the search index (it is the largest, least
    # discriminating field). Enabling this restores the original behaviour
    # of scoring against the full raw YAML fragment.
    index_raw_content: bool = False
    # Do not keep raw_content in memory; read it back from disk on demand
    # (only returned records need it, and the formatter caps it at 600 chars).
    lazy_raw_content: bool = True


@dataclass
class AgentConfig:
    """Top-level application configuration."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)


# ---------------------------------------------------------------------------
# Loading / override helpers
# ---------------------------------------------------------------------------


def _cast(value: Any, current: Any) -> Any:
    """Cast a raw config value to the type of the existing attribute."""
    if isinstance(current, bool):
        if isinstance(value, bool):
            return value
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def _apply(section: Any, values: Dict[str, Any]) -> Any:
    for key, value in values.items():
        if hasattr(section, key) and value is not None:
            setattr(section, key, _cast(value, getattr(section, key)))
    return section


def _apply_config_file(section: Any, values: Dict[str, Any]) -> Any:
    if isinstance(values, dict):
        _apply(section, values)
    return section


def _apply_env(section: Any, section_name: str) -> Any:
    for field_info in section.__dataclass_fields__.values():  # type: ignore[attr-defined]
        key = f"{_ENV_PREFIX}{section_name}_{field_info.name.upper()}"
        if key in os.environ:
            setattr(
                section,
                field_info.name,
                _cast(os.environ[key], getattr(section, field_info.name)),
            )
    return section


def _resolve_config_file(config_file: str | Path | None) -> Path:
    if config_file is not None:
        return Path(config_file)
    env_path = os.environ.get(_CONFIG_FILE_ENV)
    if env_path:
        return Path(env_path)
    return _DEFAULT_CONFIG_FILE


def load_config(config_file: str | Path | None = None) -> AgentConfig:
    """Build an :class:`AgentConfig` from file + environment overrides."""
    file_values: Dict[str, Any] = {}
    path = _resolve_config_file(config_file)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            file_values = raw

    llm = _apply_config_file(LLMConfig(), file_values.get("llm", {}))
    retrieval = _apply_config_file(RetrievalConfig(), file_values.get("retrieval", {}))
    llm = _apply_env(llm, "LLM")
    retrieval = _apply_env(retrieval, "RETRIEVER")

    return AgentConfig(llm=llm, retrieval=retrieval)


# Module-level singleton used across the codebase.
CONFIG = load_config()