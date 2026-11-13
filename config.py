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
which the rest of the code imports. Changes made through ``PUT /config`` are
written back to the same file by :func:`save_config`, so they survive a
restart.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
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
    # model_name = "google/gemma-3-270m-it"
    # model_name = "google/gemma-3-4b-it" # Is not downloaded (8GB)
    # model_name = "Qwen/Qwen3-4B" # Is not downloaded (8GB)
    model_name: str = "Qwen/Qwen3-1.7B"
    # model_name: str = "RedHatAI/Qwen3-1.7B-quantized.w4a16"
    # model_name: str = "RedHatAI/Qwen3-1.7B-quantized.w4a16"
    # 512 was chosen over the old 250 because structured explanations
    # (problem / cause / impact / manual fix / example / uncertainty) plus
    # action-unavailable disclosures routinely exceed 250 tokens, and when
    # thinking mode is enabled the model's reasoning consumes the same
    # budget. The extra 262 tokens cost negligible memory for models of
    # this size; the main effect is latency on long answers, which only
    # occurs when the answer is genuinely long.
    max_new_tokens: int = 512
    # Fact-constrained explanation, not creative writing: 0.75 turned the same
    # issue into different (sometimes invalid) recommendations across runs.
    # 0.2 keeps the wording natural while making the answer reproducible.
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int = 20
    repetition_penalty: float = 1.2
    # ``"auto"`` means the checkpoint's own dtype. Qwen3-1.7B is published in
    # bfloat16, so the unquantized model is 16-bit (~3.4 GB), NOT float32.
    # Set ``"float32"`` explicitly only if you accept the 4x size.
    torch_dtype: str = "auto"
    device_map: str = "auto"
    enable_thinking: bool = False
    # 4-bit quantization (requires bitsandbytes: pip install bitsandbytes).
    # Reduces VRAM from ~3.4 GB to ~0.9 GB for 1.7B, or ~2.1 GB for 4B.
    load_in_4bit: bool = False
    # How the weights are stored when the model is loaded:
    #   "none"  - no quantization: the checkpoint's own dtype (bfloat16,
    #             16-bit for Qwen3-1.7B — not float32)
    #   "4bit"  - bitsandbytes nf4, ~1/4 of the VRAM
    #   "8bit"  - bitsandbytes int8, ~1/2 of the VRAM
    # ``load_in_4bit`` above is the older spelling of one of these choices
    # and is still honoured when this reads "none", so an existing
    # config.json keeps doing what it always did.
    quantization: str = "none"
    # Not a generation setting, but it belongs to the section that owns
    # what the model is fed: write the most recent request's system
    # prompt, user prompt and agent output to ``prompt_log.txt`` next to
    # ``main.py`` -- overwritten on every request, and no conversation is
    # ever written to disk (see ``prompt_log``). Off = false here, in
    # config.json, or via BIDS_AGENT_LLM_SAVE_PROMPTS=false.
    save_prompts: bool = True


@dataclass
class RetrievalConfig:
    """Knowledge-base location and retrieval thresholds."""

    knowledge_base_dir: str = "./knowledge-base"
    # The single knowledge source. Every record the agent can retrieve comes
    # from this one file; there are no auxiliary files (no relationships,
    # no sources) — the retriever never opens anything else.
    knowledge_file: str = "knowledge_base.jsonl"
    # Minimum score a record must reach to be considered a real match. For a
    # large KB this floor rejects boilerplate-driven noise (e.g. generic
    # tokens like "Funding"); for a small KB it is scaled down automatically
    # (see min_best_scale) so real matches still pass.
    min_best_score: float = 3.0
    # Adaptive floor: for small knowledge bases the floor is reduced to
    # max(0.5, n * min_best_scale), never above min_best_score.
    min_best_scale: float = 0.005
    # Evidence shown to the model per request. Three (rather than one) so
    # that the exact-rule record does not crowd out the records that
    # explain *why* a value is invalid (e.g. the field's schema/format
    # definition ranked below the generic rule record). Records that
    # conflict with the issue's file context are rejected before this
    # budget is applied, and duplicate records (same rule) count once.
    default_top_k: int = 3
    # Keep raw_content out of the search index (it is the largest, least
    # discriminating field). Enabling this restores the original behaviour
    # of scoring against the full raw YAML fragment.
    index_raw_content: bool = False
    # Do not keep raw_content in memory; read it back from disk on demand
    # (only returned records need it, and the formatter caps it at 600 chars).
    lazy_raw_content: bool = True

    # ------------------------------------------------------------------
    # Retriever backend
    # ------------------------------------------------------------------
    # "lexical"   — the original deterministic, field-aware token scorer.
    # "embedding" — semantic retrieval over the same records. The lexical
    #               retriever is still used for loading, applicability,
    #               assessments and formatting; only ranking changes.
    retriever_type: str = "lexical"

    # Embedding model used by the semantic backend. BGE-small is a strong,
    # CPU-friendly English retriever (~130 MB, 384 dimensions) and fits next
    # to the quantized LLM on a 4 GB GPU.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # "auto" uses CUDA when available, otherwise CPU.
    embedding_device: str = "auto"
    # "cls" is the pooling BGE was trained with; "mean" works for most other
    # sentence-transformer checkpoints.
    embedding_pooling: str = "cls"
    embedding_max_length: int = 512
    embedding_batch_size: int = 32
    # Optional instruction prepended to the query (not the documents), e.g.
    # "Represent this sentence for searching relevant passages: ".
    embedding_query_prefix: str = ""
    # Cosine similarity below this is an explicit miss. BGE relevant pairs
    # usually score well above 0.35; unrelated BIDS prose is lower.
    embedding_min_score: float = 0.35
    # How many semantic candidates to inspect before applicability/dedupe
    # filtering. Larger than top_k so rejected records do not empty the
    # evidence set.
    embedding_candidate_k: int = 12
    # Where vectors.npy + manifest.json live. The store is rebuilt (and the
    # previous files replaced) whenever the KB or embedding model changes.
    vector_store_dir: str = "./knowledge-base/vector_store"


@dataclass
class LayaConfig:
    """Settings for the Laya decision engine (intent classification)."""

    model: str = "convaiinnovations/laya"
    # Explain-only phase: while this is False (the default for now), the
    # planner routes every request as ``explain`` and never consults the
    # model — the API reports ``classification_source: "manual"``. Set it
    # to True (config file, environment variable, or ``PUT /config``) to
    # restore Laya intent classification (explain / fix / unclear).
    classify: bool = False
    # Minimum probability the model must assign to its own ``explain`` or
    # ``fix`` choice before that choice is accepted (0.5 = the class is the
    # model's clear preference over the alternatives). ``unclear`` is
    # accepted at any support level because it is the safe outcome, and
    # anything below the threshold routes to the conservative unclear
    # handling (clarification / context-supported explanation) instead of a
    # guessed routing. The gate thresholds ``probabilities[choice]``, not
    # the package's entropy-derived ``confidence`` field, which the laya
    # package itself documents as uncalibrated for these choice buckets —
    # gating on it at 0.5 rejected ~90% of real requests (see
    # laya_decision module docstring and eval_intent.py). Behaviour around
    # the threshold is covered by tests/test_laya_decision.py.
    confidence_threshold: float = 0.5


@dataclass
class IntentConfig:
    """Intent routing: how a request is classified and whether it is forced.

    ``mode`` is the operator-facing switch:

    * ``"auto"``    — classify each request with ``classifier``.
    * ``"explain"`` — always route as explanation, never call a classifier.
    * ``"fix"``     — always route as an action request, never call a
      classifier. The planner still discloses that execution is unavailable.

    ``classifier`` selects the auto-mode engine:

    * ``"llm"``  — the project's main instruction-tuned LLM (default). A
      deterministic guard resolves unmistakable forms such as
      "How can I fix this?" (EXPLAIN) versus "Fix it for me" (FIX); only
      ambiguous paraphrases are sent to a compact few-shot prompt asking for
      EXPLAIN / FIX / UNCLEAR. This avoids Laya's explain/fix bias.
    * ``"nli"``  — the legacy zero-shot cross-encoder in
      ``intent_classifier`` (kept for comparison).
    * ``"laya"`` — the legacy Laya engine (kept for compatibility).
    """

    mode: str = "auto"
    classifier: str = "llm"


@dataclass
class QualityConfig:
    """Output-contract enforcement (see :mod:`answer_contract`).

    ``enabled`` validates every generated answer against the per-request
    contract and repairs it; ``repair_attempts`` bounds the model calls spent
    on repair; ``fallback_enabled`` allows the deterministic template as the
    final tier. Disable the whole stage to reproduce the pre-gate behaviour.
    """

    enabled: bool = True
    repair_attempts: int = 1
    fallback_enabled: bool = True


@dataclass
class AgentConfig:
    """Top-level application configuration."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    laya: LayaConfig = field(default_factory=LayaConfig)
    intent: IntentConfig = field(default_factory=IntentConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)


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
    laya = _apply_config_file(LayaConfig(), file_values.get("laya", {}))
    intent = _apply_config_file(IntentConfig(), file_values.get("intent", {}))
    quality = _apply_config_file(QualityConfig(), file_values.get("quality", {}))
    llm = _apply_env(llm, "LLM")
    retrieval = _apply_env(retrieval, "RETRIEVER")
    laya = _apply_env(laya, "LAYA")
    intent = _apply_env(intent, "INTENT")
    quality = _apply_env(quality, "QUALITY")

    return AgentConfig(
        llm=llm,
        retrieval=retrieval,
        laya=laya,
        intent=intent,
        quality=quality,
    )


def save_config(
    config: AgentConfig | None = None,
    config_file: str | Path | None = None,
) -> Path:
    """Write *config* to the file :func:`load_config` would read.

    ``PUT /config`` calls this so a setting changed in the UI survives a
    restart. The write is atomic (temp file + ``os.replace``), so a crash or
    a full disk cannot leave a half-written config behind. Unknown keys in an
    existing file are not preserved; the five known sections are rewritten in
    full.
    """
    cfg = config if config is not None else CONFIG
    path = _resolve_config_file(config_file)
    payload = {
        "llm": asdict(cfg.llm),
        "retrieval": asdict(cfg.retrieval),
        "laya": asdict(cfg.laya),
        "intent": asdict(cfg.intent),
        "quality": asdict(cfg.quality),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp, path)
    return path


# Module-level singleton used across the codebase.
CONFIG = load_config()
