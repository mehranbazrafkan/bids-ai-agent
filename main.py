"""
BIDS AI Agent — Flask server.

Endpoints:
  POST /predict          — submit an issue + user prompt, get the agent response
  GET  /config           — current configuration + what hardware it runs on
  PUT  /config           — update configuration (thinking, GPU/CPU, model, etc.)
  GET  /models           — list available LLM model names
"""

from __future__ import annotations

import os
import platform
import time
import logging
import warnings
from typing import Any, Dict, List, Optional

# Suppress noisy progress bars and Laya warnings
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=RuntimeWarning)

from flask import Flask, jsonify, request

from agent import Agent
from config import CONFIG

# ---------------------------------------------------------------------------
# Logging — keep only our own messages
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bids_agent")
logger.setLevel(logging.INFO)

# Silence noisy third-party loggers
for noisy_logger in [
    "werkzeug", "httpx", "httpcore", "huggingface_hub", "transformers",
    "retriever", "laya", "filelock",
]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Singleton agent (lazy-initialized on first request)
_agent: Optional[Agent] = None


def _get_agent() -> Agent:
    """Lazy-initialize the agent singleton."""
    global _agent
    if _agent is None:
        _agent = Agent()
    return _agent


@app.before_request
def _start_timer():
    """Store request start time for duration logging."""
    request._start_time = time.perf_counter()


@app.after_request
def _log_duration(response):
    """Log the duration of each request."""
    duration = time.perf_counter() - getattr(request, "_start_time", 0)
    logger.info(f"{request.method} {request.path} - {response.status_code} - {duration:.2f}s")
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

# How much of a transcript travels back with a follow-up question. The
# prompt already carries the finding itself, so only the exchange has to
# be sent — and only so far back, or every turn multiplies the context
# the model has to read before it can answer.
HISTORY_LIMIT = 12


def _history_from(data: Any) -> Optional[List[Dict[str, str]]]:
    """Turn whatever arrived under ``history`` into what the planner wants.

    The desktop app keeps its transcript so a follow-up question has the
    earlier ones to stand on. Anything malformed — a stray string, a
    message with no content, a role that means nothing here — is dropped
    rather than passed through to the prompt builder, and the result is
    capped so a long session cannot grow the prompt without bound.
    """
    raw = data.get("history") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return None

    out: List[Dict[str, str]] = []
    for item in raw:
        if len(out) >= HISTORY_LIMIT:
            break
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        out.append({"role": role, "content": content})
    return out or None


def _hardware() -> Dict[str, Any]:
    """What PyTorch can generate on — no model is loaded to answer this.

    Asked of the process that will actually do the work. A desktop app
    can see that a card exists; only the agent knows whether its torch
    was built with CUDA and whether it can open the device at all.
    """
    try:
        import torch
    except Exception as exc:                            # pragma: no cover
        return {"backend": "unknown", "detail": str(exc)}

    info: Dict[str, Any] = {
        "backend": "cpu",
        "device_name": "CPU",
        "python": platform.python_version(),
        "torch": str(getattr(torch, "__version__", "") or ""),
        "cpu_cores": int(os.cpu_count() or 0),
    }
    cuda = getattr(getattr(torch, "version", None), "cuda", None)
    if cuda:
        # Present whenever torch was *built* with CUDA, which is not the
        # same as there being a card to use — hence the availability
        # check below deciding the backend.
        info["cuda_version"] = str(cuda)

    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            name = str(props.name)
            total = int(getattr(props, "total_memory", 0) or 0)
        except Exception:                               # pragma: no cover
            name, total = "NVIDIA GPU", 0
        info.update({
            "backend": "cuda",
            "device_name": name,
            "device_count": int(torch.cuda.device_count()),
        })
        if total:
            info["memory_gb"] = round(total / (1024 ** 3), 1)
        return info

    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        info.update({"backend": "mps", "device_name": "Apple GPU"})
        return info

    return info


@app.route("/predict", methods=["POST"])
def predict():
    """Run the agent on the given issue + user prompt and return the response.

    The metadata reports what actually happened during the request:
      * ``intent`` — the intent the agent *used* (Laya classifies once;
        nothing is re-classified after the fact);
      * ``tool_used`` — always ``null`` in the explanation-only phase,
        because no tool executes;
      * ``execution_status`` — ``"not_available"`` when the user
        requested an action, else ``null``;
      * ``classification_source`` — ``"laya"`` when the model decided,
        ``"fallback"`` when the conservative unclear result was used.
    """
    data = request.get_json(force=True)
    if not data or "user_input" not in data or "context" not in data:
        return jsonify({"error": "Missing required fields: user_input, context"}), 400
    user_input = data["user_input"]
    context = data["context"]
    history = _history_from(data)

    try:
        agent = _get_agent()
        result = agent.run(
            user_input=user_input, context=context, history=history,
        )
    except Exception as exc:
        # Left to Flask this escapes as a bare HTML error page, which the
        # desktop app has nothing better to do than show verbatim — the
        # user reads "The server encountered an internal error" instead of
        # the one line that would tell them what to change. A local,
        # single-user service gains nothing from hiding the reason.
        logger.exception("predict failed")
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    return jsonify({
        "response": result.response,
        "intent": result.intent,
        "tool_used": None,
        "execution_status": result.execution_status,
        "classification_source": result.classification_source,
    })


@app.route("/config", methods=["GET"])
def get_config():
    """Return the current configuration."""
    llm = CONFIG.llm
    retrieval = CONFIG.retrieval
    return jsonify({
        "llm": {
            "model_name": llm.model_name,
            "max_new_tokens": llm.max_new_tokens,
            "temperature": llm.temperature,
            "top_p": llm.top_p,
            "top_k": llm.top_k,
            "repetition_penalty": llm.repetition_penalty,
            "enable_thinking": llm.enable_thinking,
            "load_in_4bit": llm.load_in_4bit,
            # The current spelling; ``load_in_4bit`` above is kept in
            # step with it for older readers.
            "quantization": llm.quantization,
            "device_map": llm.device_map,
            "torch_dtype": llm.torch_dtype,
        },
        "retrieval": {
            "knowledge_base_dir": retrieval.knowledge_base_dir,
            "min_best_score": retrieval.min_best_score,
            "default_top_k": retrieval.default_top_k,
            "max_related": retrieval.max_related,
            "index_raw_content": retrieval.index_raw_content,
            "lazy_raw_content": retrieval.lazy_raw_content,
        },
        "laya": {
            "model": CONFIG.laya.model,
            # Laya's confidence is uncalibrated for this domain; the
            # threshold is documented in config.LayaConfig and tested in
            # tests/test_laya_decision.py.
            "confidence_threshold": CONFIG.laya.confidence_threshold,
        },
        # What the answers will be generated with. Detection touches no
        # model and no disk, so this stays cheap enough to sit on the
        # same call Settings already makes when the tab opens.
        "hardware": _hardware(),
    })


@app.route("/config", methods=["PUT"])
def update_config():
    """Update configuration values. Only provided fields are changed."""
    global _agent

    data = request.get_json(force=True)
    llm = CONFIG.llm
    retrieval = CONFIG.retrieval

    # Update LLM settings
    if "model_name" in data:
        llm.model_name = data["model_name"]
    if "max_new_tokens" in data:
        llm.max_new_tokens = int(data["max_new_tokens"])
    if "temperature" in data:
        llm.temperature = float(data["temperature"])
    if "top_p" in data:
        llm.top_p = float(data["top_p"])
    if "top_k" in data:
        llm.top_k = int(data["top_k"])
    if "repetition_penalty" in data:
        llm.repetition_penalty = float(data["repetition_penalty"])
    if "enable_thinking" in data:
        llm.enable_thinking = bool(data["enable_thinking"])
    if "load_in_4bit" in data:
        # Legacy spelling; kept in step with ``quantization`` below so an
        # older client and the newer selector agree about what will load.
        llm.load_in_4bit = bool(data["load_in_4bit"])
        llm.quantization = "4bit" if llm.load_in_4bit else "none"
    if "quantization" in data:
        value = str(data["quantization"]).strip().lower()
        llm.quantization = value if value in ("none", "4bit", "8bit") else "none"
        llm.load_in_4bit = llm.quantization == "4bit"
    if "device_map" in data:
        llm.device_map = data["device_map"]
    if "torch_dtype" in data:
        llm.torch_dtype = data["torch_dtype"]

    # Update retrieval settings
    if "knowledge_base_dir" in data:
        retrieval.knowledge_base_dir = data["knowledge_base_dir"]
    if "min_best_score" in data:
        retrieval.min_best_score = float(data["min_best_score"])
    if "default_top_k" in data:
        retrieval.default_top_k = int(data["default_top_k"])
    if "max_related" in data:
        retrieval.max_related = int(data["max_related"])
    if "index_raw_content" in data:
        retrieval.index_raw_content = bool(data["index_raw_content"])
    if "lazy_raw_content" in data:
        retrieval.lazy_raw_content = bool(data["lazy_raw_content"])

    # Update Laya settings
    if "laya" in data and isinstance(data["laya"], dict):
        laya = CONFIG.laya
        if "model" in data["laya"] and data["laya"]["model"]:
            laya.model = str(data["laya"]["model"])
        if "confidence_threshold" in data["laya"] and data["laya"]["confidence_threshold"] is not None:
            laya.confidence_threshold = float(data["laya"]["confidence_threshold"])

    # Reset the agent so next /predict uses the new config
    _agent = None

    return get_config()


@app.route("/models", methods=["GET"])
def list_models():
    """Return the list of available LLM model names that can be used."""
    models = [
        # My Models #
        {
            "name": "MehranBazrafkan/gemma-3-270m-it-bids-finetuned",
            "description": "Gemma 3 270M — BIDS-finetuned version",
            "size": "270M",
            "vram_gb": "~0.5 GB (fp16)",
        },
        {
            "name": "MehranBazrafkan/qwen3-1.7B-bids-finetuned",
            "description": "Qwen3 1.7B — BIDS-finetuned version",
            "size": "1.7B",
            "vram_gb": "~3.4 GB (fp16) / ~0.9 GB (4-bit)",
        },
        # My Models #
        
        {
            "name": "google/gemma-3-270m-it",
            "description": "Gemma 3 270M — very small, fastest option",
            "size": "270M",
            "vram_gb": "~0.5 GB (fp16)",
        },
        {
            "name": "google/gemma-3-1b-it",
            "description": "Gemma 3 1B — Google's small instruction-tuned model",
            "size": "1B",
            "vram_gb": "~2 GB (fp16)",
        },
        {
            "name": "Qwen/Qwen2.5-1.5B-Instruct",
            "description": "Qwen2.5-1.5B-Instruct — balanced option for general use",
            "size": "1.5B",
            "vram_gb": "~3 GB (fp16) / ~0.75 GB (4-bit)",
        },
        {
            "name": "Qwen/Qwen2.5-Coder-1.5B",
            "description": "Qwen2.5-Coder-1.5B — balanced option for general use",
            "size": "1.5B",
            "vram_gb": "~3 GB (fp16) / ~0.75 GB (4-bit)",
        },
        {
            "name": "nvidia/OpenReasoning-Nemotron-1.5B",
            "description": "OpenReasoning Nemotron 1.5B — balanced option for general use",
            "size": "1.5B",
            "vram_gb": "~3 GB (fp16) / ~0.75 GB (4-bit)",
        },
        {
            "name": "Qwen/Qwen2-1.5B-Instruct",
            "description": "Qwen2-1.5B-Instruct — balanced option for general use",
            "size": "1.5B",
            "vram_gb": "~3 GB (fp16) / ~0.75 GB (4-bit)",
        },
        {
            "name": "stabilityai/stablelm-2-1_6b-chat",
            "description": "StableLM-2 1.6B — chat model for general use",
            "size": "1.6B",
            "vram_gb": "~3 GB (fp16) / ~0.75 GB (4-bit)",
        },
        {
            "name": "RedHatAI/Qwen3-1.7B-quantized.w4a16",
            "description": "Qwen3 1.7B pre-quantized to 4-bit — lowest VRAM usage",
            "size": "1.7B (w4a16)",
            "vram_gb": "~0.9 GB",
        },
        {
            "name": "Qwen/Qwen3-1.7B",
            "description": "Qwen3 1.7B — small, fast, good for most tasks",
            "size": "1.7B",
            "vram_gb": "~3.4 GB (fp16) / ~0.9 GB (4-bit)",
        },
        {
            "name": "Qwen/Qwen3-4B",
            "description": "Qwen3 4B — larger, more capable, needs more VRAM",
            "size": "4B",
            "vram_gb": "~8 GB (fp16) / ~2.1 GB (4-bit)",
        },
        {
            "name": "google/gemma-3-4b-it",
            "description": "Gemma 3 4B — larger Google model",
            "size": "4B",
            "vram_gb": "~8 GB (fp16)",
        },
    ]
    return jsonify({"models": models})


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Host/port come from the environment so a caller that starts this
    # server itself can move it off a busy port and keep it on loopback.
    # The defaults are the ones this script has always used, so running
    # ``python main.py`` by hand is unchanged.
    app.run(
        host=os.environ.get("BIDS_AGENT_HOST", "0.0.0.0"),
        port=int(os.environ.get("BIDS_AGENT_PORT", "8000")),
        debug=False,
    )
