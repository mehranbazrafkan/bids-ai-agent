"""
BIDS AI Agent — Flask server.

Endpoints:
  POST /predict          — submit an issue + user prompt, get the agent response
  GET  /config           — view current configuration
  PUT  /config           — update configuration (thinking, GPU/CPU, model, etc.)
  GET  /models           — list available LLM model names
"""

from __future__ import annotations

import os
import time
import logging
import warnings
from typing import Any, Dict, Optional

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


@app.route("/predict", methods=["POST"])
def predict():
    """Run the agent on the given issue + user prompt and return the response."""
    data = request.get_json(force=True)
    if not data or "user_input" not in data or "context" not in data:
        return jsonify({"error": "Missing required fields: user_input, context"}), 400
    user_input = data["user_input"]
    context = data["context"]

    agent = _get_agent()
    response = agent.run(user_input=user_input, context=context)

    # Determine intent and tool used for the response
    from utils import compact_json
    str_context = compact_json(context)
    intent = agent.planner.laya.classify_intent(user_input, str_context)
    if intent not in ("fix", "explain"):
        intent = "explain"

    tool_used = None
    if intent == "fix":
        tool_used = agent.planner.laya.select_tool(user_input, str_context)

    return jsonify({
        "response": response,
        "intent": intent,
        "tool_used": tool_used,
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
        llm.load_in_4bit = bool(data["load_in_4bit"])
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

    # Reset the agent so next /predict uses the new config
    _agent = None

    return get_config()


@app.route("/models", methods=["GET"])
def list_models():
    """Return the list of available LLM model names that can be used."""
    models = [
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
            "name": "google/gemma-3-1b-it",
            "description": "Gemma 3 1B — Google's small instruction-tuned model",
            "size": "1B",
            "vram_gb": "~2 GB (fp16)",
        },
        {
            "name": "google/gemma-3-270m",
            "description": "Gemma 3 270M — very small, fastest option",
            "size": "270M",
            "vram_gb": "~0.5 GB (fp16)",
        },
        {
            "name": "google/gemma-3-4b-it",
            "description": "Gemma 3 4B — larger Google model",
            "size": "4B",
            "vram_gb": "~8 GB (fp16)",
        },
        {
            "name": "RedHatAI/Qwen3-1.7B-quantized.w4a16",
            "description": "Qwen3 1.7B pre-quantized to 4-bit — lowest VRAM usage",
            "size": "1.7B (w4a16)",
            "vram_gb": "~0.9 GB",
        },
    ]
    return jsonify({"models": models})


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
