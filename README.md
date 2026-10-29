# BIDS AI Agent

AI agent for explaining issues found in MRI datasets according to the BIDS
specification. It uses **Laya** (a fast, non-autoregressive System 1
decision engine) as the intent classifier and a local LLM for output
generation only.

**This is the explanation-only phase.** The agent never executes tools: it
does not fix, rename, validate, or modify anything, and it never claims it
did. `tool_used` is always `null`; action requests are answered with a
grounded explanation that states execution is unavailable
(`execution_status: "not_available"`).

## Architecture

```
User Input + Issue Context
        |
        v
   [Laya] one intent classification per request: explain / fix / unclear
        |
        +---> explain --> [Retriever] --> [LLM] --> grounded explanation
        |
        +---> fix -------> [Retriever] --> [LLM] --> explanation + "execution
        |                                             not available" disclosure
        +---> unclear ---> no issue context: clarification question
                            issue context:   safe explanation-only answer
```

- **Laya** — the only intent classifier. It answers a three-way semantic
  choice question (no keyword list). An `explain`/`fix` choice is accepted
  only when the model's own probability for that class reaches
  `laya.confidence_threshold` (default `0.5`); `unclear` is accepted at any
  support level because it is the safe outcome. Model failures or weak
  support degrade to `unclear` with `classification_source: "fallback"`.
  The classifies happens **once** per request and that decision is what the
  API reports — nothing is re-classified later.
- **Retriever** — explainable, structured scoring over the local JSONL
  knowledge base (rule-code and field exact matches, then lexical; see
  `retriever.py`). No embeddings, no vector database.
- **LLM** — generates the explanation text from retrieved evidence. It
  makes no decisions (see `intent_classifier.py`: the old keyword
  classifier is deprecated and no longer called by the planner).

## Installation

```bash
pip install flask laya torch transformers pytest
pip install bitsandbytes   # optional: 4-bit / 8-bit model loading
```

Models are loaded from the local Hugging Face cache on first use.

## Running the Server

```bash
python main.py
```

The Flask server starts on `http://localhost:8000` (override with
`BIDS_AGENT_HOST` / `BIDS_AGENT_PORT`). There is no `/docs` page — this is
Flask, not FastAPI.

---

## API Endpoints

### 1. `POST /predict` — Get a response for an issue

**Request body:**

| Field | Type | Description |
|---|---|---|
| `user_input` | string | The user's prompt or question |
| `context` | object | The issue / error context (JSON object) |
| `history` | list? | Optional transcript: `[{"role": "user"\|"assistant", "content": "..."}]` (malformed entries are dropped, capped length) |

**Response:**

| Field | Type | Description |
|---|---|---|
| `response` | string | The generated response from the agent |
| `intent` | string | The intent actually used: `"explain"`, `"fix"`, or `"unclear"` |
| `tool_used` | null | Always `null` — no tool executes in this phase |
| `execution_status` | string? | `"not_available"` when the user asked for an action (`fix`), else `null` |
| `classification_source` | string | `"laya"` when Laya made the decision, `"fallback"` when the conservative unclear result was used (model failure / weak support) |

The `intent` is Laya's single classification, passed through unchanged.
A `fallback` classification is never presented as a model decision.

#### cURL Example

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"user_input\": \"Explain this to me.\", \"context\": {\"severity\": \"err\", \"rule_id\": \"JSON_SCHEMA_VALIDATION_ERROR\", \"message\": \"ReferencesAndLinks must be array  ·  List of references to publications that contain information on the dataset. Use a value of the correct type, for example {\\\"ReferencesAndLinks\\\": [\\\"text\\\"]}.\", \"field\": \"ReferencesAndLinks\", \"line\": \"null\", \"lines\": [], \"fix_label\": \"null\", \"fix_action\": \"null\", \"mirrored\": false}}"
```

#### Python Example

```python
import requests

sample_issue_001 = {
    "severity": "err",
    "rule_id": "JSON_SCHEMA_VALIDATION_ERROR",
    "message": "ReferencesAndLinks must be array  ·  List of references to publications that contain information on the dataset. Use a value of the correct type, for example {\"ReferencesAndLinks\": [\"text\"]}.",
    "field": "ReferencesAndLinks",
    "line": "null",
    "lines": [],
    "fix_label": "null",
    "fix_action": "null",
    "mirrored": False,
}

response = requests.post("http://localhost:8000/predict", json={
    "user_input": "Explain this to me.",
    "context": sample_issue_001,
})

data = response.json()
print("Intent:", data["intent"])
print("Source:", data["classification_source"])
print("Response:", data["response"])
```

#### Sample Response

```json
{
  "response": "Problem: The field 'ReferencesAndLinks' must be an array...",
  "intent": "explain",
  "tool_used": null,
  "execution_status": null,
  "classification_source": "laya"
}
```

An action request (`intent: "fix"`) looks like this — note that nothing
was executed and the response says so:

```json
{
  "response": "I cannot apply changes to your dataset in this version. ...",
  "intent": "fix",
  "tool_used": null,
  "execution_status": "not_available",
  "classification_source": "laya"
}
```

---

### 2. `GET /config` — View current configuration

Returns the current LLM, retrieval, and Laya settings plus detected
hardware (no model is loaded to answer this).

#### cURL Example

```bash
curl http://localhost:8000/config
```

#### Python Example

```python
import requests

config = requests.get("http://localhost:8000/config").json()
print("Model:", config["llm"]["model_name"])
print("Max tokens:", config["llm"]["max_new_tokens"])
print("Laya threshold:", config["laya"]["confidence_threshold"])
```

#### Sample Response

```json
{
  "llm": {
    "model_name": "Qwen/Qwen3-1.7B",
    "max_new_tokens": 512,
    "temperature": 0.75,
    "top_p": 0.95,
    "top_k": 20,
    "repetition_penalty": 1.2,
    "enable_thinking": false,
    "load_in_4bit": false,
    "quantization": "none",
    "device_map": "auto",
    "torch_dtype": "auto"
  },
  "retrieval": {
    "knowledge_base_dir": "./knowledge-base",
    "min_best_score": 3.0,
    "default_top_k": 1,
    "max_related": 3,
    "index_raw_content": false,
    "lazy_raw_content": true
  },
  "laya": {
    "model": "convaiinnovations/laya",
    "confidence_threshold": 0.5
  },
  "hardware": {
    "backend": "cuda",
    "device_name": "NVIDIA GeForce GTX 1650 Ti",
    "python": "3.10.0",
    "torch": "2.14.1+cu126",
    "cpu_cores": 8,
    "cuda_version": "12.6",
    "device_count": 1,
    "memory_gb": 4.0
  }
}
```

> `max_new_tokens` defaults to **512** (rationale documented in
> `config.py`; the value is the generation budget for one answer,
> including reasoning tokens when thinking mode is on).

---

### 3. `PUT /config` — Update configuration

Update any subset of configuration fields. Only provided fields are
changed. The agent is reset automatically so the next `/predict` uses the
new settings.

#### Controllable Settings

| Field | Type | Description |
|---|---|---|
| `model_name` | string | LLM model to use (see `/models`) |
| `max_new_tokens` | int | Maximum tokens to generate (default 512) |
| `temperature` | float | Sampling temperature (0.0–2.0) |
| `top_p` | float | Nucleus sampling probability |
| `top_k` | int | Top-k sampling |
| `repetition_penalty` | float | Penalty for repetition (>1.0) |
| `enable_thinking` | bool | Enable thinking/reasoning mode |
| `load_in_4bit` | bool | Legacy spelling; kept in step with `quantization` |
| `quantization` | string | `"none"`, `"4bit"`, or `"8bit"` |
| `device_map` | string | Device mapping (`"auto"`, `"cpu"`, `"cuda"`) |
| `torch_dtype` | string | Torch dtype (`"auto"`, `"float16"`, `"bfloat16"`) |
| `knowledge_base_dir` | string | Path to knowledge base directory |
| `min_best_score` | float | Minimum retrieval score threshold |
| `default_top_k` | int | Default number of results |
| `max_related` | int | Max related items in retrieval |
| `index_raw_content` | bool | Index raw content in retriever |
| `lazy_raw_content` | bool | Lazy-load raw content |
| `laya.model` | string | Laya checkpoint name |
| `laya.confidence_threshold` | float | Minimum class probability to accept an `explain`/`fix` choice |

#### cURL Example — Enable GPU + 4-bit quantization

```bash
curl -X PUT http://localhost:8000/config -H "Content-Type: application/json" -d "{\"device_map\": \"cuda\", \"quantization\": \"4bit\", \"enable_thinking\": false, \"max_new_tokens\": 512}"
```

#### Python Example — Full config update

```python
import requests

resp = requests.put("http://localhost:8000/config", json={
    "device_map": "cuda",
    "quantization": "4bit",
    "enable_thinking": False,
    "max_new_tokens": 512,
    "temperature": 0.5,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
})
print("Updated config:", resp.json())
```

---

### 4. `GET /models` — List available LLM models

Returns the supported LLM models with their size and VRAM requirements.

```bash
curl http://localhost:8000/models
```

```json
{
  "models": [
    {"name": "Qwen/Qwen3-1.7B", "description": "Qwen3 1.7B — small, fast, good for most tasks", "size": "1.7B", "vram_gb": "~3.4 GB (fp16) / ~0.9 GB (4-bit)"}
  ]
}
```

---

## Intent Classification — Laya

Laya is asked one three-way choice question per request with three
semantic criteria: `explain`, `fix`, and `unclear` (vague / greeting / no
concrete target). Acceptance gate:

- `explain` / `fix`: accepted only when `probabilities[choice] >=
  laya.confidence_threshold` (default `0.5`).
- `unclear`: always accepted (the safe outcome — accepting it can never
  misroute).
- Anything else (exception, malformed response, invalid label, weak
  support): conservative `unclear` with `classification_source:
  "fallback"`.

The gate thresholds the model's **class probability**, not the package's
entropy-derived `confidence` field — the installed `laya` package itself
warns its shipped temperatures are invalid and says to treat that value as
uncalibrated (history and measurements in the `laya_decision.py` module
docstring).

### Evaluating the classifier

```bash
python eval_intent.py                    # full local model, prints accuracy,
                                         # per-class recall, confusion matrix,
                                         # and every failure
python eval_intent.py --laya-mode disabled   # control: always "unclear"
                                         # (dead-classifier baseline, 12/51)
python eval_intent.py --limit 10         # smoke run
```

Dataset: `tests/data/intent_eval.jsonl` (51 labelled paraphrases:
23 explain / 16 fix / 12 unclear). Measured with the current gate (runs on
the cached local checkpoint): **0.53–0.63 accuracy** across runs (Laya
inference is not deterministic) vs **0.235** for the always-unclear
baseline; the previous entropy-confidence gate scored **0.31–0.35**.
This is a smoke signal for "the classifier is alive", not a quality score.

---

## Running Tests

```bash
pytest tests/ -v              # default: fast tests only (slow are deselected)
pytest -m slow                # tests that load the real LLM / Laya checkpoint
pytest tests/ -v -m "not slow" # explicit form of the default
```

The `slow` marker is configured in `pytest.ini` (`addopts = -m "not
slow"`); slow tests load real models and take minutes. All API, planner,
retrieval, and Laya tests are mocked and run without any model download.

---

## Knowledge Base

See `knowledge-base/KB_README.md` for structure and regeneration
instructions, and `knowledge-base/COVERAGE_GAPS.md` for documented coverage
gaps (issues that exist in the BIDS Manager output but cannot be derived
from locally available verified source material — these are documented, not
fabricated).

---

## Project Structure

```
bids-ai-agent/
  main.py                      # Flask server (API entry point)
  agent.py                     # Agent class (orchestrates the workflow)
  planner.py                   # Planner (single Laya classification -> retrieval -> LLM)
  laya_decision.py             # Laya intent classifier (3-way choice + acceptance gate)
  llm_client.py                # LLM client (model loading + generation)
  retriever.py                 # Knowledge base retriever (structured, explainable scoring)
  tool_registry.py             # Tool registry kept for a future execution phase (never dispatched here)
  intent_classifier.py         # Legacy keyword classifier (deprecated, not called by the planner)
  model_manager.py             # Model lifecycle (lazy load + offload)
  config.py                    # Central configuration
  utils.py                     # Utility functions
  eval_intent.py               # Intent-classification evaluation runner
  benchmark_timing.py          # Timing benchmark script
  pytest.ini                   # Test markers (slow excluded by default)
  knowledge-base/              # JSONL knowledge base + derivation scripts + docs
  tests/                       # Test suite (data/ holds the intent eval set)
```
