# BIDS AI Agent

AI agent for managing MRI datasets according to the BIDS specification. It uses a local LLM for output generation and **Laya** (a fast, non-autoregressive System 1 decision engine) for all decision-making — intent classification and tool selection.

## Architecture

```
User Input + Issue
        |
        v
   [Laya] Intent Classification (explain / fix)
        |
        +---> explain --> [Retriever] --> [LLM] --> Response
        |
        +---> fix --> [Retriever] --> [Laya] Tool Selection --> [Tool] --> [LLM] --> Response
```

- **Laya** (~33ms per decision): Intent classification and tool selection
- **Retriever**: Fetches relevant BIDS knowledge from the local JSONL knowledge base
- **LLM**: Generates explanations and summarizes tool results (output only, no decisions)

## Installation

```bash
pip install fastapi uvicorn
pip install laya
```

## Running the Server

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000
```

The server starts on `http://localhost:8000`. Interactive docs are available at `http://localhost:8000/docs`.

---

## API Endpoints

### 1. `POST /predict` — Get a response for an issue

**Request body:**

| Field | Type | Description |
|---|---|---|
| `user_input` | string | The user's prompt or question |
| `context` | object | The issue / error context (JSON object) |

**Response:**

| Field | Type | Description |
|---|---|---|
| `response` | string | The generated response from the agent |
| `intent` | string | Detected intent: `"explain"` or `"fix"` |
| `tool_used` | string? | Tool that was executed (only for fix intent) |

#### cURL Example

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"user_input\": \"Explain this to me.\", \"context\": {\"severity\": \"err\", \"rule_id\": \"JSON_SCHEMA_VALIDATION_ERROR\", \"message\": \"ReferencesAndLinks must be array  \u00b7  List of references to publications that contain information on the dataset. Use a value of the correct type, for example {\\\"ReferencesAndLinks\\\": [\\\"text\\\"]}.\", \"field\": \"ReferencesAndLinks\", \"line\": \"null\", \"lines\": [], \"fix_label\": \"null\", \"fix_action\": \"null\", \"mirrored\": false}}"
```

#### Python Example

```python
import requests

sample_issue_001 = {
    "severity": "err",
    "rule_id": "JSON_SCHEMA_VALIDATION_ERROR",
    "message": "ReferencesAndLinks must be array  \u00b7  List of references to publications that contain information on the dataset. Use a value of the correct type, for example {\"ReferencesAndLinks\": [\"text\"]}.",
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
print("Response:", data["response"])
```

#### Sample Response

```json
{
  "response": "Problem: The field 'ReferencesAndLinks' must be an array...",
  "intent": "explain",
  "tool_used": null
}
```

---

### 2. `GET /config` — View current configuration

Returns all current LLM and retrieval settings.

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
print("Temperature:", config["llm"]["temperature"])
```

#### Sample Response

```json
{
  "llm": {
    "model_name": "Qwen/Qwen3-1.7B",
    "max_new_tokens": 250,
    "temperature": 0.75,
    "top_p": 0.95,
    "top_k": 20,
    "repetition_penalty": 1.2,
    "enable_thinking": false,
    "load_in_4bit": false,
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
  }
}
```

---

### 3. `PUT /config` — Update configuration

Update any subset of configuration fields. Only provided fields are changed. The agent resets automatically so the next `/predict` uses the new settings.

#### Controllable Settings

| Field | Type | Description |
|---|---|---|
| `model_name` | string | LLM model to use (see `/models`) |
| `max_new_tokens` | int | Maximum tokens to generate |
| `temperature` | float | Sampling temperature (0.0–2.0) |
| `top_p` | float | Nucleus sampling probability |
| `top_k` | int | Top-k sampling |
| `repetition_penalty` | float | Penalty for repetition (>1.0) |
| `enable_thinking` | bool | Enable thinking/reasoning mode |
| `load_in_4bit` | bool | Use 4-bit quantization (reduces VRAM) |
| `device_map` | string | Device mapping (`"auto"`, `"cpu"`, `"cuda"`) |
| `torch_dtype` | string | Torch dtype (`"auto"`, `"float16"`, `"bfloat16"`) |
| `knowledge_base_dir` | string | Path to knowledge base directory |
| `min_best_score` | float | Minimum retrieval score threshold |
| `default_top_k` | int | Default number of results |
| `max_related` | int | Max related items in retrieval |
| `index_raw_content` | bool | Index raw content in retriever |
| `lazy_raw_content` | bool | Lazy-load raw content |

#### cURL Example — Enable GPU + 4-bit quantization

```bash
curl -X PUT http://localhost:8000/config -H "Content-Type: application/json" -d "{\"device_map\": \"cuda\", \"load_in_4bit\": true, \"enable_thinking\": false, \"max_new_tokens\": 250}"
```

#### cURL Example — Switch to a different model

```bash
curl -X PUT http://localhost:8000/config -H "Content-Type: application/json" -d "{\"model_name\": \"google/gemma-3-1b-it\", \"enable_thinking\": false}"
```

#### Python Example — Full config update

```python
import requests

# Switch to GPU + 4-bit + lower temperature
resp = requests.put("http://localhost:8000/config", json={
    "device_map": "cuda",
    "load_in_4bit": True,
    "enable_thinking": False,
    "max_new_tokens": 250,
    "temperature": 0.5,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
})

print("Updated config:", resp.json())
```

#### Python Example — Toggle thinking mode

```python
import requests

# Enable thinking mode
requests.put("http://localhost:8000/config", json={"enable_thinking": True})

# Disable thinking mode (faster)
requests.put("http://localhost:8000/config", json={"enable_thinking": False})
```

---

### 4. `GET /models` — List available LLM models

Returns all supported LLM models with their size and VRAM requirements.

#### cURL Example

```bash
curl http://localhost:8000/models
```

#### Python Example

```python
import requests

models = requests.get("http://localhost:8000/models").json()["models"]
for m in models:
    print(f"{m['name']:<40} {m['size']:<12} {m['vram_gb']}")
```

#### Sample Response

```json
{
  "models": [
    {
      "name": "Qwen/Qwen3-1.7B",
      "description": "Qwen3 1.7B — small, fast, good for most tasks",
      "size": "1.7B",
      "vram_gb": "~3.4 GB (fp16) / ~0.9 GB (4-bit)"
    },
    {
      "name": "Qwen/Qwen3-4B",
      "description": "Qwen3 4B — larger, more capable, needs more VRAM",
      "size": "4B",
      "vram_gb": "~8 GB (fp16) / ~2.1 GB (4-bit)"
    }
  ]
}
```

---

## Quick Start — Full Workflow Example

```python
import requests

BASE = "http://localhost:8000"

# 1. Check available models
models = requests.get(f"{BASE}/models").json()["models"]
print(f"Available models: {len(models)}")

# 2. Configure the agent
requests.put(f"{BASE}/config", json={
    "model_name": "Qwen/Qwen3-1.7B",
    "enable_thinking": False,
    "max_new_tokens": 250,
    "load_in_4bit": True,
})

# 3. Submit an issue
issue = {
    "severity": "err",
    "rule_id": "JSON_SCHEMA_VALIDATION_ERROR",
    "message": "ReferencesAndLinks must be array",
    "field": "ReferencesAndLinks",
}

result = requests.post(f"{BASE}/predict", json={
    "user_input": "Explain this to me.",
    "context": issue,
}).json()

print(f"Intent: {result['intent']}")
print(f"Response: {result['response']}")

# 4. Switch to CPU mode
requests.put(f"{BASE}/config", json={
    "device_map": "cpu",
    "load_in_4bit": False,
})
```

---

## Running Tests

```bash
# All tests except slow LLM tests
pytest tests/ -v

# Include slow tests (loads LLM, takes minutes)
pytest tests/ -v --no-skip
```

## Project Structure

```
bids-ai-agent/
  main.py              # FastAPI server (API entry point)
  agent.py             # Agent class (orchestrates the workflow)
  planner.py           # Planner (intent routing + tool selection)
  laya_decision.py     # Laya decision engine (intent + tool routing)
  llm_client.py        # LLM client (model loading + generation)
  retriever.py         # Knowledge base retriever (JSONL-based)
  tool_registry.py     # Tool registry (validate, repair, rename)
  intent_classifier.py # Legacy intent classifier (deprecated)
  model_manager.py     # Model lifecycle (lazy load + offload)
  config.py            # Central configuration
  utils.py             # Utility functions
  knowledge-base/      # BIDS knowledge base (JSONL files)
  tests/               # Test suite
```
