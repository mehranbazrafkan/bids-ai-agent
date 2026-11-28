# BIDS AI Agent

AI agent for explaining issues found in MRI datasets according to the BIDS
specification. Answer text uses a local instruction-tuned LLM by default;
intent routing is a deterministic guard with LLM fallback; knowledge-base
retrieval runs on a persistent embedding store (cosine similarity only), with
the original lexical scorer kept as a selectable alternative.

**This is the explanation-only phase.** The agent never executes tools: it
does not fix, rename, validate, or modify anything, and it never claims it
did. Action requests are answered with a grounded explanation that states
execution is unavailable (`execution_status: "not_available"`).

## Architecture

```
User Input (no issue context)
        |
        v
 [Intent router]  auto = guard + LLM fallback (EXPLAIN / FIX / UNCLEAR)
        |          explain/fix modes = forced, classifier never called
        |
        +---> explain --> [Retriever + Issue Context] --> [LLM] --> grounded explanation
        |
        +---> fix -------> [Retriever] --> [LLM] --> explanation + "execution
        |                                             not available" disclosure
        +---> unclear ---> no issue context: clarification question
                            issue context:   safe explanation-only answer
```

- **Pipeline** — the explain / fix / unclear-with-context paths run as an
  explicit LCEL graph in `planner.py` (`langchain-core` orchestrates; it
  does not decide anything): a `RunnableBranch` selects the intent lane and
  sets the truthful prompt banners → `bids_evidence` (retrieval + system-half
  rendering, deterministic) → a `RunnableParallel` runs `BIDS_PROMPT | llm`
  while retaining the prepared state → `bids_result` (deterministic contract
  check, repair/fallback, and the honest `PlannerResult` metadata). The
  clarification shortcut (unclear + no issue context) returns before the
  pipeline — no LLM call on an ungrounded answer. A `StepLogger` callback
  (`observability.py`) records every step's name and duration to the
  `bids_agent.pipeline` logger; names and durations only, never step content.
- **Intent routing** — `intent.mode` is `auto` by default and
  `intent.classifier` is `llm`: a deterministic guard resolves unmistakable
  question/imperative forms, explicitly separating "How do I fix this?"
  (EXPLAIN) from "Fix it for me" (FIX), which is the distinction Laya got
  wrong. Only ambiguous paraphrases go to a compact few-shot prompt asking
  the main local model for `EXPLAIN`, `FIX`, or `UNCLEAR`. Set
  `intent.mode` to `explain` or `fix` to
  force one lane; forced modes never call a classifier. Laya and the legacy
  zero-shot NLI encoder remain selectable for comparison via
  `intent.classifier`, but they are no longer the default. The classifier sees
  only the user request: issue metadata and conversation history are
  deliberately withheld from routing and used later for retrieval and answer
  generation. The classification happens **once** per request and that decision
  is what the API reports — nothing is re-classified later.
- **Retriever** — two interchangeable backends over the single knowledge
  file `knowledge-base/knowledge_base.jsonl`:
  - `embedding` (default) — the vector store chooses the candidates: the
    records with the highest cosine similarity, and nothing else
    (`embedding_retriever.py`). No keyword-scored record is unioned in. Within
    that candidate pool an exact rule/field match is ranked ahead, because an
    encoder alone cannot tell the record documenting `JSON_INVALID` from a
    dozen other JSON errors. Vectors are persisted as
    `vectors.npy` + `manifest.json` in `retrieval.vector_store_dir` and are
    rebuilt automatically when the KB hash, embedding model, pooling mode,
    record order, or vector shape changes — and, because the agent is
    long-running, as soon as a retrieval notices that `knowledge_base.jsonl`
    changed on disk. Old files are removed before the new store is moved into
    place, so a KB/model change cannot leave stale or duplicate vectors
    behind.
  - `lexical` — explainable, structured scoring with rule-code and field
    exact matches, then token overlap (`retriever.py`).

  `kb_retriever.KBRetriever` is a thin `BaseRetriever` facade over whichever
  core is configured: the pipeline speaks the LangChain retriever interface,
  but every call still goes through that core (same assessments, same
  applicability rejection, same explicit-miss outcome).
- **LLM** — generates the explanation text from retrieved evidence and, in
  auto mode, performs the short intent classification. The answer path makes
  no routing decisions. `local_chat_model.LocalChatModel` is a
  `BaseChatModel` wrapper around the existing `llm_client.LLM` backend: the
  pipeline hands it LangChain messages, and it reassembles the exact flat
  `chat(system_prompt, user_prompt, history=...)` call — same prompt bytes,
  same lazy loading, quantization and prompt recording. Intent calls use a
  separate greedy path with a token cap and never touch the prompt record.
- **langchain-core only** — deliberately the core package, not full
  LangChain: no chat-provider integrations, no autonomous agent loop or tool
  calling. The model is a local 1.7B-class checkpoint, the system is
  explanation-only, and free-form tool selection would reintroduce exactly
  the hallucinated-action risk the contract/output gate exists to prevent.
  The LCEL graph provides typed state, branch routing, parallel assembly,
  retriever/chat-model adapters, and callback-based observability without
  giving the model control over retrieval or execution.
- **Answer contract + output gate** — the judgements a small model gets
  wrong (is this field required/recommended/optional? is this a BIDS rule
  or only a validator/manager check? may a concrete value or an example be
  shown? must the answer admit the KB has no record?) are computed
  deterministically in `answer_contract.py` and rendered into the prompt
  as a compact `[ANSWER CHECKLIST]` block at the end of the system prompt.
  `output_gate.py` then validates the generated answer against that
  contract. Violations are repaired
  cheapest-first: mechanical repair (`answer_repair.py`), one model repair
  pass, then a template built only from the contract. The delivered answer
  is always contract-clean; the diagnostics record the tier that answered
  and every original violation. See **Output quality** below.

## Installation

```bash
uv pip install -r requirements.txt    # or: pip install -r requirements.txt
```

This installs Flask, **langchain-core** (orchestration only — see the
`langchain-core` note above), PyTorch, transformers, accelerate, NumPy,
bitsandbytes (4/8-bit quantization) and pytest. Laya is listed as an optional
legacy classifier. Nothing else is required; `torchvision` is intentionally
not a dependency (see the notes in `requirements.txt`).

**GPU (CUDA):** `torch` is pinned to the PyTorch **CUDA 12.6** wheel
(`torch==2.14.1+cu126`, fetched from `download.pytorch.org` by the
`--find-links` line in `requirements.txt`), because PyPI only ships
CPU-only torch wheels on Windows. The wheel bundles the CUDA runtime, so no
separate CUDA toolkit install is needed — a recent NVIDIA driver is
enough. Verify after starting the server:

```bash
curl http://localhost:8000/config     # look for "hardware": { "backend": "cuda", ... }
```

For a CPU-only machine (or on macOS), delete the `--find-links` line and
drop the `+cu126` pin from `torch` in `requirements.txt`.

Models are loaded from the local Hugging Face cache on first use. The
cache directory is the standard huggingface_hub one: `HF_HUB_CACHE`
wins, then `HF_HOME`, then the default under the user's home. A caller
that starts this server itself (the desktop app does) points
`HF_HUB_CACHE` at its own per-user models folder, so every downloaded
model is one visible `models--<org>--<name>` folder there — copied in
by hand or fetched through `POST /models/download`, the result is the
same.

## Running the Server

```bash
python main.py
```

The Flask server starts on `http://localhost:8000` (override with
`BIDS_AGENT_HOST` / `BIDS_AGENT_PORT`). There is no `/docs` page — this is
Flask, not FastAPI.

Settings saved through `PUT /config` are written to `config.json` in the
server's working directory (or `$BIDS_AGENT_CONFIG_FILE`) and loaded again
on the next start; the file is git-ignored. `"quantization": "none"` means
the checkpoint's own dtype — bfloat16 (16-bit) for Qwen3-1.7B — not
float32. The optional `"prompt"` section holds the desktop app's system
prompt override and the independent `include_dataset_structure` and
`include_file_content` switches; an empty `system_prompt` keeps the built-in
instructions in `system_prompt.py`.

On a 4 GB GPU, Qwen3-1.7B's bf16 weights alone are about 3.4 GB. Windows,
the desktop, and the CUDA context leave less than 4 GB free, so
`device_map="auto"` offloads some layers to CPU and accelerate logs
`Some parameters are on the meta device because they were offloaded to
the cpu`. That is expected CPU offload, not a failed load; generation is
slower because those layers run on the CPU. Use
`"quantization": "4bit"` (about 0.9 GB of weights) if the whole model must
stay on the GPU. Setting `"embedding_device": "cpu"` also reserves VRAM
for the LLM.

For a 4-bit/8-bit model on a single GPU, the loader replaces
`device_map="auto"` with an explicit GPU-only map (`{"": 0}`) and enables
double quantization. This avoids accelerate's CPU-offload path, which is
currently broken with bitsandbytes quantized weights (`Tensor.item()
cannot be called on meta tensors`). If the quantized model still does not
fit, the result is a normal CUDA out-of-memory error; free VRAM with
`"embedding_device": "cpu"`, close GPU-heavy apps, or switch to a
smaller model.

---

## Inspecting what goes into the model

The **most recent** request is recorded in memory and returned by
`POST /predict` instead of being written to a file. The response carries:

| Field | Type | Description |
|---|---|---|
| `log` | object? | The full record of this exchange: `system_prompt` (exactly as the planner assembled it — issue context, dataset structure, file digest, retrieved knowledge, any conversation-history section), `user_prompt` (the question the user actually asked) and `output` (the answer the app shows — the model's draft after the quality gate, which can repair it or substitute the grounded template). `null` when no model call was made (the clarification shortcut). |
| `model` | string | The checkpoint that wrote the answer |
| `prompt_tokens` | int? | Tokens handed to the model |
| `generated_tokens` | int? | Tokens the model produced |
| `generation_seconds` | float? | Wall-clock seconds of the whole generation |
| `model_load_seconds` | float | Seconds spent reading the checkpoint before generation began; `0.0` when it was already in memory |

Each request reports its own record; the agent saves no chat turns
itself. The desktop app keeps the record behind the "Show full data"
button under each answer, so there is no `prompt_log.txt` to find by
hand, and it appends every question and answer to a JSONL file in its
chat-log folder (Settings → AI Agent), one file per finding.

Settings → AI Agent also exports the request side of a whole validation
report: "Save prompt logs for all warnings and errors" calls
`POST /prompt/log` once per error and warning and writes one UTF-8 `.log`
file per finding (plus a manifest) into a timestamped `full-data` folder
inside the chat-log path. Nothing is generated — each file holds the full
system prompt, the user prompt and the retrieval outcome a real question
would carry — which makes the files usable as fine-tuning input and as a
view of where the knowledge base helped.

---

## API Endpoints

### 1. `POST /predict` — Get a response for an issue

**Request body:**

| Field | Type | Description |
|---|---|---|
| `user_input` | string | The user's prompt or question |
| `context` | object | The issue / error context (JSON object) |
| `history` | list? | Optional transcript: `[{"role": "user"\|"assistant", "content": "..."}]` (malformed entries are dropped). The whole transcript is forwarded as chat turns between the system prompt and the current question, **and** rendered into the system prompt as a `[Conversation History]` section built from those same turns — never truncated. |
| `dataset_structure` | string? | Optional condensed tree of the dataset the finding sits in (the desktop app compiles it: top-level entries, the directory skeleton, a handful of files per folder, the finding's folder listed properly). Rendered as its own `[Dataset Structure]` section. |
| `file_content` | string? | Optional compiled digest of the file the finding is about — a sidecar as its keys, a table as its columns and first rows, a binary file named and sized but **not** read. Rendered as its own `[File Content]` section. |

All three optional fields are omitted when there is nothing to send, and
each renders as its own section: an absent field costs no lines rather
than an empty header.

**Response:**

| Field | Type | Description |
|---|---|---|
| `response` | string | The generated response from the agent |
| `intent` | string | The intent actually used: `"explain"`, `"fix"`, or `"unclear"` |
| `execution_status` | string? | `"not_available"` when the user asked for an action (`fix`), else `null` |
| `classification_source` | string | `"llm"` / `"nli"` / `"laya"` for the selected auto classifier, `"guard"` when the deterministic explain/action guard resolved the request, `"configured"` when `intent.mode` forces explain/fix, `"manual"` when no classifier is configured, and `"fallback"` when neither could decide |
| `model` | string | The checkpoint that wrote the answer |
| `prompt_tokens` | int? | Tokens handed to the model |
| `generated_tokens` | int? | Tokens the model produced |
| `generation_seconds` | float? | Wall-clock seconds of the whole generation |
| `model_load_seconds` | float | How long the checkpoint took to read before generation began; `0.0` when it was already in memory. Shown in the app's full record for the answer (the "Show full data" window), because a cold first answer is slow for both reasons |
| `log` | object? | The full prompt record of this exchange: `system_prompt`, `user_prompt`, `output` (see *Inspecting what goes into the model*). `null` when no model call was made |

Internal diagnostics (retrieval assessments, contract checks, repair state)
are not part of the API response.

With the default configuration (`intent.mode: auto`,
`intent.classifier: llm`) each request is classified once by the hybrid
guard/LLM router. High-confidence forms report `classification_source:
"guard"`; ambiguous requests that reach the model report `"llm"`. Forced
`explain`/`fix` modes report `"configured"`. A `fallback` classification is
never presented as a model decision.

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
  "execution_status": null,
  "classification_source": "manual"
}
```

An action request — produced by the LLM router or forced with
`intent.mode: "fix"` — looks like this; note that nothing was executed and
the response says so:

```json
{
  "response": "I cannot apply changes to your dataset in this version. ...",
  "intent": "fix",
  "execution_status": "not_available",
  "classification_source": "llm"
}
```

---

### 2. `POST /prompt` — Assemble a prompt without generating

Returns the request side of what `/predict` would send for the same issue
and question — the same intent decision, retrieval, answer contract and
system-prompt rendering, stopping before the model call. **No checkpoint is
loaded and no answer is generated.**

This JSON form is useful for dashboards and scripts. The desktop app uses
`POST /prompt/log` below from Settings → AI Agent ("Save prompt logs for
all warnings and errors") to write one complete `.log` file per error and
warning in the current validation report: what the model would be fed,
without spending GPU time. Each log is a ready fine-tuning prompt, and the
retrieval evidence shows where the knowledge base helped and where it did
not.

**Request body** is the same as `/predict`:

| Field | Type | Description |
|---|---|---|
| `user_input` | string | The question the prompt is for |
| `context` | object | The issue / error context |
| `history` | list? | Optional earlier turns; usually omitted for a first ask |
| `dataset_structure` | string? | Optional condensed dataset tree |
| `file_content` | string? | Optional compiled file digest |

**Response:**

| Field | Type | Description |
|---|---|---|
| `system_prompt` | string | The exact system half `/predict` would use, including the retrieved knowledge and every evidence section |
| `user_prompt` | string | The question, unchanged |
| `intent` | string | The intent the request would run with |
| `classification_source` | string | How that intent was decided (as in `/predict`) |
| `prompt_available` | bool | `false` when `/predict` would take the clarification shortcut instead of building a prompt |
| `retrieval_found` | bool? | Whether retrieval found applicable knowledge |
| `retrieval_quality` | string? | The retriever's assessment of what it found |

No `response` field: there is no generation to report.

#### cURL Example

```bash
curl -X POST http://localhost:8000/prompt -H "Content-Type: application/json" -d "{\"user_input\": \"Explain this error to me.\", \"context\": {\"severity\": \"err\", \"rule_id\": \"JSON_SCHEMA_VALIDATION_ERROR\", \"message\": \"ReferencesAndLinks must be array\", \"field\": \"ReferencesAndLinks\"}}"
```

#### `POST /prompt/log` — the same prompt as a readable file

Accepts the same body and returns `text/plain` instead of JSON. The text
starts with `[System prompt]`, holds the complete system message (the
instructions and every evidence section), and ends with `[User prompt]`.
There is deliberately no `[Agent output]` section: this is the request side
of a fine-tuning pair. The desktop app writes one file per finding as
`<finding>.log` inside its timestamped export folder.

```bash
curl -X POST http://localhost:8000/prompt/log -H "Content-Type: application/json" -d "{\"user_input\": \"Explain this error to me.\", \"context\": {\"severity\": \"err\", \"rule_id\": \"JSON_SCHEMA_VALIDATION_ERROR\", \"message\": \"ReferencesAndLinks must be array\", \"field\": \"ReferencesAndLinks\"}}"
```

---

### 3. `GET /config` — View current configuration

Returns the current LLM, retrieval, intent and quality settings plus detected
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
print("Retriever:", config["retrieval"]["retriever_type"])
print("Intent mode:", config["intent"]["mode"])
```

#### Sample Response

```json
{
  "llm": {
    "model_name": "Qwen/Qwen3-1.7B",
    "max_new_tokens": 512,
    "temperature": 0.2,
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
    "default_top_k": 3,
    "index_raw_content": false,
    "lazy_raw_content": true,
    "retriever_type": "embedding",
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "embedding_device": "auto",
    "embedding_pooling": "cls",
    "embedding_max_length": 512,
    "embedding_batch_size": 32,
    "embedding_query_prefix": "",
    "embedding_min_score": 0.35,
    "embedding_candidate_k": 12,
    "vector_store_dir": "./knowledge-base/vector_store"
  },
  "intent": {
    "mode": "auto",
    "classifier": "llm"
  },
  "laya": {
    "model": "convaiinnovations/laya",
    "classify": false,
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

### 4. `PUT /config` — Update configuration

Update any subset of configuration fields. Only provided fields are
changed. The agent is reset automatically, the new values are written to
`config.json` (or `$BIDS_AGENT_CONFIG_FILE`) so they survive a restart,
and a change to a load-time field (`model_name`, `quantization`,
`load_in_4bit`, `device_map`, `torch_dtype`) unloads the cached model so
the next `/predict` loads the selected variant.

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
| `quantization` | string | `"none"` (checkpoint dtype, bf16 = 16-bit), `"4bit"`, or `"8bit"` |
| `device_map` | string | Device mapping (`"auto"`, `"cpu"`, `"cuda"`) |
| `torch_dtype` | string | Torch dtype (`"auto"`, `"float16"`, `"bfloat16"`) |
| `knowledge_base_dir` | string | Path to knowledge base directory (only `knowledge_base.jsonl` is read) |
| `min_best_score` | float | Minimum retrieval score threshold |
| `default_top_k` | int | Default number of retrieved records (3: room for a direct record plus background context) |
| `index_raw_content` | bool | Index raw content in retriever |
| `lazy_raw_content` | bool | Lazy-load raw content |
| `retriever_type` | string | `"embedding"` (default, vector store) or `"lexical"` |
| `embedding_model` | string | Encoder used by the embedding backend (default `BAAI/bge-small-en-v1.5`) |
| `embedding_device` | string | `"auto"`, `"cpu"`, or `"cuda"` |
| `embedding_pooling` | string | `"cls"` for BGE, `"mean"` for most other encoders |
| `embedding_max_length` | int | Token truncation limit (default 512) |
| `embedding_batch_size` | int | Documents per encoding batch |
| `embedding_query_prefix` | string | Optional instruction prepended to queries only |
| `embedding_min_score` | float | Cosine-similarity floor; below it retrieval is an explicit miss |
| `embedding_candidate_k` | int | Semantic candidates inspected before applicability/dedupe filtering |
| `vector_store_dir` | string | Directory for `vectors.npy` + `manifest.json` |
| `intent.mode` | string | `"auto"` (default), `"explain"` (force), or `"fix"` (force) |
| `intent.classifier` | string | `"llm"` (default), `"nli"`, or `"laya"` |
| `laya.model` | string | Legacy Laya checkpoint name |
| `laya.classify` | bool | Legacy switch; only used when `intent.classifier` is `"laya"` or a Laya object is injected |
| `laya.confidence_threshold` | float | Legacy Laya acceptance threshold |

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

### 5. `GET /models` — List available LLM models

Returns the supported LLM models with their size, VRAM requirements and
whether each is already usable from the local cache, plus the cache
directory itself:

```bash
curl http://localhost:8000/models
```

```json
{
  "models": [
    {"name": "Qwen/Qwen3-1.7B", "description": "Qwen3 1.7B — small, fast, good for most tasks", "size": "1.7B", "vram_gb": "~3.4 GB (fp16) / ~0.9 GB (4-bit)", "downloaded": true}
  ],
  "models_dir": "C:/Users/<user>/AppData/Local/bidsmgr/bidsmgr/models"
}
```

`downloaded` is a directory listing of the cache (no model, no network);
`models_dir` is the same cache huggingface_hub resolves (`HF_HUB_CACHE`,
then `HF_HOME`, then the default under the home). The desktop app
spawns this server with `HF_HUB_CACHE` pointed at its own per-user
models folder, so everything here — copied in by hand or fetched below —
is one visible folder per model.

### 6. `POST /models/download` — Fetch a model in the background

**Request body:** `{"name": "<repo id>"}`. The download runs in its own
thread and the answer comes back immediately with the job's status:

```json
{"name": "Qwen/Qwen3-1.7B", "status": "downloading", "downloaded_bytes": 0, "total_bytes": null, "error": "", "started": 0.0, "finished": 0.0}
```

Repeating a request that is already running (or already local) answers
with the current status rather than starting a second fetch.

### 7. `GET /models/download/status` — Progress of that download

```json
{"name": "Qwen/Qwen3-1.7B", "status": "downloading", "downloaded_bytes": 812000000, "total_bytes": 3400000000, "error": "", "started": 1700000000.0, "finished": 0.0}
```

`status` is `idle` | `downloading` | `done` | `error`. `total_bytes` is
the sum of the repo's file sizes (asked of the Hub once, before the
fetch), or `null` when it cannot be established — the client then shows
an indeterminate bar with the bytes counted so far. `downloaded_bytes`
is the size of what is actually on disk under the model's `blobs`
directory, sampled while the fetch runs.

### 8. `GET /health` — Liveness check

Returns a cheap `200` without loading any model. Health-probe requests are
excluded from the request-duration log, so a short poll interval does not
flood the console with `GET /health` lines.

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok"}
```

### 9. `GET /status` — What the process is doing

`POST /predict` does not answer until the answer exists, so a client
watching its own request cannot tell a checkpoint being read into memory
from an answer being written — on a CPU the first can take a minute. This
endpoint says which, and how long generation has been going:

```bash
curl http://localhost:8000/status
```

```json
{"phase": "generating", "generating_for": 4.2}
```

`phase` is `idle` | `loading` | `generating`. `generating_for` is `null`
unless the phase is `generating`; it is a duration rather than a
timestamp so the client never has to trust its clock to match the
agent's. Polling during a request is expected — status requests, like
health probes, are left out of the request-duration log.

---

## Retrieval Backends

`retrieval.retriever_type` selects the ranking strategy:

- `"embedding"` (default) — the same records are encoded once with
  `BAAI/bge-small-en-v1.5` (configurable). The vector store alone chooses the
  candidates: the `embedding_candidate_k` records with the highest cosine
  similarity, with no keyword-scored record unioned in. Within that pool an
  exact rule/field match is ranked ahead (cosine similarity is reported as
  the score), because the encoder cannot tell the record documenting
  `JSON_INVALID` from a dozen other JSON errors; the retriever never adds a
  record the vector store did not return. The reuse of the lexical loader,
  applicability selectors, assessments and formatter is what makes those
  `rule_exact` / `field_exact` labels available. A selected record with no
  exact identifier is reported as `quality: "semantic"` and the prompt says
  similarity alone is not proof.
- `"lexical"` — the original deterministic scorer. No model, no vector
  files; rule-code and field identifiers dominate. Useful for comparisons
  and for a deployment that cannot load an encoder.

**Vector cache.** The first embedding run writes
`retrieval.vector_store_dir/vectors.npy` and `manifest.json`. The manifest
stores a SHA-256 of `knowledge_base.jsonl`, the model name, pooling mode,
max length, record count/order hash and vector dimension. The retriever
validates all of them on startup, and — because the KB can be regenerated
while the agent runs — it also compares the file's `(mtime, size)` on every
retrieval; a change reloads the records and regenerates the whole store
before that query is answered. Old files are deleted before the new ones are
written. The generated directory is git-ignored. To force a rebuild, delete
the directory or change `embedding_model` (the next startup/request rebuilds
it).

```bash
# Use the deterministic keyword scorer instead (comparison / no encoder)
curl -X PUT http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{"retrieval": {"retriever_type": "lexical"}}'
```

The first embedding start downloads the small encoder (~130 MB) and builds
the vectors once; later starts load the cache and only encode the query.

## Intent Classification

The default (`intent.mode: "auto"`, `intent.classifier: "llm"`) is a hybrid
router. A deterministic, high-precision guard resolves the forms a small
local model still confuses, and only genuinely ambiguous paraphrases are sent
to the already-loaded LLM as one compact few-shot prompt.

The distinction Laya struggled with is therefore explicit:

- “What does this error mean?”, “Why did this fail?”, and “How do I fix
  this?” → **EXPLAIN**;
- “Fix it for me”, “Rename every subject”, and “Validate and correct my
  dataset” → **FIX**;
- greetings, thanks, and vague text → **UNCLEAR**.

When the guard resolves a request it reports `classification_source: "guard"`
and no model call is made. An ambiguous request goes to the model with greedy
decoding, an 8-token cap, thinking disabled, and a suspended prompt log; it is
not the answer generation call. A malformed or ambiguous model label (for
example, "EXPLAIN or FIX") degrades to conservative `UNCLEAR` with
`classification_source: "fallback"` rather than guessing.

### Forced explain/fix modes

```bash
# Never call a classifier: every request is explanation-only.
curl -X PUT http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{"intent": {"mode": "explain"}}'

# Every request is treated as an action request; the answer still discloses
# that execution is unavailable.
curl -X PUT http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{"intent": {"mode": "fix"}}'

# Back to automatic LLM routing.
curl -X PUT http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{"intent": {"mode": "auto", "classifier": "llm"}}'
```

Forced modes report `classification_source: "configured"` and never load or
call Laya/NLI/the LLM classifier.

### Legacy classifiers

`intent.classifier` also accepts `"nli"` (the old zero-shot DeBERTa encoder
in `intent_classifier.py`) and `"laya"` (the three-way Laya engine in
`laya_decision.py`). Set `laya.classify: true` when selecting Laya. These
paths are retained for comparison; the default no longer depends on them.
The old Laya acceptance gate, measurements and evaluation runner
(`eval_intent.py`) remain documented in `laya_decision.py` and the evaluation
dataset `tests/data/intent_eval.jsonl`.

---

## Running Tests

```bash
pytest tests/ -v              # default: fast tests only (slow are deselected)
pytest -m slow                # tests that load the real LLM / Laya checkpoint
pytest tests/ -v -m "not slow" # explicit form of the default
```

The `slow` marker is configured in `pytest.ini` (`addopts = -m "not
slow"`); slow tests load real models and take minutes. All API, planner,
lexical/embedding retrieval, intent-routing, prompt, pipeline and adapter
tests are mocked (the embedding tests use a deterministic fake encoder) and
run without any model download or GPU. The prompt wording and section layout
are pinned byte-for-byte by `tests/test_regression_scenarios.py` and
`tests/test_prompts.py`, so a change to the prompt fails loudly rather
than silently altering model behavior.

---

## Output quality

The retrieval layer already quarantines inapplicable records correctly; the
failures in the recorded prompt logs came from the *generation* layer
inverting facts ("recommended" → "required", a validator warning → "error", a
missing record → an invented rule) and fabricating values (a HED version, a
license filename, an acknowledgement naming a person). The fix moves those
judgements out of the model and into deterministic code:

1. `answer_contract.py` derives an `AnswerContract` per request from the issue
   and the retrieved records: `severity`, the field's `bids_level`, the
   record's `authority` (spec / validator / manager), whether there is
   **direct** evidence, whether the value is a `TODO` placeholder, whether an
   example may be shown, and which facts must be stated as unknown. Authority
   is taken only from a record that documents the reported rule; a field
   definition or application note cannot upgrade a validator or manager
   finding. The contract also carries the direct/supporting/background record
   ids, per-source authorities, detected conflicts and BIDS-version notes.
2. `prompts.render_system_content()` renders that contract as a compact
   `[ANSWER CHECKLIST — ...]` block at the end of the system prompt, after
   the issue, dataset tree, file digest and retrieved evidence. The model
   copies slots instead of re-deriving them. A per-request
   `[Evidence Summary]` and `[Grounding and Provenance Rules]` block labels
   each record as **direct**, **supporting** or **background**, states when
   no direct record exists, and tells the model what each role may establish.
3. `output_gate.py` validates the generated answer against the contract. It
   flags a missing "the knowledge base does not document this rule" sentence,
   a concrete value the evidence does not fix, an example where the field has
   no documented shape, invented JSON properties/TSV columns/example values,
   TSV content presented as JSON, duplicate or contradictory examples, a
   required/recommended inversion, a warning called an error, a
   BIDS-requirement overclaim, a fabricated person/institution, a rule code
   neither the issue nor the evidence supplies, attribution to a source the
   evidence does not contain, duplicated or empty sections, and any
   execution/fix claim.
4. `answer_repair.py` applies the cheapest fix first: mechanical repair
   (strip the example, replace the invented value with a visible placeholder,
   append the mandated sentence), then **one** model repair pass, then a
   grounded template built only from the contract. The user never receives a
   violating answer; the internal diagnostics trace records the original
   violations, whether it was repaired, and whether the template answered.

The gate is on by default and can be disabled with
`PUT /config {"quality": {"enabled": false}}` (or the
`BIDS_AGENT_QUALITY_ENABLED=false` environment variable) when comparing raw
model behaviour.

### Evidence grounding

`retriever.py` assigns every surviving record a deterministic grounding role
in addition to the legacy `relevance` label:

* **direct** — documents the reported rule or the exact constraint named by
  the issue message. Only a direct record may establish the rule's authority.
* **supporting** — documents the affected field, the exact format constraint,
  or the BIDS-Manager application behaviour behind the finding. It may explain
  the field, but it is not documentation for the reported rule.
* **background** — examples, unrelated rules and general file rules. It is
  context only and must never become the issue's definition or cause.
* **inapplicable** — excluded by its own BIDS selectors and shown only in the
  quarantine section.

`missing_direct` is a retrieval state, not a per-record label: retrieval found
material, but no record documents the reported rule. In that case the prompt
and contract require an explicit acknowledgement instead of a plausible rule.

Every assessment carries provenance copied from the KB record: record id,
authority class/label, source id, source file/path, section, revision,
retrieval time, and the record's BIDS/schema versions. The formatter renders
`Authority:`, `Source:` and `Trace:` lines; diagnostics expose the same data
under `provenance`. `tests/test_grounding.py` covers direct vs supporting vs
background, missing evidence, background misuse, conflicts, unsupported rule
citations and provenance.

### Example grounding

`AnswerContract.example_policy` adds a format-aware permission to every
request. It is derived deterministically from the affected file and the
retrieved records:

* **JSON** — a compact Markdown table (`Key | Expected type | Purpose |
  Evidence`) is preferred. A JSON snippet is allowed only when a record
  documents the field's shape; the contract carries the only accepted property
  names (for example `SourceDatasets`, `URL`, `DOI`, `Version`) and the gate
  rejects any other key. `key`/`value` cannot replace a documented structure.
* **TSV** — the exact supplied header is the allowed column set. The gate
  rejects extra columns, JSON content in a TSV block, and values that are not
  in the supplied file or an explicit placeholder.
* **Unknown/other format** — no file-content code block is allowed; a table or
  prose explanation is used instead.

The prompt policy distinguishes a real supplied value, a `<...>` replacement
placeholder, an illustrative value, and an unknown value. `TODO`, `<value>`
and `unknown` are always placeholders. If a reliable example cannot be built,
the Example section is omitted. The gate also detects repeated examples and
conflicting values for the same field. `tests/test_examples.py` covers
`TOO_FEW_AUTHORS` (no inventing `John Doe`/`Jane Smith`),
`JSON_KEY_RECOMMENDED` for `SourceDatasets` (no `key`/`value` structure),
TSV columns and values, placeholder handling, omission, repetition and
contradiction.

### Evaluation

`tests/data/quality_set.jsonl` is a frozen, stratified set of 36 findings:
10 specification-defined validator rules, 20 implementation-only ones, and the
6 manager `TODO` cases. `eval_quality.py` scores answers with eight binary
criteria (contract-clean, no execution claim, no fabricated entity, modality,
severity, no unsupported value, declares the unknown, complete structure) and
reports each with a Wilson 95% confidence interval.

```bash
python eval_quality.py --baseline          # the four recorded Phase-2 logs
python eval_quality.py --audit-kb          # retrieval/contract coverage
python eval_quality.py --live --save-responses responses.json
python eval_quality.py --replay responses.json --min-score 0.9
```

`baseline_scores.json` is the starting point: **50.0%** overall, with
`contract_clean 0/4`, `declares_unknown 0/4`, and one fabrication. The target
is **≥ 90% on every criterion, zero fabrications, and `fallback_used = false`**
for every scenario in the slow suite — the last condition is what forces the
model itself to conform instead of letting the safety net answer. The audit
currently reports **0 retrieval misses and 0 missing contracts** across the
36 findings. Manager rules are documented when the KB carries a
`validation_issue` record (for example, `bidsmgr.todo_placeholder`); when a
manager code has no record, the contract switches to the "no direct record"
state and requires the answer to say so.

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
  agent.py                     # Agent class (wires LLM, retriever, intent router, planner)
  prompts.py                   # Prompt content + BIDS_PROMPT template (single home for the wording)
  planner.py                   # Planner (one intent decision -> branched LCEL graph -> result)
  intent.py                    # Default guard + LLM intent router + forced modes
  answer_contract.py           # Per-request AnswerContract derived from issue + retrieved records
  output_gate.py               # Deterministic validation of an answer against its contract
  answer_repair.py             # Mechanical repair, model repair prompt, grounded template
  eval_quality.py              # Quality harness (baseline/replay/live/audit, Wilson CIs)
  baseline_scores.json         # Frozen score of the four recorded Phase-2 logs (50%)
  local_chat_model.py          # BaseChatModel wrapper over the local LLM backend
  kb_retriever.py              # BaseRetriever facade over the knowledge-base retriever
  observability.py             # StepLogger callback (step names + durations)
  retrieval_factory.py         # Selects the lexical or embedding backend from config
  embedding_retriever.py       # Semantic retriever + persistent vector cache (vectors.npy + manifest)
  laya_decision.py             # Legacy Laya intent classifier (3-way choice + acceptance gate)
  llm_client.py                # LLM backend (model loading + generation + short classify path)
  retriever.py                 # Lexical/structured knowledge-base retriever (explainable scoring)
  applicability.py             # Deterministic selector evaluation (which records apply to the issue's file)
  tool_registry.py             # Tool registry kept for a future execution phase (never dispatched here)
  intent_classifier.py         # Legacy zero-shot NLI classifier (selectable via intent.classifier="nli")
  model_manager.py             # Model lifecycle (lazy load + offload)
  config.py                    # Central configuration
  utils.py                     # Utility functions
  eval_intent.py               # Legacy Laya/intent evaluation runner
  benchmark_timing.py          # Timing benchmark script
  pytest.ini                   # Test markers (slow excluded by default)
  knowledge-base/              # Single-file JSONL KB + generated vector_store/
  tests/                       # Test suite (data/ holds the intent eval set)
```
