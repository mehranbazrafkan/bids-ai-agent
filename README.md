# BIDS AI Agent

AI agent for explaining issues found in MRI datasets according to the BIDS
specification. Answer text uses a local instruction-tuned LLM by default;
intent routing is a deterministic guard with LLM fallback; the lexical
knowledge-base retriever remains available, and a persistent embedding
retriever can be selected in the configuration.

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
  - `lexical` (default) — explainable, structured scoring with rule-code and
    field exact matches, then token overlap (`retriever.py`);
  - `embedding` — semantic cosine ranking over the same records
    (`embedding_retriever.py`). The lexical loader/assessment/formatter are
    reused; only ranking changes. Vectors are persisted as
    `vectors.npy` + `manifest.json` in `retrieval.vector_store_dir` and are
    rebuilt automatically when the KB hash, embedding model, pooling mode,
    record order, or vector shape changes. Old files are removed before the
    new store is moved into place, so a KB/model change cannot leave stale or
    duplicate vectors behind.

  `kb_retriever.KBRetriever` is a thin `BaseRetriever` facade over whichever
  core is configured: the pipeline speaks the LangChain retriever interface,
  but every call still goes through the deterministic core (same
  assessments, same applicability rejection, same explicit-miss outcome).
- **LLM** — generates the explanation text from retrieved evidence and, in
  auto mode, performs the short intent classification. The answer path makes
  no routing decisions. `local_chat_model.LocalChatModel` is a
  `BaseChatModel` wrapper around the existing `llm_client.LLM` backend: the
  pipeline hands it LangChain messages, and it reassembles the exact flat
  `chat(system_prompt, user_prompt, history=...)` call — same prompt bytes,
  same lazy loading, quantization and prompt logging. Intent calls use a
  separate greedy path with a token cap and do not overwrite `prompt_log.txt`.
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

Models are loaded from the local Hugging Face cache on first use.

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
float32.

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

The **most recent** request is written to `prompt_log.txt` next to
`main.py` in three labelled parts: the system prompt exactly as the
planner assembled it (issue context — the error or warning — dataset
structure, file digest, retrieved knowledge and any conversation-history
section), the user prompt (the question the user actually asked), and
the agent output (what the model generated). Each request overwrites
the file, so it always holds only that one exchange — no chat turns are
saved, and no conversation is ever written to disk.

Switch it off with `BIDS_AGENT_LLM_SAVE_PROMPTS=false` or
`{"llm": {"save_prompts": false}}` in `config.json`; if the file cannot
be written, the answer is still produced.

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

### 2. `GET /config` — View current configuration

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
    "retriever_type": "lexical",
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

### 3. `PUT /config` — Update configuration

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
| `retriever_type` | string | `"lexical"` (default) or `"embedding"` |
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

### 5. `GET /health` — Liveness check

Returns a cheap `200` without loading any model. Health-probe requests are
excluded from the request-duration log, so a short poll interval does not
flood the console with `GET /health` lines.

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok"}
```

---

## Retrieval Backends

`retrieval.retriever_type` selects the ranking strategy:

- `"lexical"` (default) — the original deterministic scorer. No model, no
  vector files; rule-code and field identifiers dominate.
- `"embedding"` — the same records are encoded once with
  `BAAI/bge-small-en-v1.5` (configurable) and ranked by cosine similarity.
  The lexical loader, applicability selectors, assessments and formatter are
  reused, so `rule_exact` / `field_exact` labels still take precedence over
  pure similarity. The semantic pool is unioned with the lexical retriever's
  best structured candidates, and exact rule/field matches get a small rerank
  bonus; this prevents a generic error record from outscoring the record that
  documents the issue's actual rule. A semantically selected record with no
  exact identifier is reported as `quality: "semantic"` and the prompt says
  similarity alone is not proof.

**Vector cache.** The first embedding run writes
`retrieval.vector_store_dir/vectors.npy` and `manifest.json`. The manifest
stores a SHA-256 of `knowledge_base.jsonl`, the model name, pooling mode,
max length, record count/order hash and vector dimension. On every startup
the retriever validates all of them; a mismatch deletes the old files and
rebuilds the vector store. The generated directory is git-ignored. To force
a rebuild, delete the directory or change `embedding_model` (the next
startup/request rebuilds it).

```bash
# Switch to semantic retrieval
curl -X PUT http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{"retrieval": {"retriever_type": "embedding", "embedding_device": "auto"}}'
```

The first request after switching may take longer while the vectors are
created; later starts load the cache and only encode the query.

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
   example may be shown, and which facts must be stated as unknown.
2. `prompts.render_system_content()` renders that contract as a compact
   `[ANSWER CHECKLIST — ...]` block at the end of the system prompt, after
   the issue, dataset tree, file digest and retrieved evidence. The model
   copies slots instead of re-deriving them.
3. `output_gate.py` validates the generated answer against the contract. It
   flags a missing "the knowledge base does not document this rule" sentence,
   a concrete value the evidence does not fix, an example where the field has
   no documented shape, a required/recommended inversion, a warning called an
   error, a BIDS-requirement overclaim, a fabricated person/institution,
   duplicated or empty sections, and any execution/fix claim.
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
36 findings; the six `TODO` findings are intentionally "no direct record"
(the `bidsmgr.todo_placeholder` rule is an agent-side check, not BIDS
knowledge), which is exactly the case the contract requires the answer to
state.

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
