# BIDS AI Agent — Deep Analysis and Development Plan

**Status:** analysis and plan only. No code was changed. `git status --short` is empty at commit `a490c86`.

**Scope:** the Python service in this repository (`main.py`, `agent.py`, `planner.py`, retrievers, contract, gate, repair, prompt, KB, evaluation). The desktop/manager application that calls `/predict` is out of scope except where its payload shape affects this service.

---

## How to read this report

Every non-trivial claim is tagged so repository facts are never confused with opinion or outside research.

| Tag | Meaning |
| --- | --- |
| **[V]** | Verified in this repository during this analysis: read in source, executed in a probe, or produced by the test suite. Reproducible via the commands in Appendix A. |
| **[R]** | Recommendation. A design choice, not an observed fact. |
| **[E]** | External research or third-party documentation. Not verified against this repo. See Appendix B. |
| **[I]** | Inference. My reasoning from the evidence; could be wrong. |
| **[?]** | Unknown / not established. |

---

# A. Executive Summary

## A.1 What this system actually is

The repository is **not an autonomous agent**. It is a deterministic, single-pass explanation pipeline with one (sometimes two) LLM calls wrapped in three safety-ish layers:

1. a lexical or embedding retriever over a 3,505-record BIDS knowledge base;
2. an "answer contract" derived from the retrieval;
3. a regex output gate with a repair loop and a template fallback.

The README describes it as an agent and the code uses LangChain Expression Language (`RunnableBranch`, `RunnableParallel`) in `planner.py`, but the control flow is a straight line with no tool use, no planning loop, and no state machine. **[V]**

That is a good shape for the problem. My main recommendation is to **keep it** and make the three layers actually sound — not to add agents, graphs, or frameworks.

## A.2 What is genuinely good and must be preserved

- **Explanation-only capability boundary.** The system refuses to pretend it validated or repaired anything. `tool_registry.py` deliberately holds an empty registry, and the prompt states the boundary explicitly. **[V]**
- **Honest "no evidence" path.** `requires_no_direct_statement` forces the sentence *"The knowledge base does not document this rule."* when no direct record exists, and the gate enforces it. This is exactly the abstention behaviour the literature recommends. **[V][E]**
- **Deterministic template fallback.** `render_grounded_template()` in `answer_repair.py:171` builds a complete answer from the contract alone and therefore cannot fabricate a value, invert a level, or misname a severity. This is the single most valuable piece of code in the repository. **[V]**
- **Applicability filtering.** `applicability.py` refuses to apply a rule whose selectors conflict with the finding's file context and reports "unknown" conservatively instead of guessing. **[V]**
- **A curated, sourced KB with authority labels.** `normative_specification` / `published_documentation` / `validator_implementation` / `dataset_example` / `schema_archive` are already first-class fields, not prose. **[V]**
- **Fast suite is green.** `386 passed, 29 deselected` (415 collected). **[V]**

## A.3 The four systemic failures

### F1 — The evidence *bundle* is wrong even when the top hit is right

For `TOO_FEW_AUTHORS` the correct record ranks **first** with a wide margin. The failure is in positions 2–4 and in the `default_top_k = 3` cutoff.

Verified lexical ranking using the KB's own verbatim validator message: **[V]**

| Rank | Score | Relevance | Record | Title |
| --- | --- | --- | --- | --- |
| 1 | 140.69 | direct | `kb_validationissue_f73c42d966` | TOO_FEW_AUTHORS |
| 2 | 53.99 | background | `kb_validationissue_04efbf575a` | NO_AUTHORS |
| 3 | 30.73 | background | `kb_concept_1d1777946d` | metadata field: **LabelingEfficiency** |
| 4 | 30.61 | background | `kb_concept_2bd321a9af` | metadata field: **Authors** |
| 5 | 28.81 | background | `kb_validationissue_f3224f4fd5` | JSON_KEY_RECOMMENDED |
| 6 | 28.81 | background | `kb_validationissue_f92485b45e` | JSON_KEY_REQUIRED |

Three things follow.

1. With the configured `default_top_k = 3`, the **field definition for `Authors` is excluded** while an unrelated field concept (`LabelingEfficiency`) is included.
2. The gap between rank 3 and rank 4 is **0.12 points**. Whether the answer contains the governing field definition is decided by numerical noise.
3. `LabelingEfficiency` qualifies purely on generic tokens. `retriever.explain()` decomposes its 30.726 exactly as `json` 3.546 + `one` 7.56 + `author` 4.50 + `based` 7.56 + `only` 7.56. The `author` term is a **prefix match against the word `authority` in the record's scope**, and the other four are message boilerplate. There is no BIDS-relevant overlap at all. **[V]**

The scanner scores twelve generic message tokens at **7.56 each = 90.72 points** on the correct record (verified by exact decomposition), while the structured **exact rule-ID bonus is `RULE_ID_EXACT = 20.0`** (`retriever.py:643`). **[V]**

> Rule identity is worth about 18% of what the message's filler words are worth.

The embedding backend does not fix it. Verified with the configured `retriever_type: "embedding"`: **[V]**

| Cosine | Relevance | Record |
| --- | --- | --- |
| 0.921 | direct | TOO_FEW_AUTHORS |
| 0.885 | background | NO_AUTHORS |
| 0.831 | background | Metadata rule dataset_authors |

Still no `Authors` field definition, still no `dataset_description` metadata rule.

### F2 — The contract is derived from whatever was retrieved, so it inherits every retrieval error

`answer_contract.py:317` calls `derive_bids_level(field_name, records)` with **all** retrieved records. **[V]** For `TOO_FEW_AUTHORS` (top_k=3) this returns **`bids_level = "recommended"`**. **[V]**

The record that supplies it is `NO_AUTHORS` — a *different* rule — whose `rule_definition.fields` contains `Authors: {level: "recommended"}`. **[V]**

Meanwhile the governing metadata rule `kb_metadatarule_5dad7e66a6` (`rules.json.dataset.dataset_description`) says:

```
Authors: {level: "optional", level_addendum: "recommended if `CITATION.cff` is not present"}
```

**[V]**

So the answer is not simply "wrong". It is **half-right with the condition deleted**, and it is half-right because a *sibling rule was merged into the current issue* — precisely the failure mode the requirement forbids. The truthful answer needs both clauses:

> *BIDS lists `Authors` as optional; it is recommended when `CITATION.cff` is not present. This finding (`TOO_FEW_AUTHORS`) is a separate validator hint that fires when only one author is listed.*

The current contract can only express one of those two facts, and picks the one that came from the wrong record. **[V][I]**

The KB itself makes this easy to get wrong: the `NO_AUTHORS` record stores `"severity": "recommended"` (`facts.severity`), conflating validator *severity* (error/warning) with BIDS *level* (required/recommended/optional). **[V]**

### F3 — The contract is unstable under configuration, and the gate cannot see it

`example_allowed` is `True` iff *some* retrieved record happens to contain a structural key (`anyOf`/`oneOf`/`enum`/`items`/`format`) for the field (`answer_contract.py:320`). Raising `top_k` from 3 to 6 for `TOO_FEW_AUTHORS` flips it from **False to True**. **[V]**

So a change that looks like "give the model more context" silently changes the answer's *permissions*. The naive fix (increase `top_k`) is therefore not safe. This matches the external finding that adding retrieved context reduces abstention and increases confident error, rather than improving correctness. **[E]**

### F4 — The gate validates prose form, not claims

`output_gate.py` checks execution claims, empty responses, missing no-direct declarations, single-scalar values on the issue field, versions, fenced code blocks, required/recommended inversion, a narrow BIDS-overclaim pattern, a narrow severity pattern, named entities, and duplicate/empty sections. **[V]**

I ran the gate against a contract for `TOO_FEW_AUTHORS` (`severity=warning`, `bids_level=optional`, `example_allowed=False`, `concrete_value_allowed=False`) with four realistic outputs. All four **pass**: **[V]**

| Output | Contains | Gate |
| --- | --- | --- |
| Prose example | `**Example**` + `"Authors": ["Alice Smith", "Bob Jones"]` when examples are forbidden | **PASS** |
| Invented dataset facts | "Your dataset has 12 subjects and one author listed." | **PASS** |
| Foreign-rule contamination | "This is related to `NO_AUTHORS`, which the validator raises..." | **PASS** |
| Severity leak | "The **error** `TOO_FEW_AUTHORS` is raised..." | **PASS** |

It does catch a fenced example when `example_allowed=False`, and it catches an inline `Field: "scalar"` claim. **[V]** Both are narrow: `_FENCE_RE` only sees triple-backtick blocks, and `_field_value_claims` (`output_gate.py:190`) only matches a single quoted scalar, so array values are invisible. **[V]**

The real captured failure in `prompt_log.txt` shows the same class: the model invented `"http://example.com/data"`, `"10.5281/zenodo.12345"` and `"1.0"` inside an example, and wrote *"According to the BIDS specification ... considered recommended but not strictly required"* — while `_BIDS_OVERCLAIM_RE` does not match that phrasing. **[V]**

### F5 — Evaluation cannot detect any of the above

- `eval_quality.py --audit-kb` treats `retrieval.found` as success. It never checks whether the **governing** record is present or ranked first. **[V]**
- It also only records whether `bids_level` is `None`, never whether it is **correct**. **[V]**
- `baseline_scores.json` is n=4 with overall 50.0%, `contract_clean 0/4`, `declares_unknown 0/4`, `no_unsupported_value 1/4`, one fabrication. **[V]**
- `tests/test_llm_regression.py` has six scenarios (A–F) but no `TOO_FEW_AUTHORS` case. **[V]**
- Its `SourceDatasets` test asserts that two *specific* strings from a past log are absent (`"example.org"`, `\b10\.1234/`), not that the policy holds. The captured failure used `"http://example.com/data"` and `"10.5281/zenodo.12345"`; both would pass. **[V]**
- Its severity assertions reuse the same narrow regexes as the gate (`the error (indicates|means|occurs|arises|is)`, `is a validation error`), so "The error `TOO_FEW_AUTHORS` is raised" passes. **[V]**
- `tests/test_prompts.py` pins prompt wording byte-for-byte, so the current instruction set is locked in, including the contradictions. **[V]**

This is why the suite is green while the product fails: **none of the metrics can fail in the ways that matter.**

## A.4 Root cause in one paragraph

The three layers form a closed loop that validates itself instead of the world. Retrieval decides what the evidence is; the contract is computed from that evidence; the gate checks the answer against the contract. Every check is *internally consistent*, so a retrieval error propagates into a confident, self-consistent, gate-passing, and possibly repaired answer. The system has no independent source of truth for "which rule is being explained" or "what does BIDS say about this field". **[I]** Everything in Section F below follows from breaking that loop in one place: resolve the governing record **deterministically by identity**, and let only that record define levels, authority, and permissions. **[R]**

## A.5 If you only do six things

| # | Change | Fixes | Effort | Effect |
| --- | --- | --- | --- | --- |
| 1 | Deterministic governing-record lookup by `issue_code`/`rule_id` (not similarity) | F1 | 1–2 d | Correct rule always present, independent of scoring |
| 2 | Build the contract **only** from the governing record + its conditional field rules | F2 | 2 d | Kills cross-rule level contamination |
| 3 | Make BIDS level a list of `(level, condition)` pairs | F2 | 1 d | Addenda survive; no "optional" vs "recommended" collapse |
| 4 | Structured answer plan + deterministic renderer + claim validator | F3, F4 | 5–7 d | Gate becomes exact; repair loop disappears |
| 5 | Replace prose regexes with whole-answer **value provenance** checks | F4 | 2 d | Invented URLs/DOIs/names/counts caught mechanically |
| 6 | Metrics that can fail: governing@1, level-condition integrity, severity equality, example policy | F5 | 3 d | Regressions become visible before release |

Items 1–3 alone remove the most dangerous error class (confidently stating a wrong BIDS level) and cost roughly one week. **[R]**

## A.6 What I explicitly do **not** recommend

- **No autonomous agent, no planner loop, no tool-calling.** There is exactly one workflow. A graph framework would add state machinery to a straight line. **[R]**
- **No LangGraph / no new framework.** LangChain is currently used for orchestration sugar only; the durable value is in `retriever.py`, `applicability.py`, `answer_contract.py`. Coupling the safety model to a framework's release cadence is a liability, not a feature. **[R]**
- **No second LLM as judge or validator.** It reintroduces non-determinism exactly where determinism is the point. Use LLMs offline for building the eval set, not in the request path. **[R]**
- **No vector database.** 3,505 × 384 floats is ~5 MB; brute-force cosine is microseconds. **[R]**
- **No larger `top_k` as a fix.** Verified to change contract fields. **[V]**
- **No fine-tuning.** The failures are retrieval selection and contract derivation, not language modelling. **[R]**
- **No "prompt engineering only" fix.** The gate cannot see what a better prompt still gets wrong. **[I]**

## A.7 Bottom line

The project has the right architecture for the problem and unusually good instincts about honesty (abstention, authority labels, template fallback). What it lacks is a **single source of truth for the finding being explained**. Adding one deterministic lookup and deriving everything from it is a ~2-week change that eliminates the failure class that matters most. The remaining work is turning prose validation into structured validation so the safety claims become testable.

I recommend approving phases 0–3 of Section F before any prompt or model tuning.

---

# B. Current Architecture (as built)

## B.1 Request path

```
HTTP POST /predict                       main.py
   │  lazy singleton Agent (thread-safe init)
   ▼
Agent.__init__                           agent.py
   ├─ LLMClient            → LocalChatModel (transformers, Qwen3-1.7B default)
   ├─ build_retriever()    → Retriever  or  EmbeddingRetriever
   ├─ KBRetriever(core)    → LangChain BaseRetriever facade
   ├─ ToolRegistry()       → empty, intentionally non-executable
   └─ Planner(...)
          │
          ▼
Planner._build_pipeline()                planner.py:327
   RunnableBranch(intent lane)
     └─ RunnableLambda(_prepare)         retrieval + contract + prompt render
          └─ RunnableParallel(           generate ‖ keep state
               ├─ llm_runnable
               └─ RunnablePassthrough)
                └─ RunnableLambda(_build_result)
                     └─ _apply_quality_gate()
                          ├─ output_gate.check()
                          ├─ deterministic_repair()
                          ├─ one LLM repair call
                          └─ render_grounded_template()
   ▼
PlannerResult(text, diagnostics)
```

LLM calls per request: **1** if the regex intent guard fires, **2** in auto mode (intent classifier + generation), **3** if the repair pass runs. **[V]**

## B.2 Knowledge base

`knowledge-base/knowledge_base.jsonl`, 13.8 MB. Verified counts: **[V]**

| Dimension | Distribution |
| --- | --- |
| Total records | 3,505 |
| `knowledge_type` | Concept 2606, ValidationIssue 232, MetadataRule 216, FilenameRule 184, Example 129, Version 75, TabularRule 28, Relationship 13, ValidationRule 11, Troubleshooting 11 |
| `scope.authority` | normative_specification 1775, published_documentation 1466, dataset_example 110, validator_implementation 108, schema_archive 46 |

Each record carries `id`, `knowledge_type`, `title`, `summary`, `retrieval_text`, `scope`, `bids_version`, `schema_version`, `source` (repo/file/line/revision), `raw_content`, `facts`, `authority`, `knowledge_family`, `relationships`. **[V]**

This is a strong asset. The `facts` object is machine-readable and is currently underused (see D).

## B.3 Retrieval

`Retriever` (2,060 lines) loads records lazily, builds an inverted index, expands candidates, scores with a field-weighted lexical scorer plus structured bonuses, applies `applicability`, labels each record `direct` / `background` / rejected, dedupes by rule identity, and formats the top `default_top_k` (3) as prompt sections. **[V]**

Structured bonuses include `RULE_ID_EXACT = 20.0`, rule-ID-in-title, field-in-title/id/summary, field-phrase, severity match, BIDS identity, and phrase bonus. **[V]**

`EmbeddingRetriever` (706 lines) encodes the same records with `BAAI/bge-small-en-v1.5`, caches vectors to `knowledge-base/vector_store/`, unions semantic candidates with lexical candidates, and reuses the lexical loader/applicability/formatter. Only ranking changes. **[V]**

`default_top_k = 3` is justified in `config.py:117–123` as "room for a direct record plus background context". **[V]** The premise — that the background records are relevant — is what fails in F1.

## B.4 Contract

`build_contract(issue, retrieval, file_content)` derives: `rule_id`, `field`, `path`, `message`, `severity`, `bids_level`, `authority`, `direct_evidence`, `placeholder`, `example_allowed`, `concrete_value_allowed`, `requires_no_direct_statement`, `must_state_unknown`. **[V]**

It is rendered into the prompt as a checklist placed **last**, after all evidence. **[V]** That placement is good practice; the derivation is the problem.

## B.5 Prompt

`BIDS_SYSTEM_PROMPT` (337-line module) defines: capability boundary, evidence priority (`Issue / Application Context` → file digest → dataset tree → retrieved knowledge), the direct/background/not-applicable semantics, placeholder rules, type-vs-format rules, example rules, and an output section list. **[V]**

## B.6 Gate and repair

`output_gate.check()` returns `GateResult(violations)`. `planner._apply_quality_gate` then tries, in order: deterministic repair → up to `repair_attempts=1` LLM repair → `render_grounded_template`. **[V]**

The template tier is genuinely safe. The tiers above it are not, because the gate they target is form-based.

## B.7 Configuration drift

| Setting | `config.py` default | `config.json` (present, git-ignored) | README states |
| --- | --- | --- | --- |
| `retrieval.retriever_type` | `lexical` | **`embedding`** | `lexical` (default) |
| `intent.classifier` | `llm` | **`laya`** | `llm` (default) |
| `llm.save_prompts` | `true` | `true` | documented as opt-out |

**[V]**

So the shipped runtime differs from both the documented defaults and the tested defaults. The README also points at `knowledge-base/KB_README.md`, which does not exist (only `COVERAGE_GAPS.md`). **[V]**

## B.8 Tests and CI

415 tests collected; 386 run fast; 29 are slow (real model). **[V]**

`quality_set.jsonl` holds 36 frozen findings: 20 `validator_impl`, 10 `validator_spec`, 6 `manager_todo`. **[V]**

CI (`.github/workflows/quality.yml`) runs the fast suite, `eval_quality.py --audit-kb`, and the baseline; a separate GPU job runs `pytest -m slow` and `eval_quality.py --live --min-score 0.9`. **[V]**

## B.9 Claims vs behaviour

| Claim | Reality |
| --- | --- |
| "Agent" architecture | Linear pipeline; no tools, no loop **[V]** |
| Evidence-grounded | Grounded in *retrieved* records, not in the *governing* rule **[V]** |
| Contract prevents inversion | Contract can itself carry the wrong level **[V]** |
| Gate enforces the contract | Gate enforces prose patterns, not claims **[V]** |
| Example policy | Prompt always offers an `**Example**` section; contract may forbid it **[V]** |
| KB coverage audited | Audit checks "something retrieved", not "right thing retrieved" **[V]** |

---

# C. Problems Found

Severity: **S1** = produces confidently wrong answers or blocks the product goal; **S2** = latent failure or maintenance hazard; **S3** = hygiene.

## C1 — Rule identity is treated as a ranking problem (S1)

**Evidence [V]:** exact-ID bonus 20.0 versus 90.72 of message-boilerplate contribution; unrelated `LabelingEfficiency` at rank 3 above `Authors` at rank 4 by 0.12; `default_top_k=3` excludes the field definition; embedding backend reproduces the same omission.

**Impact:** the model must explain a rule without the field definition that rule refers to, and is given a sibling rule instead.

**Class:** design defect in the retrieval contract.

## C2 — Contract inheritance from unrelated rules (S1)

**Evidence [V]:** `derive_bids_level` scans all records (`answer_contract.py:133–149`, called with `records` at `:317`); `NO_AUTHORS` supplies `Authors: recommended`; governing metadata rule says `optional` + addendum.

**Impact:** answers state a BIDS level that is false as stated, in the direction of *over-claiming obligation*. This is the exact failure the project set out to prevent.

**Class:** correctness bug + violated requirement ("must not merge related/different rules").

## C3 — Conditional levels are collapsed to scalars (S1)

**Evidence [V]:** `AnswerContract.bids_level: Optional[str]`; the addendum "recommended if `CITATION.cff` is not present" has no representation.

**Impact:** even a *correct* lookup ("optional") would lose the practically important part of the rule. Users would leave `Authors` out when they have no `CITATION.cff` — a real-world outcome the KB explicitly warns about.

**Class:** data-model defect.

## C4 — Contract permissions are retrieval-dependent (S1)

**Evidence [V]:** `example_allowed` False at `top_k=3`, True at `top_k=6`, for the same finding. `concrete_value_allowed` has the same dependency via `_has_enumerated_values`.

**Impact:** the same input produces different answer *rules* depending on tuning. Any A/B test on retrieval confounds answer policy.

**Class:** design defect.

## C5 — Gate checks form, not claims (S1)

**Evidence [V]:** four realistic violations pass (prose example, invented dataset facts, foreign rule, severity word). `_field_value_claims` ignores arrays. `_FENCE_RE` ignores inline examples. `_BIDS_OVERCLAIM_RE` misses "According to the BIDS specification...". `_WRONG_SEVERITY_RE` misses "The error X is raised".

**Impact:** the safety story is unenforced. Worse, a passing gate is treated as evidence of quality in diagnostics and eval.

**Class:** design defect.

## C6 — Prompt contradicts the contract (S1)

**Evidence [V]:** `prompts.py:82` always instructs the model to produce `**Example**`; `answer_contract.render_contract` emits `example: not allowed — omit it` when examples are forbidden; both appear in the same prompt.

**Impact:** instruction conflict resolved in favour of the general instruction — exactly what is observed in `prompt_log.txt`. External work on instruction hierarchy shows general instructions frequently override later, narrower ones. **[E]**

**Class:** bug (spec conflict).

## C7 — Repair loop targets an unreliable oracle (S2)

**Evidence [V]:** `_apply_quality_gate` loops on `output_gate.check`; deterministic repair can *add* the no-direct sentence and append unknowns, which can then pass.

**Impact:** cost (extra model call), latency, and a false sense of convergence. Repair inherits the gate's blind spots.

**Class:** architecture.

## C8 — Evaluation cannot fail meaningfully (S1)

**Evidence [V]:** `run_audit` uses `retrieval.found`; `bids_level` checked only for `None`; baseline n=4 at 50%; no `TOO_FEW_AUTHORS` slow case; byte-pinned prompt tests.

**Impact:** green CI is uncorrelated with the product requirement.

**Class:** process defect — arguably the most expensive one.

## C9 — Byte-pinned prompt tests lock in defects (S2)

**Evidence [V]:** `tests/test_prompts.py`, `tests/test_regression_scenarios.py` assert exact strings.

**Impact:** every prompt fix requires rewriting goldens, which makes fixing C6 feel expensive and thus unlikely. Tests encode the bug.

**Class:** test-design defect.

## C10 — Unnecessary second model call for intent (S2)

**Evidence [V]:** default `classifier: "llm"`; `rule_intent` already resolves unmistakable questions/actions deterministically (`intent.py:136–162`).

**Impact:** ~2× latency and an extra failure mode for a binary decision that regexes already handle for the common cases. **[I]**

**Class:** over-engineering.

## C11 — Legacy surface area (S3)

**Evidence [V]:** `intent_classifier.py` (56 lines), `laya_decision.py` (346), `eval_intent.py` (127), `tool_registry.py` (74), `test_laya_decision.py`; `LLMClient` constructs an `IntentClassifier` that is unused in the default path; `kb_retriever.py` is a facade needed only because the planner uses LCEL.

**Impact:** review burden, test time, and the impression of capability that does not exist.

**Class:** debt.

## C12 — Prompt logging writes user data by default (S2)

**Evidence [V]:** `save_prompts: bool = True` (`config.py:97`), and `prompt_log.py` says it writes the full system prompt, user question, and answer to `prompt_log.txt` next to `main.py`, overwriting each request. The module docstring says it is temporary and should be removed after testing (`prompt_log.py:26–27`).

**Impact:** dataset content and user questions land on disk in a deployed service. `prompt_log.txt` is present in the working tree (untracked, not committed). **[V]**

**Class:** privacy/hygiene.

## C13 — KB conflates severity with level (S2)

**Evidence [V]:** `NO_AUTHORS` has `facts.severity = "recommended"`. `TOO_FEW_AUTHORS` has `facts.severity = "warning"`. Both are in the same field.

**Impact:** any consumer that reads `severity` may emit "recommended" where an error/warning word is required, or treat a level as a severity. This is upstream of C2 and will keep re-seeding it.

**Class:** data-quality defect.

## C14 — Documentation drift (S3)

**Evidence [V]:** README defaults (lexical/llm) ≠ `config.json` (embedding/laya); README references a missing `KB_README.md`.

**Impact:** results are not reproducible from the docs; reviewers test a different configuration than production.

**Class:** documentation defect.

## C15 — The one safe tier is the least-used tier (S2)

**Evidence [V]:** `render_grounded_template` only runs after deterministic repair, an LLM repair, and a failed gate.

**Impact:** the only construction that is correct by design is a last resort. Inverting this is free quality. **[R]**

**Class:** architecture.

## Problem summary

| ID | Problem | Sev | Layer | Effort to fix |
| --- | --- | --- | --- | --- |
| C1 | Identity as ranking | S1 | Retrieval | 1–2 d |
| C2 | Cross-rule contract inheritance | S1 | Contract | 2 d |
| C3 | Level conditions collapsed | S1 | Contract/KB | 1 d |
| C4 | Retrieval-dependent permissions | S1 | Contract | 1 d |
| C5 | Form-based gate | S1 | Gate | 2–5 d |
| C6 | Prompt/contract conflict | S1 | Prompt | 0.5 d |
| C7 | Repair loop on weak oracle | S2 | Planner | 1 d |
| C8 | Weak evaluation | S1 | Eval | 3 d |
| C9 | Byte-pinned tests | S2 | Tests | 1 d |
| C10 | Intent LLM call | S2 | Intent | 0.5 d |
| C11 | Legacy modules | S3 | Repo | 1 d |
| C12 | Prompt logging on by default | S2 | Config | 0.25 d |
| C13 | Severity/level conflation in KB | S2 | KB | 1 d |
| C14 | Doc drift | S3 | Docs | 0.5 d |
| C15 | Safe tier is last resort | S2 | Planner | 1 d |

---

# D. Component-by-Component Analysis

Verdicts: **KEEP** (works, leave it), **REFACTOR** (keep the idea, change the contract), **DEVELOP** (extend materially), **REMOVE** (delete), **INVESTIGATE** (unclear value; decide after measurement).

| File | Lines | Role | Verdict | Rationale / required action |
| --- | --- | --- | --- | --- |
| `main.py` | — | Flask API `/predict`, `/config`, `/models`, `/health` | **KEEP** | Small, correct. Add request-size limits and structured error codes. |
| `agent.py` | — | Wiring | **REFACTOR** | Drop `ToolRegistry` and the unused classifier; expose the retriever/bundle for tests. |
| `planner.py` | 644 | Orchestration + gate loop | **REFACTOR** | Remove LCEL; make the flow a readable function. Replace the repair loop with validate-then-render (Section E). |
| `prompts.py` | 337 | System prompt + sections | **REFACTOR** | Remove the `**Example**` section from the unconditional output list; move formatting rules into the renderer. |
| `retriever.py` | 2060 | KB load, index, scoring, applicability hook, formatting | **REFACTOR** | Keep the scorer, but demote it: add `resolve_governing(finding)` (exact-key lookup) and return a typed `EvidenceBundle`. Scoring becomes supporting-context-only. |
| `embedding_retriever.py` | 706 | Semantic ranking + cache | **KEEP** | Useful as recall fallback. Add contract-invariance tests across backends. |
| `retrieval_factory.py` | 42 | Backend selection | **KEEP** | Clear and small. |
| `applicability.py` | 374 | Selector evaluation | **KEEP** | Conservative and correct. Extend to field-level rules only if needed. |
| `answer_contract.py` | 413 | Contract derivation | **REFACTOR** | Governing-record-only inputs; `levels: list[(level, condition)]`; per-field provenance; remove retrieval-order dependence. |
| `output_gate.py` | 400 | Regex validation | **DEVELOP** | Keep ~6 checks as a last-mile prose lint. Add whole-answer value provenance. Primary validation moves to the plan validator (Section E). |
| `answer_repair.py` | 244 | Deterministic + LLM repair | **REFACTOR** | Keep `render_grounded_template` (promote it). Retire LLM repair once the plan is validated before rendering. |
| `intent.py` | 338 | Regex guard + LLM/NLI classifiers | **REFACTOR** | Keep `rule_intent` and forced modes. Make the LLM classifier opt-in, not default. |
| `intent_classifier.py` | 56 | Legacy NLI | **REMOVE** | Superseded; only reachable via a non-default setting. |
| `laya_decision.py` | 346 | Legacy 3-way classifier | **REMOVE** | Dead in the default path; large surface, no product use. |
| `llm_client.py` | 339 | Model invocation | **REFACTOR** | Remove the unused `IntentClassifier`; make prompt logging explicit and off by default. |
| `local_chat_model.py` | — | transformers wrapper | **KEEP** | Works; quantization is handled. |
| `kb_retriever.py` | 181 | LangChain `BaseRetriever` facade | **REMOVE** | Only needed by the LCEL pipeline; a plain method call is simpler and testable. |
| `tool_registry.py` | 74 | Empty tool registry for a "future phase" | **REMOVE** | Explicitly speculative. Version control preserves it if a tool phase ever happens. |
| `prompt_log.py` | 112 | Writes last prompt/answer to disk | **REMOVE** | Self-declared temporary (`:26–27`); privacy cost. Replace with an explicit debug flag if still needed. |
| `observability.py` | 130 | Step timing | **REFACTOR** | Concept is good; implementation is LCEL-callback-bound. Reimplement as `@timed_step`. |
| `model_manager.py` | 65 | Model listing/selection | **KEEP** | Orthogonal to correctness. |
| `config.py` | 360 | Dataclass config | **KEEP** | Add validation + an `/config` echo of *effective* values so drift is visible. |
| `utils.py` | 46 | Formatting helpers | **KEEP** | `json_to_llm_text` may be dead — verify before deleting. |
| `eval_quality.py` | — | Baseline + audit + live | **DEVELOP** | Replace `found` with governing@1, add per-item expectation checks. |
| `eval_intent.py` | 127 | Intent eval | **REMOVE** if intent LLM is demoted; else **KEEP**. |
| `benchmark_timing.py` | 163 | Latency benchmark | **KEEP** | Useful for the latency budget once the intent call is removed. |
| `tests/test_prompts.py` | — | Byte-pinned prompt goldens | **REFACTOR** | Assert *properties* (contains no `**Example**` when forbidden) instead of exact strings. |
| `tests/test_regression_scenarios.py` | — | Byte-pinned scenarios | **REFACTOR** | Same. |
| `tests/test_output_gate.py` | — | Gate unit tests | **DEVELOP** | Add the four verified passes as failing tests first. |
| `tests/test_llm_regression.py` | — | 6 slow scenarios (A–F) | **DEVELOP** | Add `TOO_FEW_AUTHORS` and `NO_AUTHORS`; replace the hardcoded `example.org`/`10.1234` assertions with a policy check over the whole answer. |
| `knowledge-base/knowledge_base.jsonl` | — | Data | **KEEP** | High quality. Fix `facts.severity` for level-only rules (C13). |
| `knowledge-base/COVERAGE_GAPS.md` | — | Gap register | **DEVELOP** | Keep, but base it on governing-record coverage. |
| `README.md` | — | Docs | **REFACTOR** | Describe the actual defaults; remove the `KB_README.md` reference or add the file. |
| `prompt_log.txt` | — | Debug artifact in the tree (untracked) | **REMOVE** | Contains a real failing answer; keep it as a fixture under `tests/data/` instead. |

**Net effect of the verdicts:** roughly 900 lines removed (56 + 346 + 127 + 74 + 181 + 112, plus the LCEL scaffolding), several hundred changed, no capability lost.

Three verdicts deserve defence because they cut against "keep everything that works":

- **`tool_registry.py` — REMOVE.** Keeping a placeholder for a hypothetical future is exactly the "unnecessary abstraction" the project should avoid. Its own docstring says it is for a future phase.
- **`kb_retriever.py` — REMOVE.** It adds a class, a document-splitting helper, and a metadata mapping to expose what `Retriever` already returns, and it forces the planner into LCEL. The Planner calls `retrieve_issue_structured` anyway.
- **`answer_repair.py` (LLM repair) — REFACTOR, not KEEP.** A repair loop is a reasonable pattern, but it is only as good as its oracle. Once the answer is generated as a validated plan, there is nothing for the loop to repair. Keeping both mechanisms means two code paths that must agree.

---

# E. Recommended Target Architecture

## E.1 Principles

1. **Identity is not similarity.** A finding names a rule. Resolve that rule by key, not by score.
2. **One source of truth per fact.** Severity comes from the finding. Level comes from the governing field rule. Authority comes from the record's `scope.authority`. Never from a sibling rule.
3. **Deterministic before generative.** Compute everything that can be computed; generate only the explanation.
4. **Validate claims, not prose.** Check values, IDs, levels, and permissions structurally.
5. **Degrade to the template, not to prose.** If validation fails, render the safe answer.
6. **Abstain loudly.** Missing governing evidence produces the no-direct statement, not a best guess.

## E.2 Target pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. PARSE (deterministic)                                        │
│    Finding{rule_id, severity, field, path, message, selectors,  │
│            checks, provenance: validator|manager}               │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. RESOLVE (deterministic, no scoring)                          │
│    governing     := record with facts.issue_code == rule_id     │
│    governing_rule:= metadata rule whose selector matches path   │
│                     and whose fields contain `field`            │
│    levels        := [(level, condition_or_null)] from above     │
│    authority     := scope.authority of governing                │
│    permissions   := derived from governing_rule only           │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. SUPPORT (scored, restricted, non-authoritative)              │
│    siblings := records sharing (field, path) — labelled, never  │
│                used for levels or authority                     │
│    background := BM25 top-2 sharing (field AND path)            │
│    excluded   := everything else (never rendered)               │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. PLAN (LLM → JSON, schema-constrained)                        │
│    {severity, rule_id, levels[], direct_facts[], background[],  │
│     related_rules[], unknowns[], fix_steps[], example|null}     │
│    every fact carries evidence_id                               │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. VALIDATE (deterministic, whole-plan)                         │
│    severity == finding.severity                                 │
│    rule_id  == finding.rule_id                                  │
│    levels   == resolved levels (conditions preserved)           │
│    every value/number/URL/DOI/name ∈ evidence ∪ inputs          │
│    every evidence_id resolves to a real record                  │
│    example == null iff not permitted; else shape-only leaves    │
│    related_rules ⊆ computed siblings, each with a relation      │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. RENDER (deterministic template)                              │
│    Markdown: Problem / Why / How to Fix / Unknown               │
│    Example section emitted only if plan.example != null         │
└─────────────────────────────────────────────────────────────────┘
```

## E.3 Detailed stage contracts

### Stage 2 — `resolve_governing`

```python
def resolve(finding) -> EvidenceBundle:
    # 1. exact rule identity — no scoring
    governing = index.by_issue_code[finding.rule_id]        # most findings
    # 2. field-level governing rule (selectors evaluated, not scored)
    governing_rule = index.field_rule(path=finding.path, field=finding.field)
    # 3. conditional level rules that apply to this path+field
    conditions = index.level_conditions(path=finding.path, field=finding.field)
    return EvidenceBundle(governing, governing_rule, conditions, ...)
```

Rules:

- If `governing` is missing → `requires_no_direct_statement = True`. No fallback to similarity. **[R]**
- If multiple records share `issue_code` (e.g. version variants), pick by `bids_version` match, then highest `authority`, else mark ambiguous and state the version. **[R]**
- `levels` is built **only** from `governing.rule_definition.fields` and `governing_rule.fields`, preserving `level_addendum` verbatim as the condition string. **[R]**

This alone fixes C1, C2, C3, C4.

### Stage 3 — siblings vs background

"Sibling" is computed, not ranked:

```
siblings(finding) = { r : r.field == finding.field
                        and r.path == finding.path
                        and r.id != governing.id }
```

For `TOO_FEW_AUTHORS` this yields exactly `NO_AUTHORS` — and it will be rendered under a heading that says *"another check on the same field (not the cause of this finding)"*. That satisfies "distinguish related rules" without importing their levels. **[R]**

"Background" requires **both** field and path to match. A `LabelingEfficiency` concept matches neither and is excluded by construction, not by threshold. **[R]**

### Stage 4 — plan schema

```json
{
  "severity": "warning",
  "rule_id": "TOO_FEW_AUTHORS",
  "levels": [
    {"field": "Authors", "level": "optional",
     "condition": "recommended if `CITATION.cff` is not present"}
  ],
  "direct_facts": [
    {"text": "fires when only one author is listed",
     "evidence_id": "kb_validationissue_f73c42d966"}
  ],
  "background": [],
  "related_rules": [
    {"rule_id": "NO_AUTHORS", "relation": "same field, different check"}
  ],
  "unknowns": ["the intended contributor list for this dataset"],
  "fix_steps": ["add the remaining contributors to `Authors`"],
  "example": null
}
```

Why this shape:

- It makes every gate check a comparison, not a regex. **[R]**
- It makes "examples not allowed" structural: `example` is `null`, so the renderer cannot emit the section. **[R]**
- It makes "don't merge rules" structural: a foreign `rule_id` cannot appear inside `direct_facts` because `direct_facts` are bound to `evidence_id`; foreign rules can only appear in `related_rules` with a relation label. **[R]**

### Stage 5 — validator rules (exhaustive)

| Rule | Check | Failure |
| --- | --- | --- |
| V1 severity | `plan.severity == finding.severity` | reset to finding's value |
| V2 rule identity | `plan.rule_id == finding.rule_id` | reject |
| V3 levels | `plan.levels` equals resolved set, conditions verbatim | reject |
| V4 authority | no claim stronger than `bundle.authority` | strip claim |
| V5 evidence binding | every `evidence_id` ∈ bundle | reject |
| V6 support | direct fact tokens ⊆ cited record text (or a cited span) | reject |
| V7 value provenance | every numeral, URL, DOI, email, version, capitalised multiword name ∈ evidence ∪ inputs | reject |
| V8 example policy | `example is None` unless `permissions.example_allowed`; if present, all leaves match `<...>` | null the example |
| V9 foreign-rule policy | any rule ID outside `{finding.rule_id} ∪ siblings` must be absent or in `related_rules` | move or strip |
| V10 unknown declaration | `unknowns` non-empty when `bundle` lacks a governing record or a level | reject |
| V11 no dataset facts | no claim about counts, existence, or contents not present in inputs | reject |

V7 is the replacement for the current single-field value check and catches the `example.com` / `zenodo` failure. V11 catches "Your dataset has 12 subjects".

### Stage 6 — renderer

Deterministic, ~80 lines. Sections emitted conditionally:

```
**Problem**   ← finding.message + severity
**Why**       ← direct_facts (each with an inline source label)
              ← levels (level + condition, both clauses)
              ← related_rules (explicitly labelled "not the cause")
**How to Fix**← fix_steps
**Example**   ← only if plan.example is not null, rendered shape-only
**Unknown**   ← unknowns
```

No prompt can contradict this because the prompt no longer decides the structure. **[R]**

## E.4 Why not a framework / not a second LLM / not a vector DB

- **Frameworks.** The workflow has four deterministic steps and one model call. `RunnableBranch`/`RunnableParallel` currently encode "do A then B then C" with ~30 lines of indirection. Removing it improves traceback quality and testability. **[R]**
- **Second LLM as validator.** The validator's job is comparison against a resolved bundle. A model cannot do `plan.severity == finding.severity` more reliably than `==`. It also makes the gate non-reproducible, which destroys regression testing. **[R]**
- **Vector DB.** 3,505 records, brute-force cosine. A DB adds an operational dependency for a matrix multiply. **[R]**
- **But keep the embedding backend.** It is a good recall fallback for *supporting* records when lexical misses, and it is already written. The change is that it never decides the governing record. **[R]**

## E.5 What changes, what stays

| Concern | Now | Target |
| --- | --- | --- |
| Governing rule | score-ranked | key lookup |
| Field level | scalar from any record | list of (level, condition) from governing only |
| Permissions | derived from all records | derived from governing_rule only |
| Evidence shown | top-3 by score | governing + siblings + ≤2 related background |
| Answer form | free Markdown | JSON plan → template |
| Validation | regex over prose | comparisons over plan |
| Repair | deterministic + 1 LLM + template | validate, else template |
| LLM calls | 1–3 | 1 |
| Abstention | when no direct record | same, plus when level is ambiguous |
| Framework | LCEL | plain Python |

## E.6 Deliberate non-goals

- Multi-turn reasoning across findings.
- Automatic dataset repair or validator execution.
- Answering questions not tied to a finding (out of scope for v1; the no-direct path covers it honestly).
- Multilingual output.
- Streaming.

---

# F. Development Plan

Each phase is independently shippable and independently revertible. Effort is for one engineer familiar with the code; ranges reflect uncertainty about test churn.

## Phase 0 — Freeze the failure cases (0.5–1 day)

**Goal:** make the current failures executable so no fix can silently regress them.

**Work**
- Add the four gate-passing outputs from C5 to `tests/test_output_gate.py` as `xfail` (they currently pass; assert they *should* fail).
- Add `tests/data/failing_cases/` with: `TOO_FEW_AUTHORS` message, expected governing record ID, expected levels, expected forbidden strings.
- Move `prompt_log.txt` into `tests/data/fixtures/` as a captured real answer.
- Add `tests/test_contract_invariance.py`: for each of the 36 quality items, assert the contract's `(severity, rule_id, bids_level, example_allowed)` is identical across `top_k ∈ {1,3,6,10}`.

**Acceptance:** new tests fail for the documented reasons; all existing tests still pass.

**Risk:** none. This phase adds tests only.

## Phase 1 — Identity resolution (1–2 days)

**Goal:** kill F1.

**Work**
- Add `Retriever.by_issue_code` / `by_rule_id` indexes (built once in `_build_index`).
- Add `resolve_governing(finding) -> Optional[KnowledgeRecord]`.
- Add `EvidenceBundle` dataclass: `governing`, `governing_rule`, `level_conditions`, `siblings`, `background`, `excluded`, `found`.
- Restrict `background` to records matching **both** field and path.
- Keep `_score_candidates` unchanged for `background` selection.
- Delete the `author`-prefix-match class of false positive by excluding scope-only prefix matches from *background* eligibility (or by requiring ≥1 field/path token match).

**Acceptance:** `governing@1 == 1.0` for all 36 quality items; `LabelingEfficiency` never appears in a bundle for `TOO_FEW_AUTHORS`; `Authors` field definition always present for that finding.

**Risk:** some quality items may have no exact `issue_code` (e.g. `bidsmgr.*`, or findings whose `rule_id` is `null`). Mitigation: fall back to `rule_group`+`check_kind` matching, then to the existing no-direct path. Keep a metric for fallback usage.

## Phase 2 — Contract v2 (2–3 days)

**Goal:** kill F2, F3, C3.

**Work**
- Change `build_contract` to accept an `EvidenceBundle`, not a `RetrievalResult`.
- Replace `derive_bids_level` with `derive_levels(finding, bundle) -> tuple[LevelSpec, ...]` reading **only** `governing.rule_definition.fields` and `governing_rule.fields`.
- Preserve `level_addendum` verbatim.
- Derive `example_allowed` / `concrete_value_allowed` from `governing_rule` only.
- Add `provenance: dict[str, str]` mapping each contract field to its record ID.
- Add `ambiguous: bool` when two records claim different levels for the same field; the answer then states both with their conditions.

**Acceptance:** `TOO_FEW_AUTHORS` → `levels = [("optional", "recommended if `CITATION.cff` is not present")]`; `example_allowed` invariant across `top_k` and across backends; contract provenance points only at `f73c42d966` and `5dad7e66a6`.

**Risk:** the KB's `facts.severity` conflation (C13) may make `governing.rule_definition.fields` inconsistent across rule families. Mitigation: normalise on read (`severity ∈ {error, warning}`; otherwise treat as level) and fix the six affected records.

## Phase 3 — Structured plan + validator + renderer (5–7 days)

**Goal:** kill F4, C5, C7, C15.

**Work**
- Define the plan schema (Section E.3) and a strict parser (reject unknown keys, missing keys, wrong types).
- Change the generation call to request the plan with a schema-constrained prompt; use the local model's JSON mode or a constrained prefix. If the model cannot reliably emit JSON, first run it with a 3-example few-shot and a `json` grammar; measure before adopting.
- Implement V1–V11 as pure functions with unit tests per rule.
- Implement the renderer.
- Wire the planner: `prepare → plan → validate → render`. Delete the LCEL pipeline and the repair loop; keep `render_grounded_template` as the fallback when validation fails.
- Keep `output_gate` checks as a final lint over the rendered Markdown (belt and braces), not as the primary gate.

**Acceptance:**
- The four C5 outputs are rejected.
- The captured `prompt_log.txt` answer is rejected (invented URL, DOI, version, overclaim).
- `example_allowed=False` ⇒ rendered text contains no `Example` heading (verified by assertion, not regex luck).
- `TOO_FEW_AUTHORS` renders both level clauses.
- One model call per request in the default configuration.

**Risk:** the local 1.7B model may be unreliable at JSON. Mitigation: (a) keep the deterministic template as the fallback so quality never drops below it; (b) if JSON reliability is <95%, use a smaller intermediate: model emits *facts only* as bullet lines with `evidence_id` prefixes, and the renderer builds structure. Measure both on the 36-item set before choosing.

## Phase 4 — Prompt simplification (1–2 days)

**Goal:** kill C6 and reduce prompt size.

**Work**
- Remove `**Example**` from the unconditional output list.
- Remove from the system prompt every rule now enforced by the validator (severity wording, level wording, value policy, section structure). Keep only: capability boundary, evidence semantics, tone, and abstention.
- Delete the redundant evidence-priority prose that duplicates the checklist.
- Rewrite `tests/test_prompts.py` to assert properties.

**Acceptance:** prompt token count down ≥30%; the six slow scenarios still pass; no instruction in the prompt contradicts the validator.

**Risk:** less prompt scaffolding can reduce prose quality. Mitigation: prose quality is not the product goal; correctness is. Track readability informally, not as a gate.

## Phase 5 — Evaluation hardening (3–4 days, can run parallel to 3)

**Goal:** kill C8, C9.

**Work**
- Implement metrics in G.2.
- Replace `--audit-kb`'s `found` check with `governing@1` and `level-integrity`.
- Expand `quality_set.jsonl` from 36 to ≥120 by sampling the KB by rule family, with expected `(severity, levels, allowed_rules)` per item. This is annotation work, not generation work.
- Add the adversarial suite in G.3.
- Add a `--strict` mode used by CI.

**Acceptance:** the suite fails on `main` before Phase 1–4 and passes after; Wilson intervals reported; no metric can be satisfied by "something was retrieved".

**Risk:** annotation cost. Mitigation: derive expectations mechanically from `facts` for the 232 ValidationIssue records; annotate manually only the 30% that need judgement.

## Phase 6 — Cleanup and documentation (1–2 days)

**Goal:** remove the surface area that makes review expensive.

**Work**
- Delete `intent_classifier.py`, `laya_decision.py`, `eval_intent.py`, `tool_registry.py`, `kb_retriever.py`, `prompt_log.py`, `prompt_log.txt`.
- Remove the `IntentClassifier` construction in `llm_client.py`.
- Make `save_prompts` default `False`.
- Fix `README.md` defaults; add or remove the `KB_README.md` reference.
- Add `GET /config` echoing **effective** values (post-merge), and a startup log line with the effective retriever/classifier.

**Acceptance:** `pytest` green; `git grep` finds no references to deleted modules; running `python main.py` logs the effective configuration.

## Dependency graph and totals

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4
                        └────────► Phase 5 ──► Phase 6
```

| Phase | Effort | Cumulative |
| --- | --- | --- |
| 0 | 0.5–1 d | 1 d |
| 1 | 1–2 d | 3 d |
| 2 | 2–3 d | 6 d |
| 3 | 5–7 d | 13 d |
| 4 | 1–2 d | 15 d |
| 5 | 3–4 d | 19 d |
| 6 | 1–2 d | 21 d |

**~4–5 weeks for one engineer**, including evaluation and cleanup. Phases 0–2 (~1 week) deliver the highest-value correctness win and can ship alone.

---

# G. Testing Strategy

## G.1 Shape of the suite

```
        ┌──────────────────────────────┐
        │  Adversarial (deterministic) │  100% required; every C5 case
        ├──────────────────────────────┤
        │  Property / metamorphic      │  invariance across k, backend, phrasing
        ├──────────────────────────────┤
        │  Contract + validator units  │  V1–V11, one test each
        ├──────────────────────────────┤
        │  Retrieval units             │  governing@1, sibling set, exclusions
        ├──────────────────────────────┤
        │  Golden quality set (≥120)   │  per-item expectations
        ├──────────────────────────────┤
        │  Slow real-model (≥10)       │  end-to-end, sampled
        └──────────────────────────────┘
```

Fast tiers must stay under ~3 minutes. The slow tier is advisory until it is stable across three consecutive runs, then it gates merges.

## G.2 Metrics that can fail

All per-item, binary, machine-checkable. Aggregate with Wilson 95% intervals and report `n`.

| Metric | Definition | Target |
| --- | --- | --- |
| `governing@1` | the record with `issue_code == rule_id` is present and first in the bundle | 1.00 |
| `level_integrity` | `plan.levels` equals resolved levels, conditions verbatim | 1.00 |
| `severity_equal` | exact string equality with `finding.severity` | 1.00 |
| `rule_id_preserved` | exact string equality, uppercase preserved | 1.00 |
| `foreign_rule_violations` | mentions of rules outside `{rule, siblings}` without a relation label | 0 |
| `invented_value_rate` | values/URLs/DOIs/names/versions not in evidence ∪ inputs | 0 |
| `invented_dataset_fact_rate` | claims about counts/existence/contents not in inputs | 0 |
| `example_policy` | example present iff permitted; shape-only when present | 1.00 |
| `abstention_correct` | no-direct statement present exactly when required | 1.00 |
| `attribution_valid` | every `evidence_id` resolves; direct facts supported by the cited text | 1.00 |
| `p95_latency_ms` | end-to-end | budget set per deployment |
| `model_calls_per_request` | counted | 1 |

`governing@1` is the metric that would have caught `TOO_FEW_AUTHORS` on day one. `level_integrity` is the metric that would have caught the `recommended` bug. Neither exists today. **[V]**

## G.3 Adversarial cases (all must be enforced)

| ID | Input finding | Must contain | Must not contain |
| --- | --- | --- | --- |
| T1 | `TOO_FEW_AUTHORS`, severity `warning` | the word "warning"; both level clauses | "error", "validation error", "is an error" |
| T2 | same | `TOO_FEW_AUTHORS` verbatim, uppercase | any other rule ID presented as the cause |
| T3 | same | "optional" **and** "recommended if `CITATION.cff` is not present" | level stated unconditionally |
| T4 | same | `NO_AUTHORS` only as "another check on the same field" | `NO_AUTHORS` as the cause; `NO_AUTHORS` level adopted |
| T5 | finding with `example_allowed=False` | no `Example` heading at all | any JSON snippet, any array literal, any `**Example**` |
| T6 | finding with `example_allowed=True` | `<...>` placeholders only | `http://…`, `10.xxxx/…`, `1.0`, real names |
| T7 | finding with no dataset tree | — | "your dataset has N …", subject counts, file existence claims |
| T8 | `bidsmgr.todo_placeholder` | "The knowledge base does not document this rule." | invented rule text, invented spec citation |
| T9 | validator hint (e.g. `TOO_FEW_AUTHORS`) | "validator check, not itself a BIDS requirement" | "BIDS requires" |
| T10 | finding with a BIDS version in evidence only | that exact version | any other version number |
| T11 | finding with no governing record | explicit unknown statement + what is needed | a confident answer |
| T12 | `SIDECAR_KEY_RECOMMENDED` with `SourceDatasets` | the field name and the recommended/required distinction **as stated in evidence** | the captured `prompt_log.txt` URL/DOI/version |

T5–T7, T9, T12 are derived directly from verified current failures. **[V]**

## G.4 Metamorphic tests (cheap, high value)

These encode architectural invariants without needing new annotations.

| ID | Transformation | Invariant |
| --- | --- | --- |
| M1 | Append the validator's boilerplate sentence 3× to the message | governing record and level set unchanged |
| M2 | `top_k ∈ {1,3,6,10}` | `(severity, rule_id, levels, example_allowed)` unchanged |
| M3 | Lexical ↔ embedding backend | same invariant as M2 |
| M4 | Reword the user question ("explain this" / "what does this mean" / "why did this fail") | same facts, same level |
| M5 | Remove `file_content` | no new concrete values appear |
| M6 | Add an unrelated `LabelingEfficiency` record to the path+field neighbourhood | it is still excluded |

M1 and M2 are the tests that would have caught F1 and C4. **[V]**

## G.5 LLM evaluation protocol

- Fixed `temperature=0.2` (current) — but for the eval, run each item 3× and report pass@3 and pass-all-3. A metric that passes 2/3 is not a pass.
- Record the model hash and the effective config with every run.
- Separate deterministic-tier failures (bug) from model-tier failures (capability).
- Do not tune prompts against the same 36 items you report on. Hold out ≥40%.

## G.6 CI gates

| Gate | Scope | Blocking |
| --- | --- | --- |
| Fast suite | unit + property | yes, <3 min |
| Contract invariance (M2, M3) | 120 items | yes |
| Adversarial (T1–T12) | deterministic | yes |
| `governing@1` | 120 items | yes, threshold 0.95 |
| Slow real-model | 10 items × 3 runs | advisory → blocking after 3 stable runs |

## G.7 What not to test

- Exact prompt wording (C9). Test properties instead.
- Full-answer string equality. Test claims.
- Retrieval *score* values. Test *membership and order of the governing record*.
- Model prose style. Not a product requirement.

---

# H. Final Prioritized Roadmap

## 🔴 P0 — Correctness of the claim (ship in ~1 week)

| ID | Item | Why | Effort | Risk if skipped |
| --- | --- | --- | --- | --- |
| P0-1 | Freeze the 4 verified gate misses + `TOO_FEW_AUTHORS` fixture as tests | Without this, every later change is unverifiable | 0.5 d | Fixes silently regress |
| P0-2 | `resolve_governing()` by exact key; typed `EvidenceBundle` | Removes the root cause of wrong evidence | 1.5 d | Keeps explaining the wrong rule |
| P0-3 | Contract from governing records only | Removes cross-rule contamination | 2 d | Still states a false BIDS level |
| P0-4 | `levels: list[(level, condition)]` | Restores the `CITATION.cff` condition | 1 d | Advice is wrong for datasets without `CITATION.cff` |
| P0-5 | Delete `**Example**` from the unconditional output list | Removes an internal contradiction | 0.5 d | Models keep producing forbidden examples |
| P0-6 | Whole-answer value provenance (V7) | Catches invented URLs/DOIs/names/counts | 2 d | Hallucinated values keep shipping |

**Exit criteria:** `TOO_FEW_AUTHORS` answer contains "warning", both level clauses, no foreign rule as cause, no example; the captured `prompt_log.txt` answer is rejected.

## 🟠 P1 — Make the guarantee structural (ship in ~3 weeks)

| ID | Item | Why | Effort | Risk if skipped |
| --- | --- | --- | --- | --- |
| P1-1 | Structured plan + validator V1–V11 | Turns prose regexes into comparisons | 5–7 d | Gate remains guesswork |
| P1-2 | Deterministic renderer | Removes format conflicts and repair churn | included | Structure stays model-dependent |
| P1-3 | Delete the LLM repair loop; template as fallback | One model call, no convergence theatre | 1 d | Latency, cost, false confidence |
| P1-4 | Metrics: `governing@1`, `level_integrity`, `severity_equal`, `attribution_valid` | Makes regressions visible | 3 d | Green CI stays meaningless |
| P1-5 | Adversarial suite T1–T12 | Encodes the requirements as tests | 2 d | Requirements remain aspirational |
| P1-6 | Fix `facts.severity` conflation in the KB | Removes the upstream seed of C2 | 1 d | Same bug returns via a new path |
| P1-7 | Replace byte-pinned prompt tests | Unblocks prompt work | 1 d | Every fix looks expensive |

## 🟢 P2 — Simplify and harden (ship when P1 is stable)

| ID | Item | Why | Effort |
| --- | --- | --- | --- |
| P2-1 | Remove LCEL; plain-Python planner | Tracebacks, testability, one less dependency | 2 d |
| P2-2 | Remove `intent_classifier.py`, `laya_decision.py`, `eval_intent.py`, `tool_registry.py`, `kb_retriever.py`, `prompt_log.py` | ~900 lines of non-product surface | 1 d |
| P2-3 | Make intent deterministic by default; LLM classifier opt-in | Halves latency, removes a failure mode | 1 d |
| P2-4 | `save_prompts=False`; remove `prompt_log.txt` from the tree | Privacy and hygiene | 0.25 d |
| P2-5 | `/config` returns **effective** config; startup logs it | Ends doc/runtime drift | 0.5 d |
| P2-6 | Fix README defaults; resolve `KB_README.md` | Reproducibility | 0.5 d |
| P2-7 | Expand quality set to ≥120 with held-out split | Statistical honesty | 2 d |
| P2-8 | Reimplement `observability.py` as a decorator | Keep step timings without LCEL | 0.5 d |

## Sequencing

```
Week 1        P0-1 … P0-6                    ← the week that changes correctness
Week 2        P1-1/P1-2 (plan + renderer)
Week 3        P1-3/P1-4/P1-5 (loop, metrics, adversarial)
Week 4        P1-6/P1-7 + P2-1/P2-2
Week 5        P2-3 … P2-8 + slow-suite stabilisation
```

## Definition of done

The project can claim "explains validator findings without inventing facts" only when:

1. `governing@1 = 1.00` on the quality set.
2. `level_integrity = 1.00`, including conditions.
3. `severity_equal = 1.00` and `rule_id_preserved = 1.00`.
4. `invented_value_rate = 0` and `invented_dataset_fact_rate = 0` on the adversarial suite.
5. `example_policy = 1.00` in both directions (present iff permitted; shape-only when present).
6. `abstention_correct = 1.00` on the six `bidsmgr.*` items plus a synthetic no-evidence set.
7. One model call per request, p95 within the deployment budget.
8. The effective configuration is visible at startup and matches the README.

## What I would not do

- Do not tune the model before P0-2/P0-3. A better model given the wrong record still gives the wrong answer. **[I]**
- Do not raise `default_top_k` to compensate for ranking. It changes contract permissions (verified) and adds noise. **[V]**
- Do not add a "confidence" number to answers. It would be a fabricated quantity unless derived from the retrieval score, which is not calibrated to correctness. **[I]**
- Do not ship the structured-output change without the template fallback; on a 1.7B model, JSON reliability is a measured unknown. **[?]**

---

# Appendix A — Reproduction commands

All probes used the repository's real code and the real KB.

```powershell
# 1. Baseline suite state
python -m pytest --collect-only -q        # 386/415 collected (29 deselected)

# 2. Lexical ranking for TOO_FEW_AUTHORS (verbatim KB message)
python tools/probe_retrieval.py           # see snippet below

# 3. Contract derivation at top_k = 3 and 6
python tools/probe_contract.py            # bids_level = "recommended" in both

# 4. Gate behaviour on four realistic violations
python tools/probe_gate.py                # all four print ok=True

# 5. Embedding backend reproduces the omission
python tools/probe_embedding.py           # 0.921 / 0.885 / 0.831, no Authors field
```

The probe scripts are intentionally not committed; they are four short files that:
- build the issue dict `{message, rule_id, field, severity, file}`,
- call `retriever.retrieve_issue_structured(issue, top_k=…)`,
- print `(score, relevance, record.id, record.title)` per assessment,
- call `build_contract(issue, retrieval)` and print `bids_level`, `example_allowed`, `authority`,
- call `output_gate.check(text, contract)` on candidate outputs.

They can be turned directly into the Phase 0 tests.

---

# Appendix B — External research consulted

Used for design principles only; none of it is evidence about this repository.

| Topic | Source | Relevance here |
| --- | --- | --- |
| Context sufficiency and abstention | Joren et al., *Sufficient Context: A New Lens on Retrieval Augmented Generation Systems*, ICLR 2025 (arXiv:2411.06037) | Models answer confidently from insufficient context; adding context reduces abstention. Supports "abstain when the governing record is missing" and "don't raise top_k as a fix". |
| Structured outputs do not guarantee correctness | OpenAI, *Structured Outputs* documentation ("Structured Outputs can still contain mistakes", "validate values in your application") | Justifies a deterministic validator after the plan, not reliance on schema conformance. |
| Attribution and citation verification | Gao et al., *Enabling Large Language Models to Generate Text with Citations* (ALCE), EMNLP 2023; Yue et al., *Automatic Evaluation of Attribution by Large Language Models*, 2023 | Basis for `evidence_id` binding and support checks (V5, V6, `attribution_valid`). |
| Instruction conflicts | Wallace et al., *The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions*, 2024 | Explains why the unconditional `**Example**` section beats the later "omit it" checklist item. |
| Long-context position effects | Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, TACL 2024 | Supports keeping the evidence bundle small and ordered (governing first). |
| Lexical vs dense retrieval | Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, 2009; Cormack et al., Reciprocal Rank Fusion, SIGIR 2009; Nogueira & Cho, *Passage Re-ranking with BERT*, 2019 | Exact-identifier queries favour sparse/lexical matching; dense is a recall aid; fusion and reranking help within a candidate set. Supports keeping lexical as primary and embedding as fallback. |
| BIDS authority | BIDS Specification (bids-standard/bids-specification), bids-validator source | The KB already encodes these as `source` + `authority`; the design keeps them as the only authority signal. |

---

# Appendix C — Open questions

These need a decision before Phase 3, not before Phase 1.

1. **JSON reliability of Qwen3-1.7B.** If constrained JSON fails >5% of the time, use the intermediate "labelled facts" format instead of a full plan. Measure on 20 items first. **[?]**
2. **Findings with `rule_id = null`.** 20 of the 36 quality items are `validator_impl`; some may lack a stable code. The fallback chain (rule_group+check_kind → path+field → no-direct) needs a measured usage rate. **[?]**
3. **Multi-version KB.** Records carry `bids_version`; `NO_AUTHORS` and `TOO_FEW_AUTHORS` are from `1.11.3-dev`. Should the agent answer for the user's dataset version or for the KB's version? Currently implicit. **[?]**
4. **`facts.severity` conflation.** Fixing the six `manager_todo`/level records changes KB content; confirm this is in scope. **[?]**
5. **Latency budget.** Removing the intent call and the repair call should net ~2× improvement, but the structured plan may be longer to generate. Benchmark both before committing. **[?]**
6. **Deployment privacy.** If `save_prompts` is used for support, it must be opt-in per request, not a config default. **[?]**

---

*Prepared from a clean tree at `a490c86`. No source, test, configuration, or knowledge-base file was modified. The only added file is this report (`DEVELOPMENT_PLAN.md`), which can be deleted if you prefer the analysis delivered only in conversation.*
