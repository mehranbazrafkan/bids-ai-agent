"""Benchmark: measure per-step timing for each of the 4 test cases in main.py."""

import time
from agent import Agent
from utils import compact_json

sample_issue_001 = {
    "severity": "err",
    "rule_id": "JSON_SCHEMA_VALIDATION_ERROR",
    "message": "ReferencesAndLinks must be array  \u00b7  List of references to publications that contain information on the dataset. Use a value of the correct type, for example {\"ReferencesAndLinks\": [\"text\"]}.",
    "field": "ReferencesAndLinks",
    "line": "null",
    "lines": [],
    "fix_label": "null",
    "fix_action": "null",
    "mirrored": False
}

sample_issue_002 = {
    "severity": "warn",
    "rule_id": "bidsmgr.todo_placeholder",
    "message": "field 'EthicsApprovals' contains a TODO placeholder",
    "field": "EthicsApprovals",
    "line": "null",
    "lines": [],
    "fix_label": "Set a real value",
    "fix_action": "set_field",
    "mirrored": False
}

sample_issue_003 = {
    "path": "participants.tsv",
    "severity": "err",
    "datatype": "null",
    "suffix": "participants",
    "issues": [
        {
            "severity": "err",
            "rule_id": "TSV_VALUE_INCORRECT_TYPE",
            "message": "column 'age': value '020Y' is not valid for its type  \u00b7  Each value must be a valid number, or 'n/a'.",
            "field": "age",
            "line": 2,
            "lines": [2, 3],
            "fix_label": "null",
            "fix_action": "null",
            "mirrored": False
        }
    ],
    "sidecar_fields": []
}

sample_issue_004 = {
    "severity": "warn",
    "rule_id": "SIDECAR_KEY_RECOMMENDED",
    "message": "missing recommended field 'SequenceName'  \u00b7  Manufacturer's designation of the sequence name. Add it to the JSON, for example {\"SequenceName\": \"text\"}.",
    "field": "SequenceName",
    "line": "null",
    "lines": [],
    "fix_label": "Fix",
    "fix_action": "add_field",
    "mirrored": True
}

user_prompt = "Explain this to me."

issues = [
    ("sample_issue_001", sample_issue_001),
    ("sample_issue_002", sample_issue_002),
    ("sample_issue_003", sample_issue_003),
    ("sample_issue_004", sample_issue_004),
]

print("=" * 80)
print("BIDS AI Agent -- Per-Step Timing Benchmark")
print("=" * 80)

# -- Shared setup (one-time costs) --------------------------------------
t0 = time.perf_counter()
agent = Agent()
t_agent_init = time.perf_counter() - t0
print(f"\n[SETUP] Agent() construction (loads retriever + Laya): {t_agent_init:.2f}s")

# -- Per-test timing ----------------------------------------------------
all_results = []

for name, issue in issues:
    print(f"\n{'-' * 80}")
    print(f"TEST: {name}")
    print(f"{'-' * 80}")

    timings = {}

    # Step 1: compact_json
    t0 = time.perf_counter()
    str_context = compact_json(issue)
    timings["compact_json"] = time.perf_counter() - t0

    # Step 2: Laya intent classification (once per request)
    t0 = time.perf_counter()
    laya = agent.planner.laya
    decision = laya.classify_intent(user_prompt, str_context)
    intent = decision.intent
    timings["laya_classify_intent"] = time.perf_counter() - t0

    # Step 3: Retriever (structured, explicit outcome)
    t0 = time.perf_counter()
    retrieval = agent.planner.retriever.retrieve_issue_structured(
        issue, user_question=user_prompt
    )
    timings["retriever"] = time.perf_counter() - t0

    # Steps 4-5: Laya tool selection / tool execution are N/A in the
    # explanation-only phase: no tool is selected or executed.
    tool_decision = None
    execution_result = None
    timings["laya_select_tool"] = 0.0
    timings["tool_execution"] = 0.0

    # Step 6: LLM generation (the main bottleneck)
    from agent import system_prompt as agent_system_prompt
    t0 = time.perf_counter()
    built_system_prompt = agent.planner.build_system_prompt(
        system_prompt=agent_system_prompt,
        context=str_context,
        retrieval=retrieval,
        action_unavailable=(intent == "fix"),
        unclear=(intent == "unclear"),
    )
    response = agent.planner.llm.generate_response(
        system_prompt=built_system_prompt,
        user_prompt=user_prompt,
    )
    timings["llm_generation"] = time.perf_counter() - t0

    total = sum(timings.values())
    timings["TOTAL"] = total

    print(f"  Intent detected: {intent}")
    if tool_decision:
        print(f"  Tool selected:   {tool_decision}")
    print(f"  compact_json:         {timings['compact_json']:.3f}s")
    print(f"  laya_classify_intent: {timings['laya_classify_intent']:.3f}s")
    print(f"  retriever:            {timings['retriever']:.3f}s")
    print(f"  laya_select_tool:     {timings['laya_select_tool']:.3f}s")
    print(f"  tool_execution:       {timings['tool_execution']:.3f}s")
    print(f"  llm_generation:       {timings['llm_generation']:.3f}s")
    print(f"  " + "-" * 30)
    print(f"  TOTAL:                {total:.3f}s")

    all_results.append((name, intent, timings))

# -- Summary ------------------------------------------------------------
print(f"\n{'=' * 80}")
print("SUMMARY")
print(f"{'=' * 80}")
print(f"{'Test':<20} {'Intent':<10} {'Total':>8} {'LLM':>8} {'Laya':>8} {'Retrieve':>10}")
print(f"{'-' * 20} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 10}")
for name, intent, timings in all_results:
    laya_total = timings["laya_classify_intent"] + timings["laya_select_tool"]
    print(f"{name:<20} {intent:<10} {timings['TOTAL']:>7.2f}s {timings['llm_generation']:>7.2f}s {laya_total:>7.2f}s {timings['retriever']:>9.2f}s")

grand_total = sum(t["TOTAL"] for _, _, t in all_results)
print(f"\nGrand total (all 4 tests): {grand_total:.2f}s")
