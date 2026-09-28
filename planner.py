from typing import Optional, Dict, List
from llm_client import LLMClient
from retriever import Retriever
from tool_registry import ToolRegistry
from laya_decision import LayaDecisionEngine
from utils import compact_json

# Fixed-keyword fallback when Laya intent classifier is unavailable.
_FIX_KEYWORDS = {"fix", "repair", "correct", "resolve", "clean", "rename"}


class Planner:
    def __init__(self, llm: LLMClient, retriever: Retriever, tools: ToolRegistry):
        self.llm = llm
        self.retriever = retriever
        self.tools = tools
        self.laya = LayaDecisionEngine()

    def execute(self, system_prompt: str, user_input: str, context: dict):
        str_context = compact_json(context)
        intent = self._decided_intent(user_input, str_context)
        if intent == "fix":
            return self.fix(system_prompt, user_input, str_context)
        return self.explain(system_prompt, user_input, str_context, issue=context)

    # ------------------------------------------------------------------
    # Intent classification — Laya-first, keyword fallback
    # ------------------------------------------------------------------

    def _decided_intent(self, user_input: str, context: str) -> str:
        """Return 'fix' or 'explain'.

        Uses Laya (non-autoregressive decision engine) for intent
        classification. Falls back to a simple keyword heuristic when
        Laya cannot make a confident decision.
        """
        laya_intent = self.laya.classify_intent(user_input, context)
        if laya_intent in ("fix", "explain"):
            return laya_intent
        # Keyword fallback
        if any(kw in user_input.lower() for kw in _FIX_KEYWORDS):
            return "fix"
        return "explain"

    # ------------------------------------------------------------------
    # Workflows
    # ------------------------------------------------------------------

    def explain(
        self,
        system_prompt: str,
        user_prompt: str,
        context: str,
        issue: Optional[dict] = None,
    ) -> str:
        if issue is not None:
            retrieved_docs = self.retriever.retrieve_issue(
                issue, user_question=user_prompt, top_k=1
            )
        else:
            retrieved_docs = self.retriever.retrieve(f"{user_prompt} {context}")
        built_system_prompt = self.build_system_prompt(
            system_prompt=system_prompt,
            context=context,
            retrieved_docs=retrieved_docs,
        )
        return self.llm.generate_response(
            system_prompt=built_system_prompt,
            user_prompt=user_prompt,
        )

    def fix(self, system_prompt: str, user_prompt: str, context: str) -> str:
        """Workflow 2: Retrieve context, pick & execute tool, then explain outcome."""
        retrieved_docs = self.retriever.retrieve(f"{user_prompt} {context}")
        built_system_prompt = self.build_system_prompt(
            system_prompt=system_prompt,
            context=context,
            retrieved_docs=retrieved_docs,
        )

        # 1. Use Laya to decide which tool to call (no LLM decision-making)
        tool_decision = self.laya.select_tool(user_prompt, context)

        # 2. Execute selected tool (dict-key match, no brittle substring)
        execution_result = self._call_tool(tool_decision, context)

        # 3. Summarize the tool result back to the user (LLM = output only)
        final_prompt = (
            "This is a Summarization task. There is everything related to "
            "the task that is done by an AI agent. The AI agent has already "
            "executed a tool to fix the issue. Here is the context of the "
            "task:\n\n"
            f"[System prompt]\n{built_system_prompt}\n\n"
            f"[Action Taken]\n{execution_result}\n\n"
            "Task: Briefly explain to the user what fix was applied and the "
            "result."
        )
        return self.llm.generate_response(
            system_prompt=final_prompt,
            user_prompt=user_prompt,
        )

    # ------------------------------------------------------------------
    # Tool routing
    # ------------------------------------------------------------------

    def _call_tool(self, tool_decision: Optional[str], context: str) -> str:
        """Route execution through ToolRegistry based on Laya's decision."""
        if tool_decision is None:
            return "No matching tool could be determined."

        tool_name = tool_decision.strip().lower()

        # Exact match in the registry
        if tool_name in self.tools.tools:
            try:
                res = self.tools.execute(tool_name, context=context)
                return f"Successfully executed tool with output: {res}"
            except Exception as e:
                return f"Failed to execute tool: {e}"

        # Substring fallback for robustness.
        for name in self.tools.tools:
            if name in tool_name:
                try:
                    res = self.tools.execute(name, context=context)
                    return f"Successfully executed tool with output: {res}"
                except Exception as e:
                    return f"Failed to execute tool: {e}"

        return "No matching tool could be determined."

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def build_system_prompt(
        self,
        system_prompt: str,
        context: str,
        retrieved_docs: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        formatted_history = ""
        if history:
            formatted_history = "\n[Conversation History]\n" + "\n".join(
                f"{msg['role']}: {msg['content']}" for msg in history
            )
        return (
            f"System: {system_prompt}\n\n"
            f"[Application Error / Context]\n{context}\n"
            f"[Retrieved BIDS Documentation]\n{retrieved_docs}\n\n"
            f"{formatted_history}\n"
        )
