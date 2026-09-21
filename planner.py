# System Prompt + User Prompt + Retrieved Documents + Current Error + Conversation History(If Needed) => Model
from typing import Optional, Dict, List
from llm_client import LLMClient
from retriever import Retriever
from tool_registry import ToolRegistry
from utils import compact_json

class Planner:
    def __init__(self, llm: LLMClient, retriever: Retriever, tools: ToolRegistry):
        self.llm = llm
        self.retriever = retriever
        self.tools = tools

    def execute(self, system_prompt: str, user_input: str, context: dict):
        str_context = compact_json(context)

        intent = self._decided_intent(user_input, str_context)

        if intent == "fix":
            return self.fix(system_prompt, user_input, str_context)
        return self.explain(system_prompt, user_input, str_context, issue=context)

    def _decided_intent(self, user_input: str, context: str):
        fix_keywords = ["fix", "repair", "correct", "resolve", "clean", "rename"]
        if any(kw in user_input.lower() for kw in fix_keywords):
            return "fix"
        return "explain"

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
            retrieved_docs=retrieved_docs
        )

        return self.llm.generate_response(
            system_prompt=built_system_prompt,
            user_prompt=user_prompt
        )

    def fix(self, system_prompt:str, user_prompt: str, context: str) -> str:
        """Workflow 2: Retrieve context, pick & execute tool, then explain outcome."""
        retrieved_docs = self.retriever.retrieve(f"{user_prompt} {context}")
        built_system_prompt = self.build_system_prompt(
            system_prompt=system_prompt,
            context=context,
            retrieved_docs=retrieved_docs
            # user_input=user_input,
        )

        # 1. Ask LLM which tool to call
        tool_decision = self.llm.tool_chat(
            system_prompt=built_system_prompt,
            user_prompt=user_prompt,
            tools_schema=self.tools.get_tool_schemas()
        )

        # 2. Execute selected tool
        execution_result = self._call_tool(tool_decision, context)

        # 3. Summarize the tool result back to the user
        final_prompt = (
            f"This is a Summarization task. There is everything related to the task that is done by an AI agent. The AI agent has already executed a tool to fix the issue. Here is the context of the task:\n\n"
            f"[System prompt]\n{built_system_prompt}\n\n"
            f"[Action Taken]\n{execution_result}\n\n"
            f"Task: Briefly explain to the user what fix was applied and the result."
        )
        return self.llm.generate_response(
            # system_prompt=system_prompt, 
            # prompt=final_prompt
            system_prompt=final_prompt, 
            user_prompt=user_prompt
            )

    def _call_tool(self, tool_decision: str, context: str) -> str:
        """Parses decision and routes tool execution through the ToolRegistry."""
        try:
            # Basic parsing logic; adaptable to function call JSON payloads
            if "repair_metadata" in tool_decision:
                res = self.tools.execute("repair_metadata", context=context)
            elif "rename_subject" in tool_decision:
                res = self.tools.execute("rename_subject", context=context)
            elif "validate" in tool_decision:
                res = self.tools.execute("validate", context=context)
            else:
                return "No matching tool could be determined."
            return f"Successfully executed tool with output: {res}"
        except Exception as e:
            return f"Failed to execute tool: {str(e)}"

    def build_system_prompt(
        self,
        system_prompt: str,
        context: str,
        retrieved_docs: str,
        history: Optional[List[Dict[str, str]]] = None
        # user_input: str,
    ) -> str:
        """Constructs a structured prompt for the LLM."""
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
            # f"[User Question / Request]\n{user_input}"
        )