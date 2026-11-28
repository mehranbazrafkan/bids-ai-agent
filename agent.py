from __future__ import annotations

from typing import Optional

from intent import build_intent_classifier
from kb_retriever import KBRetriever
from local_chat_model import LocalChatModel
from planner import Planner
from prompt_export import render_prompt_log
from prompts import BIDS_SYSTEM_PROMPT
from retrieval_factory import build_retriever
from tool_registry import ToolRegistry
from llm_client import LLMClient
from config import CONFIG, AgentConfig
from utils import *

#: The agent's system instructions, re-exported from ``prompts`` — the
#: module that owns the wording (and the tests that pin it). The old
#: import location stays valid for callers that use it (benchmark
#: scripts, external tools): ``from agent import system_prompt``.
system_prompt = BIDS_SYSTEM_PROMPT


class Agent:
    def __init__(self, config: AgentConfig = CONFIG):
        self.config = config
        # The legacy flat client stays available for call sites that
        # still use it; the planner itself runs through the LangChain
        # chat model wrapping the same backend ``LLM`` object.
        self.llm = LLMClient(config=self.config)
        self.chat_model = LocalChatModel(
            llm=self.llm.llm,
            model_name=self.config.llm.model_name,
        )
        self.retriever = build_retriever(self.config.retrieval)
        # The LangChain facade over the same knowledge base: the planner
        # speaks the retriever interface, direct callers keep the exact
        # structured API through the delegation.
        self.kb_retriever = KBRetriever(core=self.retriever)
        self.tools = ToolRegistry()
        # Auto mode uses the hybrid guard/LLM router by default; forced
        # explain/fix modes deliberately build no classifier.
        self.intent_classifier = build_intent_classifier(
            self.config.intent, self.llm.llm
        )
        # Compatibility alias for the timing benchmark and external callers
        # that used ``agent.laya`` / ``agent.planner.laya``.
        self.laya = self.intent_classifier

        self.planner = Planner(
            llm=self.chat_model,
            retriever=self.kb_retriever,
            tools=self.tools,
            intent_classifier=self.intent_classifier)

    def _system_prompt(self) -> str:
        """The instructions configured for this agent, or the built-in ones."""
        return self.config.prompt.effective_system_prompt

    def _dataset_structure(self, value: Optional[str]) -> Optional[str]:
        """Drop the tree when Settings says the request should not carry it."""
        return value if self.config.prompt.include_dataset_structure else None

    def _file_content(self, value: Optional[str]) -> Optional[str]:
        """Drop the compiled file digest when Settings turns it off."""
        return value if self.config.prompt.include_file_content else None

    def run(
        self,
        user_input: str,
        context: dict,
        history: Optional[list] = None,
        dataset_structure: Optional[str] = None,
        file_content: Optional[str] = None,
    ):
        """
        Main Entry point for the agent.

        Returns a ``PlannerResult`` carrying the generated response plus
        the metadata that describes what actually happened (the intent
        that was used, the classification source, the honest
        ``tool_used=None`` / ``execution_status`` for action requests,
        and the retrieval outcome).

        ``history`` is the conversation so far — earlier questions and
        answers about this same finding. ``dataset_structure`` and
        ``file_content`` are the evidence about the dataset itself: its
        shape, and a compiled digest of the file in question. All three
        default rather than being required, because a first ask about a
        finding with no path behind it owes none of them.
        """
        return self.planner.execute(
            system_prompt=self._system_prompt(),
            user_input=user_input,
            context=context,
            history=history,
            dataset_structure=self._dataset_structure(dataset_structure),
            file_content=self._file_content(file_content),
        )

    def build_prompt(
        self,
        user_input: str,
        context: dict,
        history: Optional[list] = None,
        dataset_structure: Optional[str] = None,
        file_content: Optional[str] = None,
    ) -> dict:
        """Assemble the request side of an answer without running the model.

        See :meth:`Planner.build_prompt`: the same intent decision,
        retrieval and prompt rendering as :meth:`run`, with no generation
        and no checkpoint load. The JSON form is returned by ``POST
        /prompt``; the desktop export uses :meth:`build_prompt_log` for the
        plain-text file it writes per validation finding.
        """
        return self.planner.build_prompt(
            system_prompt=self._system_prompt(),
            user_input=user_input,
            context=context,
            history=history,
            dataset_structure=self._dataset_structure(dataset_structure),
            file_content=self._file_content(file_content),
        )

    def build_prompt_log(
        self,
        user_input: str,
        context: dict,
        history: Optional[list] = None,
        dataset_structure: Optional[str] = None,
        file_content: Optional[str] = None,
    ) -> str:
        """The prompt record formatted as one plain-text export file.

        This is what ``POST /prompt/log`` returns to the desktop app. It is
        the same request :meth:`build_prompt` assembles, rendered without a
        generated answer so each file can be used directly as a fine-tuning
        prompt or reviewed beside the validation finding.
        """
        return render_prompt_log(self.build_prompt(
            user_input=user_input,
            context=context,
            history=history,
            dataset_structure=dataset_structure,
            file_content=file_content,
        ))
