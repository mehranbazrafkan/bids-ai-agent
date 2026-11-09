"""LangChain retriever facade over the custom BIDS knowledge base.

The knowledge base is a hand-tuned keyword system — deterministic
scoring, applicability selectors, quarantine of inapplicable records.
It is not a vector store, and this wrapper does not turn it into one.

The facade exists so the retriever speaks the LangChain interface
(``BaseRetriever``): it can be dropped into chains, it participates in
the callback system (its runs show up in :class:`observability.StepLogger`
and, later, in tracing tools), and downstream code can type against
``BaseRetriever`` without knowing the internals. Every call still goes
straight through :class:`retriever.Retriever` — the same scoring, the
same assessments, the same explicit miss outcome. Nothing here is
approximate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from retriever import RetrievalResult, Retriever

#: The marker a formatted record starts with (see ``Formatter.format``).
_RECORD_HEADER = "=== Knowledge Item"


class KBRetriever(BaseRetriever):
    """LangChain retriever facade over the custom BIDS knowledge base.

    ``core`` is the project's own :class:`retriever.Retriever`. The
    structured retrieval methods are delegated unchanged so existing
    call sites (planner, benchmark) keep their exact contract.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    core: Retriever
    #: ``top_k`` override for the flat LangChain path; ``None`` keeps the
    #: core's configured default (structured calls pass their own).
    top_k: Optional[int] = None

    # ------------------------------------------------------------------
    # LangChain retriever contract
    # ------------------------------------------------------------------

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """The kept records, one document each; a miss is an empty list.

        A retrieval *miss* returns no documents — the LangChain interface
        has no "miss" marker, and inventing one would paper over exactly
        the outcome the agent is built to declare honestly. Callers that
        need the distinction use :meth:`retrieve_structured` (as the
        planner does) and read the explicit ``found``/``quality`` fields.
        """
        result = self.core.retrieve_structured(query, top_k=self.top_k)
        if not result.found or not result.text:
            return []
        return self._documents(result)

    @staticmethod
    def _documents(result: RetrievalResult) -> List[Document]:
        """One :class:`~langchain_core.documents.Document` per kept record.

        The formatted blocks of ``result.text`` and the per-record
        assessments arrive in the same order, so they zip exactly. The
        assessment (relevance, match, applicability, score) travels in
        the document metadata — the same reasoning the prompt and the
        API diagnostics show, available to anything downstream that
        treats this as a plain LangChain retriever.
        """
        docs: List[Document] = []
        for block, assessment in zip(
            KBRetriever._split_records(result.text), result.assessments
        ):
            docs.append(
                Document(
                    page_content=block,
                    metadata={
                        "id": assessment.id,
                        "title": assessment.title,
                        "match": assessment.match,
                        "relevance": assessment.relevance,
                        "applicability": assessment.applicability,
                        "score": assessment.score,
                    },
                )
            )
        return docs

    @staticmethod
    def _split_records(text: str) -> List[str]:
        """Split the formatted evidence into one document per record.

        Records are formatted by ``Formatter`` as blocks beginning with
        ``=== Knowledge Item N ===`` and joined with single newlines, so
        a line-level split on the header is exact, not a guess.
        """
        docs: List[str] = []
        current: List[str] = []
        for line in text.splitlines():
            if line.startswith(_RECORD_HEADER) and current:
                docs.append("\n".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            docs.append("\n".join(current))
        return [doc for doc in docs if doc.strip()]

    # ------------------------------------------------------------------
    # Delegated structured API (the planner's actual contract)
    # ------------------------------------------------------------------

    def retrieve_structured(
        self, query: str, top_k: Optional[int] = None
    ) -> RetrievalResult:
        """Explicit-outcome retrieval (``found``/``quality``/assessments)."""
        return self.core.retrieve_structured(query, top_k=top_k)

    def retrieve_issue_structured(
        self,
        issue: dict,
        user_question: str = "",
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """Structured-issue retrieval with the bounded fallback."""
        return self.core.retrieve_issue_structured(
            issue, user_question=user_question, top_k=top_k
        )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> str:
        """Formatted text for a query; empty string on a miss."""
        return self.core.retrieve(query, top_k=top_k)

    def retrieve_issue(
        self,
        issue: dict,
        user_question: str = "",
        top_k: Optional[int] = None,
    ) -> str:
        """Formatted text for a structured issue; empty on a miss."""
        return self.core.retrieve_issue(
            issue, user_question=user_question, top_k=top_k
        )

    def explain(
        self,
        query: str,
        top_k: int = 5,
        record_id: Optional[str] = None,
        structured_query: Optional[Dict[str, str]] = None,
        issue: Optional[dict] = None,
    ) -> str:
        """Human-readable score breakdown for auditing retrieval."""
        return self.core.explain(
            query,
            top_k=top_k,
            record_id=record_id,
            structured_query=structured_query,
            issue=issue,
        )

    # ------------------------------------------------------------------
    # Introspection (kept for compatibility)
    # ------------------------------------------------------------------

    def get_record_count(self) -> int:
        return self.core.get_record_count()

    def get_record_types(self) -> Dict[str, int]:
        return self.core.get_record_types()
