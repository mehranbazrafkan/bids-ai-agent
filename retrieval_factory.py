"""Construct the configured knowledge-base retriever.

Keeping this in one place means the API, tests and the agent all switch
backends the same way, and the embedding module (with its optional model
load) is imported only when it is actually selected.
"""

from __future__ import annotations

import logging
from typing import Optional

from config import CONFIG, RetrievalConfig
from retriever import Retriever

logger = logging.getLogger(__name__)

VALID_RETRIEVER_TYPES = ("lexical", "embedding")


def build_retriever(config: Optional[RetrievalConfig] = None) -> Retriever:
    """Return the retriever selected by ``config.retriever_type``.

    ``lexical`` is the default and keeps the verified deterministic path.
    ``embedding`` returns an :class:`embedding_retriever.EmbeddingRetriever`
    (a ``Retriever`` subclass), so the planner and the LangChain
    ``KBRetriever`` facade keep the same interface.
    """
    cfg = config or CONFIG.retrieval
    kind = (cfg.retriever_type or "lexical").strip().lower()
    if kind == "embedding":
        # Imported lazily so lexical-only deployments never load
        # transformers/NumPy for the retriever.
        from embedding_retriever import EmbeddingRetriever

        return EmbeddingRetriever(config=cfg)
    if kind != "lexical":
        logger.warning(
            "Unknown retriever_type %r; falling back to the lexical retriever.",
            cfg.retriever_type,
        )
    return Retriever(config=cfg)
