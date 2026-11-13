"""Tests for the semantic retriever and its persistent vector cache.

A deterministic fake encoder keeps these tests fast and offline; the real
transformer backend is exercised only by the slow/model-machine suite.
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pytest

from config import RetrievalConfig
from embedding_retriever import EmbeddingRetriever
from retrieval_factory import build_retriever
from retriever import Retriever

DIM = 64


class FakeEmbedder:
    """Stable bag-of-tokens encoder with L2-normalized rows."""

    def __init__(self) -> None:
        self.calls = []

    def encode(self, texts, batch_size=32):
        self.calls.append(list(texts))
        rows = []
        for text in texts:
            vector = np.zeros(DIM, dtype=np.float32)
            for token in re.findall(r"[a-z0-9]+", str(text).lower()):
                h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % DIM
                vector[h] += 1.0
            norm = float(np.linalg.norm(vector))
            if norm:
                vector /= norm
            rows.append(vector)
        if not rows:
            return np.zeros((0, DIM), dtype=np.float32)
        return np.stack(rows)


def _record(record_id, title, text, *, code="", selectors=None):
    data = {
        "id": record_id,
        "knowledge_type": "field",
        "title": title,
        "summary": text,
        "retrieval_text": text,
        "selectors": selectors or [],
        "raw_content": {},
    }
    if code:
        data["facts"] = {"issue_code": code}
    return data


def _write_kb(path, records):
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


@pytest.fixture()
def kb(tmp_path):
    path = tmp_path / "knowledge_base.jsonl"
    records = [
        _record(
            "kb-license",
            "License field",
            "License must be a valid license identifier such as CC0.",
            code="LICENSE_INVALID",
        ),
        _record(
            "kb-events",
            "Events file",
            "Events files must use the events suffix and a TSV extension.",
        ),
        _record(
            "kb-description",
            "Dataset description",
            "DatasetDescription is a JSON sidecar describing the dataset.",
        ),
    ]
    _write_kb(path, records)
    return path


def _cfg(tmp_path, **overrides):
    values = dict(
        knowledge_base_dir=str(tmp_path),
        vector_store_dir=str(tmp_path / "vectors"),
        retriever_type="embedding",
        embedding_model="fake-embedder",
        embedding_min_score=0.0,
        default_top_k=2,
        embedding_candidate_k=4,
    )
    values.update(overrides)
    return RetrievalConfig(**values)


def test_builds_and_reuses_vector_store(tmp_path, kb):
    fake = FakeEmbedder()
    retriever = EmbeddingRetriever(config=_cfg(tmp_path), embedder=fake)
    assert retriever.vectors_from_cache is False
    assert retriever.vector_count == 3
    assert (tmp_path / "vectors" / "vectors.npy").exists()
    assert (tmp_path / "vectors" / "manifest.json").exists()
    assert len(fake.calls) == 1
    assert len(fake.calls[0]) == 3

    # A second process/instance must load the cache without encoding the
    # documents again. Only the query is encoded.
    fake2 = FakeEmbedder()
    retriever2 = EmbeddingRetriever(config=_cfg(tmp_path), embedder=fake2)
    assert retriever2.vectors_from_cache is True
    assert fake2.calls == []
    result = retriever2.retrieve_structured("license")
    assert result.found is True
    assert "License" in result.text
    assert fake2.calls and len(fake2.calls[0]) == 1

    manifest = json.loads(
        (tmp_path / "vectors" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["count"] == 3
    assert manifest["model_name"] == "fake-embedder"
    assert manifest["dimension"] == DIM


def test_changed_knowledge_base_rebuilds_and_deletes_old_store(tmp_path, kb):
    fake = FakeEmbedder()
    EmbeddingRetriever(config=_cfg(tmp_path), embedder=fake).retrieve_structured("license")

    with open(kb, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(_record(
            "kb-new", "New task rule", "A new task entity rule for task files."
        )) + "\n")

    fake2 = FakeEmbedder()
    retriever = EmbeddingRetriever(config=_cfg(tmp_path), embedder=fake2)
    assert retriever.vectors_from_cache is False
    assert retriever.vector_count == 4
    assert len(fake2.calls) == 1 and len(fake2.calls[0]) == 4

    store = tmp_path / "vectors"
    assert not list(store.glob("*.tmp"))
    assert len(list(store.glob("vectors*"))) == 1
    manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["count"] == 4


def test_changed_embedding_model_rebuilds(tmp_path, kb):
    EmbeddingRetriever(config=_cfg(tmp_path), embedder=FakeEmbedder())
    fake = FakeEmbedder()
    retriever = EmbeddingRetriever(
        config=_cfg(tmp_path, embedding_model="fake-embedder-v2"),
        embedder=fake,
    )
    assert retriever.vectors_from_cache is False
    assert retriever.vector_status()["model"] == "fake-embedder-v2"
    manifest = json.loads(
        (tmp_path / "vectors" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["model_name"] == "fake-embedder-v2"


def test_structured_issue_exact_rule_is_rule_exact(tmp_path, kb):
    retriever = EmbeddingRetriever(config=_cfg(tmp_path), embedder=FakeEmbedder())
    result = retriever.retrieve_issue_structured(
        {
            "rule_id": "LICENSE_INVALID",
            "field": "License",
            "message": "License is invalid",
            "severity": "error",
        },
        user_question="What does this error mean?",
    )
    assert result.found is True
    assert result.quality == "rule_exact"
    assert result.has_direct is True


def test_plain_semantic_match_is_marked_semantic(tmp_path, kb):
    retriever = EmbeddingRetriever(config=_cfg(tmp_path), embedder=FakeEmbedder())
    result = retriever.retrieve_structured("license identifier")
    assert result.found is True
    assert result.quality == "semantic"
    assert "License" in result.text


def test_factory_selects_backend(tmp_path, kb, monkeypatch):
    lexical = build_retriever(RetrievalConfig(
        knowledge_base_dir=str(tmp_path),
        retriever_type="lexical",
    ))
    assert isinstance(lexical, Retriever)
    assert not isinstance(lexical, EmbeddingRetriever)

    class DummyEmbedding(Retriever):
        pass

    monkeypatch.setattr("embedding_retriever.EmbeddingRetriever", DummyEmbedding)
    selected = build_retriever(RetrievalConfig(
        knowledge_base_dir=str(tmp_path),
        retriever_type="embedding",
    ))
    assert isinstance(selected, DummyEmbedding)
