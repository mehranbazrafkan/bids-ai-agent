#!/usr/bin/env python3
"""Semantic knowledge-base retrieval with an on-disk vector cache.

The project kept the original lexical :class:`retriever.Retriever` because
it is deterministic, explainable and performs well on BIDS rule identifiers.
This module adds a second backend without touching that code:

* the same knowledge-base loader, record model, applicability selectors,
  assessments and formatter are reused;
* only the *ranking* step is replaced by cosine similarity between the
  query embedding and pre-computed record embeddings;
* vectors are persisted as ``vectors.npy`` + ``manifest.json`` in
  ``RetrievalConfig.vector_store_dir`` and rebuilt automatically when the
  knowledge base or the embedding model changes.

The cache is deliberately dependency-light (NumPy + Transformers, both
already required by the LLM side of the project).  The manifest records a
SHA-256 of the knowledge file, the model name, pooling mode, vector shape and
the record-id order.  Any mismatch deletes the old files and writes a new
store, so stale vectors can never be silently mixed with a changed KB.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from applicability import IssueScope, build_issue_scope, classify_selectors
from config import RetrievalConfig
from retriever import (
    MAX_REJECTED,
    MSG_EMPTY_KB,
    MSG_EMPTY_QUERY,
    MSG_NO_KB,
    MSG_NO_MATCH,
    Formatter,
    KnowledgeRecord,
    RecordAssessment,
    RetrievalResult,
    Retriever,
    _issue_structured,
    _parse_query,
    _rejection_text,
)

logger = logging.getLogger(__name__)

_STORE_FORMAT_VERSION = 1
_VECTORS_FILE = "vectors.npy"
_MANIFEST_FILE = "manifest.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_ids(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row; zero rows remain zero."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)


class TransformerEmbedder:
    """Small wrapper around a Hugging Face encoder.

    ``pooling="cls"`` is the correct setting for BGE checkpoints;
    ``pooling="mean"`` works for most other sentence-transformer models.
    Both paths L2-normalize the output so cosine similarity is a dot
    product.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_length: int = 512,
        pooling: str = "cls",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.requested_device = device
        self.max_length = int(max_length)
        self.pooling = (pooling or "cls").strip().lower()
        self.batch_size = max(1, int(batch_size))
        self.dimension: Optional[int] = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: Optional[str] = None

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = self._resolve_device()
        return self._device

    def _resolve_device(self) -> str:
        requested = (self.requested_device or "auto").strip().lower()
        if requested == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:  # pragma: no cover - torch is a hard dependency
                return "cpu"
        if requested.startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    logger.warning(
                        "Embedding device %r requested but CUDA is unavailable; "
                        "falling back to CPU.",
                        requested,
                    )
                    return "cpu"
            except Exception:  # pragma: no cover
                return "cpu"
        return requested

    def load(self) -> None:
        """Load the tokenizer/model once."""
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - transformers is pinned
            raise RuntimeError(
                "The embedding retriever needs transformers and torch. "
                "Install requirements.txt or switch retrieval.retriever_type "
                "back to 'lexical'."
            ) from exc

        logger.info(
            "Loading embedding model %s on %s (pooling=%s).",
            self.model_name,
            self.device,
            self.pooling,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(self.device)
        hidden = getattr(self._model.config, "hidden_size", None)
        if hidden is None:
            hidden = getattr(self._model.config, "d_model", None)
        if hidden is None:
            # Last-resort probe; avoids failing after a successful model load.
            with torch.inference_mode():
                probe = self._tokenizer(
                    ["probe"], return_tensors="pt", truncation=True,
                    max_length=8,
                ).to(self.device)
                probe_out = self._model(**probe)
            hidden = int(probe_out.last_hidden_state.shape[-1])
        self.dimension = int(hidden)

    def encode(
        self,
        texts: Sequence[str],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Encode texts into a float32, L2-normalized matrix."""
        self.load()
        import torch

        assert self._tokenizer is not None and self._model is not None
        batch_size = max(1, int(batch_size or self.batch_size))
        outputs: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = [str(text) for text in texts[start:start + batch_size]]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                model_output = self._model(**encoded)
                hidden = model_output.last_hidden_state
                if self.pooling == "mean":
                    mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                else:
                    pooled = hidden[:, 0]
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            outputs.append(pooled.detach().cpu().numpy().astype(np.float32))
        if not outputs:
            return np.zeros((0, self.dimension or 0), dtype=np.float32)
        return np.vstack(outputs)


class EmbeddingRetriever(Retriever):
    """Retriever that ranks the same KB records by embedding similarity.

    It inherits loading, indexing, applicability assessment and formatting
    from :class:`Retriever`; the lexical index is still built because the
    structured ``rule_exact`` / ``field_exact`` assessments are valuable and
    cheap (about 3.5k records).  ``_retrieve_result`` is the only ranking
    method overridden.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        config: Optional[RetrievalConfig] = None,
        embedder: Optional[Any] = None,
    ) -> None:
        self._embedder = embedder
        self._vector_lock = threading.RLock()
        self._vectors: Optional[np.ndarray] = None
        self._manifest: Dict[str, Any] = {}
        self._cache_hit = False

        super().__init__(data_dir=data_dir, config=config)

        self._store_dir = Path(self.config.vector_store_dir)
        self._vectors_path = self._store_dir / _VECTORS_FILE
        self._manifest_path = self._store_dir / _MANIFEST_FILE
        self._ensure_vectors()

    # -- Public introspection ------------------------------------------

    @property
    def vectors_from_cache(self) -> bool:
        """True when the last ``_ensure_vectors`` call reused valid files."""
        return self._cache_hit

    @property
    def vector_count(self) -> int:
        return 0 if self._vectors is None else int(self._vectors.shape[0])

    def vector_status(self) -> Dict[str, Any]:
        """Small, JSON-safe description used by diagnostics and tests."""
        return {
            "backend": "embedding",
            "model": self.config.embedding_model,
            "dimension": int(self._vectors.shape[1]) if self._vectors is not None else 0,
            "count": self.vector_count,
            "store_dir": str(self._store_dir),
            "cache_hit": self._cache_hit,
            "knowledge_sha256": self._manifest.get("knowledge_sha256", ""),
        }

    # -- Vector cache ----------------------------------------------------

    def _knowledge_path(self) -> Path:
        return self.data_dir / self.knowledge_file

    def _knowledge_fingerprint(self) -> Dict[str, Any]:
        path = self._knowledge_path()
        if not path.exists():
            return {}
        return {
            "knowledge_sha256": _sha256_file(path),
            "knowledge_size": path.stat().st_size,
        }

    def _current_ids(self) -> List[str]:
        return [record.id for record in self.records]

    def _expected_manifest(self, fingerprint: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "format_version": _STORE_FORMAT_VERSION,
            "backend": "embedding",
            "model_name": self.config.embedding_model,
            "pooling": (self.config.embedding_pooling or "cls").lower(),
            "max_length": int(self.config.embedding_max_length),
            "count": len(self.records),
            "record_ids_sha256": _sha256_ids(self._current_ids()),
            **fingerprint,
        }

    def _manifest_matches(self, manifest: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        # ``created_at`` and ``dimension`` are intentionally not compared
        # here; dimension is validated against the .npy shape below.
        for key in (
            "format_version",
            "backend",
            "model_name",
            "pooling",
            "max_length",
            "count",
            "record_ids_sha256",
            "knowledge_sha256",
            "knowledge_size",
        ):
            if manifest.get(key) != expected.get(key):
                return False
        return True

    def _load_cached_vectors(self, expected: Dict[str, Any]) -> bool:
        if not self._vectors_path.exists() or not self._manifest_path.exists():
            return False
        try:
            manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable embedding manifest: %s", exc)
            return False
        if not isinstance(manifest, dict) or not self._manifest_matches(manifest, expected):
            return False
        try:
            vectors = np.load(self._vectors_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            logger.warning("Ignoring unreadable embedding vectors: %s", exc)
            return False
        if vectors.ndim != 2 or vectors.shape[0] != len(self.records):
            logger.warning(
                "Embedding cache shape %s does not match %d records; rebuilding.",
                getattr(vectors, "shape", None),
                len(self.records),
            )
            return False
        if int(manifest.get("dimension", -1)) != int(vectors.shape[1]):
            logger.warning(
                "Embedding manifest dimension %r does not match vector shape %r; "
                "rebuilding.",
                manifest.get("dimension"),
                vectors.shape[1],
            )
            return False
        if not np.isfinite(vectors).all():
            logger.warning("Embedding cache contains non-finite values; rebuilding.")
            return False
        self._vectors = vectors.astype(np.float32, copy=False)
        self._manifest = manifest
        self._cache_hit = True
        logger.info(
            "Loaded %d cached embeddings from %s (dim=%d, model=%s).",
            vectors.shape[0],
            self._vectors_path,
            vectors.shape[1],
            manifest.get("model_name"),
        )
        return True

    def _clear_store(self) -> None:
        self._store_dir.mkdir(parents=True, exist_ok=True)
        for path in (
            self._vectors_path,
            self._manifest_path,
            self._vectors_path.with_name(_VECTORS_FILE + ".tmp"),
            self._manifest_path.with_name(_MANIFEST_FILE + ".tmp"),
        ):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:  # pragma: no cover - defensive
                logger.warning("Could not remove stale vector file %s: %s", path, exc)

    def _ensure_vectors(self) -> None:
        if not self.records:
            return
        with self._vector_lock:
            if self._vectors is not None:
                return
            fingerprint = self._knowledge_fingerprint()
            expected = self._expected_manifest(fingerprint)
            if self._vectors is None and self._load_cached_vectors(expected):
                return
            self._rebuild_vectors(expected)

    def _rebuild_vectors(self, expected: Dict[str, Any]) -> None:
        logger.info(
            "Embedding cache missing or stale; generating vectors for %d records "
            "with %s.",
            len(self.records),
            self.config.embedding_model,
        )
        embedder = self._get_embedder()
        documents = [self._document_text(record) for record in self.records]
        vectors = embedder.encode(
            documents,
            batch_size=self.config.embedding_batch_size,
        )
        vectors = _normalize_rows(vectors)
        if vectors.ndim != 2 or vectors.shape[0] != len(self.records):
            raise RuntimeError(
                "Embedding backend returned an unexpected matrix shape "
                f"{vectors.shape}; expected ({len(self.records)}, dim)."
            )
        # Old files are removed before the new ones are moved into place.
        # This is what guarantees a changed KB/model cannot leave duplicate
        # or mixed-generation vectors behind.
        self._clear_store()
        vectors_tmp = self._vectors_path.with_name(_VECTORS_FILE + ".tmp")
        manifest_tmp = self._manifest_path.with_name(_MANIFEST_FILE + ".tmp")
        # Pass a file handle: np.save would append ".npy" to a path such as
        # "vectors.npy.tmp" and break the atomic replace below.
        with open(vectors_tmp, "wb") as handle:
            np.save(handle, vectors.astype(np.float32, copy=False))
        manifest = {
            **expected,
            "dimension": int(vectors.shape[1]),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_tmp.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(vectors_tmp, self._vectors_path)
        os.replace(manifest_tmp, self._manifest_path)
        self._vectors = vectors.astype(np.float32, copy=False)
        self._manifest = manifest
        self._cache_hit = False
        logger.info(
            "Saved %d embeddings (dim=%d) to %s.",
            vectors.shape[0],
            vectors.shape[1],
            self._store_dir,
        )

    def _get_embedder(self) -> Any:
        if self._embedder is None:
            self._embedder = TransformerEmbedder(
                model_name=self.config.embedding_model,
                device=self.config.embedding_device,
                max_length=self.config.embedding_max_length,
                pooling=self.config.embedding_pooling,
                batch_size=self.config.embedding_batch_size,
            )
        return self._embedder

    # -- Record / query text ---------------------------------------------

    @staticmethod
    def _document_text(record: KnowledgeRecord) -> str:
        parts: List[str] = [
            record.title,
            record.summary,
            record.retrieval_text,
        ]
        if record.rule_codes:
            parts.append("Rule codes: " + ", ".join(record.rule_codes))
        if record.scope:
            parts.append("Scope: " + json.dumps(record.scope, ensure_ascii=False, default=str))
        if record.requirements:
            parts.append(
                "Requirements: "
                + json.dumps(record.requirements, ensure_ascii=False, default=str)
            )
        if record.conditions:
            parts.append(
                "Conditions: "
                + json.dumps(record.conditions, ensure_ascii=False, default=str)
            )
        if record.allowed_values:
            parts.append(
                "Allowed values: "
                + json.dumps(record.allowed_values, ensure_ascii=False, default=str)
            )
        # The tokenizer truncates at max_length anyway; capping here avoids
        # constructing huge strings for records with long enrichment blocks.
        return "\n".join(part for part in parts if part).strip()[:6000]

    def _encode_query(self, text: str) -> np.ndarray:
        vector = self._get_embedder().encode([text], batch_size=1)[0]
        return _normalize_rows(vector)[0]

    def _query_text(
        self,
        query: str,
        structured_query: Optional[Dict[str, str]],
    ) -> str:
        if structured_query:
            # Natural order, no "field:"/"rule_id:" labels. BGE handles a
            # short sentence-like query better than a labelled key/value
            # dump; the exact identifiers are still present for the lexical
            # candidate union below.
            parts: List[str] = []
            for key in (
                "general_text",
                "message_text",
                "field",
                "rule_id",
                "path",
                "suffix",
                "datatype",
            ):
                value = str(structured_query.get(key) or "").strip()
                if value and value not in parts:
                    parts.append(value)
            text = " ".join(parts) if parts else query
        else:
            text = query
        prefix = self.config.embedding_query_prefix or ""
        return (prefix + str(text)).strip()

    # -- Retrieval -------------------------------------------------------

    def _retrieve_result(
        self,
        query: str,
        top_k: int = 1,
        structured_query: Optional[Dict[str, str]] = None,
        scope: Optional[IssueScope] = None,
    ) -> RetrievalResult:
        """Rank records by embedding similarity, then assess and format.

        Applicability rejection and deduplication are identical to the
        lexical retriever; only ``score`` carries cosine similarity.
        """
        if not self.records:
            msg = (
                MSG_EMPTY_KB if self._knowledge_file_exists()
                else MSG_NO_KB.format(self.data_dir)
            )
            return RetrievalResult(False, "none", [], [], msg)
        if not query or not query.strip():
            return RetrievalResult(False, "none", [], [], MSG_EMPTY_QUERY)

        if structured_query is None:
            structured_query = _parse_query(query)
        if scope is None:
            scope = build_issue_scope(structured_query) if structured_query else IssueScope()

        self._ensure_vectors()
        if self._vectors is None or self._vectors.shape[0] != len(self.records):
            return RetrievalResult(
                False,
                "none",
                [],
                [],
                "The embedding cache is unavailable; no semantic retrieval was performed.",
            )

        query_text = self._query_text(query, structured_query)
        query_vector = self._encode_query(query_text)
        scores = self._vectors @ query_vector.astype(np.float32, copy=False)

        top_k = max(1, int(top_k))
        candidate_k = min(
            len(self.records),
            max(
                top_k,
                int(self.config.embedding_candidate_k),
                top_k * 4,
                top_k + MAX_REJECTED,
            ),
        )
        ranked = np.argsort(-scores)[:candidate_k]
        candidate_indices = {int(idx) for idx in ranked}

        # Exact identifiers are too important to leave to an encoder. A
        # record documenting the issue's rule or field may be semantically
        # similar to several generic error records; union the semantic
        # candidates with the lexical retriever's best structured
        # candidates, then rerank the union with a small exact-match bonus.
        if structured_query and any(
            structured_query.get(key)
            for key in ("rule_id", "field", "path", "suffix", "datatype")
        ):
            try:
                lexical_candidates = self._candidate_indices(query, structured_query)
                lexical_scored = self._score_candidates(
                    query, lexical_candidates, structured_query
                )
                lexical_limit = max(candidate_k * 2, 24)
                candidate_indices.update(
                    int(idx) for _score, idx in lexical_scored[:lexical_limit]
                )
            except Exception:  # pragma: no cover - defensive
                logger.exception("Lexical candidate union failed; using semantic only.")

        assessed: List[Tuple[float, float, int, RecordAssessment, Any]] = []
        for idx in candidate_indices:
            semantic_score = float(scores[idx])
            record = self.records[idx]
            applicability = classify_selectors(record.selectors, scope)
            assessment = self._assess(
                query, idx, semantic_score, structured_query, scope, applicability
            )
            boost = (
                0.30 if assessment.match == "rule_exact"
                else 0.20 if assessment.match == "field_exact"
                else 0.0
            )
            assessed.append(
                (semantic_score + boost, semantic_score, idx, assessment, applicability)
            )
        assessed.sort(key=lambda item: item[0], reverse=True)

        best_semantic = max((item[1] for item in assessed), default=0.0)
        has_exact = any(item[3].match in ("rule_exact", "field_exact") for item in assessed)
        if best_semantic < float(self.config.embedding_min_score) and not has_exact:
            return RetrievalResult(
                False, "none", [], [], MSG_NO_MATCH.format(query)
            )

        selected: List[Tuple[float, int, RecordAssessment]] = []
        rejected: List[RecordAssessment] = []
        seen_keys: set[str] = set()
        for _combined, semantic_score, idx, assessment, applicability in assessed:
            if len(selected) >= top_k:
                break
            if (
                semantic_score < float(self.config.embedding_min_score)
                and assessment.match not in ("rule_exact", "field_exact")
            ):
                continue
            if applicability.is_inapplicable:
                if len(rejected) < MAX_REJECTED:
                    rejected.append(assessment)
                continue
            record = self.records[idx]
            key = self._dedupe_key(record)
            if key:
                if key in seen_keys:
                    continue
                seen_keys.add(key)
            # Keep the reported score as the cosine similarity; ``combined``
            # exists only to put exact identifiers ahead of generic
            # semantically-similar records.
            selected.append((semantic_score, idx, assessment))

        if not selected:
            return RetrievalResult(
                False,
                "none",
                [],
                [],
                MSG_NO_MATCH.format(query),
                assessments=[],
                rejected=rejected,
                rejection_text=_rejection_text(rejected),
            )

        chunks = []
        for rank, (_score, idx, assessment) in enumerate(selected, 1):
            record = self.records[idx]
            chunks.append(Formatter.format(record, rank, assessment=assessment))

        quality = selected[0][2].match
        if quality == "lexical":
            # The record was selected by semantic similarity rather than an
            # exact rule/field identifier. Keep the distinction visible to
            # the prompt and diagnostics instead of calling it "lexical".
            quality = "semantic"

        return RetrievalResult(
            True,
            quality,
            [idx for _s, idx, _a in selected],
            [round(s, 3) for s, _idx, _a in selected],
            "\n".join(chunks),
            assessments=[a for _s, _i, a in selected],
            rejected=rejected,
            rejection_text=_rejection_text(rejected),
        )

    def explain(
        self,
        query: str,
        top_k: int = 5,
        record_id: Optional[str] = None,
        structured_query: Optional[Dict[str, str]] = None,
        issue: Optional[dict] = None,
    ) -> str:
        """Human-readable semantic ranking, used by audit scripts/tests."""
        if issue is not None:
            structured_query = _issue_structured(issue)
        if structured_query is None:
            structured_query = _parse_query(query)
        self._ensure_vectors()
        if self._vectors is None or not self.records:
            return "No embedding vectors are available."
        query_vector = self._encode_query(
            self._query_text(query, structured_query)
        )
        scores = self._vectors @ query_vector.astype(np.float32, copy=False)
        order = np.argsort(-scores)
        lines = [
            f"[EmbeddingRetriever] model={self.config.embedding_model} "
            f"dim={self._vectors.shape[1]} records={len(self.records)} "
            f"store={self._store_dir}"
        ]
        shown = 0
        for raw_idx in order:
            idx = int(raw_idx)
            record = self.records[idx]
            if record_id and record.id != record_id:
                continue
            lines.append(f"{float(scores[idx]):.4f}  {record.id}  {record.title}")
            shown += 1
            if shown >= max(1, int(top_k)):
                break
        return "\n".join(lines)
