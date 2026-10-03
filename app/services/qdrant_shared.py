"""Shared Qdrant client and embedding model singleton.

Both RAGService (``rag_service.py``) and the Mem0 long-term memory backend
(``memory.py``) need a Qdrant vector store.  Opening two embedded QdrantClients
against the same on-disk path takes an exclusive file lock, causing whichever
service initialises second to fail silently and fall back to ``:memory:``.
They also each load their own copy of ``all-MiniLM-L6-v2``, doubling RAM usage
and startup time.

This module provides:
* A single ``QdrantClient`` instance (embedded path or ``:memory:`` fallback).
* A single ``HuggingFaceEmbeddings`` instance (for HuggingFace embedding path).
* A ``vector_store_degraded`` flag that the ``/api/health`` endpoint can expose.

Both services import from here instead of constructing their own clients.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public state (read by the health endpoint)
# ---------------------------------------------------------------------------

#: Set to True when the embedded Qdrant path is locked and we fell back to RAM.
vector_store_degraded: bool = False
#: Human-readable reason when degraded; None when healthy.
vector_store_degraded_reason: Optional[str] = None

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_qdrant_client: Optional["QdrantClient"] = None  # type: ignore[name-defined]
_hf_embeddings = None


def get_qdrant_client():
    """Return the module-level QdrantClient, initialising it on first call.

    Uses the embedded path configured in ``settings.VECTOR_STORE_PATH``.
    Falls back to ``:memory:`` if the path is locked or unavailable, sets
    ``vector_store_degraded = True``, and logs a WARNING (not DEBUG) so the
    problem is visible in the startup log and the ``/api/health`` response.
    """
    global _qdrant_client, vector_store_degraded, vector_store_degraded_reason

    if _qdrant_client is not None:
        return _qdrant_client

    from qdrant_client import QdrantClient

    path = settings.VECTOR_STORE_PATH
    os.makedirs(path, exist_ok=True)
    try:
        _qdrant_client = QdrantClient(path=path)
        vector_store_degraded = False
        vector_store_degraded_reason = None
        logger.info("[Qdrant] Shared client initialised (path=%s).", path)
    except Exception as exc:
        vector_store_degraded = True
        vector_store_degraded_reason = (
            f"Embedded Qdrant path '{path}' locked or unavailable: {exc}. "
            "RAG retrieval will return no results until the lock is released."
        )
        logger.warning(
            "[Qdrant] DEGRADED — %s Falling back to :memory: client.",
            vector_store_degraded_reason,
        )
        print(f"[WARNING] {vector_store_degraded_reason}")
        _qdrant_client = QdrantClient(location=":memory:")

    return _qdrant_client


def get_hf_embeddings():
    """Return the shared HuggingFaceEmbeddings instance, loading on first call.

    Only used when ``settings.EMBEDDING_PROVIDER`` is ``"huggingface"`` or
    unrecognised.  The same object is passed to both RAGService and mem0 so
    the sentence-transformer model is loaded exactly once.
    """
    global _hf_embeddings

    if _hf_embeddings is not None:
        return _hf_embeddings

    from langchain_huggingface import HuggingFaceEmbeddings

    model_name = settings.EMBEDDING_MODEL or "all-MiniLM-L6-v2"
    logger.info("[Embeddings] Loading HuggingFace model '%s' (shared singleton).", model_name)
    _hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return _hf_embeddings
