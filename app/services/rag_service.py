"""RAG ingestion pipeline service.

Pipeline stages
---------------
1. Validation       — extension allow-list, file-size limit, basic sanity
2. Parsing          — format-aware document loading (PDF, TXT, MD)
3. Cleaning         — strip control characters, normalise whitespace
4. Structure detect — heuristic heading/section detection
5. Metadata extract — build per-chunk metadata envelope
6. Intelligent chunk — semantic/structural boundary-aware splitting
7. Embedding        — HuggingFace or OpenAI (lazy-loaded, unchanged)
8. Vector storage   — Qdrant upsert with full metadata payload

Deduplication
-------------
SHA-256 of raw file bytes.  If a matching hash is found the upload is
rejected unless ``force=True`` is passed, in which case the old vectors
are deleted from Qdrant and the document is re-ingested from scratch.

Document lifecycle
------------------
IngestedDocument.status:  "pending" → "processing" → "ingested" | "failed"
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, Filter, FieldCondition, MatchValue, 
    VectorParams, SparseVectorParams, SparseIndexParams, SparseVector
)
from sqlalchemy import UniqueConstraint
from sqlmodel import Field, Session, SQLModel, select
from flashrank import Ranker, RerankRequest

from app.config import settings
from app.services.qdrant_shared import get_qdrant_client, get_hf_embeddings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RetrievedChunk — structured retrieval result carrying full citation metadata
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """A single retrieved document chunk with full citation and scoring metadata.

    Returned by :meth:`RAGService.retrieve_structured`. Retains all provenance
    information needed to produce grounded, citable LLM responses.

    Score fields
    ------------
    dense_score:
        Raw cosine-similarity score returned by Qdrant for the dense vector
        query.  Always populated (range ~0.0–1.0).
    fusion_score:
        Reciprocal Rank Fusion (RRF) score combining dense + sparse rankings.
        Only meaningful when ``ENABLE_HYBRID_SEARCH=True``; ``None`` otherwise.
    rerank_score:
        FlashRank cross-encoder score.  Only set when ``ENABLE_RERANKING=True``
        and the reranker succeeded; ``None`` on fallback.
    """

    # Content
    text: str

    # Citation metadata (populated from Qdrant payload)
    document_id: str = ""
    filename: str = ""
    page_number: Optional[int] = None
    section: Optional[str] = None
    chunk_id: str = ""
    chunk_index: int = 0
    source: str = ""  # same as filename for documents

    # Scoring — always a raw cosine similarity from the dense index (~0.0–1.0)
    dense_score: float = 0.0
    # Fusion score from RRF over dense+sparse; None when hybrid search is disabled
    fusion_score: Optional[float] = None
    # FlashRank cross-encoder score; None when reranking is off or failed
    rerank_score: Optional[float] = None

    # Telemetry (set once on the list, carried per-chunk for convenience)
    retrieval_latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# format_cited_context — convert chunks to 【Doc N】-marked context + citation map
# ---------------------------------------------------------------------------


def format_cited_context(
    chunks: List[RetrievedChunk],
) -> Tuple[str, Dict[int, dict]]:
    """Format a list of RetrievedChunks into a grounded context string and citation map.

    Parameters
    ----------
    chunks:
        Ordered list of retrieved chunks (best first).

    Returns
    -------
    context_block : str
        Formatted text with 【Doc N】 markers for each chunk, ready to inject
        into the system prompt or tool message.
    citation_map : dict
        Maps 1-based integer index → citation metadata dict containing
        filename, page_number, section, chunk_id, document_id, dense_score,
        fusion_score (None when hybrid search is off), and rerank_score
        (None when reranking is off or failed).
    """
    if not chunks:
        return "", {}

    parts: List[str] = []
    citation_map: Dict[int, dict] = {}

    for i, chunk in enumerate(chunks, 1):
        # Build a compact source label for the header line
        source_parts = [chunk.filename or chunk.source or "Unknown"]
        if chunk.page_number:
            source_parts.append(f"p.{chunk.page_number}")
        if chunk.section:
            source_parts.append(f"\u00a7 {chunk.section}")
        source_label = " | ".join(source_parts)

        parts.append(f"\u3010Doc {i}\u3011 {source_label}\n{chunk.text}")

        citation_map[i] = {
            "filename": chunk.filename or chunk.source,
            "page_number": chunk.page_number,
            "section": chunk.section,
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            # dense_score is always a raw cosine similarity (~0.0-1.0)
            "dense_score": round(float(chunk.dense_score), 4),
            # fusion_score is only meaningful when hybrid search is enabled
            "fusion_score": (
                round(float(chunk.fusion_score), 4)
                if chunk.fusion_score is not None else None
            ),
            "rerank_score": (
                round(float(chunk.rerank_score), 4)
                if chunk.rerank_score is not None else None
            ),
        }

    context_block = "\n\n".join(parts)
    return context_block, citation_map


# ---------------------------------------------------------------------------
# DB Model
# ---------------------------------------------------------------------------


class IngestedDocument(SQLModel, table=True):
    """Tracks every file ingested into the RAG vector store.

    Used for deduplication, lifecycle management, and admin listing.

    Deduplication is scoped per-user: the composite unique constraint on
    (user_id, file_hash) allows two different users to ingest the same
    file content independently.  document_id is a stable UUID generated
    per ingestion and is unique globally (no two rows share a document_id).
    """

    __table_args__ = (
        # Dedup key: same file content uploaded by the same user counts as a
        # duplicate; byte-identical files from different users are independent.
        UniqueConstraint("user_id", "file_hash", name="uq_ingesteddocument_user_file_hash"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)

    # Stable external identifier (UUID4 hex, generated at ingest time).
    # Globally unique — no unique=True needed on the SA column because the
    # primary-key auto-increment already guarantees row uniqueness and
    # the UUID itself is random enough; an index is sufficient.
    document_id: str = Field(
        index=True,
        description="Stable UUID for this document",
    )

    # Uploader context
    user_id: str = Field(default="default_user", index=True, description="Uploader user ID")
    session_id: str = Field(default="default", index=True, description="Uploader session ID")

    # File identity
    filename: str = Field(index=True, description="Original filename")
    document_type: str = Field(default="txt", description="File extension / type (pdf, txt, md)")
    # Not unique by itself — the composite (user_id, file_hash) constraint
    # above handles dedup; different users may share the same hash.
    file_hash: str = Field(
        index=True,
        description="SHA-256 hash of raw file content",
    )

    # Stats
    chunk_count: int = Field(default=0, description="Number of vector chunks stored")

    # Lifecycle
    status: str = Field(
        default="pending",
        description="pending | processing | ingested | failed",
    )
    error_message: Optional[str] = Field(
        default=None, description="Error detail when status=failed"
    )

    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# RAGService
# ---------------------------------------------------------------------------


class RAGService:
    """Full 8-stage RAG ingestion pipeline with deduplication and lifecycle tracking."""

    def __init__(self):
        self._embeddings = None
        self._sparse_embeddings = None
        self._reranker = None
        self._vector_store = None
        self._client: Optional[QdrantClient] = None

    # ------------------------------------------------------------------
    # Lazy-loaded infrastructure properties
    # ------------------------------------------------------------------

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = self._get_embeddings()
        return self._embeddings

    @property
    def sparse_embeddings(self):
        if getattr(settings, "ENABLE_HYBRID_SEARCH", False) and self._sparse_embeddings is None:
            self._sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        return self._sparse_embeddings

    @property
    def reranker(self):
        if getattr(settings, "ENABLE_RERANKING", False) and self._reranker is None:
            model_name = getattr(settings, "RERANK_MODEL", "ms-marco-TinyBERT-L-2-v2")
            self._reranker = Ranker(model_name=model_name, cache_dir="./.flashrank_cache")
        return self._reranker

    @property
    def vector_store(self):
        if self._vector_store is None:
            self._init_vector_store()
        return self._vector_store

    # ------------------------------------------------------------------
    # Infrastructure setup
    # ------------------------------------------------------------------

    def _init_vector_store(self):
        # Use the shared client — only one QdrantClient opens the on-disk path
        # so there is no double file-lock with the mem0 Qdrant instance.
        client = get_qdrant_client()

        collection_name = "chatbot_documents"
        is_hybrid = getattr(settings, "ENABLE_HYBRID_SEARCH", False)
        try:
            if not client.collection_exists(collection_name=collection_name):
                sample_vec = self.embeddings.embed_query("init")
                create_params = {
                    "collection_name": collection_name,
                    "vectors_config": VectorParams(
                        size=len(sample_vec), distance=Distance.COSINE
                    ),
                }
                
                # Configure sparse vectors if hybrid search is enabled
                if is_hybrid:
                    create_params["sparse_vectors_config"] = {
                        "langchain-sparse": SparseVectorParams(
                            index=SparseIndexParams(on_disk=False)
                        )
                    }

                client.create_collection(**create_params)
        except Exception as exc:
            logger.warning("Error checking/creating Qdrant collection: %s", exc)

        self._client = client
        
        # Instantiate VectorStore with optional sparse_embedding
        vector_store_kwargs = {
            "client": client,
            "collection_name": collection_name,
            "embedding": self.embeddings,
        }
        
        if is_hybrid:
            vector_store_kwargs["sparse_embedding"] = self.sparse_embeddings
            vector_store_kwargs["retrieval_mode"] = RetrievalMode.HYBRID
            
        self._vector_store = QdrantVectorStore(**vector_store_kwargs)

    def _get_embeddings(self):
        provider = settings.EMBEDDING_PROVIDER.lower()
        if provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set for OpenAI embeddings.")
            return OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
            )
        else:
            # Both "huggingface" and any unrecognised provider use the shared
            # HuggingFaceEmbeddings singleton so the model is loaded only once.
            return get_hf_embeddings()

    # ------------------------------------------------------------------
    # Stage 1 — Validation
    # ------------------------------------------------------------------

    def _validate(self, file_path: str, filename: str) -> None:
        """Raise ValueError for invalid files (type, size).

        Checks:
        - Extension against ALLOWED_EXTENSIONS config list.
        - File size against MAX_UPLOAD_SIZE_MB config.
        """
        ext = os.path.splitext(filename)[1].lstrip(".").lower()
        allowed = [e.lower() for e in settings.ALLOWED_EXTENSIONS]
        if ext not in allowed:
            raise ValueError(
                f"Unsupported file type '.{ext}'. "
                f"Allowed: {', '.join('.' + e for e in allowed)}"
            )

        size_bytes = os.path.getsize(file_path)
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if size_bytes > max_bytes:
            raise ValueError(
                f"File '{filename}' is {size_bytes / (1024*1024):.1f} MB, "
                f"which exceeds the {settings.MAX_UPLOAD_SIZE_MB} MB limit."
            )

    # ------------------------------------------------------------------
    # Stage 2 — Parsing
    # ------------------------------------------------------------------

    def _parse(self, file_path: str, ext: str) -> List[Document]:
        """Load document bytes into LangChain Document objects.

        - PDF:  pdfplumber → extracts text and font size for layout-aware structure detection.
        - TXT:  TextLoader   → single Document.
        - MD:   TextLoader   → single Document (markdown treated as plain text).
        """
        if ext == "pdf":
            import pdfplumber
            docs = []
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    words = page.extract_words(extra_attrs=["size", "fontname"])
                    if not words:
                        continue
                    
                    # Sort words by vertical position, then horizontal
                    words.sort(key=lambda w: (round(w["top"], 1), w["x0"]))
                    
                    lines = []
                    current_line = []
                    current_top = None
                    
                    for w in words:
                        if current_top is None or abs(w["top"] - current_top) < 3: # 3 pt tolerance
                            current_line.append(w)
                            if current_top is None:
                                current_top = w["top"]
                        else:
                            lines.append(current_line)
                            current_line = [w]
                            current_top = w["top"]
                    if current_line:
                        lines.append(current_line)
                    
                    for line_words in lines:
                        text = " ".join(w["text"] for w in line_words)
                        max_size = max((w["size"] for w in line_words), default=0)
                        is_bold = any("bold" in str(w.get("fontname", "")).lower() for w in line_words)
                        
                        docs.append(Document(
                            page_content=text,
                            metadata={
                                "page_number": page_num,
                                "layout": {"size": max_size, "bold": is_bold}
                            }
                        ))
            return docs
        elif ext in ("txt", "md"):
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["page_number"] = 1
            return docs
        else:
            raise ValueError(f"No parser available for extension '.{ext}'")

    # ------------------------------------------------------------------
    # Stage 3 — Cleaning
    # ------------------------------------------------------------------

    # Regex: remove null bytes, carriage returns, and other control chars
    # (except tab \x09 and newline \x0a which are semantically useful)
    _CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")
    # Collapse runs of 3+ blank lines into exactly two newlines
    _EXCESS_BLANK = re.compile(r"\n{3,}")
    # Collapse horizontal whitespace runs (spaces/tabs) to a single space
    _MULTI_SPACE = re.compile(r"[ \t]{2,}")

    def _clean(self, documents: List[Document]) -> List[Document]:
        """Normalise document text in-place; returns the same list."""
        for doc in documents:
            text = doc.page_content
            # Unicode normalisation (NFC) — resolves composed vs decomposed chars
            text = unicodedata.normalize("NFC", text)
            # Strip invisible control characters
            text = self._CONTROL_CHARS.sub("", text)
            # Collapse excessive blank lines
            text = self._EXCESS_BLANK.sub("\n\n", text)
            # Collapse horizontal whitespace
            text = self._MULTI_SPACE.sub(" ", text)
            doc.page_content = text.strip()
        return documents

    # ------------------------------------------------------------------
    # Stage 4 — Structure Detection
    # ------------------------------------------------------------------

    # Patterns that indicate a heading/section title
    # Priority order: Markdown headers, ALL-CAPS lines, numbered headings
    _MD_HEADING = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
    _CAPS_LINE = re.compile(r"^([A-Z][A-Z\s\-:]{4,60})$", re.MULTILINE)
    _NUMBERED = re.compile(r"^(\d{1,2}(?:\.\d{1,2})*)\s+([A-Z].{3,60})$", re.MULTILINE)

    def _detect_structure(self, documents: List[Document]) -> List[Document]:
        """Annotate each Document with a 'section' metadata field.

        Heuristic priority:
        1. Layout metadata (font size/bold) for PDFs
        2. Markdown `## Heading` patterns
        3. ALL-CAPS short lines (common in Word/PDF exports)
        4. Numbered section headings (e.g. "1.2 Introduction")
        """
        if any("layout" in doc.metadata for doc in documents):
            sizes = [doc.metadata["layout"]["size"] for doc in documents if "layout" in doc.metadata]
            baseline_size = max(set(sizes), key=sizes.count) if sizes else 0
            
            current_section = None
            for doc in documents:
                layout = doc.metadata.get("layout")
                if layout:
                    size = layout.get("size", 0)
                    is_bold = layout.get("bold", False)
                    text = doc.page_content.strip()
                    if (size > baseline_size + 0.5 or (is_bold and size >= baseline_size)) and len(text) < 150:
                        current_section = text
                doc.metadata["section"] = current_section
            return documents
        else:
            new_docs = []
            for doc in documents:
                text = doc.page_content
                headings = []
                for match in self._MD_HEADING.finditer(text):
                    headings.append((match.start(), match.group(1).strip()))
                if not headings:
                    for match in self._NUMBERED.finditer(text):
                        headings.append((match.start(), f"{match.group(1)} {match.group(2)}".strip()))
                if not headings:
                    for match in self._CAPS_LINE.finditer(text):
                        if len(match.group(1).strip()) > 4:
                            headings.append((match.start(), match.group(1).strip()))
                
                if not headings:
                    doc.metadata["section"] = None
                    new_docs.append(doc)
                    continue
                
                headings.sort(key=lambda x: x[0])
                
                last_idx = 0
                current_section = None
                
                for start_idx, title in headings:
                    if start_idx > last_idx:
                        chunk_text = text[last_idx:start_idx].strip()
                        if chunk_text:
                            new_doc = Document(page_content=chunk_text, metadata=doc.metadata.copy())
                            new_doc.metadata["section"] = current_section
                            new_docs.append(new_doc)
                    current_section = title
                    last_idx = start_idx
                
                if last_idx < len(text):
                    chunk_text = text[last_idx:].strip()
                    if chunk_text:
                        new_doc = Document(page_content=chunk_text, metadata=doc.metadata.copy())
                        new_doc.metadata["section"] = current_section
                        new_docs.append(new_doc)
            return new_docs

    # ------------------------------------------------------------------
    # Stage 5 — Metadata Extraction
    # ------------------------------------------------------------------

    def _extract_metadata(
        self,
        documents: List[Document],
        *,
        document_id: str,
        user_id: str,
        session_id: str,
        filename: str,
        document_type: str,
        file_hash: str,
        created_at: str,
    ) -> List[Document]:
        """Merge the per-document envelope into each Document's metadata.

        After this stage every Document carries:
            document_id, user_id, session_id, filename, document_type,
            file_hash, created_at, page_number, section, source
        """
        envelope = {
            "document_id": document_id,
            "user_id": user_id,
            "session_id": session_id,
            "filename": filename,
            "document_type": document_type,
            "file_hash": file_hash,
            "created_at": created_at,
            "source": filename,
        }
        for doc in documents:
            doc.metadata.update(envelope)
            # Ensure page_number is always present (default 1 for non-PDF)
            doc.metadata.setdefault("page_number", 1)
            doc.metadata.setdefault("section", None)
        return documents

    # ------------------------------------------------------------------
    # Stage 6 — Intelligent Chunking
    # ------------------------------------------------------------------

    @staticmethod
    def _make_chunk_id(document_id: str, chunk_index: int) -> str:
        """Deterministic 12-char chunk fingerprint derived from doc ID + position."""
        raw = f"{document_id}:{chunk_index}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    @staticmethod
    def _merge_lines(line_docs: List[Document]) -> str:
        """Join per-line Documents into clean, embedding-friendly prose.

        Steps applied to the raw ``"\n".join(...)`` text:

        1. **De-hyphenation** — remove line-end soft hyphens so that
           ``"infor-\nmation"`` becomes ``"information"``.
        2. **Paragraph-aware join** — blank lines in the original PDF layout
           produce a ``"\n\n"`` paragraph break; all other single ``"\n"``
           separators (normal line-wraps within the same paragraph) are
           replaced by a single space so the text reads as continuous prose.
        """
        raw = "\n".join(d.page_content for d in line_docs)
        # Step 1 — remove hyphenation at line endings (e.g. "infor-\nmation")
        raw = re.sub(r"-\n(?=\w)", "", raw)
        # Step 2 — collapse single \n (soft line wrap) → space;
        #           double \n (paragraph boundary) is left intact.
        raw = re.sub(r"(?<!\n)\n(?!\n)", " ", raw)
        return raw

    def _chunk(self, documents: List[Document]) -> List[Document]:
        """Split documents into semantically-bounded chunks.

        Separator priority (coarsest → finest):
            triple-newline → double-newline (paragraph) → single-newline (line)
            → sentence-ending period+space → word boundary → character

        This preserves paragraph and sentence context before resorting to
        word or character splits, avoiding arbitrary mid-sentence breaks.

        Each chunk gets:
            chunk_id      — deterministic fingerprint (document_id + index)
            chunk_index   — 0-based position within document
            total_chunks  — filled after splitting (requires a second pass)
        """
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
            chunk_size=settings.RAG_CHUNK_SIZE,
            chunk_overlap=settings.RAG_CHUNK_OVERLAP,
            add_start_index=True,
        )
        
        # Group contiguous documents by section and page_number
        grouped_docs = []
        if documents:
            current_group = [documents[0]]
            
            for doc in documents[1:]:
                prev_doc = current_group[-1]
                same_section = doc.metadata.get("section") == prev_doc.metadata.get("section")
                same_page = doc.metadata.get("page_number") == prev_doc.metadata.get("page_number")
                same_doc = doc.metadata.get("document_id") == prev_doc.metadata.get("document_id")
                
                if same_section and same_page and same_doc:
                    current_group.append(doc)
                else:
                    merged_text = self._merge_lines(current_group)
                    merged_metadata = current_group[0].metadata.copy()
                    merged_metadata.pop("layout", None)
                    grouped_docs.append(Document(page_content=merged_text, metadata=merged_metadata))
                    current_group = [doc]
            
            if current_group:
                merged_text = self._merge_lines(current_group)
                merged_metadata = current_group[0].metadata.copy()
                merged_metadata.pop("layout", None)
                grouped_docs.append(Document(page_content=merged_text, metadata=merged_metadata))
        
        chunks = []
        for g_doc in grouped_docs:
            g_chunks = splitter.split_documents([g_doc])
            chunks.extend(g_chunks)
            
        total = len(chunks)

        # Group chunks by document_id to assign per-document chunk_index
        # (important when multiple pages produce separate Document objects)
        doc_chunk_counters: dict[str, int] = {}
        for chunk in chunks:
            doc_id = chunk.metadata.get("document_id", "unknown")
            idx = doc_chunk_counters.get(doc_id, 0)
            chunk.metadata["chunk_id"] = self._make_chunk_id(doc_id, idx)
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["total_chunks"] = total
            

            
            doc_chunk_counters[doc_id] = idx + 1

        return chunks

    # ------------------------------------------------------------------
    # Deduplication helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_file_hash(file_path: str) -> str:
        """Compute SHA-256 of the file's raw bytes."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as fh:
            for block in iter(lambda: fh.read(65536), b""):
                sha256.update(block)
        return sha256.hexdigest()

    def _find_existing(self, file_hash: str, user_id: str) -> Optional[IngestedDocument]:
        """Return the IngestedDocument matching (user_id, file_hash), or None.

        Deduplication is intentionally scoped to the uploading user: two
        different users uploading byte-identical files receive independent
        ingestion records and their own Qdrant vectors tagged with their
        user_id, so per-user RAG queries return results for both.
        """
        from app.services.database import db_service

        try:
            with Session(db_service.engine) as db:
                return db.exec(
                    select(IngestedDocument).where(
                        IngestedDocument.file_hash == file_hash,
                        IngestedDocument.user_id == user_id,
                    )
                ).first()
        except Exception:
            return None

    def _upsert_db_record(
        self,
        *,
        document_id: str,
        filename: str,
        file_hash: str,
        user_id: str,
        session_id: str,
        document_type: str,
        chunk_count: int,
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        """Insert or update the IngestedDocument DB row."""
        from app.services.database import db_service

        try:
            with Session(db_service.engine) as db:
                existing = db.exec(
                    select(IngestedDocument).where(
                        IngestedDocument.document_id == document_id
                    )
                ).first()
                if existing:
                    existing.status = status
                    existing.chunk_count = chunk_count
                    existing.error_message = error_message
                    db.add(existing)
                else:
                    db.add(
                        IngestedDocument(
                            document_id=document_id,
                            filename=filename,
                            file_hash=file_hash,
                            user_id=user_id,
                            session_id=session_id,
                            document_type=document_type,
                            chunk_count=chunk_count,
                            status=status,
                            error_message=error_message,
                        )
                    )
                db.commit()
        except Exception as exc:
            logger.warning("Failed to upsert ingestion record: %s", exc)

    def _delete_vectors_for_document(self, document_id: str) -> int:
        """Delete all Qdrant points for a given document_id.

        Returns the number of points deleted (0 if none found or on error).
        """
        if self._client is None:
            # Force vector store initialisation so _client is populated
            _ = self.vector_store

        try:
            collection_name = "chatbot_documents"
            # Count before deletion so we can report how many were removed
            count_result = self._client.count(
                collection_name=collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                ),
                exact=True,
            )
            deleted_count = count_result.count if count_result else 0

            self._client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                ),
            )
            return deleted_count
        except Exception as exc:
            logger.warning("Failed to delete Qdrant vectors for document %s: %s", document_id, exc)
            return 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_file(
        self,
        file_path: str,
        metadata: Optional[dict] = None,
        force: bool = False,
    ) -> dict:
        """Ingest a file through the full 8-stage pipeline.

        Parameters
        ----------
        file_path : str
            Path to the temporary file on disk.
        metadata : dict, optional
            Caller-supplied context: ``filename``, ``user_id``, ``session_id``.
        force : bool
            If True and a duplicate is detected, delete the old vectors and
            re-ingest. If False (default) the duplicate is rejected immediately.

        Returns
        -------
        dict with keys:
            status          : "ingested" | "duplicate" | "failed"
            document_id     : stable UUID for this document
            chunks_added    : int
            filename        : original filename
            document_type   : file extension
            user_id         : uploader user ID
            session_id      : uploader session ID
            duplicate_of    : filename of prior ingestion (duplicate only)
            originally_ingested_at : ISO timestamp (duplicate only)
            error           : error message (failed only)
        """
        meta = metadata or {}
        original_filename = meta.get("filename", os.path.basename(file_path))
        user_id = meta.get("user_id", "default_user")
        session_id = meta.get("session_id", "default")
        ext = os.path.splitext(original_filename)[1].lstrip(".").lower()

        # ── Stage 1: Validation ──────────────────────────────────────────
        try:
            self._validate(file_path, original_filename)
        except ValueError as exc:
            return {
                "status": "failed",
                "document_id": None,
                "chunks_added": 0,
                "filename": original_filename,
                "document_type": ext,
                "user_id": user_id,
                "session_id": session_id,
                "error": str(exc),
            }

        # ── Deduplication check ──────────────────────────────────────────
        file_hash = self._compute_file_hash(file_path)
        existing = self._find_existing(file_hash, user_id)

        if existing and not force:
            return {
                "status": "duplicate",
                "document_id": existing.document_id,
                "chunks_added": 0,
                "filename": original_filename,
                "document_type": ext,
                "user_id": user_id,
                "session_id": session_id,
                "duplicate_of": existing.filename,
                "originally_ingested_at": existing.ingested_at.isoformat(),
            }

        # If force=True, delete old vectors and DB record so we can re-ingest
        if existing and force:
            logger.info(
                "Force re-ingestion requested for '%s' (document_id=%s). "
                "Deleting %d old vectors.",
                original_filename,
                existing.document_id,
                existing.chunk_count,
            )
            self._delete_vectors_for_document(existing.document_id)
            from app.services.database import db_service
            try:
                with Session(db_service.engine) as db:
                    record = db.get(IngestedDocument, existing.id)
                    if record:
                        db.delete(record)
                        db.commit()
            except Exception as exc:
                logger.warning("Could not remove old DB record for force re-ingest: %s", exc)

        # Generate a fresh document_id for this ingestion
        document_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()

        # Write a "processing" record so failures are visible
        self._upsert_db_record(
            document_id=document_id,
            filename=original_filename,
            file_hash=file_hash,
            user_id=user_id,
            session_id=session_id,
            document_type=ext,
            chunk_count=0,
            status="processing",
        )

        try:
            # ── Stage 2: Parsing ─────────────────────────────────────────
            documents = self._parse(file_path, ext)

            # ── Stage 3: Cleaning ────────────────────────────────────────
            documents = self._clean(documents)

            # Filter out any pages/blocks that are empty after cleaning
            documents = [d for d in documents if d.page_content.strip()]

            if not documents:
                raise ValueError("No extractable text content found in file.")

            # ── Stage 4: Structure Detection ─────────────────────────────
            documents = self._detect_structure(documents)

            # ── Stage 5: Metadata Extraction ─────────────────────────────
            documents = self._extract_metadata(
                documents,
                document_id=document_id,
                user_id=user_id,
                session_id=session_id,
                filename=original_filename,
                document_type=ext,
                file_hash=file_hash,
                created_at=created_at,
            )

            # ── Stage 6: Intelligent Chunking ────────────────────────────
            chunks = self._chunk(documents)

            # ── Stages 7 & 8: Embedding + Vector Storage ─────────────────
            if chunks:
                self.vector_store.add_documents(chunks)

        except Exception as exc:
            error_msg = str(exc)
            logger.error("Ingestion failed for '%s': %s", original_filename, error_msg)
            self._upsert_db_record(
                document_id=document_id,
                filename=original_filename,
                file_hash=file_hash,
                user_id=user_id,
                session_id=session_id,
                document_type=ext,
                chunk_count=0,
                status="failed",
                error_message=error_msg,
            )
            return {
                "status": "failed",
                "document_id": document_id,
                "chunks_added": 0,
                "filename": original_filename,
                "document_type": ext,
                "user_id": user_id,
                "session_id": session_id,
                "error": error_msg,
            }

        # Persist the final "ingested" record
        self._upsert_db_record(
            document_id=document_id,
            filename=original_filename,
            file_hash=file_hash,
            user_id=user_id,
            session_id=session_id,
            document_type=ext,
            chunk_count=len(chunks),
            status="ingested",
        )

        logger.info(
            "Ingested '%s' → document_id=%s, %d chunks, user=%s, session=%s",
            original_filename,
            document_id,
            len(chunks),
            user_id,
            session_id,
        )

        return {
            "status": "ingested",
            "document_id": document_id,
            "chunks_added": len(chunks),
            "filename": original_filename,
            "document_type": ext,
            "user_id": user_id,
            "session_id": session_id,
        }

    # ------------------------------------------------------------------
    # Split-phase helpers for the async ingest endpoint
    # ------------------------------------------------------------------

    def preflight_check(
        self,
        file_path: str,
        metadata: Optional[dict] = None,
        force: bool = False,
    ) -> dict:
        """Run fast pre-flight checks and write the 'processing' DB row.

        Covers Stage 1 (validation) and the deduplication hash check.
        Heavy work (parsing, embedding) is NOT performed here.

        Returns
        -------
        dict with ``status`` one of:

        ``"ready"``
            Checks passed.  DB row written with ``status="processing"``.
            Caller should invoke :meth:`run_pipeline` in a background task.
            Extra keys: ``document_id``, ``file_hash``, ``filename``,
            ``document_type``, ``user_id``, ``session_id``.

        ``"duplicate"``
            File already ingested and ``force=False``.
            Keys mirror those returned by :meth:`ingest_file`.

        ``"failed"``
            Validation error.  ``error`` key contains the message.
        """
        meta = metadata or {}
        original_filename = meta.get("filename", os.path.basename(file_path))
        user_id = meta.get("user_id", "default_user")
        session_id = meta.get("session_id", "default")
        ext = os.path.splitext(original_filename)[1].lstrip(".").lower()

        # Stage 1 — Validation
        try:
            self._validate(file_path, original_filename)
        except ValueError as exc:
            return {
                "status": "failed",
                "document_id": None,
                "chunks_added": 0,
                "filename": original_filename,
                "document_type": ext,
                "user_id": user_id,
                "session_id": session_id,
                "error": str(exc),
            }

        # Deduplication
        file_hash = self._compute_file_hash(file_path)
        existing = self._find_existing(file_hash, user_id)

        if existing and not force:
            return {
                "status": "duplicate",
                "document_id": existing.document_id,
                "chunks_added": 0,
                "filename": original_filename,
                "document_type": ext,
                "user_id": user_id,
                "session_id": session_id,
                "duplicate_of": existing.filename,
                "originally_ingested_at": existing.ingested_at.isoformat(),
            }

        if existing and force:
            logger.info(
                "Force re-ingestion requested for '%s' (document_id=%s). "
                "Deleting %d old vectors.",
                original_filename,
                existing.document_id,
                existing.chunk_count,
            )
            self._delete_vectors_for_document(existing.document_id)
            from app.services.database import db_service
            try:
                with Session(db_service.engine) as db:
                    record = db.get(IngestedDocument, existing.id)
                    if record:
                        db.delete(record)
                        db.commit()
            except Exception as exc:
                logger.warning("Could not remove old DB record for force re-ingest: %s", exc)

        # Allocate stable document_id and write a "processing" placeholder
        document_id = uuid.uuid4().hex
        self._upsert_db_record(
            document_id=document_id,
            filename=original_filename,
            file_hash=file_hash,
            user_id=user_id,
            session_id=session_id,
            document_type=ext,
            chunk_count=0,
            status="processing",
        )

        return {
            "status": "ready",
            "document_id": document_id,
            "file_hash": file_hash,
            "filename": original_filename,
            "document_type": ext,
            "user_id": user_id,
            "session_id": session_id,
        }

    def run_pipeline(
        self,
        file_path: str,
        metadata: Optional[dict] = None,
        document_id: Optional[str] = None,
        file_hash: Optional[str] = None,
    ) -> dict:
        """Run the heavy pipeline stages (2–8) for a pre-flighted document.

        Intended to be called from a background thread/task after
        :meth:`preflight_check` has returned ``status="ready"``.
        Updates the DB row to ``"ingested"`` on success or ``"failed"`` on
        any exception, so :meth:`get_document` reflects the final state.

        Parameters
        ----------
        file_path:
            Path to the (still-present) temp file on disk.
        metadata:
            Same dict passed to :meth:`preflight_check`.
        document_id:
            The UUID allocated by :meth:`preflight_check`.
        file_hash:
            The SHA-256 hash computed by :meth:`preflight_check` (avoids
            re-hashing the file a second time).

        Returns
        -------
        dict with keys matching those of :meth:`ingest_file`
        (``status="ingested"`` or ``status="failed"``).
        """
        meta = metadata or {}
        original_filename = meta.get("filename", os.path.basename(file_path))
        user_id = meta.get("user_id", "default_user")
        session_id = meta.get("session_id", "default")
        ext = os.path.splitext(original_filename)[1].lstrip(".").lower()
        created_at = datetime.now(timezone.utc).isoformat()

        try:
            # Stage 2 — Parsing
            documents = self._parse(file_path, ext)

            # Stage 3 — Cleaning
            documents = self._clean(documents)
            documents = [d for d in documents if d.page_content.strip()]
            if not documents:
                raise ValueError("No extractable text content found in file.")

            # Stage 4 — Structure Detection
            documents = self._detect_structure(documents)

            # Stage 5 — Metadata Extraction
            documents = self._extract_metadata(
                documents,
                document_id=document_id,
                user_id=user_id,
                session_id=session_id,
                filename=original_filename,
                document_type=ext,
                file_hash=file_hash,
                created_at=created_at,
            )

            # Stage 6 — Intelligent Chunking
            chunks = self._chunk(documents)

            # Stages 7 & 8 — Embedding + Vector Storage
            if chunks:
                self.vector_store.add_documents(chunks)

        except Exception as exc:
            error_msg = str(exc)
            logger.error("Background ingestion failed for '%s': %s", original_filename, error_msg)
            self._upsert_db_record(
                document_id=document_id,
                filename=original_filename,
                file_hash=file_hash,
                user_id=user_id,
                session_id=session_id,
                document_type=ext,
                chunk_count=0,
                status="failed",
                error_message=error_msg,
            )
            return {
                "status": "failed",
                "document_id": document_id,
                "chunks_added": 0,
                "filename": original_filename,
                "document_type": ext,
                "user_id": user_id,
                "session_id": session_id,
                "error": error_msg,
            }

        self._upsert_db_record(
            document_id=document_id,
            filename=original_filename,
            file_hash=file_hash,
            user_id=user_id,
            session_id=session_id,
            document_type=ext,
            chunk_count=len(chunks),
            status="ingested",
        )

        logger.info(
            "Background ingestion complete: '%s' -> document_id=%s, %d chunks",
            original_filename,
            document_id,
            len(chunks),
        )

        return {
            "status": "ingested",
            "document_id": document_id,
            "chunks_added": len(chunks),
            "filename": original_filename,
            "document_type": ext,
            "user_id": user_id,
            "session_id": session_id,
        }

    def retrieve_structured(
        self,
        query: str,
        top_k: Optional[int] = None,
        candidate_k: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Retrieve relevant chunks with full citation metadata and telemetry.

        This is the **structured** counterpart to :meth:`retrieve`. It returns
        :class:`RetrievedChunk` objects instead of a plain string, preserving
        all provenance data needed for source-grounded citation.

        Pipeline
        --------
        1. Fetch ``candidate_k`` dense + sparse hits (20–50 recommended).
        2. Fuse via Reciprocal Rank Fusion (RRF).
        3. Rerank with FlashRank cross-encoder → select ``top_k`` best.
           Falls back gracefully to RRF order if reranker is unavailable.
        4. Log retrieval latency, initial candidate count, final chunk count,
           and relevance scores.

        Parameters
        ----------
        query:
            The user's (possibly rewritten) search query.
        top_k:
            Final number of chunks to return. Defaults to ``settings.RERANK_TOP_K``.
        candidate_k:
            Number of candidates to fetch before reranking.
            Defaults to ``settings.RERANK_CANDIDATE_K``.
        user_id:
            Optional filter — restrict retrieval to this user's documents.

        Returns
        -------
        List[RetrievedChunk]
            Ordered best-first list (at most ``top_k`` items).
            Chunks scoring below ``MIN_RERANK_SCORE`` (when reranking is on) or
            ``MIN_RRF_SCORE`` (when reranking is off) are excluded.
            Returns an empty list when no chunk clears the minimum threshold.
        """
        t0 = time.perf_counter()

        # Resolve configurable defaults
        _top_k = top_k if top_k is not None else getattr(settings, "RERANK_TOP_K", 5)
        _candidate_k = candidate_k if candidate_k is not None else getattr(settings, "RERANK_CANDIDATE_K", 20)
        # Always fetch at least top_k; reranking needs the extra candidates to be useful
        _candidate_k = max(_candidate_k, _top_k)

        is_hybrid = getattr(settings, "ENABLE_HYBRID_SEARCH", False)
        is_reranking = getattr(settings, "ENABLE_RERANKING", False)
        dense_weight = getattr(settings, "HYBRID_DENSE_WEIGHT", 0.5)
        sparse_weight = getattr(settings, "HYBRID_SPARSE_WEIGHT", 0.5)

        # Ensure vector store is initialised
        if self._client is None:
            _ = self.vector_store

        # ── 1. Build query filter ────────────────────────────────────────────
        query_filter = None
        if user_id:
            query_filter = Filter(
                must=[FieldCondition(
                    key="metadata.user_id", match=MatchValue(value=user_id)
                )]
            )

        # ── 2. Execute dense (and optionally sparse) search ──────────────────
        dense_hits = []
        sparse_hits = []

        try:
            dense_vec = self.embeddings.embed_query(query)
            dense_hits = self._client.query_points(
                collection_name="chatbot_documents",
                query=dense_vec,
                using="",
                query_filter=query_filter,
                limit=_candidate_k,
                with_payload=True,
            ).points
        except Exception as exc:
            logger.warning("[retrieve_structured] Dense search failed: %s", exc)

        if is_hybrid and self.sparse_embeddings:
            try:
                sparse_vec_obj = self.sparse_embeddings.embed_query(query)
                sparse_hits = self._client.query_points(
                    collection_name="chatbot_documents",
                    query=SparseVector(
                        indices=sparse_vec_obj.indices,
                        values=sparse_vec_obj.values,
                    ),
                    using="langchain-sparse",
                    query_filter=query_filter,
                    limit=_candidate_k,
                    with_payload=True,
                ).points
            except Exception as exc:
                logger.warning("[retrieve_structured] Sparse search failed: %s", exc)

        # ── 3. RRF fusion ────────────────────────────────────────────────────
        # Build a lookup of raw dense scores so we can always populate dense_score
        # on RetrievedChunk regardless of hybrid mode.
        dense_score_by_id: Dict[str, float] = {
            str(hit.id): hit.score for hit in dense_hits
        }

        fused: dict = {}

        def _apply_rrf(hits, weight: float) -> None:
            for rank, hit in enumerate(hits):
                pid = str(hit.id)
                score = weight * (1.0 / (rank + 60))
                if pid not in fused:
                    fused[pid] = {
                        "payload": hit.payload,
                        "fusion_score": 0.0,
                        "dense_score": dense_score_by_id.get(pid, 0.0),
                        "point_id": pid,
                    }
                fused[pid]["fusion_score"] += score

        if is_hybrid:
            _apply_rrf(dense_hits, dense_weight)
            _apply_rrf(sparse_hits, sparse_weight)
        else:
            # Non-hybrid: no RRF — store raw dense cosine similarity directly.
            # fusion_score is left as None to signal that hybrid was not active.
            for hit in dense_hits:
                fused[str(hit.id)] = {
                    "payload": hit.payload,
                    "fusion_score": None,
                    "dense_score": hit.score,
                    "point_id": str(hit.id),
                }

        # Sort by fusion_score in hybrid mode, dense_score otherwise
        sorted_candidates = sorted(
            fused.values(),
            key=lambda x: x["fusion_score"] if x["fusion_score"] is not None else x["dense_score"],
            reverse=True,
        )
        initial_candidates = len(sorted_candidates)

        # ── 4. Reranking stage ───────────────────────────────────────────────
        reranker_used = False
        final_items = sorted_candidates[:_top_k]  # fallback default
        rerank_scores: Dict[str, float] = {}

        if is_reranking and self.reranker and sorted_candidates:
            passages = []
            for item in sorted_candidates:
                payload = item["payload"]
                passages.append({
                    "id": item["point_id"],
                    "text": payload.get("page_content", ""),
                    "meta": payload.get("metadata", {}),
                })
            try:
                rerank_req = RerankRequest(query=query, passages=passages)
                reranked = self.reranker.rerank(rerank_req)
                # Build a score lookup by point_id
                for r in reranked:
                    rerank_scores[r["id"]] = r["score"]
                # Re-sort original candidates by rerank score, take top_k
                scored_candidates = [
                    c for c in sorted_candidates if c["point_id"] in rerank_scores
                ]
                scored_candidates.sort(
                    key=lambda c: rerank_scores.get(c["point_id"], 0.0), reverse=True
                )
                final_items = scored_candidates[:_top_k]
                reranker_used = True
            except Exception as exc:
                logger.warning(
                    "[retrieve_structured] Reranker failed (%s). "
                    "Falling back to RRF ordering.", exc
                )
                final_items = sorted_candidates[:_top_k]

        # ── 4b. Minimum-relevance cutoff ─────────────────────────────────────
        # Each branch compares against the score that was used for ranking so
        # the threshold is always in the same numeric range as the setting.
        if reranker_used:
            min_score = getattr(settings, "MIN_RERANK_SCORE", 0.1)
            pre_filter = len(final_items)
            final_items = [
                item for item in final_items
                if rerank_scores.get(item["point_id"], 0.0) >= min_score
            ]
            if len(final_items) < pre_filter:
                logger.info(
                    "[retrieve_structured] Score cutoff (MIN_RERANK_SCORE=%.3f) dropped "
                    "%d/%d chunks.",
                    min_score, pre_filter - len(final_items), pre_filter,
                )
        elif is_hybrid:
            # Hybrid without reranking: compare fusion (RRF) score
            min_score = getattr(settings, "MIN_RRF_SCORE", 0.005)
            pre_filter = len(final_items)
            final_items = [
                item for item in final_items
                if (item["fusion_score"] or 0.0) >= min_score
            ]
            if len(final_items) < pre_filter:
                logger.info(
                    "[retrieve_structured] Score cutoff (MIN_RRF_SCORE=%.4f) dropped "
                    "%d/%d chunks.",
                    min_score, pre_filter - len(final_items), pre_filter,
                )
        else:
            # Dense-only without reranking: compare raw cosine similarity
            min_score = getattr(settings, "MIN_DENSE_SCORE", 0.3)
            pre_filter = len(final_items)
            final_items = [
                item for item in final_items
                if item["dense_score"] >= min_score
            ]
            if len(final_items) < pre_filter:
                logger.info(
                    "[retrieve_structured] Score cutoff (MIN_DENSE_SCORE=%.3f) dropped "
                    "%d/%d chunks.",
                    min_score, pre_filter - len(final_items), pre_filter,
                )

        # ── 5. Build RetrievedChunk objects ──────────────────────────────────
        latency_ms = (time.perf_counter() - t0) * 1000.0
        chunks: List[RetrievedChunk] = []

        for item in final_items:
            payload = item["payload"]
            meta = payload.get("metadata", {})
            pid = item.get("point_id", "")
            rs = rerank_scores.get(pid) if reranker_used else None
            chunk = RetrievedChunk(
                text=payload.get("page_content", ""),
                document_id=meta.get("document_id", ""),
                filename=meta.get("filename") or meta.get("source", ""),
                page_number=meta.get("page_number"),
                section=meta.get("section"),
                chunk_id=meta.get("chunk_id", ""),
                chunk_index=meta.get("chunk_index", 0),
                source=meta.get("source", meta.get("filename", "")),
                dense_score=item["dense_score"],
                fusion_score=item["fusion_score"],  # None when hybrid search is off
                rerank_score=rs,
                retrieval_latency_ms=latency_ms,
            )
            chunks.append(chunk)

        # ── 6. Telemetry log ─────────────────────────────────────────────────
        def _primary_score(c: RetrievedChunk) -> float:
            """Return the most informative score for logging purposes."""
            if c.rerank_score is not None:
                return round(c.rerank_score, 3)
            if c.fusion_score is not None:
                return round(c.fusion_score, 4)
            return round(c.dense_score, 4)

        final_scores = [_primary_score(c) for c in chunks]
        logger.info(
            "[Retrieval Diagnostics] latency=%.1fms | candidate_k=%d | "
            "initial_candidates=%d | final_chunks=%d | reranker=%s | scores=%s",
            latency_ms,
            _candidate_k,
            initial_candidates,
            len(chunks),
            "flashrank" if reranker_used else "rrf_fallback",
            final_scores,
        )

        return chunks

    def delete_document(self, document_id: str) -> dict:
        """Delete a document's vectors from Qdrant and its DB record.

        Returns
        -------
        dict with keys: status, document_id, vectors_deleted
        """
        from app.services.database import db_service

        vectors_deleted = self._delete_vectors_for_document(document_id)

        db_deleted = False
        try:
            with Session(db_service.engine) as db:
                record = db.exec(
                    select(IngestedDocument).where(
                        IngestedDocument.document_id == document_id
                    )
                ).first()
                if record:
                    db.delete(record)
                    db.commit()
                    db_deleted = True
        except Exception as exc:
            logger.warning("Failed to delete DB record for document %s: %s", document_id, exc)

        if not db_deleted and vectors_deleted == 0:
            return {
                "status": "not_found",
                "document_id": document_id,
                "vectors_deleted": 0,
            }

        return {
            "status": "deleted",
            "document_id": document_id,
            "vectors_deleted": vectors_deleted,
        }

    def get_document(self, document_id: str) -> Optional[dict]:
        """Return metadata for a single document by document_id, or None."""
        from app.services.database import db_service

        try:
            with Session(db_service.engine) as db:
                record = db.exec(
                    select(IngestedDocument).where(
                        IngestedDocument.document_id == document_id
                    )
                ).first()
                if not record:
                    return None
                return self._doc_to_dict(record)
        except Exception as exc:
            logger.warning("Failed to fetch document %s: %s", document_id, exc)
            return None

    def list_documents(self) -> List[dict]:
        """Return metadata for all ingested documents, newest first."""
        from app.services.database import db_service

        try:
            with Session(db_service.engine) as db:
                records = db.exec(
                    select(IngestedDocument).order_by(
                        IngestedDocument.ingested_at.desc()
                    )
                ).all()
                return [self._doc_to_dict(r) for r in records]
        except Exception as exc:
            logger.warning("Failed to list ingested documents: %s", exc)
            return []

    def has_documents(self, user_id: Optional[str] = None) -> bool:
        """Return True when at least one successfully ingested document exists.

        Used as a fast pre-check (single indexed COUNT) before the LLM call so
        the ``retrieve_documents`` tool is only offered when there is actually
        something in the vector store to retrieve.

        Parameters
        ----------
        user_id:
            When supplied, the count is scoped to that uploader.  When None or
            empty the check is global (useful for admin or shared corpora).
        """
        from app.services.database import db_service
        from sqlalchemy import func

        try:
            with Session(db_service.engine) as db:
                stmt = select(func.count()).select_from(IngestedDocument).where(
                    IngestedDocument.status == "ingested"
                )
                if user_id:
                    stmt = stmt.where(IngestedDocument.user_id == user_id)
                count = db.exec(stmt).one()
                return count > 0
        except Exception as exc:
            # On any DB error fall back to True so retrieval is never wrongly blocked.
            logger.warning("[has_documents] COUNT query failed — assuming True: %s", exc)
            return True

    @staticmethod
    def _doc_to_dict(r: IngestedDocument) -> dict:
        return {
            "id": r.id,
            "document_id": r.document_id,
            "filename": r.filename,
            "document_type": r.document_type,
            "file_hash": r.file_hash[:16] + "…",
            "chunk_count": r.chunk_count,
            "user_id": r.user_id,
            "session_id": r.session_id,
            "status": r.status,
            "error_message": r.error_message,
            "ingested_at": r.ingested_at.isoformat(),
        }


# Singleton
rag_service = RAGService()
