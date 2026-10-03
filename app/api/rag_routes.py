"""RAG API routes — document ingestion, retrieval admin, and document management."""

import asyncio
import os
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from app.config import settings
from app.services.rag_service import rag_service

router = APIRouter(prefix="/rag", tags=["RAG"])

UPLOAD_DIR = "./temp_uploads"


# ---------------------------------------------------------------------------
# POST /rag/ingest
# ---------------------------------------------------------------------------


@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    user_id: str = Form(default="default_user"),
    session_id: str = Form(default="default"),
    force: bool = Query(
        default=False,
        description=(
            "If true, delete the existing vectors for a duplicate file and "
            "re-ingest from scratch. Has no effect on genuinely new files."
        ),
    ),
):
    """Upload a document (.txt, .pdf, or .md) to the RAG vector store.

    Returns **202 Accepted** immediately once pre-flight checks pass and the
    slow pipeline has been scheduled as a background task.

    The caller should poll ``GET /rag/documents/{document_id}`` every 1-2 s
    until ``status`` transitions to ``"ingested"`` or ``"failed"``.

    Pre-flight (synchronous, fast)
    --------------------------------
    * Extension allow-list + file-size validation
    * SHA-256 deduplication check

    Background pipeline (asynchronous)
    ------------------------------------
    parsing -> cleaning -> structure detection ->
    metadata extraction -> intelligent chunking -> embedding -> vector storage

    Deduplication
    -------------
    Uploading a byte-identical file a second time returns ``status="duplicate"``
    immediately (no 202) -- unless ``force=true`` is passed, which deletes the
    old vectors and re-ingests.

    Context fields
    --------------
    ``user_id`` and ``session_id`` are stored in per-chunk metadata so that
    retrieval results can be attributed to their uploader context.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe_filename = os.path.basename(file.filename or "upload")
    unique_filename = f"{uuid.uuid4().hex}_{safe_filename}"
    temp_file_path = os.path.join(UPLOAD_DIR, unique_filename)

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    written = 0
    try:
        with open(temp_file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # stream 1 MiB at a time
                written += len(chunk)
                if written > max_bytes:
                    # Partial file on disk — remove it before rejecting.
                    buffer.close()
                    os.remove(temp_file_path)
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File exceeds the {settings.MAX_UPLOAD_SIZE_MB} MB upload limit. "
                            "Please upload a smaller file."
                        ),
                    )
                buffer.write(chunk)
    except HTTPException:
        raise  # re-raise 413 without wrapping it
    except Exception as exc:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Could not save upload: {exc}") from exc

    meta = {"filename": safe_filename, "user_id": user_id, "session_id": session_id}

    # Synchronous pre-flight (fast: validation + dedup hash)
    # Runs inline so the caller immediately knows about invalid/duplicate files
    # without waiting for the embedding pipeline.
    try:
        preflight = await asyncio.to_thread(
            rag_service.preflight_check,
            file_path=temp_file_path,
            metadata=meta,
            force=force,
        )
    except Exception as exc:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Pre-flight error: {exc}") from exc

    if preflight["status"] == "failed":
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(
            status_code=422,
            detail={
                "message": f"Failed to ingest '{safe_filename}'.",
                "error": preflight.get("error"),
                "document_id": preflight.get("document_id"),
            },
        )

    if preflight["status"] == "duplicate":
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return {
            "message": (
                f"'{safe_filename}' was already ingested as "
                f"'{preflight['duplicate_of']}' on {preflight['originally_ingested_at']}. "
                "Skipped to avoid duplicate chunks. Pass force=true to re-ingest."
            ),
            "status": "duplicate",
            "document_id": preflight["document_id"],
            "chunks_added": 0,
            "duplicate_of": preflight["duplicate_of"],
            "originally_ingested_at": preflight["originally_ingested_at"],
            "user_id": preflight["user_id"],
            "session_id": preflight["session_id"],
        }

    # preflight["status"] == "ready"
    # The DB row already has status="processing" (written by preflight_check).
    document_id = preflight["document_id"]

    # Background pipeline
    # Fire-and-forget: slow stages (parse -> embed -> upsert) run after response.
    # Temp-file cleanup is done here so the file still exists when pipeline runs.
    async def _background_ingest():
        try:
            await asyncio.to_thread(
                rag_service.run_pipeline,
                file_path=temp_file_path,
                metadata=meta,
                document_id=document_id,
                file_hash=preflight["file_hash"],
            )
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    asyncio.ensure_future(_background_ingest())

    return JSONResponse(
        status_code=202,
        content={
            "message": (
                f"'{safe_filename}' is being processed in the background. "
                "Poll GET /rag/documents/{document_id} for status updates."
            ),
            "status": "processing",
            "document_id": document_id,
            "filename": safe_filename,
            "user_id": user_id,
            "session_id": session_id,
        },
    )


# ---------------------------------------------------------------------------
# GET /rag/documents
# ---------------------------------------------------------------------------


@router.get("/documents")
async def list_ingested_documents():
    """List all documents tracked in the RAG vector store.

    Returns ingestion status, chunk counts, document IDs, and uploader context.
    """
    docs = await asyncio.to_thread(rag_service.list_documents)
    return {
        "total": len(docs),
        "documents": docs,
    }


# ---------------------------------------------------------------------------
# GET /rag/documents/{document_id}
# ---------------------------------------------------------------------------


@router.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Fetch metadata for a single document by its document_id."""
    doc = await asyncio.to_thread(rag_service.get_document, document_id)
    if doc is None:
        raise HTTPException(
            status_code=404,
            detail=f"No document found with document_id='{document_id}'.",
        )
    return doc


# ---------------------------------------------------------------------------
# DELETE /rag/documents/{document_id}
# ---------------------------------------------------------------------------


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document's vectors from Qdrant and its DB record.

    This is permanent -- the document must be re-uploaded to be available
    for retrieval again.
    """
    result = await asyncio.to_thread(rag_service.delete_document, document_id)

    if result["status"] == "not_found":
        raise HTTPException(
            status_code=404,
            detail=f"No document found with document_id='{document_id}'.",
        )

    return {
        "message": f"Document '{document_id}' deleted successfully.",
        "status": "deleted",
        "document_id": document_id,
        "vectors_deleted": result["vectors_deleted"],
    }
