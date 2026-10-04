"""FastAPI API routes for the simple chatbot with database persistence."""

import asyncio
import json
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.agent.graph import agent
from app.services.database import db_service
from app.services.memory import memory_service
from app.api import rag_routes


router = APIRouter(prefix="/api")
router.include_router(rag_routes.router)


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message text")
    session_id: str = Field(default="default", description="Conversation session thread ID")
    user_id: Optional[str] = Field(default="default_user", description="User identifier for long-term memory")
    provider: Optional[str] = Field(default=None, description="LLM provider: groq, ollama, openai, gemini, anthropic")
    model: Optional[str] = Field(default=None, description="Model identifier")
    system_prompt: Optional[str] = Field(default=None, description="Custom system instructions")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ChatResponse(BaseModel):
    content: str
    session_id: str
    provider: str
    model: Optional[str] = None
    citations: Optional[List[dict]] = Field(
        default=None,
        description=(
            "Source citations when the answer draws from retrieved documents. "
            "Each entry contains: index, filename, page_number, section, chunk_id, "
            "document_id, dense_score, fusion_score (None when hybrid search is off), "
            "rerank_score (None when reranking is off or failed)."
        ),
    )


class SessionCreateRequest(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = "New Chat"


@router.get("/sessions")
async def get_sessions():
    """Retrieve all chat sessions from the database."""
    sessions = await asyncio.to_thread(db_service.get_all_sessions)
    return [
        {
            "id": s.id,
            "title": s.title,
            "created_at": s.created_at.isoformat(),
            "updated_at": s.updated_at.isoformat(),
        }
        for s in sessions
    ]


@router.post("/sessions")
async def create_session(payload: SessionCreateRequest):
    """Create a new chat session in the database."""
    import uuid
    session_id = payload.id or f"sess_{uuid.uuid4().hex[:8]}"
    chat_session = await asyncio.to_thread(db_service.create_session, session_id=session_id, title=payload.title or "New Chat")
    return {
        "id": chat_session.id,
        "title": chat_session.title,
        "created_at": chat_session.created_at.isoformat(),
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session and its messages from the database."""
    success = await asyncio.to_thread(db_service.delete_session, session_id)
    await agent.clear_history(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    """Non-streaming chat endpoint with database persistence."""
    try:
        # Save user message to database
        await asyncio.to_thread(db_service.add_message, payload.session_id, role="user", content=payload.message)

        res = await agent.get_response(
            message=payload.message,
            session_id=payload.session_id,
            user_id=payload.user_id or "default_user",
            provider=payload.provider or settings.DEFAULT_PROVIDER,
            model=payload.model or settings.DEFAULT_MODEL,
            system_prompt=payload.system_prompt,
            temperature=payload.temperature if payload.temperature is not None else settings.TEMPERATURE,
        )

        # Save assistant response to database
        await asyncio.to_thread(db_service.add_message, payload.session_id, role="assistant", content=res["content"])

        return ChatResponse(
            content=res["content"],
            session_id=res["session_id"],
            provider=res["provider"],
            model=res.get("model"),
            citations=res.get("citations") or None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream_endpoint(payload: ChatRequest):
    """Server-Sent Events (SSE) streaming endpoint with database persistence."""
    # Record user message in DB safely
    try:
        await asyncio.to_thread(db_service.add_message, payload.session_id, role="user", content=payload.message)
    except Exception as e:
        print(f"[WARNING] Failed to record user message in database: {e}")

    async def event_generator():
        accumulated_response = ""
        last_tool = None
        try:
            async for event in agent.stream_response(
                message=payload.message,
                session_id=payload.session_id,
                user_id=payload.user_id or "default_user",
                provider=payload.provider or settings.DEFAULT_PROVIDER,
                model=payload.model or settings.DEFAULT_MODEL,
                system_prompt=payload.system_prompt,
                temperature=payload.temperature if payload.temperature is not None else settings.TEMPERATURE,
            ):
                if event.get("type") == "token":
                    accumulated_response += event.get("content", "")
                elif event.get("type") == "tool_start":
                    last_tool = event.get("name")

                yield f"data: {json.dumps(event)}\n\n"

            # Save full assistant response to database on completion safely
            if accumulated_response:
                try:
                    await asyncio.to_thread(
                        db_service.add_message,
                        payload.session_id,
                        role="assistant",
                        content=accumulated_response,
                        tool_name=last_tool,
                    )
                except Exception as e:
                    print(f"[WARNING] Failed to record assistant response in database: {e}")

        except Exception as e:
            err_data = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(err_data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/memories/{user_id}")
async def get_user_memories(user_id: str):
    """Get all long-term memory facts stored for a user."""
    memories = await memory_service.get_all(user_id)
    return {"user_id": user_id, "memories": memories}


@router.delete("/memories/{user_id}/{memory_id}")
async def delete_user_memory(user_id: str, memory_id: str):
    """Delete a specific long-term memory fact (accepts UUID strings and integer IDs)."""
    success = await memory_service.delete(user_id, memory_id)
    return {"status": "deleted" if success else "not_found", "memory_id": memory_id}


@router.delete("/memories/{user_id}")
async def clear_user_memories(user_id: str):
    """Clear all long-term memories for a user."""
    await memory_service.clear_all(user_id)
    return {"status": "cleared", "user_id": user_id}


@router.get("/models")
async def get_models():
    """Return available providers and models."""
    return {
        "current_default_provider": settings.DEFAULT_PROVIDER,
        "current_default_model": settings.DEFAULT_MODEL,
        "providers": {
            "groq": {
                "name": "Groq (Ultra-Fast)",
                "description": "GPT-OSS 120B / Qwen 3.8 / 3.6 / GPT-OSS 20B / Compound",
                "models": ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "qwen/qwen3.6-27b", "openai/gpt-oss-20b", "groq/compound"],
                "requires_key": True,
                "is_configured": bool(settings.GROQ_API_KEY),
            },
            "ollama": {
                "name": "Ollama (Local LLM)",
                "description": "100% free local models running on your computer",
                "models": ["llama3.2", "mistral", "deepseek-r1:latest", "qwen2.5", "phi3"],
                "requires_key": False,
                "is_configured": True,
                "base_url": settings.OLLAMA_BASE_URL,
            },
            "openai": {
                "name": "OpenAI",
                "description": "GPT-4o, GPT-4o-mini",
                "models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                "requires_key": True,
                "is_configured": bool(settings.OPENAI_API_KEY),
            },
            "gemini": {
                "name": "Google Gemini",
                "description": "Gemini 1.5 Flash, Gemini 1.5 Pro, Gemini 2.0 Flash",
                "models": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"],
                "requires_key": True,
                "is_configured": bool(settings.GEMINI_API_KEY),
            },
            "anthropic": {
                "name": "Anthropic Claude",
                "description": "Claude 3.5 Sonnet, Claude 3.5 Haiku",
                "models": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"],
                "requires_key": True,
                "is_configured": bool(settings.ANTHROPIC_API_KEY),
            },
            "openrouter": {
                "name": "OpenRouter",
                "description": "Unified API for multiple LLM providers",
                "models": ["poolside/laguna-s-2.1:free", "meta-llama/llama-3.3-70b-instruct:free", "deepseek/deepseek-r1:free"],
                "requires_key": True,
                "is_configured": bool(settings.OPENROUTER_API_KEY),
            },
        },
    }


@router.get("/history/{session_id}")
async def get_session_history(session_id: str):
    """Get conversation history from the database for a session."""
    messages = await asyncio.to_thread(db_service.get_session_messages, session_id)
    return {
        "session_id": session_id,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "tool_name": m.tool_name,
                "created_at": m.created_at.isoformat(),
            }
            for m in messages
        ],
    }


@router.delete("/history/{session_id}")
async def clear_session_history(session_id: str):
    """Clear message history in the database for a session."""
    await asyncio.to_thread(db_service.clear_session_messages, session_id)
    await agent.clear_history(session_id)
    return {"status": "cleared", "session_id": session_id}


@router.get("/health")
async def health_check():
    """Health check endpoint — reports the active backend for every infrastructure layer."""
    from app.agent.graph import agent
    from app.config import settings as s

    db_healthy = await asyncio.to_thread(db_service.health_check)
    db_url = s.DATABASE_URL

    # Determine relational DB backend label
    if db_url.startswith(("postgresql", "postgres")):
        db_backend = "postgresql"
    else:
        db_backend = "sqlite"

    # Determine LangGraph checkpointer backend
    if agent._checkpointer_ready and agent.checkpointer is not None:
        checkpointer_type = type(agent.checkpointer).__name__   # "AsyncPostgresSaver" or "MemorySaver"
        checkpointer_persistent = "PostgresSaver" in checkpointer_type
        checkpointer_error = agent._checkpointer_error  # None on success, error string on fallback
    else:
        checkpointer_type = "not yet initialised (lazy — fires on first chat)"
        checkpointer_persistent = None
        checkpointer_error = None

    # Determine vector store backend
    from app.services.qdrant_shared import vector_store_degraded, vector_store_degraded_reason
    vector_store_path = s.VECTOR_STORE_PATH
    if vector_store_degraded:
        vector_backend = "qdrant-memory (DEGRADED — path locked)"
    elif vector_store_path:
        vector_backend = "qdrant-local"
    else:
        vector_backend = "qdrant-memory"

    # Determine long-term memory (mem0) backend
    from app.services.memory import memory_service
    if memory_service._initialized:
        memory_backend = "mem0+qdrant" if memory_service._use_mem0 else "db-keyword-fallback"
    else:
        memory_backend = "not yet initialised (lazy — fires on first chat)"

    return {
        "status": "healthy" if (db_healthy and not vector_store_degraded) else "degraded",
        "app": s.APP_NAME,
        "version": s.VERSION,
        "infrastructure": {
            "relational_db": {
                "backend": db_backend,
                "url": db_url.split("@")[-1] if "@" in db_url else db_url,   # strip credentials
                "status": "healthy" if db_healthy else "unhealthy",
                "stores": ["sessions", "messages", "rag_document_index", "user_memories_fallback"],
            },
            "checkpointer": {
                "backend": checkpointer_type,
                "persistent_across_restarts": checkpointer_persistent,
                "init_error": checkpointer_error,
                "note": "PostgresSaver writes to relational_db; MemorySaver is in-process RAM only",
            },
            "vector_store": {
                "backend": vector_backend,
                "path": vector_store_path,
                "degraded": vector_store_degraded,
                "degraded_reason": vector_store_degraded_reason,
                "embedding_model": s.EMBEDDING_MODEL,
                "stores": ["rag_document_chunks", "user_memories_hf"],
            },
            "long_term_memory": {
                "backend": memory_backend,
                "vector_collection": "user_memories_hf",
                "note": "mem0 extracts facts via LLM and stores embeddings in a separate Qdrant collection",
            },
        },
        "features": {
            "tools_enabled": s.ENABLE_TOOLS,
            "rag_enabled": True,
            "long_term_memory_enabled": True,
            "streaming": True,
        },
    }

