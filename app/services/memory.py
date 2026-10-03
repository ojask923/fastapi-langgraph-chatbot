"""Long-Term Memory service (mem0 + qdrant / local DB fallback).

Architecture role
-----------------
Mem0 is used ONLY for long-term, stable user memory:
  preferences, facts, projects, goals, and important corrections.

Two lightweight gate heuristics control Mem0 access:
  read gate  (_should_retrieve): skip trivial / short / greeting messages.
  write gate (_should_store):    only persist long-term-worthy content.

Search results are capped at MEM0_MAX_RESULTS (default 5).

The local UserMemory DB table is kept for display/management only
(GET /memories, DELETE /memories).  It is NOT mixed into retrieval.
"""

import asyncio
from typing import Optional, List, Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from sqlmodel import SQLModel, Field, Session, select

from app.config import settings
from app.services.database import db_service


# ---------------------------------------------------------------------------
# DB model (management / display only)
# ---------------------------------------------------------------------------

class UserMemory(SQLModel, table=True):
    """Fallback database table for long-term user facts (display and management only)."""

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True)
    memory_text: str = Field(description="The extracted fact or preference")
    category: Optional[str] = Field(default="general")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Gate heuristics
# ---------------------------------------------------------------------------

# Words/phrases that indicate a trivial, greeting-style message
_GREETING_TOKENS: Set[str] = {
    "hi", "hello", "hey", "hiya", "howdy", "sup", "yo",
    "ok", "okay", "k",
    "yes", "no", "yep", "nope", "yeah", "nah",
    "thanks", "thank", "ty", "thx", "cheers",
    "bye", "goodbye", "cya", "later",
    "sure", "fine", "good", "great", "awesome", "cool", "nice",
    "lol", "lmao", "haha", "hehe",
}

# Tokens that trigger a memory retrieval (first-person pronouns and memory words)
_PERSONAL_PRONOUNS: Set[str] = {
    "i", "me", "my", "mine", "we", "us", "our", "i'm", "i've", "i'd", "i'll"
}

_MEMORY_KEYWORDS: Set[str] = {
    "remember", "forgot", "forget", "recall", "previous", "yesterday",
    "earlier", "before", "discuss", "discussed", "told", "said",
    "mention", "mentioned", "project", "prefer", "preference", "goal",
    # identity / personal-info queries
    "name", "about", "know", "who", "what",
}

# Keywords that strongly suggest long-term-worthy content
_STORE_INDICATORS: List[str] = [
    "my name is", "i am called", "call me",
    "i live in", "i'm from", "i work at", "i work as", "i work for",
    "i prefer", "my preference", "i strongly dislike",
    "my goal is", "my project is", "i'm building", "i am building",
    "remember that", "remember this", "don't forget", "keep in mind",
    "always use", "never use", "please always", "please never",
    "my favourite", "my favorite",
    "my email", "my phone", "my address",
    "i was born", "i'm a ", "i am a ",
    "i study", "i'm studying", "i graduated",
    "my team", "my company", "my boss",
    "correction:", "actually,", "that's wrong", "to clarify",
]


def _should_retrieve(query: str) -> bool:
    """Return True if the query warrants a Mem0 search.

    Uses a lightweight heuristic checking for personal pronouns or memory-related keywords.
    Skips purely trivial greetings and general/impersonal questions.
    """
    words = query.strip().split()
    if not words:
        return False
        
    lower_words = {w.lower().strip(".,!?'\"") for w in words}
    
    # 1. Skip if it's purely a greeting/acknowledgement
    if lower_words.issubset(_GREETING_TOKENS):
        return False
        
    # 2. Check for personal/memory indicators
    for w in lower_words:
        if w in _PERSONAL_PRONOUNS or w in _MEMORY_KEYWORDS:
            return True
            
    return False


def _should_store(message: str) -> bool:
    """Return True if the message likely contains long-term-worthy content."""
    lower = message.lower()
    return any(indicator in lower for indicator in _STORE_INDICATORS)


# ---------------------------------------------------------------------------
# MemoryService
# ---------------------------------------------------------------------------

# Thread pool for blocking I/O (model loading) so the async event loop stays free
_mem0_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mem0-init")


class MemoryService:
    """Long-term memory management using Mem0 with Qdrant backend."""

    def __init__(self):
        self._mem0_instance = None
        self._use_mem0 = False
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self):
        """Initialize mem0 with qdrant backend; fall back to DB-only mode.

        Runs the blocking model-load (sentence-transformers embedding) in a
        ThreadPoolExecutor so the async event loop stays responsive.
        """
        if self._initialized:
            return
        async with self._init_lock:
            # Double-checked locking: another coroutine may have initialised
            # while we were waiting for the lock.
            if self._initialized:
                return

            # Use MEM0_LLM_PROVIDER (decoupled from DEFAULT_PROVIDER) so
            # memory extraction always uses a reliable, high-TPM model.
            mem0_provider = settings.MEM0_LLM_PROVIDER.lower()

            def _blocking_init():
                """Runs synchronously in a thread — safe to call blocking loaders here."""
                from mem0 import AsyncMemory  # noqa: PLC0415

                llm_config: Dict[str, Any] = {}
                if mem0_provider == "groq":
                    llm_config = {
                        "provider": "groq",
                        "config": {
                            "model": settings.MEM0_LLM_MODEL,
                            "api_key": settings.GROQ_API_KEY,
                            "max_tokens": 2000,
                        },
                    }
                elif mem0_provider == "openai":
                    llm_config = {
                        "provider": "openai",
                        "config": {
                            "model": settings.MEM0_LLM_MODEL,
                            "api_key": settings.OPENAI_API_KEY,
                        },
                    }
                elif mem0_provider == "anthropic":
                    llm_config = {
                        "provider": "anthropic",
                        "config": {
                            "model": settings.MEM0_LLM_MODEL,
                            "api_key": settings.ANTHROPIC_API_KEY,
                        },
                    }
                elif mem0_provider == "gemini":
                    llm_config = {
                        "provider": "gemini",
                        "config": {
                            "model": settings.MEM0_LLM_MODEL,
                            "api_key": settings.GEMINI_API_KEY,
                        },
                    }
                else:
                    llm_config = {
                        "provider": "ollama",
                        "config": {"model": settings.MEM0_LLM_MODEL},
                    }

                from app.services.qdrant_shared import get_qdrant_client, get_hf_embeddings  # noqa: PLC0415

                # Reuse the shared QdrantClient — avoids a second exclusive
                # file lock on the embedded Qdrant data directory and avoids
                # loading all-MiniLM-L6-v2 a second time.
                shared_client = get_qdrant_client()

                cfg = {
                    "vector_store": {
                        "provider": "qdrant",
                        "config": {
                            "collection_name": "user_memories_hf",
                            "client": shared_client,
                            "embedding_model_dims": 384,
                        },
                    },
                    "llm": llm_config,
                    "embedder": {
                        "provider": "huggingface",
                        "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
                    },
                }
                return AsyncMemory.from_config(cfg)

            try:
                loop = asyncio.get_event_loop()
                self._mem0_instance = await loop.run_in_executor(
                    _mem0_executor, _blocking_init
                )
                self._use_mem0 = True
                print(f"[INFO] mem0 initialized using {mem0_provider} for long-term memory (Qdrant).")
            except Exception as e:
                print(f"[WARNING] Could not initialize mem0, using DB fallback for management only: {e}")
                self._use_mem0 = False

            self._initialized = True

    # ------------------------------------------------------------------
    # Read  (read gate applied here)
    # ------------------------------------------------------------------

    async def search(self, user_id: str, query: str) -> str:
        """Search relevant long-term Mem0 memories for a user.

        Returns an empty string when:
          - user_id is empty
          - the read gate deems the query trivial
          - Mem0 is not available

        Results are capped at MEM0_MAX_RESULTS.
        """
        if not user_id:
            return ""

        if not _should_retrieve(query):
            return ""

        await self.initialize()

        if not (self._use_mem0 and self._mem0_instance):
            return ""

        try:
            results = await self._mem0_instance.search(
                query=query,
                filters={"user_id": str(user_id)},
                top_k=settings.MEM0_MAX_RESULTS,
            )
            if results and isinstance(results, dict) and "results" in results:
                items = results["results"][: settings.MEM0_MAX_RESULTS]
            elif isinstance(results, list):
                items = results[: settings.MEM0_MAX_RESULTS]
            else:
                items = []

            return "\n".join(f"- {r['memory']}" for r in items if r.get("memory"))
        except Exception as e:
            print(f"[WARNING] mem0 search error: {e}")
            return ""

    # ------------------------------------------------------------------
    # Write  (write gate applied here)
    # ------------------------------------------------------------------

    async def add(
        self,
        user_id: str,
        messages: List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Extract and persist facts to long-term memory, write-gated.

        Only stores when the user message contains likely long-term-worthy
        content (personal facts, preferences, corrections, goals, etc.).
        """
        if not user_id or not messages:
            return

        user_text = next(
            (m["content"] for m in messages if m.get("role") == "user"), ""
        )
        if not user_text or not _should_store(user_text):
            return

        await self.initialize()

        if self._use_mem0 and self._mem0_instance:
            try:
                await self._mem0_instance.add(
                    messages, user_id=str(user_id), metadata=metadata
                )
                return
            except Exception as e:
                print(f"[WARNING] mem0 add error: {e}")

        # Fallback: save to local UserMemory DB table
        try:
            lower = user_text.lower()
            for indicator in _STORE_INDICATORS:
                if indicator in lower:
                    with Session(db_service.engine) as db:
                        existing = db.exec(
                            select(UserMemory).where(
                                UserMemory.user_id == str(user_id),
                                UserMemory.memory_text == user_text,
                            )
                        ).first()
                        if not existing:
                            mem = UserMemory(
                                user_id=str(user_id),
                                memory_text=user_text,
                                category="preference",
                            )
                            db.add(mem)
                            db.commit()
                    break
        except Exception as e:
            print(f"[WARNING] Memory DB save error: {e}")

    # ------------------------------------------------------------------
    # Management helpers (display / admin only)
    # ------------------------------------------------------------------

    async def get_all(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all stored long-term memories for a user."""
        if not user_id:
            return []

        await self.initialize()

        all_memories: List[Dict[str, Any]] = []

        if self._use_mem0 and self._mem0_instance:
            try:
                res = await self._mem0_instance.get_all(
                    filters={"user_id": str(user_id)}
                )
                if isinstance(res, dict) and "results" in res:
                    all_memories.extend(res["results"])
                elif isinstance(res, list):
                    all_memories.extend(res)
            except Exception as e:
                print(f"[WARNING] mem0 get_all error: {e}")

        try:
            with Session(db_service.engine) as db:
                memories = db.exec(
                    select(UserMemory).where(UserMemory.user_id == str(user_id))
                ).all()
                all_memories.extend(
                    {
                        "id": m.id,
                        "memory": m.memory_text,
                        "created_at": m.created_at.isoformat(),
                        "category": m.category,
                    }
                    for m in memories
                )
        except Exception:
            pass

        return all_memories

    async def delete(self, user_id: str, memory_id: Any) -> bool:
        """Delete a specific memory by ID."""
        success = False

        if self._use_mem0 and self._mem0_instance:
            try:
                await self._mem0_instance.delete(memory_id=str(memory_id))
                success = True
            except Exception as e:
                print(f"[WARNING] mem0 delete error: {e}")

        try:
            with Session(db_service.engine) as db:
                mem = db.get(UserMemory, int(memory_id))
                if mem and mem.user_id == str(user_id):
                    db.delete(mem)
                    db.commit()
                    return True
        except Exception:
            pass

        return success

    async def clear_all(self, user_id: str):
        """Delete all memories for a user."""
        if self._use_mem0 and self._mem0_instance:
            try:
                await self._mem0_instance.delete_all(user_id=str(user_id))
            except Exception as e:
                print(f"[WARNING] mem0 clear_all error: {e}")

        try:
            with Session(db_service.engine) as db:
                mems = db.exec(
                    select(UserMemory).where(UserMemory.user_id == str(user_id))
                ).all()
                for m in mems:
                    db.delete(m)
                db.commit()
        except Exception:
            pass


memory_service = MemoryService()
