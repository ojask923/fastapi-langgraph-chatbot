"""LangGraph agent workflow with multi-provider LLM support and PostgresSaver checkpointing.

Memory architecture
-------------------
* AgentState extends MessagesState with a ``summary`` field that stores the
  rolling compact summary of older conversation turns.
* Context is assembled by ContextEngine at model-call time — never injected
  through config or accumulated in persistent state as SystemMessages.
* A ``summarize`` node runs after each ``agent`` node invocation when the
  message count exceeds SUMMARY_THRESHOLD; it compacts old messages and
  updates ``AgentState.summary`` via the normal LangGraph state-update path.
* Mem0 search is called inside the graph node (gated by memory_service's
  read gate) — not outside in get_response/stream_response.
"""

import asyncio
import logging
from typing import AsyncGenerator, Dict, Any, List, Optional, Annotated

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from app.config import settings
from app.agent.tools import get_available_tools
from app.services.memory import memory_service
from app.services.database import db_service
from app.services.context_engine import context_engine
from app.services.summarizer import summarizer
from app.services.llm_factory import get_llm
from app.services.rag_service import rag_service  # for has_documents pre-check

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Background-task helper
# ---------------------------------------------------------------------------

# Module-level set keeps references alive so the GC never collects running tasks.
_background_tasks: set = set()


def _fire_and_forget(coro) -> None:
    """Schedule a coroutine as a fire-and-forget background task.

    Holds a strong reference to the task until it completes so Python's GC
    cannot collect it mid-execution.  Any exception is logged rather than
    silently swallowed.
    """
    task = asyncio.create_task(coro)
    _background_tasks.add(task)

    def _on_done(t: asyncio.Task) -> None:
        _background_tasks.discard(t)
        exc = t.exception() if not t.cancelled() else None
        if exc:
            logger.warning("[background task] %s raised: %s", t.get_name(), exc)

    task.add_done_callback(_on_done)


# ---------------------------------------------------------------------------
# DB connection helper
# ---------------------------------------------------------------------------

def _build_pg_conn_string(db_url: str) -> str:
    """Convert a SQLAlchemy DATABASE_URL to a plain psycopg3 connection string."""
    conn = db_url
    for suffix in ("+psycopg2", "+psycopg", "+asyncpg", "+pg8000"):
        conn = conn.replace(f"postgresql{suffix}://", "postgresql://")
    conn = conn.replace("postgres://", "postgresql://")
    return conn


# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------

def _merge_citations(left: dict, right: dict | None) -> dict:
    """Reducer for rag_citations that supports parallel tool-call updates.

    LangGraph applies this function when multiple graph nodes (e.g. two
    parallel retrieve_documents calls) write to the same state key in a
    single step.  Without a reducer, LangGraph raises InvalidUpdateError.

    ``right=None`` is the turn-start reset sentinel sent by get_response /
    stream_response at the beginning of each user turn so that stale
    citations from a prior turn are cleared before new retrieval runs.
    """
    if right is None:
        return {}
    return {**(left or {}), **right}


class AgentState(TypedDict):
    """LangGraph state carrying message history, a rolling summary, and RAG citations.

    The ``summary`` field stores compact text produced by the Summarizer
    when the conversation grows beyond SUMMARY_THRESHOLD messages.
    The ``rag_citations`` field is populated by the retrieve_documents tool
    and contains the citation map from the last RAG retrieval.
    Both fields are persisted by PostgresSaver across restarts.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    summary: str
    rag_citations: Annotated[dict, _merge_citations]  # merged by reducer; None resets


# ---------------------------------------------------------------------------
# LLM factory is now in app/services/llm_factory.py
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ChatAgent
# ---------------------------------------------------------------------------

class ChatAgent:
    """Manages LangGraph compilation and async-safe PostgresSaver checkpointing."""

    def __init__(self):
        self.checkpointer = None
        self._checkpointer_ready = False
        self._checkpointer_error: Optional[str] = None
        self._pool = None
        self._init_lock = asyncio.Lock()  # guards _ensure_checkpointer against concurrent callers
        self.tools = get_available_tools()
        self._compiled_graphs: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Checkpointer lifecycle
    # ------------------------------------------------------------------

    async def _ensure_checkpointer(self) -> None:
        """Initialise the checkpointer exactly once, safe against concurrent callers.

        Uses double-checked locking (mirroring MemoryService._init_lock) so that
        two concurrent first requests cannot both pass the ready-check and each
        create their own AsyncConnectionPool, leaking one.
        """
        if self._checkpointer_ready:
            return

        async with self._init_lock:
            # Second check inside the lock: another coroutine may have finished
            # initialisation while we waited to acquire it.
            if self._checkpointer_ready:
                return

            db_url = settings.DATABASE_URL
            is_postgres = db_url.startswith(("postgresql://", "postgresql+", "postgres://"))

            if is_postgres:
                try:
                    import psycopg
                    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
                    from psycopg_pool import AsyncConnectionPool

                    conn_string = _build_pg_conn_string(db_url)

                    async with await psycopg.AsyncConnection.connect(
                        conn_string, autocommit=True
                    ) as setup_conn:
                        await AsyncPostgresSaver(setup_conn).setup()

                    self._pool = AsyncConnectionPool(
                        conninfo=conn_string,
                        min_size=1,
                        max_size=10,
                        open=False,
                    )
                    await self._pool.open(wait=True, timeout=10)
                    self.checkpointer = AsyncPostgresSaver(self._pool)
                    self._checkpointer_error = None
                    logger.info("[Checkpointer] PostgresSaver ready — checkpoints persist across restarts.")
                    print("[INFO] LangGraph checkpointer: PostgresSaver (persistent across restarts).")
                except Exception as exc:
                    import traceback
                    self._checkpointer_error = f"{type(exc).__name__}: {exc}"
                    logger.warning(
                        "[Checkpointer] PostgresSaver init failed — falling back to MemorySaver.\n%s",
                        traceback.format_exc(),
                    )
                    print("[WARNING] PostgresSaver FAILED — using MemorySaver fallback.")
                    print(f"[WARNING] Error: {type(exc).__name__}: {exc}")
                    self.checkpointer = MemorySaver()
            else:
                # SQLite path — use AsyncSqliteSaver for cross-restart persistence
                db_path = db_url.replace("sqlite:///", "").replace("sqlite+aiosqlite:///", "")
                if not db_path:
                    db_path = "./chatbot.db"
                try:
                    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

                    self._sqlite_ctx = AsyncSqliteSaver.from_conn_string(db_path)
                    self.checkpointer = await self._sqlite_ctx.__aenter__()
                    self._checkpointer_error = None
                    logger.info("[Checkpointer] AsyncSqliteSaver ready — checkpoints persist across restarts.")
                    print(f"[INFO] LangGraph checkpointer: AsyncSqliteSaver ({db_path}) — persistent across restarts.")
                except Exception as exc:
                    import traceback
                    self._checkpointer_error = f"{type(exc).__name__}: {exc}"
                    logger.warning(
                        "[Checkpointer] AsyncSqliteSaver init failed — falling back to MemorySaver.\n%s",
                        traceback.format_exc(),
                    )
                    print("[WARNING] AsyncSqliteSaver FAILED — using MemorySaver fallback.")
                    print(f"[WARNING] Error: {type(exc).__name__}: {exc}")
                    self.checkpointer = MemorySaver()

            self._checkpointer_ready = True
            self._compiled_graphs.clear()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_graph(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        """Construct a compiled StateGraph for the selected provider and model."""
        assert self.checkpointer is not None, "call await _ensure_checkpointer() before build_graph()"

        provider = provider or settings.DEFAULT_PROVIDER
        temperature = temperature if temperature is not None else settings.TEMPERATURE
        cache_key = f"{provider}_{model}_{temperature}"

        if cache_key in self._compiled_graphs:
            return self._compiled_graphs[cache_key]

        tools = self.tools if settings.ENABLE_TOOLS else []
        llm = get_llm(provider, model, temperature, tools=tools)
        tool_node = ToolNode(self.tools)

        # ---- agent node ---------------------------------------------------
        async def call_model(state: AgentState, config: RunnableConfig):
            """Build context via ContextEngine and call the LLM.

            RAG routing pre-check
            ---------------------
            Before building the context we run a fast COUNT query
            (``rag_service.has_documents``) scoped to the current user.

            * No documents  →  ``retrieve_documents`` is removed from the
              bound tool list so the model structurally cannot call it.
            * Documents exist →  a short routing instruction is appended to
              the system prompt so the model reliably calls
              ``retrieve_documents`` when the user references an uploaded
              file, rather than guessing or skipping retrieval.
            """
            configurable = config.get("configurable", {})
            system_instructions = configurable.get(
                "system_prompt", "You are a helpful, smart AI assistant."
            )
            user_id = configurable.get("user_id", "")

            messages = state["messages"]

            # ── RAG pre-check ────────────────────────────────────────────────
            # Runs in a thread so the indexed COUNT never blocks the event loop.
            user_has_docs = await asyncio.to_thread(
                rag_service.has_documents, user_id or None
            )

            if user_has_docs:
                # (b) Append a compact routing rule to the system instructions
                # so the model calls retrieve_documents for document questions.
                system_instructions = (
                    system_instructions.rstrip()
                    + "\n\n"
                    + "DOCUMENT RETRIEVAL RULE:\n"
                    + "The user has documents in the knowledge base. "
                    + "ALWAYS call the `retrieve_documents` tool first when the "
                    + "user's message references \"the document\", \"this file\", "
                    + "\"what I uploaded\", \"the report\", \"the PDF\", or any "
                    + "phrasing that implies they want information from an uploaded "
                    + "file. Do NOT answer document questions from memory alone. "
                    + "Call `retrieve_documents` AT MOST ONCE per turn; combine "
                    + "multiple sub-questions into a single query string."
                )
                active_tools = self.tools  # full tool list
            else:
                # (a) Drop retrieve_documents — nothing to retrieve for this user.
                active_tools = [
                    t for t in self.tools if t.name != "retrieve_documents"
                ]

            # Rebind the LLM with the per-call tool list.
            # get_llm is cheap after the first call (returns a cached/new instance
            # with the correct tool binding applied via bind_tools).
            active_llm = get_llm(
                configurable.get("provider", provider),
                configurable.get("model", model),
                configurable.get("temperature", temperature),
                tools=active_tools if settings.ENABLE_TOOLS else [],
            )

            # Extract the last HumanMessage's text as the current query for Mem0 search
            current_query = ""
            for i in range(len(messages) - 1, -1, -1):
                m = messages[i]
                if isinstance(m, HumanMessage):
                    content = m.content
                    if isinstance(content, str):
                        current_query = content
                    elif isinstance(content, list):
                        current_query = " ".join(
                            p.get("text", "") if isinstance(p, dict) else str(p)
                            for p in content
                        )
                    break

            # Retrieve Mem0 long-term memories (read gate is inside memory_service)
            mem0_memories = ""
            if user_id and current_query:
                try:
                    mem0_memories = await memory_service.search(
                        user_id=user_id, query=current_query
                    )
                except Exception as exc:
                    logger.warning("[call_model] Mem0 search failed: %s", exc)

            # Build the final context via ContextEngine
            # Pass any RAG citations accumulated in state (from retrieve_documents tool calls)
            rag_citations = state.get("rag_citations", {})
            final_messages = context_engine.build(
                system_instructions=system_instructions,
                conversation_summary=state.get("summary", ""),
                mem0_memories=mem0_memories,
                messages=messages,
                rag_citations=rag_citations,
            )

            response = await active_llm.ainvoke(final_messages)
            return {"messages": [response]}

        # ---- summarize node -----------------------------------------------
        async def maybe_summarize(state: AgentState, config: RunnableConfig):
            """Compact old messages into a rolling summary when threshold is hit."""
            messages = state["messages"]
            should_compact = await summarizer.should_summarize(messages)
            if not should_compact:
                return {}

            configurable = config.get("configurable", {})
            updates = await summarizer.compact(
                messages=messages,
                existing_summary=state.get("summary", ""),
                provider=configurable.get("provider", provider),
                model=configurable.get("model", model),
            )
            logger.info(
                "[Graph] Summarization complete. Summary: %d chars. Messages kept: %d.",
                len(updates.get("summary", "")),
                len(updates.get("messages", messages)),
            )
            return updates

        # ---- routing ------------------------------------------------------
        def route_after_agent(state: AgentState):
            """Route to tools if the last message has tool calls, else summarize."""
            last = state["messages"][-1] if state["messages"] else None
            if last and hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return "summarize"

        # ---- wire the graph -----------------------------------------------
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)
        workflow.add_node("summarize", maybe_summarize)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            route_after_agent,
            {"tools": "tools", "summarize": "summarize"},
        )
        workflow.add_edge("tools", "agent")
        workflow.add_edge("summarize", END)

        compiled = workflow.compile(checkpointer=self.checkpointer)
        self._compiled_graphs[cache_key] = compiled
        return compiled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_response(
        self,
        message: str,
        session_id: str = "default",
        user_id: str = "default_user",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Invoke the graph and return the final reply."""
        await self._ensure_checkpointer()

        provider = provider or settings.DEFAULT_PROVIDER
        temperature = temperature if temperature is not None else settings.TEMPERATURE
        graph = self.build_graph(provider, model, temperature)

        config = {
            "configurable": {
                "thread_id": session_id,
                "user_id": user_id,
                "provider": provider,
                "model": model,
                "system_prompt": system_prompt or "You are a helpful, smart AI assistant.",
            }
        }

        # Only send the new HumanMessage; LangGraph loads prior state from checkpointer
        input_state: Dict[str, Any] = {
            "messages": [HumanMessage(content=message)],
            "rag_citations": None,  # None sentinel triggers _merge_citations reset
        }
        result = await graph.ainvoke(input_state, config=config)

        all_messages = result.get("messages", [])
        last_ai_msg = next(
            (m for m in reversed(all_messages) if isinstance(m, AIMessage) and m.content),
            None,
        )
        content = last_ai_msg.content if last_ai_msg else "No response generated."

        # Conditionally store to Mem0 in background (write gate is inside add())
        _fire_and_forget(
            memory_service.add(
                user_id=user_id,
                messages=[
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": content},
                ],
            )
        )

        # Extract citations from the final state
        raw_citations = result.get("rag_citations", {})
        citations = [
            {"index": idx, **meta}
            for idx, meta in raw_citations.items()
        ] if raw_citations else []

        return {
            "content": content,
            "session_id": session_id,
            "provider": provider,
            "model": model,
            "memories_used": False,
            "citations": citations,
        }

    async def stream_response(
        self,
        message: str,
        session_id: str = "default",
        user_id: str = "default_user",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response tokens and tool call notifications."""
        await self._ensure_checkpointer()

        provider_clean = (provider or settings.DEFAULT_PROVIDER).lower()
        temperature = temperature if temperature is not None else settings.TEMPERATURE
        graph = self.build_graph(provider_clean, model, temperature)

        config = {
            "configurable": {
                "thread_id": session_id,
                "user_id": user_id,
                "provider": provider_clean,
                "model": model,
                "system_prompt": system_prompt or "You are a helpful, smart AI assistant.",
            }
        }

        input_state: Dict[str, Any] = {
            "messages": [HumanMessage(content=message)],
            "rag_citations": None,  # None sentinel triggers _merge_citations reset
        }
        accumulated_text = ""

        try:
            async for event in graph.astream_events(input_state, config=config, version="v2"):
                kind = event.get("event")

                if kind == "on_chat_model_stream":
                    node = event.get("metadata", {}).get("langgraph_node")
                    if node != "agent":
                        continue
                    
                    chunk = event["data"].get("chunk")
                    if chunk and chunk.content:
                        if isinstance(chunk.content, str):
                            accumulated_text += chunk.content
                            yield {"type": "token", "content": chunk.content}
                        elif isinstance(chunk.content, list):
                            for part in chunk.content:
                                if isinstance(part, dict) and "text" in part:
                                    accumulated_text += part["text"]
                                    yield {"type": "token", "content": part["text"]}

                elif kind == "on_tool_start":
                    tool_name = event.get("name", "tool")
                    tool_args = event.get("data", {}).get("input", {})
                    yield {"type": "tool_start", "name": tool_name, "args": tool_args}

                elif kind == "on_tool_end":
                    tool_name = event.get("name", "tool")
                    tool_output = str(event.get("data", {}).get("output", ""))
                    yield {"type": "tool_end", "name": tool_name, "result": tool_output}

        finally:
            # Always fire mem0 write, even on early client disconnect
            if accumulated_text:
                _fire_and_forget(
                    memory_service.add(
                        user_id=user_id,
                        messages=[
                            {"role": "user", "content": message},
                            {"role": "assistant", "content": accumulated_text},
                        ],
                    )
                )

        # Extract and yield citations from the final state
        final_state = await graph.aget_state(config)
        raw_citations = final_state.values.get("rag_citations", {})
        if raw_citations:
            citations = [
                {"index": idx, **meta}
                for idx, meta in raw_citations.items()
            ]
            yield {"type": "citations", "citations": citations}

        yield {"type": "done"}

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Retrieve stored chat history for a session from the relational DB."""
        db_msgs = db_service.get_session_messages(session_id)
        return [
            {"role": m.role, "content": m.content}
            for m in db_msgs
            if m.role in ("user", "assistant") and m.content
        ]

    async def clear_history(self, session_id: str) -> None:
        """Clear LangGraph checkpoint rows for a session (async, non-blocking).

        Uses the checkpointer's own async delete API so the event loop is never
        blocked by synchronous psycopg I/O.

        For AsyncPostgresSaver / AsyncSqliteSaver: calls adelete_thread().
        For MemorySaver: pops from the in-process dict (cheap, no I/O).
        Silently no-ops if the checkpointer has not yet been initialised.
        """
        if not self._checkpointer_ready or self.checkpointer is None:
            return

        if isinstance(self.checkpointer, MemorySaver):
            try:
                if hasattr(self.checkpointer, "storage"):
                    self.checkpointer.storage.pop(session_id, None)
            except Exception:
                pass
            return

        # AsyncPostgresSaver / AsyncSqliteSaver — use their own async delete method.
        try:
            await self.checkpointer.adelete_thread(session_id)
        except Exception as exc:
            logger.warning(
                "[clear_history] Could not delete checkpoints for session %s: %s",
                session_id,
                exc,
            )


agent = ChatAgent()
