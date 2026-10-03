"""ContextEngine: single layer that assembles the final LLM context each turn.

Combines:
  1. System instructions (static + dynamic persona)
  2. Conversation summary (compressed older history, stored in AgentState)
  3. Relevant Mem0 memories (long-term user facts, already retrieved and gated upstream)
  4. Windowed recent messages (last N from MessagesState)
  5. Current user query

Applies a configurable token budget so context never grows unbounded.
Uses tiktoken for accurate token counting against provider limits.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
)
from langchain_core.messages.utils import trim_messages

from app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Lazy singleton — loaded on first call to avoid a network fetch at import time.
_tokenizer = None


def _get_tokenizer():
    """Return the tiktoken encoder, initialising it lazily.

    Falls back to None if the BPE data cannot be fetched (e.g. no internet),
    in which case callers use the len(text)//4 heuristic.
    """
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    try:
        import tiktoken
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[ContextEngine] tiktoken unavailable — using len//4 heuristic. Reason: %s", exc
        )
        _tokenizer = False  # sentinel: tried and failed
    return _tokenizer


def _encode_text(text: str) -> int:
    """Return token count for a plain string, using tiktoken or the heuristic."""
    enc = _get_tokenizer()
    if enc:
        return len(enc.encode(text))
    return max(1, len(text) // 4)


def _count_tokens(messages: List[BaseMessage]) -> int:
    """Token-count estimate for a list of messages.

    Accounts for:
    * ``content`` — str or list-of-dicts (multi-part / tool-use blocks)
    * ``tool_calls`` on AIMessage — serialised as JSON so tool-heavy turns
      are not silently undercounted against the context budget.
    """
    total = 0
    for m in messages:
        content = m.content
        if isinstance(content, str):
            total += _encode_text(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += _encode_text(str(part.get("text", "")))

        # Include tool_calls payload (present on AIMessage when tools are invoked)
        if isinstance(m, AIMessage) and m.tool_calls:
            try:
                total += _encode_text(json.dumps(m.tool_calls))
            except Exception:  # noqa: BLE001
                total += _encode_text(str(m.tool_calls))

    return total


def _count_tokens_for_trim(messages: List[BaseMessage]) -> int:
    """Token counter compatible with trim_messages() signature (accepts a list)."""
    return _count_tokens(messages)


# ---------------------------------------------------------------------------
# ContextEngine
# ---------------------------------------------------------------------------


class ContextEngine:
    """Assembles the final message list sent to the LLM each turn."""

    def build(
        self,
        *,
        system_instructions: str,
        conversation_summary: str,
        mem0_memories: str,
        messages: Sequence[BaseMessage],
        rag_citations: Optional[Dict[int, dict]] = None,
    ) -> List[BaseMessage]:
        """Return a fully assembled List[BaseMessage] ready for llm.ainvoke().

        Parameters
        ----------
        system_instructions:
            Static system prompt / persona text.
        conversation_summary:
            Rolling summary of older conversation turns. Empty string if none.
        mem0_memories:
            Pre-retrieved, pre-gated Mem0 long-term memories as formatted text.
        messages:
            Sequence of current conversation messages from MessagesState (including tool calls).
            Any stale SystemMessages are stripped out here.
        rag_citations:
            Optional citation map produced by :func:`format_cited_context`.
            When non-empty and ``settings.RAG_GROUNDING_ENABLED`` is True,
            a citation rules block is injected into the system prompt.
        """
        # --- 1. Build the composite system block ----------------------------
        system_parts: List[str] = [
            system_instructions.strip(),
            (
                "\nIMPORTANT INSTRUCTIONS:\n"
                "- Do NOT output raw internal citation tags like `【retrieve_documents†source=1】`. "
                "Integrate the information naturally into your answer instead.\n"
                "- If the user asks about themselves (their name, preferences, history, goals, etc.), "
                "answer using the <long_term_memory> context block below. "
                "Do NOT call retrieve_documents or search_web for personal user information."
            )
        ]

        if conversation_summary:
            system_parts.append(
                "<conversation_summary>\n"
                "The following is a summary of the older part of this conversation "
                "(older messages have been compacted to save space):\n"
                + conversation_summary.strip()
                + "\n</conversation_summary>"
            )

        if mem0_memories:
            system_parts.append(
                "<long_term_memory>\n"
                "The following are facts remembered about this user from past conversations:\n"
                + mem0_memories.strip()
                + "\n</long_term_memory>"
            )

        # --- 1b. Inject RAG citation grounding rules when documents are present ---
        grounding_enabled = getattr(settings, "RAG_GROUNDING_ENABLED", True)
        if grounding_enabled and rag_citations:
            system_parts.append(
                "<rag_grounding>\n"
                "The retrieved document chunks below were injected into this conversation via "
                "the retrieve_documents tool. Each chunk is identified by a \u3010Doc N\u3011 marker.\n\n"
                "CITATION RULES (follow strictly):\n"
                "1. When your answer uses information from a retrieved document, reference it "
                "inline as \u3010Doc N\u3011 (e.g. \u300caccording to \u3010Doc 1\u3011, the conclusion was...\u300d).\n"
                "2. If the retrieved documents do NOT contain sufficient evidence to answer the "
                "question, you MUST explicitly say: 'This information was not found in the "
                "provided documents.' Do NOT invent an answer.\n"
                "3. Do NOT fabricate citations. Do NOT cite a \u3010Doc N\u3011 that was not retrieved.\n"
                "4. Clearly distinguish between:\n"
                "   (a) information sourced from retrieved documents \u2192 cite with \u3010Doc N\u3011\n"
                "   (b) information from your general training knowledge \u2192 note it as general knowledge\n"
                "   (c) information from web search results \u2192 note the source URL where available\n"
                "</rag_grounding>"
            )

        system_content = "\n\n".join(system_parts)
        system_msg = SystemMessage(content=system_content)

        # --- 2. Strip any stale SystemMessages from history -----------------
        filtered_messages: List[BaseMessage] = [
            m for m in messages if not isinstance(m, SystemMessage)
        ]

        # --- 3 & 5. Window + token budget via trim_messages (tool-call aware) -
        # trim_messages keeps the most-recent messages, always starts on a
        # HumanMessage boundary, and never splits an AIMessage/ToolMessage pair.
        windowed = trim_messages(
            filtered_messages,
            strategy="last",
            token_counter=_count_tokens_for_trim,
            max_tokens=settings.CONTEXT_TOKEN_BUDGET,
            start_on="human",
            include_system=False,
        )

        # --- 4. Assemble the final list ------------------------------------
        final_messages: List[BaseMessage] = [system_msg] + windowed

        logger.debug(
            "[ContextEngine] Context built: system=%d tokens | recent=%d msgs | total=%d tokens",
            _encode_text(system_content),
            len(windowed),
            _count_tokens(final_messages),
        )

        return final_messages


# Module-level singleton
context_engine = ContextEngine()
