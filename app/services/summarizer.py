"""Summarizer: rolling conversation compactor for LangGraph AgentState.

When the total number of stored messages exceeds SUMMARY_THRESHOLD, the
oldest messages (those outside the RECENT_MESSAGES_WINDOW) are summarised
into a compact text block (~300-500 tokens).  The summary is stored in
``AgentState["summary"]`` and the old messages are removed from state,
keeping only the recent window alive.

The summary is *replaced* each compaction cycle, not appended to, so it
cannot grow indefinitely.

Design notes
------------
* The summarizer reuses the same provider/model already selected for the
  conversation so no extra API keys are needed.
* It is called from within the LangGraph ``summarize`` node which runs
  asynchronously after each ``agent`` node invocation when needed.
* It never mutates the checkpointer directly; it returns updated state
  dicts that LangGraph's normal checkpoint mechanism persists.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    RemoveMessage,
)

from app.config import settings

logger = logging.getLogger(__name__)

_SUMMARY_SYSTEM_PROMPT = (
    "You are a concise conversation summarizer. "
    "Your job is to summarize the provided conversation messages into a "
    "compact paragraph (300-500 words maximum). "
    "Preserve all important facts, decisions, preferences, and context. "
    "Do NOT include greetings, filler, or pleasantries. "
    "Write in third-person narrative style: 'The user said... The assistant explained...'. "
    "Output ONLY the summary text, no preamble."
)


def _format_messages_for_summary(messages: List[BaseMessage]) -> str:
    """Convert a list of messages into a plain-text transcript for summarisation."""
    lines: List[str] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = "User"
        elif isinstance(m, AIMessage):
            role = "Assistant"
        elif isinstance(m, SystemMessage):
            continue  # skip system messages
        else:
            role = type(m).__name__

        content = m.content
        if isinstance(content, list):
            # Handle multi-part content (e.g., tool use)
            content = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


class ConversationSummarizer:
    """Compacts old conversation messages into a rolling summary."""

    async def should_summarize(self, messages: List[BaseMessage]) -> bool:
        """Return True when the message count exceeds SUMMARY_THRESHOLD."""
        # Count only non-system messages
        user_ai_count = sum(
            1 for m in messages if not isinstance(m, SystemMessage)
        )
        return user_ai_count > settings.SUMMARY_THRESHOLD

    async def compact(
        self,
        messages: List[BaseMessage],
        existing_summary: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Summarise the oldest messages and return updated state fields.

        Returns a dict with:
          - "summary": the new compact summary string
          - "messages": trimmed message list (only the recent window kept)
        """
        provider = (provider or settings.DEFAULT_PROVIDER).lower()
        window = settings.RECENT_MESSAGES_WINDOW

        # Separate non-system messages
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]

        if len(non_system) <= window:
            # Nothing old enough to compact yet
            return {"summary": existing_summary, "messages": messages}

        # Messages to summarize: everything except the recent window
        to_summarize = non_system[:-window]
        to_keep = non_system[-window:]

        # Build the text to summarize, optionally incorporating the prior summary
        transcript_parts: List[str] = []
        if existing_summary:
            transcript_parts.append(
                f"[Prior summary]\n{existing_summary}\n\n[New messages to incorporate]"
            )
        transcript_parts.append(_format_messages_for_summary(to_summarize))
        transcript_text = "\n".join(transcript_parts)

        # Call LLM for summarization
        new_summary = existing_summary  # fallback to old if call fails
        try:
            from app.services.llm_factory import get_llm

            summary_llm = get_llm(provider=provider, model=model, temperature=0.3)
            summary_input = [
                SystemMessage(content=_SUMMARY_SYSTEM_PROMPT),
                HumanMessage(content=transcript_text),
            ]
            response = await summary_llm.ainvoke(summary_input, config={"tags": ["internal"]})
            new_summary_text = response.content
            if isinstance(new_summary_text, list):
                new_summary_text = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in new_summary_text
                )
            new_summary = new_summary_text.strip()
            logger.info(
                "[Summarizer] Compacted %d messages into summary (%d chars)",
                len(to_summarize),
                len(new_summary),
            )
        except Exception as exc:
            logger.warning("[Summarizer] LLM summarization failed: %s", exc)
            # Keep the old summary; still trim the messages so state doesn't balloon
            if not new_summary:
                new_summary = f"[Summary unavailable — {len(to_summarize)} older messages compacted]"

        return {
            "summary": new_summary,
            "messages": [RemoveMessage(id=m.id) for m in to_summarize if m.id],
        }


# Module-level singleton
summarizer = ConversationSummarizer()
