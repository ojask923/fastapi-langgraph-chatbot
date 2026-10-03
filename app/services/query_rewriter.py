import logging
import re
from typing import Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

from app.services.llm_factory import get_llm

logger = logging.getLogger(__name__)

# Pronouns that signal a query might refer to earlier conversation context.
# Presence of any of these words makes the rewrite worthwhile; absence means
# the query is almost certainly already standalone and we can skip the LLM call.
_REFERENCE_PRONOUNS = re.compile(
    r"\b(it|its|this|that|they|them|their|those|these|he|she|him|her)\b",
    re.IGNORECASE,
)


class QueryRewriter:
    async def rewrite(
        self,
        query: str,
        messages: Sequence[BaseMessage],
        provider: str,
        model: str,
    ) -> str:
        """Rewrite a search query to be standalone based on conversation history.

        Short-circuits (returns the original query unchanged) when:
        - There is no meaningful conversation history, OR
        - The query contains no reference pronouns *and* there is no prior
          assistant turn (i.e. it's effectively a first-turn question).

        This avoids a costly LLM round-trip for queries that are already
        self-contained.
        """

        # Format the history into a string.
        # We only need the last few messages for context to keep it fast.
        history_text = ""
        recent_messages = messages[-10:] if len(messages) > 10 else messages

        has_prior_assistant_turn = False
        for m in recent_messages:
            if isinstance(m, HumanMessage):
                content = m.content
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in content
                    )
                history_text += f"User: {content}\n"
            elif isinstance(m, AIMessage) and m.content:
                history_text += f"Assistant: {m.content}\n"
                has_prior_assistant_turn = True

        # Early exit 1: no history at all.
        if not history_text.strip():
            return query

        # Early exit 2: query has no reference pronouns and there has been no
        # prior assistant reply — the query is almost certainly standalone.
        if not has_prior_assistant_turn and not _REFERENCE_PRONOUNS.search(query):
            logger.debug(
                "[QueryRewriter] Skipping rewrite (no pronouns, first turn): '%s'", query
            )
            return query

        system_prompt = """You are an expert search query rewriter. 
Your task is to analyze the conversation history and a given search query.
If the query is already standalone and clear, output it exactly as is.
If the query relies on context (e.g. uses pronouns like 'it', 'they', 'this', 'that', or refers to previous topics implicitly), rewrite it into a single standalone query that captures the full context.
Do not answer the query. Do not add conversational filler.
Preserve important technical terminology, names, numbers, constraints, and user intent.
Return ONLY the final query text."""

        user_prompt = f"Conversation History:\n{history_text}\n\nCurrent Search Query: {query}"

        try:
            # We don't bind tools for the rewriter and use 0.0 temperature for consistency
            llm = get_llm(provider, model, temperature=0.0)
            response = await llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ],
                config={"tags": ["query_rewriter"]},
            )

            # Normalize content — some providers (e.g. Anthropic) return a list
            # of content blocks instead of a plain string.
            content = response.content
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                )
            rewritten_query = content.strip()

            logger.info(
                "[QueryRewriter] Original: '%s' -> Rewritten: '%s'", query, rewritten_query
            )
            return rewritten_query
        except Exception as exc:
            logger.warning("[QueryRewriter] Failed to rewrite query: %s", exc)
            return query


query_rewriter = QueryRewriter()
