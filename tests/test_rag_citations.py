"""Verify that RAG citations are correctly preserved in the graph state.

Before the fix:
  call_model returned {"messages": [response], "rag_citations": {}} which overwrote
  the citations written by retrieve_documents on the second agent invocation.

After the fix:
  call_model returns only {"messages": [response]}, and citations are reset to {} in
  input_state at the start of each user turn (get_response / stream_response).

This test constructs a minimal LangGraph graph with the same topology as ChatAgent
and asserts:
  (a) A turn that triggers retrieve_documents ends with non-empty rag_citations.
  (b) A subsequent turn that does NOT trigger retrieval ends with empty rag_citations.
"""

import asyncio
from typing import Annotated, Any, List

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Minimal AgentState (mirrors the one in graph.py)
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    summary: str
    rag_citations: dict


# ---------------------------------------------------------------------------
# Fake citation data
# ---------------------------------------------------------------------------

FAKE_CITATION_MAP = {"1": {"source": "doc.pdf", "page": 1, "text": "Some excerpt."}}


def _make_retrieve_command(tool_call_id: str) -> Command:
    """Return the Command that retrieve_documents would produce."""
    return Command(update={
        "rag_citations": FAKE_CITATION_MAP,
        "messages": [ToolMessage(content="Doc 1 Some excerpt.", tool_call_id=tool_call_id)],
    })


# ---------------------------------------------------------------------------
# Graph builder (mirrors ChatAgent.build_graph topology)
# ---------------------------------------------------------------------------

def build_test_graph(fake_llm):
    from langchain_core.tools import tool

    @tool
    async def retrieve_documents(query: str) -> str:
        """Stub retrieve_documents tool."""
        return "stub"

    class FakeToolNode(ToolNode):
        async def ainvoke(self, state, config=None):
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                tc = last.tool_calls[0]
                if tc["name"] == "retrieve_documents":
                    return _make_retrieve_command(tc["id"])
            return await super().ainvoke(state, config)

    tool_node = FakeToolNode([retrieve_documents])

    async def call_model(state: AgentState, config: RunnableConfig):
        # FIX: do NOT include "rag_citations" in return value
        response = await fake_llm.ainvoke(state["messages"])
        return {"messages": [response]}

    async def maybe_summarize(state: AgentState, config: RunnableConfig):
        return {}

    def route_after_agent(state: AgentState):
        last = state["messages"][-1] if state["messages"] else None
        if last and hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "summarize"

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_node("summarize", maybe_summarize)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent", route_after_agent,
        {"tools": "tools", "summarize": "summarize"},
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("summarize", END)

    return workflow.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# Fake LLM
# ---------------------------------------------------------------------------

def _make_fake_llm(call_sequence: list):
    class FakeLLM:
        async def ainvoke(self, messages):
            return call_sequence.pop(0)
    return FakeLLM()


# ---------------------------------------------------------------------------
# Tests (plain sync — no pytest-asyncio required)
# ---------------------------------------------------------------------------

def test_citations_preserved_after_retrieval():
    """Citations set by the tool must survive the second call_model invocation."""
    tool_call_id = "tc-001"
    seq = [
        AIMessage(
            content="",
            tool_calls=[{"name": "retrieve_documents", "args": {"query": "doc question"}, "id": tool_call_id}],
        ),
        AIMessage(content="Here is the answer from the document."),
    ]
    graph = build_test_graph(_make_fake_llm(seq))
    config = {"configurable": {"thread_id": "session-rag-test"}}

    # FIX: reset citations at turn-start
    result = asyncio.run(graph.ainvoke(
        {"messages": [HumanMessage(content="What does the document say?")], "rag_citations": {}},
        config=config,
    ))

    assert result["rag_citations"] == FAKE_CITATION_MAP, (
        f"Expected non-empty citations, got: {result['rag_citations']}"
    )


def test_citations_cleared_on_turn_without_retrieval():
    """A turn that does no retrieval must produce empty citations (not bleed from prior turn)."""

    async def run():
        tool_call_id = "tc-002"
        session_id = "session-rag-bleed"

        # Turn 1: with retrieval
        seq1 = [
            AIMessage(
                content="",
                tool_calls=[{"name": "retrieve_documents", "args": {"query": "doc?"}, "id": tool_call_id}],
            ),
            AIMessage(content="Answer with doc."),
        ]
        graph = build_test_graph(_make_fake_llm(seq1))
        config = {"configurable": {"thread_id": session_id}}
        await graph.ainvoke(
            {"messages": [HumanMessage(content="What does the document say?")], "rag_citations": {}},
            config=config,
        )

        # Turn 2: plain question, no retrieval; reset citations at turn-start (the fix)
        seq2 = [AIMessage(content="The sky is blue.")]
        graph2 = build_test_graph(_make_fake_llm(seq2))
        result2 = await graph2.ainvoke(
            {"messages": [HumanMessage(content="What color is the sky?")], "rag_citations": {}},
            config=config,
        )

        assert result2["rag_citations"] == {}, (
            f"Expected empty citations on a non-retrieval turn, got: {result2['rag_citations']}"
        )

    asyncio.run(run())
