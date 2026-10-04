"""Verify the _merge_citations reducer.

Test 1 – parallel writes no longer raise InvalidUpdateError:
  Two nodes writing rag_citations in the same superstep are merged by the
  reducer instead of colliding.

Test 2 – None sentinel resets citations at turn-start:
  Passing rag_citations=None as input clears the accumulated state, so a
  turn that does no retrieval does not bleed citations from the prior turn.
"""
import asyncio
from typing import Annotated, Any, List

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# Import the reducer directly from the module under test.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.agent.graph import _merge_citations


# ---------------------------------------------------------------------------
# Minimal state using the real reducer
# ---------------------------------------------------------------------------
class TestState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    rag_citations: Annotated[dict, _merge_citations]


# ---------------------------------------------------------------------------
# Test 1 – parallel writes are merged (not raising InvalidUpdateError)
# ---------------------------------------------------------------------------
def test_parallel_writes_are_merged():
    """Two fan-out nodes each writing rag_citations must be merged, not crashed."""

    async def node_a(state: TestState):
        return {"rag_citations": {"1": {"source": "a.pdf", "text": "chunk A"}}}

    async def node_b(state: TestState):
        return {"rag_citations": {"2": {"source": "b.pdf", "text": "chunk B"}}}

    async def join(state: TestState):
        return {}

    wf = StateGraph(TestState)
    wf.add_node("a", node_a)
    wf.add_node("b", node_b)
    wf.add_node("join", join)
    wf.add_edge(START, "a")
    wf.add_edge(START, "b")
    wf.add_edge("a", "join")
    wf.add_edge("b", "join")
    wf.add_edge("join", END)
    graph = wf.compile(checkpointer=MemorySaver())

    result = asyncio.run(graph.ainvoke(
        {"messages": [HumanMessage(content="hi")], "rag_citations": None},
        config={"configurable": {"thread_id": "t1"}},
    ))

    assert "1" in result["rag_citations"], "Key from node_a missing"
    assert "2" in result["rag_citations"], "Key from node_b missing"
    print("PASS test_parallel_writes_are_merged:", result["rag_citations"])


# ---------------------------------------------------------------------------
# Test 2 – None sentinel clears previous citations
# ---------------------------------------------------------------------------
def test_none_sentinel_resets_citations():
    """rag_citations=None in input_state must wipe citations from prior turns."""

    async def write_citations(state: TestState):
        return {"rag_citations": {"1": {"source": "doc.pdf", "text": "excerpt"}}}

    async def noop(state: TestState):
        return {}

    wf = StateGraph(TestState)
    wf.add_node("write", write_citations)
    wf.add_node("noop", noop)
    wf.add_edge(START, "write")
    wf.add_edge("write", END)

    wf2 = StateGraph(TestState)
    wf2.add_node("noop", noop)
    wf2.add_edge(START, "noop")
    wf2.add_edge("noop", END)

    checkpointer = MemorySaver()
    g1 = wf.compile(checkpointer=checkpointer)
    g2 = wf2.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "t2"}}

    # Turn 1: write some citations
    r1 = asyncio.run(g1.ainvoke(
        {"messages": [HumanMessage(content="q1")], "rag_citations": None},
        config=config,
    ))
    assert r1["rag_citations"] == {"1": {"source": "doc.pdf", "text": "excerpt"}}, \
        f"Turn 1 citations wrong: {r1['rag_citations']}"
    print("PASS turn 1 citations:", r1["rag_citations"])

    # Turn 2: reset via None sentinel — no retrieval this turn
    r2 = asyncio.run(g2.ainvoke(
        {"messages": [HumanMessage(content="q2")], "rag_citations": None},
        config=config,
    ))
    assert r2["rag_citations"] == {}, \
        f"FAIL: citations not cleared on turn 2: {r2['rag_citations']}"
    print("PASS turn 2 citations cleared:", r2["rag_citations"])


# ---------------------------------------------------------------------------
# Unit-test the reducer function directly
# ---------------------------------------------------------------------------
def test_merge_citations_unit():
    assert _merge_citations({"1": "a"}, None) == {}, "None should reset"
    assert _merge_citations(None, None) == {}, "None+None should reset"
    assert _merge_citations({"1": "a"}, {"2": "b"}) == {"1": "a", "2": "b"}, "merge"
    assert _merge_citations(None, {"1": "a"}) == {"1": "a"}, "left=None baseline"
    assert _merge_citations({"1": "a"}, {}) == {"1": "a"}, "empty right keeps left"
    print("PASS test_merge_citations_unit")


if __name__ == "__main__":
    test_merge_citations_unit()
    test_parallel_writes_are_merged()
    test_none_sentinel_resets_citations()
    print()
    print("ALL TESTS PASSED")