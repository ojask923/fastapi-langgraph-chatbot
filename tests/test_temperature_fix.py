"""Verify that temperature=0 is respected and omitting it uses settings.TEMPERATURE.

Tests are pure unit tests — no running server, no real LLM calls.
We patch get_llm to record what temperature it was called with, then call
get_response / stream_response directly and assert the captured value.
"""
import asyncio
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DATABASE_URL", "sqlite:///./chatbot.db")

from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage

# Import after env is set so pydantic-settings picks up SQLite
from app.config import settings
from app.agent.graph import ChatAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent_with_spy():
    """Return a ChatAgent whose build_graph is replaced by a spy.

    We don't need a real graph — we just want to assert what temperature
    value reached build_graph (which is what get_llm would receive).
    """
    agent = ChatAgent.__new__(ChatAgent)
    agent._checkpointer_ready = True
    agent.checkpointer = MagicMock()
    agent._compiled_graphs = {}
    agent._init_lock = asyncio.Lock()
    agent.tools = []

    captured = {}

    def spy_build_graph(provider=None, model=None, temperature=None):
        captured["temperature"] = temperature
        # Return a minimal fake compiled graph
        fake_graph = AsyncMock()
        fake_result = {
            "messages": [AIMessage(content="ok")],
            "rag_citations": {},
        }
        fake_graph.ainvoke = AsyncMock(return_value=fake_result)
        return fake_graph

    agent.build_graph = spy_build_graph
    return agent, captured


# ---------------------------------------------------------------------------
# Test 1 — temperature=0.0 must reach build_graph as 0.0, not 0.7
# ---------------------------------------------------------------------------
def test_temperature_zero_is_respected():
    agent, captured = _make_agent_with_spy()

    async def run():
        # Patch memory_service.add so fire-and-forget doesn't crash
        with patch("app.agent.graph.memory_service.add", new_callable=AsyncMock):
            await agent.get_response(
                message="hello",
                session_id="s1",
                user_id="u1",
                temperature=0.0,
            )

    asyncio.run(run())

    assert captured["temperature"] == 0.0, (
        f"Expected 0.0, got {captured['temperature']} — temperature=0 was treated as falsy"
    )
    print("PASS test_temperature_zero_is_respected: temperature =", captured["temperature"])


# ---------------------------------------------------------------------------
# Test 2 — omitting temperature uses settings.TEMPERATURE
# ---------------------------------------------------------------------------
def test_omitted_temperature_uses_settings():
    agent, captured = _make_agent_with_spy()

    async def run():
        with patch("app.agent.graph.memory_service.add", new_callable=AsyncMock):
            await agent.get_response(
                message="hello",
                session_id="s2",
                user_id="u1",
                # temperature not passed — should default to settings.TEMPERATURE
            )

    asyncio.run(run())

    assert captured["temperature"] == settings.TEMPERATURE, (
        f"Expected settings.TEMPERATURE={settings.TEMPERATURE}, got {captured['temperature']}"
    )
    print(f"PASS test_omitted_temperature_uses_settings: temperature = {captured['temperature']} (settings.TEMPERATURE={settings.TEMPERATURE})")


# ---------------------------------------------------------------------------
# Test 3 — ChatRequest.temperature default is None (not 0.7)
# ---------------------------------------------------------------------------
def test_chat_request_default_is_none():
    from app.api.routes import ChatRequest
    req = ChatRequest(message="hi")
    assert req.temperature is None, (
        f"Expected None, got {req.temperature} — default was not changed from 0.7"
    )
    print("PASS test_chat_request_default_is_none: ChatRequest.temperature =", req.temperature)


# ---------------------------------------------------------------------------
# Test 4 — ChatRequest with temperature=0 passes 0, not settings.TEMPERATURE
# ---------------------------------------------------------------------------
def test_chat_request_zero_preserved():
    from app.api.routes import ChatRequest
    req = ChatRequest(message="hi", temperature=0.0)
    resolved = req.temperature if req.temperature is not None else settings.TEMPERATURE
    assert resolved == 0.0, f"Expected 0.0, got {resolved}"
    print("PASS test_chat_request_zero_preserved: resolved temperature =", resolved)


if __name__ == "__main__":
    test_chat_request_default_is_none()
    test_chat_request_zero_preserved()
    test_temperature_zero_is_respected()
    test_omitted_temperature_uses_settings()
    print()
    print("ALL TESTS PASSED")