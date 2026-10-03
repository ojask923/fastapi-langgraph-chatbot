"""Agent tools for the chatbot."""

import math
import ast
import asyncio
import operator
import datetime
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig
from typing import Annotated


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.
    Examples: '2 + 2', 'sqrt(144) * 3', 'sin(radians(90))', '2 ** 8'.
    """
    allowed_names = {
        k: v for k, v in math.__dict__.items() if not k.startswith("__")
    }
    allowed_names.update({
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
    })
    try:
        clean_expr = expression.strip()
        
        # Supported operators
        bin_ops = {
            ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
            ast.Pow: operator.pow, ast.Mod: operator.mod
        }
        unary_ops = {
            ast.UAdd: operator.pos, ast.USub: operator.neg
        }

        def eval_node(node):
            if isinstance(node, ast.Expression):
                return eval_node(node.body)
            elif isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                left = eval_node(node.left)
                right = eval_node(node.right)
                return bin_ops[type(node.op)](left, right)
            elif isinstance(node, ast.UnaryOp):
                operand = eval_node(node.operand)
                return unary_ops[type(node.op)](operand)
            elif isinstance(node, ast.Call):
                func_name = node.func.id
                if func_name not in allowed_names:
                    raise ValueError(f"Function {func_name} not allowed")
                args = [eval_node(arg) for arg in node.args]
                return allowed_names[func_name](*args)
            elif isinstance(node, ast.Name):
                if node.id in allowed_names:
                    return allowed_names[node.id]
                raise ValueError(f"Variable {node.id} not allowed")
            else:
                raise TypeError(f"Unsupported operation: {type(node).__name__}")

        parsed_expr = ast.parse(clean_expr, mode='eval')
        result = eval_node(parsed_expr)
        return f"Result of `{expression}` = {result}"
    except Exception as e:
        return f"Error evaluating expression '{expression}': {str(e)}"


@tool
def get_current_time(timezone: str = "local") -> str:
    """Get the current date, time, and day of the week."""
    now = datetime.datetime.now()
    return f"Current date and time: {now.strftime('%A, %Y-%m-%d %H:%M:%S')} ({timezone} time)"


@tool
def search_web(query: str) -> str:
    """Search the web for up-to-date information, news, or general knowledge using DuckDuckGo.
    Do NOT use this for questions about the user's personal facts, name, or preferences —
    those are already available in the system context from long-term memory.
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results:
                return f"No search results found for query: '{query}'"
            formatted = []
            for i, r in enumerate(results, 1):
                formatted.append(f"{i}. [{r.get('title', 'No Title')}]({r.get('href', '')}): {r.get('body', '')}")
            return "\n\n".join(formatted)
    except Exception as e:
        return f"Web search could not be completed: {str(e)}. DO NOT retry the web search, answer using existing knowledge or context."


@tool
async def retrieve_documents(
    query: str,
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    runtime: ToolRuntime,
) -> Command:
    """Search the uploaded knowledge base for relevant document snippets.

    Use this tool ONLY when the user asks about content from documents they have
    uploaded (e.g. PDFs, policy files, reports). Do NOT use this tool for:
    - Questions about the user's personal facts, name, preferences, or history
      (those are already provided in the system context via long-term memory).
    - General knowledge or current events (use search_web instead).

    Returns a 【Doc N】-marked context block with source attribution. Citation
    metadata is written into ``rag_citations`` in AgentState via a
    ``Command(update=...)`` so it is properly persisted by LangGraph and
    available to the API layer in the response.
    """
    tool_call_id = runtime.tool_call_id

    try:
        # Deferred imports — avoids loading torch/flashrank/HuggingFace at startup.
        # Both rag_service and query_rewriter are module-level singletons; importing
        # them here is cheap after the first call (Python caches sys.modules).
        from app.services.rag_service import rag_service, format_cited_context  # noqa: PLC0415
        from app.services.query_rewriter import query_rewriter  # noqa: PLC0415

        configurable = config.get("configurable", {})
        provider = configurable.get("provider", "openai")
        model = configurable.get("model", "gpt-4o-mini")
        user_id = configurable.get("user_id", "")

        rewritten_query = await query_rewriter.rewrite(
            query=query,
            messages=state.get("messages", []),
            provider=provider,
            model=model
        )

        # retrieve_structured is synchronous (Qdrant + HuggingFace).
        # Running it in a thread pool keeps the async event loop unblocked.
        chunks = await asyncio.to_thread(
            rag_service.retrieve_structured,
            rewritten_query,
            user_id=user_id or None,
        )

        if not chunks:
            return Command(update={
                "rag_citations": {},
                "messages": [ToolMessage(
                    content="No relevant documents found in the vector store.",
                    tool_call_id=tool_call_id,
                )],
            })

        context_block, citation_map = format_cited_context(chunks)

        # Return a Command so LangGraph writes citation_map into the real
        # persisted AgentState.  InjectedState is a read-only snapshot and
        # direct mutation of it is silently ignored by the graph runtime.
        # ToolNode requires a ToolMessage with the matching tool_call_id to be
        # present in Command.update["messages"] — it carries the context block
        # that the LLM sees, while rag_citations is persisted in state.
        return Command(update={
            "rag_citations": citation_map,
            "messages": [ToolMessage(
                content=context_block,
                tool_call_id=tool_call_id,
            )],
        })
    except Exception as e:
        return Command(update={
            "rag_citations": {},
            "messages": [ToolMessage(
                content=(
                    f"Error retrieving documents: {str(e)}. "
                    "DO NOT retry document retrieval, answer using existing knowledge or context."
                ),
                tool_call_id=tool_call_id,
            )],
        })


def get_available_tools():
    """Return the list of active tools for the agent."""
    return [calculate, get_current_time, search_web, retrieve_documents]
