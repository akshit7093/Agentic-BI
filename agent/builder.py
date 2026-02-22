# =============================================================
# agent/builder.py — Assemble the 5-node LangGraph cognitive agent
# =============================================================
"""
Graph topology:
    START → planner → agent ⇄ tools → reflect ⇄ agent/planner → quality_gate → planner/END
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Optional, Tuple

from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from ..core.mmm_engine import MMMEngine
from ..tools.registry import ToolRegistry
from ..workflows.state import AgentState, initial_state
from ..workflows.nodes import (
    build_planner_node,
    build_agent_node,
    build_tool_node,
    build_reflection_node,
    build_quality_gate_node,
    planner_router,
    agent_router,
    reflect_router,
    gate_router,
)
from ..agent.prompts import build_system_message

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# TOOL REGISTRY BUILDER
# ─────────────────────────────────────────────

def _build_tool_registry(engine: MMMEngine, llm=None) -> ToolRegistry:
    """
    Build a partitioned tool registry for the supervisor agent.

    The Databricks LLM endpoint caps at 32 tools per call.
    We bypass this via multi-agent partitioning:
      - Supervisor sees: 3 dispatch tools + 9 meta tools = 12 tools ✅
      - Each sub-agent holds its own tool partition (<= 32 tools) internally.
    """
    from ..tools.custom_tools import build_meta_tools
    from ..tools.sub_agents import build_dispatch_tools

    registry = ToolRegistry()

    # 3 dispatch tools — each routes to a specialized sub-agent
    # (Data Agent: 14+ tools, Analytics Agent: 12 tools, MMM/Viz Agent: 15 tools)
    if llm is not None:
        dispatch_tools = build_dispatch_tools(llm, engine)
        registry.register_many(dispatch_tools)
        logger.info(f"[Registry] {len(dispatch_tools)} dispatch tools registered")
    else:
        logger.warning("[Registry] No LLM provided — dispatch tools skipped")

    # Meta tools (9 tools): ask_user, add_analysis_note, create_custom_tool, etc.
    # These stay on the supervisor so it can interact with the user directly.
    registry.register_many(build_meta_tools(registry, engine))

    total = len(registry)
    logger.info(f"[Registry] Supervisor tool count: {total} (limit=32)")
    assert total <= 32, (
        f"Supervisor exceeds 32-tool limit: {total} tools. "
        "Check dispatch + meta tool counts."
    )
    return registry


# ─────────────────────────────────────────────
# LLM INITIALIZATION
# ─────────────────────────────────────────────

def _init_llm(endpoint: str, temperature: float, max_tokens: int):
    """Initialize the Databricks-hosted LLM."""
    from langchain_databricks import ChatDatabricks

    return ChatDatabricks(
        endpoint=endpoint,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ─────────────────────────────────────────────
# GRAPH BUILDER
# ─────────────────────────────────────────────

def build_agent(
    engine: MMMEngine,
    llm_endpoint: str = "databricks-llama-4-maverick",
    temperature: float = 0.2,
    max_tokens: int = 4096,
    console=None,
) -> tuple:
    """
    Build the 5-node cognitive agent graph.

    Returns
    -------
    (compiled_graph, tool_registry)
    """
    # 1. LLM
    logger.info(f"[Builder] Initializing LLM: endpoint={llm_endpoint}, temp={temperature}")
    llm = _init_llm(llm_endpoint, temperature, max_tokens)

    # 2. Tool registry — partitioned so supervisor sees ≤ 12 tools
    #    (3 dispatch tools + 9 meta tools)
    logger.info("[Builder] Building partitioned tool registry...")
    registry = _build_tool_registry(engine, llm=llm)
    lc_tools = registry.get_all()
    logger.info(f"[Builder] Supervisor tool count: {len(lc_tools)}")

    # 3. Bind tools to the supervisor LLM (≤ 32 guaranteed by assertion in _build_tool_registry)
    llm_with_tools = llm.bind_tools(lc_tools)
    logger.info(f"[Builder] LLM bound to {len(lc_tools)} supervisor tools")

    # 4. System message factory (curried with engine)
    def system_message_fn(state: AgentState) -> SystemMessage:
        return build_system_message(state, engine)

    # 5. Build nodes
    logger.info("[Builder] Building graph nodes...")
    planner_node      = build_planner_node(llm, system_message_fn, registry=registry)
    agent_node        = build_agent_node(llm_with_tools, system_message_fn)
    tool_node         = build_tool_node(registry, console=console)
    reflection_node   = build_reflection_node(llm)
    quality_gate_node = build_quality_gate_node(console=console)

    # 6. Assemble graph
    workflow = StateGraph(AgentState)

    workflow.add_node("planner",      planner_node)
    workflow.add_node("agent",        agent_node)
    workflow.add_node("tools",        tool_node)
    workflow.add_node("reflect",      reflection_node)
    workflow.add_node("quality_gate", quality_gate_node)

    # Entry point
    workflow.set_entry_point("planner")

    # Edges
    workflow.add_conditional_edges(
        "planner",
        planner_router,
        {"agent": "agent", "end": END},
    )

    workflow.add_conditional_edges(
        "agent",
        agent_router,
        {
            "tools":   "tools",
            "reflect": "reflect",
            "agent":   "agent",     # self-retry on recoverable error
            "end":     END,
        },
    )

    workflow.add_edge("tools", "agent")

    workflow.add_conditional_edges(
        "reflect",
        reflect_router,
        {
            "agent":        "agent",         # continue working
            "planner":      "planner",       # re-plan (retry)
            "quality_gate": "quality_gate",  # phase complete
            "end":          END,
        },
    )

    workflow.add_conditional_edges(
        "quality_gate",
        gate_router,
        {
            "planner": "planner",   # next phase
            "end":     END,
        },
    )

    # 7. Compile with checkpointer
    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)

    logger.info(
        "Cognitive agent graph compiled: "
        "planner → agent ⇄ tools → reflect → quality_gate → planner/END"
    )

    return graph, registry
