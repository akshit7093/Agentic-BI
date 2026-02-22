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

def _build_tool_registry(engine: MMMEngine) -> ToolRegistry:
    """Register all built-in tools from each tool module."""
    from ..tools.data_tools import build_data_tools
    from ..tools.mmm_tools import build_mmm_tools
    from ..tools.custom_tools import build_meta_tools
    from ..tools.analytics_tools import build_analytics_tools
    from ..tools.viz_tools import build_viz_tools

    registry = ToolRegistry()

    # Data tools (14 tools)
    registry.register_many(build_data_tools(engine))

    # MMM tools (9 tools)
    registry.register_many(build_mmm_tools(engine))

    # Advanced analytics tools (12 tools) — ML, PCA, VIF, Granger, etc.
    registry.register_many(build_analytics_tools(engine))

    # Visualization tools (6 tools) — charts, heatmaps, scatter matrices
    registry.register_many(build_viz_tools(engine))

    # Meta / custom tools (9 tools) — needs registry for self-extension
    registry.register_many(build_meta_tools(registry, engine))

    logger.info(f"Tool registry: {len(registry)} tools registered")
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
    llm = _init_llm(llm_endpoint, temperature, max_tokens)

    # 2. Tool registry
    registry = _build_tool_registry(engine)
    lc_tools = registry.get_all()

    # 3. Bind tools to the LLM (for the agent node)
    llm_with_tools = llm.bind_tools(lc_tools)

    # 4. System message factory (curried with engine)
    def system_message_fn(state: AgentState) -> SystemMessage:
        return build_system_message(state, engine)

    # 5. Build nodes
    planner_node     = build_planner_node(llm, system_message_fn, registry=registry)
    agent_node       = build_agent_node(llm_with_tools, system_message_fn)
    tool_node        = build_tool_node(registry, console=console)
    reflection_node  = build_reflection_node(llm)
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
