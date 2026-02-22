# =============================================================
# tools/sub_agents.py — Specialized sub-agent dispatch tools
#
# Bypasses the Databricks 32-tool-per-call limit by partitioning
# all tools into 3 specialized sub-agents, each invoked as a
# single tool from the supervisor LLM.
#
# Pattern: ReAct generate_code → validator → executor loop
#   (per https://langchain-ai.github.io/langgraph/ canonical pattern)
# =============================================================

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
)
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages

from ..core.mmm_engine import MMMEngine

logger = logging.getLogger(__name__)

# ─── Max retries for the validator loop ─────────────────────────
_MAX_SUB_STEPS = 15
_MAX_RETRIES   = 7


# ─── Shared sub-agent state ────────────────────────────────────

class _SubState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    retries: int


# ─── Input schemas ─────────────────────────────────────────────

class CallDataAgentInput(BaseModel):
    task: str = Field(
        description=(
            "Natural-language instruction for the Data Agent. Examples: "
            "'Load the CSV at /path/to/data.csv', 'Show me the column statistics', "
            "'How many null values are in each column?', 'Aggregate sales by week'. "
            "The agent can run arbitrary Python/Pandas code and SQL queries."
        )
    )


class CallAnalyticsAgentInput(BaseModel):
    target_col: str = Field(description="Target / KPI column name to analyse")
    task: str = Field(
        description=(
            "Analysis instruction for the Analytics Agent. Examples: "
            "'Find the best features to predict totalPrice using RF, mutual info, and VIF', "
            "'Run PCA on all numeric columns', "
            "'Perform stepwise feature selection', "
            "'Cross-validate OLS vs Ridge vs Random Forest'. "
            "The agent will run multiple methods and synthesize a consensus result."
        )
    )
    feature_cols: Optional[str] = Field(
        default=None,
        description="Comma-separated candidate feature columns (leave empty to use all numeric)"
    )


class CallMMMAgentInput(BaseModel):
    task: str = Field(
        description=(
            "MMM/visualization instruction. Examples: "
            "'Run Bayesian MMM with kpi=revenue and channels=tv_spend,digital_spend', "
            "'Generate a correlation heatmap for all numeric columns', "
            "'Optimize budget of $1M across channels tv,digital,ooh', "
            "'Compare scenario A=1M vs scenario B=2M budget'. "
        )
    )
    kpi_col: Optional[str] = Field(default=None, description="KPI column name (if known)")
    channel_cols: Optional[str] = Field(
        default=None,
        description="Comma-separated channel column names (if known)"
    )


# ─── Core ReAct sub-agent runner ───────────────────────────────

def _run_react_subagent(
    llm,
    tools: List,
    system_prompt: str,
    user_task: str,
    agent_name: str,
) -> str:
    """
    Run a minimal ReAct loop:
      generate_code → should_continue? → tool_executor → generate_code
                                       ↓ (text only)
                                      END

    Includes a validator: if executor returns an error, re-prompt
    with the error message (up to _MAX_RETRIES times).

    Returns the final text response from the LLM as a string.
    """
    logger.info(f"[{agent_name}] Starting sub-agent for task: {user_task[:80]}")

    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def generate_code(state: _SubState) -> Dict:
        retries = state.get("retries", 0)
        if retries >= _MAX_RETRIES:
            logger.warning(f"[{agent_name}] Max retries ({_MAX_RETRIES}) reached")
            stop_msg = AIMessage(content=f"⚠ Max retries reached. Last state: {state['messages'][-1].content[:300]}")
            return {"messages": [stop_msg], "retries": retries}

        sys = SystemMessage(content=system_prompt)
        history = [sys] + list(state["messages"])
        logger.debug(f"[{agent_name}] generate_code call #{retries+1}, messages={len(history)}")

        try:
            response = llm_with_tools.invoke(history)
            tool_calls = getattr(response, "tool_calls", None)
            logger.info(
                f"[{agent_name}] LLM responded: "
                f"{'tool_calls=' + str(len(tool_calls)) if tool_calls else 'text response'}"
            )
            return {"messages": [response], "retries": retries}
        except Exception as exc:
            logger.error(f"[{agent_name}] LLM invoke failed: {exc}")
            err = AIMessage(content=f"⚠ LLM error: {exc}")
            return {"messages": [err], "retries": retries + 1}

    def execute_tools(state: _SubState) -> Dict:
        """Run tool calls and catch errors, re-inject as validation feedback."""
        logger.debug(f"[{agent_name}] Executing tool calls")
        try:
            result = tool_node.invoke(state)
            # Check if any tool result is an error string
            new_msgs = result.get("messages", [])
            for msg in new_msgs:
                if isinstance(msg, ToolMessage) and "Error" in str(msg.content):
                    logger.warning(f"[{agent_name}] Tool error detected: {str(msg.content)[:200]}")
            return {**result, "retries": state.get("retries", 0) + 1}
        except Exception as exc:
            logger.error(f"[{agent_name}] Tool execution failed: {exc}")
            err_msg = ToolMessage(
                content=f"Tool execution error: {exc}. Please try a different approach.",
                tool_call_id="error",
            )
            return {"messages": [err_msg], "retries": state.get("retries", 0) + 1}

    def should_continue(state: _SubState) -> str:
        last = state["messages"][-1]
        retries = state.get("retries", 0)

        if retries >= _MAX_RETRIES:
            logger.info(f"[{agent_name}] Routing → END (max retries)")
            return "end"

        has_tool_calls = bool(getattr(last, "tool_calls", None))
        route = "continue" if has_tool_calls else "end"
        logger.debug(f"[{agent_name}] Routing → {route}")
        return route

    # Build the mini-graph
    workflow = StateGraph(_SubState)
    workflow.add_node("generate_code", generate_code)
    workflow.add_node("python_executor", execute_tools)
    workflow.set_entry_point("generate_code")
    workflow.add_conditional_edges(
        "generate_code",
        should_continue,
        {"continue": "python_executor", "end": END},
    )
    workflow.add_edge("python_executor", "generate_code")

    graph = workflow.compile()

    # Run the graph
    init_state: _SubState = {
        "messages": [HumanMessage(content=user_task)],
        "retries": 0,
    }

    step = 0
    final_response = f"[{agent_name}] No response generated."
    for event in graph.stream(init_state):
        step += 1
        if step > _MAX_SUB_STEPS:
            logger.warning(f"[{agent_name}] Step limit ({_MAX_SUB_STEPS}) reached")
            break
        for node_name, node_data in event.items():
            msgs = node_data.get("messages", [])
            if msgs:
                last = msgs[-1]
                if isinstance(last, AIMessage) and last.content:
                    if not getattr(last, "tool_calls", None):
                        final_response = str(last.content)
                        logger.info(f"[{agent_name}] Final response captured ({len(final_response)} chars)")

    logger.info(f"[{agent_name}] Completed in {step} steps")
    return final_response


# ─────────────────────────────────────────────
# SUB-AGENT 1: DATA AGENT
# Tools: PythonAstREPL + DuckDB + data tools
# ─────────────────────────────────────────────

_DATA_AGENT_SYSTEM = """You are a Data Agent specialized in loading, inspecting, and querying data.

You have access to:
- `python_repl_ast`: Run any Python/Pandas code. The variable `df` contains the loaded dataframe.
- `duckdb_query`: Run SQL queries on the data using DuckDB (faster for large files).
- Pre-built data tools: load_data, get_data_status, inspect_data, column_stats, top_values,
  sample_rows, filter_agg, get_correlation, detect_outliers, execute_query, clean_data,
  add_time_features, aggregate_weekly, get_adstock_recs.

BEHAVIOR:
- For simple queries, use the pre-built tools first (they return structured JSON).
- For complex or custom queries, write Python/Pandas code via python_repl_ast.
- For SQL-style analysis on large data, use duckdb_query.
- Always return a clear, structured answer.
- If a tool fails, try a different approach — use python_repl_ast as the fallback.
"""


def build_data_agent_tool(llm, engine: MMMEngine) -> StructuredTool:
    """Build the Data Agent dispatch tool."""
    from ..tools.data_tools import build_data_tools

    # Build data-specific tools
    data_tools = build_data_tools(engine)

    # Add PythonAstREPLTool (gives the LLM a live Python REPL against df)
    try:
        from langchain_experimental.tools import PythonAstREPLTool
        df = engine.df
        repl_locals = {"df": df, "pd": None, "np": None}
        try:
            import pandas as pd
            import numpy as np
            repl_locals["pd"] = pd
            repl_locals["np"] = np
        except ImportError:
            pass
        repl_tool = PythonAstREPLTool(locals=repl_locals)
        data_tools = [repl_tool] + data_tools
        logger.info("[DataAgent] PythonAstREPLTool added")
    except ImportError:
        logger.warning("[DataAgent] langchain_experimental not available — PythonAstREPLTool skipped")

    # Add DuckDB tool
    try:
        import duckdb

        def run_duckdb(query: str) -> str:
            """Run a SQL query on the loaded dataset using DuckDB.
            The table is accessible as 'data'. Example: SELECT COUNT(*) FROM data"""
            from pydantic import BaseModel, Field as PField

            df = engine.df
            if df is None:
                return json.dumps({"error": "No data loaded"})
            try:
                conn = duckdb.connect()
                conn.register("data", df)
                result = conn.execute(query).fetchdf()
                return json.dumps(result.to_dict(orient="records"), default=str)
            except Exception as exc:
                return json.dumps({"error": str(exc)})

        class DuckDBInput(BaseModel):
            query: str = Field(description="SQL query. Table name is 'data'. E.g. SELECT COUNT(*) FROM data")

        duckdb_tool = StructuredTool.from_function(
            func=run_duckdb,
            name="duckdb_query",
            description=(
                "Run SQL on the loaded dataset using DuckDB. Table name is 'data'. "
                "Fast for aggregations, GROUP BY, filtering. Use instead of Pandas for large files."
            ),
            args_schema=DuckDBInput,
        )
        data_tools.append(duckdb_tool)
        logger.info("[DataAgent] DuckDB tool added")
    except ImportError:
        logger.warning("[DataAgent] duckdb not available — DuckDB tool skipped")

    logger.info(f"[DataAgent] Tool count: {len(data_tools)}")
    assert len(data_tools) <= 32, f"DataAgent exceeds 32 tool limit: {len(data_tools)}"

    def call_data_agent(task: str) -> str:
        # Keep df reference fresh
        try:
            from langchain_experimental.tools import PythonAstREPLTool
            for t in data_tools:
                if isinstance(t, PythonAstREPLTool):
                    t.locals["df"] = engine.df
        except Exception:
            pass

        logger.info(f"[Supervisor] Dispatching to DataAgent: {task[:80]}")
        result = _run_react_subagent(llm, data_tools, _DATA_AGENT_SYSTEM, task, "DataAgent")
        return result

    return StructuredTool.from_function(
        func=call_data_agent,
        name="call_data_agent",
        description=(
            "Delegate a data loading, inspection, or query task to the Data Agent. "
            "The Data Agent can: load files, profile columns, run SQL, run Python/Pandas, "
            "detect outliers, aggregate data, show sample rows, compute correlations. "
            "Pass a natural-language instruction."
        ),
        args_schema=CallDataAgentInput,
    )


# ─────────────────────────────────────────────
# SUB-AGENT 2: ANALYTICS AGENT
# Tools: ML feature importance, PCA, VIF, etc.
# ─────────────────────────────────────────────

_ANALYTICS_AGENT_SYSTEM = """You are an Analytics Agent specialized in advanced statistical and machine learning analysis.

You have access to:
- feature_importance_rf: Random Forest feature importance with cross-validation
- feature_importance_gb: Gradient Boosting feature importance
- mutual_information: Non-linear dependency scoring (catches what correlation misses)
- pca_analysis: Principal Component Analysis
- vif_analysis: Variance Inflation Factor (multicollinearity detection)
- granger_causality: Granger causality test for time series
- stationarity_test: ADF-like stationarity test
- time_series_decompose: Trend/seasonal/residual decomposition
- cross_validate_model: K-fold cross-validation for OLS, Ridge, Lasso, RF, GB
- compare_models: Compare model performance head-to-head
- stepwise_selection: Forward/backward stepwise feature selection
- auto_feature_engineering: Generate lag, rolling, and interaction features

BEHAVIOR:
- NEVER use just one method — use at least 3 methods and synthesize a consensus.
- For feature selection: run RF importance + mutual_information + VIF + stepwise_selection.
- For model comparison: run cross_validate_model with all models.
- After each tool, interpret the result in the context of the overall task.
- At the end, synthesize a consensus ranking across all methods.
- Present numbered findings with actual values from the tool outputs.
"""


def build_analytics_agent_tool(llm, engine: MMMEngine) -> StructuredTool:
    """Build the Analytics Agent dispatch tool."""
    from ..tools.analytics_tools import build_analytics_tools
    analytics_tools = build_analytics_tools(engine)

    logger.info(f"[AnalyticsAgent] Tool count: {len(analytics_tools)}")
    assert len(analytics_tools) <= 32, f"AnalyticsAgent exceeds 32 tool limit: {len(analytics_tools)}"

    def call_analytics_agent(task: str, target_col: str, feature_cols: Optional[str] = None) -> str:
        full_task = task
        if target_col:
            full_task = f"Target column: '{target_col}'. " + (
                f"Candidate features: {feature_cols}. " if feature_cols else ""
            ) + task
        logger.info(f"[Supervisor] Dispatching to AnalyticsAgent: {full_task[:100]}")
        result = _run_react_subagent(llm, analytics_tools, _ANALYTICS_AGENT_SYSTEM, full_task, "AnalyticsAgent")
        return result

    return StructuredTool.from_function(
        func=call_analytics_agent,
        name="call_analytics_agent",
        description=(
            "Delegate advanced ML and statistical analysis to the Analytics Agent. "
            "The Analytics Agent can: compute feature importance (RF, GB, mutual info), "
            "run PCA, detect multicollinearity (VIF), perform stepwise selection, "
            "cross-validate multiple models, test stationarity, decompose time series, "
            "and synthesize a consensus ranking. Always runs multiple methods."
        ),
        args_schema=CallAnalyticsAgentInput,
    )


# ─────────────────────────────────────────────
# SUB-AGENT 3: MMM / VIZ AGENT
# Tools: adstock, OLS, Bayesian, budget + charts
# ─────────────────────────────────────────────

_MMM_AGENT_SYSTEM = """You are an MMM (Marketing Mix Modelling) and Visualization Agent.

You have access to:
MMM tools:
- optimize_adstock: Optimize adstock parameters for a single channel
- optimize_all_adstock: Optimize adstock for all channels at once
- run_ols: Run OLS regression MMM
- run_bayesian: Run Bayesian MMM (most robust)
- roi_summary: Get ROI estimates per channel
- optimize_budget: Optimize budget allocation across channels
- simulate_scenario: Simulate a budget scenario
- compare_scenarios: Compare two budget scenarios

Visualization tools:
- plot_correlation_heatmap: Generate a correlation heatmap chart
- plot_feature_importance: 4-method feature importance comparison chart
- plot_time_series: Plot time series for one or more columns
- plot_distributions: Distribution histograms for columns
- plot_scatter_matrix: Pairwise scatter plots
- plot_model_comparison: R²/RMSE boxplot chart comparing models

BEHAVIOR:
- For MMM: always optimize adstock first, then fit the model, then compute ROI.
- For visualization: generate appropriate charts based on the task.
- Always interpret results — don't just return raw numbers.
- Generate charts proactively when doing analysis.
"""


def build_mmm_agent_tool(llm, engine: MMMEngine) -> StructuredTool:
    """Build the MMM/Viz Agent dispatch tool."""
    from ..tools.mmm_tools import build_mmm_tools
    from ..tools.viz_tools import build_viz_tools

    mmm_tools = build_mmm_tools(engine)
    viz_tools = build_viz_tools(engine)
    all_tools = mmm_tools + viz_tools

    logger.info(f"[MMMAgent] Tool count: {len(all_tools)}")
    assert len(all_tools) <= 32, f"MMMAgent exceeds 32 tool limit: {len(all_tools)}"

    def call_mmm_agent(task: str, kpi_col: Optional[str] = None, channel_cols: Optional[str] = None) -> str:
        full_task = task
        if kpi_col:
            full_task = f"KPI column: '{kpi_col}'. " + (
                f"Channel columns: {channel_cols}. " if channel_cols else ""
            ) + task
        logger.info(f"[Supervisor] Dispatching to MMMAgent: {full_task[:100]}")
        result = _run_react_subagent(llm, all_tools, _MMM_AGENT_SYSTEM, full_task, "MMMAgent")
        return result

    return StructuredTool.from_function(
        func=call_mmm_agent,
        name="call_mmm_agent",
        description=(
            "Delegate Marketing Mix Modelling or visualization tasks to the MMM Agent. "
            "The MMM Agent can: optimize adstock parameters, run OLS/Bayesian MMM, "
            "compute ROI, optimize budget allocation, simulate scenarios, and generate "
            "any of 6 types of charts (heatmap, feature importance, time series, distributions, "
            "scatter matrix, model comparison)."
        ),
        args_schema=CallMMMAgentInput,
    )


# ─────────────────────────────────────────────
# CONVENIENCE: BUILD ALL 3 DISPATCH TOOLS
# ─────────────────────────────────────────────

def build_dispatch_tools(llm, engine: MMMEngine) -> List[StructuredTool]:
    """Build all 3 sub-agent dispatch tools for the supervisor."""
    tools = [
        build_data_agent_tool(llm, engine),
        build_analytics_agent_tool(llm, engine),
        build_mmm_agent_tool(llm, engine),
    ]
    logger.info(f"[Supervisor] {len(tools)} dispatch tools ready")
    return tools
