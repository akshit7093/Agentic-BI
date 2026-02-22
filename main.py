# =============================================================
# main.py — Entry point: NotebookMMM class + init_mmm()
# =============================================================

import json
import logging
import os
import traceback
import uuid
from typing import Any, Dict, Optional

import re
from tqdm.auto import tqdm

def _clean_print(text):
    # Strip basic rich markup tags
    clean = re.sub(r'\[/?(?:bold|dim|italic|underline|strike|color|blink|reverse|red|green|yellow|blue|magenta|cyan|white|black)[^\]]*\]', '', text)
    tqdm.write(clean)

class BasicConsole:
    def print(self, *args, **kwargs):
        text = " ".join(str(a) for a in args)
        _clean_print(text)
    
    def print_json(self, text, **kwargs):
        _clean_print(text)

from .core.mmm_engine import MMMEngine
from .workflows.state import initial_state, Phase
from .config import DEFAULT_TABLE, DEFAULT_KPI_COL, DEFAULT_LLM_ENDPOINT

try:
    from langchain_core.messages import HumanMessage, AIMessage
    LC_AVAILABLE = True
except ImportError:
    LC_AVAILABLE = False

try:
    from pyspark.sql import SparkSession as _SS
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False

logger = logging.getLogger(__name__)
console = BasicConsole()


# =============================================================
def _get_spark():
    if not SPARK_AVAILABLE:
        return None
    try:
        spark = _SS.getActiveSession()
        if spark:
            return spark
    except Exception:
        pass
    try:
        from databricks.connect import DatabricksSession
        env_val = os.environ.get("SPARK_REMOTE", "")
        if env_val and not env_val.startswith("sc://"):
            os.environ.pop("SPARK_REMOTE", None)
        return DatabricksSession.builder.getOrCreate()
    except Exception:
        return None


# =============================================================
class NotebookMMM:
    """
    High-level interface for the Agentic MMM system.
    Usable both in Databricks notebooks and from the terminal.

    Quick start:
        nb = NotebookMMM()
        nb.ask("Show me top 10 products by revenue")
        nb.chat()          # interactive loop
    """

    def __init__(
        self,
        table: str = DEFAULT_TABLE,
        kpi_col: str = DEFAULT_KPI_COL,
        llm_endpoint: str = DEFAULT_LLM_ENDPOINT,
        auto_load: bool = True,
        spark=None,
    ):
        console.print(
            "\n🤖 Agentic MMM System\n"
            "Intelligent Data Analyst & Marketing Mix Modelling Agent\n"
        )

        self._spark = spark or (_get_spark() if SPARK_AVAILABLE else None)
        if self._spark:
            console.print("✅ Spark session active")
        else:
            console.print("⚠ Spark unavailable — CSV/local mode only")

        self.engine = MMMEngine(self._spark)
        self._table = table
        self._kpi_col = kpi_col
        self._llm_endpoint = llm_endpoint
        self._graph = None
        self._registry = None
        self._thread_id = str(uuid.uuid4())
        self._state = initial_state(interactive=True)

        if auto_load and table:
            self.load(table)

    # ─────────────────────────────────────────────
    # DATA OPERATIONS (without agent)
    # ─────────────────────────────────────────────

    def load(self, path: str) -> bool:
        """Load data directly (no agent)."""
        console.print(f"Loading: {path}")
        res = self.engine.load_data(path)
        if res["success"]:
            console.print(f"✅ {res['rows']:,} rows × {len(res['columns'])} columns")
            if res.get("potential_spend_columns"):
                console.print(f"   💡 Spend columns detected: {res['potential_spend_columns']}")
            if res.get("potential_kpi_columns"):
                console.print(f"   💡 KPI columns detected: {res['potential_kpi_columns']}")
            return True
        console.print(f"❌ Load failed: {res['error']}")
        return False

    def inspect(self) -> Dict[str, Any]:
        """Inspect loaded data directly."""
        res = self.engine.inspect_data()
        if res.get("success"):
            self._print_inspection(res)
        else:
            console.print(f"[red]❌ {res['error']}[/red]")
        return res

    def _print_inspection(self, res: Dict[str, Any]) -> None:
        console.print("\n--- Dataset Profile ---")
        console.print(f"Shape:      {res['rows']:,} rows × {len(res['columns'])} cols")
        console.print(f"Spend cols: {res.get('potential_spend_columns', [])}")
        console.print(f"KPI cols:   {res.get('potential_kpi_columns', [])}")
        console.print(f"Time col:   {res.get('time_column')}")
        console.print(f"Nulls(any): {any(v > 0 for v in res.get('null_pct', {}).values())}")
        console.print("-----------------------\n")

    # ─────────────────────────────────────────────
    # AGENT OPERATIONS
    # ─────────────────────────────────────────────

    def _ensure_agent(self) -> None:
        if self._graph is not None:
            return
        from .agent.builder import build_agent
        try:
            self._graph, self._registry = build_agent(
                self.engine,
                llm_endpoint=self._llm_endpoint,
                console=console,
            )
        except Exception as exc:
            console.print(f"[red]❌ Agent init failed: {exc}[/red]")
            raise

    def ask(
        self,
        question: str,
        thread_id: Optional[str] = None,
        phase: Optional[str] = None,
    ) -> str:
        """
        Send a question / command to the agent.
        Maintains conversation context across calls when thread_id is the same.
        """
        self._ensure_agent()
        tid = thread_id or self._thread_id
        console.print(f"\n[bold cyan]You:[/bold cyan] {question}")

        config = {"configurable": {"thread_id": tid}}
        input_state = {
            "messages": [HumanMessage(content=question)],
            "next_step": "planner",
            "phase": phase or self._state.get("phase", Phase.IDLE.value),
            "context": self._state.get("context", {}),
            "iteration": 0,
            "interactive": True,
            "plan": self._state.get("plan", {}),
            "reflection": self._state.get("reflection", {}),
            "quality_scores": self._state.get("quality_scores", {}),
            "error_recovery": self._state.get("error_recovery", {}),
            "phase_results": self._state.get("phase_results", {}),
        }

        final_response: Optional[str] = None
        tool_call_count = 0
        # Intent-based limits: set after planner runs
        # Default high limit; planner output refines it
        max_tool_calls = 50  # default for analysis

        try:
            with tqdm(desc="Agent Steps", unit="step", dynamic_ncols=True) as pbar:
                for event in self._graph.stream(input_state, config):
                    pbar.update(1)
                    for node_name, node_data in event.items():

                        # After planner runs, set intent-based tool limit
                        if node_name == "planner" and "plan" in node_data:
                            intent = node_data["plan"].get("intent", "analysis")
                            max_tool_calls = {
                                "simple": 5,
                                "data_query": 15,
                                "analysis": 50,
                            }.get(intent, 50)

                        # Count only actual tool executions
                        if node_name == "tools":
                            tool_call_count += 1
                            if tool_call_count > max_tool_calls:
                                console.print(
                                    f"⚠ Safety limit: {max_tool_calls} "
                                    f"tool calls reached for intent '{intent}'"
                                )
                                break

                        if node_name == "agent" and "messages" in node_data:
                            msgs = node_data["messages"]
                            if msgs:
                                msg = msgs[-1]
                                if isinstance(msg, AIMessage) and msg.content:
                                    tc = getattr(msg, "tool_calls", None)
                                    if not tc:
                                        final_response = str(msg.content)
                    else:
                        continue  # inner loop didn't break
                    break  # inner loop broke → stop outer loop too
        except Exception as exc:
            traceback.print_exc()
            final_response = f"⚠ Agent error: {exc}"

        # Fallback: read from checkpoint
        if final_response is None:
            try:
                saved = self._graph.get_state(config)
                for msg in reversed(saved.values.get("messages", [])):
                    if isinstance(msg, AIMessage) and msg.content:
                        tc = getattr(msg, "tool_calls", None)
                        if not tc:
                            final_response = str(msg.content)
                            break
            except Exception:
                pass

        # Capture final state for continuity
        try:
            saved = self._graph.get_state(config)
            if saved and saved.values:
                self._state = {**self._state, **{
                    k: saved.values[k]
                    for k in ("phase", "context", "plan", "reflection",
                              "quality_scores", "error_recovery", "phase_results")
                    if k in saved.values
                }}
        except Exception:
            pass

        output = final_response or "(No text response generated — see tool outputs above)"
        console.print(f"\n🤖 Agent:\n{output}\n")
        return output

    # ─────────────────────────────────────────────
    # FULL AUTONOMOUS ANALYSIS
    # ─────────────────────────────────────────────

    def run_full_analysis(self, kpi_col: Optional[str] = None, channel_cols: Optional[str] = None) -> str:
        """
        Trigger a fully autonomous end-to-end MMM analysis.
        The cognitive graph (planner → agent → tools → reflection → quality gate)
        manages multi-phase orchestration automatically.
        """
        self._ensure_agent()
        kpi = kpi_col or self._kpi_col

        # Pre-populate context so the planner has hints
        ctx = dict(self._state.get("context", {}))
        if kpi:
            ctx["kpi_col"] = kpi
        if channel_cols:
            if isinstance(channel_cols, str):
                ctx["channel_cols"] = [c.strip() for c in channel_cols.split(",")]
            else:
                ctx["channel_cols"] = list(channel_cols)
        self._state["context"] = ctx

        channels_hint = f" Use these as channel columns: {channel_cols}." if channel_cols else ""
        prompt = (
            f"Run a complete end-to-end MMM analysis. "
            f"The KPI column is '{kpi}'.{channels_hint} "
            f"Autonomously: profile the data, clean if needed, "
            f"identify spend channels, optimise adstock parameters for each, "
            f"fit a Bayesian MMM, evaluate model quality, "
            f"then optimise budget allocation. "
            f"Log key findings at each step and present a final executive summary."
        )
        return self.ask(prompt, phase=Phase.DATA_LOADING.value)

    # ─────────────────────────────────────────────
    # INTERACTIVE CHAT LOOP
    # ─────────────────────────────────────────────

    def chat(self) -> None:
        """Start an interactive chat session with the agent."""
        self._ensure_agent()

        console.print(
            "\n────────── Chat Mode ──────────\n"
            "🗨️  Interactive MMM Agent Chat\n\n"
            "Try:\n"
            " - 'Show me basic analytics'\n"
            " - 'Which channels drive the most sales?'\n"
            " - 'Run an OLS model'\n"
            "Type 'exit' to quit.\n"
            "─────────────────────────────────\n"
        )

        thread_id = str(uuid.uuid4())

        while True:
            try:
                user_input = input("\n🧑 You: ").strip()
                if not user_input:
                    continue

                cmd = user_input.lower().strip()

                if cmd in ("quit", "exit", "q", "bye"):
                    console.print("[yellow]👋 Goodbye![/yellow]")
                    break

                if cmd == "new":
                    thread_id = str(uuid.uuid4())
                    console.print("[cyan]🔄 New conversation thread started[/cyan]")
                    continue

                if cmd == "status":
                    status = self.engine.get_status()
                    console.print_json(json.dumps(status, default=str))
                    continue

                if cmd == "tools":
                    if self._registry:
                        tools = self._registry.list_tools()
                        for t in tools:
                            tag = "[cyan][custom][/cyan] " if t.get("dynamic") else ""
                            console.print(f"  {tag}[bold]{t['name']}[/bold]: {t['description']}")
                    continue

                if cmd.startswith("load "):
                    path = user_input[5:].strip()
                    self.load(path)
                    continue

                self.ask(user_input, thread_id=thread_id)

            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted — type 'quit' to exit[/yellow]")
            except Exception as exc:
                console.print(f"[red]Error: {exc}[/red]")

    # ─────────────────────────────────────────────
    # DIRECT ENGINE ACCESS (no agent)
    # ─────────────────────────────────────────────

    @property
    def data(self):
        return self.engine.data

    @property
    def model_results(self):
        return self.engine.model_results

    @property
    def budget_results(self):
        return self.engine.budget_results

    def add_tool(self, name: str, description: str, code: str, params: dict = None) -> Dict[str, Any]:
        """
        Manually register a custom tool from Python (not through the agent).

        Example:
            nb.add_tool(
                name="weekly_trend",
                description="Compute 4-week rolling average of a column",
                code='''
def tool_fn(column):
    import json
    result = df[column].rolling(4).mean().dropna()
    return json.dumps({"mean_last4": float(result.iloc[-1])})
''',
                params={"column": "str"},
            )
        """
        self._ensure_agent()
        from .tools.registry import ToolSpec
        spec = ToolSpec(
            name=name,
            description=description,
            code=code,
            params=params or {},
        )
        extra = {"df": self.engine.data, "engine": self.engine}
        result = self._registry.register_from_spec(spec, extra_globals=extra)
        if result.get("success"):
            # Rebind LLM tools to include new tool
            console.print(f"[green]✅ Custom tool '{name}' registered[/green]")
        else:
            console.print(f"[red]❌ Tool registration failed: {result.get('error')}[/red]")
        return result

    def list_tools(self) -> None:
        """Print all available tools."""
        if not self._registry:
            console.print("Agent not yet initialised — call ask() or chat() first")
            return
        tools = self._registry.list_tools()
        console.print(f"\n--- Available Tools ({len(tools)}) ---")
        for t in tools:
            kind = "custom" if t.get("dynamic") else "built-in"
            console.print(f"{t['name'].ljust(30)} | {kind.ljust(10)} | {t['description'][:80]}")
        console.print("--------------------------------")


# =============================================================
# CONVENIENCE FACTORY
# =============================================================

def init_mmm(
    table: str = DEFAULT_TABLE,
    kpi_col: str = DEFAULT_KPI_COL,
    llm_endpoint: str = DEFAULT_LLM_ENDPOINT,
    auto_load: bool = True,
) -> NotebookMMM:
    """
    One-line initialisation.

    Usage:
        nb = init_mmm()
        nb.ask("Which columns should I use for MMM?")
        nb.run_full_analysis()
        nb.chat()
    """
    return NotebookMMM(
        table=table,
        kpi_col=kpi_col,
        llm_endpoint=llm_endpoint,
        auto_load=auto_load,
    )
