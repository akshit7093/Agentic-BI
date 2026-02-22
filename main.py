# =============================================================
# main.py — Entry point: NotebookMMM class + init_mmm()
# =============================================================

import json
import logging
import os
import traceback
import uuid
from typing import Any, Dict, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

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
console = Console()


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
        console.print(Panel.fit(
            "[bold blue]🤖 Agentic MMM System[/bold blue]\n"
            "[dim]Intelligent Data Analyst & Marketing Mix Modelling Agent[/dim]",
        ))

        self._spark = spark or (_get_spark() if SPARK_AVAILABLE else None)
        if self._spark:
            console.print("[green]✅ Spark session active[/green]")
        else:
            console.print("[yellow]⚠ Spark unavailable — CSV/local mode only[/yellow]")

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
        console.print(f"Loading: [cyan]{path}[/cyan]")
        res = self.engine.load_data(path)
        if res["success"]:
            console.print(f"[green]✅ {res['rows']:,} rows × {len(res['columns'])} columns[/green]")
            if res.get("potential_spend_columns"):
                console.print(f"   💡 Spend columns detected: {res['potential_spend_columns']}")
            if res.get("potential_kpi_columns"):
                console.print(f"   💡 KPI columns detected: {res['potential_kpi_columns']}")
            return True
        console.print(f"[red]❌ Load failed: {res['error']}[/red]")
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
        tbl = Table(title="Dataset Profile", show_header=True)
        tbl.add_column("Property", style="cyan")
        tbl.add_column("Value")
        tbl.add_row("Shape", f"{res['rows']:,} rows × {len(res['columns'])} cols")
        tbl.add_row("Spend cols", str(res.get("potential_spend_columns", [])))
        tbl.add_row("KPI cols", str(res.get("potential_kpi_columns", [])))
        tbl.add_row("Time col", str(res.get("time_column")))
        tbl.add_row("Nulls (any)", str(any(v > 0 for v in res.get("null_pct", {}).values())))
        console.print(tbl)

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
        step_count = 0

        with console.status("[bold green]Agent working…[/bold green]"):
            try:
                for event in self._graph.stream(input_state, config):
                    step_count += 1
                    if step_count > 25:
                        console.print("[yellow]⚠ Safety limit: max steps reached[/yellow]")
                        break
                    for node_name, node_data in event.items():
                        if node_name == "agent" and "messages" in node_data:
                            msgs = node_data["messages"]
                            if msgs:
                                msg = msgs[-1]
                                if isinstance(msg, AIMessage) and msg.content:
                                    tc = getattr(msg, "tool_calls", None)
                                    if not tc:
                                        final_response = str(msg.content)
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
        console.print(Panel(Markdown(output), title="[bold blue]🤖 Agent[/bold blue]", border_style="blue"))
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

        console.print(Panel.fit(
            "[bold green]💬 Interactive MMM Agent Chat[/bold green]\n\n"
            "[cyan]Try:[/cyan]\n"
            "  • [white]'profile the data'[/white]\n"
            "  • [white]'which columns are suitable for MMM?'[/white]\n"
            "  • [white]'run full MMM analysis'[/white]\n"
            "  • [white]'optimise my $1M budget'[/white]\n"
            "  • [white]'create a tool to detect seasonality'[/white]\n"
            "  • [white]'show me the analysis history'[/white]\n\n"
            "[yellow]Type 'quit' to exit | 'new' for a new conversation thread[/yellow]",
            title="Chat Mode",
            border_style="green",
        ))

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
            console.print("[yellow]Agent not yet initialised — call ask() or chat() first[/yellow]")
            return
        tools = self._registry.list_tools()
        tbl = Table(title=f"Available Tools ({len(tools)})")
        tbl.add_column("Name", style="bold cyan")
        tbl.add_column("Type")
        tbl.add_column("Description")
        for t in tools:
            kind = "[green]custom[/green]" if t.get("dynamic") else "built-in"
            tbl.add_row(t["name"], kind, t["description"][:80])
        console.print(tbl)


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
