# =============================================================
# workflows/nodes.py — LangGraph node implementations
# 5-node cognitive architecture:
#   Planner → Agent → Tools → Reflection → Quality Gate
# =============================================================

import json
import logging
import re
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from .state import (
    AgentState, Phase, PHASE_TRANSITIONS,
    validate_phase_transition, get_next_phase, PHASE_QUALITY_THRESHOLDS,
    empty_plan, empty_reflection, empty_error_recovery,
)
from ..tools.registry import ToolRegistry

# ─────────────────────────────────────────────
# CONFIGURATION CONSTANTS (was hardcoded magic numbers)
# ─────────────────────────────────────────────
MAX_ITERATIONS = 100
MAX_MESSAGES_KEEP = 50
MAX_RETRIES = 3
REFLECTION_MESSAGE_WINDOW = 15
QUALITY_GATE_RETRY_THRESHOLD = 35

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _extract_tool_calls(message: BaseMessage) -> List[Dict[str, Any]]:
    """Extract tool calls from an AIMessage regardless of SDK version."""
    calls: List[Dict[str, Any]] = []
    if not isinstance(message, AIMessage):
        return calls
    raw = getattr(message, "tool_calls", None) or []
    for call in raw:
        if isinstance(call, dict):
            calls.append({
                "name": call.get("name", ""),
                "args": call.get("args", call.get("arguments", {})),
                "id":   call.get("id", str(uuid.uuid4())),
            })
        else:
            args = getattr(call, "args", None) or getattr(call, "arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            calls.append({
                "name": getattr(call, "name", ""),
                "args": args,
                "id":   getattr(call, "id", str(uuid.uuid4())),
            })
    return calls


def _has_tool_calls(message: BaseMessage) -> bool:
    """Check if a message contains tool calls."""
    return bool(_extract_tool_calls(message))


def _parse_json_from_response(content: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from LLM text output.
    Handles ```json fences and raw JSON with improved robustness.
    """
    if not content or not isinstance(content, str):
        return {}
    
    # Try fenced JSON block first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    
    # Try to find any JSON object with better nesting support
    start = content.find("{")
    end = content.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(content[start:end])
        except Exception:
            pass
    
    return {}


def _summarise_args(args: Dict[str, Any]) -> str:
    """Create a compact string representation of tool arguments."""
    parts: List[str] = []
    for k, v in args.items():
        sv = str(v)
        parts.append(f"{k}={sv[:40]!r}" if len(sv) > 40 else f"{k}={v!r}")
    return ", ".join(parts)


def _log_message(console: Optional[Any], msg: str, level: str = "info") -> None:
    """
    Standardized logging function.
    Logs to both console (if available) and Python logger.
    """
    if console:
        console.print(msg)
    log_func = getattr(logger, level, logger.info)
    log_func(msg)


def _truncate_messages(messages: List[BaseMessage], max_count: int = MAX_MESSAGES_KEEP) -> List[BaseMessage]:
    """Keep only the most recent messages to prevent memory issues."""
    return list(messages)[-max_count:] if len(messages) > max_count else list(messages)


def _validate_tool_exists(registry: ToolRegistry, tool_name: str) -> Tuple[bool, str]:
    """Check if a tool exists in the registry."""
    available_tools = {t["name"] for t in registry.list_tools()}
    if tool_name in available_tools:
        return True, ""
    return False, f"Unknown tool: {tool_name}. Available: {', '.join(sorted(available_tools))}"


# ─────────────────────────────────────────────
# 1. PLANNER NODE
# ─────────────────────────────────────────────

def build_planner_node(llm, system_message_fn, registry=None, console=None):
    """
    Creates a step-by-step plan for the current phase.
    The planner receives phase context, available tools, and any
    reflection feedback from a previous iteration.

    Now accepts `registry` so it can inject the live tool list into the
    planning prompt — new tools are automatically discovered.
    """

    def planner_node(state: AgentState) -> Dict[str, Any]:
        from langchain_core.messages import SystemMessage

        phase = state.get("phase", Phase.IDLE.value)
        ctx = dict(state.get("context", {}))
        reflection = state.get("reflection", empty_reflection())
        iteration = state.get("iteration", 0)
        plan = state.get("plan", empty_plan())
        error_recovery = state.get("error_recovery", empty_error_recovery())
        quality_scores = dict(state.get("quality_scores", {}))
        phase_results = dict(state.get("phase_results", {}))

        # ── Build available tools list from registry ──
        available_tools_block = ""
        if registry:
            tool_list = registry.list_tools()
            tool_lines = [f"  - `{t['name']}`: {t['description']}" for t in tool_list]
            available_tools_block = (
                "\n## AVAILABLE TOOLS:\n"
                + "\n".join(tool_lines)
            )

        # Build reflection feedback if retrying
        reflection_feedback = ""
        if reflection.get("gaps"):
            reflection_feedback = (
                f"\n\n## FEEDBACK FROM PREVIOUS ATTEMPT:\n"
                f"Assessment: {reflection.get('assessment', 'N/A')}\n"
                f"Gaps: {json.dumps(reflection.get('gaps', []))}\n"
                f"Quality Score: {reflection.get('quality_score', 0)}\n"
                f"Previous decision: {reflection.get('decision', 'N/A')}\n"
                f"Reasoning: {reflection.get('reasoning', 'N/A')}\n"
                f"You MUST address these gaps in your revised plan."
            )

        # Get the user's latest message to understand intent
        user_message = ""
        for msg in reversed(state.get("messages", [])):
            if hasattr(msg, "type") and msg.type == "human":
                user_message = getattr(msg, "content", "")
                break
            elif hasattr(msg, "content") and isinstance(msg, HumanMessage):
                user_message = msg.content
                break

        plan_prompt = f"""You are a STRATEGIC PLANNER for an intelligent data analysis agent.

## YOUR TASK:
Classify the user's intent and create an appropriate execution plan.

## USER'S REQUEST: "{user_message}"

## CURRENT PHASE: {phase}
## ANALYSIS CONTEXT:
- KPI Column: {ctx.get('kpi_col', 'Not set')}
- Channel Columns: {ctx.get('channel_cols', [])}
- Date Column: {ctx.get('date_col', 'Not set')}
- Data Loaded: {bool(ctx.get('table_path'))}
- Model R²: {ctx.get('r2', 'Not yet modelled')}
- Findings so far: {len(ctx.get('findings', []))} logged
- Iteration: {iteration}
{reflection_feedback}
{available_tools_block}

## INSTRUCTIONS:
First, classify the user's request:
- **simple**: Greetings, yes/no questions, clarifications → 1 step plan (just respond)
- **data_query**: Questions about the data → 1-3 step plan (inspect, query, respond)
- **analysis**: "run MMM", "profile data", "full analysis" → 3-6 step plan

**CRITICAL REASONING RULES:**
1. If the user asks for modelling/MMM and column roles (KPI, channels) are not yet confirmed in the context, your plan MUST start with data inspection steps
2. If column roles are ambiguous, include an `ask_user` step to confirm before modelling
3. Never plan a modelling step using columns you haven't verified through data profiling

Respond with ONLY a JSON object (no extra text):
{{
    "intent": "simple" or "data_query" or "analysis",
    "goal": "What this plan should accomplish",
    "steps": [
        {{"action": "description of step", "tool": "tool_name_to_use", "expected_output": "what success looks like"}},
        ...
    ],
    "success_criteria": ["criterion 1", "criterion 2", ...]
}}

Keep the plan focused: 1-6 steps based on complexity.
Only use tools from the AVAILABLE TOOLS list above.
"""

        try:
            response = llm.invoke([SystemMessage(content=plan_prompt)])
            content = getattr(response, "content", "") or ""
            parsed = _parse_json_from_response(content)

            intent = parsed.get("intent", "analysis")

            new_plan = {
                "goal": parsed.get("goal", f"Complete {phase} phase"),
                "current_phase": phase,
                "intent": intent,
                "steps": parsed.get("steps", []),
                "success_criteria": parsed.get("success_criteria", []),
                "current_step_idx": 0,
                "revision_count": plan.get("revision_count", 0) + (
                    1 if reflection.get("decision") == "retry" else 0
                ),
            }

            logger.info(f"[PLANNER] Phase={phase}, Intent={intent}, Steps={len(new_plan['steps'])}, Goal={new_plan['goal'][:80]}")

            plan_summary = f"📋 **Plan** ({intent}): {new_plan['goal']}\n"
            for i, step in enumerate(new_plan.get("steps", []), 1):
                plan_summary += f"  {i}. {step.get('action', '?')} → `{step.get('tool', '?')}`\n"

            plan_msg = AIMessage(content=plan_summary)

            # Increment iteration consistently
            new_iteration = iteration + 1

            return {
                "messages": _truncate_messages(state.get("messages", []) + [plan_msg]),
                "next_step": "agent",
                "phase": phase,
                "context": ctx,
                "plan": new_plan,
                "reflection": reflection,
                "error_recovery": error_recovery,
                "quality_scores": quality_scores,
                "phase_results": phase_results,
                "iteration": new_iteration,
            }

        except Exception as exc:
            logger.warning(f"[PLANNER] Failed: {exc}, falling back to agent")
            _log_message(console, f"[yellow]⚠ Planner failed: {exc}[/yellow]", "warning")
            
            new_iteration = iteration + 1
            return {
                "messages": _truncate_messages(state.get("messages", [])),
                "next_step": "agent",
                "phase": phase,
                "context": ctx,
                "plan": empty_plan(),
                "reflection": reflection,
                "error_recovery": error_recovery,
                "quality_scores": quality_scores,
                "phase_results": phase_results,
                "iteration": new_iteration,
            }

    return planner_node


# ─────────────────────────────────────────────
# 2. AGENT NODE (enhanced with error recovery)
# ─────────────────────────────────────────────

def build_agent_node(llm_with_tools, system_message_fn, console=None):
    """
    Factory: returns a node function that calls the LLM.
    Enhanced with adaptive error recovery and plan awareness.
    """

    def agent_node(state: AgentState) -> Dict[str, Any]:
        from langchain_core.messages import SystemMessage

        iteration = state.get("iteration", 0)
        error_recovery = dict(state.get("error_recovery", empty_error_recovery()))
        phase = state.get("phase", Phase.IDLE.value)
        ctx = dict(state.get("context", {}))
        plan = state.get("plan", empty_plan())
        reflection = state.get("reflection", empty_reflection())
        quality_scores = dict(state.get("quality_scores", {}))
        phase_results = dict(state.get("phase_results", {}))

        # Safety: max iterations
        if iteration >= MAX_ITERATIONS:
            logger.warning(f"[AGENT] Max iterations ({MAX_ITERATIONS}) reached — stopping")
            _log_message(console, f"[yellow]⚠ Max iterations ({MAX_ITERATIONS}) reached[/yellow]", "warning")
            err_msg = HumanMessage(
                content="[System Note] ⚠️ Maximum iteration limit reached. Routing to reflection for assessment."
            )
            return {
                "messages": _truncate_messages(state.get("messages", []) + [err_msg]),
                "next_step": "reflect",
                "phase": phase,
                "context": ctx,
                "plan": plan,
                "reflection": reflection,
                "error_recovery": error_recovery,
                "quality_scores": quality_scores,
                "phase_results": phase_results,
                "iteration": iteration + 1,
            }

        sys_msg = system_message_fn(state)
        messages = [sys_msg] + list(state["messages"])

        try:
            response: AIMessage = llm_with_tools.invoke(messages)

            # Reset error recovery on success
            error_recovery = {
                **error_recovery,
                "consecutive_failures": 0,
                "alternative_tried": False,
            }

        except Exception as exc:
            logger.exception("LLM invoke failed")
            _log_message(console, f"[red]❌ LLM invoke failed: {exc}[/red]", "error")

            error_recovery["consecutive_failures"] = error_recovery.get("consecutive_failures", 0) + 1
            error_recovery["retry_count"] = error_recovery.get("retry_count", 0) + 1

            if error_recovery["consecutive_failures"] <= error_recovery.get("max_retries", MAX_RETRIES):
                # Retry: inject error context and try again
                logger.info(f"[AGENT] Error recovery attempt {error_recovery['consecutive_failures']}")
                err_msg = HumanMessage(
                    content=f"[System Note] ⚠️ LLM error (attempt {error_recovery['consecutive_failures']}): {exc}. "
                            f"Please try a different approach."
                )
                return {
                    "messages": _truncate_messages(state.get("messages", []) + [err_msg]),
                    "next_step": "agent",  # retry
                    "phase": phase,
                    "context": ctx,
                    "plan": plan,
                    "reflection": reflection,
                    "error_recovery": error_recovery,
                    "quality_scores": quality_scores,
                    "phase_results": phase_results,
                    "iteration": iteration + 1,
                }
            else:
                # Give up — route to reflection
                err_msg = AIMessage(
                    content=f"⚠️ Repeated failures after {error_recovery['retry_count']} attempts. "
                            f"Routing to reflection for assessment."
                )
                return {
                    "messages": _truncate_messages(state.get("messages", []) + [err_msg]),
                    "next_step": "reflect",
                    "phase": phase,
                    "context": ctx,
                    "plan": plan,
                    "reflection": reflection,
                    "error_recovery": error_recovery,
                    "quality_scores": quality_scores,
                    "phase_results": phase_results,
                    "iteration": iteration + 1,
                }

        has_calls = _has_tool_calls(response)

        # Determine routing based on plan intent and whether there are tool calls
        if has_calls:
            next_step = "tools"
            call_names = [c["name"] for c in _extract_tool_calls(response)]
            logger.info(f"[AGENT] iter={iteration} → tools: {call_names}")
        else:
            # No tool calls → agent is done acting
            plan_intent = plan.get("intent", "analysis")
            if plan_intent in ("simple", "data_query"):
                # For simple/data queries, respond and finish — no need for
                # full reflection + quality gate cycle
                next_step = "end"
            else:
                # For full analysis, route to reflection for quality assessment
                next_step = "reflect"
            logger.info(f"[AGENT] iter={iteration} → {next_step} (intent={plan_intent}, no tool calls)")

        # Increment iteration consistently
        new_iteration = iteration + 1

        return {
            "messages": _truncate_messages(state.get("messages", []) + [response]),
            "next_step": next_step,
            "phase": phase,
            "context": ctx,
            "plan": plan,
            "reflection": reflection,
            "error_recovery": error_recovery,
            "quality_scores": quality_scores,
            "phase_results": phase_results,
            "iteration": new_iteration,
        }

    return agent_node


# ─────────────────────────────────────────────
# 3. TOOL NODE (enhanced with auto context extraction)
# ─────────────────────────────────────────────

def build_tool_node(registry: ToolRegistry, console=None):
    """
    Factory: returns a node function that executes tool calls.
    Enhanced with automatic context extraction from tool results.
    """

    def tool_node(state: AgentState) -> Dict[str, Any]:
        last_message = state["messages"][-1]
        tool_calls = _extract_tool_calls(last_message)

        phase = state.get("phase", Phase.IDLE.value)
        ctx = dict(state.get("context", {}))
        error_recovery = dict(state.get("error_recovery", empty_error_recovery()))
        plan = state.get("plan", empty_plan())
        reflection = state.get("reflection", empty_reflection())
        quality_scores = dict(state.get("quality_scores", {}))
        phase_results = dict(state.get("phase_results", {}))
        iteration = state.get("iteration", 0)

        if not tool_calls:
            # ✅ FIX: Preserve all state even when no tool calls
            return {
                "messages": _truncate_messages(state.get("messages", [])),
                "next_step": "agent",
                "phase": phase,
                "context": ctx,
                "plan": plan,
                "reflection": reflection,
                "error_recovery": error_recovery,
                "quality_scores": quality_scores,
                "phase_results": phase_results,
                "iteration": iteration + 1,
            }

        tool_results: List[ToolMessage] = []

        logger.info(f"[TOOLS] Executing {len(tool_calls)} tool call(s): {[c['name'] for c in tool_calls]}")

        for call in tool_calls:
            name = call["name"]
            args = call["args"]
            call_id = call["id"]

            _log_message(console, f"  ⚙  {name}({_summarise_args(args)})", "info")

            # ✅ FIX: Validate tool exists before invoking
            tool_exists, error_msg = _validate_tool_exists(registry, name)
            if not tool_exists:
                result_str = json.dumps({"success": False, "error": error_msg})
                _log_message(console, f"  ↩  [red]❌ {error_msg}[/red]", "error")
                error_recovery["consecutive_failures"] = error_recovery.get("consecutive_failures", 0) + 1
                error_recovery["failed_tools"] = error_recovery.get("failed_tools", []) + [{
                    "tool": name,
                    "args": {k: str(v)[:100] for k, v in args.items()},
                    "error": error_msg[:200],
                    "timestamp": datetime.now().isoformat(),
                }]
                tool_results.append(
                    ToolMessage(
                        content=result_str,
                        tool_call_id=call_id,
                        name=name,
                    )
                )
                continue

            result_str = registry.invoke(name, args)
            logger.debug(f"[TOOLS] {name} returned {len(result_str)} chars")

            # Parse result for context updates and phase transitions
            try:
                if name.startswith("call_") and name.endswith("_agent"):
                    # Dispatch tools return raw text, not JSON
                    result_dict = {"success": True, "message": result_str}
                else:
                    result_dict = json.loads(result_str)

                    # ✅ FIX: Validate result is a dict
                    if not isinstance(result_dict, dict):
                        logger.warning(f"Tool {name} returned non-dict result")
                        result_dict = {"success": False, "error": "Invalid result format"}

                # ── Auto-extract context from tool results ──
                _auto_update_context(name, args, result_dict, ctx)

                # ── Phase transition ──
                phase = _update_phase(name, result_dict, phase)

                # ── Track tool failures for error recovery ──
                if not result_dict.get("success", True):
                    error_recovery["consecutive_failures"] = error_recovery.get("consecutive_failures", 0) + 1
                    error_recovery["failed_tools"] = error_recovery.get("failed_tools", []) + [{
                        "tool": name,
                        "args": {k: str(v)[:100] for k, v in args.items()},
                        "error": str(result_dict.get("error", "Unknown"))[:200],
                        "timestamp": datetime.now().isoformat(),
                    }]
                else:
                    error_recovery["consecutive_failures"] = 0

            except Exception as e:
                logger.warning(f"Tool {name} result parsing failed: {e}")
                result_dict = {"success": False, "error": f"Parse error: {str(e)}"}
                error_recovery["consecutive_failures"] = error_recovery.get("consecutive_failures", 0) + 1

            display = result_str[:300] + "…" if len(result_str) > 300 else result_str
            _log_message(console, f"  ↩  {display}", "info")

            tool_results.append(
                ToolMessage(
                    content=result_str,
                    tool_call_id=call_id,
                    name=name,
                )
            )

        # Increment iteration consistently
        new_iteration = iteration + 1

        return {
            "messages": _truncate_messages(state.get("messages", []) + tool_results),
            "next_step": "agent",
            "phase": phase,
            "context": ctx,
            "plan": plan,
            "reflection": reflection,
            "error_recovery": error_recovery,
            "quality_scores": quality_scores,
            "phase_results": phase_results,
            "iteration": new_iteration,
        }

    return tool_node


def _auto_update_context(
    tool_name: str, args: Dict, result: Dict, ctx: Dict
) -> None:
    """
    Automatically extract structured context from tool results.
    Replaces the fragile <!--context: {...} --> HTML-comment approach.
    """
    if not result.get("success", False):
        return

    if "load_data" in tool_name:
        # ✅ FIX: Only set if not already defined (preserve user settings)
        if not ctx.get("table_path"):
            ctx["table_path"] = args.get("path")
        if result.get("potential_spend_columns"):
            ctx["channel_cols"] = result["potential_spend_columns"]
        if result.get("potential_kpi_columns"):
            kpis = result["potential_kpi_columns"]
            if kpis and not ctx.get("kpi_col"):
                ctx["kpi_col"] = kpis[0]  # auto-select first candidate
        if result.get("time_column") and not ctx.get("date_col"):
            ctx["date_col"] = result["time_column"]

    elif "inspect_data" in tool_name:
        if result.get("potential_spend_columns"):
            ctx["channel_cols"] = result["potential_spend_columns"]
        if result.get("potential_kpi_columns"):
            kpis = result["potential_kpi_columns"]
            if kpis and not ctx.get("kpi_col"):
                ctx["kpi_col"] = kpis[0]
        if result.get("time_column") and not ctx.get("date_col"):
            ctx["date_col"] = result["time_column"]

    elif "run_ols_mmm" in tool_name or "run_bayesian_mmm" in tool_name:
        if result.get("r2") is not None:
            ctx["r2"] = result["r2"]
        ctx["model_type"] = "bayesian" if "bayesian" in tool_name else "ols"

    elif "optimize_budget" in tool_name:
        if result.get("optimal_allocation"):
            ctx["budget_allocation"] = result["optimal_allocation"]


def _update_phase(tool_name: str, result: Dict, current_phase: str) -> str:
    """Automatically advance workflow phase based on tool calls and results."""
    if not result.get("success", True):
        return current_phase  # keep phase on failures

    transitions = {
        "load_data":                    Phase.DATA_PROFILING.value,
        "inspect_data":                 Phase.DATA_VALIDATION.value,
        "clean_data":                   Phase.FEATURE_ENG.value,
        "aggregate_weekly":             Phase.FEATURE_ENG.value,
        "add_time_features":            Phase.FEATURE_ENG.value,
        "optimize_adstock_parameters":  Phase.ADSTOCK_OPT.value,
        "optimize_all_adstock":         Phase.ADSTOCK_OPT.value,
        "run_ols_mmm":                  Phase.EVALUATION.value,
        "run_bayesian_mmm":             Phase.EVALUATION.value,
        "optimize_budget":              Phase.REPORTING.value,
    }

    for pattern, new_phase in transitions.items():
        if pattern in tool_name:
            # ✅ FIX: Validate transition but allow with warning if blocked
            if validate_phase_transition(current_phase, new_phase):
                return new_phase
            # Log warning but still allow transition - Quality Gate will validate
            logger.warning(
                f"Phase transition {current_phase} → {new_phase} unusual but allowing "
                f"(Quality Gate will validate)"
            )
            return new_phase

    return current_phase


# ─────────────────────────────────────────────
# 4. REFLECTION NODE
# ─────────────────────────────────────────────

def build_reflection_node(llm, console=None):
    """
    Self-evaluation node. After the agent finishes acting, the reflection
    node evaluates what was accomplished vs. the plan, scores quality,
    and decides the next routing.
    """

    def reflection_node(state: AgentState) -> Dict[str, Any]:
        from langchain_core.messages import SystemMessage

        phase = state.get("phase", Phase.IDLE.value)
        plan = state.get("plan", empty_plan())
        ctx = dict(state.get("context", {}))
        iteration = state.get("iteration", 0)
        error_recovery = dict(state.get("error_recovery", empty_error_recovery()))
        quality_scores = dict(state.get("quality_scores", {}))
        phase_results = dict(state.get("phase_results", {}))

        # For truly empty plans (no steps), skip reflection gracefully
        if not plan.get("steps") and phase == Phase.IDLE.value:
            return {
                "messages": _truncate_messages(state.get("messages", [])),
                "next_step": "end",
                "phase": phase,
                "context": ctx,
                "plan": plan,
                "reflection": empty_reflection(),
                "error_recovery": error_recovery,
                "quality_scores": quality_scores,
                "phase_results": phase_results,
                "iteration": iteration + 1,
            }

        # Gather recent agent actions from message history
        recent_actions: List[str] = []
        for msg in state.get("messages", [])[-REFLECTION_MESSAGE_WINDOW:]:
            if isinstance(msg, ToolMessage):
                content = getattr(msg, "content", "")
                name = getattr(msg, "name", "tool")
                recent_actions.append(f"Tool `{name}`: {content[:200]}")
            elif isinstance(msg, AIMessage):
                content = getattr(msg, "content", "")
                if content:
                    recent_actions.append(f"Agent: {content[:200]}")

        actions_text = "\n".join(recent_actions[-10:]) if recent_actions else "No actions recorded."

        plan_text = json.dumps(plan, indent=2, default=str)

        reflect_prompt = f"""You are a CRITICAL SELF-EVALUATOR for an MMM analysis agent.

## YOUR TASK:
Evaluate what the agent accomplished in the current phase and decide what to do next.

## CURRENT PHASE: {phase}
## PLAN:
{plan_text}

## RECENT ACTIONS & RESULTS:
{actions_text}

## ANALYSIS CONTEXT:
- KPI: {ctx.get('kpi_col', 'Not set')}
- Channels: {ctx.get('channel_cols', [])}
- Model R²: {ctx.get('r2', 'Not modelled')}
- Iteration: {iteration}

## INSTRUCTIONS:
Respond with ONLY a JSON object:
{{
    "assessment": "Brief narrative of what was accomplished",
    "quality_score": 0.0 to 1.0 rating of phase completion quality,
    "achievements": ["what was done well"],
    "gaps": ["what is still missing or needs improvement"],
    "decision": "phase_complete" or "continue" or "retry",
    "reasoning": "Why this decision"
}}

DECISION RULES:
- "phase_complete": The phase goals are met, quality is sufficient, move to next phase
- "continue": More work needed in this phase, send back to agent with specific guidance
- "retry": The approach failed, re-plan from scratch with reflection feedback
"""

        try:
            response = llm.invoke([SystemMessage(content=reflect_prompt)])
            content = getattr(response, "content", "") or ""
            parsed = _parse_json_from_response(content)

            reflection = {
                "assessment": parsed.get("assessment", "Assessment unavailable"),
                "quality_score": float(parsed.get("quality_score", 0.5)),
                "achievements": parsed.get("achievements", []),
                "gaps": parsed.get("gaps", []),
                "decision": parsed.get("decision", "phase_complete"),
                "reasoning": parsed.get("reasoning", ""),
            }

        except Exception as exc:
            logger.warning(f"[REFLECT] Parse failed: {exc}, defaulting to phase_complete")
            _log_message(console, f"[yellow]⚠ Reflection parse failed: {exc}[/yellow]", "warning")
            reflection = {
                "assessment": "Reflection could not parse LLM response",
                "quality_score": 0.5,
                "achievements": [],
                "gaps": [],
                "decision": "phase_complete",
                "reasoning": f"Defaulting due to parse error: {exc}",
            }

        logger.info(
            f"[REFLECT] Phase={phase}, Score={reflection['quality_score']:.2f}, "
            f"Decision={reflection['decision']}"
        )

        # Build a reflection summary message
        reflect_summary = (
            f"🪞 **Reflection** (Phase: {phase})\n"
            f"- Quality: {reflection['quality_score']:.0%}\n"
            f"- Decision: **{reflection['decision']}**\n"
            f"- {reflection['assessment'][:200]}"
        )
        reflect_msg = HumanMessage(content=reflect_summary)

        # Route based on decision
        decision = reflection["decision"]
        if decision == "phase_complete":
            next_step = "quality_gate"
        elif decision == "retry":
            next_step = "planner"  # re-plan with feedback
        else:  # "continue"
            next_step = "agent"

        # Store quality score
        quality_scores[phase] = reflection["quality_score"]

        # Increment iteration consistently
        new_iteration = iteration + 1

        return {
            "messages": _truncate_messages(state.get("messages", []) + [reflect_msg]),
            "next_step": next_step,
            "phase": phase,
            "context": ctx,
            "plan": plan,
            "reflection": reflection,
            "error_recovery": error_recovery,  # ✅ FIX: Preserve error_recovery
            "quality_scores": quality_scores,
            "phase_results": phase_results,
            "iteration": new_iteration,
        }

    return reflection_node


# ─────────────────────────────────────────────
# 5. QUALITY GATE NODE (programmatic, no LLM)
# ─────────────────────────────────────────────

def build_quality_gate_node(console=None):
    """
    Programmatic gate that validates phase completion, enforces
    quality thresholds, and advances the workflow phase.
    """

    def quality_gate_node(state: AgentState) -> Dict[str, Any]:
        phase = state.get("phase", Phase.IDLE.value)
        quality_scores = state.get("quality_scores", {})
        phase_results_all = dict(state.get("phase_results", {}))
        reflection = state.get("reflection", empty_reflection())
        iteration = state.get("iteration", 0)
        ctx = dict(state.get("context", {}))
        plan = state.get("plan", empty_plan())
        error_recovery = dict(state.get("error_recovery", empty_error_recovery()))

        current_score = quality_scores.get(phase, 0.0)
        threshold = PHASE_QUALITY_THRESHOLDS.get(phase, 0.3)

        # Record phase result
        phase_results_all[phase] = {
            "quality_score": current_score,
            "achievements": reflection.get("achievements", []),
            "iteration_completed": iteration,
        }

        # Check quality threshold
        if current_score < threshold and iteration < QUALITY_GATE_RETRY_THRESHOLD:
            _log_message(
                console,
                f"  🚧 [GATE] Phase '{phase}' quality {current_score:.0%} < "
                f"threshold {threshold:.0%} — sending back for retry",
                "warning"
            )
            gate_msg = HumanMessage(
                content=(
                    f"🚧 **Quality Gate**: Phase '{phase}' scored {current_score:.0%} "
                    f"(threshold: {threshold:.0%}). Sending back to planner for improvement."
                )
            )
            return {
                "messages": _truncate_messages(state.get("messages", []) + [gate_msg]),
                "next_step": "planner",  # re-plan with feedback
                "phase": phase,
                "context": ctx,
                "plan": plan,
                "reflection": reflection,
                "error_recovery": error_recovery,
                "quality_scores": quality_scores,
                "phase_results": phase_results_all,
                "iteration": iteration + 1,
            }

        # Advance to next phase
        next_phase = get_next_phase(phase)

        # ✅ FIX: Check next_phase instead of current phase
        if next_phase is None or next_phase == Phase.DONE.value:
            _log_message(console, f"  ✅ [GATE] All phases complete!", "info")
            gate_msg = HumanMessage(
                content=f"✅ **Quality Gate**: All phases complete! Final quality scores: "
                        f"{json.dumps({k: f'{v:.0%}' for k, v in quality_scores.items()})}"
            )
            return {
                "messages": _truncate_messages(state.get("messages", []) + [gate_msg]),
                "next_step": "end",
                "phase": Phase.DONE.value,
                "context": ctx,
                "plan": plan,
                "reflection": reflection,
                "error_recovery": error_recovery,
                "quality_scores": quality_scores,
                "phase_results": phase_results_all,
                "iteration": iteration + 1,
            }

        _log_message(
            console,
            f"  ✅ [GATE] Phase '{phase}' passed ({current_score:.0%} ≥ {threshold:.0%}) "
            f"→ advancing to '{next_phase}'",
            "info"
        )

        # Record phase transition in context
        phase_history = list(ctx.get("phase_history", []))
        phase_history.append({
            "from": phase,
            "to": next_phase,
            "score": current_score,
            "iteration": iteration,
        })
        new_ctx = {**ctx, "phase_history": phase_history}

        gate_msg = HumanMessage(
            content=(
                f"[System Note] ✅ **Quality Gate**: Phase '{phase}' passed with {current_score:.0%}. "
                f"Advancing to **{next_phase}**."
            )
        )

        # Increment iteration consistently
        new_iteration = iteration + 1

        return {
            "messages": _truncate_messages(state.get("messages", []) + [gate_msg]),
            "next_step": "planner",
            "phase": next_phase,
            "context": new_ctx,
            "plan": empty_plan(),          # fresh plan for new phase
            "reflection": empty_reflection(),  # reset reflection
            "error_recovery": empty_error_recovery(),  # reset errors
            "quality_scores": quality_scores,
            "phase_results": phase_results_all,
            "iteration": new_iteration,
        }

    return quality_gate_node


# ─────────────────────────────────────────────
# ROUTERS
# ─────────────────────────────────────────────

def agent_router(state: AgentState) -> str:
    """Route after agent_node: tools, reflect, or self-retry."""
    return state.get("next_step", "reflect")


def reflect_router(state: AgentState) -> str:
    """Route after reflection_node: agent (continue), planner (retry), or quality_gate."""
    return state.get("next_step", "quality_gate")


def gate_router(state: AgentState) -> str:
    """Route after quality_gate: planner (next phase) or end."""
    return state.get("next_step", "end")


def planner_router(state: AgentState) -> str:
    """Route after planner: agent or end."""
    return state.get("next_step", "agent")