# =============================================================
# workflows/nodes.py — LangGraph node implementations
# 5-node cognitive architecture:
#   Planner → Agent → Tools → Reflection → Quality Gate
# =============================================================

import json
import logging
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from .state import (
    AgentState, Phase, PHASE_TRANSITIONS,
    validate_phase_transition, get_next_phase, PHASE_QUALITY_THRESHOLDS,
    empty_plan, empty_reflection, empty_error_recovery,
)
from ..tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _extract_tool_calls(message: BaseMessage) -> List[Dict[str, Any]]:
    """Extract tool calls from an AIMessage regardless of SDK version."""
    calls = []
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
    return bool(_extract_tool_calls(message))


def _parse_json_from_response(content: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from LLM text output.
    Handles ```json fences and raw JSON.
    """
    import re
    # Try fenced JSON block first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Try raw JSON
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {}


def _summarise_args(args: Dict[str, Any]) -> str:
    parts = []
    for k, v in args.items():
        sv = str(v)
        parts.append(f"{k}={sv[:40]!r}" if len(sv) > 40 else f"{k}={v!r}")
    return ", ".join(parts)


# ─────────────────────────────────────────────
# 1. PLANNER NODE
# ─────────────────────────────────────────────

def build_planner_node(llm, system_message_fn, registry=None):
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
        ctx = state.get("context", {})
        reflection = state.get("reflection", {})
        iteration = state.get("iteration", 0)

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

            plan = {
                "goal": parsed.get("goal", f"Complete {phase} phase"),
                "current_phase": phase,
                "intent": intent,
                "steps": parsed.get("steps", []),
                "success_criteria": parsed.get("success_criteria", []),
                "current_step_idx": 0,
                "revision_count": state.get("plan", {}).get("revision_count", 0) + (
                    1 if reflection.get("decision") == "retry" else 0
                ),
            }

            logger.info(f"[PLANNER] Phase={phase}, Intent={intent}, Steps={len(plan['steps'])}, Goal={plan['goal'][:80]}")

            plan_summary = f"📋 **Plan** ({intent}): {plan['goal']}\n"
            for i, step in enumerate(plan.get("steps", []), 1):
                plan_summary += f"  {i}. {step.get('action', '?')} → `{step.get('tool', '?')}`\n"

            plan_msg = AIMessage(content=plan_summary)

            return {
                "messages": [plan_msg],
                "next_step": "agent",
                "plan": plan,
                "iteration": iteration,
            }

        except Exception as exc:
            logger.warning(f"[PLANNER] Failed: {exc}, falling back to agent")
            return {
                "messages": [],
                "next_step": "agent",
                "plan": empty_plan(),
                "iteration": iteration,
            }

    return planner_node


# ─────────────────────────────────────────────
# 2. AGENT NODE (enhanced with error recovery)
# ─────────────────────────────────────────────

def build_agent_node(llm_with_tools, system_message_fn):
    """
    Factory: returns a node function that calls the LLM.
    Enhanced with adaptive error recovery and plan awareness.
    """

    def agent_node(state: AgentState) -> Dict[str, Any]:
        from langchain_core.messages import SystemMessage

        iteration = state.get("iteration", 0)
        error_state = state.get("error_recovery", empty_error_recovery())

        # Safety: max iterations
        max_iter = 40
        if iteration >= max_iter:
            logger.warning(f"[AGENT] Max iterations ({max_iter}) reached — stopping")
            err_msg = AIMessage(
                content="⚠️ Maximum iteration limit reached. Routing to reflection for assessment."
            )
            return {
                "messages": [err_msg],
                "next_step": "reflect",
                "iteration": iteration + 1,
            }

        sys_msg = system_message_fn(state)
        messages = [sys_msg] + list(state["messages"])

        try:
            response: AIMessage = llm_with_tools.invoke(messages)

            # Reset error recovery on success
            error_state = {
                **error_state,
                "consecutive_failures": 0,
                "alternative_tried": False,
            }

        except Exception as exc:
            logger.exception("LLM invoke failed")

            error_state["consecutive_failures"] = error_state.get("consecutive_failures", 0) + 1
            error_state["retry_count"] = error_state.get("retry_count", 0) + 1

            if error_state["consecutive_failures"] <= error_state.get("max_retries", 3):
                # Retry: inject error context and try again
                logger.info(f"[AGENT] Error recovery attempt {error_state['consecutive_failures']}")
                err_msg = AIMessage(
                    content=f"⚠️ LLM error (attempt {error_state['consecutive_failures']}): {exc}. "
                            f"I will try a different approach."
                )
                return {
                    "messages": [err_msg],
                    "next_step": "agent",  # retry
                    "iteration": iteration + 1,
                    "error_recovery": error_state,
                }
            else:
                # Give up — route to reflection
                err_msg = AIMessage(
                    content=f"⚠️ Repeated failures after {error_state['retry_count']} attempts. "
                            f"Routing to reflection for assessment."
                )
                return {
                    "messages": [err_msg],
                    "next_step": "reflect",
                    "iteration": iteration + 1,
                    "error_recovery": error_state,
                }

        has_calls = _has_tool_calls(response)

        # Determine routing
        if has_calls:
            next_step = "tools"
        else:
            # No more tool calls → route to reflection for assessment
            next_step = "reflect"

        return {
            "messages": [response],
            "next_step": next_step,
            "iteration": iteration + 1,
            "error_recovery": error_state,
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

    def _log(msg: str, style: str = "cyan") -> None:
        if console:
            console.print(f"[{style}]{msg}[/{style}]")
        else:
            logger.info(msg)

    def tool_node(state: AgentState) -> Dict[str, Any]:
        last_message = state["messages"][-1]
        tool_calls = _extract_tool_calls(last_message)

        if not tool_calls:
            return {"messages": [], "next_step": "agent"}

        tool_results: List[ToolMessage] = []
        phase = state.get("phase", Phase.IDLE.value)
        ctx = dict(state.get("context", {}))
        error_state = dict(state.get("error_recovery", empty_error_recovery()))

        for call in tool_calls:
            name = call["name"]
            args = call["args"]
            call_id = call["id"]

            _log(f"  ⚙  {name}({_summarise_args(args)})", "dim cyan")

            result_str = registry.invoke(name, args)

            # Parse result for context updates and phase transitions
            try:
                result_dict = json.loads(result_str)

                # ── Auto-extract context from tool results ──
                _auto_update_context(name, args, result_dict, ctx)

                # ── Phase transition ──
                phase = _update_phase(name, result_dict, phase)

                # ── Track tool failures for error recovery ──
                if not result_dict.get("success", True):
                    error_state["consecutive_failures"] = error_state.get("consecutive_failures", 0) + 1
                    error_state["failed_tools"] = error_state.get("failed_tools", []) + [{
                        "tool": name,
                        "args": {k: str(v)[:100] for k, v in args.items()},
                        "error": str(result_dict.get("error", "Unknown"))[:200],
                        "timestamp": datetime.now().isoformat(),
                    }]
                else:
                    error_state["consecutive_failures"] = 0

            except Exception:
                pass

            display = result_str[:300] + "…" if len(result_str) > 300 else result_str
            _log(f"  ↩  {display}", "dim green")

            tool_results.append(
                ToolMessage(
                    content=result_str,
                    tool_call_id=call_id,
                    name=name,
                )
            )

        return {
            "messages": tool_results,
            "next_step": "agent",
            "phase": phase,
            "context": ctx,
            "error_recovery": error_state,
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
        ctx["table_path"] = args.get("path", ctx.get("table_path"))
        if result.get("potential_spend_columns"):
            ctx["channel_cols"] = result["potential_spend_columns"]
        if result.get("potential_kpi_columns"):
            kpis = result["potential_kpi_columns"]
            if kpis and not ctx.get("kpi_col"):
                ctx["kpi_col"] = kpis[0]  # auto-select first candidate
        if result.get("time_column"):
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
            # Validate transition is allowed
            if validate_phase_transition(current_phase, new_phase):
                return new_phase
            # If not valid, stay in current — the quality gate will handle it
            logger.debug(
                f"Phase transition {current_phase} → {new_phase} blocked by validator"
            )
            return current_phase

    return current_phase


# ─────────────────────────────────────────────
# 4. REFLECTION NODE
# ─────────────────────────────────────────────

def build_reflection_node(llm):
    """
    Self-evaluation node. After the agent finishes acting, the reflection
    node evaluates what was accomplished vs. the plan, scores quality,
    and decides the next routing.
    """

    def reflection_node(state: AgentState) -> Dict[str, Any]:
        from langchain_core.messages import SystemMessage

        phase = state.get("phase", Phase.IDLE.value)
        plan = state.get("plan", empty_plan())
        ctx = state.get("context", {})
        iteration = state.get("iteration", 0)

        # For truly empty plans (no steps), skip reflection gracefully
        if not plan.get("steps") and phase == Phase.IDLE.value:
            return {
                "messages": [],
                "next_step": "end",
                "reflection": empty_reflection(),
            }

        # Gather recent agent actions from message history
        recent_actions = []
        for msg in state.get("messages", [])[-15:]:  # last 15 messages
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
        reflect_msg = AIMessage(content=reflect_summary)

        # Route based on decision
        decision = reflection["decision"]
        if decision == "phase_complete":
            next_step = "quality_gate"
        elif decision == "retry":
            next_step = "planner"  # re-plan with feedback
        else:  # "continue"
            next_step = "agent"

        # Store quality score
        quality_scores = dict(state.get("quality_scores", {}))
        quality_scores[phase] = reflection["quality_score"]

        return {
            "messages": [reflect_msg],
            "next_step": next_step,
            "reflection": reflection,
            "quality_scores": quality_scores,
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

    def _log(msg: str, style: str = "cyan") -> None:
        if console:
            console.print(f"[{style}]{msg}[/{style}]")
        else:
            logger.info(msg)

    def quality_gate_node(state: AgentState) -> Dict[str, Any]:
        phase = state.get("phase", Phase.IDLE.value)
        quality_scores = state.get("quality_scores", {})
        phase_results_all = dict(state.get("phase_results", {}))
        reflection = state.get("reflection", {})
        iteration = state.get("iteration", 0)
        ctx = state.get("context", {})

        current_score = quality_scores.get(phase, 0.0)
        threshold = PHASE_QUALITY_THRESHOLDS.get(phase, 0.3)

        # Record phase result
        phase_results_all[phase] = {
            "quality_score": current_score,
            "achievements": reflection.get("achievements", []),
            "iteration_completed": iteration,
        }

        # Check quality threshold
        if current_score < threshold and iteration < 35:
            _log(
                f"  🚧 [GATE] Phase '{phase}' quality {current_score:.0%} < "
                f"threshold {threshold:.0%} — sending back for retry",
                "yellow"
            )
            gate_msg = AIMessage(
                content=(
                    f"🚧 **Quality Gate**: Phase '{phase}' scored {current_score:.0%} "
                    f"(threshold: {threshold:.0%}). Sending back to planner for improvement."
                )
            )
            return {
                "messages": [gate_msg],
                "next_step": "planner",  # re-plan with feedback
                "phase_results": phase_results_all,
            }

        # Advance to next phase
        next_phase = get_next_phase(phase)

        if next_phase is None or phase == Phase.DONE.value:
            _log(f"  ✅ [GATE] All phases complete!", "bold green")
            gate_msg = AIMessage(
                content=f"✅ **Quality Gate**: All phases complete! Final quality scores: "
                        f"{json.dumps({k: f'{v:.0%}' for k, v in quality_scores.items()})}"
            )
            return {
                "messages": [gate_msg],
                "next_step": "end",
                "phase": Phase.DONE.value,
                "phase_results": phase_results_all,
            }

        _log(
            f"  ✅ [GATE] Phase '{phase}' passed ({current_score:.0%} ≥ {threshold:.0%}) "
            f"→ advancing to '{next_phase}'",
            "bold green"
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

        gate_msg = AIMessage(
            content=(
                f"✅ **Quality Gate**: Phase '{phase}' passed with {current_score:.0%}. "
                f"Advancing to **{next_phase}**."
            )
        )

        return {
            "messages": [gate_msg],
            "next_step": "planner",
            "phase": next_phase,
            "phase_results": phase_results_all,
            "context": new_ctx,
            "plan": empty_plan(),          # fresh plan for new phase
            "reflection": empty_reflection(),  # reset reflection
            "error_recovery": empty_error_recovery(),  # reset errors
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
