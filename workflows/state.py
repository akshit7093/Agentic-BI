# =============================================================
# workflows/state.py — LangGraph agent state + workflow phase tracking
# =============================================================

from __future__ import annotations

import operator
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, Sequence

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


# ─────────────────────────────────────────────
# WORKFLOW PHASES
# ─────────────────────────────────────────────

class Phase(str, Enum):
    """Agent lifecycle phases — defines the analysis pipeline."""
    IDLE            = "idle"
    DATA_LOADING    = "data_loading"
    DATA_PROFILING  = "data_profiling"
    DATA_VALIDATION = "data_validation"
    FEATURE_ENG     = "feature_engineering"
    ADSTOCK_OPT     = "adstock_optimisation"
    MODELING        = "modeling"
    EVALUATION      = "evaluation"
    BUDGET_OPT      = "budget_optimisation"
    REPORTING       = "reporting"
    USER_INPUT      = "awaiting_user_input"
    DONE            = "done"


# Phase transitions (directed graph)
PHASE_TRANSITIONS: Dict[Phase, List[Phase]] = {
    Phase.IDLE:            [Phase.DATA_LOADING],
    Phase.DATA_LOADING:    [Phase.DATA_PROFILING, Phase.IDLE],
    Phase.DATA_PROFILING:  [Phase.DATA_VALIDATION, Phase.DATA_LOADING],
    Phase.DATA_VALIDATION: [Phase.FEATURE_ENG, Phase.DATA_PROFILING],
    Phase.FEATURE_ENG:     [Phase.ADSTOCK_OPT, Phase.MODELING],
    Phase.ADSTOCK_OPT:     [Phase.MODELING, Phase.FEATURE_ENG],
    Phase.MODELING:        [Phase.EVALUATION, Phase.ADSTOCK_OPT],
    Phase.EVALUATION:      [Phase.BUDGET_OPT, Phase.MODELING, Phase.REPORTING],
    Phase.BUDGET_OPT:      [Phase.REPORTING, Phase.MODELING],
    Phase.REPORTING:       [Phase.DONE, Phase.IDLE],
    Phase.USER_INPUT:      [Phase.DATA_LOADING, Phase.MODELING, Phase.REPORTING, Phase.IDLE],
    Phase.DONE:            [Phase.IDLE],
}


# ─────────────────────────────────────────────
# ANALYSIS CONTEXT  (mutable dict the agent populates)
# ─────────────────────────────────────────────

def empty_context() -> Dict[str, Any]:
    return {
        "table_path":       None,
        "kpi_col":          None,
        "channel_cols":     [],
        "date_col":         None,
        "model_type":       None,   # "ols" | "bayesian"
        "r2":               None,
        "iteration_count":  0,
        "phase_history":    [],
        "findings":         [],     # agent's logged insights
        "pending_questions": [],    # questions to ask user
    }


def empty_plan() -> Dict[str, Any]:
    """Empty plan structure — populated by the planner node."""
    return {
        "goal":             "",     # high-level objective
        "current_phase":    "",     # phase this plan targets
        "steps":            [],     # list of {action, tool, expected_output, status}
        "success_criteria":  [],    # measurable criteria for phase completion
        "current_step_idx": 0,      # which step we're executing
        "revision_count":   0,      # how many times the plan was revised
    }


def empty_reflection() -> Dict[str, Any]:
    """Empty reflection structure — populated by the reflection node."""
    return {
        "assessment":       "",     # narrative evaluation of actions vs plan
        "quality_score":    0.0,    # 0.0 – 1.0 quality rating for current phase
        "achievements":     [],     # what was accomplished
        "gaps":             [],     # what's still missing
        "decision":         "",     # "continue" | "phase_complete" | "retry" | "backtrack"
        "reasoning":        "",     # why this decision
    }


def empty_error_recovery() -> Dict[str, Any]:
    """Tracks error state for adaptive retry logic."""
    return {
        "consecutive_failures": 0,
        "failed_tools":         [],     # list of {tool, args, error, timestamp}
        "retry_count":          0,
        "max_retries":          3,
        "alternative_tried":    False,  # whether we attempted an alternative approach
    }


# ─────────────────────────────────────────────
# LANGGRAPH STATE
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    # Message history (LangGraph reducer: append-only)
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # Current workflow phase
    phase: str                          # Phase enum value

    # Routing decision set by agent_node
    next_step: str                      # "tools" | "end" | "reflect" | "planner" | "quality_gate"

    # Mutable analysis context (replaced each update)
    context: Dict[str, Any]

    # Iteration counter (for loop detection / safety)
    iteration: int

    # Whether we're in interactive (chat) mode
    interactive: bool

    # ── Agentic enhancements ─────────────────────

    # Structured plan for the current phase
    plan: Dict[str, Any]

    # Last reflection assessment
    reflection: Dict[str, Any]

    # Per-phase quality scores  {phase_value: float}
    quality_scores: Dict[str, float]

    # Error recovery tracking
    error_recovery: Dict[str, Any]

    # Accumulated results per phase  {phase_value: {...}}
    phase_results: Dict[str, Any]


def initial_state(interactive: bool = True) -> AgentState:
    return {
        "messages": [],
        "phase": Phase.IDLE.value,
        "next_step": "planner",
        "context": empty_context(),
        "iteration": 0,
        "interactive": interactive,
        "plan": empty_plan(),
        "reflection": empty_reflection(),
        "quality_scores": {},
        "error_recovery": empty_error_recovery(),
        "phase_results": {},
    }


# ─────────────────────────────────────────────
# PHASE TRANSITION VALIDATION
# ─────────────────────────────────────────────

def validate_phase_transition(current: str, proposed: str) -> bool:
    """
    Check whether moving from *current* to *proposed* phase is allowed
    according to the PHASE_TRANSITIONS directed graph.
    """
    try:
        current_phase = Phase(current)
    except ValueError:
        return False
    allowed = PHASE_TRANSITIONS.get(current_phase, [])
    return any(p.value == proposed for p in allowed)


def get_next_phase(current: str) -> Optional[str]:
    """Return the primary (first) allowed next phase, or None if done."""
    try:
        current_phase = Phase(current)
    except ValueError:
        return None
    allowed = PHASE_TRANSITIONS.get(current_phase, [])
    return allowed[0].value if allowed else None


# Quality thresholds that the quality gate enforces per phase
PHASE_QUALITY_THRESHOLDS: Dict[str, float] = {
    Phase.DATA_PROFILING.value:  0.3,   # basic profiling done
    Phase.DATA_VALIDATION.value: 0.4,   # data quality checks passed
    Phase.FEATURE_ENG.value:     0.3,   # features created
    Phase.ADSTOCK_OPT.value:     0.4,   # adstock params optimised
    Phase.MODELING.value:        0.5,   # model has reasonable R²
    Phase.EVALUATION.value:      0.4,   # evaluation completed
    Phase.BUDGET_OPT.value:      0.4,   # budget optimised
    Phase.REPORTING.value:       0.3,   # report generated
}
