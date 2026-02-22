# =============================================================
# tools/mmm_tools.py — MMM modelling, adstock, budget optimisation tools
# =============================================================

import json
from typing import List, Optional

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from ..core.mmm_engine import MMMEngine


# ─── Input schemas ────────────────────────────────────────────

class AdstockOptInput(BaseModel):
    channel_col: str = Field(description="Channel / spend column")
    kpi_col: str = Field(description="KPI / outcome column")

class AllAdstockOptInput(BaseModel):
    channel_cols: str = Field(description="Comma-separated channel column names")
    kpi_col: str = Field(description="KPI / outcome column")

class RunOLSInput(BaseModel):
    kpi_col: str = Field(description="KPI column (dependent variable)")
    channel_cols: str = Field(description="Comma-separated channel columns")
    use_adstock: bool = Field(default=True, description="Apply adstock + Hill transforms before fitting")

class RunBayesianInput(BaseModel):
    kpi_col: str = Field(description="KPI column")
    channel_cols: str = Field(description="Comma-separated channel columns")
    n_samples: int = Field(default=500, description="MCMC samples per chain")
    tune: int = Field(default=200, description="Burn-in / tuning steps")
    use_adstock: bool = Field(default=True, description="Apply transforms before fitting")

class ROISummaryInput(BaseModel):
    pass

class OptimizeBudgetInput(BaseModel):
    total_budget: float = Field(description="Total budget to allocate")
    channel_cols: str = Field(description="Comma-separated channel names to include in optimisation")
    min_pct: float = Field(default=0.05, description="Minimum % per channel (0.05 = 5%)")
    max_pct: float = Field(default=0.60, description="Maximum % per channel (0.60 = 60%)")

class ScenarioInput(BaseModel):
    budgets: str = Field(description="JSON string: {channel: budget_amount, ...}")

class CompareScenariosInput(BaseModel):
    scenario_a: str = Field(description="JSON: {channel: budget}")
    scenario_b: str = Field(description="JSON: {channel: budget}")


# ─────────────────────────────────────────────
def build_mmm_tools(engine: MMMEngine) -> List[StructuredTool]:
    """Return all MMM modelling tools wired to *engine*."""

    def _j(obj) -> str:
        return json.dumps(obj, default=str)

    def _chan(s: str) -> List[str]:
        return [c.strip() for c in s.split(",") if c.strip()]

    # ── Adstock parameter optimisation ──

    def optimize_adstock(channel_col: str, kpi_col: str) -> str:
        return _j(engine.optimize_adstock_parameters(channel_col.strip(), kpi_col.strip()))

    def optimize_all_adstock(channel_cols: str, kpi_col: str) -> str:
        return _j(engine.optimize_all_adstock_parameters(_chan(channel_cols), kpi_col.strip()))

    # ── Modelling ──

    def run_ols(kpi_col: str, channel_cols: str, use_adstock: bool = True) -> str:
        return _j(engine.run_ols_mmm(kpi_col.strip(), _chan(channel_cols), use_adstock))

    def run_bayesian(
        kpi_col: str,
        channel_cols: str,
        n_samples: int = 500,
        tune: int = 200,
        use_adstock: bool = True,
    ) -> str:
        return _j(engine.run_bayesian_mmm(kpi_col.strip(), _chan(channel_cols), n_samples, tune, use_adstock))

    def roi_summary() -> str:
        return _j(engine.roi_summary())

    # ── Budget optimisation ──

    def optimize_budget(
        total_budget: float,
        channel_cols: str,
        min_pct: float = 0.05,
        max_pct: float = 0.60,
    ) -> str:
        return _j(engine.optimize_budget(float(total_budget), _chan(channel_cols), min_pct, max_pct))

    def simulate_scenario(budgets: str) -> str:
        """Simulate a budget scenario and return estimated revenue."""
        if not engine.model_results.get("success"):
            return _j({"success": False, "error": "Fit a model first."})
        try:
            import numpy as np
            bdict = json.loads(budgets)
            rois = {c: r for c, r in zip(engine.model_results["channels"], engine.model_results.get("roi_per_unit", engine.model_results.get("coefficients", [])))}
            total_rev = sum(rois.get(c, 0) * (float(b) ** 0.5) for c, b in bdict.items())
            return _j({
                "success": True,
                "scenario": bdict,
                "estimated_revenue_index": round(float(total_rev), 4),
                "note": "Revenue index — relative to current model scale",
            })
        except Exception as exc:
            return _j({"success": False, "error": str(exc)})

    def compare_scenarios(scenario_a: str, scenario_b: str) -> str:
        """Compare two budget scenarios."""
        try:
            import numpy as np
            if not engine.model_results.get("success"):
                return _j({"success": False, "error": "Fit a model first."})
            rois = {c: r for c, r in zip(engine.model_results["channels"], engine.model_results.get("roi_per_unit", engine.model_results.get("coefficients", [])))}
            def _rev(bdict):
                return sum(rois.get(c, 0) * (float(b) ** 0.5) for c, b in bdict.items())
            a, b = json.loads(scenario_a), json.loads(scenario_b)
            ra, rb = float(_rev(a)), float(_rev(b))
            return _j({
                "success": True,
                "scenario_a": {"budget": a, "estimated_revenue_index": round(ra, 4)},
                "scenario_b": {"budget": b, "estimated_revenue_index": round(rb, 4)},
                "lift_b_over_a_pct": round((rb - ra) / (abs(ra) + 1e-10) * 100, 2),
                "winner": "B" if rb > ra else "A",
            })
        except Exception as exc:
            return _j({"success": False, "error": str(exc)})

    return [
        StructuredTool(
            name="optimize_adstock_parameters",
            func=optimize_adstock,
            args_schema=AdstockOptInput,
            description=(
                "Grid-search optimal adstock decay + Hill saturation parameters for one channel vs KPI. "
                "Run this BEFORE fitting an MMM for each spend column."
            ),
        ),
        StructuredTool(
            name="optimize_all_adstock_parameters",
            func=optimize_all_adstock,
            args_schema=AllAdstockOptInput,
            description="Optimise adstock + Hill parameters for ALL channels at once vs a KPI column.",
        ),
        StructuredTool(
            name="run_ols_mmm",
            func=run_ols,
            args_schema=RunOLSInput,
            description=(
                "Fit a fast Ridge-regression MMM (no PyMC required). "
                "Good for quick iteration and data with few rows. Returns R², coefficients, ROI table."
            ),
        ),
        StructuredTool(
            name="run_bayesian_mmm",
            func=run_bayesian,
            args_schema=RunBayesianInput,
            description=(
                "Fit a full Bayesian MMM using PyMC (MCMC sampling). "
                "Returns posterior means/std, ROI per unit, R². Falls back to OLS if PyMC unavailable."
            ),
        ),
        StructuredTool(
            name="roi_summary",
            func=roi_summary,
            args_schema=ROISummaryInput,
            description="Show ranked ROI table from the most recently fitted model.",
        ),
        StructuredTool(
            name="optimize_budget",
            func=optimize_budget,
            args_schema=OptimizeBudgetInput,
            description=(
                "Find optimal budget allocation across channels to maximise predicted revenue. "
                "Requires a fitted model. Returns allocation amounts, percentages, and revenue lift %."
            ),
        ),
        StructuredTool(
            name="simulate_scenario",
            func=simulate_scenario,
            args_schema=ScenarioInput,
            description=(
                'Simulate revenue for a custom budget scenario. '
                'Input: JSON string {"channel_name": budget_amount, ...}. '
                "Requires a fitted model."
            ),
        ),
        StructuredTool(
            name="compare_scenarios",
            func=compare_scenarios,
            args_schema=CompareScenariosInput,
            description=(
                "Compare two budget scenarios A and B. Returns revenue index for each and the winner. "
                'Input: JSON strings {"channel": budget}.'
            ),
        ),
    ]
