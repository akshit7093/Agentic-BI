# =============================================================
# config.py — Centralized configuration for Agentic MMM System
# =============================================================

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─────────────────────────────────────────────
# LLM SETTINGS
# ─────────────────────────────────────────────
@dataclass
class LLMConfig:
    endpoint: str = "databricks-llama-4-maverick"
    temperature: float = 0.2
    max_tokens: int = 4096
    max_agent_steps: int = 25          # max tool-call rounds per turn
    max_iterations: int = 10           # max analysis iterations


# ─────────────────────────────────────────────
# AGENT BEHAVIOUR
# ─────────────────────────────────────────────
@dataclass
class AgentConfig:
    # Workflow phases the agent cycles through
    phases: List[str] = field(default_factory=lambda: [
        "data_loading",
        "data_understanding",
        "data_validation",
        "feature_engineering",
        "modeling",
        "optimization",
        "reporting",
    ])
    # How many EDA iterations before moving to modeling
    max_eda_iterations: int = 5
    # Minimum rows for MMM to be meaningful
    min_rows_for_mmm: int = 52
    # Ask user for confirmation before expensive operations
    confirm_expensive_ops: bool = True


# ─────────────────────────────────────────────
# KEYWORD CLASSIFIERS  (used in MMMCore)
# ─────────────────────────────────────────────
SPEND_KEYWORDS: List[str] = [
    "spend", "cost", "budget", "investment", "ad_spend",
    "media", "tv", "digital", "radio", "print", "social",
    "search", "display", "video", "email", "affiliate",
    "influencer", "content", "promotion", "marketing",
]

KPI_KEYWORDS: List[str] = [
    "revenue", "sales", "conversion", "orders", "transactions",
    "quantity", "price", "total", "kpi", "target", "goal",
    "outcome", "response",
]

TIME_KEYWORDS: List[str] = [
    "date", "week", "month", "year", "period", "time", "day",
]

CHANNEL_KEYWORDS: List[str] = [
    "channel", "media", "source", "campaign", "platform",
    "network", "publisher",
]


# ─────────────────────────────────────────────
# SAFE CODE EXECUTION
# ─────────────────────────────────────────────
FORBIDDEN_PATTERNS: List[str] = [
    r"__import__", r"exec\s*\(", r"eval\s*\(", r"compile\s*\(",
    r"open\s*\(", r"subprocess", r"os\.", r"sys\.", r"shutil",
    r"import\s+os", r"import\s+sys", r"import\s+subprocess",
    r"socket", r"urllib", r"requests",
]

ALLOWED_IMPORTS: List[str] = [
    "pd", "pandas", "np", "numpy", "math", "json", "re",
    "datetime", "scipy", "sklearn", "statsmodels",
]


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
@dataclass
class PathConfig:
    custom_tools_dir: str = "/tmp/agentic_mmm_custom_tools"
    output_dir: str = "/tmp/agentic_mmm_outputs"
    checkpoint_dir: str = "/tmp/agentic_mmm_checkpoints"


# ─────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────
DEFAULT_TABLE: str = "samples.bakehouse.sales_transactions"
DEFAULT_KPI_COL: str = "totalPrice"
DEFAULT_LLM_ENDPOINT: str = "databricks-llama-4-maverick"

LLM_CFG = LLMConfig()
AGENT_CFG = AgentConfig()
PATH_CFG = PathConfig()
