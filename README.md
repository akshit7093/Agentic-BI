# Agentic MMM System v2.0

A fully agentic, intelligent Marketing Mix Modelling (MMM) system built on LangGraph + Databricks. The agent autonomously profiles data, engineers features, fits probabilistic models, optimises budgets, and can create its own tools on-the-fly.

---

## Architecture

```
agentic_mmm/
├── __init__.py
├── config.py                    ← Centralised configuration
├── requirements.txt
│
├── core/
│   ├── mmm_engine.py            ← Data loading, profiling, modelling, optimisation
│   ├── transforms.py            ← Adstock, Hill, saturation transforms
│   └── executor.py              ← Sandboxed Python execution
│
├── tools/
│   ├── registry.py              ← Dynamic tool registry (add tools at runtime)
│   ├── data_tools.py            ← Load, inspect, EDA, transform tools
│   ├── mmm_tools.py             ← Adstock opt, OLS/Bayesian MMM, budget opt
│   └── custom_tools.py          ← Meta-tools: create tools, ask user, log notes
│
├── workflows/
│   ├── state.py                 ← AgentState, Phase enum, phase transitions
│   └── nodes.py                 ← agent_node, tool_node, router
│
├── agent/
│   ├── prompts.py               ← Dynamic system prompt (phase-aware)
│   └── builder.py               ← Assemble LangGraph graph
│
└── main.py                      ← NotebookMMM entry point, init_mmm()
```

---

## Quick Start

```python
from agentic_mmm import init_mmm

# Initialise with a Databricks Unity Catalog table
nb = init_mmm(
    table="catalog.schema.mmm_weekly_spend",
    kpi_col="revenue",
    llm_endpoint="databricks-llama-4-maverick",
)

# One-shot questions
nb.ask("Profile this dataset and tell me if it's suitable for MMM")
nb.ask("Which channels have the highest ROI?")
nb.ask("Optimise my $500,000 monthly budget across all channels")

# Full autonomous end-to-end analysis
nb.run_full_analysis(kpi_col="revenue")

# Interactive chat loop
nb.chat()
```

---

## Key Features

### 🧠 Intelligent Data Profiling
The agent automatically:
- Detects spend, KPI, datetime, and channel columns by keyword matching
- Checks data quality (nulls, outliers, skewness)
- Determines if aggregation is needed (transaction → weekly time series)
- Warns if data volume is insufficient for reliable MMM

### ⚙️ Dynamic Workflow Phases
The agent moves through defined phases autonomously:
```
IDLE → DATA_LOADING → DATA_PROFILING → DATA_VALIDATION →
FEATURE_ENGINEERING → ADSTOCK_OPTIMISATION → MODELING →
EVALUATION → BUDGET_OPTIMISATION → REPORTING
```
Each phase has specific guidance injected into the system prompt.

### 🔧 On-the-Run Custom Tools
The agent can CREATE new tools mid-analysis:

```python
# Agent creates this autonomously, OR you create it manually:
nb.add_tool(
    name="detect_seasonality",
    description="Test for weekly/monthly seasonality using autocorrelation",
    code="""
def tool_fn(column, lags=52):
    import json, numpy as np
    from pandas import Series
    s = df[column].values
    acf = [np.corrcoef(s[:-lag], s[lag:])[0, 1] for lag in range(1, lags+1)]
    peak_lag = int(np.argmax(np.abs(acf))) + 1
    return json.dumps({
        "peak_autocorrelation_lag": peak_lag,
        "peak_acf_value": round(acf[peak_lag-1], 4),
        "likely_seasonality": f"{peak_lag}-period cycle"
    })
""",
    params={"column": "str", "lags": "int"},
)
```

### 📊 Probabilistic Modelling
- **OLS/Ridge**: Fast iteration (no PyMC required)
- **Bayesian MMM (PyMC)**: Full posterior distributions, uncertainty quantification
- **Adstock optimisation**: Grid-search + scipy for decay, half-saturation, slope
- **Budget optimisation**: Differential evolution + SLSQP for revenue maximisation

### 🗣️ User Interaction
The agent asks smart questions when genuinely needed:
- Column disambiguation
- Budget parameters
- Model selection (OLS vs Bayesian)
- Confirmation before expensive MCMC runs

---

## Configuration (`config.py`)

```python
from agentic_mmm.config import LLM_CFG, AGENT_CFG

LLM_CFG.endpoint = "databricks-claude-3-5-sonnet"
LLM_CFG.temperature = 0.1
LLM_CFG.max_agent_steps = 30
AGENT_CFG.min_rows_for_mmm = 104  # require 2 years of weekly data
```

---

## Direct Engine Access (no agent)

```python
from agentic_mmm.core.mmm_engine import MMMEngine

engine = MMMEngine()
engine.load_data("catalog.schema.spend_data")
print(engine.get_column_stats("tv_spend"))
print(engine.optimize_adstock_parameters("tv_spend", "revenue"))
print(engine.run_ols_mmm("revenue", ["tv_spend", "digital_spend", "radio_spend"]))
print(engine.optimize_budget(1_000_000, ["tv_spend", "digital_spend", "radio_spend"]))
```

---

## Supported Data Sources

| Format | Example |
|--------|---------|
| Unity Catalog | `catalog.schema.table` |
| CSV | `/dbfs/mnt/data/mmm.csv` |
| Parquet | `/dbfs/mnt/data/mmm.parquet` |
| Excel | `/dbfs/mnt/data/mmm.xlsx` |
| JSON | `/dbfs/mnt/data/mmm.json` |

---

## Available Tools (37 total)

### Data Tools
| Tool | Description |
|------|-------------|
| `get_data_status` | Check data load state |
| `load_data` | Load from any source |
| `inspect_data` | Full profile with EDA |
| `get_column_stats` | Per-column statistics |
| `get_top_values` | Top N with groupby |
| `sample_rows` | Random sample |
| `filter_aggregate` | Filter + agg |
| `get_correlation_matrix` | Pearson correlation |
| `detect_outliers` | IQR / Z-score |
| `execute_query` | Custom pandas code |
| `clean_data` | Remove nulls/dupes |
| `add_time_features` | Week/month/quarter |
| `aggregate_weekly` | Resample to weekly |

### MMM Tools
| Tool | Description |
|------|-------------|
| `get_adstock_recommendations` | Column suitability analysis |
| `optimize_adstock_parameters` | Single channel decay/hill opt |
| `optimize_all_adstock_parameters` | All channels at once |
| `run_ols_mmm` | Fast Ridge regression MMM |
| `run_bayesian_mmm` | Full Bayesian MMM (PyMC) |
| `roi_summary` | Ranked ROI table |
| `optimize_budget` | Revenue-maximising allocation |
| `simulate_scenario` | Custom budget scenario |
| `compare_scenarios` | A/B budget comparison |

### Meta Tools (Agent Self-Extension)
| Tool | Description |
|------|-------------|
| `create_custom_tool` | Build a new tool on the fly |
| `list_custom_tools` | See all dynamic tools |
| `inspect_custom_tool` | View tool source code |
| `remove_custom_tool` | Remove a dynamic tool |
| `ask_user` | Get user clarification |
| `ask_user_to_choose` | Present options to user |
| `add_analysis_note` | Log insight to history |
| `get_analysis_history` | Review all logged notes |
| `list_all_tools` | Full tool inventory |

---

## Installation

```bash
pip install langgraph langchain-core databricks-langchain \
            pandas numpy scipy scikit-learn rich pydantic \
            pymc arviz  # optional for Bayesian modelling
```

---

## Environment Variables

```bash
export DATABRICKS_HOST=https://your-workspace.azuredatabricks.net
export DATABRICKS_TOKEN=dapi...
```

Or set in notebook:
```python
import os
os.environ["DATABRICKS_HOST"] = "https://..."
os.environ["DATABRICKS_TOKEN"] = "dapi..."
```
