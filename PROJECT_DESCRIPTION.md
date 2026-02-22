# Agentic MMM — Autonomous Marketing Mix Modelling on Databricks

> **An intelligent, self-planning AI agent that autonomously executes end-to-end Marketing Mix Modelling (MMM) analysis** — from raw data ingestion through adstock optimisation, Bayesian modelling, budget allocation, and executive reporting — running entirely on Databricks with LLM-powered reasoning.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Cognitive Graph — 5-Node Architecture](#3-cognitive-graph--5-node-architecture)
4. [Project Structure](#4-project-structure)
5. [Core Modules](#5-core-modules)
6. [Agent System](#6-agent-system)
7. [Workflow Engine](#7-workflow-engine)
8. [Tool System](#8-tool-system)
9. [Configuration](#9-configuration)
10. [Workflow Phases](#10-workflow-phases)
11. [Data Flow](#11-data-flow)
12. [Key Design Decisions](#12-key-design-decisions)
13. [Quick Start](#13-quick-start)
14. [API Reference](#14-api-reference)
15. [Dependencies](#15-dependencies)

---

## 1. Project Overview

### What Is This?

Agentic MMM is a **fully autonomous AI agent** designed for marketers and data scientists who need to understand the ROI impact of their marketing channels. Instead of writing code manually, users describe their analysis goals in natural language, and the agent:

- 📊 **Loads and profiles** data from Unity Catalog, CSV, Parquet, Excel, or JSON
- 🔍 **Validates and cleans** data autonomously (outlier detection, null handling, deduplication)
- ⚙️ **Engineers features** (time features, weekly aggregation, adstock/saturation transforms)
- 📈 **Fits MMM models** (OLS regression and/or Bayesian MCMC) with adstock parameter optimisation
- 💰 **Optimises budget allocation** using constrained optimisation (scipy SLSQP)
- 📝 **Generates executive summaries** with actionable business insights
- 🛠️ **Creates custom tools at runtime** when built-in tools don't cover a specific analysis need

### What Makes It "Agentic"?

Unlike a simple LLM wrapper that calls tools in a flat loop, this system implements a **5-node cognitive architecture** with genuine autonomous intelligence:

| Capability | Implementation |
|---|---|
| **Plans before acting** | Planner Node creates step-by-step JSON plans via LLM before each phase |
| **Self-evaluates** | Reflection Node scores quality 0–1 and decides continue/retry/advance |
| **Enforces quality** | Quality Gate Node validates phase transitions + minimum quality thresholds |
| **Recovers from errors** | Agent tracks failures and automatically tries alternative approaches |
| **Extends itself** | Creates new tools at runtime using sandboxed code execution |
| **Maintains structured context** | Auto-extracts analysis context from tool results (no fragile parsing) |

### Target Environment

- **Runtime**: Databricks notebooks (Python)
- **LLM**: Databricks-hosted models via `ChatDatabricks` (default: `databricks-llama-4-maverick`)
- **Data**: Unity Catalog tables, CSV, Parquet, Excel, JSON
- **Orchestration**: LangGraph (stateful, graph-based agent orchestration)

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                        │
│   NotebookMMM.ask()  |  .chat()  |  .run_full_analysis()    │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│                     AGENT LAYER                              │
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌────────────┐              │
│  │ Planner  │───▶│  Agent   │◀──▶│   Tools    │              │
│  │  Node    │    │  Node    │    │   Node     │              │
│  └──────────┘    └────┬─────┘    └────────────┘              │
│       ▲               │                                      │
│       │               ▼                                      │
│  ┌──────────┐    ┌──────────┐                                │
│  │ Quality  │◀───│Reflection│                                │
│  │  Gate    │    │  Node    │                                │
│  └──────────┘    └──────────┘                                │
│                                                              │
│  prompts.py  │  builder.py  │  nodes.py  │  state.py         │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│                     TOOL REGISTRY                            │
│                                                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐      │
│  │ Data Tools │  │ MMM Tools  │  │   Meta / Custom    │      │
│  │  (14)      │  │   (9)      │  │   Tools (9+)       │      │
│  └─────┬──────┘  └─────┬──────┘  └─────────┬──────────┘      │
│        │               │                    │                │
│  registry.py  — dynamic registration, invocation, discovery  │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│                     CORE ENGINE                              │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  MMMEngine   │  │  transforms  │  │  executor    │        │
│  │  (data +     │  │  (adstock,   │  │  (sandboxed  │        │
│  │   modelling) │  │   saturation │  │   code exec) │        │
│  └──────────────┘  │   response)  │  └──────────────┘        │
│                    └──────────────┘                           │
│  mmm_engine.py  │  transforms.py  │  executor.py             │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Cognitive Graph — 5-Node Architecture

The agent uses a **LangGraph `StateGraph`** with five functionally distinct nodes connected by conditional routing edges:

```
START → Planner → Agent ⇄ Tools → Reflection ⇄ Agent/Planner → Quality Gate → Planner/END
```

### Node Descriptions

#### 🗺️ Planner Node (`planner`)
- **Purpose**: Creates a structured, step-by-step execution plan for the current workflow phase
- **Input**: Current phase, analysis context, available tools, reflection feedback (if retrying)
- **Output**: JSON plan with `goal`, `steps[]`, `success_criteria[]`
- **LLM Call**: Yes — uses a dedicated planning prompt
- **Routing**: `agent` (normal) or `end` (if in IDLE with interactive question)

#### 🤖 Agent Node (`agent`)
- **Purpose**: Executes the plan by calling tools through the LLM
- **Input**: System prompt (plan-aware), message history, tool bindings
- **Output**: Tool calls or text response
- **LLM Call**: Yes — main conversational LLM with tools bound
- **Error Recovery**:
  - Tracks consecutive failures in `error_recovery` state
  - After ≤3 failures: retries with different approach (`next_step: "agent"`)
  - After >3 failures: escalates to reflection (`next_step: "reflect"`)
- **Safety**: Hard limit of 40 iterations
- **Routing**: `tools` (has tool calls), `reflect` (no tool calls), `agent` (self-retry on error)

#### ⚙️ Tool Node (`tools`)
- **Purpose**: Executes tool calls from the agent and returns results
- **Input**: Last message's tool calls
- **Output**: `ToolMessage` results
- **LLM Call**: No — pure execution
- **Auto Context Extraction**: Automatically populates `state.context` from tool results:
  - `load_data` → sets `table_path`, `channel_cols`, `kpi_col`, `date_col`
  - `inspect_data` → refines column classifications
  - `run_ols_mmm` / `run_bayesian_mmm` → sets `r2`, `model_type`
  - `optimize_budget` → sets `budget_allocation`
- **Phase Transitions**: Advances workflow phase based on successful tool execution, validated against `PHASE_TRANSITIONS` graph
- **Routing**: Always → `agent`

#### 🪞 Reflection Node (`reflect`)
- **Purpose**: Self-evaluates what the agent accomplished vs. the plan
- **Input**: Recent actions, plan, analysis context
- **Output**: Quality score (0.0–1.0), achievements, gaps, and routing decision
- **LLM Call**: Yes — dedicated reflection/evaluation prompt
- **Decisions**:
  - `"phase_complete"` → route to Quality Gate
  - `"continue"` → route back to Agent for more work
  - `"retry"` → route back to Planner to re-plan with feedback
- **Routing**: `quality_gate`, `agent`, or `planner`

#### 🚧 Quality Gate Node (`quality_gate`)
- **Purpose**: Programmatic gate that enforces phase completion standards
- **Input**: Quality scores, phase results, current phase
- **LLM Call**: No — purely programmatic
- **Checks**:
  - Quality score ≥ phase-specific threshold (configurable per phase)
  - Phase transition is valid according to `PHASE_TRANSITIONS` directed graph
- **If quality < threshold**: Routes back to Planner for retry (up to iteration 35)
- **If quality ≥ threshold**: Advances to next phase, resets plan/reflection/error state
- **Routing**: `planner` (next phase or retry) or `end` (all phases complete)

### Quality Thresholds

| Phase | Min Quality | What's Assessed |
|-------|-------------|-----------------|
| Data Profiling | 30% | Basic profiling completed |
| Data Validation | 40% | Data quality checks passed |
| Feature Engineering | 30% | Features created and validated |
| Adstock Optimisation | 40% | Adstock params optimised |
| Modelling | 50% | Model has acceptable R² |
| Evaluation | 40% | Model evaluation completed |
| Budget Optimisation | 40% | Budget allocation optimised |
| Reporting | 30% | Report generated |

---

## 4. Project Structure

```
agentic_mmm/
│
├── __init__.py              # Package exports: NotebookMMM, init_mmm, MMMEngine
├── main.py                  # Entry point: NotebookMMM class, chat loop, full analysis
├── config.py                # Centralized configuration (LLM, agent, security, paths)
├── requirements.txt         # Python dependencies
├── README.md                # Quick start guide
├── PROJECT_DESCRIPTION.md   # This file
│
├── agent/                   # Agent assembly + prompts
│   ├── __init__.py
│   ├── builder.py           # LangGraph graph construction (5-node wiring)
│   └── prompts.py           # Dynamic system prompts (phase-aware, plan-aware)
│
├── core/                    # Data engine + mathematical transforms
│   ├── __init__.py
│   ├── mmm_engine.py        # MMMEngine: load → profile → model → optimise (704 lines)
│   ├── transforms.py        # Adstock (geometric, Weibull), saturation (Hill, log, negexp)
│   └── executor.py          # SafeCodeExecutor: sandboxed Python execution
│
├── tools/                   # Tool definitions for LLM function calling
│   ├── __init__.py
│   ├── registry.py          # ToolRegistry: dynamic tool management + runtime creation
│   ├── data_tools.py        # 14 data tools (load, inspect, clean, aggregate, query, etc.)
│   ├── mmm_tools.py         # 9 MMM tools (adstock, OLS, Bayesian, budget, scenarios)
│   └── custom_tools.py      # 9 meta-tools (create tools, ask user, log notes, etc.)
│
└── workflows/               # State management + node implementations
    ├── __init__.py
    ├── state.py             # AgentState (TypedDict), Phase enum, transitions, quality thresholds
    └── nodes.py             # 5 node builders + 4 routers + helpers
```

---

## 5. Core Modules

### 5.1 `core/mmm_engine.py` — MMMEngine (704 lines)

The `MMMEngine` class is the **central data and modelling engine**. Every tool delegates its heavy lifting to this class. It maintains the loaded DataFrame, profile metadata, fitted model results, and adstock parameters.

#### Capabilities

| Category | Methods |
|----------|---------|
| **Data Loading** | `load_data(path)` — Unity Catalog, CSV, Parquet, Excel, JSON, or Python dict |
| **Profiling** | `_build_profile()`, `_profile_summary()` — auto-detects spend/KPI/date/channel columns using keyword classifiers |
| **Inspection** | `get_status()`, `inspect_data()`, `get_column_stats(col)`, `sample_rows(n)` |
| **EDA** | `get_top_values(...)`, `filter_and_aggregate(...)`, `get_correlation_matrix(cols)`, `detect_outliers(col, method)` |
| **Transforms** | `clean_data(...)`, `add_time_features(date_col)`, `aggregate_to_weekly(...)` |
| **Adstock** | `get_adstock_recommendations()`, `optimize_adstock_parameters(channel, kpi)`, `optimize_all_adstock_parameters(channels, kpi)` |
| **Modelling** | `run_ols_mmm(kpi, channels, use_adstock)`, `run_bayesian_mmm(kpi, channels, n_samples, tune, use_adstock)` |
| **Optimisation** | `optimize_budget(total, channels, min_pct, max_pct)`, `roi_summary()` |
| **Custom Execution** | `execute_custom_query(code)` — delegates to `SafeCodeExecutor` |
| **Context** | `get_data_context()` — formatted markdown for system prompts |

#### Internal State

```python
self.data          # pd.DataFrame — the loaded dataset
self._path         # str — source path for reload
self._profile      # dict — column types, dtypes, classification results
self._model_result # dict — fitted model coefficients, R², residuals
self._adstock_params # dict — optimised decay/half_sat/slope per channel
self._iteration    # int — analysis iteration counter
self._analysis_log # list — timestamped insight log
```

### 5.2 `core/transforms.py` — Signal Transforms (164 lines)

Implements the mathematical transformations fundamental to MMM:

#### Adstock Functions
| Function | Formula | Use Case |
|----------|---------|----------|
| `adstock_geometric(series, decay)` | `y[t] = x[t] + decay × y[t-1]` | Classic carryover effect (TV, radio) |
| `adstock_weibull_pdf(series, shape, scale)` | Weibull PDF convolution | Flexible lag distributions (digital, delayed effects) |

#### Saturation / Response Curves
| Function | Formula | Use Case |
|----------|---------|----------|
| `hill_transform(series, half_sat, slope)` | `x^s / (h^s + x^s)` | S-curve diminishing returns |
| `log_saturation(series, alpha)` | `α × log(1 + x)` | Simple concave response |
| `negative_exponential(series, alpha, gamma)` | `α × (1 - e^(-γx))` | Bounded exponential response |

#### Pipeline & Optimisation
- `apply_transforms(df, channels, ...)` — Applies adstock → saturation pipeline to all channels
- `fit_hill_params(x, y)` — Fits Hill curve via `scipy.optimize.curve_fit`
- `optimize_adstock_params(df, channel, kpi)` — Grid-search over decay values + Hill fitting to find optimal parameters

### 5.3 `core/executor.py` — SafeCodeExecutor (180 lines)

Provides **sandboxed Python execution** for the `execute_query` tool and dynamic tool creation:

- **Security**: Blocks dangerous patterns (`os.`, `subprocess`, `__import__`, `eval`, `open`, etc.) via regex scanning against `FORBIDDEN_PATTERNS`
- **Restricted Globals**: Exposes only safe builtins (`len`, `range`, `sorted`, etc.) plus data science libraries (`pandas`, `numpy`, `scipy`, `sklearn`, `statsmodels`)
- **Result Packaging**: Automatically serialises DataFrames, Series, arrays, dicts, and scalars to JSON-compatible formats
- **Stdout Capture**: Captures `print()` output during execution
- **Variables**: Code uses `df` for the current dataset and stores results in `result`

---

## 6. Agent System

### 6.1 `agent/prompts.py` — Dynamic Prompt Engine

The system prompt is **dynamically constructed** from the current agent state, incorporating:

1. **Base Identity**: Expert Agentic AI Data Analyst persona with behavioural rules (DOs/DON'Ts)
2. **Error Recovery Instructions**: 3-step escalation strategy for tool failures
3. **Tool Creation Guidance**: Template for `create_custom_tool()`
4. **Analysis Context**: Current KPI, channels, R², findings, failed tools
5. **Active Plan Block**: When a plan exists, each step is listed with status markers (`→` for current step)
6. **Phase Guidance**: Phase-specific instructions (e.g., "In DATA_LOADING, call `load_data(path)` first")
7. **Data Context**: Engine-provided dataset summary (shape, columns, dtypes)

#### Phase Guidance System

Each workflow phase has dedicated guidance injected into the prompt:

| Phase | Key Guidance |
|-------|-------------|
| IDLE | Call `get_data_status()` first |
| DATA_LOADING | Call `load_data(path)` with Unity Catalog or file path |
| DATA_PROFILING | Call `inspect_data()`, `get_column_stats()` for each key column |
| DATA_VALIDATION | Check for nulls, outliers, duplicates; call `clean_data()` |
| FEATURE_ENG | Add time features, aggregate to weekly, create interaction terms |
| ADSTOCK_OPT | Optimise adstock parameters for each channel |
| MODELLING | Fit OLS and/or Bayesian MMM, compare R² |
| EVALUATION | Assess model quality, check coefficients, review residuals |
| BUDGET_OPT | Call `optimize_budget()` with business constraints |
| REPORTING | Summarise all findings, provide actionable recommendations |

### 6.2 `agent/builder.py` — Graph Assembly

Assembles the full cognitive agent:

1. **Initialises LLM** via `ChatDatabricks` (Databricks-hosted endpoint)
2. **Builds Tool Registry** by calling factory functions from each tool module
3. **Binds tools** to the LLM for function calling
4. **Creates system message factory** (curried with `MMMEngine`)
5. **Instantiates 5 node functions** via their builder factories
6. **Wires the `StateGraph`** with conditional edges for all routing decisions
7. **Compiles with `MemorySaver` checkpointer** for conversation continuity

---

## 7. Workflow Engine

### 7.1 `workflows/state.py` — State Management

#### `AgentState` (TypedDict)

The central state object that flows through every graph node:

| Field | Type | Purpose |
|-------|------|---------|
| `messages` | `Sequence[BaseMessage]` | Append-only conversation history (LangGraph reducer) |
| `phase` | `str` | Current workflow phase (Phase enum value) |
| `next_step` | `str` | Routing decision: `"tools"`, `"end"`, `"reflect"`, `"planner"`, `"quality_gate"` |
| `context` | `Dict[str, Any]` | Mutable analysis context (KPI, channels, R², findings, etc.) |
| `iteration` | `int` | Global step counter (for safety limits) |
| `interactive` | `bool` | Whether we're in chat mode (affects planner/reflection skip logic) |
| `plan` | `Dict[str, Any]` | Current phase plan: goal, steps, success criteria, progress |
| `reflection` | `Dict[str, Any]` | Last reflection: assessment, quality score, gaps, decision |
| `quality_scores` | `Dict[str, float]` | Accumulated quality scores per phase |
| `error_recovery` | `Dict[str, Any]` | Failure tracking: consecutive failures, failed tools, retry counts |
| `phase_results` | `Dict[str, Any]` | Results accumulated per completed phase |

#### Phase Enum

```python
class Phase(str, Enum):
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
```

#### Phase Transition Graph (Enforced)

```
IDLE → DATA_LOADING → DATA_PROFILING → DATA_VALIDATION → FEATURE_ENG
                                                              ↓
                                                        ADSTOCK_OPT → MODELING → EVALUATION
                                                                                     ↓
                                                                               BUDGET_OPT → REPORTING → DONE
```

Transitions are validated by `validate_phase_transition(current, proposed)` before any phase advance. The agent **cannot** skip phases (e.g., jump from DATA_LOADING directly to MODELING).

### 7.2 `workflows/nodes.py` — Node Implementations

Contains 5 node builder functions and 4 router functions:

| Builder | Returns | LLM? |
|---------|---------|------|
| `build_planner_node(llm, sys_msg_fn)` | `planner_node(state) → dict` | ✅ |
| `build_agent_node(llm_with_tools, sys_msg_fn)` | `agent_node(state) → dict` | ✅ |
| `build_tool_node(registry, console)` | `tool_node(state) → dict` | ❌ |
| `build_reflection_node(llm)` | `reflection_node(state) → dict` | ✅ |
| `build_quality_gate_node(console)` | `quality_gate_node(state) → dict` | ❌ |

| Router | Logic |
|--------|-------|
| `planner_router(state)` | Reads `state.next_step` → `"agent"` or `"end"` |
| `agent_router(state)` | Reads `state.next_step` → `"tools"`, `"reflect"`, `"agent"`, or `"end"` |
| `reflect_router(state)` | Reads `state.next_step` → `"agent"`, `"planner"`, `"quality_gate"`, or `"end"` |
| `gate_router(state)` | Reads `state.next_step` → `"planner"` or `"end"` |

---

## 8. Tool System

### 8.1 `tools/registry.py` — Dynamic Tool Registry

The `ToolRegistry` class manages all tools (built-in + dynamically created):

- **`register(tool)`** / **`register_many(tools)`** — Register `StructuredTool` instances
- **`register_from_spec(spec)`** — Compile a `ToolSpec` (name + code + params) into a live tool at runtime
- **`invoke(name, args)`** — Execute a tool by name with error handling
- **`get_all()`** — Return all tools for LLM binding
- **`list_tools()`** — Discovery: list all tools with descriptions and dynamic/static flag
- **Shared Context** — Key-value store shared across tool calls

### 8.2 Data Tools (14 tools)

| Tool | Description |
|------|-------------|
| `get_data_status` | Check load status, table path, row/col counts, model status |
| `load_data` | Load from Unity Catalog, CSV, Parquet, Excel, JSON |
| `inspect_data` | Full dataset profile: types, stats, detected columns, sample rows |
| `get_adstock_recommendations` | AI recommendations for adstock-eligible columns |
| `get_column_stats` | Detailed single-column statistics |
| `get_top_values` | Top/bottom N values with optional group-by |
| `sample_rows` | Random sample of dataset rows |
| `filter_aggregate` | Filter + aggregate with optional grouping |
| `get_correlation_matrix` | Pearson correlation matrix |
| `detect_outliers` | IQR or Z-score outlier detection |
| `execute_query` | Sandboxed Python/Pandas code execution |
| `clean_data` | Remove duplicates, handle nulls |
| `add_time_features` | Extract week, month, quarter, year from datetime |
| `aggregate_weekly` | Aggregate transaction data to weekly time series |

### 8.3 MMM Tools (9 tools)

| Tool | Description |
|------|-------------|
| `optimize_adstock_parameters` | Grid-search optimal decay + Hill params for one channel |
| `optimize_all_adstock` | Batch adstock optimisation for multiple channels |
| `run_ols_mmm` | Fit OLS linear regression MMM (with optional adstock transforms) |
| `run_bayesian_mmm` | Fit Bayesian MMM via PyMC MCMC sampling |
| `roi_summary` | ROI breakdown per channel from fitted model |
| `optimize_budget` | Constrained budget allocation (scipy SLSQP) |
| `simulate_scenario` | Simulate revenue for a custom budget scenario |
| `compare_scenarios` | Side-by-side comparison of two budget scenarios |
| *(model evaluation)* | Part of the OLS/Bayesian output (R², coefficients, residuals) |

### 8.4 Meta / Custom Tools (9 tools)

| Tool | Description |
|------|-------------|
| `create_custom_tool` | **Runtime tool creation** — write Python code, specify params, creates a live tool |
| `list_custom_tools` | List all dynamically created tools |
| `remove_custom_tool` | Delete a dynamic tool |
| `inspect_custom_tool` | View source code and parameters of a dynamic tool |
| `ask_user` | Ask the user a question (blocking, with optional choices) |
| `ask_user_to_choose` | Present multiple-choice options to the user |
| `add_analysis_note` | Log an insight/finding/hypothesis with category tags |
| `get_analysis_history` | Retrieve all logged analysis notes |
| `list_all_tools` | List every available tool (built-in + dynamic) |

---

## 9. Configuration

### `config.py`

Central configuration with dataclass-based settings:

#### LLM Settings (`LLMConfig`)
```python
endpoint: str = "databricks-llama-4-maverick"
temperature: float = 0.2
max_tokens: int = 4096
max_agent_steps: int = 25
max_iterations: int = 10
```

#### Agent Behaviour (`AgentConfig`)
```python
phases: List[str]                    # Workflow phases
max_eda_iterations: int = 5          # EDA iteration limit
min_rows_for_mmm: int = 52           # Minimum rows for meaningful MMM
confirm_expensive_ops: bool = True   # Ask before expensive operations
```

#### Keyword Classifiers

The engine uses keyword matching to auto-classify columns:

| Classifier | Keywords (subset) | Used For |
|------------|-------------------|----------|
| `SPEND_KEYWORDS` | spend, cost, budget, tv, digital, social, search, display | Identifying media spend columns |
| `KPI_KEYWORDS` | revenue, sales, conversion, orders, transactions | Identifying target/KPI columns |
| `TIME_KEYWORDS` | date, week, month, year, period, time | Identifying datetime columns |
| `CHANNEL_KEYWORDS` | channel, media, source, campaign, platform | Identifying channel grouping |

#### Security (`FORBIDDEN_PATTERNS`)
Regex patterns blocked in sandboxed execution:
- `__import__`, `exec(`, `eval(`, `compile(`
- `open(`, `subprocess`, `os.`, `sys.`, `shutil`
- `socket`, `urllib`, `requests`

#### Path Configuration (`PathConfig`)
```python
custom_tools_dir: str = "/tmp/agentic_mmm_custom_tools"
output_dir: str = "/tmp/agentic_mmm_outputs"
checkpoint_dir: str = "/tmp/agentic_mmm_checkpoints"
```

---

## 10. Workflow Phases

The agent progresses through a structured pipeline. Each phase has specific goals, tools, and quality thresholds:

### Phase 1: IDLE → DATA_LOADING
- **Goal**: Load the dataset from user-specified source
- **Key Tool**: `load_data(path)`
- **Success**: Data loaded, columns and types identified

### Phase 2: DATA_PROFILING
- **Goal**: Understand the dataset structure and statistics
- **Key Tools**: `inspect_data()`, `get_column_stats()`, `sample_rows()`
- **Success**: All columns profiled, spend/KPI/date columns identified

### Phase 3: DATA_VALIDATION
- **Goal**: Ensure data quality
- **Key Tools**: `detect_outliers()`, `clean_data()`, `get_correlation_matrix()`
- **Success**: Nulls handled, duplicates removed, outliers identified

### Phase 4: FEATURE_ENGINEERING
- **Goal**: Prepare data for modelling
- **Key Tools**: `add_time_features()`, `aggregate_weekly()`, `execute_query()`
- **Success**: Time features added, data at weekly granularity

### Phase 5: ADSTOCK_OPTIMISATION
- **Goal**: Find optimal adstock decay and saturation parameters per channel
- **Key Tools**: `optimize_adstock_parameters()`, `optimize_all_adstock()`
- **Success**: Adstock parameters optimised with R² for each channel

### Phase 6: MODELLING
- **Goal**: Fit marketing mix model(s)
- **Key Tools**: `run_ols_mmm()`, `run_bayesian_mmm()`
- **Success**: Model fitted with acceptable R², coefficients make business sense

### Phase 7: EVALUATION
- **Goal**: Assess model quality and business validity
- **Key Tools**: `roi_summary()`, `execute_query()` (for residual analysis)
- **Success**: Model validated, ROI per channel computed

### Phase 8: BUDGET_OPTIMISATION
- **Goal**: Optimise marketing budget allocation
- **Key Tools**: `optimize_budget()`, `simulate_scenario()`, `compare_scenarios()`
- **Success**: Optimal allocation found with revenue estimate

### Phase 9: REPORTING
- **Goal**: Generate executive summary with recommendations
- **Key Tools**: `add_analysis_note()`, `get_analysis_history()`
- **Success**: Clear summary with actionable insights delivered

---

## 11. Data Flow

### Single `ask()` Call

```
User message
  ↓
Planner creates/updates plan for current phase
  ↓
Agent receives plan-aware system prompt + message history
  ↓
LLM generates tool call(s)
  ↓
Tool Node executes via ToolRegistry → MMMEngine → result
  ↓
Auto-context extraction updates state.context
  ↓
Phase transition check (validated against PHASE_TRANSITIONS)
  ↓
Agent receives tool result, interprets, calls next tool or stops
  ↓
Reflection evaluates: score quality, decide continue/retry/advance
  ↓
Quality Gate: enforce threshold, advance phase or retry
  ↓
Loop back to Planner for next phase, or END
```

### `run_full_analysis()` Flow

```
1. Pre-populate context (KPI col, channel cols from user hints)
2. Set phase = DATA_LOADING
3. Single call to ask() — the cognitive graph manages everything:
   └─ Planner plans DATA_LOADING
      └─ Agent loads data
      └─ Reflection: phase complete
      └─ Quality Gate: advance to DATA_PROFILING
      └─ Planner plans DATA_PROFILING
      └─ Agent profiles data
      └─ ... (continues through all phases)
      └─ Quality Gate: REPORTING complete → END
4. State persisted for follow-up questions
```

---

## 12. Key Design Decisions

### Why LangGraph over LangChain AgentExecutor?
LangGraph provides **explicit graph-based control flow** with conditional edges, typed state, and checkpointing. This is essential for the multi-node cognitive architecture — a flat AgentExecutor cannot express "reflect after acting, then gate before advancing".

### Why Databricks LLMs?
The system is designed for enterprise environments where data governance matters. Databricks-hosted models (`ChatDatabricks`) ensure data doesn't leave the workspace. The agent can access Unity Catalog tables directly.

### Why Runtime Tool Creation?
Marketing data is highly variable. The built-in tools cover 80% of needs, but analysts often need bespoke analysis (seasonality detection, custom segmentation, etc.). The `create_custom_tool` meta-tool lets the agent manufacture exactly the tool it needs, validated through `SafeCodeExecutor`.

### Why Sandboxed Execution?
The agent can generate and execute arbitrary Python code. Without sandboxing, this would be a massive security risk. `SafeCodeExecutor` blocks system access, network calls, and file operations while allowing data science operations.

### Why Per-Phase Quality Gates?
Without quality gates, the agent could skip phases or advance with inadequate work. The programmatic quality gate ensures minimum standards before allowing phase transitions, preventing garbage-in/garbage-out model fitting.

### Why Plan → Act → Reflect?
The Plan-Act-Reflect cycle mirrors expert human reasoning. Planning reduces wasted tool calls. Reflection catches when the agent is stuck or going in circles. Together they produce significantly more reliable end-to-end analysis than a flat tool-calling loop.

---

## 13. Quick Start

### On Databricks

```python
# In a Databricks notebook cell:
from agentic_mmm import init_mmm

# Initialize with a Unity Catalog table
mmm = init_mmm(
    table_path="catalog.schema.marketing_data",
    kpi_col="revenue"
)

# Interactive chat
mmm.chat()

# Or run full autonomous analysis
result = mmm.run_full_analysis(
    kpi_col="revenue",
    channel_cols="tv_spend, digital_spend, social_spend, search_spend"
)
```

### Key API Methods

```python
# Ask a specific question
mmm.ask("Which channel has the highest ROI?")

# Load different data
mmm.load_data("catalog.schema.different_table")

# Run with custom LLM endpoint
mmm = init_mmm(
    table_path="...",
    llm_endpoint="databricks-meta-llama-3-1-70b-instruct"
)
```

---

## 14. API Reference

### `NotebookMMM` Class

| Method | Description |
|--------|-------------|
| `load_data(path)` | Load data from any supported source |
| `quick_profile()` | Print a Rich-formatted data profile table |
| `ask(question, thread_id?, phase?)` | Send a question to the cognitive agent |
| `run_full_analysis(kpi_col?, channel_cols?)` | Trigger autonomous end-to-end MMM |
| `chat()` | Start interactive chat session |
| `engine` | Access the underlying `MMMEngine` instance |

### `init_mmm()` Factory

```python
def init_mmm(
    table_path: str = "samples.bakehouse.sales_transactions",
    kpi_col: str = "totalPrice",
    llm_endpoint: str = "databricks-llama-4-maverick",
) -> NotebookMMM
```

---

## 15. Dependencies

| Category | Packages |
|----------|----------|
| **LLM / Agent** | `langchain-core`, `langchain-databricks`, `langgraph` |
| **Data** | `pandas`, `numpy`, `scipy`, `pyarrow` |
| **Modelling** | `scikit-learn`, `statsmodels`, `pymc` (Bayesian) |
| **Databricks** | `databricks-sdk`, `pyspark` (optional) |
| **UI** | `rich` (terminal formatting), `ipywidgets` (notebook) |
| **Schema** | `pydantic` |

---

*Built for Databricks. Powered by LangGraph. Designed for autonomous intelligence.*
