# =============================================================
# tools/data_tools.py — Data loading, inspection, EDA, transformation tools
# =============================================================

import json
from typing import Any, List, Optional

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from ..core.mmm_engine import MMMEngine


# ─── Input schemas ────────────────────────────────────────────

class LoadDataInput(BaseModel):
    path: str = Field(description="Unity Catalog (catalog.schema.table), CSV, Parquet, Excel, or JSON path")

class GetStatusInput(BaseModel):
    pass

class InspectDataInput(BaseModel):
    pass

class GetAdstockRecsInput(BaseModel):
    pass

class ColumnStatsInput(BaseModel):
    column: str = Field(description="Column name to analyse")


class FilterAggInput(BaseModel):
    column: str = Field(description="Column to aggregate")
    agg: str = Field(default="sum", description="Aggregation: sum | mean | count | max | min | std")
    filter_col: Optional[str] = Field(default=None, description="Column to filter on")
    filter_val: Optional[Any] = Field(default=None, description="Value to keep")
    group_by: Optional[str] = Field(default=None, description="Group by this column")

class CorrelationInput(BaseModel):
    columns: Optional[str] = Field(default=None, description="Comma-separated column names (leave empty for all numeric)")

class OutlierInput(BaseModel):
    column: str = Field(description="Column to check for outliers")
    method: str = Field(default="iqr", description="Method: iqr | zscore")

class ExecuteQueryInput(BaseModel):
    code: str = Field(description="Python/Pandas code. Use `df` for data. Store final result in `result` variable.")

class CleanDataInput(BaseModel):
    drop_nulls: bool = Field(default=False, description="Drop rows with any null values")
    fill_value: Optional[float] = Field(default=None, description="Fill numeric nulls with this value")
    drop_duplicates: bool = Field(default=True, description="Remove duplicate rows")

class AddTimeFeaturesInput(BaseModel):
    date_col: str = Field(description="Name of the datetime column")

class AggregateWeeklyInput(BaseModel):
    date_col: str = Field(description="Datetime column to resample by")
    value_cols: str = Field(description="Comma-separated column names to aggregate")
    agg: str = Field(default="sum", description="Aggregation: sum | mean | max | min")


# ─────────────────────────────────────────────
def build_data_tools(engine: MMMEngine) -> List[StructuredTool]:
    """Return all data-layer tools wired to *engine*."""

    def _j(obj) -> str:
        return json.dumps(obj, default=str)

    def load_data(path: str) -> str:
        return _j(engine.load_data(path.strip()))

    def get_status() -> str:
        return _j(engine.get_status())

    def inspect_data() -> str:
        return _j(engine.inspect_data())

    def get_adstock_recs() -> str:
        return _j(engine.get_adstock_recommendations())


    def filter_agg(
        column: str,
        agg: str = "sum",
        filter_col: Optional[str] = None,
        filter_val: Optional[Any] = None,
        group_by: Optional[str] = None,
    ) -> str:
        return _j(engine.filter_and_aggregate(column.strip(), agg, filter_col, filter_val, group_by))


    def detect_outliers(column: str, method: str = "iqr") -> str:
        return _j(engine.detect_outliers(column.strip(), method))
    def execute_query(code: str) -> str:
        return _j(engine.execute_custom_query(code))

    def clean_data(
        drop_nulls: bool = False,
        fill_value: Optional[float] = None,
        drop_duplicates: bool = True,
    ) -> str:
        return _j(engine.clean_data(drop_nulls, fill_value, drop_duplicates))

    def add_time_features(date_col: str) -> str:
        return _j(engine.add_time_features(date_col.strip()))

    def aggregate_weekly(date_col: str, value_cols: str, agg: str = "sum") -> str:
        cols = [c.strip() for c in value_cols.split(",")]
        return _j(engine.aggregate_to_weekly(date_col.strip(), cols, agg))

    return [
        StructuredTool(
            name="get_data_status",
            func=get_status,
            args_schema=GetStatusInput,
            description=(
                "Check whether data is loaded, get current table path, row/column counts, "
                "model fit status, and analysis iteration count. ALWAYS call this FIRST."
            ),
        ),
        StructuredTool(
            name="load_data",
            func=load_data,
            args_schema=LoadDataInput,
            description=(
                "Load data from Unity Catalog (catalog.schema.table), CSV, Parquet, Excel, or JSON. "
                "Returns column names, dtypes, detected spend/KPI columns, and null percentages."
            ),
        ),
        StructuredTool(
            name="inspect_data",
            func=inspect_data,
            args_schema=InspectDataInput,
            description=(
                "Get full dataset profile: shape, column types, descriptive statistics, "
                "detected spend/KPI/channel columns, skewness, correlations, sample rows."
            ),
        ),
        StructuredTool(
            name="get_adstock_recommendations",
            func=get_adstock_recs,
            args_schema=GetAdstockRecsInput,
            description=(
                "Get AI recommendations for which columns suit adstock parameter optimisation. "
                "Includes warnings about data suitability (time series, row count, etc.)."
            ),
        ),

        StructuredTool(
            name="filter_aggregate",
            func=filter_agg,
            args_schema=FilterAggInput,
            description="Filter dataset and aggregate a column, with optional grouping.",
        ),

        StructuredTool(
            name="detect_outliers",
            func=detect_outliers,
            args_schema=OutlierInput,
            description="Detect outliers in a column using IQR or Z-score method.",
        ),
        StructuredTool(
            name="execute_query",
            func=execute_query,
            args_schema=ExecuteQueryInput,
            description=(
                "Execute arbitrary Python/Pandas code in a sandbox. "
                "Use `df` for the dataframe. Store final result in `result` variable. "
                "scipy, sklearn, statsmodels available. "
                "Example: result = df.groupby('product')['revenue'].sum().reset_index()"
            ),
        ),
        StructuredTool(
            name="clean_data",
            func=clean_data,
            args_schema=CleanDataInput,
            description="Clean data: remove duplicates, drop/fill nulls.",
        ),
        StructuredTool(
            name="add_time_features",
            func=add_time_features,
            args_schema=AddTimeFeaturesInput,
            description="Add week, month, quarter, year, dayofweek columns from a datetime column.",
        ),
        StructuredTool(
            name="aggregate_weekly",
            func=aggregate_weekly,
            args_schema=AggregateWeeklyInput,
            description="Aggregate transaction data to weekly time series (required for MMM on granular data).",
        ),
    ]
