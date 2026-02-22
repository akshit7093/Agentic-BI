# =============================================================
# core/mmm_engine.py — Central data + model engine
# =============================================================

import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import (
    SPEND_KEYWORDS, KPI_KEYWORDS, TIME_KEYWORDS, CHANNEL_KEYWORDS,
)
from .executor import SafeCodeExecutor
from .transforms import (
    apply_transforms, optimize_adstock_params, adstock_geometric, hill_transform,
)

logger = logging.getLogger(__name__)

try:
    from scipy.optimize import minimize, differential_evolution
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False

try:
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from pyspark.sql import SparkSession
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False


# =============================================================
class MMMEngine:
    """
    Core engine: load → profile → transform → model → optimise.
    All public methods return typed dicts so tools can serialise easily.
    """

    def __init__(self, spark=None):
        self.spark = spark
        self.data: Optional[pd.DataFrame] = None
        self.data_loaded: bool = False
        self.table_path: Optional[str] = None
        self._profile: Dict[str, Any] = {}
        self.model_results: Dict[str, Any] = {}
        self.budget_results: Dict[str, Any] = {}
        self.adstock_params: Dict[str, Dict] = {}
        self.transformations_applied: List[str] = []
        self.analysis_history: List[Dict] = []
        self.executor = SafeCodeExecutor()

    # ─────────────────────────────────────────────
    # DATA LOADING
    # ─────────────────────────────────────────────

    def load_data(self, path: str) -> Dict[str, Any]:
        """Load from Unity Catalog, CSV, JSON, or a dict."""
        try:
            if isinstance(path, dict):
                self.data = pd.DataFrame(path)
            elif isinstance(path, pd.DataFrame):
                self.data = path.copy()
            elif isinstance(path, str):
                lower = path.lower()
                if lower.endswith(".csv"):
                    if not os.path.exists(path):
                        return {"success": False, "error": f"CSV not found: {path}"}
                    self.data = pd.read_csv(path)
                elif lower.endswith(".json"):
                    self.data = pd.read_json(path)
                elif lower.endswith(".parquet"):
                    self.data = pd.read_parquet(path)
                elif lower.endswith(".xlsx") or lower.endswith(".xls"):
                    self.data = pd.read_excel(path)
                elif self.spark:
                    self.data = self.spark.table(path).toPandas()
                elif os.path.exists(path):
                    # guess
                    self.data = pd.read_csv(path)
                else:
                    return {
                        "success": False,
                        "error": (
                            f"Cannot load '{path}'. Spark unavailable for Unity Catalog "
                            "and file not found locally."
                        ),
                    }
            else:
                return {"success": False, "error": f"Unsupported path type: {type(path)}"}

            self.table_path = str(path)
            self.data_loaded = True
            self._build_profile()
            return {"success": True, **self._profile_summary()}

        except Exception as exc:
            logger.exception("load_data failed")
            hint = ""
            if "TABLE_OR_VIEW_NOT_FOUND" in str(exc):
                hint = f" Verify catalog.schema.table path: '{path}'"
            return {"success": False, "error": str(exc) + hint}

    def reload_data(self) -> Dict[str, Any]:
        """Reload from same path (after external changes)."""
        if not self.table_path:
            return {"success": False, "error": "No table_path stored — call load_data() first."}
        return self.load_data(self.table_path)

    # ─────────────────────────────────────────────
    # DATA PROFILING
    # ─────────────────────────────────────────────

    def _build_profile(self) -> None:
        if self.data is None:
            return
        df = self.data

        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)
        dt_cols  = list(df.select_dtypes(include=["datetime64"]).columns)

        # Try to parse date-like string columns
        for col in cat_cols:
            if any(kw in col.lower() for kw in TIME_KEYWORDS):
                try:
                    self.data[col] = pd.to_datetime(df[col])
                    dt_cols.append(col)
                    cat_cols.remove(col)
                except Exception:
                    pass

        spend_cols   = [c for c in df.columns if any(kw in c.lower() for kw in SPEND_KEYWORDS)]
        kpi_cols     = [c for c in df.columns if any(kw in c.lower() for kw in KPI_KEYWORDS)]
        channel_cols = [c for c in df.columns if any(kw in c.lower() for kw in CHANNEL_KEYWORDS)]

        # Null analysis
        null_pct = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
        # Unique counts
        unique_counts = {c: int(df[c].nunique()) for c in df.columns}
        # Skewness for numerics
        skewness = df[num_cols].skew().round(3).to_dict() if num_cols else {}
        # Correlation with first KPI if any
        correlations: Dict[str, float] = {}
        if kpi_cols and num_cols:
            kpi = kpi_cols[0]
            if kpi in df.columns:
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        corr = df[num_cols].corrwith(df[kpi]).drop(index=kpi, errors="ignore")
                        correlations = corr.round(4).to_dict()
                except Exception:
                    pass

        self._profile = {
            "columns": list(df.columns),
            "dtypes": {k: str(v) for k, v in df.dtypes.items()},
            "rows": len(df),
            "numeric_columns": num_cols,
            "categorical_columns": cat_cols,
            "datetime_columns": dt_cols,
            "potential_spend_columns": spend_cols,
            "potential_kpi_columns": kpi_cols,
            "potential_channel_columns": channel_cols,
            "null_pct": null_pct,
            "unique_counts": unique_counts,
            "skewness": skewness,
            "correlations_with_first_kpi": correlations,
            "has_time_column": len(dt_cols) > 0,
            "time_column": dt_cols[0] if dt_cols else None,
        }

    def _profile_summary(self) -> Dict[str, Any]:
        p = self._profile
        return {
            "rows": p.get("rows"),
            "columns": p.get("columns"),
            "numeric_columns": p.get("numeric_columns"),
            "categorical_columns": p.get("categorical_columns"),
            "datetime_columns": p.get("datetime_columns"),
            "potential_spend_columns": p.get("potential_spend_columns"),
            "potential_kpi_columns": p.get("potential_kpi_columns"),
            "null_pct": p.get("null_pct"),
        }

    # ─────────────────────────────────────────────
    # PUBLIC INSPECTION METHODS
    # ─────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        return {
            "data_loaded": self.data_loaded,
            "table_path": self.table_path,
            "rows": len(self.data) if self.data is not None else 0,
            **{k: self._profile.get(k, []) for k in [
                "columns", "numeric_columns", "categorical_columns",
                "datetime_columns", "potential_spend_columns",
                "potential_kpi_columns", "potential_channel_columns",
            ]},
            "model_fitted": bool(self.model_results.get("success")),
            "analysis_iterations": len(self.analysis_history),
        }

    def inspect_data(self) -> Dict[str, Any]:
        if not self.data_loaded or self.data is None:
            return {"success": False, "error": "No data loaded."}
        return {
            "success": True,
            **self._profile,
            "describe": self.data.describe(include="all").round(4).to_dict(),
            "sample": self.data.head(5).to_dict(orient="records"),
        }

    def get_column_stats(self, column: str) -> Dict[str, Any]:
        if self.data is None:
            return {"success": False, "error": "No data loaded."}
        if column not in self.data.columns:
            return {"success": False, "error": f"Column '{column}' not found. Available: {list(self.data.columns)}"}
        col = self.data[column]
        stats: Dict[str, Any] = {
            "success": True,
            "column": column,
            "dtype": str(col.dtype),
            "count": int(col.count()),
            "null_count": int(col.isnull().sum()),
            "null_pct": round(float(col.isnull().mean() * 100), 2),
            "unique": int(col.nunique()),
        }
        if pd.api.types.is_numeric_dtype(col):
            stats.update({
                "mean": round(float(col.mean()), 4),
                "median": round(float(col.median()), 4),
                "std": round(float(col.std()), 4),
                "min": round(float(col.min()), 4),
                "max": round(float(col.max()), 4),
                "q25": round(float(col.quantile(0.25)), 4),
                "q75": round(float(col.quantile(0.75)), 4),
                "skewness": round(float(col.skew()), 4),
                "kurtosis": round(float(col.kurtosis()), 4),
            })
        else:
            top5 = col.value_counts().head(5).to_dict()
            stats["top_values"] = {str(k): int(v) for k, v in top5.items()}
        return stats

    def get_top_values(
        self,
        column: str,
        n: int = 10,
        ascending: bool = False,
        group_by: Optional[str] = None,
        agg: str = "sum",
    ) -> Dict[str, Any]:
        if self.data is None:
            return {"success": False, "error": "No data loaded."}
        if column not in self.data.columns:
            return {"success": False, "error": f"Column '{column}' not found."}
        try:
            agg_fns = {"sum": "sum", "mean": "mean", "count": "count", "max": "max", "min": "min"}
            agg_fn = agg_fns.get(agg, "sum")
            if group_by:
                if group_by not in self.data.columns:
                    return {"success": False, "error": f"group_by column '{group_by}' not found."}
                result = (
                    self.data.groupby(group_by)[column]
                    .agg(agg_fn)
                    .reset_index()
                    .sort_values(column, ascending=ascending)
                    .head(n)
                )
            else:
                result = (
                    self.data[[column]]
                    .sort_values(column, ascending=ascending)
                    .head(n)
                )
            return {"success": True, "rows": result.to_dict(orient="records"), "columns": list(result.columns)}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def sample_rows(self, n: int = 10) -> Dict[str, Any]:
        if self.data is None:
            return {"success": False, "error": "No data loaded."}
        sample = self.data.sample(min(n, len(self.data)), random_state=42)
        return {"success": True, "rows": sample.to_dict(orient="records"), "columns": list(sample.columns)}

    def filter_and_aggregate(
        self,
        column: str,
        agg: str = "sum",
        filter_col: Optional[str] = None,
        filter_val: Optional[Any] = None,
        group_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.data is None:
            return {"success": False, "error": "No data loaded."}
        try:
            df = self.data.copy()
            if filter_col and filter_val is not None:
                df = df[df[filter_col] == filter_val]
            agg_fns = {"sum": "sum", "mean": "mean", "count": "count", "max": "max", "min": "min", "std": "std"}
            agg_fn = agg_fns.get(agg, "sum")
            if group_by:
                result = df.groupby(group_by)[column].agg(agg_fn).reset_index()
                return {"success": True, "rows": result.to_dict(orient="records"), "columns": list(result.columns)}
            else:
                val = getattr(df[column], agg_fn)()
                return {"success": True, "result": float(val), "agg": agg, "column": column}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def execute_custom_query(self, code: str) -> Dict[str, Any]:
        """Run arbitrary pandas code against self.data."""
        if self.data is None:
            return {"success": False, "error": "No data loaded."}
        result = self.executor.execute(code, df=self.data)
        self.analysis_history.append({"type": "custom_query", "code": code, "success": result["success"]})
        return result

    def get_correlation_matrix(self, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        if self.data is None:
            return {"success": False, "error": "No data loaded."}
        num = self.data.select_dtypes(include=[np.number])
        if columns:
            cols = [c for c in columns if c in num.columns]
            num = num[cols]
        if num.empty:
            return {"success": False, "error": "No numeric columns to correlate."}
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            corr = num.corr().round(4)
            
        return {"success": True, "matrix": corr.to_dict(), "columns": list(corr.columns)}

    def detect_outliers(self, column: str, method: str = "iqr") -> Dict[str, Any]:
        if self.data is None:
            return {"success": False, "error": "No data loaded."}
        if column not in self.data.columns:
            return {"success": False, "error": f"Column '{column}' not found."}
        col = self.data[column].dropna()
        if method == "iqr":
            q1, q3 = col.quantile([0.25, 0.75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (self.data[column] < lo) | (self.data[column] > hi)
        else:  # z-score
            z = (self.data[column] - col.mean()) / (col.std() + 1e-10)
            mask = z.abs() > 3
            lo, hi = float(col.mean() - 3 * col.std()), float(col.mean() + 3 * col.std())
        outlier_rows = self.data[mask]
        return {
            "success": True,
            "method": method,
            "n_outliers": int(mask.sum()),
            "pct_outliers": round(float(mask.mean() * 100), 2),
            "lower_bound": round(float(lo), 4),
            "upper_bound": round(float(hi), 4),
            "sample_outliers": outlier_rows.head(10).to_dict(orient="records"),
        }

    # ─────────────────────────────────────────────
    # ADSTOCK / HILL PARAMETER OPTIMISATION
    # ─────────────────────────────────────────────

    def get_adstock_recommendations(self) -> Dict[str, Any]:
        if not self.data_loaded:
            return {"success": False, "error": "No data loaded."}
        spend_cols = self._profile.get("potential_spend_columns", [])
        dt_cols    = self._profile.get("datetime_columns", [])
        warnings_: List[str] = []
        recs = []
        for col in spend_cols:
            recs.append({"column": col, "confidence": "high", "reason": "Spend keyword in column name"})
        if not dt_cols:
            warnings_.append("⚠️ No datetime column detected — adstock requires time-series ordering.")
        if len(self.data) < 52:
            warnings_.append(f"⚠️ Only {len(self.data)} rows — recommend ≥52 weeks for reliable MMM.")
        if not spend_cols:
            warnings_.append("⚠️ No spend-like columns found — consider aggregating transactions to weekly spend.")
        return {
            "success": True,
            "recommended_columns": recs,
            "warnings": warnings_,
            "has_time_series": bool(dt_cols),
            "total_rows": len(self.data),
        }

    def optimize_adstock_parameters(
        self,
        channel_col: str,
        kpi_col: str,
    ) -> Dict[str, Any]:
        if self.data is None:
            return {"success": False, "error": "No data loaded."}
        for c in [channel_col, kpi_col]:
            if c not in self.data.columns:
                return {"success": False, "error": f"Column '{c}' not found."}
        result = optimize_adstock_params(self.data, channel_col, kpi_col)
        self.adstock_params[channel_col] = result
        return {"success": True, "channel": channel_col, "kpi": kpi_col, **result}

    def optimize_all_adstock_parameters(
        self, channel_cols: List[str], kpi_col: str
    ) -> Dict[str, Any]:
        results = {}
        for col in channel_cols:
            res = self.optimize_adstock_parameters(col, kpi_col)
            results[col] = res
        return {"success": True, "results": results}

    # ─────────────────────────────────────────────
    # QUICK OLS MODEL (sklearn / numpy fallback)
    # ─────────────────────────────────────────────

    def run_ols_mmm(
        self,
        kpi_col: str,
        channel_cols: List[str],
        use_adstock: bool = True,
    ) -> Dict[str, Any]:
        if self.data is None or not self.data_loaded:
            return {"success": False, "error": "No data loaded."}
        missing = [c for c in [kpi_col] + channel_cols if c not in self.data.columns]
        if missing:
            return {"success": False, "error": f"Columns not found: {missing}"}

        try:
            decay_map = {c: self.adstock_params.get(c, {}).get("decay", 0.5) for c in channel_cols}
            half_sat_map = {c: self.adstock_params.get(c, {}).get("half_sat", None) for c in channel_cols}
            slope_map = {c: self.adstock_params.get(c, {}).get("slope", 1.0) for c in channel_cols}
            df = apply_transforms(self.data, channel_cols, decay_map, half_sat_map, slope_map) if use_adstock else self.data.copy()
            t_cols = [f"{c}_transformed" for c in channel_cols] if use_adstock else channel_cols

            y = df[kpi_col].values.astype(float)
            X = df[t_cols].values.astype(float)

            if SKLEARN_AVAILABLE:
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                X_s = scaler_X.fit_transform(X)
                y_s = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
                mdl = Ridge(alpha=1.0)
                mdl.fit(X_s, y_s)
                y_hat = mdl.predict(X_s)
                r2 = float(r2_score(y_s, y_hat))
                coefs = mdl.coef_
                # Unscale coefficients to get ROI
                roi = coefs * scaler_y.scale_[0] / (scaler_X.scale_ + 1e-10)
            else:
                # numpy fallback
                X_aug = np.column_stack([np.ones(len(y)), X])
                coefs, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
                coefs = coefs[1:]
                y_hat = X_aug @ np.concatenate([[0], coefs])
                ss_res = np.sum((y - y_hat) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = float(1 - ss_res / (ss_tot + 1e-10))
                roi = coefs

            result = {
                "success": True,
                "model_type": "OLS/Ridge",
                "channels": channel_cols,
                "coefficients": roi.tolist(),
                "r2": round(r2, 4),
                "n_obs": len(y),
                "roi_table": [
                    {"channel": c, "roi": round(float(r), 6)}
                    for c, r in zip(channel_cols, roi)
                ],
            }
            self.model_results = result
            return result
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ─────────────────────────────────────────────
    # BAYESIAN MMM (PyMC)
    # ─────────────────────────────────────────────

    def run_bayesian_mmm(
        self,
        kpi_col: str,
        channel_cols: List[str],
        n_samples: int = 500,
        tune: int = 200,
        use_adstock: bool = True,
    ) -> Dict[str, Any]:
        if self.data is None or not self.data_loaded:
            return {"success": False, "error": "No data loaded."}
        if not PYMC_AVAILABLE:
            logger.warning("PyMC not available — falling back to OLS.")
            return self.run_ols_mmm(kpi_col, channel_cols, use_adstock)
        missing = [c for c in [kpi_col] + channel_cols if c not in self.data.columns]
        if missing:
            return {"success": False, "error": f"Columns not found: {missing}"}

        try:
            decay_map = {c: self.adstock_params.get(c, {}).get("decay", 0.5) for c in channel_cols}
            half_sat_map = {c: self.adstock_params.get(c, {}).get("half_sat", None) for c in channel_cols}
            slope_map = {c: self.adstock_params.get(c, {}).get("slope", 1.0) for c in channel_cols}
            df = apply_transforms(self.data, channel_cols, decay_map, half_sat_map, slope_map) if use_adstock else self.data.copy()
            t_cols = [f"{c}_transformed" for c in channel_cols] if use_adstock else channel_cols

            y = df[kpi_col].values.astype(float)
            X = df[t_cols].values.astype(float)
            y_mean, y_std = y.mean(), y.std() + 1e-8
            X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
            y_n = (y - y_mean) / y_std
            X_n = (X - X_mean) / X_std

            with pm.Model():
                intercept = pm.Normal("intercept", mu=0, sigma=1)
                betas = pm.HalfNormal("betas", sigma=1, shape=len(channel_cols))
                sigma = pm.HalfNormal("sigma", sigma=1)
                mu = intercept + pm.math.dot(X_n, betas)
                pm.Normal("obs", mu=mu, sigma=sigma, observed=y_n)
                trace = pm.sample(n_samples, tune=tune, chains=2, progressbar=False, return_inferencedata=True)

            beta_means = trace.posterior["betas"].mean(dim=["chain", "draw"]).values
            beta_sds   = trace.posterior["betas"].std(dim=["chain", "draw"]).values
            y_hat = trace.posterior["intercept"].mean().item() + X_n @ beta_means
            ss_res = np.sum((y_n - y_hat) ** 2)
            ss_tot = np.sum((y_n - y_n.mean()) ** 2)
            r2 = float(1 - ss_res / (ss_tot + 1e-10))
            roi = (beta_means * y_std) / X_std

            result = {
                "success": True,
                "model_type": "Bayesian (PyMC)",
                "channels": channel_cols,
                "betas_mean": beta_means.tolist(),
                "betas_std": beta_sds.tolist(),
                "roi_per_unit": roi.tolist(),
                "r2": round(r2, 4),
                "n_obs": len(y),
                "roi_table": sorted(
                    [{"channel": c, "roi": round(float(r), 6), "beta_std": round(float(sd), 6)}
                     for c, r, sd in zip(channel_cols, roi, beta_sds)],
                    key=lambda x: x["roi"], reverse=True,
                ),
            }
            self.model_results = result
            return result
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ─────────────────────────────────────────────
    # BUDGET OPTIMISATION
    # ─────────────────────────────────────────────

    def optimize_budget(
        self,
        total_budget: float,
        channel_cols: List[str],
        min_pct: float = 0.05,
        max_pct: float = 0.60,
    ) -> Dict[str, Any]:
        if not self.model_results.get("success"):
            return {"success": False, "error": "Fit a model first (run_bayesian_mmm or run_ols_mmm)."}
        if not SCIPY_AVAILABLE:
            return {"success": False, "error": "scipy required for budget optimisation."}

        channels_in_model = self.model_results["channels"]
        idx = [channels_in_model.index(c) for c in channel_cols if c in channels_in_model]
        if not idx:
            return {"success": False, "error": f"None of {channel_cols} are in model channels: {channels_in_model}"}

        rois = np.array(self.model_results.get("roi_per_unit", self.model_results.get("coefficients", [])))
        rois_sub = rois[idx]

        lb = total_budget * min_pct
        ub = total_budget * max_pct
        bounds = [(lb, ub)] * len(channel_cols)
        constraint = {"type": "eq", "fun": lambda x: x.sum() - total_budget}

        def neg_rev(x):
            return -float(np.dot(rois_sub, np.sqrt(np.clip(x, 0, None) + 1e-8)))

        try:
            de = differential_evolution(neg_rev, bounds, maxiter=500, seed=42, constraints=[constraint], tol=1e-8)
            res = minimize(neg_rev, de.x, method="SLSQP", bounds=bounds, constraints=[constraint])
            alloc = np.clip(res.x, lb, ub)
            alloc *= total_budget / alloc.sum()

            equal_alloc = np.full(len(channel_cols), total_budget / len(channel_cols))
            uplift_pct = (-neg_rev(alloc) - (-neg_rev(equal_alloc))) / (-neg_rev(equal_alloc) + 1e-10) * 100

            result = {
                "success": True,
                "total_budget": total_budget,
                "optimal_allocation": {c: round(float(a), 2) for c, a in zip(channel_cols, alloc)},
                "allocation_pct": {c: round(float(a / total_budget * 100), 2) for c, a in zip(channel_cols, alloc)},
                "expected_revenue_lift_pct": round(float(uplift_pct), 2),
            }
            self.budget_results = result
            return result
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def roi_summary(self) -> Dict[str, Any]:
        if not self.model_results.get("success"):
            return {"success": False, "error": "No model fitted yet."}
        return {
            "success": True,
            "model_type": self.model_results.get("model_type"),
            "r2": self.model_results.get("r2"),
            "roi_table": self.model_results.get("roi_table", []),
        }

    # ─────────────────────────────────────────────
    # DATA TRANSFORMATION HELPERS
    # ─────────────────────────────────────────────

    def clean_data(
        self,
        drop_nulls: bool = False,
        fill_value: Optional[float] = None,
        drop_duplicates: bool = True,
    ) -> Dict[str, Any]:
        if self.data is None:
            return {"success": False, "error": "No data loaded."}
        before = len(self.data)
        if drop_duplicates:
            self.data = self.data.drop_duplicates()
        if drop_nulls:
            self.data = self.data.dropna()
        elif fill_value is not None:
            self.data = self.data.fillna(fill_value)
        after = len(self.data)
        self._build_profile()
        return {"success": True, "rows_before": before, "rows_after": after, "rows_removed": before - after}

    def add_time_features(self, date_col: str) -> Dict[str, Any]:
        if self.data is None:
            return {"success": False, "error": "No data loaded."}
        if date_col not in self.data.columns:
            return {"success": False, "error": f"Column '{date_col}' not found."}
        try:
            self.data[date_col] = pd.to_datetime(self.data[date_col])
            self.data["_week"]      = self.data[date_col].dt.isocalendar().week.astype(int)
            self.data["_month"]     = self.data[date_col].dt.month
            self.data["_quarter"]   = self.data[date_col].dt.quarter
            self.data["_year"]      = self.data[date_col].dt.year
            self.data["_dayofweek"] = self.data[date_col].dt.dayofweek
            self._build_profile()
            return {"success": True, "added_columns": ["_week", "_month", "_quarter", "_year", "_dayofweek"]}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def aggregate_to_weekly(
        self,
        date_col: str,
        value_cols: List[str],
        agg: str = "sum",
    ) -> Dict[str, Any]:
        if self.data is None:
            return {"success": False, "error": "No data loaded."}
        try:
            df = self.data.copy()
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
            agg_fns = {"sum": "sum", "mean": "mean", "max": "max", "min": "min"}
            agg_fn = agg_fns.get(agg, "sum")
            weekly = df[value_cols].resample("W").agg(agg_fn).reset_index()
            self.data = weekly
            self._build_profile()
            return {"success": True, "rows": len(weekly), "columns": list(weekly.columns), "sample": weekly.head(3).to_dict(orient="records")}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def get_data_context(self) -> str:
        """Return formatted markdown string for use in system prompts."""
        if not self.data_loaded or self.data is None:
            return "No data loaded yet."
        p = self._profile
        lines = [
            f"- **Table**: `{self.table_path}`",
            f"- **Shape**: {p['rows']} rows × {len(p['columns'])} columns",
            f"- **Spend/Channel cols**: {p.get('potential_spend_columns') or 'None detected'}",
            f"- **KPI cols**: {p.get('potential_kpi_columns') or 'None detected'}",
            f"- **Time col**: {p.get('time_column') or 'None detected'}",
            "",
            "**All columns:**",
        ]
        for col in p["columns"]:
            dtype = p["dtypes"].get(col, "?")
            lines.append(f"  - `{col}` ({dtype})")
        return "\n".join(lines)
