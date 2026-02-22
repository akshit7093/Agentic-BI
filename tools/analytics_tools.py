# =============================================================
# tools/analytics_tools.py — Advanced ML, statistical, and
# feature-engineering tools for deep autonomous analysis
# =============================================================

import json
import logging
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from ..core.mmm_engine import MMMEngine

logger = logging.getLogger(__name__)


# ─── Input schemas ────────────────────────────────────────────

class FeatureImportanceRFInput(BaseModel):
    target_col: str = Field(description="Target / KPI column to predict")
    feature_cols: Optional[str] = Field(
        default=None,
        description="Comma-separated feature columns (leave empty for all numeric)",
    )
    n_estimators: int = Field(default=100, description="Number of trees")
    cv_folds: int = Field(default=5, description="Cross-validation folds")


class FeatureImportanceGBInput(BaseModel):
    target_col: str = Field(description="Target column")
    feature_cols: Optional[str] = Field(default=None, description="Comma-separated features")
class MutualInfoInput(BaseModel):
    target_col: str = Field(description="Target column")
    feature_cols: Optional[str] = Field(default=None, description="Comma-separated features")

class CrossValidateInput(BaseModel):
    target_col: str = Field(description="Target column")
    feature_cols: str = Field(description="Comma-separated feature columns")
    models: str = Field(
        default="ols,ridge,lasso,rf,gb",
        description="Comma-separated models: ols, ridge, lasso, rf, gb",
    )
    cv_folds: int = Field(default=5, description="Number of CV folds")

class StepwiseInput(BaseModel):
    target_col: str = Field(description="Target / KPI column")
    feature_cols: Optional[str] = Field(default=None, description="Comma-separated candidate features")
    direction: str = Field(default="forward", description="forward | backward | both")
    metric: str = Field(default="r2", description="Scoring metric: r2 | rmse | aic")


class AutoFeatureInput(BaseModel):
    columns: Optional[str] = Field(default=None, description="Comma-separated columns to engineer")
    date_col: Optional[str] = Field(default=None, description="Date column for lag/rolling features")
    lags: str = Field(default="1,2,3", description="Comma-separated lag periods")
    rolling_windows: str = Field(default="3,7", description="Comma-separated rolling window sizes")
    interactions: bool = Field(default=True, description="Generate interaction terms")


# ─────────────────────────────────────────────
# BUILDER
# ─────────────────────────────────────────────


def build_analytics_tools(engine: MMMEngine) -> list:
    """Return all advanced analytics tools wired to *engine*."""

    def _j(obj):
        return json.dumps(obj, default=str)

    def _get_df():
        df = engine.df
        if df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        return df

    def _numeric_cols(df, cols_str=None):
        if cols_str:
            return [c.strip() for c in cols_str.split(",") if c.strip() in df.columns]
        return [c for c in df.select_dtypes(include=[np.number]).columns]

    # ─── 1. Random Forest Feature Importance ──────────

    def feature_importance_rf(
        target_col: str,
        feature_cols: Optional[str] = None,
        n_estimators: int = 100,
        cv_folds: int = 5,
    ) -> str:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        df = _get_df().dropna()
        features = _numeric_cols(df, feature_cols)
        features = [c for c in features if c != target_col]

        if not features:
            return _j({"error": "No numeric feature columns found"})

        X = df[features].values
        y = df[target_col].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        cv_scores = cross_val_score(rf, X_scaled, y, cv=min(cv_folds, len(df)), scoring="r2")
        rf.fit(X_scaled, y)

        importances = dict(zip(features, [round(float(v), 4) for v in rf.feature_importances_]))
        ranked = sorted(importances.items(), key=lambda x: x[1], reverse=True)

        return _j({
            "method": "RandomForest",
            "n_estimators": n_estimators,
            "cv_r2_mean": round(float(cv_scores.mean()), 4),
            "cv_r2_std": round(float(cv_scores.std()), 4),
            "feature_importances": dict(ranked),
            "top_5": [r[0] for r in ranked[:5]],
            "recommendation": f"Top features: {', '.join(r[0] for r in ranked[:5])}",
        })

    # ─── 2. Gradient Boosting Feature Importance ──────

    def feature_importance_gb(
        target_col: str,
        feature_cols: Optional[str] = None,
        n_estimators: int = 100,
    ) -> str:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        df = _get_df().dropna()
        features = _numeric_cols(df, feature_cols)
        features = [c for c in features if c != target_col]

        X = df[features].values
        y = df[target_col].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        gb = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
        cv_scores = cross_val_score(gb, X_scaled, y, cv=min(5, len(df)), scoring="r2")
        gb.fit(X_scaled, y)

        importances = dict(zip(features, [round(float(v), 4) for v in gb.feature_importances_]))
        ranked = sorted(importances.items(), key=lambda x: x[1], reverse=True)

        return _j({
            "method": "GradientBoosting",
            "n_estimators": n_estimators,
            "cv_r2_mean": round(float(cv_scores.mean()), 4),
            "cv_r2_std": round(float(cv_scores.std()), 4),
            "feature_importances": dict(ranked),
            "top_5": [r[0] for r in ranked[:5]],
        })

    # ─── 3. Mutual Information ────────────────────────

    def mutual_information(
        target_col: str,
        feature_cols: Optional[str] = None,
    ) -> str:
        from sklearn.feature_selection import mutual_info_regression

        df = _get_df().dropna()
        features = _numeric_cols(df, feature_cols)
        features = [c for c in features if c != target_col]

        X = df[features].values
        y = df[target_col].values

        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_dict = dict(zip(features, [round(float(v), 4) for v in mi_scores]))
        ranked = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)



    # ─── 10. Compare Models (fit + rank) ─────────────

    def compare_models(
        target_col: str,
        feature_cols: str,
        models: str = "ols,ridge,lasso,rf,gb",
        cv_folds: int = 5,
    ) -> str:
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        df = _get_df().dropna()
        features = [c.strip() for c in feature_cols.split(",") if c.strip() in df.columns]
        if not features or target_col not in df.columns:
            return _j({"error": "Invalid columns"})

        X = df[features].values
        y = df[target_col].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model_map = {
            "ols": ("OLS Linear Regression", LinearRegression()),
            "ridge": ("Ridge Regression", Ridge(alpha=1.0)),
            "lasso": ("Lasso Regression", Lasso(alpha=0.1, max_iter=5000)),
            "rf": ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
            "gb": ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
        }

        requested_models = [m.strip().lower() for m in models.split(",")]
        folds = min(cv_folds, len(df))
        results = {}

        for key in requested_models:
            if key not in model_map:
                continue
            name, estimator = model_map[key]
            try:
                r2_scores = cross_val_score(estimator, X_scaled, y, cv=folds, scoring="r2")
                neg_rmse = cross_val_score(estimator, X_scaled, y, cv=folds, scoring="neg_root_mean_squared_error")
                results[name] = {
                    "r2_mean": round(float(r2_scores.mean()), 4),
                    "r2_std": round(float(r2_scores.std()), 4),
                    "rmse_mean": round(float(-neg_rmse.mean()), 4),
                    "rmse_std": round(float(neg_rmse.std()), 4),
                }
            except Exception as exc:
                results[name] = {"error": str(exc)}

        # Rank by R²
        valid = {k: v for k, v in results.items() if "r2_mean" in v}
        ranking = sorted(valid.items(), key=lambda x: x[1]["r2_mean"], reverse=True)

        return _j({
            "method": "CompareModels",
            "features": features,
            "target": target_col,
            "cv_folds": folds,
            "model_results": results,
            "ranking": [r[0] for r in ranking],
            "best_model": ranking[0][0] if ranking else "N/A",
            "best_r2": ranking[0][1]["r2_mean"] if ranking else None,
        })

    # ─── 11. Stepwise Feature Selection ──────────────

    def stepwise_selection(
        target_col: str,
        feature_cols: Optional[str] = None,
        direction: str = "forward",
        metric: str = "r2",
    ) -> str:
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score

        df = _get_df().dropna()
        candidates = _numeric_cols(df, feature_cols)
        candidates = [c for c in candidates if c != target_col]

        if not candidates:
            return _j({"error": "No candidate features"})

        y = df[target_col].values
        folds = min(5, len(df))

        if direction == "forward":
            selected = []
            remaining = list(candidates)
            history = []

            for step in range(len(candidates)):
                best_score = -np.inf
                best_feature = None

                for feat in remaining:
                    test_features = selected + [feat]
                    X = df[test_features].values
                    lr = LinearRegression()
                    scores = cross_val_score(lr, X, y, cv=folds, scoring="r2")
                    score = scores.mean()
                    if score > best_score:
                        best_score = score
                        best_feature = feat

                if best_feature is None or (selected and best_score <= history[-1]["r2"]):
                    break

                selected.append(best_feature)
                remaining.remove(best_feature)
                history.append({
                    "step": step + 1,
                    "added": best_feature,
                    "r2": round(float(best_score), 4),
                    "features": list(selected),
                })

        elif direction == "backward":
            selected = list(candidates)
            history = []

            X_all = df[selected].values
            lr = LinearRegression()
            baseline = cross_val_score(lr, X_all, y, cv=folds, scoring="r2").mean()
            history.append({"step": 0, "removed": None, "r2": round(float(baseline), 4), "features": list(selected)})

            for step in range(len(candidates) - 1):
                best_score = -np.inf
                worst_feature = None

                for feat in selected:
                    test_features = [f for f in selected if f != feat]
                    if not test_features:
                        continue
                    X = df[test_features].values
                    scores = cross_val_score(LinearRegression(), X, y, cv=folds, scoring="r2")
                    score = scores.mean()
                    if score > best_score:
                        best_score = score
                        worst_feature = feat

                if worst_feature is None or best_score <= history[-1]["r2"] - 0.01:
                    break

                selected.remove(worst_feature)
                history.append({
                    "step": step + 1,
                    "removed": worst_feature,
                    "r2": round(float(best_score), 4),
                    "features": list(selected),
                })
        else:
            return _j({"error": "direction must be 'forward' or 'backward'"})

        return _j({
            "method": f"StepwiseSelection_{direction}",
            "selected_features": selected,
            "final_r2": round(float(history[-1]["r2"]), 4) if history else None,
            "steps": history,
            "recommendation": f"Best feature subset: {', '.join(selected)}",
        })

    # ─── 12. Auto Feature Engineering ────────────────

    def auto_feature_engineering(
        columns: Optional[str] = None,
        date_col: Optional[str] = None,
        lags: str = "1,2,3",
        rolling_windows: str = "3,7",
        interactions: bool = True,
    ) -> str:
        df = _get_df().copy()
        cols = _numeric_cols(df, columns)
        new_features = []

        # Lag features (only if data has a time ordering)
        if date_col and date_col in df.columns:
            df = df.sort_values(date_col).reset_index(drop=True)
            lag_list = [int(l.strip()) for l in lags.split(",") if l.strip()]
            for col in cols[:5]:  # Limit to top 5 columns
                for lag in lag_list:
                    fname = f"{col}_lag{lag}"
                    df[fname] = df[col].shift(lag)
                    new_features.append(fname)

            # Rolling features
            window_list = [int(w.strip()) for w in rolling_windows.split(",") if w.strip()]
            for col in cols[:5]:
                for w in window_list:
                    fname_mean = f"{col}_rolling_mean_{w}"
                    fname_std = f"{col}_rolling_std_{w}"
                    df[fname_mean] = df[col].rolling(w).mean()
                    df[fname_std] = df[col].rolling(w).std()
                    new_features.extend([fname_mean, fname_std])

        # Interaction features
        if interactions and len(cols) >= 2:
            for i in range(min(len(cols), 4)):
                for j in range(i + 1, min(len(cols), 4)):
                    fname = f"{cols[i]}_x_{cols[j]}"
                    df[fname] = df[cols[i]] * df[cols[j]]
                    new_features.append(fname)

        # Update engine's dataframe
        engine.df = df.copy()

        return _j({
            "method": "AutoFeatureEngineering",
            "new_features_created": len(new_features),
            "features": new_features,
            "total_columns_now": len(df.columns),
            "note": "DataFrame updated in-place. New features available for modelling.",
        })

    # ─────────────────────────────────────────────
    # ASSEMBLE TOOLS
    # ─────────────────────────────────────────────

    tools = [
        StructuredTool.from_function(
            func=feature_importance_rf,
            name="feature_importance_rf",
            description=(
                "Compute Random Forest feature importance with cross-validation. "
                "Returns ranked features, importances, and CV R² score."
            ),
            args_schema=FeatureImportanceRFInput,
        ),
        StructuredTool.from_function(
            func=feature_importance_gb,
            name="feature_importance_gb",
            description=(
                "Compute Gradient Boosting feature importance with cross-validation. "
                "Returns ranked features and importances."
            ),
            args_schema=FeatureImportanceGBInput,
        ),
        StructuredTool.from_function(
            func=mutual_information,
            name="mutual_information",
            description=(
                "Compute mutual information scores for non-linear dependency detection. "
                "Reveals relationships that correlation misses."
            ),
            args_schema=MutualInfoInput,
        ),

        StructuredTool.from_function(
            func=compare_models,
            name="compare_models",
            description=(
                "Compare multiple regression models head-to-head with cross-validation. "
                "Returns a ranking of models by R² — use this to pick the best approach."
            ),
            args_schema=CrossValidateInput,
        ),
        StructuredTool.from_function(
            func=stepwise_selection,
            name="stepwise_selection",
            description=(
                "Forward or backward stepwise feature selection. Iteratively adds/removes features "
                "to find the optimal subset that maximises R²."
            ),
            args_schema=StepwiseInput,
        ),
        StructuredTool.from_function(
            func=auto_feature_engineering,
            name="auto_feature_engineering",
            description=(
                "Automatically generate lag features, rolling averages, rolling std, and interaction terms. "
                "Enriches the dataset for better model performance."
            ),
            args_schema=AutoFeatureInput,
        ),
    ]

    logger.info(f"Analytics tools: {len(tools)} tools built")
    return tools
