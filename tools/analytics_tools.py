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
    n_estimators: int = Field(default=100, description="Number of boosting rounds")


class MutualInfoInput(BaseModel):
    target_col: str = Field(description="Target column")
    feature_cols: Optional[str] = Field(default=None, description="Comma-separated features")


class PCAInput(BaseModel):
    columns: Optional[str] = Field(default=None, description="Comma-separated numeric columns (empty=all)")
    n_components: int = Field(default=5, description="Number of principal components")


class VIFInput(BaseModel):
    columns: Optional[str] = Field(default=None, description="Comma-separated numeric columns (empty=all)")


class GrangerInput(BaseModel):
    cause_col: str = Field(description="Potential cause column")
    effect_col: str = Field(description="Potential effect column")
    max_lag: int = Field(default=4, description="Maximum lag to test")


class StationarityInput(BaseModel):
    column: str = Field(description="Column to test for stationarity")


class DecomposeInput(BaseModel):
    column: str = Field(description="Column to decompose")
    date_col: str = Field(description="Date column for time index")
    period: int = Field(default=7, description="Seasonal period (e.g. 7=weekly, 52=yearly)")


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

        return _j({
            "method": "MutualInformation",
            "scores": dict(ranked),
            "top_5": [r[0] for r in ranked[:5]],
            "interpretation": "Higher MI = stronger (possibly non-linear) dependency with target",
        })

    # ─── 4. PCA Analysis ─────────────────────────────

    def pca_analysis(
        columns: Optional[str] = None,
        n_components: int = 5,
    ) -> str:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        df = _get_df().dropna()
        cols = _numeric_cols(df, columns)

        X = df[cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_comp = min(n_components, len(cols), len(df))
        pca = PCA(n_components=n_comp)
        pca.fit(X_scaled)

        components_detail = {}
        for i in range(n_comp):
            loadings = dict(zip(cols, [round(float(v), 4) for v in pca.components_[i]]))
            top_contributors = sorted(loadings.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            components_detail[f"PC{i+1}"] = {
                "explained_variance_pct": round(float(pca.explained_variance_ratio_[i] * 100), 2),
                "top_contributors": dict(top_contributors),
            }

        return _j({
            "method": "PCA",
            "n_components": n_comp,
            "total_explained_variance_pct": round(float(sum(pca.explained_variance_ratio_) * 100), 2),
            "components": components_detail,
            "recommendation": (
                f"First {n_comp} components explain "
                f"{round(float(sum(pca.explained_variance_ratio_) * 100), 1)}% of variance"
            ),
        })

    # ─── 5. VIF (Variance Inflation Factor) ──────────

    def vif_analysis(columns: Optional[str] = None) -> str:
        from sklearn.linear_model import LinearRegression

        df = _get_df().dropna()
        cols = _numeric_cols(df, columns)

        if len(cols) < 2:
            return _j({"error": "Need at least 2 numeric columns for VIF"})

        vif_data = {}
        X = df[cols].values
        for i, col in enumerate(cols):
            y_i = X[:, i]
            X_others = np.delete(X, i, axis=1)
            if X_others.shape[1] == 0:
                vif_data[col] = 1.0
                continue
            lr = LinearRegression().fit(X_others, y_i)
            r2 = lr.score(X_others, y_i)
            vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
            vif_data[col] = round(float(vif), 2)

        ranked = sorted(vif_data.items(), key=lambda x: x[1], reverse=True)
        high_vif = [r for r in ranked if r[1] > 5]

        return _j({
            "method": "VIF",
            "vif_scores": dict(ranked),
            "high_multicollinearity": [r[0] for r in high_vif],
            "interpretation": "VIF > 5 = moderate multicollinearity, VIF > 10 = severe",
            "recommendation": (
                f"Consider removing: {', '.join(r[0] for r in high_vif)}" if high_vif
                else "No severe multicollinearity detected"
            ),
        })

    # ─── 6. Granger Causality ────────────────────────

    def granger_causality(cause_col: str, effect_col: str, max_lag: int = 4) -> str:
        from sklearn.linear_model import LinearRegression

        df = _get_df().dropna()
        if cause_col not in df.columns or effect_col not in df.columns:
            return _j({"error": f"Column(s) not found"})

        cause = df[cause_col].values
        effect = df[effect_col].values

        results = {}
        for lag in range(1, max_lag + 1):
            if lag >= len(effect):
                break
            y = effect[lag:]
            # Restricted model: only own lags
            X_restricted = np.column_stack([effect[lag - i - 1: len(effect) - i - 1] for i in range(lag)])
            # Unrestricted model: own lags + cause lags
            X_unrestricted = np.column_stack([
                X_restricted,
                *[cause[lag - i - 1: len(cause) - i - 1] for i in range(lag)],
            ])

            lr_r = LinearRegression().fit(X_restricted, y)
            lr_u = LinearRegression().fit(X_unrestricted, y)

            rss_r = np.sum((y - lr_r.predict(X_restricted)) ** 2)
            rss_u = np.sum((y - lr_u.predict(X_unrestricted)) ** 2)

            n = len(y)
            f_stat = ((rss_r - rss_u) / lag) / (rss_u / max(n - 2 * lag - 1, 1))
            results[f"lag_{lag}"] = {
                "f_statistic": round(float(f_stat), 4),
                "improvement_pct": round(float((rss_r - rss_u) / rss_r * 100), 2),
            }

        best_lag = max(results.items(), key=lambda x: x[1]["f_statistic"]) if results else None

        return _j({
            "method": "GrangerCausality",
            "cause": cause_col,
            "effect": effect_col,
            "lag_results": results,
            "best_lag": best_lag[0] if best_lag else None,
            "interpretation": (
                f"{cause_col} Granger-causes {effect_col} at {best_lag[0]} "
                f"(F={best_lag[1]['f_statistic']})" if best_lag and best_lag[1]["f_statistic"] > 3.0
                else f"Weak evidence that {cause_col} Granger-causes {effect_col}"
            ),
        })

    # ─── 7. Stationarity Test ────────────────────────

    def stationarity_test(column: str) -> str:
        df = _get_df()
        if column not in df.columns:
            return _j({"error": f"Column '{column}' not found"})

        series = df[column].dropna().values

        # Simple ADF-like test using rolling statistics
        n = len(series)
        if n < 10:
            return _j({"error": "Need at least 10 data points"})

        # Calculate rolling mean/std stability
        half = n // 2
        mean_first = float(np.mean(series[:half]))
        mean_second = float(np.mean(series[half:]))
        std_first = float(np.std(series[:half]))
        std_second = float(np.std(series[half:]))

        mean_change = abs(mean_second - mean_first) / max(abs(mean_first), 1e-10)
        std_change = abs(std_second - std_first) / max(std_first, 1e-10)

        # Autocorrelation at lag 1
        if n > 1:
            autocorr = float(np.corrcoef(series[:-1], series[1:])[0, 1])
        else:
            autocorr = 0.0

        is_stationary = mean_change < 0.3 and std_change < 0.5 and abs(autocorr) < 0.9

        return _j({
            "method": "StationarityTest",
            "column": column,
            "n_observations": n,
            "mean_first_half": round(mean_first, 4),
            "mean_second_half": round(mean_second, 4),
            "mean_change_pct": round(mean_change * 100, 2),
            "std_change_pct": round(std_change * 100, 2),
            "autocorrelation_lag1": round(autocorr, 4),
            "is_stationary": is_stationary,
            "recommendation": (
                f"Series appears {'stationary' if is_stationary else 'non-stationary'}. "
                + ("" if is_stationary else "Consider differencing or detrending before modelling.")
            ),
        })

    # ─── 8. Time Series Decomposition ────────────────

    def time_series_decompose(column: str, date_col: str, period: int = 7) -> str:
        df = _get_df().copy()
        if column not in df.columns or date_col not in df.columns:
            return _j({"error": f"Column not found"})

        df = df.sort_values(date_col).reset_index(drop=True)
        series = df[column].dropna().values

        n = len(series)
        if n < 2 * period:
            return _j({"error": f"Need at least {2 * period} data points for period={period}"})

        # Moving average trend
        trend = pd.Series(series).rolling(window=period, center=True).mean().values
        # Detrended
        detrended = series - np.nan_to_num(trend, nan=np.nanmean(trend))
        # Seasonal component (average by position in period)
        seasonal = np.zeros(n)
        for i in range(period):
            mask = np.arange(i, n, period)
            seasonal[mask] = np.mean(detrended[mask])
        # Residual
        residual = series - np.nan_to_num(trend, nan=np.nanmean(trend)) - seasonal

        return _j({
            "method": "TimeSeriesDecomposition",
            "column": column,
            "period": period,
            "trend_summary": {
                "start": round(float(np.nanmean(trend[:period])), 2),
                "end": round(float(np.nanmean(trend[-period:])), 2),
                "direction": "increasing" if np.nanmean(trend[-period:]) > np.nanmean(trend[:period]) else "decreasing",
            },
            "seasonal_strength": round(float(np.std(seasonal) / max(np.std(series), 1e-10)), 4),
            "residual_std": round(float(np.std(residual)), 4),
            "signal_to_noise": round(float(np.std(series) / max(np.std(residual), 1e-10)), 4),
        })

    # ─── 9. Cross-Validate Multiple Models ──────────

    def cross_validate_model(
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
            "method": "CrossValidation",
            "features": features,
            "target": target_col,
            "cv_folds": folds,
            "model_results": results,
            "ranking": [r[0] for r in ranking],
            "best_model": ranking[0][0] if ranking else "N/A",
            "best_r2": ranking[0][1]["r2_mean"] if ranking else None,
        })

    # ─── 10. Compare Models (fit + rank) ─────────────

    def compare_models(
        target_col: str,
        feature_cols: str,
        models: str = "ols,ridge,lasso,rf,gb",
        cv_folds: int = 5,
    ) -> str:
        # Delegate to cross_validate_model with a different wrapper
        return cross_validate_model(target_col, feature_cols, models, cv_folds)

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
            func=pca_analysis,
            name="pca_analysis",
            description=(
                "Principal Component Analysis — reduce dimensionality, identify dominant patterns, "
                "and find which columns contribute most to data variance."
            ),
            args_schema=PCAInput,
        ),
        StructuredTool.from_function(
            func=vif_analysis,
            name="vif_analysis",
            description=(
                "Variance Inflation Factor — detect multicollinearity between features. "
                "VIF > 5 = moderate, > 10 = severe multicollinearity."
            ),
            args_schema=VIFInput,
        ),
        StructuredTool.from_function(
            func=granger_causality,
            name="granger_causality",
            description=(
                "Granger causality test — does one column's past values help predict another? "
                "Useful for MMM to test if spend leads to revenue."
            ),
            args_schema=GrangerInput,
        ),
        StructuredTool.from_function(
            func=stationarity_test,
            name="stationarity_test",
            description=(
                "Test time-series stationarity via rolling statistics and autocorrelation. "
                "Non-stationary series need differencing before modelling."
            ),
            args_schema=StationarityInput,
        ),
        StructuredTool.from_function(
            func=time_series_decompose,
            name="time_series_decompose",
            description=(
                "Decompose a time series into trend, seasonal, and residual components. "
                "Shows signal-to-noise ratio and trend direction."
            ),
            args_schema=DecomposeInput,
        ),
        StructuredTool.from_function(
            func=cross_validate_model,
            name="cross_validate_model",
            description=(
                "Cross-validate multiple models (OLS, Ridge, Lasso, Random Forest, Gradient Boosting) "
                "on the same data. Returns R² and RMSE for each, ranked by performance."
            ),
            args_schema=CrossValidateInput,
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
