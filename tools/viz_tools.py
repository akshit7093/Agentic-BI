# =============================================================
# tools/viz_tools.py — Visualization tools for charts & graphs
# =============================================================

import json
import logging
import os
import tempfile
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from ..core.mmm_engine import MMMEngine

logger = logging.getLogger(__name__)


# ─── Input schemas ────────────────────────────────────────────


class FeatureImportanceChartInput(BaseModel):
    target_col: str = Field(description="Target column to analyze")
    feature_cols: Optional[str] = Field(default=None, description="Comma-separated features")
    title: str = Field(default="Feature Importance Comparison", description="Chart title")


class TimeSeriesPlotInput(BaseModel):
    columns: str = Field(description="Comma-separated columns to plot")
    date_col: str = Field(description="Date column for x-axis")
    title: str = Field(default="Time Series", description="Chart title")



class ModelComparisonChartInput(BaseModel):
    target_col: str = Field(description="Target column")
    feature_cols: str = Field(description="Comma-separated feature columns")
    title: str = Field(default="Model Performance Comparison", description="Chart title")


# ─────────────────────────────────────────────
# BUILDER
# ─────────────────────────────────────────────


def build_viz_tools(engine: MMMEngine) -> list:
    """Return all visualization tools wired to *engine*."""

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

    def _save_fig(fig, name):
        """Save figure and display in Databricks notebook."""
        try:
            import matplotlib
            matplotlib.use("Agg")
        except Exception:
            pass

        # Save to temp file
        path = os.path.join(tempfile.gettempdir(), f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")

        # Display in notebook
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            pass

        return path


    # ─── 2. Feature Importance Chart ─────────────────

    def plot_feature_importance(
        target_col: str,
        feature_cols: Optional[str] = None,
        title: str = "Feature Importance Comparison",
    ) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import mutual_info_regression

        df = _get_df().dropna()
        features = _numeric_cols(df, feature_cols)
        features = [c for c in features if c != target_col]

        if not features:
            return _j({"error": "No numeric features found"})

        X = df[features].values
        y = df[target_col].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Run 3 methods
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_scaled, y)
        rf_imp = rf.feature_importances_

        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb.fit(X_scaled, y)
        gb_imp = gb.feature_importances_

        mi_scores = mutual_info_regression(X, y, random_state=42)
        # Normalize MI to 0-1
        mi_max = max(mi_scores) if max(mi_scores) > 0 else 1
        mi_norm = mi_scores / mi_max

        # Correlation with target
        corr_scores = np.array([abs(float(df[f].corr(df[target_col]))) for f in features])

        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=16, fontweight="bold")

        methods = [
            ("Random Forest", rf_imp, "steelblue"),
            ("Gradient Boosting", gb_imp, "darkorange"),
            ("Mutual Information", mi_norm, "seagreen"),
            ("Correlation", corr_scores, "crimson"),
        ]

        for ax, (name, scores, color) in zip(axes.flat, methods):
            idx = np.argsort(scores)[::-1]
            sorted_features = [features[i] for i in idx]
            sorted_scores = scores[idx]

            bars = ax.barh(range(len(sorted_features)), sorted_scores, color=color, alpha=0.8)
            ax.set_yticks(range(len(sorted_features)))
            ax.set_yticklabels(sorted_features, fontsize=9)
            ax.set_title(name, fontsize=12, fontweight="bold")
            ax.invert_yaxis()
            ax.set_xlim(0, max(sorted_scores) * 1.2)

            for bar, score in zip(bars, sorted_scores):
                ax.text(score + max(sorted_scores) * 0.02, bar.get_y() + bar.get_height() / 2,
                        f"{score:.3f}", va="center", fontsize=8)

        fig.tight_layout()
        path = _save_fig(fig, "feature_importance")
        plt.close(fig)

        # Aggregate ranking
        all_ranks = {}
        for name, scores, _ in methods:
            ranked = np.argsort(scores)[::-1]
            for rank, idx in enumerate(ranked):
                feat = features[idx]
                all_ranks[feat] = all_ranks.get(feat, 0) + rank

        consensus = sorted(all_ranks.items(), key=lambda x: x[1])

        return _j({
            "chart": "feature_importance_comparison",
            "saved_to": path,
            "displayed": True,
            "methods_used": ["RandomForest", "GradientBoosting", "MutualInformation", "Correlation"],
            "consensus_ranking": [c[0] for c in consensus],
            "top_3_consensus": [c[0] for c in consensus[:3]],
        })

    # ─── 3. Time Series Plot ─────────────────────────

    def plot_time_series(
        columns: str,
        date_col: str,
        title: str = "Time Series",
    ) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = _get_df().copy()
        cols = [c.strip() for c in columns.split(",") if c.strip() in df.columns]

        if not cols or date_col not in df.columns:
            return _j({"error": "Invalid columns"})

        df = df.sort_values(date_col)

        n_cols = len(cols)
        fig, axes = plt.subplots(n_cols, 1, figsize=(14, 3 * n_cols), sharex=True)
        if n_cols == 1:
            axes = [axes]

        colors = plt.cm.Set2(np.linspace(0, 1, n_cols))

        for ax, col, color in zip(axes, cols, colors):
            ax.plot(df[date_col], df[col], color=color, linewidth=1.5, label=col)
            ax.fill_between(df[date_col], df[col], alpha=0.1, color=color)
            ax.set_ylabel(col, fontsize=10)
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Date", fontsize=11)
        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout()

        path = _save_fig(fig, "time_series")
        plt.close(fig)

        return _j({
            "chart": "time_series",
            "saved_to": path,
            "displayed": True,
            "columns_plotted": cols,
            "date_range": f"{df[date_col].min()} to {df[date_col].max()}",
            "n_data_points": len(df),
        })


    # ─── 6. Model Comparison Chart ──────────────────

    def plot_model_comparison(
        target_col: str,
        feature_cols: str,
        title: str = "Model Performance Comparison",
    ) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
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
        folds = min(5, len(df))

        models = {
            "OLS": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.1, max_iter=5000),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "Gradient Boost": GradientBoostingRegressor(n_estimators=100, random_state=42),
        }

        r2_results = {}
        rmse_results = {}

        for name, model in models.items():
            try:
                r2 = cross_val_score(model, X_scaled, y, cv=folds, scoring="r2")
                rmse = -cross_val_score(model, X_scaled, y, cv=folds, scoring="neg_root_mean_squared_error")
                r2_results[name] = r2
                rmse_results[name] = rmse
            except Exception:
                pass

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # R² boxplot
        labels = list(r2_results.keys())
        r2_data = [r2_results[l] for l in labels]
        bp1 = ax1.boxplot(r2_data, labels=labels, patch_artist=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        for patch, color in zip(bp1["boxes"], colors):
            patch.set_facecolor(color)
        ax1.set_title("R² Score (higher = better)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("R²")
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis="x", rotation=30)

        # RMSE boxplot
        rmse_data = [rmse_results[l] for l in labels]
        bp2 = ax2.boxplot(rmse_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp2["boxes"], colors):
            patch.set_facecolor(color)
        ax2.set_title("RMSE (lower = better)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("RMSE")
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis="x", rotation=30)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout()

        path = _save_fig(fig, "model_comparison")
        plt.close(fig)

        # Summary stats
        summary = {}
        for name in labels:
            summary[name] = {
                "r2_mean": round(float(r2_results[name].mean()), 4),
                "r2_std": round(float(r2_results[name].std()), 4),
                "rmse_mean": round(float(rmse_results[name].mean()), 4),
            }

        ranking = sorted(summary.items(), key=lambda x: x[1]["r2_mean"], reverse=True)

        return _j({
            "chart": "model_comparison",
            "saved_to": path,
            "displayed": True,
            "model_results": summary,
            "ranking": [r[0] for r in ranking],
            "best_model": ranking[0][0],
            "best_r2": ranking[0][1]["r2_mean"],
        })

    # ─────────────────────────────────────────────
    # ASSEMBLE TOOLS
    # ─────────────────────────────────────────────

    tools = [

        StructuredTool.from_function(
            func=plot_feature_importance,
            name="plot_feature_importance",
            description=(
                "Generate a 4-panel feature importance chart comparing Random Forest, "
                "Gradient Boosting, Mutual Information, and Correlation methods side by side. "
                "Returns consensus ranking of features."
            ),
            args_schema=FeatureImportanceChartInput,
        ),
        StructuredTool.from_function(
            func=plot_time_series,
            name="plot_time_series",
            description=(
                "Plot one or more columns as time series with date on x-axis. "
                "Each column gets its own subplot. Good for visualizing trends and seasonality."
            ),
            args_schema=TimeSeriesPlotInput,
        ),

        StructuredTool.from_function(
            func=plot_model_comparison,
            name="plot_model_comparison",
            description=(
                "Train 5 models (OLS, Ridge, Lasso, RF, GB), cross-validate, and generate "
                "boxplot comparison charts for R² and RMSE. Returns ranking of best models."
            ),
            args_schema=ModelComparisonChartInput,
        ),
    ]

    logger.info(f"Visualization tools: {len(tools)} tools built")
    return tools
