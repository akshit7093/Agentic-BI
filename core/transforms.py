# =============================================================
# core/transforms.py — Signal transformations for MMM
# =============================================================

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# ADSTOCK
# ─────────────────────────────────────────────

def adstock_geometric(series: np.ndarray, decay: float) -> np.ndarray:
    """Classic geometric (exponential) adstock."""
    decay = float(np.clip(decay, 0.0, 0.999))
    result = np.zeros_like(series, dtype=float)
    result[0] = series[0]
    for t in range(1, len(series)):
        result[t] = series[t] + decay * result[t - 1]
    return result


def adstock_weibull_pdf(series: np.ndarray, shape: float, scale: float, max_lag: int = 13) -> np.ndarray:
    """Weibull PDF adstock — more flexible lag distribution."""
    from scipy.stats import weibull_min
    lags = np.arange(0, max_lag)
    weights = weibull_min.pdf(lags + 1, c=shape, scale=scale)
    weights /= weights.sum() + 1e-10
    padded = np.concatenate([np.zeros(max_lag - 1), series])
    result = np.convolve(padded, weights[::-1], mode="valid")[: len(series)]
    return result


# ─────────────────────────────────────────────
# SATURATION / RESPONSE CURVES
# ─────────────────────────────────────────────

def hill_transform(series: np.ndarray, half_sat: float, slope: float) -> np.ndarray:
    """Hill / S-curve saturation."""
    x = np.clip(series, 0.0, None)
    denom = half_sat ** slope + x ** slope + 1e-10
    return x ** slope / denom


def log_saturation(series: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Logarithmic saturation: alpha * log(1 + x)."""
    x = np.clip(series, 0.0, None)
    return alpha * np.log1p(x)


def negative_exponential(series: np.ndarray, alpha: float = 1.0, gamma: float = 1.0) -> np.ndarray:
    """Negative-exponential saturation: alpha * (1 - exp(-gamma * x))."""
    x = np.clip(series, 0.0, None)
    return alpha * (1.0 - np.exp(-gamma * x))


# ─────────────────────────────────────────────
# APPLY PIPELINE
# ─────────────────────────────────────────────

def apply_transforms(
    df: pd.DataFrame,
    channel_cols: List[str],
    decay_map: Optional[Dict[str, float]] = None,
    half_sat_map: Optional[Dict[str, float]] = None,
    slope_map: Optional[Dict[str, float]] = None,
    saturation_fn: str = "hill",
    adstock_fn: str = "geometric",
    adstock_params: Optional[Dict[str, Dict]] = None,
) -> pd.DataFrame:
    """
    Apply adstock then saturation to each channel column.
    Returns df with new `<col>_transformed` columns.
    """
    df = df.copy()
    decay_map = decay_map or {}
    half_sat_map = half_sat_map or {}
    slope_map = slope_map or {}
    adstock_params = adstock_params or {}

    sat_fns = {"hill": hill_transform, "log": log_saturation, "negexp": negative_exponential}
    sat_f = sat_fns.get(saturation_fn, hill_transform)

    for col in channel_cols:
        arr = df[col].values.astype(float)

        # Adstock
        if adstock_fn == "weibull":
            params = adstock_params.get(col, {})
            arr = adstock_weibull_pdf(arr, params.get("shape", 2.0), params.get("scale", 3.0))
        else:
            arr = adstock_geometric(arr, decay_map.get(col, 0.5))

        # Saturation
        if saturation_fn == "hill":
            hs = half_sat_map.get(col, max(float(arr.mean()), 1e-5))
            sl = slope_map.get(col, 1.0)
            arr = hill_transform(arr, hs, sl)
        elif saturation_fn == "log":
            arr = log_saturation(arr)
        else:
            arr = negative_exponential(arr)

        df[f"{col}_transformed"] = arr

    return df


# ─────────────────────────────────────────────
# PARAMETER OPTIMISATION (grid + scipy)
# ─────────────────────────────────────────────

def fit_hill_params(
    x: np.ndarray, y: np.ndarray
) -> Tuple[float, float, float]:
    """
    Fit Hill curve params (half_sat, slope) via scipy curve_fit.
    Returns (half_sat, slope, r2).
    """
    from scipy.optimize import curve_fit

    x = np.clip(x.astype(float), 0, None)
    y = y.astype(float)

    def _hill(x_, hs, sl):
        return x_ ** sl / (hs ** sl + x_ ** sl + 1e-10)

    try:
        p0 = [float(np.median(x[x > 0])) if np.any(x > 0) else 1.0, 1.0]
        bounds = ([1e-6, 0.1], [np.ptp(x) * 10 + 1, 10.0])
        popt, _ = curve_fit(_hill, x, y, p0=p0, bounds=bounds, maxfev=5000)
        y_hat = _hill(x, *popt)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-10))
        return float(popt[0]), float(popt[1]), r2
    except Exception:
        return float(np.median(x[x > 0]) if np.any(x > 0) else 1.0), 1.0, 0.0


def optimize_adstock_params(
    df: pd.DataFrame,
    channel_col: str,
    kpi_col: str,
    decay_grid: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Grid-search optimal adstock decay + hill params for a channel vs KPI.
    Returns best {decay, half_sat, slope, r2}.
    """
    decay_grid = decay_grid if decay_grid is not None else np.linspace(0.1, 0.9, 17)
    x_raw = df[channel_col].values.astype(float)
    y = df[kpi_col].values.astype(float)

    best = {"decay": 0.5, "half_sat": 1.0, "slope": 1.0, "r2": -np.inf}
    for decay in decay_grid:
        x_ads = adstock_geometric(x_raw, decay)
        hs, sl, r2 = fit_hill_params(x_ads, y)
        if r2 > best["r2"]:
            best = {"decay": float(decay), "half_sat": hs, "slope": sl, "r2": r2}
    return best
