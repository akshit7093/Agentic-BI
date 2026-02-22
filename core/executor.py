# =============================================================
# core/executor.py — Safe sandboxed code execution
# =============================================================

import sys
import io
import re
import json
import math
import traceback
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import numpy as np

from ..config import FORBIDDEN_PATTERNS


class SafeCodeExecutor:
    """
    Sandboxed Python execution.
    - Blocks dangerous imports / builtins
    - Captures stdout
    - Returns structured result
    """

    @staticmethod
    def validate_code(code: str) -> Tuple[bool, str]:
        """Check for forbidden patterns."""
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Forbidden pattern detected: {pattern}"
        return True, "OK"

    @classmethod
    def _build_safe_globals(cls, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Build a restricted globals dict for exec()."""
        # Lazy imports so only what's available is exposed
        safe = {
            "__builtins__": {
                "print": print,
                "len": len, "range": range, "enumerate": enumerate,
                "zip": zip, "map": map, "filter": filter,
                "list": list, "dict": dict, "set": set, "tuple": tuple,
                "str": str, "int": int, "float": float, "bool": bool,
                "min": min, "max": max, "sum": sum, "abs": abs,
                "round": round, "sorted": sorted, "reversed": reversed,
                "isinstance": isinstance, "hasattr": hasattr,
                "getattr": getattr, "type": type,
                "True": True, "False": False, "None": None,
            },
            "pd": pd,
            "pandas": pd,
            "np": np,
            "numpy": np,
            "math": math,
            "json": json,
            "re": re,
            "df": df,
            "result": None,
            "output": None,
        }
        # Optional: scipy, sklearn
        try:
            import scipy.stats as _sts
            import scipy.optimize as _sco
            safe["scipy_stats"] = _sts
            safe["scipy_optimize"] = _sco
            from scipy.optimize import curve_fit, minimize, differential_evolution
            safe["curve_fit"] = curve_fit
            safe["minimize"] = minimize
            safe["differential_evolution"] = differential_evolution
        except ImportError:
            pass
        try:
            from sklearn import preprocessing, linear_model, metrics
            safe["preprocessing"] = preprocessing
            safe["linear_model"] = linear_model
            safe["sk_metrics"] = metrics
        except ImportError:
            pass
        try:
            import statsmodels.api as sm
            safe["sm"] = sm
        except ImportError:
            pass
        return safe

    @classmethod
    def execute(
        cls,
        code: str,
        df: Optional[pd.DataFrame] = None,
        extra_vars: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute *code* in sandbox.
        Returns:
            success, output (stdout), result (any), error, columns (if df)
        """
        is_valid, msg = cls.validate_code(code)
        if not is_valid:
            return {"success": False, "error": msg, "output": "", "result": None, "columns": []}

        safe_globals = cls._build_safe_globals(df)
        if extra_vars:
            safe_globals.update(extra_vars)
        safe_locals: Dict[str, Any] = {}

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            exec(compile(code, "<sandbox>", "exec"), safe_globals, safe_locals)  # noqa: S102
            captured = sys.stdout.getvalue()

            # Prefer local 'result', then global
            result = safe_locals.get("result", safe_globals.get("result"))

            return cls._package_result(result, captured)

        except Exception as exc:
            captured = sys.stdout.getvalue()
            return {
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=5),
                "output": captured,
                "result": None,
                "columns": [],
            }
        finally:
            sys.stdout = old_stdout

    @staticmethod
    def _package_result(result: Any, output: str) -> Dict[str, Any]:
        """Convert result into a JSON-serialisable dict."""
        if isinstance(result, pd.DataFrame):
            return {
                "success": True,
                "output": output,
                "result": {
                    "type": "dataframe",
                    "columns": list(result.columns),
                    "rows": result.reset_index(drop=True).head(100).to_dict(orient="records"),
                    "shape": list(result.shape),
                },
                "columns": list(result.columns),
            }
        if isinstance(result, pd.Series):
            df_r = result.reset_index()
            df_r.columns = [str(c) for c in df_r.columns]
            return {
                "success": True,
                "output": output,
                "result": {
                    "type": "series",
                    "columns": list(df_r.columns),
                    "rows": df_r.head(100).to_dict(orient="records"),
                },
                "columns": list(df_r.columns),
            }
        if isinstance(result, np.ndarray):
            return {"success": True, "output": output, "result": result.tolist(), "columns": []}
        if isinstance(result, (np.integer, np.floating)):
            return {"success": True, "output": output, "result": float(result), "columns": []}
        if isinstance(result, dict):
            # Convert numpy types inside dict
            def _clean(v: Any) -> Any:
                if isinstance(v, (np.integer, np.floating)):
                    return float(v)
                if isinstance(v, np.ndarray):
                    return v.tolist()
                return v
            cleaned = {k: _clean(v) for k, v in result.items()}
            return {"success": True, "output": output, "result": cleaned, "columns": list(cleaned.keys())}
        if isinstance(result, list):
            return {"success": True, "output": output, "result": result, "columns": []}
        return {"success": True, "output": output, "result": str(result) if result is not None else output, "columns": []}
