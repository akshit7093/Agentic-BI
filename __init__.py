# agentic_mmm/__init__.py

# CRITICAL: Must be the very first import to neutralize Rich's FileProxy
# before any print() calls trigger the recursion loop in Databricks/Jupyter
from . import _deproxy  # noqa: F401

from .main import NotebookMMM, init_mmm
from .core.mmm_engine import MMMEngine

__all__ = ["NotebookMMM", "init_mmm", "MMMEngine"]
__version__ = "2.0.0"
