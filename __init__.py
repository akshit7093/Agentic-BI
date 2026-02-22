# agentic_mmm/__init__.py
from .main import NotebookMMM, init_mmm
from .core.mmm_engine import MMMEngine

__all__ = ["NotebookMMM", "init_mmm", "MMMEngine"]
__version__ = "2.0.0"
