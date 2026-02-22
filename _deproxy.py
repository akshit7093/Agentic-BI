# =============================================================
# _deproxy.py — Neutralize Rich's FileProxy on sys.stdout/stderr
# =============================================================
# Rich installs a FileProxy on sys.stdout/stderr when imported.
# In Databricks/Jupyter, this creates an infinite recursion:
#   print() → stdout.flush() → FileProxy.flush() → Console.print()
#   → ipython_display() → publish_display_data() → _flush_streams()
#   → stdout.flush() → FileProxy.flush() → ...
#
# This module provides two defences:
#   1. deproxy_stdio() — strips FileProxy from sys.stdout/stderr
#   2. safe_print() — always writes to sys.__stdout__ (never proxied)
# =============================================================

import sys


def deproxy_stdio():
    """
    Remove Rich's FileProxy from sys.stdout and sys.stderr.

    Rich's FileProxy stores the original file as `_FileProxy__file`.
    We extract it and put it back as the real stdout/stderr.
    """
    for stream_name in ('stdout', 'stderr'):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue

        # Check by class name to avoid importing rich
        cls_name = type(stream).__name__
        cls_module = getattr(type(stream), '__module__', '')

        if cls_name == 'FileProxy' and 'rich' in cls_module:
            # Rich's FileProxy stores the original as _FileProxy__file
            original = getattr(stream, '_FileProxy__file', None)
            if original is not None:
                setattr(sys, stream_name, original)


def safe_print(*args, **kwargs):
    """
    Print that safely bypasses Rich's FileProxy but STILL goes to Jupyter/Databricks cell output.
    """
    out = sys.stdout
    
    # If sys.stdout is currently Rich's FileProxy, extract the real underlying file 
    # (e.g. Jupyter's OutStream) to avoid recursion, while still printing to the frontend.
    cls_name = type(out).__name__
    cls_module = getattr(type(out), '__module__', '')
    if cls_name == 'FileProxy' and 'rich' in cls_module:
        original = getattr(out, '_FileProxy__file', None)
        if original is not None:
            out = original
            
    kwargs['file'] = out
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


# Run deproxy immediately on import
deproxy_stdio()
