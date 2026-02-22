# =============================================================
# tools/registry.py — Dynamic tool registry
# Allows the agent to CREATE, REGISTER, and USE new tools at runtime
# =============================================================

import uuid
import json
import logging
import textwrap
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, Field, create_model
from langchain_core.tools import StructuredTool

from ..core.executor import SafeCodeExecutor

logger = logging.getLogger(__name__)


# =============================================================
class ToolSpec(BaseModel):
    """Specification for a dynamically created tool."""
    name: str
    description: str
    code: str                          # Python code that implements the tool
    params: Dict[str, str] = Field(   # param_name -> type hint string
        default_factory=dict
    )
    author: str = "agent"
    version: int = 1
    tags: List[str] = Field(default_factory=list)


# =============================================================
class ToolRegistry:
    """
    Maintains all tools (built-in + dynamic).
    Agent can call register_custom_tool() to create new tools on the fly.
    """

    def __init__(self):
        self._tools: Dict[str, StructuredTool] = {}
        self._specs: Dict[str, ToolSpec]       = {}
        self._executor = SafeCodeExecutor()
        self._shared_context: Dict[str, Any]   = {}  # shared state across tool calls

    # ─────────────────────────────────────────────
    # REGISTER
    # ─────────────────────────────────────────────

    def register(self, tool: StructuredTool) -> None:
        """Register a pre-built StructuredTool."""
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def register_many(self, tools: List[StructuredTool]) -> None:
        for t in tools:
            self.register(t)

    def register_from_spec(self, spec: ToolSpec, extra_globals: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Compile a ToolSpec into a StructuredTool and register it.
        The code in spec.code must define a function named `tool_fn(*args, **kwargs) -> str`.
        """
        is_valid, msg = self._executor.validate_code(spec.code)
        if not is_valid:
            return {"success": False, "error": f"Unsafe code: {msg}"}

        # Build the pydantic input model dynamically
        type_map = {"str": str, "int": int, "float": float, "bool": bool, "list": list, "dict": dict, "any": Any}
        fields = {
            pname: (type_map.get(ptype.lower(), str), Field(description=pname))
            for pname, ptype in spec.params.items()
        }
        InputModel: Type[BaseModel] = create_model(f"{spec.name}Input", **fields)  # type: ignore[call-overload]

        # Exec the code to extract tool_fn
        globs: Dict[str, Any] = {
            "__builtins__": __builtins__,
            "json": json,
        }
        if extra_globals:
            globs.update(extra_globals)

        try:
            exec(spec.code, globs)  # noqa: S102
            fn: Callable = globs["tool_fn"]
        except Exception as exc:
            return {"success": False, "error": f"Code compilation error: {exc}"}

        def _wrapped(**kwargs):  # type: ignore[return]
            try:
                return str(fn(**kwargs))
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)})

        structured_tool = StructuredTool(
            name=spec.name,
            func=_wrapped,
            args_schema=InputModel,
            description=spec.description,
        )
        self._tools[spec.name] = structured_tool
        self._specs[spec.name] = spec
        logger.info(f"Dynamic tool registered: {spec.name}")
        return {"success": True, "tool_name": spec.name, "description": spec.description}

    # ─────────────────────────────────────────────
    # ACCESS
    # ─────────────────────────────────────────────

    def get(self, name: str) -> Optional[StructuredTool]:
        return self._tools.get(name)

    def get_all(self) -> List[StructuredTool]:
        return list(self._tools.values())

    def list_tools(self) -> List[Dict[str, str]]:
        return [
            {
                "name": t.name,
                "description": t.description[:120],
                "dynamic": t.name in self._specs,
            }
            for t in self._tools.values()
        ]

    def invoke(self, name: str, args: Dict[str, Any]) -> str:
        tool = self.get(name)
        if tool is None:
            return json.dumps({"success": False, "error": f"Tool '{name}' not found. Available: {list(self._tools.keys())}"})
        try:
            return str(tool.invoke(args))
        except Exception as exc:
            return json.dumps({"success": False, "error": str(exc)})

    def remove(self, name: str) -> bool:
        removed = self._tools.pop(name, None)
        self._specs.pop(name, None)
        return removed is not None

    def set_shared(self, key: str, value: Any) -> None:
        self._shared_context[key] = value

    def get_shared(self, key: str) -> Any:
        return self._shared_context.get(key)

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry({len(self._tools)} tools)"
