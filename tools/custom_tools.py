# =============================================================
# tools/custom_tools.py — Agent-created on-the-run custom tools
# =============================================================

import json
import logging
import sys
from typing import Any, Dict, List, Optional

from .._deproxy import safe_print

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from .registry import ToolRegistry, ToolSpec
from ..core.mmm_engine import MMMEngine

logger = logging.getLogger(__name__)


# ─── Input schemas ────────────────────────────────────────────

class CreateCustomToolInput(BaseModel):
    name: str = Field(description="Unique snake_case name for the new tool")
    description: str = Field(description="What this tool does (used as LLM tool description)")
    code: str = Field(
        description=(
            "Python code defining `tool_fn(**kwargs) -> str`. "
            "Must return a string (ideally JSON). "
            "Has access to: json, pd, np, df (current dataframe), scipy, sklearn."
        )
    )
    params: str = Field(
        default="{}",
        description='JSON string mapping param_name -> type. E.g. {"column": "str", "n": "int"}',
    )
    tags: str = Field(default="", description="Comma-separated tags for discovery")


class RemoveCustomToolInput(BaseModel):
    name: str = Field(description="Name of the custom tool to remove")


class UserFeedbackToolInput(BaseModel):
    question: str = Field(description="Question or clarification to present to the user")
    options: Optional[str] = Field(
        default=None,
        description="Optional comma-separated options (e.g. 'Yes,No,Maybe')",
    )


class AddAnalysisNoteInput(BaseModel):
    note: str = Field(description="Insight, finding or hypothesis to record in analysis log")
    category: str = Field(default="general", description="Category: insight | warning | hypothesis | action | finding")


# ─────────────────────────────────────────────
def build_meta_tools(registry: ToolRegistry, engine: MMMEngine) -> List[StructuredTool]:
    """
    Meta-tools: create tools, ask user, log notes, inspect registry.
    """

    def _j(obj) -> str:
        return json.dumps(obj, default=str)

    # ── Dynamic tool creation ──

    def create_custom_tool(
        name: str,
        description: str,
        code: str,
        params: str = "{}",
        tags: str = "",
    ) -> str:
        try:
            params_dict: Dict[str, str] = json.loads(params) if params.strip() else {}
            tags_list = [t.strip() for t in tags.split(",") if t.strip()]
        except Exception as exc:
            return _j({"success": False, "error": f"Invalid params JSON: {exc}"})

        spec = ToolSpec(
            name=name.strip().replace(" ", "_"),
            description=description.strip(),
            code=code,
            params=params_dict,
            tags=tags_list,
        )
        # Pass engine data access into the tool
        extra = {"df": engine.data, "engine": engine}
        result = registry.register_from_spec(spec, extra_globals=extra)
        return _j(result)


    def remove_custom_tool(name: str) -> str:
        removed = registry.remove(name.strip())
        return _j({"success": removed, "removed_tool": name if removed else None})


    # ── User interaction ──

    def ask_user_for_input(question: str, options: Optional[str] = None) -> str:
        """Ask user a question and return their response (blocking).
        Uses safe_print() to avoid Rich FileProxy recursion in Databricks."""
        safe_print(f"\n{'='*60}")
        safe_print(f"🤖 Agent needs clarification:")
        safe_print(f"   {question}")
        if options:
            opt_list = [o.strip() for o in options.split(",")]
            for i, o in enumerate(opt_list, 1):
                safe_print(f"   {i}. {o}")
            try:
                user_in = input("Your choice (number or text): ").strip()
                # If numeric, map to option
                try:
                    idx = int(user_in) - 1
                    if 0 <= idx < len(opt_list):
                        user_in = opt_list[idx]
                except ValueError:
                    pass
            except EOFError:
                user_in = ""
        else:
            try:
                user_in = input("Your answer: ").strip()
            except EOFError:
                user_in = ""
        safe_print(f"{'='*60}")
        return _j({"success": True, "question": question, "user_response": user_in})


    # ── Analysis logging ──

    def add_analysis_note(note: str, category: str = "general") -> str:
        entry = {
            "type": "analysis_note",
            "category": category,
            "note": note.strip(),
        }
        engine.analysis_history.append(entry)
        return _j({"success": True, "logged": entry, "total_notes": len(engine.analysis_history)})


    return [
        StructuredTool(
            name="create_custom_tool",
            func=create_custom_tool,
            args_schema=CreateCustomToolInput,
            description=(
                "CREATE a brand-new analysis tool on the fly. "
                "Define a Python function `tool_fn(**kwargs) -> str` and it becomes immediately callable. "
                "Use this when you need a specialised analysis not covered by existing tools. "
                "The code has access to: df (dataframe), engine, pd, np, json, scipy, sklearn."
            ),
        ),

        StructuredTool(
            name="remove_custom_tool",
            func=remove_custom_tool,
            args_schema=RemoveCustomToolInput,
            description="Remove a dynamically created tool from the registry.",
        ),
        StructuredTool(
            name="ask_user",
            func=ask_user_for_input,
            args_schema=UserFeedbackToolInput,
            description=(
                "Ask the user a clarifying question and get their response. "
                "Use when you need user confirmation, preferences, or missing information "
                "(e.g., which KPI to use, confirm an expensive operation)."
            ),
        ),

        StructuredTool(
            name="add_analysis_note",
            func=add_analysis_note,
            args_schema=AddAnalysisNoteInput,
            description=(
                "Log an insight, finding, warning, or hypothesis to the analysis history. "
                "Use this to track what you've discovered and what to investigate next."
            ),
        ),
    ]
