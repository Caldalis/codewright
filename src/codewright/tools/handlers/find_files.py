from __future__ import annotations

from pydantic import Field

from codewright.tools.errors import RespondToModelError
from codewright.tools.handler import ToolHandler
from codewright.tools.handlers._file_tools import (
    DEFAULT_FIND_LIMIT,
    MAX_BODY_CHARS,
    MAX_MATCH_LIMIT,
    coerce_params,
    matches_glob,
    truncate_text,
    walk_files,
)
from codewright.tools.invocation import ToolInvocation
from codewright.tools.result import ToolResult
from codewright.tools.spec import ParameterModel, ToolSpec


class FindFilesParams(ParameterModel):
    query: str | None = Field(
        None,
        min_length=1,
        description=(
            "Filename or relative-path substring to search for. If it contains "
            "glob metacharacters such as '*' or '?', it is treated as a glob."
        ),
    )
    glob: str | None = Field(
        None,
        min_length=1,
        description=(
            "Optional glob filter. Bare patterns like '*.py' match filenames in "
            "any directory; patterns with '/' match workspace-relative paths."
        ),
    )
    limit: int = Field(
        DEFAULT_FIND_LIMIT,
        ge=1,
        le=MAX_MATCH_LIMIT,
        description="Maximum number of file paths to return.",
    )

class FindFilesHandler(ToolHandler):
    @property
    def tool_name(self) -> str:
        return "find_files"

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_name,
            description=(
                "Find files under the workspace by substring or glob. Recursion "
                "does not follow directory symlinks and skips common generated "
                "directories such as .git, .venv, and node_modules."
            ),
            parameters=FindFilesParams.to_json_schema(),
            supports_parallel=True,
        )

    async def handle(self, invocation: ToolInvocation) -> ToolResult:
        params = coerce_params(FindFilesParams, invocation.arguments, self.tool_name)
        if not params.query and not params.glob:
            raise RespondToModelError("find_files: provide query and/or glob")

        workspace = invocation.session.workspace
        matches: list[str] = []
        limit_reached = False
        async for entry in walk_files(workspace.root, workspace.root, invocation):
            if _matches(entry.rel, params.query, params.glob):
                matches.append(entry.rel)
                if len(matches) >= params.limit:
                    limit_reached = True
                    break

        body = _format_body(matches, limit_reached)
        workspace.audit(
            {
                "tool": self.tool_name,
                "call_id": invocation.call_id,
                "query": params.query,
                "glob": params.glob,
                "matches": len(matches),
                "limit_reached": limit_reached,
            }
        )
        return ToolResult(
            success=True,
            body=body,
            structured_data={
                "matches": matches,
                "limit_reached": limit_reached,
            },
        )

def _matches(rel: str, query: str | None, glob: str | None) -> bool:
    if query:
        if _is_glob_query(query):
            if not matches_glob(rel, query):
                return False
        elif query.lower() not in rel.lower():
            return False
    if glob and not matches_glob(rel, glob):
        return False
    return True

def _is_glob_query(query: str) -> bool:
    return any(ch in query for ch in "*?[")

def _format_body(matches: list[str], limit_reached: bool) -> str:
    if not matches:
        return "find_files: no matches"
    header = f"find_files: {len(matches)} match"
    if len(matches) != 1:
        header += "es"
    if limit_reached:
        header += " (limit reached)"
    return truncate_text(header + "\n" + "\n".join(matches), MAX_BODY_CHARS)
