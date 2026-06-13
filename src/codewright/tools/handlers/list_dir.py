from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from codewright.tools.errors import RespondToModelError
from codewright.tools.handler import ToolHandler
from codewright.tools.handlers._file_tools import (
    DEFAULT_LIST_LIMIT,
    MAX_BODY_CHARS,
    MAX_MATCH_LIMIT,
    coerce_params,
    entry_kind,
    reject_private_path,
    relative_to_workspace,
    truncate_text,
)
from codewright.tools.invocation import ToolInvocation
from codewright.tools.result import ToolResult
from codewright.tools.spec import ParameterModel, ToolSpec


class ListDirParams(ParameterModel):
    path: str = Field(..., min_length=1, description="Workspace-relative directory path to list.")
    limit: int = Field(
        DEFAULT_LIST_LIMIT,
        ge=1,
        le=MAX_MATCH_LIMIT,
        description="Maximum number of entries to return (directories first).",
    )


class ListDirHandler(ToolHandler):
    @property
    def tool_name(self) -> str:
        return "list_dir"

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_name,
            description=(
                "List one workspace directory without recursing. Entries are "
                "stable-sorted with directories first."
            ),
            parameters=ListDirParams.to_json_schema(),
            supports_parallel=True,
        )

    async def handle(self, invocation: ToolInvocation) -> ToolResult:
        params = coerce_params(ListDirParams, invocation.arguments, self.tool_name)
        workspace = invocation.session.workspace
        path = workspace.canonicalize(params.path)
        reject_private_path(path, workspace.root, self.tool_name)
        rel = relative_to_workspace(path, workspace.root)
        entries, limit_reached = _list_entries(path, rel, params.limit)
        body = _format_body(rel, entries, limit_reached)
        workspace.audit(
            {
                "tool": self.tool_name,
                "call_id": invocation.call_id,
                "path": str(path),
                "entries": len(entries),
                "limit_reached": limit_reached,
            }
        )
        return ToolResult(
            success=True,
            body=body,
            structured_data={
                "path": rel,
                "entries": entries,
                "limit_reached": limit_reached,
            },
        )

def _list_entries(
    path: Path, label: str, limit: int
) -> tuple[list[dict[str, Any]], bool]:
    if not path.exists():
        raise RespondToModelError(f"list_dir: missing directory '{label}'")
    if not path.is_dir():
        raise RespondToModelError(f"list_dir: '{label}' is not a directory")
    entries: list[dict[str, Any]] = []
    try:
        children = sorted(
            path.iterdir(),
            key=lambda p: (_sort_rank(entry_kind(p)), p.name.lower(), p.name),
        )
    except OSError as exc:
        raise RespondToModelError(
            f"list_dir: cannot list '{label}': {type(exc).__name__}: {exc}"
        ) from exc
    limit_reached = len(children) > limit
    for child in children[:limit]:
        try:
            stat = child.lstat()
            kind = entry_kind(child)
            entries.append(
                {
                    "name": child.name,
                    "kind": kind,
                    "size": stat.st_size if kind == "file" else None,
                }
            )
        except OSError:
            entries.append({"name": child.name, "kind": "unknown", "size": None})
    return entries, limit_reached


def _sort_rank(kind: str) -> int:
    if kind == "dir":
        return 0
    if kind == "file":
        return 1
    if kind == "symlink":
        return 2
    return 3

def _format_body(
    rel: str, entries: list[dict[str, Any]], limit_reached: bool
) -> str:
    if not entries:
        return f"Directory: {rel}\n(empty)"
    suffix = " (limit reached)" if limit_reached else ""
    lines = [f"Directory: {rel}", f"Entries: {len(entries)}{suffix}"]
    for entry in entries:
        name = entry["name"] + ("/" if entry["kind"] == "dir" else "")
        size = "" if entry["size"] is None else f" ({entry['size']} bytes)"
        lines.append(f"[{entry['kind']}] {name}{size}")
    return truncate_text("\n".join(lines), MAX_BODY_CHARS)
