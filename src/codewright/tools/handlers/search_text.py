from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from pydantic import Field

from codewright.tools.errors import RespondToModelError
from codewright.tools.handler import ToolHandler
from codewright.tools.handlers._file_tools import (
    DEFAULT_SEARCH_LIMIT,
    MAX_BODY_CHARS,
    MAX_MATCH_LIMIT,
    FileEntry,
    coerce_params,
    looks_binary,
    matches_glob,
    path_kind,
    reject_private_path,
    relative_to_workspace,
    safe_relative,
    strip_line_end,
    trim_line,
    truncate_text,
    walk_files,
)
from codewright.tools.invocation import ToolInvocation
from codewright.tools.result import ToolResult
from codewright.tools.spec import ParameterModel, ToolSpec

_MAX_REGEX_LINE_CHARS = 4000


class SearchTextParams(ParameterModel):
    pattern: str = Field(
        ...,
        min_length=1,
        description="Text to search for. Treated as a literal string unless regex=true.",
    )
    path: str = Field(
        ".",
        min_length=1,
        description="Workspace-relative file or directory to search. Defaults to the workspace root.",
    )
    glob: str | None = Field(
        None,
        min_length=1,
        description=(
            "Optional file glob filter. Bare patterns like '*.py' match filenames "
            "in any directory; patterns with '/' match workspace-relative paths."
        ),
    )
    limit: int = Field(
        DEFAULT_SEARCH_LIMIT,
        ge=1,
        le=MAX_MATCH_LIMIT,
        description="Maximum number of text matches to return.",
    )
    regex: bool = Field(
        False,
        description=(
            "Interpret pattern as a Python regular expression. Defaults to false "
            "so normal searches are safe literal substring searches."
        ),
    )

class SearchTextHandler(ToolHandler):
    @property
    def tool_name(self) -> str:
        return "search_text"

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_name,
            description=(
                "Search UTF-8 text files. By default pattern is a literal "
                "substring; set regex=true to use a Python regular expression. "
                "Returns path, line, column, and a trimmed line preview for each "
                "match. Note: in regex mode only the first "
                f"{_MAX_REGEX_LINE_CHARS} characters of each line are searched, "
                "so matches further into a very long (e.g. minified) line can be "
                "missed; for long lines prefer a literal search or narrow the "
                "scope with path/glob. Literal search has no per-line limit."
            ),
            parameters=SearchTextParams.to_json_schema(),
            supports_parallel=True,
        )

    async def handle(self, invocation: ToolInvocation) -> ToolResult:
        params = coerce_params(SearchTextParams, invocation.arguments, self.tool_name)
        regex = _compile_regex(params.pattern) if params.regex else None
        workspace = invocation.session.workspace
        start = workspace.canonicalize(params.path)
        reject_private_path(start, workspace.root, self.tool_name)
        start_rel = relative_to_workspace(start, workspace.root)
        matches: list[dict[str, Any]] = []
        skipped: list[str] = []
        limit_reached = False
        regex_truncated_lines = 0
        async for entry in _iter_files(workspace.root, start, invocation):
            if params.glob and not matches_glob(entry.rel, params.glob):
                continue
            result = _search_file(
                entry,
                pattern=params.pattern,
                regex=regex,
                cancellation_token=invocation.cancellation_token,
            )
            if result is None:
                skipped.append(entry.rel)
                continue
            file_matches, truncated = result
            regex_truncated_lines += truncated
            for match in file_matches:
                matches.append(match)
                if len(matches) >= params.limit:
                    limit_reached = True
                    break
            if limit_reached:
                break

        body = _format_body(
            params.pattern, start_rel, matches, skipped, limit_reached,
            regex_truncated_lines,
        )
        workspace.audit(
            {
                "tool": self.tool_name,
                "call_id": invocation.call_id,
                "pattern": params.pattern,
                "regex": params.regex,
                "path": str(start),
                "glob": params.glob,
                "matches": len(matches),
                "skipped": len(skipped),
                "limit_reached": limit_reached,
                "regex_truncated_lines": regex_truncated_lines,
            }
        )
        return ToolResult(
            success=True,
            body=body,
            structured_data={
                "matches": matches,
                "skipped": skipped,
                "limit_reached": limit_reached,
                "regex_truncated_lines": regex_truncated_lines,
            },
        )

async def _iter_files(root: Path, start: Path, invocation: ToolInvocation):
    kind = path_kind(start)
    if kind == "file":
        rel = safe_relative(start, root)
        if rel is not None:
            yield FileEntry(path=start, rel=rel)
        return
    if kind == "missing":
        label = relative_to_workspace(start, root)
        raise RespondToModelError(f"search_text: missing path '{label}'")
    if kind != "dir":
        label = relative_to_workspace(start, root)
        raise RespondToModelError(f"search_text: '{label}' is not a file or directory")
    async for entry in walk_files(root, start, invocation):
        yield entry

def _compile_regex(pattern: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise RespondToModelError(f"search_text: invalid regex pattern: {exc}") from exc

def _search_file(
    entry: FileEntry,
    *,
    pattern: str,
    regex: re.Pattern[str] | None,
    cancellation_token,
) -> tuple[list[dict[str, Any]], int] | None:
    """Return (matches, regex_truncated_lines), or None if the file is skipped.
    regex_truncated_lines counts lines longer than the regex search window: in
    regex mode only their prefix is searched, so matches past it are missed.
    """
    try:
        if looks_binary(entry.path):
            return None
    except RespondToModelError:
        return None

    matches: list[dict[str, Any]] = []
    truncated = 0
    try:
        with entry.path.open("r", encoding="utf-8", newline="") as fh:
            for line_number, raw in enumerate(fh, start=1):
                if cancellation_token.is_cancelled():
                    raise asyncio.CancelledError
                line = strip_line_end(raw)
                if regex is not None and len(line) > _MAX_REGEX_LINE_CHARS:
                    truncated += 1
                for column in _match_columns(line, pattern, regex):
                    matches.append(
                        {
                            "path": entry.rel,
                            "line": line_number,
                            "column": column + 1,
                            "text": trim_line(line),
                        }
                    )
    except asyncio.CancelledError:
        raise
    except (UnicodeDecodeError, OSError):
        return None
    return matches, truncated


def _match_columns(
    line: str,
    pattern: str,
    regex: re.Pattern[str] | None,
) -> list[int]:
    if regex is not None:
        search_line = line[:_MAX_REGEX_LINE_CHARS]
        return [found.start() for found in regex.finditer(search_line)]

    columns: list[int] = []
    start = 0
    while True:
        column = line.find(pattern, start)
        if column < 0:
            return columns
        columns.append(column)
        start = column + len(pattern)


def _format_body(
    pattern: str,
    path: str,
    matches: list[dict[str, Any]],
    skipped: list[str],
    limit_reached: bool,
    regex_truncated_lines: int = 0,
) -> str:
    lines = [f"Pattern: {pattern}", f"Path: {path}"]
    if not matches:
        lines.append("Matches: none")
    else:
        suffix = " (limit reached)" if limit_reached else ""
        lines.append(f"Matches: {len(matches)}{suffix}")
        for item in matches:
            lines.append(
                f"{item['path']}:{item['line']}:{item['column']}: {item['text']}"
            )
    if skipped:
        shown = ", ".join(skipped[:5])
        more = "" if len(skipped) <= 5 else f", +{len(skipped) - 5} more"
        lines.append(f"Skipped non-text/unreadable files: {shown}{more}")
    if regex_truncated_lines:
        lines.append(
            f"Note: {regex_truncated_lines} line(s) exceeded the "
            f"{_MAX_REGEX_LINE_CHARS}-char regex window and were only searched "
            "up to that point; matches past it may be missed. For long lines use "
            "a literal search or narrow the scope with path/glob."
        )
    return truncate_text("\n".join(lines), MAX_BODY_CHARS)
