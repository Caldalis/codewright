from __future__ import annotations

from pathlib import Path

from pydantic import Field

from codewright.tools.errors import RespondToModelError
from codewright.tools.handler import ToolHandler
from codewright.tools.handlers._file_tools import (
    DEFAULT_READ_LIMIT,
    MAX_BODY_CHARS,
    MAX_READ_LIMIT,
    coerce_params,
    looks_binary,
    reject_private_path,
    relative_to_workspace,
    strip_line_end,
    trim_line,
    truncate_text,
)
from codewright.tools.invocation import ToolInvocation
from codewright.tools.result import ToolResult
from codewright.tools.spec import ParameterModel, ToolSpec


class ReadFileParams(ParameterModel):
    path: str = Field(..., min_length=1, description="Workspace-relative file path to read.")
    offset: int = Field(
        0,
        ge=0,
        description="Zero-based line offset. The default reads from the first line.",
    )
    limit: int = Field(
        DEFAULT_READ_LIMIT,
        ge=1,
        le=MAX_READ_LIMIT,
        description="Maximum number of lines to return.",
    )

class ReadFileHandler(ToolHandler):
    @property
    def tool_name(self) -> str:
        return "read_file"

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_name,
            description=(
                "Read a UTF-8 text file inside the workspace. Returns numbered "
                "lines and a next_offset when more content is available."
            ),
            parameters=ReadFileParams.to_json_schema(),
            supports_parallel=True,
        )

    async def handle(self, invocation: ToolInvocation) -> ToolResult:
        params = coerce_params(ReadFileParams, invocation.arguments, self.tool_name)
        workspace = invocation.session.workspace
        path = workspace.canonicalize(params.path)
        reject_private_path(path, workspace.root, self.tool_name)
        rel = relative_to_workspace(path, workspace.root)
        lines, has_more = _read_numbered_lines(path, rel, params.offset, params.limit)
        body = _format_body(rel, lines, params.offset, has_more)
        workspace.audit(
            {
                "tool": self.tool_name,
                "call_id": invocation.call_id,
                "path": str(path),
                "offset": params.offset,
                "limit": params.limit,
                "returned_lines": len(lines),
                "has_more": has_more,
            }
        )
        return ToolResult(
            success=True,
            body=body,
            structured_data={
                "path": rel,
                "offset": params.offset,
                "returned_lines": len(lines),
                "has_more": has_more,
                "next_offset": params.offset + len(lines) if has_more else None,
            },
        )


def _read_numbered_lines(
    path: Path,
    label: str,
    offset: int,
    limit: int,
) -> tuple[list[tuple[int, str]], bool]:
    if not path.exists():
        raise RespondToModelError(f"read_file: missing file '{label}'")
    if not path.is_file():
        raise RespondToModelError(f"read_file: '{label}' is not a file")
    if looks_binary(path):
        raise RespondToModelError(f"read_file: '{label}' appears to be binary")

    lines: list[tuple[int, str]] = []
    has_more = False
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            for line_number, raw in enumerate(fh, start=1):
                if line_number <= offset:
                    continue
                if len(lines) >= limit:
                    has_more = True
                    break
                lines.append((line_number, strip_line_end(raw)))
    except UnicodeDecodeError as exc:
        raise RespondToModelError(
            f"read_file: cannot read '{label}': file is not valid UTF-8"
        ) from exc
    except OSError as exc:
        raise RespondToModelError(
            f"read_file: cannot read '{label}': {type(exc).__name__}: {exc}"
        ) from exc
    return lines, has_more


def _format_body(
    rel: str,
    lines: list[tuple[int, str]],
    offset: int,
    has_more: bool,
) -> str:
    if lines:
        last_line = lines[-1][0]
        header = f"File: {rel}\nLines: {lines[0][0]}-{last_line}"
    else:
        header = f"File: {rel}\nLines: none at offset {offset}"
    if has_more:
        header += f"\nNext offset: {offset + len(lines)}"
    content = _format_numbered_lines(lines) if lines else "(no lines returned)"
    return truncate_text(f"{header}\n{content}", MAX_BODY_CHARS)


def _format_numbered_lines(lines: list[tuple[int, str]]) -> str:
    if not lines:
        return ""
    width = max(4, len(str(lines[-1][0])))
    return "\n".join(f"{line_no:>{width}} | {trim_line(text)}" for line_no, text in lines)
