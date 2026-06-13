from __future__ import annotations

import asyncio
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from codewright.tools.errors import RespondToModelError
from codewright.tools.invocation import ToolInvocation
from codewright.tools.spec import ParameterModel

DEFAULT_READ_LIMIT = 200
MAX_READ_LIMIT = 1000
DEFAULT_FIND_LIMIT = 100
DEFAULT_SEARCH_LIMIT = 100
DEFAULT_LIST_LIMIT = 1000
MAX_MATCH_LIMIT = 1000
MAX_BODY_CHARS = 30_000
MAX_LINE_CHARS = 500
BINARY_SAMPLE_BYTES = 8192
YIELD_EVERY = 100
SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        ".codewright",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
    }
)
PRIVATE_DIR_NAMES = frozenset({".codewright"})

@dataclass(frozen=True, slots=True)
class FileEntry:
    path: Path
    rel: str




def coerce_params(model: type[ParameterModel], arguments: dict, tool_name: str) -> Any:
    try:
        return model(**arguments)
    except Exception as exc:
        raise RespondToModelError(f"{tool_name}: invalid arguments: {exc}") from exc

def relative_to_workspace(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError as exc:
        raise RespondToModelError(f"path escapes workspace root: {path}") from exc
    text = rel.as_posix()
    return "." if text == "" else text

def reject_private_path(path: Path, root: Path, tool_name: str) -> None:
    rel = relative_to_workspace(path, root)
    if rel == ".":
        return
    first = rel.split("/", 1)[0]
    if first in PRIVATE_DIR_NAMES:
        raise RespondToModelError(
            f"{tool_name}: access to agent-private path '{first}' is not allowed"
        )

def path_kind(path: Path) -> str:
    if path.is_file():
        return "file"
    if path.is_dir():
        return "dir"
    if path.exists():
        return "other"
    return "missing"

def entry_kind(path: Path) -> str:
    try:
        mode = path.lstat().st_mode
    except OSError:
        return "unknown"
    if stat.S_ISLNK(mode) or _is_junction(path):
        return "symlink"
    if stat.S_ISDIR(mode):
        return "dir"
    if stat.S_ISREG(mode):
        return "file"
    return "other"

def _is_junction(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    if is_junction is None:
        return False
    try:
        return bool(is_junction())
    except OSError:
        return False

async def walk_files(root: Path, start: Path, invocation: ToolInvocation):
    visited = 0
    for dirpath_text, dirnames, filenames in os.walk(start, topdown=True, followlinks=False):
        dirpath = Path(dirpath_text)
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in SKIP_DIR_NAMES and entry_kind(dirpath / name) == "dir"
        )
        for name in sorted(filenames):
            if invocation.cancellation_token.is_cancelled():
                raise asyncio.CancelledError
            path = dirpath / name
            rel = safe_relative(path, root)
            if rel is None:
                continue
            yield FileEntry(path=path, rel=rel)
            visited += 1
            if visited % YIELD_EVERY == 0:
                await asyncio.sleep(0)

def safe_relative(path: Path, root: Path) -> str | None:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
        rel = path.relative_to(root)
    except (OSError, ValueError):
        return None
    return rel.as_posix()

def matches_glob(rel: str, pattern: str) -> bool:
    normalized = pattern.replace("\\", "/")
    path = PurePosixPath(rel)
    if path.match(normalized):
        return True
    if "**/" in normalized:
        return path.match(normalized.replace("**/", ""))
    if "/" not in normalized:
        return PurePosixPath(path.name).match(normalized)
    return False
def looks_binary(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return b"\0" in fh.read(BINARY_SAMPLE_BYTES)
    except OSError as exc:
        label = path.as_posix()
        raise RespondToModelError(
            f"cannot inspect '{label}': {type(exc).__name__}: {exc}"
        ) from exc
def strip_line_end(line: str) -> str:
    return line.removesuffix("\n").removesuffix("\r")
def trim_line(text: str) -> str:
    if len(text) <= MAX_LINE_CHARS:
        return text
    marker = " ...[line truncated]... "
    keep = max(1, (MAX_LINE_CHARS - len(marker)) // 2)
    return text[:keep] + marker + text[-keep:]
def truncate_text(text: str, max_chars: int = MAX_BODY_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    marker = f"\n...[{len(text) - max_chars} chars truncated]...\n"
    keep = max(1, (max_chars - len(marker)) // 2)
    return text[:keep] + marker + text[-keep:]
