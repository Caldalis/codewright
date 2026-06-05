from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath

BEGIN_PATCH = "*** Begin Patch"
END_PATCH = "*** End Patch"
ADD_FILE = "*** Add File: "
DELETE_FILE = "*** Delete File: "
UPDATE_FILE = "*** Update File: "
MOVE_TO = "*** Move to: "
END_OF_FILE = "*** End of File"
CHANGE_CONTEXT = "@@ "
EMPTY_CHANGE_CONTEXT = "@@"


class PatchParseError(ValueError):
    """Failed to parse a patch envelope. The handler turns this into a
    ``RespondToModelError`` so the model can repair its own patch."""


@dataclass
class Hunk:
    """One ``@@`` block inside an UpdateFile."""

    change_context: str | None
    old_lines: list[str] = field(default_factory=list)
    new_lines: list[str] = field(default_factory=list)
    is_end_of_file: bool = False


@dataclass
class FileOp:
    """One of three file-level operations."""

    kind: str  # "add" | "delete" | "update"
    path: str
    contents: str | None = None         # only for "add"
    move_to: str | None = None          # only for "update"
    hunks: list[Hunk] = field(default_factory=list)  # only for "update"


def parse_patch(text: str) -> list[FileOp]:

    lines = text.rstrip("\n").splitlines()
    if not lines:
        raise PatchParseError("empty patch")
    if lines[0].strip() != BEGIN_PATCH:
        raise PatchParseError("missing '*** Begin Patch' header")
    if lines[-1].strip() != END_PATCH:
        raise PatchParseError("missing '*** End Patch' footer")

    body = lines[1:-1]
    ops: list[FileOp] = []
    i = 0
    while i < len(body):
        line = body[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith(ADD_FILE):
            path = _validate_relative_path(stripped[len(ADD_FILE):].strip())
            contents_lines: list[str] = []
            i += 1
            while i < len(body):
                add_line = body[i]
                if not add_line.startswith("+"):
                    break
                contents_lines.append(add_line[1:])
                i += 1
            ops.append(
                FileOp(
                    kind="add",
                    path=path,
                    contents="\n".join(contents_lines) + ("\n" if contents_lines else ""),
                )
            )
            continue
        if stripped.startswith(DELETE_FILE):
            path = _validate_relative_path(stripped[len(DELETE_FILE):].strip())
            ops.append(FileOp(kind="delete", path=path))
            i += 1
            continue
        if stripped.startswith(UPDATE_FILE):
            path = _validate_relative_path(stripped[len(UPDATE_FILE):].strip())
            i += 1
            move_to: str | None = None
            if i < len(body) and body[i].lstrip().startswith(MOVE_TO):
                move_to = _validate_relative_path(
                    body[i].lstrip()[len(MOVE_TO):].strip()
                )
                i += 1
            hunks, consumed = _parse_hunks(body[i:])
            if not hunks:
                raise PatchParseError(
                    f"Update file hunk for '{path}' has no @@ chunks"
                )
            ops.append(
                FileOp(
                    kind="update",
                    path=path,
                    move_to=move_to,
                    hunks=hunks,
                )
            )
            i += consumed
            continue
        raise PatchParseError(
            f"unexpected line at top level: {line!r}; "
            f"expected '*** Add File:', '*** Delete File:', '*** Update File:'"
        )
    return ops


def _parse_hunks(lines: list[str]) -> tuple[list[Hunk], int]:
    hunks: list[Hunk] = []
    i = 0
    current_hunk: Hunk | None = None
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("*** "):
            # Next file op — stop consuming.
            break
        if not stripped:
            if current_hunk is not None:
                # Treat as a context line (blank).
                current_hunk.old_lines.append("")
                current_hunk.new_lines.append("")
            i += 1
            continue
        if stripped == EMPTY_CHANGE_CONTEXT:
            current_hunk = Hunk(change_context=None)
            hunks.append(current_hunk)
            i += 1
            continue
        if stripped.startswith(CHANGE_CONTEXT):
            current_hunk = Hunk(change_context=stripped[len(CHANGE_CONTEXT):])
            hunks.append(current_hunk)
            i += 1
            continue
        if stripped == END_OF_FILE:
            if current_hunk is None:
                raise PatchParseError("'*** End of File' marker outside any hunk")
            current_hunk.is_end_of_file = True
            i += 1
            continue
        if current_hunk is None:
            # Allow chunks without an explicit @@ header (codex's lenient mode).
            current_hunk = Hunk(change_context=None)
            hunks.append(current_hunk)
        prefix = line[:1]
        body = line[1:]
        if prefix == " ":
            current_hunk.old_lines.append(body)
            current_hunk.new_lines.append(body)
        elif prefix == "+":
            current_hunk.new_lines.append(body)
        elif prefix == "-":
            current_hunk.old_lines.append(body)
        else:
            raise PatchParseError(
                f"unexpected hunk line {line!r}; "
                f"every line must start with ' ', '+', '-', or '@@'"
            )
        i += 1
    return hunks, i


def _validate_relative_path(raw: str) -> str:
    """Reject absolute / `..`-escape paths early in the envelope parser.

    Path safety is *also* enforced by ``WorkspaceManager.canonicalize`` once
    handler code resolves the path against cwd, but rejecting at parse time
    gives the model a clearer error message.
    """
    if not raw:
        raise PatchParseError("empty path in patch header")
    pp = PurePosixPath(raw)
    if pp.is_absolute() or raw.startswith("/") or (len(raw) >= 2 and raw[1] == ":"):
        raise PatchParseError(f"absolute paths are not allowed in patches: {raw}")
    if any(part == ".." for part in pp.parts):
        raise PatchParseError(f"parent-directory escape not allowed: {raw}")
    return raw
