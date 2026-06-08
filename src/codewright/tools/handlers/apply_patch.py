from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field

from codewright.protocol import PendingAction, ReviewDecision
from codewright.tools.errors import FatalToolError, RespondToModelError
from codewright.tools.handler import ToolHandler
from codewright.tools.handlers._patch_matcher import seek_sequence
from codewright.tools.handlers._patch_parser import FileOp, Hunk, PatchParseError, parse_patch
from codewright.tools.invocation import ToolInvocation
from codewright.tools.result import ToolResult
from codewright.tools.spec import ParameterModel, ToolSpec


@dataclass(frozen=True)
class _ResolvedOp:
    op: FileOp
    path: Path
    move_to: Path | None


@dataclass
class _VirtualFile:
    original_exists: bool
    exists: bool
    content: str | None
    content_loaded: bool
    label: str


@dataclass(frozen=True)
class _PlannedWrite:
    path: Path
    body: str
    label: str


@dataclass(frozen=True)
class _PlannedDelete:
    path: Path
    label: str


@dataclass(frozen=True)
class _PatchPlan:
    writes: list[_PlannedWrite]
    deletes: list[_PlannedDelete]


@dataclass(frozen=True)
class _Replacement:
    sequence: int
    start: int
    old_len: int
    new_lines: list[str]


class ApplyPatchParams(ParameterModel):

    patch: str = Field(
        ...,
        description=(
            "Structured patch envelope for workspace-relative file edits. The "
            "text must start with '*** Begin Patch' and end with '*** End "
            "Patch'. Supported file ops are: '*** Add File: <path>' followed "
            "by new file lines prefixed with '+'; '*** Delete File: <path>'; "
            "and '*** Update File: <path>' optionally followed by '*** Move "
            "to: <new path>'. Paths must be relative and must not contain '..' "
            "or drive/absolute prefixes. Add requires a missing path; "
            "Delete/Update require an existing path; Move writes the updated "
            "contents to the destination and deletes the source, and the "
            "destination must not exist unless an earlier op deletes it. Each "
            "Update needs one or more hunks. Start a hunk with '@@' or '@@ "
            "<anchor>'; anchors and hunks are matched in file order from the "
            "previous hunk, so do not jump backward to an earlier anchor. "
            "Within a hunk, prefix context lines with one space, removed lines "
            "with '-', and added lines with '+'. Use '*** End of File' inside "
            "a hunk to require a match at EOF. Do not create overlapping "
            "hunks; combine conflicting edits into one hunk."
        ),
    )


class ApplyPatchHandler(ToolHandler):
    @property
    def tool_name(self) -> str:
        return "apply_patch"

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_name,
            description=(
                "Edit files via a structured patch envelope. Use this instead "
                "of shell utilities like sed/awk/cat. Operates on files "
                "relative to the workspace root only."
            ),
            parameters=ApplyPatchParams.to_json_schema(),
            supports_parallel=False,
            requires_approval=True,
        )

    async def handle(self, invocation: ToolInvocation) -> ToolResult:
        params = _coerce_params(invocation.arguments)
        try:
            ops = parse_patch(params.patch)
        except PatchParseError as exc:
            raise RespondToModelError(f"apply_patch: {exc}") from exc
        if not ops:
            raise RespondToModelError("apply_patch: patch contained no file ops")

        workspace = invocation.session.workspace
        resolved_ops = [
            _ResolvedOp(
                op=op,
                path=workspace.canonicalize(op.path),
                move_to=workspace.canonicalize(op.move_to) if op.move_to else None,
            )
            for op in ops
        ]
        affected_paths = _affected_paths(resolved_ops)

        action = PendingAction(
            action_id=uuid.uuid4().hex,
            kind="patch",
            summary=_summarize(ops),
            details={
                "paths": [str(p) for p in affected_paths],
                "workspace_root": str(workspace.root),
                "cwd": str(invocation.turn_context.cwd),
            },
        )
        decision = await workspace.check_action(action, invocation.session)
        if decision == ReviewDecision.DENIED:
            raise RespondToModelError(
                f"apply_patch denied by user/policy: {action.summary}"
            )
        if decision == ReviewDecision.ABORT:
            raise FatalToolError("user aborted apply_patch")


        plan = _build_plan(resolved_ops)
        _commit_plan(plan)

        workspace.audit(
            {
                "tool": "apply_patch",
                "call_id": invocation.call_id,
                "ops": [
                    {
                        "kind": resolved.op.kind,
                        "path": str(resolved.path),
                        "move_to": str(resolved.move_to) if resolved.move_to else None,
                    }
                    for resolved in resolved_ops
                ],
            }
        )
        body = "Applied " + ", ".join(_describe(op) for op in ops)
        return ToolResult(success=True, body=body)




def _coerce_params(arguments: dict) -> ApplyPatchParams:
    try:
        return ApplyPatchParams(**arguments)
    except Exception as exc:  # pydantic.ValidationError, etc.
        raise RespondToModelError(f"apply_patch: invalid arguments: {exc}") from exc


def _summarize(ops: list[FileOp]) -> str:
    parts: list[str] = []
    for op in ops[:3]:
        parts.append(_describe(op))
    if len(ops) > 3:
        parts.append(f"… (+{len(ops) - 3} more)")
    return "; ".join(parts)


def _describe(op: FileOp) -> str:
    if op.kind == "add":
        return f"add {op.path}"
    if op.kind == "delete":
        return f"delete {op.path}"
    if op.move_to:
        return f"update {op.path} -> {op.move_to}"
    return f"update {op.path}"


def _affected_paths(resolved_ops: list[_ResolvedOp]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for resolved in resolved_ops:
        for path in (resolved.path, resolved.move_to):
            if path is not None and path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def _build_plan(resolved_ops: list[_ResolvedOp]) -> _PatchPlan:
    states: dict[Path, _VirtualFile] = {}

    def state_for(path: Path, label: str) -> _VirtualFile:
        state = states.get(path)
        if state is None:
            exists = _path_exists(path, label)
            state = _VirtualFile(
                original_exists=exists,
                exists=exists,
                content=None,
                content_loaded=False,
                label=label,
            )
            states[path] = state
        return state

    def require_existing(path: Path, label: str, action: str) -> _VirtualFile:
        state = state_for(path, label)
        if not state.exists:
            raise RespondToModelError(
                f"apply_patch: cannot {action} missing file '{label}'"
            )
        return state

    def content_for(path: Path, label: str) -> str:
        state = require_existing(path, label, "update")
        if not state.content_loaded:
            state.content = _read_utf8(path, label)
            state.content_loaded = True
        return state.content or ""

    def set_present(path: Path, body: str, label: str) -> None:
        state = state_for(path, label)
        state.exists = True
        state.content = body
        state.content_loaded = True
        state.label = label

    def set_deleted(path: Path, label: str) -> None:
        state = state_for(path, label)
        state.exists = False
        state.content = None
        state.content_loaded = False
        state.label = label

    for resolved in resolved_ops:
        op = resolved.op
        if op.kind == "add":
            state = state_for(resolved.path, op.path)
            if state.exists:
                raise RespondToModelError(
                    f"apply_patch: '{op.path}' already exists; use Update"
                )
            set_present(resolved.path, op.contents or "", op.path)
        elif op.kind == "delete":
            require_existing(resolved.path, op.path, "delete")
            set_deleted(resolved.path, op.path)
        elif op.kind == "update":
            original = content_for(resolved.path, op.path)
            new_body = _apply_hunks(op, original)
            if resolved.move_to is not None and resolved.move_to != resolved.path:
                destination_label = op.move_to or op.path
                destination = state_for(resolved.move_to, destination_label)
                if destination.exists:
                    raise RespondToModelError(
                        "apply_patch: cannot move "
                        f"'{op.path}' to '{destination_label}': destination already exists"
                    )
                set_present(resolved.move_to, new_body, destination_label)
                set_deleted(resolved.path, op.path)
            else:
                set_present(resolved.path, new_body, op.path)

    writes: list[_PlannedWrite] = []
    deletes: list[_PlannedDelete] = []
    for path, state in states.items():
        if state.exists:
            if state.content_loaded:
                writes.append(_PlannedWrite(path=path, body=state.content or "", label=state.label))
        elif state.original_exists:
            deletes.append(_PlannedDelete(path=path, label=state.label))
    return _PatchPlan(writes=writes, deletes=deletes)


def _commit_plan(plan: _PatchPlan) -> None:
    for write in plan.writes:
        _atomic_write(write.path, write.body, write.label)
    for delete in plan.deletes:
        _delete_file(delete.path, delete.label)


def _apply_hunks(op: FileOp, original: str) -> str:

    has_trailing_newline = original.endswith("\n") or original == ""
    lines = original.split("\n")
    if has_trailing_newline and lines and lines[-1] == "":
        lines.pop()


    replacements: list[_Replacement] = []
    cursor = 0
    for sequence, hunk in enumerate(op.hunks):
        replacement = _compute_replacement(op, hunk, lines, cursor, sequence)
        replacements.append(replacement)
        if replacement.old_len > 0:
            cursor = replacement.start + replacement.old_len

    _validate_replacements(op, replacements)
    lines = _apply_replacements(lines, replacements)
    rebuilt = "\n".join(lines)
    if has_trailing_newline and rebuilt and not rebuilt.endswith("\n"):
        rebuilt += "\n"
    elif rebuilt == "" and has_trailing_newline and original != "":
        rebuilt = "\n"
    return rebuilt


def _compute_replacement(
    op: FileOp,
    hunk: Hunk,
    lines: list[str],
    cursor: int,
    sequence: int,
) -> _Replacement:
    pattern = hunk.old_lines
    new_lines = hunk.new_lines
    start = _locate_hunk_start(hunk, lines, cursor, pattern)
    if start is not None:
        return _Replacement(
            sequence=sequence,
            start=start,
            old_len=len(pattern),
            new_lines=list(new_lines),
        )

    if pattern and pattern[-1] == "":
        trimmed_pattern = pattern[:-1]
        trimmed_new_lines = new_lines[:-1] if new_lines and new_lines[-1] == "" else new_lines
        start = _locate_hunk_start(hunk, lines, cursor, trimmed_pattern)
        if start is not None:
            return _Replacement(
                sequence=sequence,
                start=start,
                old_len=len(trimmed_pattern),
                new_lines=list(trimmed_new_lines),
            )

    raise RespondToModelError(_hunk_failure_message(op, hunk, lines, cursor))


def _locate_hunk_start(
    hunk: Hunk,
    lines: list[str],
    cursor: int,
    pattern: list[str],
) -> int | None:
    if not pattern:
        if hunk.is_end_of_file:
            return len(lines)
        if hunk.change_context is None:
            return cursor
        context_index = _locate_context(hunk.change_context, lines, cursor)
        if context_index is None:
            return None
        return max(cursor, context_index + 1)

    if hunk.change_context is None:
        return seek_sequence(lines, pattern, cursor, hunk.is_end_of_file)

    context_index = _locate_context(hunk.change_context, lines, cursor)
    if context_index is None:
        return None
    return seek_sequence(
        lines,
        pattern,
        max(cursor, context_index + 1),
        hunk.is_end_of_file,
    )


def _locate_context(context: str, lines: list[str], cursor: int) -> int | None:
    return _find_context_line(context, lines, cursor)


def _find_context_line(context: str, lines: list[str], start: int) -> int | None:
    needle = context.strip()
    if not needle:
        return start
    for i in range(start, len(lines)):
        if needle in lines[i]:
            return i
    return seek_sequence(lines, [context], start)


def _validate_replacements(op: FileOp, replacements: list[_Replacement]) -> None:
    non_empty = [replacement for replacement in replacements if replacement.old_len > 0]
    consumed_until = 0
    for replacement in sorted(non_empty, key=lambda item: (item.start, item.sequence)):
        if replacement.start < consumed_until:
            raise RespondToModelError(
                f"apply_patch: overlapping hunks in '{op.path}' near line "
                f"{replacement.start + 1}"
            )
        consumed_until = replacement.start + replacement.old_len

    for insertion in (replacement for replacement in replacements if replacement.old_len == 0):
        for replacement in non_empty:
            replacement_end = replacement.start + replacement.old_len
            if replacement.start < insertion.start < replacement_end:
                raise RespondToModelError(
                    f"apply_patch: overlapping hunks in '{op.path}' near line "
                    f"{insertion.start + 1}"
                )


def _apply_replacements(
    lines: list[str],
    replacements: list[_Replacement],
) -> list[str]:
    updated = list(lines)
    ordered = sorted(replacements, key=lambda item: (item.start, item.sequence))
    for replacement in reversed(ordered):
        updated[
            replacement.start : replacement.start + replacement.old_len
        ] = replacement.new_lines
    return updated


def _hunk_failure_message(
    op: FileOp, hunk: Hunk, lines: list[str], cursor: int
) -> str:
    near: list[str] = []
    for i in range(max(0, cursor - 2), min(len(lines), cursor + 5)):
        near.append(f"  {i + 1:4d}| {lines[i]}")
    near_block = "\n".join(near) if near else "(file is shorter than expected)"
    return (
        f"apply_patch: failed to locate hunk in '{op.path}'. "
        f"Anchor='{hunk.change_context or '@@'}'. "
        f"Searched from line {cursor + 1}. Nearby file contents:\n{near_block}"
    )


def _path_exists(path: Path, label: str) -> bool:
    try:
        return path.exists()
    except OSError as exc:
        raise RespondToModelError(
            f"apply_patch: cannot inspect '{label}': {type(exc).__name__}: {exc}"
        ) from exc


def _read_utf8(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise RespondToModelError(
            f"apply_patch: cannot read '{label}': file is not valid UTF-8"
        ) from exc
    except OSError as exc:
        raise RespondToModelError(
            f"apply_patch: cannot read '{label}': {type(exc).__name__}: {exc}"
        ) from exc


def _atomic_write(path: Path, body: str, label: str) -> None:
    tmp = path.with_suffix(path.suffix + f".cw.tmp.{os.getpid()}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(body, encoding="utf-8", newline="")
        tmp.replace(path)
    except OSError as exc:
        raise RespondToModelError(
            f"apply_patch: cannot write '{label}': {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def _delete_file(path: Path, label: str) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError as exc:
        raise RespondToModelError(
            f"apply_patch: cannot delete '{label}': {type(exc).__name__}: {exc}"
        ) from exc
