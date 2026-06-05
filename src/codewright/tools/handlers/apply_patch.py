from __future__ import annotations

import os
import uuid
from pathlib import Path

from pydantic import Field

from codewright.protocol import PendingAction, ReviewDecision
from codewright.tools.errors import FatalToolError, RespondToModelError
from codewright.tools.handler import ToolHandler
from codewright.tools.handlers._patch_matcher import find_hunk
from codewright.tools.handlers._patch_parser import FileOp, Hunk, PatchParseError, parse_patch
from codewright.tools.invocation import ToolInvocation
from codewright.tools.result import ToolResult
from codewright.tools.spec import ParameterModel, ToolSpec


class ApplyPatchParams(ParameterModel):

    patch: str = Field(
        ...,
        description=(
            "Envelope text. Begins with '*** Begin Patch' and ends with "
            "'*** End Patch'. Inside: '*** Add File: <path>' (each new line "
            "prefixed with '+'); '*** Delete File: <path>'; '*** Update File: "
            "<path>' optionally followed by '*** Move to: <new path>' and one "
            "or more '@@ <header>' hunks. Within a hunk: ' line' = context, "
            "'-line' = remove, '+line' = add."
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
        affected = [workspace.canonicalize(op.path) for op in ops]
        move_targets = [
            workspace.canonicalize(op.move_to) if op.move_to else None for op in ops
        ]

        action = PendingAction(
            action_id=uuid.uuid4().hex,
            kind="patch",
            summary=_summarize(ops),
            details={
                "paths": [str(p) for p in affected],
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


        pending_writes: list[tuple[Path, str]] = []
        pending_deletes: list[Path] = []
        pending_renames: list[tuple[Path, Path]] = []
        for op, abs_path, move_target in zip(ops, affected, move_targets, strict=True):
            if op.kind == "add":
                if abs_path.exists():
                    raise RespondToModelError(
                        f"apply_patch: '{op.path}' already exists; use Update"
                    )
                pending_writes.append((abs_path, op.contents or ""))
            elif op.kind == "delete":
                if not abs_path.exists():
                    raise RespondToModelError(
                        f"apply_patch: cannot delete missing file '{op.path}'"
                    )
                pending_deletes.append(abs_path)
            elif op.kind == "update":
                if not abs_path.exists():
                    raise RespondToModelError(
                        f"apply_patch: cannot update missing file '{op.path}'"
                    )
                original = abs_path.read_text(encoding="utf-8")
                new_body = _apply_hunks(op, original)
                if move_target is not None and move_target != abs_path:
                    pending_renames.append((abs_path, move_target))
                    pending_writes.append((move_target, new_body))

                    pending_deletes.append(abs_path)
                else:
                    pending_writes.append((abs_path, new_body))


        for path, body in pending_writes:
            _atomic_write(path, body)
        for src, dst in pending_renames:

            if src != dst and src.exists() and dst.exists():

                pass
        for path in pending_deletes:
            if path.exists():
                path.unlink()

        workspace.audit(
            {
                "tool": "apply_patch",
                "call_id": invocation.call_id,
                "ops": [
                    {"kind": o.kind, "path": str(p), "move_to": str(m) if m else None}
                    for o, p, m in zip(ops, affected, move_targets, strict=True)
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


def _apply_hunks(op: FileOp, original: str) -> str:

    has_trailing_newline = original.endswith("\n") or original == ""
    lines = original.split("\n")
    if has_trailing_newline and lines and lines[-1] == "":
        lines.pop()


    anchors: list[str] = []
    cursor = 0
    for hunk in op.hunks:
        if hunk.change_context is not None:
            anchors.append(hunk.change_context)
        idx = find_hunk(lines, anchors, hunk.old_lines, hunk.is_end_of_file)
        if idx is None or idx < cursor:
            raise RespondToModelError(
                _hunk_failure_message(op, hunk, lines, cursor)
            )

        lines[idx : idx + len(hunk.old_lines)] = hunk.new_lines
        cursor = idx + len(hunk.new_lines)
    rebuilt = "\n".join(lines)
    if has_trailing_newline and rebuilt and not rebuilt.endswith("\n"):
        rebuilt += "\n"
    elif rebuilt == "" and has_trailing_newline and original != "":
        rebuilt = "\n"
    return rebuilt


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


def _atomic_write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".cw.tmp.{os.getpid()}")
    try:
        tmp.write_text(body, encoding="utf-8", newline="")
        tmp.replace(path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
