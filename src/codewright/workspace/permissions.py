from __future__ import annotations

from pathlib import Path
from typing import Literal

from codewright.protocol import PendingAction, PermissionProfile

Verdict = Literal["auto_allow", "auto_deny", "ask"]


def action_signature(action: PendingAction) -> str:

    details = action.details or {}
    if action.kind == "exec":
        if details.get("flagged"):
            return f"exec:{action.summary}"
        segments = details.get("segments")
        if isinstance(segments, list) and segments:
            heads = ",".join(sorted({str(s) for s in segments}))
            return f"exec:{heads}"
        command = details.get("command")
        if isinstance(command, list) and command:
            head = str(command[0])
            return f"exec:{head}"
        return f"exec:{action.summary}"
    if action.kind == "patch":
        return f"patch:{details.get('workspace_root') or details.get('cwd') or ''}"
    return f"{action.kind}:{action.summary}"


def assess_action(
    profile: PermissionProfile,
    action: PendingAction,
    session_approvals: set[str],
    cwd: Path,
    workspace_root: Path,
) -> Verdict:

    if action_signature(action) in session_approvals:
        return "auto_allow"

    if profile == PermissionProfile.READ_ONLY:

        return "auto_deny"

    if profile == PermissionProfile.DANGEROUS:
        return "ask"


    if action.kind == "patch":
        paths = action.details.get("paths") or []
        if not paths:
            return "ask"
        for raw in paths:
            try:
                p = Path(raw).resolve()
            except OSError:
                return "ask"
            if not _is_within(p, workspace_root):
                return "ask"
        return "auto_allow"

    if action.kind == "exec":
        # Segment analysis
        if action.details.get("flagged"):
            return "ask"
        target_cwd = action.details.get("cwd")
        try:
            resolved = Path(target_cwd).resolve() if target_cwd else cwd
        except OSError:
            return "ask"
        if _is_within(resolved, workspace_root):
            return "auto_allow"
        # Outside the workspace always asks, even for read-only commands: the
        # workspace root is a hard boundary (system.md), and a persistent shell
        # session can `cd` out of it.
        return "ask"

    return "ask"


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False
