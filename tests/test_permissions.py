"""Pure permission decision table tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from codewright.protocol import PendingAction, PermissionProfile
from codewright.workspace.permissions import action_signature, assess_action


def _exec(cwd: str, command: list[str] | None = None) -> PendingAction:
    return PendingAction(
        action_id="x",
        kind="exec",
        summary=" ".join(command or ["ls"]),
        details={"command": command or ["ls"], "cwd": cwd},
    )


def _patch(paths: list[str]) -> PendingAction:
    return PendingAction(
        action_id="x",
        kind="patch",
        summary="apply patch",
        details={"paths": paths},
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


def test_read_only_denies_all_exec(workspace: Path) -> None:
    action = _exec(str(workspace))
    verdict = assess_action(
        PermissionProfile.READ_ONLY, action, set(), workspace, workspace
    )
    assert verdict == "auto_deny"


def test_read_only_denies_all_patches(workspace: Path) -> None:
    action = _patch([str(workspace / "x.txt")])
    verdict = assess_action(
        PermissionProfile.READ_ONLY, action, set(), workspace, workspace
    )
    assert verdict == "auto_deny"


def test_dangerous_always_asks(workspace: Path) -> None:
    for action in (_exec(str(workspace)), _patch([str(workspace / "a.txt")])):
        verdict = assess_action(
            PermissionProfile.DANGEROUS, action, set(), workspace, workspace
        )
        assert verdict == "ask"


def test_workspace_write_inside_auto_allows_exec(workspace: Path) -> None:
    verdict = assess_action(
        PermissionProfile.WORKSPACE_WRITE,
        _exec(str(workspace)),
        set(),
        workspace,
        workspace,
    )
    assert verdict == "auto_allow"


def test_workspace_write_outside_asks_exec(workspace: Path, tmp_path_factory) -> None:
    other = tmp_path_factory.mktemp("elsewhere")
    verdict = assess_action(
        PermissionProfile.WORKSPACE_WRITE,
        _exec(str(other)),
        set(),
        workspace,
        workspace,
    )
    assert verdict == "ask"


def test_workspace_write_in_workspace_patch(workspace: Path) -> None:
    verdict = assess_action(
        PermissionProfile.WORKSPACE_WRITE,
        _patch([str(workspace / "x.txt"), str(workspace / "y.txt")]),
        set(),
        workspace,
        workspace,
    )
    assert verdict == "auto_allow"


def test_workspace_write_escape_patch_asks(workspace: Path, tmp_path_factory) -> None:
    other = tmp_path_factory.mktemp("else")
    verdict = assess_action(
        PermissionProfile.WORKSPACE_WRITE,
        _patch([str(workspace / "a.txt"), str(other / "b.txt")]),
        set(),
        workspace,
        workspace,
    )
    assert verdict == "ask"


def test_session_approval_short_circuits(workspace: Path) -> None:
    action = _exec(str(workspace), ["git", "commit"])
    approvals = {action_signature(action)}
    verdict = assess_action(
        PermissionProfile.DANGEROUS, action, approvals, workspace, workspace
    )
    assert verdict == "auto_allow"


def test_signatures_are_kind_aware(workspace: Path) -> None:
    a = _exec(str(workspace), ["git", "commit"])
    b = _exec(str(workspace), ["ls", "-la"])
    assert action_signature(a) != action_signature(b)
    assert action_signature(a).startswith("exec:git")
