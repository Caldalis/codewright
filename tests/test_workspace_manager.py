"""Workspace canonicalize / AGENTS.md / audit tests."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from codewright.protocol import PermissionProfile
from codewright.tools.errors import RespondToModelError
from codewright.workspace import WorkspaceManager


def test_canonicalize_resolves_relative(tmp_path: Path) -> None:
    wm = WorkspaceManager(tmp_path, PermissionProfile.WORKSPACE_WRITE)
    p = wm.canonicalize("foo/bar.txt")
    assert p == (tmp_path / "foo" / "bar.txt").resolve()


def test_canonicalize_rejects_absolute_escape(tmp_path: Path) -> None:
    other = tmp_path.parent / "outside"
    other.mkdir(exist_ok=True)
    wm = WorkspaceManager(tmp_path, PermissionProfile.WORKSPACE_WRITE)
    with pytest.raises(RespondToModelError):
        wm.canonicalize(other / "x.txt")


def test_canonicalize_rejects_parent_escape(tmp_path: Path) -> None:
    wm = WorkspaceManager(tmp_path, PermissionProfile.WORKSPACE_WRITE)
    with pytest.raises(RespondToModelError):
        wm.canonicalize("../etc/passwd")


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks require privileges on Windows")
def test_canonicalize_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_target"
    outside.mkdir(exist_ok=True)
    secret = outside / "secret.txt"
    secret.write_text("nope", encoding="utf-8")
    inside_link = tmp_path / "escape"
    os.symlink(outside, inside_link)
    wm = WorkspaceManager(tmp_path, PermissionProfile.WORKSPACE_WRITE)
    with pytest.raises(RespondToModelError):
        wm.canonicalize("escape/secret.txt")


def test_audit_appends_jsonl(tmp_path: Path) -> None:
    wm = WorkspaceManager(tmp_path, PermissionProfile.WORKSPACE_WRITE)
    wm.audit({"tool": "run_shell", "exit_code": 0})
    wm.audit({"tool": "apply_patch", "ops": [{"kind": "add", "path": "x"}]})
    path = tmp_path / ".codewright" / "audit.jsonl"
    assert path.exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["tool"] == "run_shell"
    assert "ts" in first


def test_audit_failure_is_swallowed(tmp_path: Path, monkeypatch) -> None:
    wm = WorkspaceManager(tmp_path, PermissionProfile.WORKSPACE_WRITE)

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "mkdir", boom)
    wm.audit({"tool": "test"})  # must not raise
