"""AGENTS.md upward walk tests."""

from __future__ import annotations

from pathlib import Path

from codewright.protocol import PermissionProfile
from codewright.workspace import WorkspaceManager


def _wm(root: Path) -> WorkspaceManager:
    return WorkspaceManager(root, PermissionProfile.WORKSPACE_WRITE)


def test_finds_agents_md_at_root(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("hello root", encoding="utf-8")
    assert _wm(tmp_path).resolve_agents_md(tmp_path) == "hello root"


def test_finds_dot_agents_md_when_no_canonical(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / ".agents.md").write_text("sub doc", encoding="utf-8")
    deep = sub / "deep"
    deep.mkdir()
    wm = _wm(tmp_path)
    assert wm.resolve_agents_md(deep) == "sub doc"


def test_prefers_closer_match(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("ROOT", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "AGENTS.md").write_text("SUB", encoding="utf-8")
    assert _wm(tmp_path).resolve_agents_md(sub) == "SUB"


def test_returns_none_when_none_found(tmp_path: Path) -> None:
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    assert _wm(tmp_path).resolve_agents_md(deep) is None


def test_refuses_to_walk_above_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "AGENTS.md"
    outside.write_text("must not see", encoding="utf-8")
    deep = workspace / "deep"
    deep.mkdir()
    assert _wm(workspace).resolve_agents_md(deep) is None


def test_cwd_outside_root_returns_none(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "AGENTS.md").write_text("not mine", encoding="utf-8")
    assert _wm(workspace).resolve_agents_md(elsewhere) is None


def test_default_cwd_is_root(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("doc", encoding="utf-8")
    assert _wm(tmp_path).resolve_agents_md() == "doc"
