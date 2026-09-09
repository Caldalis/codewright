"""Semantic read/search tools."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from codewright.agent.cancellation import CancellationToken
from codewright.agent.session import Session
from codewright.agent.turn_context import TurnContext
from codewright.cli import _register_builtin_tools
from codewright.protocol import AskForApproval, PermissionProfile
from codewright.tools.errors import RespondToModelError
from codewright.tools.handlers.find_files import FindFilesHandler
from codewright.tools.handlers.list_dir import ListDirHandler
from codewright.tools.handlers.read_file import ReadFileHandler
from codewright.tools.handlers.search_text import SearchTextHandler
from codewright.tools.invocation import ToolInvocation
from codewright.workspace import WorkspaceManager


async def _session(tmp_path: Path) -> Session:
    sess = Session(
        session_id=uuid.uuid4().hex,
        cwd=tmp_path,
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
        workspace=WorkspaceManager(tmp_path, PermissionProfile.WORKSPACE_WRITE),
    )
    await sess.next_event()
    return sess


def _inv(sess: Session, tool_name: str, args: dict, cwd: Path) -> ToolInvocation:
    return ToolInvocation(
        session=sess,
        turn_context=TurnContext(
            turn_id="t",
            cwd=cwd,
            model="m",
            permission_profile=PermissionProfile.WORKSPACE_WRITE,
            approval_policy=AskForApproval.NEVER,
            cancellation_token=CancellationToken(),
        ),
        call_id=uuid.uuid4().hex,
        tool_name=tool_name,
        arguments=args,
        cancellation_token=CancellationToken(),
    )


@pytest.mark.asyncio
async def test_read_file_offset_limit_and_next_offset(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    sess = await _session(tmp_path)
    try:
        result = await ReadFileHandler().handle(
            _inv(
                sess,
                "read_file",
                {"path": "notes.txt", "offset": 1, "limit": 2},
                tmp_path,
            )
        )
        assert result.success is True
        assert "| two" in result.body
        assert "| three" in result.body
        assert "| one" not in result.body
        assert "Next offset: 3" in result.body
        assert result.structured_data["next_offset"] == 3
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_read_file_blocks_workspace_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"outside-{uuid.uuid4().hex}.txt"
    outside.write_text("secret", encoding="utf-8")
    sess = await _session(tmp_path)
    try:
        with pytest.raises(RespondToModelError, match="path escapes workspace root"):
            await ReadFileHandler().handle(
                _inv(sess, "read_file", {"path": str(outside)}, tmp_path)
            )
    finally:
        outside.unlink(missing_ok=True)
        await sess.shutdown()


@pytest.mark.asyncio
async def test_list_dir_is_non_recursive_and_dirs_first(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "src" / "nested.py").write_text("pass\n", encoding="utf-8")
    sess = await _session(tmp_path)
    try:
        result = await ListDirHandler().handle(_inv(sess, "list_dir", {"path": "."}, tmp_path))
        entries = result.structured_data["entries"]
        assert [entry["name"] for entry in entries] == ["src", "a.txt", "b.txt"]
        assert "nested.py" not in result.body
        assert "[dir] src/" in result.body
        assert result.structured_data["limit_reached"] is False
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_list_dir_limit_caps_entries(tmp_path: Path) -> None:
    for i in range(20):
        (tmp_path / f"f{i:02d}.txt").write_text("x", encoding="utf-8")
    sess = await _session(tmp_path)
    try:
        result = await ListDirHandler().handle(
            _inv(sess, "list_dir", {"path": ".", "limit": 5}, tmp_path)
        )
        entries = result.structured_data["entries"]
        assert len(entries) == 5  # structured_data is bounded, not just the body
        assert result.structured_data["limit_reached"] is True
        assert "(limit reached)" in result.body
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_list_dir_reports_symlink_without_following(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"outside-{uuid.uuid4().hex}.txt"
    outside.write_text("secret-size-should-not-leak", encoding="utf-8")
    link = tmp_path / "outside_link"
    try:
        link.symlink_to(outside)
    except (NotImplementedError, OSError) as exc:
        outside.unlink(missing_ok=True)
        pytest.skip(f"symlink creation unavailable: {exc}")

    sess = await _session(tmp_path)
    try:
        result = await ListDirHandler().handle(_inv(sess, "list_dir", {"path": "."}, tmp_path))
        entry = next(item for item in result.structured_data["entries"] if item["name"] == "outside_link")
        assert entry["kind"] == "symlink"
        assert entry["size"] is None
        assert "[symlink] outside_link" in result.body
        assert "secret-size-should-not-leak" not in result.body
    finally:
        outside.unlink(missing_ok=True)
        await sess.shutdown()


@pytest.mark.asyncio
async def test_find_files_supports_query_glob_and_skips_generated_dirs(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".codewright").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('app')\n", encoding="utf-8")
    (tmp_path / "tests" / "test_app.py").write_text("def test_app(): pass\n", encoding="utf-8")
    (tmp_path / ".venv" / "app.py").write_text("ignored\n", encoding="utf-8")
    (tmp_path / ".codewright" / "app.py").write_text("ignored\n", encoding="utf-8")
    sess = await _session(tmp_path)
    try:
        result = await FindFilesHandler().handle(
            _inv(sess, "find_files", {"query": "app", "glob": "*.py"}, tmp_path)
        )
        matches = result.structured_data["matches"]
        assert matches == ["src/app.py", "tests/test_app.py"]
        assert not any(".venv" in match for match in matches)
        assert not any(".codewright" in match for match in matches)
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_explicit_codewright_access_is_rejected(tmp_path: Path) -> None:
    private = tmp_path / ".codewright"
    private.mkdir()
    (private / "audit.jsonl").write_text("alpha\n", encoding="utf-8")
    sess = await _session(tmp_path)
    try:
        with pytest.raises(RespondToModelError, match="agent-private path"):
            await ReadFileHandler().handle(
                _inv(sess, "read_file", {"path": ".codewright/audit.jsonl"}, tmp_path)
            )
        with pytest.raises(RespondToModelError, match="agent-private path"):
            await ListDirHandler().handle(
                _inv(sess, "list_dir", {"path": ".codewright"}, tmp_path)
            )
        with pytest.raises(RespondToModelError, match="agent-private path"):
            await SearchTextHandler().handle(
                _inv(sess, "search_text", {"pattern": "alpha", "path": ".codewright"}, tmp_path)
            )
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_search_text_supports_path_glob_and_limit(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text(
        "alpha = 1\nbeta = alpha + 1\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "notes.txt").write_text("alpha text\n", encoding="utf-8")
    (tmp_path / "src" / "bad.bin").write_bytes(b"\0alpha")
    sess = await _session(tmp_path)
    try:
        result = await SearchTextHandler().handle(
            _inv(
                sess,
                "search_text",
                {"pattern": "alpha", "path": "src", "glob": "*.py", "limit": 1},
                tmp_path,
            )
        )
        matches = result.structured_data["matches"]
        assert result.structured_data["limit_reached"] is True
        assert len(matches) == 1
        assert matches[0]["path"] == "src/a.py"
        assert matches[0]["line"] == 1
        assert "src/notes.txt" not in result.body
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_search_text_defaults_to_literal_search(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("alpha(value)\n", encoding="utf-8")
    sess = await _session(tmp_path)
    try:
        result = await SearchTextHandler().handle(
            _inv(sess, "search_text", {"pattern": "(", "path": "."}, tmp_path)
        )
        assert result.success is True
        assert result.structured_data["matches"][0]["path"] == "a.py"
        assert result.structured_data["matches"][0]["column"] == 6
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_search_text_reports_invalid_regex_when_enabled(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("alpha\n", encoding="utf-8")
    sess = await _session(tmp_path)
    try:
        with pytest.raises(RespondToModelError, match="invalid regex pattern"):
            await SearchTextHandler().handle(
                _inv(sess, "search_text", {"pattern": "(", "path": ".", "regex": True}, tmp_path)
            )
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_search_text_regex_flags_overlong_lines(tmp_path: Path) -> None:
    # Finding B: regex mode only searches the first N chars of a line; a match
    # past that is missed, and the result must say so. Literal mode is unaffected.
    from codewright.tools.handlers.search_text import _MAX_REGEX_LINE_CHARS

    needle = "NEEDLEZ"
    long_line = ("a" * (_MAX_REGEX_LINE_CHARS + 50)) + needle + "\n"
    (tmp_path / "min.js").write_text(long_line, encoding="utf-8")
    sess = await _session(tmp_path)
    try:
        # regex mode: the needle is past the window -> missed + flagged.
        rx = await SearchTextHandler().handle(
            _inv(sess, "search_text", {"pattern": needle, "path": ".", "regex": True}, tmp_path)
        )
        assert rx.structured_data["matches"] == []
        assert rx.structured_data["regex_truncated_lines"] == 1
        assert "regex window" in rx.body

        # literal mode: no per-line cap -> the needle is found, no note.
        lit = await SearchTextHandler().handle(
            _inv(sess, "search_text", {"pattern": needle, "path": "."}, tmp_path)
        )
        assert len(lit.structured_data["matches"]) == 1
        assert lit.structured_data["regex_truncated_lines"] == 0
        assert "regex window" not in lit.body
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_builtin_registration_includes_semantic_tools(tmp_path: Path) -> None:
    sess = await _session(tmp_path)
    try:
        await _register_builtin_tools(sess)
        names = set(sess.tool_registry.names())
        assert {"read_file", "list_dir", "find_files", "search_text"} <= names
    finally:
        await sess.shutdown()
