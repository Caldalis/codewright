"""apply_patch: parser, 4-level fallback, multi-anchor, path safety, end-to-end."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

from codewright.agent.cancellation import CancellationToken
from codewright.agent.turn_context import TurnContext
from codewright.protocol import AskForApproval, PermissionProfile, ReviewDecision
from codewright.tools.errors import RespondToModelError
from codewright.tools.executor import ToolExecutor
from codewright.tools.handlers._patch_matcher import find_hunk, unicode_normalize
from codewright.tools.handlers._patch_parser import PatchParseError, parse_patch
from codewright.tools.handlers.apply_patch import (
    ApplyPatchHandler,
    _Replacement,
    _validate_replacements,
)
from codewright.tools.invocation import ToolInvocation
from codewright.tools.registry import ToolRegistry
from codewright.workspace import WorkspaceManager

# -- parser -----------------------------------------------------------------


def test_parse_simple_add() -> None:
    ops = parse_patch(
        "*** Begin Patch\n*** Add File: foo.txt\n+hello\n+world\n*** End Patch\n"
    )
    assert len(ops) == 1
    assert ops[0].kind == "add"
    assert ops[0].path == "foo.txt"
    assert ops[0].contents == "hello\nworld\n"


def test_parse_simple_delete() -> None:
    ops = parse_patch("*** Begin Patch\n*** Delete File: gone.txt\n*** End Patch\n")
    assert ops[0].kind == "delete"
    assert ops[0].path == "gone.txt"


def test_parse_update_with_move() -> None:
    ops = parse_patch(
        "*** Begin Patch\n"
        "*** Update File: src/old.py\n"
        "*** Move to: src/new.py\n"
        "@@ def f():\n"
        "-    pass\n"
        "+    return 1\n"
        "*** End Patch\n"
    )
    assert ops[0].kind == "update"
    assert ops[0].move_to == "src/new.py"
    assert ops[0].hunks[0].change_context == "def f():"


def test_parse_missing_begin() -> None:
    with pytest.raises(PatchParseError, match="Begin Patch"):
        parse_patch("*** Add File: x\n+y\n*** End Patch\n")


def test_parse_rejects_absolute_path() -> None:
    with pytest.raises(PatchParseError):
        parse_patch("*** Begin Patch\n*** Add File: /etc/passwd\n+x\n*** End Patch\n")


def test_parse_rejects_parent_escape() -> None:
    with pytest.raises(PatchParseError):
        parse_patch("*** Begin Patch\n*** Delete File: ../escape\n*** End Patch\n")


# -- matcher: 4 levels ------------------------------------------------------


def test_match_exact() -> None:
    assert find_hunk(["a", "b", "c"], [], ["b", "c"]) == 1


def test_match_rstrip() -> None:
    assert find_hunk(["a   ", "b\t"], [], ["a", "b"]) == 0


def test_match_full_strip() -> None:
    assert find_hunk(["  a", "    b"], [], ["a", "b"]) == 0


def test_match_unicode_normalize_curly_quotes() -> None:
    # File has fancy curly quotes; pattern uses ASCII straight quotes.
    assert find_hunk(['x = “hi”', "y = 'bye'"], [], ['x = "hi"', "y = 'bye'"]) == 0


def test_match_unicode_normalize_dashes() -> None:
    # File uses U+2014 (em dash) and U+2013 (en dash); pattern uses ASCII '-'.
    # rstrip/trim cannot bridge a non-whitespace codepoint diff, so this case
    # locks the 4th-level normalize path specifically.
    em = "—"
    en = "–"
    file_lines = [f"range: 1{em}5", f"step: 1{en}1"]
    pattern = ["range: 1-5", "step: 1-1"]
    # Levels 1-3 cannot match: em/en dash is not whitespace.
    assert file_lines[0] != pattern[0]
    assert file_lines[0].rstrip() != pattern[0].rstrip()
    assert file_lines[0].strip() != pattern[0].strip()
    assert find_hunk(file_lines, [], pattern) == 0


def test_match_unicode_normalize_nbsp_midline() -> None:
    # File uses NBSP (U+00A0) inside the line; pattern uses ASCII space.
    # The NBSP is in the *middle* of the line so rstrip/strip cannot remove it;
    # only the level-4 unicode-normalize pass folds it to ASCII space.
    nbsp = " "
    file_lines = [f"foo{nbsp}bar{nbsp}baz", "next"]
    pattern = ["foo bar baz"]
    assert file_lines[0] != pattern[0]
    assert file_lines[0].rstrip() != pattern[0].rstrip()
    assert file_lines[0].strip() != pattern[0].strip()
    assert find_hunk(file_lines, [], pattern) == 0


def test_match_unicode_normalize_curly_quotes_locks_level_4() -> None:
    # Pure level-4 assertion: file has curly quotes, pattern has ASCII; pre-
    # level-4 passes cannot bridge the diff.
    file_lines = ['msg = “hi”', "next"]
    pattern = ['msg = "hi"']
    assert file_lines[0] != pattern[0]
    assert file_lines[0].rstrip() != pattern[0].rstrip()
    assert file_lines[0].strip() != pattern[0].strip()
    assert find_hunk(file_lines, [], pattern) == 0


def test_match_all_levels_fail() -> None:
    assert find_hunk(["alpha", "beta"], [], ["zeta"]) is None


def test_unicode_normalize_table() -> None:
    assert unicode_normalize("“x”") == '"x"'
    assert unicode_normalize("a—b") == "a-b"
    assert unicode_normalize("a b") == "a b"


# -- multi-anchor -----------------------------------------------------------


def test_single_anchor() -> None:
    file_lines = ["class Foo:", "    def bar(self):", "        pass"]
    idx = find_hunk(file_lines, ["class Foo"], ["    def bar(self):"])
    assert idx == 1


def test_two_level_anchor_disambiguates() -> None:
    file_lines = [
        "class Foo:",
        "    def go(self):",
        "        pass",
        "",
        "class Bar:",
        "    def go(self):",
        "        return 1",
    ]
    # Find go() but only inside Bar.
    idx = find_hunk(file_lines, ["class Bar", "def go"], ["        return 1"])
    assert idx == 6


def test_missing_anchor_returns_none() -> None:
    assert find_hunk(["x", "y"], ["class NonExistent"], ["x"]) is None


def test_second_anchor_multiple_occurrences_disambiguates_by_hunk() -> None:
    """First anchor pins us inside class Bar; second anchor 'def go' appears
    twice in that range; the hunk's own old_lines must pick the right one."""
    file_lines = [
        "class Foo:",
        "    def go(self):",
        "        return 0",
        "",
        "class Bar:",
        "    def go(self):",
        "        return 1",
        "    def helper(self):",
        "        pass",
        "    def go(self, x):",
        "        return x + 2",
    ]
    # We're inside Bar; 'def go' appears at lines 5 and 9. The hunk wants the
    # second one, distinguishable by its body 'return x + 2'.
    idx = find_hunk(
        file_lines,
        ["class Bar", "def go"],
        ["        return x + 2"],
    )
    assert idx == 10
    # Sanity: the same anchors with a body that matches the *first* def go
    # locks to the earlier match.
    idx_first = find_hunk(
        file_lines,
        ["class Bar", "def go"],
        ["        return 1"],
    )
    assert idx_first == 6


# -- replacement validation --------------------------------------------------


def test_apply_patch_overlap_validation_rejects_intersecting_replacements() -> None:
    ops = parse_patch(
        "*** Begin Patch\n"
        "*** Update File: overlap.txt\n"
        "@@\n"
        "-a\n"
        "+A\n"
        "*** End Patch\n"
    )
    with pytest.raises(RespondToModelError, match="overlapping hunks"):
        _validate_replacements(
            ops[0],
            [
                _Replacement(sequence=0, start=1, old_len=3, new_lines=["x"]),
                _Replacement(sequence=1, start=2, old_len=1, new_lines=["y"]),
            ],
        )


def test_apply_patch_overlap_validation_allows_adjacent_replacements() -> None:
    ops = parse_patch(
        "*** Begin Patch\n"
        "*** Update File: adjacent.txt\n"
        "@@\n"
        "-a\n"
        "+A\n"
        "*** End Patch\n"
    )
    _validate_replacements(
        ops[0],
        [
            _Replacement(sequence=0, start=0, old_len=1, new_lines=["A"]),
            _Replacement(sequence=1, start=1, old_len=1, new_lines=["B"]),
            _Replacement(sequence=2, start=1, old_len=0, new_lines=["inserted"]),
        ],
    )


def test_apply_patch_overlap_validation_rejects_insertion_inside_replacement() -> None:
    ops = parse_patch(
        "*** Begin Patch\n"
        "*** Update File: insertion-overlap.txt\n"
        "@@\n"
        "-a\n"
        "+A\n"
        "*** End Patch\n"
    )
    with pytest.raises(RespondToModelError, match="overlapping hunks"):
        _validate_replacements(
            ops[0],
            [
                _Replacement(sequence=0, start=2, old_len=0, new_lines=["inserted"]),
                _Replacement(sequence=1, start=1, old_len=3, new_lines=["replacement"]),
            ],
        )


def test_apply_patch_overlap_validation_allows_insertion_at_replacement_boundary() -> None:
    ops = parse_patch(
        "*** Begin Patch\n"
        "*** Update File: insertion-boundary.txt\n"
        "@@\n"
        "-a\n"
        "+A\n"
        "*** End Patch\n"
    )
    _validate_replacements(
        ops[0],
        [
            _Replacement(sequence=0, start=1, old_len=3, new_lines=["replacement"]),
            _Replacement(sequence=1, start=1, old_len=0, new_lines=["before"]),
            _Replacement(sequence=2, start=4, old_len=0, new_lines=["after"]),
        ],
    )


# -- end-to-end handler -----------------------------------------------------


@pytest.fixture
def session(tmp_path: Path):
    from codewright.agent.session import Session

    async def make() -> Session:
        wm = WorkspaceManager(tmp_path, PermissionProfile.WORKSPACE_WRITE)
        sess = Session(
            session_id="s1",
            cwd=tmp_path,
            permission_profile=PermissionProfile.WORKSPACE_WRITE,
            workspace=wm,
        )
        # Drain the bootstrap event so it doesn't sit in the queue.
        await sess.next_event()
        return sess

    return make


def _inv(session, tool_name: str, args: dict, cwd: Path) -> ToolInvocation:
    return ToolInvocation(
        session=session,
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
async def test_apply_patch_add_file_end_to_end(tmp_path: Path, session) -> None:
    sess = await session()
    try:
        handler = ApplyPatchHandler()
        patch = (
            "*** Begin Patch\n"
            "*** Add File: hello.txt\n"
            "+hello\n"
            "+world\n"
            "*** End Patch\n"
        )
        result = await handler.handle(_inv(sess, "apply_patch", {"patch": patch}, tmp_path))
        assert result.success is True
        assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "hello\nworld\n"
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_apply_patch_update_with_anchor(tmp_path: Path, session) -> None:
    sess = await session()
    try:
        target = tmp_path / "mod.py"
        target.write_text("class Foo:\n    def bar(self):\n        pass\n", encoding="utf-8")
        handler = ApplyPatchHandler()
        patch = (
            "*** Begin Patch\n"
            "*** Update File: mod.py\n"
            "@@ class Foo:\n"
            "     def bar(self):\n"
            "-        pass\n"
            "+        return 42\n"
            "*** End Patch\n"
        )
        result = await handler.handle(_inv(sess, "apply_patch", {"patch": patch}, tmp_path))
        assert result.success is True
        assert "return 42" in target.read_text(encoding="utf-8")
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_apply_patch_move_to_existing_target_is_rejected(
    tmp_path: Path, session
) -> None:
    sess = await session()
    try:
        source = tmp_path / "source.txt"
        target = tmp_path / "target.txt"
        source.write_text("before\n", encoding="utf-8")
        target.write_text("do not overwrite\n", encoding="utf-8")
        handler = ApplyPatchHandler()
        patch = (
            "*** Begin Patch\n"
            "*** Update File: source.txt\n"
            "*** Move to: target.txt\n"
            "@@\n"
            "-before\n"
            "+after\n"
            "*** End Patch\n"
        )
        with pytest.raises(RespondToModelError, match="destination already exists"):
            await handler.handle(_inv(sess, "apply_patch", {"patch": patch}, tmp_path))
        assert source.read_text(encoding="utf-8") == "before\n"
        assert target.read_text(encoding="utf-8") == "do not overwrite\n"
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_apply_patch_delete_then_move_to_same_target_succeeds(
    tmp_path: Path, session
) -> None:
    sess = await session()
    try:
        source = tmp_path / "source.txt"
        target = tmp_path / "target.txt"
        source.write_text("before\n", encoding="utf-8")
        target.write_text("old target\n", encoding="utf-8")
        handler = ApplyPatchHandler()
        patch = (
            "*** Begin Patch\n"
            "*** Delete File: target.txt\n"
            "*** Update File: source.txt\n"
            "*** Move to: target.txt\n"
            "@@\n"
            "-before\n"
            "+after\n"
            "*** End Patch\n"
        )
        result = await handler.handle(_inv(sess, "apply_patch", {"patch": patch}, tmp_path))
        assert result.success is True
        assert not source.exists()
        assert target.read_text(encoding="utf-8") == "after\n"
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_apply_patch_move_target_is_in_approval_paths(
    tmp_path: Path, session, monkeypatch
) -> None:
    sess = await session()
    try:
        source = tmp_path / "source.txt"
        source.write_text("before\n", encoding="utf-8")
        captured: dict[str, list[str]] = {}

        async def check_action(action, _session):
            captured["paths"] = action.details["paths"]
            return ReviewDecision.APPROVED

        monkeypatch.setattr(sess.workspace, "check_action", check_action)
        handler = ApplyPatchHandler()
        patch = (
            "*** Begin Patch\n"
            "*** Update File: source.txt\n"
            "*** Move to: moved/target.txt\n"
            "@@\n"
            "-before\n"
            "+after\n"
            "*** End Patch\n"
        )
        result = await handler.handle(_inv(sess, "apply_patch", {"patch": patch}, tmp_path))
        assert result.success is True
        assert captured["paths"] == [
            str(tmp_path / "source.txt"),
            str(tmp_path / "moved" / "target.txt"),
        ]
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_apply_patch_repeated_updates_to_same_file_accumulate(
    tmp_path: Path, session
) -> None:
    sess = await session()
    try:
        target = tmp_path / "multi.txt"
        target.write_text("one\ntwo\nthree\n", encoding="utf-8")
        handler = ApplyPatchHandler()
        patch = (
            "*** Begin Patch\n"
            "*** Update File: multi.txt\n"
            "@@\n"
            "-one\n"
            "+ONE\n"
            "*** Update File: multi.txt\n"
            "@@\n"
            "-two\n"
            "+TWO\n"
            "*** End Patch\n"
        )
        result = await handler.handle(_inv(sess, "apply_patch", {"patch": patch}, tmp_path))
        assert result.success is True
        assert target.read_text(encoding="utf-8") == "ONE\nTWO\nthree\n"
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_apply_patch_add_then_update_uses_virtual_file_state(
    tmp_path: Path, session
) -> None:
    sess = await session()
    try:
        target = tmp_path / "created.txt"
        handler = ApplyPatchHandler()
        patch = (
            "*** Begin Patch\n"
            "*** Add File: created.txt\n"
            "+one\n"
            "+two\n"
            "*** Update File: created.txt\n"
            "@@\n"
            "-two\n"
            "+TWO\n"
            "*** End Patch\n"
        )
        result = await handler.handle(_inv(sess, "apply_patch", {"patch": patch}, tmp_path))
        assert result.success is True
        assert target.read_text(encoding="utf-8") == "one\nTWO\n"
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_apply_patch_move_then_update_destination_uses_virtual_file_state(
    tmp_path: Path, session
) -> None:
    sess = await session()
    try:
        source = tmp_path / "source.txt"
        target = tmp_path / "target.txt"
        source.write_text("one\ntwo\n", encoding="utf-8")
        handler = ApplyPatchHandler()
        patch = (
            "*** Begin Patch\n"
            "*** Update File: source.txt\n"
            "*** Move to: target.txt\n"
            "@@\n"
            "-one\n"
            "+ONE\n"
            "*** Update File: target.txt\n"
            "@@\n"
            "-two\n"
            "+TWO\n"
            "*** End Patch\n"
        )
        result = await handler.handle(_inv(sess, "apply_patch", {"patch": patch}, tmp_path))
        assert result.success is True
        assert not source.exists()
        assert target.read_text(encoding="utf-8") == "ONE\nTWO\n"
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_apply_patch_plan_build_failure_does_not_commit_prior_virtual_add(
    tmp_path: Path, session
) -> None:
    sess = await session()
    try:
        handler = ApplyPatchHandler()
        patch = (
            "*** Begin Patch\n"
            "*** Add File: created.txt\n"
            "+created\n"
            "*** Update File: missing.txt\n"
            "@@\n"
            "-missing\n"
            "+updated\n"
            "*** End Patch\n"
        )
        with pytest.raises(RespondToModelError, match=r"cannot update missing file 'missing\.txt'"):
            await handler.handle(_inv(sess, "apply_patch", {"patch": patch}, tmp_path))
        assert not (tmp_path / "created.txt").exists()
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_apply_patch_delete_then_add_same_path_commits_final_write(
    tmp_path: Path, session
) -> None:
    sess = await session()
    try:
        target = tmp_path / "replace.txt"
        target.write_text("old\n", encoding="utf-8")
        handler = ApplyPatchHandler()
        patch = (
            "*** Begin Patch\n"
            "*** Delete File: replace.txt\n"
            "*** Add File: replace.txt\n"
            "+new\n"
            "*** End Patch\n"
        )
        result = await handler.handle(_inv(sess, "apply_patch", {"patch": patch}, tmp_path))
        assert result.success is True
        assert target.read_text(encoding="utf-8") == "new\n"
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_apply_patch_commit_write_failure_does_not_run_planned_deletes(
    tmp_path: Path, session
) -> None:
    sess = await session()
    try:
        keep = tmp_path / "keep.txt"
        keep.write_text("do not delete\n", encoding="utf-8")
        (tmp_path / "blocked").write_text("not a directory", encoding="utf-8")
        handler = ApplyPatchHandler()
        patch = (
            "*** Begin Patch\n"
            "*** Delete File: keep.txt\n"
            "*** Add File: blocked/new.txt\n"
            "+new\n"
            "*** End Patch\n"
        )
        with pytest.raises(RespondToModelError, match=r"cannot write 'blocked/new\.txt'"):
            await handler.handle(_inv(sess, "apply_patch", {"patch": patch}, tmp_path))
        assert keep.read_text(encoding="utf-8") == "do not delete\n"
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_apply_patch_commit_write_failure_keeps_prior_successful_write(
    tmp_path: Path, session
) -> None:
    sess = await session()
    try:
        (tmp_path / "blocked").write_text("not a directory", encoding="utf-8")
        handler = ApplyPatchHandler()
        patch = (
            "*** Begin Patch\n"
            "*** Add File: created.txt\n"
            "+created\n"
            "*** Add File: blocked/new.txt\n"
            "+new\n"
            "*** End Patch\n"
        )
        with pytest.raises(RespondToModelError, match=r"cannot write 'blocked/new\.txt'"):
            await handler.handle(_inv(sess, "apply_patch", {"patch": patch}, tmp_path))
        assert (tmp_path / "created.txt").read_text(encoding="utf-8") == "created\n"
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_apply_patch_repeated_prior_context_anchor_is_rejected(
    tmp_path: Path, session
) -> None:
    sess = await session()
    try:
        target = tmp_path / "user.py"
        target.write_text(
            "class User:\n"
            "    def name(self):\n"
            "        return \"old name\"\n"
            "\n"
            "    def save(self):\n"
            "        return False\n",
            encoding="utf-8",
        )
        handler = ApplyPatchHandler()
        patch = (
            "*** Begin Patch\n"
            "*** Update File: user.py\n"
            "@@ class User:\n"
            "-        return \"old name\"\n"
            "+        return \"new name\"\n"
            "@@ class User:\n"
            "-        return False\n"
            "+        return True\n"
            "*** End Patch\n"
        )
        with pytest.raises(RespondToModelError, match="failed to locate hunk"):
            await handler.handle(_inv(sess, "apply_patch", {"patch": patch}, tmp_path))
        assert target.read_text(encoding="utf-8") == (
            "class User:\n"
            "    def name(self):\n"
            "        return \"old name\"\n"
            "\n"
            "    def save(self):\n"
            "        return False\n"
        )
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_apply_patch_out_of_order_context_does_not_misapply(
    tmp_path: Path, session
) -> None:
    sess = await session()
    try:
        target = tmp_path / "out_of_order.py"
        original = (
            "class A:\n"
            "    x = 1\n"
            "\n"
            "class B:\n"
            "    x = 1\n"
        )
        target.write_text(original, encoding="utf-8")
        handler = ApplyPatchHandler()
        patch = (
            "*** Begin Patch\n"
            "*** Update File: out_of_order.py\n"
            "@@ class A:\n"
            "-    x = 1\n"
            "+    x = 2\n"
            "@@ class A:\n"
            "-    x = 1\n"
            "+    x = 3\n"
            "*** End Patch\n"
        )
        with pytest.raises(RespondToModelError, match="failed to locate hunk"):
            await handler.handle(_inv(sess, "apply_patch", {"patch": patch}, tmp_path))
        assert target.read_text(encoding="utf-8") == original
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_apply_patch_context_anchor_keeps_substring_matching(
    tmp_path: Path, session
) -> None:
    sess = await session()
    try:
        target = tmp_path / "partial.py"
        target.write_text(
            "class Service:\n"
            "    def compute(self):\n"
            "        return 1\n",
            encoding="utf-8",
        )
        handler = ApplyPatchHandler()
        patch = (
            "*** Begin Patch\n"
            "*** Update File: partial.py\n"
            "@@ def compute\n"
            "-        return 1\n"
            "+        return 2\n"
            "*** End Patch\n"
        )
        result = await handler.handle(_inv(sess, "apply_patch", {"patch": patch}, tmp_path))
        assert result.success is True
        assert target.read_text(encoding="utf-8") == (
            "class Service:\n"
            "    def compute(self):\n"
            "        return 2\n"
        )
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_apply_patch_anchor_miss_is_respond_to_model(tmp_path: Path, session) -> None:
    sess = await session()
    try:
        (tmp_path / "x.py").write_text("class Real:\n    pass\n", encoding="utf-8")
        handler = ApplyPatchHandler()
        patch = (
            "*** Begin Patch\n"
            "*** Update File: x.py\n"
            "@@ class NonExistent:\n"
            "-    pass\n"
            "+    new\n"
            "*** End Patch\n"
        )
        with pytest.raises(RespondToModelError):
            await handler.handle(_inv(sess, "apply_patch", {"patch": patch}, tmp_path))
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_apply_patch_non_utf8_update_is_tool_failure(tmp_path: Path, session) -> None:
    sess = await session()
    try:
        target = tmp_path / "bad.bin"
        target.write_bytes(b"\xff\xfe\x00")
        patch = (
            "*** Begin Patch\n"
            "*** Update File: bad.bin\n"
            "@@\n"
            "-old\n"
            "+new\n"
            "*** End Patch\n"
        )
        registry = ToolRegistry().register(ApplyPatchHandler())
        executor = ToolExecutor(registry)
        results = await executor.dispatch_batch(
            [_inv(sess, "apply_patch", {"patch": patch}, tmp_path)]
        )
        assert results[0].success is False
        assert "apply_patch: cannot read 'bad.bin': file is not valid UTF-8" in results[0].body
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_apply_patch_write_error_is_respond_to_model(tmp_path: Path, session) -> None:
    sess = await session()
    try:
        (tmp_path / "blocked").write_text("not a directory", encoding="utf-8")
        handler = ApplyPatchHandler()
        patch = (
            "*** Begin Patch\n"
            "*** Add File: blocked/new.txt\n"
            "+hello\n"
            "*** End Patch\n"
        )
        with pytest.raises(RespondToModelError, match=r"cannot write 'blocked/new\.txt'"):
            await handler.handle(_inv(sess, "apply_patch", {"patch": patch}, tmp_path))
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_apply_patch_delete_error_is_respond_to_model(
    tmp_path: Path, session, monkeypatch
) -> None:
    sess = await session()
    try:
        target = tmp_path / "gone.txt"
        target.write_text("delete me", encoding="utf-8")
        original_unlink = Path.unlink

        def fail_target_unlink(self: Path, *args, **kwargs):
            if self == target:
                raise PermissionError("locked")
            return original_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_target_unlink)
        handler = ApplyPatchHandler()
        patch = "*** Begin Patch\n*** Delete File: gone.txt\n*** End Patch\n"
        with pytest.raises(RespondToModelError, match=r"cannot delete 'gone\.txt'"):
            await handler.handle(_inv(sess, "apply_patch", {"patch": patch}, tmp_path))
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
async def test_apply_patch_path_safety_absolute(tmp_path: Path, session) -> None:
    sess = await session()
    try:
        handler = ApplyPatchHandler()
        patch = (
            "*** Begin Patch\n"
            "*** Add File: /etc/passwd\n"
            "+pwn\n"
            "*** End Patch\n"
        )
        with pytest.raises(RespondToModelError):
            await handler.handle(_inv(sess, "apply_patch", {"patch": patch}, tmp_path))
    finally:
        await sess.shutdown()


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on win")
async def test_apply_patch_symlink_escape_blocked(tmp_path: Path, session) -> None:
    sess = await session()
    try:
        outside = tmp_path.parent / "elsewhere_apply"
        outside.mkdir(exist_ok=True)
        link = tmp_path / "out"
        if not link.exists():
            os.symlink(outside, link)
        handler = ApplyPatchHandler()
        # The parser accepts the relative path; canonicalize must reject the
        # symlink target that resolves outside ``tmp_path``.
        patch = (
            "*** Begin Patch\n"
            "*** Add File: out/secret.txt\n"
            "+pwn\n"
            "*** End Patch\n"
        )
        with pytest.raises(RespondToModelError):
            await handler.handle(_inv(sess, "apply_patch", {"patch": patch}, tmp_path))
    finally:
        await sess.shutdown()
