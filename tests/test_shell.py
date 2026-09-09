"""shell tool family: state persistence, output pipeline, jobs, safety analysis.

Acceptance set (ADR D-3-008): flood output -> truncate + spill + paging;
progress bars -> \r collapse; interactive prompts -> env armor; long-running
server -> background trio; empty output -> explicit confirmation; compound
command -> per-segment permission analysis.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

import pytest

from codewright.agent.cancellation import CancellationToken
from codewright.agent.session import Session
from codewright.agent.turn_context import TurnContext
from codewright.protocol import AskForApproval, PendingAction, PermissionProfile
from codewright.tools.handlers._shell import ShellManager, clean_output
from codewright.tools.handlers._shell_safety import analyze_command
from codewright.tools.handlers.shell import ShellHandler
from codewright.tools.handlers.shell_kill import ShellKillHandler
from codewright.tools.handlers.shell_output import ShellOutputHandler
from codewright.tools.invocation import ToolInvocation
from codewright.workspace import WorkspaceManager
from codewright.workspace.permissions import action_signature, assess_action

_MANAGER = ShellManager()
needs_bash = pytest.mark.skipif(
    _MANAGER.dialect.flavor != "bash", reason="bash not found on this machine"
)


async def _session(tmp_path: Path) -> Session:
    wm = WorkspaceManager(tmp_path, PermissionProfile.WORKSPACE_WRITE)
    sess = Session(
        session_id=uuid.uuid4().hex[:8],
        cwd=tmp_path,
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
        workspace=wm,
    )
    await sess.next_event()
    return sess


def _inv(sess, tool_name: str, arguments: dict, cwd: Path) -> ToolInvocation:
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
        arguments=arguments,
        cancellation_token=CancellationToken(),
    )


# ---------------------------------------------------------------------------
# Foreground basics
# ---------------------------------------------------------------------------


@needs_bash
async def test_simple_success_echoes_cwd(tmp_path: Path) -> None:
    sess = await _session(tmp_path)
    manager = ShellManager()
    try:
        h = ShellHandler(manager)
        result = await h.handle(_inv(sess, "shell", {"command": "echo hello"}, tmp_path))
        assert result.success is True
        assert "hello" in result.body
        assert result.structured_data["exit_code"] == 0
        assert result.structured_data["cwd"]
        assert "Cwd:" in result.body
    finally:
        await manager.aclose()
        await sess.shutdown()


@needs_bash
async def test_empty_output_is_confirmed(tmp_path: Path) -> None:
    sess = await _session(tmp_path)
    manager = ShellManager()
    try:
        h = ShellHandler(manager)
        result = await h.handle(_inv(sess, "shell", {"command": "true"}, tmp_path))
        assert result.success is True
        assert "(command succeeded, no output)" in result.body
    finally:
        await manager.aclose()
        await sess.shutdown()


@needs_bash
async def test_nonzero_exit_is_surfaced_not_failed(tmp_path: Path) -> None:
    sess = await _session(tmp_path)
    manager = ShellManager()
    try:
        h = ShellHandler(manager)
        result = await h.handle(_inv(sess, "shell", {"command": "exit 7"}, tmp_path))
        assert result.success is True  # tool ran; exit code is data (D-3-003)
        assert result.structured_data["exit_code"] == 7
        assert "Exit code: 7" in result.body
    finally:
        await manager.aclose()
        await sess.shutdown()


@needs_bash
async def test_missing_command_reports_127(tmp_path: Path) -> None:
    sess = await _session(tmp_path)
    manager = ShellManager()
    try:
        h = ShellHandler(manager)
        result = await h.handle(
            _inv(sess, "shell", {"command": "definitely_not_a_command_xyz_123"}, tmp_path)
        )
        assert result.structured_data["exit_code"] == 127
        assert "not found" in result.body.lower()
    finally:
        await manager.aclose()
        await sess.shutdown()


async def test_spawn_dir_failure_becomes_spawn_error(tmp_path: Path, monkeypatch) -> None:
    # Bug 4 regression: a filesystem failure while creating the jobs/state dirs
    # must surface as a clean spawn_error ToolResult, not an unhandled OSError
    # escaping the handler (the executor only folds RespondToModelError).
    sess = await _session(tmp_path)
    manager = ShellManager()
    try:
        def boom(*_a, **_k):
            raise OSError("disk full")

        monkeypatch.setattr(manager, "jobs_dir", boom)
        h = ShellHandler(manager)
        result = await h.handle(_inv(sess, "shell", {"command": "echo hi"}, tmp_path))
        assert result.success is False
        assert result.structured_data["spawn_error"] is True
        assert "failed to start" in result.body.lower()
    finally:
        await manager.aclose()
        await sess.shutdown()


@needs_bash
async def test_quoted_path_with_spaces(tmp_path: Path) -> None:
    """Regression: cmd /c used to mangle quoted space-paths; bash must not."""
    spaced = tmp_path / "a b"
    spaced.mkdir()
    (spaced / "s.txt").write_text("needle-in-spaced-path\n", encoding="utf-8")
    sess = await _session(tmp_path)
    manager = ShellManager()
    try:
        h = ShellHandler(manager)
        result = await h.handle(
            _inv(sess, "shell", {"command": 'cat "a b/s.txt"'}, tmp_path)
        )
        assert result.structured_data["exit_code"] == 0
        assert "needle-in-spaced-path" in result.body
    finally:
        await manager.aclose()
        await sess.shutdown()


# ---------------------------------------------------------------------------
# Session state (plan B: persistent state, ephemeral process)
# ---------------------------------------------------------------------------


@needs_bash
async def test_cd_persists_across_calls(tmp_path: Path) -> None:
    sess = await _session(tmp_path)
    manager = ShellManager()
    try:
        h = ShellHandler(manager)
        r1 = await h.handle(
            _inv(sess, "shell", {"command": "mkdir -p subdir && cd subdir"}, tmp_path)
        )
        assert r1.structured_data["exit_code"] == 0
        r2 = await h.handle(_inv(sess, "shell", {"command": "pwd"}, tmp_path))
        assert r2.structured_data["cwd"].replace("\\", "/").endswith("subdir")
    finally:
        await manager.aclose()
        await sess.shutdown()


@needs_bash
async def test_export_persists_and_restart_clears(tmp_path: Path) -> None:
    sess = await _session(tmp_path)
    manager = ShellManager()
    try:
        h = ShellHandler(manager)
        await h.handle(
            _inv(sess, "shell", {"command": "export CW_TEST_FOO=bar123"}, tmp_path)
        )
        r2 = await h.handle(
            _inv(sess, "shell", {"command": 'echo "v=$CW_TEST_FOO"'}, tmp_path)
        )
        assert "v=bar123" in r2.body
        r3 = await h.handle(
            _inv(
                sess,
                "shell",
                {"command": 'echo "v=$CW_TEST_FOO"', "restart": True},
                tmp_path,
            )
        )
        assert "v=bar123" not in r3.body
        assert "v=" in r3.body
    finally:
        await manager.aclose()
        await sess.shutdown()


@needs_bash
async def test_named_sessions_are_isolated(tmp_path: Path) -> None:
    sess = await _session(tmp_path)
    manager = ShellManager()
    try:
        h = ShellHandler(manager)
        await h.handle(
            _inv(
                sess,
                "shell",
                {"command": "mkdir -p deep && cd deep", "session": "a"},
                tmp_path,
            )
        )
        r = await h.handle(
            _inv(sess, "shell", {"command": "pwd", "session": "b"}, tmp_path)
        )
        assert not r.structured_data["cwd"].replace("\\", "/").endswith("deep")
    finally:
        await manager.aclose()
        await sess.shutdown()


@needs_bash
async def test_env_armor_suppresses_prompts(tmp_path: Path) -> None:
    sess = await _session(tmp_path)
    manager = ShellManager()
    try:
        h = ShellHandler(manager)
        result = await h.handle(
            _inv(
                sess,
                "shell",
                {"command": 'echo "g=$GIT_TERMINAL_PROMPT ci=$CI p=$PAGER"'},
                tmp_path,
            )
        )
        assert "g=0 ci=true p=cat" in result.body
    finally:
        await manager.aclose()
        await sess.shutdown()


# ---------------------------------------------------------------------------
# Time: timeout kills the tree
# ---------------------------------------------------------------------------


@needs_bash
async def test_cancelling_foreground_kills_child(tmp_path: Path) -> None:
    # Interrupting a running foreground command (the coroutine is cancelled via
    # task.cancel, as the turn-interrupt path does) must reap the child process
    # promptly, not leak it until session shutdown.
    sess = await _session(tmp_path)
    manager = ShellManager()
    try:
        key = manager.session_key(sess.session_id, "main")
        run = asyncio.create_task(
            manager.run_foreground(
                root=tmp_path,
                session_key=key,
                session_name="main",
                command="sleep 30",
                start_cwd=tmp_path,
                timeout_ms=60_000,
                cancellation_token=CancellationToken(),
            )
        )
        # Wait until the child has actually spawned.
        for _ in range(100):
            await asyncio.sleep(0.05)
            if manager._jobs:
                break
        job = next(iter(manager._jobs.values()))
        assert job.proc is not None
        assert job.proc.returncode is None  # still running

        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

        # The CancelledError handler ran (status flipped) and the child is being
        # reaped — its returncode lands shortly (not held until aclose()).
        assert job.status == "killed"
        for _ in range(50):
            if job.proc.returncode is not None:
                break
            await asyncio.sleep(0.1)
        assert job.proc.returncode is not None, "child not reaped on cancellation"
    finally:
        await manager.aclose()
        await sess.shutdown()


@needs_bash
async def test_timeout_kills_process_tree(tmp_path: Path) -> None:
    sess = await _session(tmp_path)
    manager = ShellManager()
    try:
        h = ShellHandler(manager)
        start = time.monotonic()
        result = await h.handle(
            _inv(
                sess,
                "shell",
                {"command": "sleep 30", "timeout_ms": 1500},
                tmp_path,
            )
        )
        elapsed = time.monotonic() - start
        assert result.success is False
        assert result.structured_data["timed_out"] is True
        assert "larger timeout_ms" in result.body
        assert elapsed < 15
    finally:
        await manager.aclose()
        await sess.shutdown()


# ---------------------------------------------------------------------------
# Output pipeline
# ---------------------------------------------------------------------------


def test_clean_output_collapses_progress_bars() -> None:
    assert clean_output("download 10%\rdownload 50%\rdownload 100%\ndone\n") == (
        "download 100%\ndone\n"
    )


def test_clean_output_strips_ansi() -> None:
    assert clean_output("\x1b[31mred\x1b[0m plain\x1b]0;title\x07\n") == "red plain\n"


def test_clean_output_folds_repeated_lines() -> None:
    cleaned = clean_output("same\n" * 50 + "tail\n")
    assert cleaned.count("same") == 1
    assert "[previous line repeated 49 more times]" in cleaned
    assert "tail" in cleaned


@needs_bash
async def test_flood_is_truncated_spilled_and_pageable(tmp_path: Path) -> None:
    sess = await _session(tmp_path)
    manager = ShellManager()
    try:
        h = ShellHandler(manager)
        # ~390 KB of distinct lines (no fold) > head+tail caps -> middle dropped.
        result = await h.handle(
            _inv(
                sess,
                "shell",
                {"command": "seq -w 1 30000 | sed 's/^/line-/'"},
                tmp_path,
            )
        )
        assert result.structured_data["exit_code"] == 0
        assert result.structured_data["truncated"] is True
        assert "shell_output" in result.body
        job_id = result.structured_data["job_id"]

        out = ShellOutputHandler(manager)
        cursor = 0
        pages = 0
        seen: set[str] = set()
        while pages < 200:
            page = await out.handle(
                _inv(
                    sess,
                    "shell_output",
                    {"job_id": job_id, "cursor": cursor},
                    tmp_path,
                )
            )
            for token in page.body.split():
                if token.startswith("line-"):
                    seen.add(token)
            cursor = page.structured_data["next_cursor"]
            pages += 1
            if page.structured_data["eof"]:
                break
        assert page.structured_data["eof"] is True
        # Bug 1 regression: paging must expose EVERY byte, not just head+tail of
        # each page. The old per-page middle-truncation dropped ~45% (~1795 of
        # 4000 in the probe). The only legitimate gaps now are lines split across
        # a page boundary — at most one per page — so tolerate < pages misses.
        expected = {f"line-{i:05d}" for i in range(1, 30001)}
        missing = expected - seen
        assert len(missing) <= pages, (
            f"paging dropped {len(missing)} lines over {pages} pages "
            f"(only boundary splits, < {pages}, are acceptable); "
            f"e.g. {sorted(missing)[:3]}"
        )
    finally:
        await manager.aclose()
        await sess.shutdown()


@needs_bash
async def test_shell_output_unknown_job_errors(tmp_path: Path) -> None:
    sess = await _session(tmp_path)
    manager = ShellManager()
    try:
        out = ShellOutputHandler(manager)
        from codewright.tools.errors import RespondToModelError

        with pytest.raises(RespondToModelError):
            await out.handle(
                _inv(sess, "shell_output", {"job_id": "nope1234"}, tmp_path)
            )
    finally:
        await manager.aclose()
        await sess.shutdown()


# ---------------------------------------------------------------------------
# Background trio
# ---------------------------------------------------------------------------


@needs_bash
async def test_background_job_lifecycle(tmp_path: Path) -> None:
    sess = await _session(tmp_path)
    manager = ShellManager()
    try:
        h = ShellHandler(manager)
        spawned = await h.handle(
            _inv(
                sess,
                "shell",
                {"command": "echo listening; sleep 30", "background": True},
                tmp_path,
            )
        )
        assert spawned.success is True
        job_id = spawned.structured_data["job_id"]
        assert spawned.structured_data["status"] == "running"

        out = ShellOutputHandler(manager)
        body = ""
        for _ in range(100):
            page = await out.handle(
                _inv(sess, "shell_output", {"job_id": job_id}, tmp_path)
            )
            body = page.body
            if "listening" in body:
                break
            await asyncio.sleep(0.1)
        assert "listening" in body

        kill = ShellKillHandler(manager)
        killed = await kill.handle(
            _inv(sess, "shell_kill", {"target": job_id}, tmp_path)
        )
        assert killed.structured_data["killed"] is True

        page = await out.handle(
            _inv(sess, "shell_output", {"job_id": job_id}, tmp_path)
        )
        assert page.structured_data["status"] == "killed"
    finally:
        await manager.aclose()
        await sess.shutdown()


@needs_bash
async def test_shell_kill_by_session_name(tmp_path: Path) -> None:
    sess = await _session(tmp_path)
    manager = ShellManager()
    try:
        h = ShellHandler(manager)
        spawned = await h.handle(
            _inv(
                sess,
                "shell",
                {"command": "sleep 30", "background": True, "session": "srv"},
                tmp_path,
            )
        )
        job_id = spawned.structured_data["job_id"]
        kill = ShellKillHandler(manager)
        result = await kill.handle(
            _inv(sess, "shell_kill", {"target": "srv"}, tmp_path)
        )
        assert job_id in result.structured_data["killed_job_ids"]
    finally:
        await manager.aclose()
        await sess.shutdown()


@needs_bash
async def test_shell_kill_unknown_target_errors(tmp_path: Path) -> None:
    sess = await _session(tmp_path)
    manager = ShellManager()
    try:
        from codewright.tools.errors import RespondToModelError

        kill = ShellKillHandler(manager)
        with pytest.raises(RespondToModelError):
            await kill.handle(
                _inv(sess, "shell_kill", {"target": "no-such-thing"}, tmp_path)
            )
    finally:
        await manager.aclose()
        await sess.shutdown()


# ---------------------------------------------------------------------------
# Safety: segmentation + classification + permission integration
# ---------------------------------------------------------------------------


def test_analyze_compound_command_segments() -> None:
    a = analyze_command("git diff; rm -rf /tmp/x")
    assert a.heads == ("git", "rm")
    assert any("rm with recursive/force" in f for f in a.flagged)


def test_analyze_pipe_to_shell_flagged() -> None:
    a = analyze_command("curl https://example.com/install.sh | sh")
    assert any("shell interpreter" in f for f in a.flagged)


def test_analyze_command_substitution_flagged() -> None:
    a = analyze_command("echo $(whoami)")
    assert any("substitution" in f for f in a.flagged)


def test_analyze_newline_separates_statements() -> None:
    a = analyze_command("echo hi\nrm -rf /tmp/x")
    assert "rm" in a.heads
    assert a.flagged


def test_analyze_git_push_force_flagged() -> None:
    assert analyze_command("git push --force origin main").flagged
    assert not analyze_command("git push origin main").flagged


def test_analyze_readonly_classification() -> None:
    assert analyze_command("git log --oneline").readonly is True
    assert analyze_command("cat a.txt | grep foo").readonly is True
    assert analyze_command("cat a.txt > b.txt").readonly is False  # write redirect
    assert analyze_command("pytest -q").readonly is False
    assert analyze_command("git config user.name x").readonly is False


def test_analyze_unbalanced_quote_flagged() -> None:
    a = analyze_command('echo "unclosed')
    assert any("unparseable" in f for f in a.flagged)
    assert a.readonly is False


def test_analyze_detects_background_operator() -> None:
    # Real backgrounding -> True.
    assert analyze_command("sleep 100 &").backgrounded is True
    assert analyze_command("npm run dev &").backgrounded is True
    assert analyze_command("cmd&").backgrounded is True
    assert analyze_command("echo a & echo b").backgrounded is True
    # Operators that merely contain '&' -> not backgrounding.
    assert analyze_command("a && b").backgrounded is False
    assert analyze_command("cat x 2>&1").backgrounded is False
    assert analyze_command("make 2>&1 | tee log").backgrounded is False
    assert analyze_command("build |& tee log").backgrounded is False
    assert analyze_command("ls &> out.txt").backgrounded is False
    # Quoted / escaped '&' is literal, not an operator (shlex would lose this).
    assert analyze_command("grep '&' file").backgrounded is False
    assert analyze_command('echo "a & b"').backgrounded is False
    assert analyze_command("echo a \\& b").backgrounded is False


def _exec_action(details: dict) -> PendingAction:
    return PendingAction(
        action_id="a1",
        kind="exec",
        summary=details.get("command", "x"),
        details=details,
    )


def test_assess_flagged_asks_even_inside_workspace(tmp_path: Path) -> None:
    action = _exec_action(
        {"command": "rm -rf x", "segments": ["rm"], "flagged": ["rm -rf"], "cwd": str(tmp_path)}
    )
    verdict = assess_action(
        PermissionProfile.WORKSPACE_WRITE, action, set(), tmp_path, tmp_path
    )
    assert verdict == "ask"


def test_assess_readonly_outside_workspace_still_asks(tmp_path: Path) -> None:
    # The workspace root is a hard boundary: even a read-only command running
    # outside it must ask (a persistent shell can `cd` out). See Bug 2 fix.
    outside = tmp_path.parent
    action = _exec_action(
        {
            "command": "cat secret",
            "segments": ["cat"],
            "flagged": [],
            "readonly": True,
            "cwd": str(outside),
        }
    )
    verdict = assess_action(
        PermissionProfile.WORKSPACE_WRITE, action, set(), tmp_path, tmp_path / "ws"
    )
    assert verdict == "ask"


def test_assess_non_readonly_outside_workspace_asks(tmp_path: Path) -> None:
    outside = tmp_path.parent
    action = _exec_action(
        {
            "command": "pytest",
            "segments": ["pytest"],
            "flagged": [],
            "readonly": False,
            "cwd": str(outside),
        }
    )
    verdict = assess_action(
        PermissionProfile.WORKSPACE_WRITE, action, set(), tmp_path, tmp_path / "ws"
    )
    assert verdict == "ask"


def test_signature_flagged_uses_exact_command() -> None:
    flagged = _exec_action(
        {"command": "rm -rf x", "segments": ["rm"], "flagged": ["rm -rf"]}
    )
    assert action_signature(flagged) == "exec:rm -rf x"
    plain = _exec_action({"command": "git diff; ls", "segments": ["git", "ls"]})
    assert action_signature(plain) == "exec:git,ls"


@needs_bash
async def test_read_only_profile_denies_shell(tmp_path: Path) -> None:
    wm = WorkspaceManager(tmp_path, PermissionProfile.READ_ONLY)
    sess = Session(
        session_id=uuid.uuid4().hex[:8],
        cwd=tmp_path,
        permission_profile=PermissionProfile.READ_ONLY,
        workspace=wm,
    )
    await sess.next_event()
    manager = ShellManager()
    try:
        from codewright.tools.errors import RespondToModelError

        h = ShellHandler(manager)
        with pytest.raises(RespondToModelError, match="denied"):
            await h.handle(_inv(sess, "shell", {"command": "echo hi"}, tmp_path))
    finally:
        await manager.aclose()
        await sess.shutdown()


# ---------------------------------------------------------------------------
# Spec / dialect declaration
# ---------------------------------------------------------------------------


def test_spec_declares_dialect() -> None:
    manager = ShellManager()
    desc = ShellHandler(manager).spec().description
    assert manager.dialect.flavor in ("bash", "cmd")
    if manager.dialect.flavor == "bash":
        assert "bash" in desc.lower()
        assert manager.dialect.persistent is True
    else:
        assert "cmd.exe" in desc


async def test_blank_command_rejected(tmp_path: Path) -> None:
    from codewright.tools.errors import RespondToModelError

    sess = await _session(tmp_path)
    manager = ShellManager()
    try:
        h = ShellHandler(manager)
        with pytest.raises(RespondToModelError):
            await h.handle(_inv(sess, "shell", {"command": "   "}, tmp_path))
    finally:
        await manager.aclose()
        await sess.shutdown()


async def test_shell_rejects_background_ampersand(tmp_path: Path) -> None:
    # Finding 3: shell '&' orphans an untracked process; reject and steer the
    # model to background=true. The rejection happens before any spawn.
    from codewright.tools.errors import RespondToModelError

    sess = await _session(tmp_path)
    manager = ShellManager()
    try:
        h = ShellHandler(manager)
        with pytest.raises(RespondToModelError, match="background=true"):
            await h.handle(_inv(sess, "shell", {"command": "sleep 30 &"}, tmp_path))
        # '&&' is logical-and, not backgrounding — must NOT be rejected.
        assert manager.dialect.flavor in ("bash", "cmd")
    finally:
        await manager.aclose()
        await sess.shutdown()


# ---------------------------------------------------------------------------
# cmd.exe fallback is announced (Option 1: startup warning; Option 2: per-call)
# ---------------------------------------------------------------------------


def _force_cmd_dialect(manager: ShellManager) -> None:
    from codewright.tools.handlers._shell import ShellDialect

    manager._dialect = ShellDialect(
        flavor="cmd",
        path="",
        label="cmd.exe on Windows (bash not found — session state DISABLED)",
        persistent=False,
    )


def test_degraded_banner_only_in_cmd_mode() -> None:
    manager = ShellManager()
    h = ShellHandler(manager)
    # bash machine (or any persistent dialect): no banner.
    if manager.dialect.persistent:
        assert h._degraded_banner() == ""
    _force_cmd_dialect(manager)
    banner = h._degraded_banner()
    assert "degraded" in banner and "cmd.exe" in banner
    assert banner.endswith("\n")


async def test_cmd_fallback_warns_at_registration(tmp_path: Path) -> None:
    # Option 1: a bad/non-existent shell_path forces the cmd fallback, which must
    # surface one EvWarning at registration (cross-platform: the explicit-path
    # branch returns None on any OS).
    from codewright.cli import _register_builtin_tools
    from codewright.protocol import EvWarning

    sess = await _session(tmp_path)
    try:
        await _register_builtin_tools(
            sess, shell_path=str(tmp_path / "no_such_bash_exe")
        )
        warning = None
        for _ in range(5):
            ev = await asyncio.wait_for(sess.next_event(), timeout=2)
            if isinstance(ev.msg, EvWarning):
                warning = ev.msg
                break
        assert warning is not None, "expected an EvWarning for the cmd fallback"
        assert "cmd.exe" in warning.message
    finally:
        await sess.shutdown()


async def test_foreground_result_carries_banner_in_cmd_mode(tmp_path: Path) -> None:
    # Option 2: every shell result in cmd mode prefixes the degraded banner.
    sess = await _session(tmp_path)
    manager = ShellManager()
    _force_cmd_dialect(manager)
    try:
        h = ShellHandler(manager)
        result = await h.handle(_inv(sess, "shell", {"command": "echo hi"}, tmp_path))
        assert result.body.startswith("[shell degraded")
        assert "hi" in result.body
    finally:
        await manager.aclose()
        await sess.shutdown()
