"""TUI smoke: import + event dispatch without launching a real terminal."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path

import pytest
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from codewright.agent.session import Session
from codewright.context import ContextManager
from codewright.llm.base import CanonicalMessage, LLMProvider, StreamEvent
from codewright.prompts.builder import PromptBuilder
from codewright.protocol import (
    EvAgentMessage,
    EvSessionConfigured,
    EvTokenCount,
    EvTurnAborted,
    EvTurnCompleted,
    EvTurnStarted,
    EvWarning,
    OpShutdown,
    PendingAction,
    PermissionProfile,
    PlanItem,
    PlanItemStatus,
    ReviewDecision,
)
from codewright.tui.app import TuiApp
from codewright.tui.status_bar import StatusBarState


class _SilentProvider(LLMProvider):
    async def stream(  # type: ignore[override]
        self,
        messages: list[CanonicalMessage],
        tools: list,
        turn_context,
    ) -> AsyncIterator[StreamEvent]:
        async def _gen() -> AsyncIterator[StreamEvent]:
            if False:
                yield  # type: ignore[unreachable]

        return _gen()


@pytest.mark.asyncio
async def test_tui_app_import_and_instantiate():
    from io import StringIO

    from rich.console import Console

    session = Session(
        session_id="s-tui",
        cwd=Path("/tmp"),
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
        llm=_SilentProvider(),
        context_manager=ContextManager(max_context_tokens=4000),
        prompt_builder=PromptBuilder("SYS"),
    )
    try:
        app = TuiApp(session, console=Console(file=StringIO()))
        assert app.session is session
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_tui_queues_multiple_approval_requests():
    from io import StringIO

    from rich.console import Console

    session = Session(
        session_id="s-tui",
        cwd=Path("/tmp"),
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
        llm=_SilentProvider(),
        context_manager=ContextManager(max_context_tokens=4000),
        prompt_builder=PromptBuilder("SYS"),
    )
    try:
        app = TuiApp(session, console=Console(file=StringIO()))
        first = PendingAction("a1", "exec", "first", {})
        second = PendingAction("a2", "patch", "second", {})

        app._start_approval("r1", first, is_exec=True)
        app._start_approval("r2", second, is_exec=False)
        assert app.has_pending_approval
        assert app._status.pending_approvals == 2
        assert app._pending_approval is not None
        assert app._pending_approval.action.summary == "first"

        app.submit_approval(ReviewDecision.DENIED)
        await asyncio.sleep(0)
        assert app.has_pending_approval
        assert app._status.pending_approvals == 1
        assert app._pending_approval is not None
        assert app._pending_approval.action.summary == "second"

        app.submit_approval(ReviewDecision.DENIED)
        await asyncio.sleep(0)
        assert not app.has_pending_approval
        assert app._status.pending_approvals == 0
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_tui_clears_pending_approvals_on_turn_abort():
    from io import StringIO

    from rich.console import Console

    session = Session(
        session_id="s-tui",
        cwd=Path("/tmp"),
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
        llm=_SilentProvider(),
        context_manager=ContextManager(max_context_tokens=4000),
        prompt_builder=PromptBuilder("SYS"),
    )
    try:
        app = TuiApp(session, console=Console(file=StringIO()))
        app._start_approval(
            "r1", PendingAction("a1", "exec", "first", {}), is_exec=True
        )
        app._start_approval(
            "r2", PendingAction("a2", "patch", "second", {}), is_exec=False
        )

        await app._handle_event(EvTurnAborted(turn_id="t1", reason="interrupted"))

        assert not app.has_pending_approval
        assert app._approval_queue == []
        assert app._status.pending_approvals == 0
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_tui_application_runs_and_exits_headless():
    session = Session(
        session_id="s-tui",
        cwd=Path("/tmp"),
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
        llm=_SilentProvider(),
        context_manager=ContextManager(max_context_tokens=4000),
        prompt_builder=PromptBuilder("SYS"),
    )
    pipe_cm = create_pipe_input()
    pipe_input = pipe_cm.__enter__()
    try:
        app = TuiApp(
            session,
            pt_input=pipe_input,
            pt_output=DummyOutput(),
        )
        task = asyncio.create_task(app.run())
        await asyncio.sleep(0.05)
        await session.submit(OpShutdown())
        await asyncio.wait_for(task, timeout=3)
        assert app._application is None
    finally:
        pipe_cm.__exit__(None, None, None)
        await session.shutdown()


@pytest.mark.asyncio
async def test_tui_prompt_input_submits_user_turn():
    session = Session(
        session_id="s-tui",
        cwd=Path("/tmp"),
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
        llm=_SilentProvider(),
        context_manager=ContextManager(max_context_tokens=4000),
        prompt_builder=PromptBuilder("SYS"),
    )
    submitted: list[str] = []
    original_submit = session.submit

    async def _spy(op):
        submitted.append(type(op).__name__)
        return await original_submit(op)

    session.submit = _spy  # type: ignore[method-assign]

    pipe_cm = create_pipe_input()
    pipe_input = pipe_cm.__enter__()
    try:
        app = TuiApp(session, pt_input=pipe_input, pt_output=DummyOutput())
        task = asyncio.create_task(app.run())
        await asyncio.sleep(0.1)
        # Type a prompt and press Enter (\r) — the inline REPL must turn it into
        # an OpUserTurn, and the startup banner must have been recorded.
        pipe_input.send_text("hello there\r")
        await asyncio.sleep(0.1)
        await session.submit(OpShutdown())
        await asyncio.wait_for(task, timeout=3)
        assert "OpUserTurn" in submitted
        assert any("CODEWRIGHT" in item.plain for item in app._history)
    finally:
        pipe_cm.__exit__(None, None, None)
        await session.shutdown()


@pytest.mark.asyncio
async def test_tui_background_task_errors_are_reported():
    session = Session(
        session_id="s-tui",
        cwd=Path("/tmp"),
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
        llm=_SilentProvider(),
        context_manager=ContextManager(max_context_tokens=4000),
        prompt_builder=PromptBuilder("SYS"),
    )

    async def boom():
        raise RuntimeError("submit failed")

    try:
        app = TuiApp(session)
        task = app._spawn(boom())
        with suppress(RuntimeError):
            await task
        assert any("submit failed" in item.plain for item in app._history)
    finally:
        await session.shutdown()


def test_status_bar_renders():
    bar = StatusBarState(model="m", cwd="/x", turn_state="idle")
    rendered = bar.render()
    assert "model=m" in rendered.plain
    assert "state=idle" in rendered.plain


_ = asyncio  # silence unused-import warning when only async tests run below


@pytest.mark.asyncio
async def test_handle_event_updates_history_and_status_for_each_kind():
    """Walk every EventMsg branch the TUI cares about.

    This is a *smoke* test — we don't render to a real terminal, we just
    ensure the dispatcher doesn't crash and updates state predictably.
    """
    from io import StringIO

    from rich.console import Console

    session = Session(
        session_id="s-tui",
        cwd=Path("/tmp"),
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
        llm=_SilentProvider(),
        context_manager=ContextManager(max_context_tokens=4000),
        prompt_builder=PromptBuilder("SYS"),
    )
    try:
        app = TuiApp(session, console=Console(file=StringIO()))
        await app._handle_event(
            EvSessionConfigured(
                session_id="s-tui", model="m", cwd="/tmp", permission_profile="workspace_write"
            )
        )
        await app._handle_event(EvTurnStarted(turn_id="t1"))
        await app._handle_event(EvAgentMessage(content="hello"))
        await app._handle_event(EvTokenCount(input=10, output=5, total=15))
        await app._handle_event(EvWarning(message="warn here"))
        await app._handle_event(
            EvTurnCompleted(turn_id="t1", last_agent_message="hello")
        )
        from codewright.protocol import EvPlanUpdate

        await app._handle_event(
            EvPlanUpdate(
                plan=[PlanItem(step="x", status=PlanItemStatus.PENDING)],
                explanation="trying",
            )
        )
        assert app._status.input_tokens == 10
        assert app._status.output_tokens == 5
        assert app._status.turn_state == "idle"
        # History got messages.
        assert len(app._history) >= 3
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_tui_reports_compaction_progress():
    """Auto-compaction must surface as a visible front-end signal."""
    from io import StringIO

    from rich.console import Console

    from codewright.protocol import EvCompactionCompleted, EvCompactionStarted

    session = Session(
        session_id="s-tui",
        cwd=Path("/tmp"),
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
        llm=_SilentProvider(),
        context_manager=ContextManager(max_context_tokens=4000),
        prompt_builder=PromptBuilder("SYS"),
    )
    try:
        app = TuiApp(session, console=Console(file=StringIO()))

        await app._handle_event(
            EvCompactionStarted(reason="auto", tokens_before=120_000)
        )
        assert app._status.turn_state == "compacting"
        assert any("Compacting" in item.plain for item in app._history)

        await app._handle_event(
            EvCompactionCompleted(tokens_before=120_000, tokens_after=38_000)
        )
        assert app._status.turn_state == "running"
        assert any("Compacted" in item.plain for item in app._history)
    finally:
        await session.shutdown()
