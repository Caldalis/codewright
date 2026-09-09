"""submission_loop dispatch behavior + Session bootstrap event."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from codewright.agent.session import Session
from codewright.context import ContextManager
from codewright.llm.base import CanonicalMessage, LLMProvider, StreamEvent
from codewright.prompts.builder import PromptBuilder
from codewright.protocol import (
    EvError,
    EvSessionConfigured,
    EvShutdownComplete,
    EvTurnAborted,
    EvTurnCompleted,
    EvTurnStarted,
    EvWarning,
    OpCompact,
    OpInterrupt,
    OpOverrideTurnContext,
    OpShutdown,
    OpUserTurn,
    PermissionProfile,
    UserInputText,
)


async def _make_session() -> Session:
    return Session(
        session_id="t-session",
        cwd=Path("/tmp/cw"),
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
    )


# Fixed budget keeps tests fast; ASYNC109 prefers a constant over a kwarg here.
_DRAIN_DEADLINE_S = 1.0


async def _drain_until(session: Session, predicate):
    """Pull events until predicate(event) is True; returns that event."""

    async def _loop():
        while True:
            ev = await session.next_event()
            if predicate(ev):
                return ev

    async with asyncio.timeout(_DRAIN_DEADLINE_S):
        return await _loop()


class _InterruptThenCompleteProvider(LLMProvider):
    def __init__(self) -> None:
        self.calls = 0
        self.blocking_started = asyncio.Event()

    async def stream(  # type: ignore[override]
        self,
        messages: list[CanonicalMessage],
        tools: list,
        turn_context,
    ) -> AsyncIterator[StreamEvent]:
        del messages, tools, turn_context
        self.calls += 1
        if self.calls == 1:
            blocker = asyncio.Event()

            async def _blocked() -> AsyncIterator[StreamEvent]:
                self.blocking_started.set()
                await blocker.wait()
                yield StreamEvent(kind="message_completed")

            return _blocked()

        async def _complete() -> AsyncIterator[StreamEvent]:
            yield StreamEvent(kind="text_delta", text="next ok")

        return _complete()


async def test_session_emits_configured_event_on_startup():
    sess = await _make_session()
    try:
        ev = await asyncio.wait_for(sess.next_event(), timeout=1.0)
        assert isinstance(ev.msg, EvSessionConfigured)
        assert ev.msg.permission_profile == "workspace_write"
    finally:
        await sess.shutdown()


async def test_user_turn_emits_started_then_completed():
    sess = await _make_session()
    try:
        # Discard the bootstrap event.
        await sess.next_event()
        sub_id = await sess.submit(OpUserTurn(items=[UserInputText(text="hi")]))
        started = await _drain_until(sess, lambda e: isinstance(e.msg, EvTurnStarted))
        completed = await _drain_until(sess, lambda e: isinstance(e.msg, EvTurnCompleted))
        assert started.id == sub_id
        assert started.msg.turn_id == sub_id
        assert completed.msg.turn_id == sub_id
    finally:
        await sess.shutdown()


async def test_compact_emits_warning_stub():
    sess = await _make_session()
    try:
        await sess.next_event()  # bootstrap
        sub_id = await sess.submit(OpCompact())
        ev = await _drain_until(sess, lambda e: isinstance(e.msg, EvWarning))
        assert ev.id == sub_id
        assert "compact" in ev.msg.message.lower()
    finally:
        await sess.shutdown()


async def test_idle_interrupt_does_not_cancel_session_token_or_exit_loop():
    sess = await _make_session()
    try:
        await sess.next_event()  # bootstrap
        assert sess.cancellation_token.is_cancelled() is False
        await sess.submit(OpInterrupt())
        # Give the out-of-band handler a tick to process.
        await asyncio.sleep(0.05)
        assert sess.cancellation_token.is_cancelled() is False
        # Loop is still alive: an OpCompact still produces a Warning event.
        await sess.submit(OpCompact())
        await _drain_until(sess, lambda e: isinstance(e.msg, EvWarning))
    finally:
        await sess.shutdown()


async def test_interrupt_aborts_active_turn_without_poisoning_next_turn():
    provider = _InterruptThenCompleteProvider()
    sess = Session(
        session_id="t-interrupt",
        cwd=Path("/tmp/cw"),
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
        llm=provider,
        context_manager=ContextManager(max_context_tokens=4000),
        prompt_builder=PromptBuilder("SYS"),
    )
    try:
        await sess.next_event()  # bootstrap
        await sess.submit(OpUserTurn(items=[UserInputText(text="first")]))
        await _drain_until(sess, lambda e: isinstance(e.msg, EvTurnStarted))
        await asyncio.wait_for(provider.blocking_started.wait(), timeout=1.0)

        await sess.submit(OpInterrupt())
        aborted = await _drain_until(sess, lambda e: isinstance(e.msg, EvTurnAborted))
        assert aborted.msg.reason == "interrupted"
        assert sess.cancellation_token.is_cancelled() is False

        await sess.submit(OpUserTurn(items=[UserInputText(text="second")]))
        completed = await _drain_until(sess, lambda e: isinstance(e.msg, EvTurnCompleted))
        assert completed.msg.last_agent_message == "next ok"
    finally:
        await sess.shutdown()


async def test_shutdown_cancels_active_turn_without_timeout():
    provider = _InterruptThenCompleteProvider()
    sess = Session(
        session_id="t-shutdown-active",
        cwd=Path("/tmp/cw"),
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
        llm=provider,
        context_manager=ContextManager(max_context_tokens=4000),
        prompt_builder=PromptBuilder("SYS"),
    )
    try:
        await sess.next_event()  # bootstrap
        await sess.submit(OpUserTurn(items=[UserInputText(text="first")]))
        await _drain_until(sess, lambda e: isinstance(e.msg, EvTurnStarted))
        await asyncio.wait_for(provider.blocking_started.wait(), timeout=1.0)

        await asyncio.wait_for(sess.shutdown(), timeout=1.0)
        assert sess.cancellation_token.is_cancelled() is True
    finally:
        await sess.shutdown()


async def test_shutdown_emits_shutdown_complete_and_exits():
    sess = await _make_session()
    try:
        await sess.next_event()  # bootstrap
        await sess.submit(OpShutdown())
        ev = await _drain_until(sess, lambda e: isinstance(e.msg, EvShutdownComplete))
        assert isinstance(ev.msg, EvShutdownComplete)
        # Loop task should be done shortly after the event.
        await asyncio.wait_for(sess._loop_task, timeout=1.0)
        assert sess._loop_task.done()
    finally:
        # shutdown() is idempotent even after the loop already exited.
        await sess.shutdown()


async def test_override_turn_context_emits_error_instead_of_crashing_loop():
    """D-1-004: NotImplementedError raised by the handler is caught by the
    outer try/except in submission_loop and surfaced as EvError; the loop
    stays alive so the session does not become unusable."""
    sess = await _make_session()
    try:
        await sess.next_event()  # bootstrap
        sub_id = await sess.submit(OpOverrideTurnContext(model="foo"))
        ev = await _drain_until(sess, lambda e: isinstance(e.msg, EvError))
        assert ev.id == sub_id
        assert "override_turn_context" in ev.msg.message.lower()
        # And the loop is still serving — issue another op and observe a reply.
        await sess.submit(OpCompact())
        await _drain_until(sess, lambda e: isinstance(e.msg, EvWarning))
    finally:
        await sess.shutdown()


async def test_shutdown_is_idempotent():
    sess = await _make_session()
    await sess.shutdown()
    # Second call must not raise.
    await sess.shutdown()


@pytest.mark.parametrize("payload", ["abc", "🙂 with emoji", ""])
async def test_user_turn_with_various_text_payloads(payload: str):
    sess = await _make_session()
    try:
        await sess.next_event()  # bootstrap
        await sess.submit(OpUserTurn(items=[UserInputText(text=payload)]))
        await _drain_until(sess, lambda e: isinstance(e.msg, EvTurnCompleted))
    finally:
        await sess.shutdown()
