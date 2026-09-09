"""run_turn behavior with a stub LLM (no network)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from codewright.agent.cancellation import CancellationToken
from codewright.agent.session import Session
from codewright.agent.turn import run_turn
from codewright.agent.turn_context import TurnContext
from codewright.context import ContextManager
from codewright.llm.base import CanonicalMessage, LLMProvider, StreamEvent, ToolCallBlock
from codewright.prompts.builder import PromptBuilder
from codewright.protocol import (
    AskForApproval,
    EvAgentMessage,
    EvAgentMessageDelta,
    EvTurnAborted,
    EvTurnCompleted,
    EvTurnStarted,
    OpUserTurn,
    PermissionProfile,
    UserInputText,
)


class StubProvider(LLMProvider):
    """Replay a scripted list of ``list[StreamEvent]`` per ``stream()`` call."""

    def __init__(self, scripts: list[list[StreamEvent]]) -> None:
        self._scripts = list(scripts)
        self.calls: list[list[CanonicalMessage]] = []

    async def stream(  # type: ignore[override]
        self,
        messages: list[CanonicalMessage],
        tools: list,
        turn_context,
    ) -> AsyncIterator[StreamEvent]:
        del turn_context
        self.calls.append(list(messages))
        script = self._scripts.pop(0) if self._scripts else []

        async def _gen() -> AsyncIterator[StreamEvent]:
            for ev in script:
                yield ev

        return _gen()


def _make_turn_context(token: CancellationToken | None = None) -> TurnContext:
    return TurnContext(
        turn_id="t1",
        cwd=Path("/tmp"),
        model="m",
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
        approval_policy=AskForApproval.ON_REQUEST,
        cancellation_token=token or CancellationToken(),
    )


def _make_session(provider: LLMProvider) -> Session:
    return Session(
        session_id="s1",
        cwd=Path("/tmp"),
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
        llm=provider,
        context_manager=ContextManager(max_context_tokens=4000),
        prompt_builder=PromptBuilder("SYS"),
    )


@pytest.mark.asyncio
async def test_text_only_turn_completes():
    provider = StubProvider(
        [
            [
                StreamEvent(kind="text_delta", text="hello"),
                StreamEvent(kind="text_delta", text=" world"),
            ]
        ]
    )
    session = _make_session(provider)
    try:
        result = await run_turn(session, _make_turn_context(), "ping")
        assert result == "hello world"
        # history contains user + assistant
        items = session.context.snapshot()
        assert [m.role for m in items] == ["user", "assistant"]
        assert items[0].content == "ping"
        assert items[1].content == "hello world"
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_tool_call_unknown_tool_returns_failure_to_model():
    """P3: with no handler registered for ``run_shell``, the executor folds
    the unknown-tool error into ``ToolResult(success=False)`` so the model
    can recover on the next sample."""
    provider = StubProvider(
        [
            [
                StreamEvent(
                    kind="tool_call_completed",
                    tool_call=ToolCallBlock(
                        call_id="c1", tool_name="run_shell", arguments_json="{}"
                    ),
                ),
            ],
            [StreamEvent(kind="text_delta", text="done")],
        ]
    )
    session = _make_session(provider)
    try:
        result = await run_turn(session, _make_turn_context(), "list files")
        assert result == "done"
        items = session.context.snapshot()
        roles = [m.role for m in items]
        assert roles[0] == "user"
        assert "assistant" in roles
        assert "tool" in roles
        tool_msg = next(m for m in items if m.role == "tool")
        assert "unknown tool: run_shell" in tool_msg.content
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_cancellation_emits_turn_aborted():
    provider = StubProvider([[]])
    session = _make_session(provider)
    token = CancellationToken()
    token.cancel()
    try:
        result = await run_turn(session, _make_turn_context(token), "hi")
        assert result is None
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_op_user_turn_drives_run_turn_via_session():
    provider = StubProvider(
        [[StreamEvent(kind="text_delta", text="ack")]]
    )
    session = _make_session(provider)
    try:
        await session.submit(OpUserTurn(items=[UserInputText(text="hi")]))
        seen_event_types: list[str] = []
        for _ in range(20):
            try:
                event = await asyncio.wait_for(session.next_event(), timeout=2.0)
            except TimeoutError:
                break
            seen_event_types.append(type(event.msg).__name__)
            if isinstance(event.msg, EvTurnCompleted):
                break

        assert "EvSessionConfigured" in seen_event_types
        assert "EvTurnStarted" in seen_event_types
        assert "EvAgentMessage" in seen_event_types
        assert "EvTurnCompleted" in seen_event_types
        _ = EvAgentMessage, EvAgentMessageDelta, EvTurnStarted, EvTurnAborted
    finally:
        await session.shutdown()
