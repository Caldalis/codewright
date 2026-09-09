"""End-to-end approval round-trip: tool-side request_approval ↔ frontend Op response.

This is _SHARED.md I3 in action: no callbacks, only events + ops.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from codewright.agent.session import Session
from codewright.context import ContextManager
from codewright.llm.base import (
    CanonicalMessage,
    LLMProvider,
    StreamEvent,
    ToolCallBlock,
)
from codewright.prompts.builder import PromptBuilder
from codewright.protocol import (
    EvExecApprovalRequest,
    EvPatchApprovalRequest,
    EvSessionConfigured,
    EvTurnAborted,
    EvTurnCompleted,
    EvWarning,
    OpExecApprovalResponse,
    OpInterrupt,
    OpPatchApprovalResponse,
    OpUserTurn,
    PendingAction,
    PermissionProfile,
    ReviewDecision,
    UserInputText,
)
from codewright.tools.handler import ToolHandler
from codewright.tools.invocation import ToolInvocation
from codewright.tools.result import ToolResult
from codewright.tools.spec import ToolSpec


class _ApprovalToolHandler(ToolHandler):
    @property
    def tool_name(self) -> str:
        return "needs_approval"

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_name,
            description="Test tool that waits for frontend approval.",
            parameters={},
            requires_approval=True,
        )

    async def handle(self, invocation: ToolInvocation) -> ToolResult:
        action = PendingAction(
            action_id="approval-tool",
            kind="exec",
            summary="approve test tool",
        )
        decision = await invocation.session.request_approval(action)
        return ToolResult(success=True, body=f"decision={decision.value}")


class _ScriptedProvider(LLMProvider):
    def __init__(self, scripts: list[list[StreamEvent]]) -> None:
        self._scripts = list(scripts)

    async def stream(  # type: ignore[override]
        self,
        messages: list[CanonicalMessage],
        tools: list,
        turn_context,
    ) -> AsyncIterator[StreamEvent]:
        del messages, tools, turn_context
        script = self._scripts.pop(0) if self._scripts else []

        async def _gen() -> AsyncIterator[StreamEvent]:
            for ev in script:
                yield ev

        return _gen()


async def _make_session() -> Session:
    sess = Session(
        session_id="approval-test",
        cwd=Path("/tmp/cw"),
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
    )
    # discard bootstrap EvSessionConfigured
    boot = await sess.next_event()
    assert isinstance(boot.msg, EvSessionConfigured)
    return sess


async def test_exec_approval_resolves_to_user_decision():
    sess = await _make_session()
    try:
        action = PendingAction(
            action_id="a1",
            kind="exec",
            summary="rm -rf /tmp/x",
            details={"argv": ["rm", "-rf", "/tmp/x"]},
        )

        # Tool task: suspends on approval, then we'll observe its decision.
        handler_task = asyncio.create_task(sess.request_approval(action))

        # Frontend: read the approval-request event, extract request_id.
        ev = await asyncio.wait_for(sess.next_event(), timeout=1.0)
        assert isinstance(ev.msg, EvExecApprovalRequest)
        request_id = ev.msg.request_id

        # Frontend: submit user's approval back.
        await sess.submit(OpExecApprovalResponse(
            request_id=request_id, decision=ReviewDecision.APPROVED
        ))

        decision = await asyncio.wait_for(handler_task, timeout=1.0)
        assert decision is ReviewDecision.APPROVED
    finally:
        await sess.shutdown()


async def test_exec_approval_denied():
    sess = await _make_session()
    try:
        action = PendingAction(action_id="a", kind="exec", summary="x")
        handler_task = asyncio.create_task(sess.request_approval(action))
        ev = await asyncio.wait_for(sess.next_event(), timeout=1.0)
        assert isinstance(ev.msg, EvExecApprovalRequest)
        await sess.submit(OpExecApprovalResponse(
            request_id=ev.msg.request_id, decision=ReviewDecision.DENIED
        ))
        assert (await handler_task) is ReviewDecision.DENIED
    finally:
        await sess.shutdown()


async def test_patch_approval_emits_patch_event_and_resolves():
    sess = await _make_session()
    try:
        action = PendingAction(action_id="a", kind="patch", summary="update README")
        handler_task = asyncio.create_task(sess.request_approval(action))
        ev = await asyncio.wait_for(sess.next_event(), timeout=1.0)
        assert isinstance(ev.msg, EvPatchApprovalRequest)
        await sess.submit(OpPatchApprovalResponse(
            request_id=ev.msg.request_id, decision=ReviewDecision.APPROVED_FOR_SESSION
        ))
        assert (await handler_task) is ReviewDecision.APPROVED_FOR_SESSION
    finally:
        await sess.shutdown()


async def test_approval_response_bypasses_busy_submission_loop():
    provider = _ScriptedProvider(
        [
            [
                StreamEvent(
                    kind="tool_call_completed",
                    tool_call=ToolCallBlock(
                        call_id="c1",
                        tool_name="needs_approval",
                        arguments_json="{}",
                    ),
                )
            ],
            [StreamEvent(kind="text_delta", text="done")],
        ]
    )
    sess = Session(
        session_id="approval-busy-loop-test",
        cwd=Path("/tmp/cw"),
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
        llm=provider,
        context_manager=ContextManager(max_context_tokens=4000),
        prompt_builder=PromptBuilder("SYS"),
    )
    sess.tool_registry.register(_ApprovalToolHandler())
    try:
        boot = await sess.next_event()
        assert isinstance(boot.msg, EvSessionConfigured)

        await sess.submit(OpUserTurn(items=[UserInputText(text="run tool")]))

        request_id = ""
        for _ in range(10):
            ev = await asyncio.wait_for(sess.next_event(), timeout=1.0)
            if isinstance(ev.msg, EvExecApprovalRequest):
                request_id = ev.msg.request_id
                break
        assert request_id

        await sess.submit(
            OpExecApprovalResponse(
                request_id=request_id,
                decision=ReviewDecision.APPROVED,
            )
        )

        completed = False
        for _ in range(20):
            ev = await asyncio.wait_for(sess.next_event(), timeout=1.0)
            if isinstance(ev.msg, EvTurnCompleted):
                completed = True
                break
        assert completed
    finally:
        await sess.shutdown()


async def test_interrupt_during_approval_aborts_turn_without_poisoning_session():
    provider = _ScriptedProvider(
        [
            [
                StreamEvent(
                    kind="tool_call_completed",
                    tool_call=ToolCallBlock(
                        call_id="c1",
                        tool_name="needs_approval",
                        arguments_json="{}",
                    ),
                )
            ],
            [StreamEvent(kind="text_delta", text="next ok")],
        ]
    )
    sess = Session(
        session_id="approval-interrupt-test",
        cwd=Path("/tmp/cw"),
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
        llm=provider,
        context_manager=ContextManager(max_context_tokens=4000),
        prompt_builder=PromptBuilder("SYS"),
    )
    sess.tool_registry.register(_ApprovalToolHandler())
    try:
        boot = await sess.next_event()
        assert isinstance(boot.msg, EvSessionConfigured)

        await sess.submit(OpUserTurn(items=[UserInputText(text="run tool")]))

        request_id = ""
        for _ in range(10):
            ev = await asyncio.wait_for(sess.next_event(), timeout=1.0)
            if isinstance(ev.msg, EvExecApprovalRequest):
                request_id = ev.msg.request_id
                break
        assert request_id

        await sess.submit(OpInterrupt())

        aborted = False
        for _ in range(20):
            ev = await asyncio.wait_for(sess.next_event(), timeout=1.0)
            if isinstance(ev.msg, EvTurnAborted):
                assert ev.msg.reason == "interrupted"
                aborted = True
                break
        assert aborted
        assert sess.cancellation_token.is_cancelled() is False

        await sess.submit(OpUserTurn(items=[UserInputText(text="next")]))

        completed = False
        for _ in range(20):
            ev = await asyncio.wait_for(sess.next_event(), timeout=1.0)
            if isinstance(ev.msg, EvTurnCompleted):
                assert ev.msg.last_agent_message == "next ok"
                completed = True
                break
        assert completed
    finally:
        await sess.shutdown()


async def test_late_approval_for_unknown_request_emits_warning():
    """If a stale Op.ApprovalResponse arrives after the future is gone, the
    loop logs a Warning instead of crashing."""
    sess = await _make_session()
    try:
        await sess.submit(OpExecApprovalResponse(
            request_id="does-not-exist", decision=ReviewDecision.APPROVED
        ))
        ev = await asyncio.wait_for(sess.next_event(), timeout=1.0)
        assert isinstance(ev.msg, EvWarning)
        assert "does-not-exist" in ev.msg.message
    finally:
        await sess.shutdown()


async def test_cancellation_during_approval_raises_cancelled_error():
    sess = await _make_session()
    try:
        action = PendingAction(action_id="a", kind="exec", summary="long wait")
        handler_task = asyncio.create_task(sess.request_approval(action))
        # Wait for the approval event so the handler is genuinely blocked.
        ev = await asyncio.wait_for(sess.next_event(), timeout=1.0)
        assert isinstance(ev.msg, EvExecApprovalRequest)

        # Simulate session-level cancellation, such as programmatic shutdown.
        sess.cancellation_token.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(handler_task, timeout=1.0)
    finally:
        await sess.shutdown()


async def test_unsupported_network_approval_kind_raises():
    sess = await _make_session()
    try:
        action = PendingAction(action_id="a", kind="network", summary="https://x")
        with pytest.raises(NotImplementedError):
            await sess.request_approval(action)
    finally:
        await sess.shutdown()
