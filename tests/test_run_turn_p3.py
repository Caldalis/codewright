"""Extension to test_run_turn: mock LLM emits a tool_call and run_turn
dispatches it through ToolExecutor.
"""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from codewright.agent.cancellation import CancellationToken
from codewright.agent.session import Session
from codewright.agent.turn import run_turn
from codewright.agent.turn_context import TurnContext
from codewright.context.manager import ContextManager
from codewright.llm.base import LLMProvider, StreamEvent, TokenUsage, ToolCallBlock
from codewright.prompts.builder import PromptBuilder
from codewright.protocol import AskForApproval, PermissionProfile
from codewright.tools.handlers.run_shell import RunShellHandler
from codewright.workspace import WorkspaceManager


class _ToolThenTextProvider(LLMProvider):
    """First sample: emit a single run_shell tool_call. Second sample: emit
    'all done' as final text."""

    def __init__(self, command: list[str]) -> None:
        self._command = command
        self._sample = 0

    async def stream(  # type: ignore[override]
        self, messages, tools, turn_context
    ) -> AsyncIterator[StreamEvent]:
        async def gen() -> AsyncIterator[StreamEvent]:
            self._sample += 1
            if self._sample == 1:
                yield StreamEvent(
                    kind="tool_call_completed",
                    tool_call=ToolCallBlock(
                        call_id="c1",
                        tool_name="run_shell",
                        arguments_json=json.dumps(
                            {"command": self._command, "timeout_ms": 5000}
                        ),
                    ),
                )
                yield StreamEvent(
                    kind="usage", usage=TokenUsage(input=10, output=2, total=12)
                )
            else:
                yield StreamEvent(kind="text_delta", text="all done")
                yield StreamEvent(
                    kind="usage", usage=TokenUsage(input=12, output=4, total=16)
                )

        return gen()


@pytest.mark.asyncio
async def test_run_turn_dispatches_real_tool(tmp_path: Path) -> None:
    wm = WorkspaceManager(tmp_path, PermissionProfile.WORKSPACE_WRITE)
    sess = Session(
        session_id="s",
        cwd=tmp_path,
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
        llm=_ToolThenTextProvider([sys.executable, "-c", "print('hello shell')"]),
        context_manager=ContextManager(),
        prompt_builder=PromptBuilder("you are codewright"),
        workspace=wm,
    )
    sess.tool_registry.register(RunShellHandler())
    try:
        await sess.next_event()  # SessionConfigured
        tc = TurnContext(
            turn_id="t1",
            cwd=tmp_path,
            model="mock",
            permission_profile=PermissionProfile.WORKSPACE_WRITE,
            approval_policy=AskForApproval.NEVER,
            cancellation_token=CancellationToken(),
        )
        last = await run_turn(sess, tc, "do it", sub_id="sub")
        assert last == "all done"
        snap = sess.context.snapshot()
        # We expect at least: user, assistant-with-call, tool result, assistant final.
        roles = [m.role for m in snap]
        assert "tool" in roles
        tool_msg = next(m for m in snap if m.role == "tool")
        assert "hello shell" in str(tool_msg.content)
    finally:
        await sess.shutdown()
