"""End-to-end equivalents (Tier B fallbacks) using a mock LLM.

These cover the Tier B commands in ``_STRUCTURAL_CHECKS.md §6`` for which a
real API key is not available in this environment. See ``VALIDATION_REPORT.md``
for the mapping of e2e command -> mock test.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from codewright.agent.cancellation import CancellationToken
from codewright.agent.resume import resume_session
from codewright.agent.session import Session
from codewright.agent.turn import run_turn
from codewright.agent.turn_context import TurnContext
from codewright.context import ContextManager
from codewright.llm import ChatCompletionsAdapter
from codewright.llm.base import (
    CanonicalMessage,
    LLMProvider,
    StreamEvent,
    ToolCallBlock,
)
from codewright.persistence.rollout import SessionMeta
from codewright.persistence.session_store import SessionStore
from codewright.prompts.builder import PromptBuilder
from codewright.protocol import (
    AskForApproval,
    EvTurnCompleted,
    OpUserTurn,
    PermissionProfile,
    UserInputText,
)
from codewright.tools.handlers import RunShellHandler
from codewright.workspace.manager import WorkspaceManager


class ScriptedProvider(LLMProvider):
    """LLMProvider yielding a pre-baked list of events per turn."""

    def __init__(self, scripts: list[list[StreamEvent]]) -> None:
        self._scripts = list(scripts)
        self.calls: list[list[CanonicalMessage]] = []

    async def stream(  # type: ignore[override]
        self,
        messages: list[CanonicalMessage],
        tools: list,
        turn_context,
    ) -> AsyncIterator[StreamEvent]:
        del tools, turn_context
        self.calls.append(list(messages))
        script = self._scripts.pop(0) if self._scripts else []

        async def _gen() -> AsyncIterator[StreamEvent]:
            for ev in script:
                yield ev

        return _gen()


def _turn_context(token: CancellationToken | None = None) -> TurnContext:
    return TurnContext(
        turn_id="tx",
        cwd=Path("/tmp"),
        model="m",
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
        approval_policy=AskForApproval.ON_REQUEST,
        cancellation_token=token or CancellationToken(),
    )


# 1. summarize README — Tier B #3 ---------------------------------------------


@pytest.mark.asyncio
async def test_summarize_readme(tmp_path: Path):
    readme = tmp_path / "README.md"
    readme.write_text("Codewright is a Codex-inspired Python coding agent.\n")
    provider = ScriptedProvider(
        [[StreamEvent(kind="text_delta",
                       text="Codewright is a Codex-inspired Python coding agent.")]]
    )
    session = Session(
        session_id="s-readme",
        cwd=tmp_path,
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
        llm=provider,
        context_manager=ContextManager(max_context_tokens=4000),
        prompt_builder=PromptBuilder("SYS"),
    )
    try:
        result = await run_turn(session, _turn_context(), "summarize README.md")
        assert "Codex-inspired" in (result or "")
    finally:
        await session.shutdown()


# 2. self-correction — Tier B #4 ----------------------------------------------


@pytest.mark.asyncio
async def test_self_correction_run_shell_then_fix(tmp_path: Path):
    """First turn invokes a non-existent shell command, gets an error, then
    the next sample is a clean text completion (mirrors the "fix the bug"
    flow without needing real code edits)."""
    missing_command = "codewright_command_that_should_not_exist_12345"
    provider = ScriptedProvider(
        [
            [StreamEvent(kind="tool_call_completed",
                          tool_call=ToolCallBlock(call_id="c1",
                                                   tool_name="run_shell",
                                                   arguments_json=json.dumps(
                                                       {"command": [missing_command], "cwd": str(tmp_path)})))],
            [StreamEvent(kind="text_delta", text="done after recovery")],
        ]
    )
    session = Session(
        session_id="s-fix",
        cwd=tmp_path,
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
        llm=provider,
        context_manager=ContextManager(max_context_tokens=4000),
        prompt_builder=PromptBuilder("SYS"),
        workspace=WorkspaceManager(tmp_path, PermissionProfile.WORKSPACE_WRITE),
    )
    session.tool_registry.register(RunShellHandler())
    try:
        result = await run_turn(session, _turn_context(), "fix the bug")
        assert "done after recovery" in (result or "")
    finally:
        await session.shutdown()


# 3. auto-compact — Tier B #6 -------------------------------------------------


@pytest.mark.asyncio
async def test_auto_compact_triggered_under_tight_budget(tmp_path: Path):
    class _Summarizer:
        async def summarize(
            self, messages: list[CanonicalMessage], compact_prompt: str
        ) -> str:
            return "summary of prior conversation"

    # Compose a tiny budget so even a small message trips the threshold.
    cm = ContextManager(max_context_tokens=200, compact_threshold=0.5)
    # Pre-populate history with a long user message so should_compact() flips.
    cm.append(CanonicalMessage(role="user", content="x" * 500))
    provider = ScriptedProvider(
        [[StreamEvent(kind="text_delta", text="after compact")]]
    )
    session = Session(
        session_id="s-compact",
        cwd=tmp_path,
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
        llm=provider,
        context_manager=cm,
        prompt_builder=PromptBuilder("SYS"),
        summarizer=_Summarizer(),
    )
    try:
        result = await run_turn(session, _turn_context(), "more info")
        assert result == "after compact"
    finally:
        await session.shutdown()


# 4. session resume — Tier B #7 -----------------------------------------------


@pytest.mark.asyncio
async def test_session_resume_replays_history(tmp_path: Path):
    """Run a turn, persist, resume, ensure history is restored."""
    workspace = tmp_path
    provider = ScriptedProvider(
        [[StreamEvent(kind="text_delta", text="first answer")]]
    )
    store = SessionStore(workspace)
    sid = "s-resume"
    rollout = await store.create_recorder(
        SessionMeta(
            session_id=sid, cwd=str(workspace), model="m",
            permission_profile="workspace_write", start_time=0.0,
        )
    )
    sess = Session(
        session_id=sid,
        cwd=workspace,
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
        llm=provider,
        context_manager=ContextManager(max_context_tokens=4000),
        prompt_builder=PromptBuilder("SYS"),
        rollout=rollout,
    )
    try:
        await sess.submit(OpUserTurn(items=[UserInputText(text="hi")]))
        async for ev in _drain_until(sess, EvTurnCompleted):
            del ev
    finally:
        await sess.shutdown()

    # Now resume — history should contain the prior user + assistant.
    provider2 = ScriptedProvider([])
    resumed = await resume_session(
        workspace_root=workspace,
        session_id=sid,
        llm=provider2,
        prompt_builder=PromptBuilder("SYS"),
    )
    try:
        roles = [m.role for m in resumed.context.snapshot()]
        assert "user" in roles
        assert "assistant" in roles
    finally:
        await resumed.shutdown()


# 5. cross-provider base_url — Tier B #11 -------------------------------------


@pytest.mark.asyncio
async def test_cross_provider_base_url_carried_through_transport():
    """When the user passes --provider-base-url, the adapter wires that exact
    URL into the HTTP call (verified by httpx.MockTransport)."""
    seen_urls: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        # Minimal valid SSE-ish body so the adapter terminates cleanly.
        body = (
            b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
            b"data: [DONE]\n\n"
        )
        return httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )

    transport = httpx.MockTransport(_handler)
    client = httpx.AsyncClient(transport=transport)
    adapter = ChatCompletionsAdapter(
        api_key="sk-test",
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        http_client=client,
    )
    stream = adapter.stream(
        [CanonicalMessage(role="user", content="hi")],
        tools=[],
        turn_context=_turn_context(),
    )
    if hasattr(stream, "__aiter__"):
        ait = stream
    else:
        ait = await stream
    async for _ in ait:
        pass
    assert any("api.deepseek.com/v1" in u for u in seen_urls), seen_urls


# helper -------------------------------------------------------------------


async def _drain_until(session: Session, event_type) -> AsyncIterator[object]:
    for _ in range(50):
        try:
            ev = await asyncio.wait_for(session.next_event(), timeout=2.0)
        except TimeoutError:
            return
        yield ev
        if isinstance(ev.msg, event_type):
            return
