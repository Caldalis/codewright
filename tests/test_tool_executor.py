"""ToolExecutor: parallel reads, serial writes, error routing, cancellation."""

from __future__ import annotations

import asyncio

import pytest

from codewright.tools.errors import FatalToolError, RespondToModelError
from codewright.tools.executor import ToolExecutor
from codewright.tools.handler import ToolHandler
from codewright.tools.invocation import ToolInvocation
from codewright.tools.registry import ToolRegistry
from codewright.tools.result import ToolResult
from codewright.tools.spec import ToolSpec


class _Sleep(ToolHandler):
    def __init__(self, name: str, *, parallel: bool, sleep: float = 0.05) -> None:
        self._name = name
        self._parallel = parallel
        self._sleep = sleep
        self.invocations: list[str] = []

    @property
    def tool_name(self) -> str:
        return self._name

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self._name,
            description="sleep then return",
            parameters={},
            supports_parallel=self._parallel,
        )

    async def handle(self, invocation: ToolInvocation) -> ToolResult:
        self.invocations.append(invocation.call_id)
        await asyncio.sleep(self._sleep)
        return ToolResult(success=True, body=f"{self._name}:{invocation.call_id}")


def _make_inv(name: str, call_id: str) -> ToolInvocation:
    from codewright.agent.cancellation import CancellationToken
    from codewright.agent.turn_context import TurnContext
    from codewright.protocol import AskForApproval, PermissionProfile

    return ToolInvocation(
        session=None,
        turn_context=TurnContext(
            turn_id="t",
            cwd=__import__("pathlib").Path("."),
            model="m",
            permission_profile=PermissionProfile.WORKSPACE_WRITE,
            approval_policy=AskForApproval.NEVER,
            cancellation_token=CancellationToken(),
        ),
        call_id=call_id,
        tool_name=name,
        arguments={},
        cancellation_token=CancellationToken(),
    )


@pytest.mark.asyncio
async def test_parallel_handlers_run_concurrently() -> None:
    reg = ToolRegistry()
    reg.register(_Sleep("a", parallel=True, sleep=0.1))
    reg.register(_Sleep("b", parallel=True, sleep=0.1))
    ex = ToolExecutor(reg)
    loop = asyncio.get_event_loop()
    start = loop.time()
    results = await ex.dispatch_batch([_make_inv("a", "1"), _make_inv("b", "2")])
    elapsed = loop.time() - start
    assert [r.body for r in results] == ["a:1", "b:2"]
    # Two 0.1s tasks in parallel should be << 0.18s; serial would be ~0.2s.
    assert elapsed < 0.18, f"expected parallel execution, took {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_serial_handler_blocks_after_reader() -> None:
    reg = ToolRegistry()
    reg.register(_Sleep("a", parallel=True, sleep=0.1))
    reg.register(_Sleep("w", parallel=False, sleep=0.1))
    ex = ToolExecutor(reg)
    loop = asyncio.get_event_loop()
    start = loop.time()
    await ex.dispatch_batch([_make_inv("a", "1"), _make_inv("w", "2")])
    elapsed = loop.time() - start
    # Writer must wait for the reader to finish; sequential ~ 0.2s.
    assert elapsed >= 0.18, f"expected serialization, took {elapsed:.3f}s"


class _Boom(ToolHandler):
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    @property
    def tool_name(self) -> str:
        return "boom"

    def spec(self) -> ToolSpec:
        return ToolSpec(name="boom", description="raises", parameters={})

    async def handle(self, invocation: ToolInvocation) -> ToolResult:
        raise self._exc


@pytest.mark.asyncio
async def test_respond_to_model_becomes_failure_result() -> None:
    reg = ToolRegistry()
    reg.register(_Boom(RespondToModelError("oh no")))
    ex = ToolExecutor(reg)
    results = await ex.dispatch_batch([_make_inv("boom", "1")])
    assert results == [ToolResult(success=False, body="oh no")]


@pytest.mark.asyncio
async def test_fatal_tool_error_propagates() -> None:
    reg = ToolRegistry()
    reg.register(_Boom(FatalToolError("engine on fire")))
    ex = ToolExecutor(reg)
    with pytest.raises(FatalToolError):
        await ex.dispatch_batch([_make_inv("boom", "1")])


@pytest.mark.asyncio
async def test_cancelled_error_propagates() -> None:
    reg = ToolRegistry()
    reg.register(_Boom(asyncio.CancelledError()))
    ex = ToolExecutor(reg)
    with pytest.raises(asyncio.CancelledError):
        await ex.dispatch_batch([_make_inv("boom", "1")])


@pytest.mark.asyncio
async def test_unexpected_exception_becomes_failure_result() -> None:
    # A handler that escapes the error taxonomy (here KeyError) must not bubble
    # out of dispatch_batch; it becomes a failed ToolResult so run_turn always
    # gets a result for every tool_call and the conversation stays valid.
    reg = ToolRegistry()
    reg.register(_Boom(KeyError("boom-unexpected")))
    ex = ToolExecutor(reg)
    results = await ex.dispatch_batch([_make_inv("boom", "1")])
    assert results[0].success is False
    assert "failed unexpectedly" in results[0].body
    assert "KeyError" in results[0].body


@pytest.mark.asyncio
async def test_unexpected_exception_does_not_discard_sibling_result() -> None:
    # In a mixed batch, one crashing tool must not lose a healthy sibling's
    # result (the old gather-without-backstop discarded the whole batch).
    reg = ToolRegistry()
    reg.register(_Sleep("ok", parallel=True, sleep=0.0))
    reg.register(_Boom(KeyError("boom")))
    ex = ToolExecutor(reg)
    results = await ex.dispatch_batch(
        [_make_inv("ok", "1"), _make_inv("boom", "2")]
    )
    assert results[0].success is True and results[0].body == "ok:1"
    assert results[1].success is False and "failed unexpectedly" in results[1].body


@pytest.mark.asyncio
async def test_unknown_tool_routes_to_failure_result() -> None:
    reg = ToolRegistry()
    ex = ToolExecutor(reg)
    results = await ex.dispatch_batch([_make_inv("ghost", "1")])
    assert results[0].success is False
    assert "unknown tool" in results[0].body


# -- freeze guard ----------------------------------------------------------


def test_registry_freeze_blocks_register() -> None:
    reg = ToolRegistry()
    reg.register(_Sleep("a", parallel=True))
    assert reg.is_frozen is False
    reg.freeze()
    assert reg.is_frozen is True
    with pytest.raises(RuntimeError, match="frozen"):
        reg.register(_Sleep("b", parallel=True))


def test_registry_freeze_is_idempotent() -> None:
    reg = ToolRegistry()
    reg.freeze()
    reg.freeze()  # second call is a no-op, must not raise
    assert reg.is_frozen is True


@pytest.mark.asyncio
async def test_registry_freeze_keeps_dispatch_working() -> None:
    """Existing handlers stay dispatchable after freeze."""
    reg = ToolRegistry()
    reg.register(_Sleep("a", parallel=True, sleep=0.0))
    reg.freeze()
    ex = ToolExecutor(reg)
    results = await ex.dispatch_batch([_make_inv("a", "1")])
    assert results[0].success is True
    assert results[0].body == "a:1"


@pytest.mark.asyncio
async def test_run_turn_freezes_registry() -> None:
    """run_turn must call freeze() before doing anything else so a slow
    background MCP discovery cannot mutate the tool set mid-turn."""
    from pathlib import Path

    from codewright.agent.session import Session
    from codewright.agent.turn import run_turn
    from codewright.agent.turn_context import TurnContext
    from codewright.context.manager import ContextManager
    from codewright.llm.base import LLMProvider, StreamEvent
    from codewright.prompts.builder import PromptBuilder
    from codewright.protocol import AskForApproval, PermissionProfile

    class _Empty(LLMProvider):
        async def stream(self, messages, tools, turn_context):  # type: ignore[override]
            async def gen():
                yield StreamEvent(kind="text_delta", text="ok")

            return gen()

    sess = Session(
        session_id="freeze",
        cwd=Path("."),
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
        llm=_Empty(),
        context_manager=ContextManager(),
        prompt_builder=PromptBuilder("SYS"),
    )
    try:
        await sess.next_event()  # drain SessionConfigured
        assert sess.tool_registry.is_frozen is False
        tc = TurnContext(
            turn_id="t",
            cwd=Path("."),
            model="m",
            permission_profile=PermissionProfile.WORKSPACE_WRITE,
            approval_policy=AskForApproval.NEVER,
            cancellation_token=__import__(
                "codewright.agent.cancellation", fromlist=["CancellationToken"]
            ).CancellationToken(),
        )
        await run_turn(sess, tc, "hi")
        assert sess.tool_registry.is_frozen is True
        with pytest.raises(RuntimeError, match="frozen"):
            sess.tool_registry.register(_Sleep("late", parallel=True))
    finally:
        await sess.shutdown()
