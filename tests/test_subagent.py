"""AgentControl + subagent tool handlers (spawn / followup / wait / list / close).

All tests use a stub ``LLMProvider`` so subagent turns finish deterministically
without network.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from codewright.agent.cancellation import CancellationToken
from codewright.agent.control import AgentControl
from codewright.agent.roles import load_builtin_roles
from codewright.agent.session import Session
from codewright.agent.turn_context import TurnContext
from codewright.context import ContextManager
from codewright.llm.base import CanonicalMessage, LLMProvider, StreamEvent
from codewright.prompts.builder import PromptBuilder
from codewright.protocol import (
    AskForApproval,
    PermissionProfile,
)
from codewright.protocol.agent_messages import AgentPath
from codewright.tools.handlers import (
    CloseAgentHandler,
    FollowupTaskHandler,
    ListAgentsHandler,
    SendMessageHandler,
    SpawnAgentHandler,
    WaitAgentHandler,
)
from codewright.tools.invocation import ToolInvocation


class CannedProvider(LLMProvider):
    """LLMProvider that returns the same single-line answer for every turn."""

    def __init__(self, answer: str = "ok") -> None:
        self._answer = answer
        self.turn_count = 0

    async def stream(  # type: ignore[override]
        self,
        messages: list[CanonicalMessage],
        tools: list,
        turn_context,
    ) -> AsyncIterator[StreamEvent]:
        del messages, tools, turn_context
        self.turn_count += 1

        async def _gen() -> AsyncIterator[StreamEvent]:
            yield StreamEvent(kind="text_delta", text=self._answer)

        return _gen()


def _make_root_session(provider: LLMProvider) -> Session:
    return Session(
        session_id="root",
        cwd=Path("/tmp"),
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
        llm=provider,
        context_manager=ContextManager(max_context_tokens=4000),
        prompt_builder=PromptBuilder("SYS"),
    )


def _make_turn_context() -> TurnContext:
    return TurnContext(
        turn_id="t-x",
        cwd=Path("/tmp"),
        model="m",
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
        approval_policy=AskForApproval.ON_REQUEST,
        cancellation_token=CancellationToken(),
    )


async def _drain_until_idle(control: AgentControl, paths: list[AgentPath]) -> None:
    await control.wait_agent(paths, timeout_ms=3000)


# ---- AgentControl direct API --------------------------------------------------


@pytest.mark.asyncio
async def test_spawn_agent_runs_turn_and_records_terminal_status():
    provider = CannedProvider("done")
    root = _make_root_session(provider)
    try:
        control = root.agent_control
        path = await control.spawn_agent("explorer", "find_logs", "look around")
        assert str(path) == "/root/explorer_find_logs"
        terminals = await control.wait_agent([path], timeout_ms=3000)
        assert len(terminals) == 1
        info = terminals[0]
        assert info.status == "completed"
        assert info.last_message == "done"
    finally:
        await root.shutdown()


@pytest.mark.asyncio
async def test_followup_task_triggers_second_turn_with_mailbox_content():
    provider = CannedProvider("ack")
    root = _make_root_session(provider)
    try:
        control = root.agent_control
        path = await control.spawn_agent("worker", "build_x", "start")
        await control.wait_agent([path], timeout_ms=3000)
        first_turns = provider.turn_count
        # Followup should start a new turn that drains the mailbox content.
        await control.followup_task(path, "do the next step")
        await control.wait_agent([path], timeout_ms=3000)
        assert provider.turn_count == first_turns + 1
        sub = control.get_session(path)
        # History contains a user message whose body includes the followup.
        user_msgs = [m for m in sub.context.snapshot() if m.role == "user"]
        assert any("do the next step" in (m.content if isinstance(m.content, str) else "")
                   for m in user_msgs)
    finally:
        await root.shutdown()


@pytest.mark.asyncio
async def test_send_message_queues_without_starting_turn():
    provider = CannedProvider("done")
    root = _make_root_session(provider)
    try:
        control = root.agent_control
        path = await control.spawn_agent("worker", "a", "go")
        await control.wait_agent([path], timeout_ms=3000)
        prev = provider.turn_count
        await control.send_message(path, "fyi")
        await asyncio.sleep(0.05)
        # send_message must NOT trigger a new turn on its own.
        assert provider.turn_count == prev
        # The message is sitting in the mailbox.
        assert control.get_session(path).mailbox.has_pending()
    finally:
        await root.shutdown()


@pytest.mark.asyncio
async def test_list_agents_reflects_full_tree():
    provider = CannedProvider("ok")
    root = _make_root_session(provider)
    try:
        control = root.agent_control
        p1 = await control.spawn_agent("explorer", "one", "investigate one")
        p2 = await control.spawn_agent("explorer", "two", "investigate two")
        await control.wait_agent([p1, p2], timeout_ms=3000)
        await control.wait_agent([p1, p2], timeout_ms=3000)
        agents = control.list_agents()
        paths = sorted(str(a.path) for a in agents)
        assert paths == ["/root/explorer_one", "/root/explorer_two"]
    finally:
        await root.shutdown()


@pytest.mark.asyncio
async def test_close_agent_marks_closed_and_is_idempotent():
    provider = CannedProvider("ok")
    root = _make_root_session(provider)
    try:
        control = root.agent_control
        path = await control.spawn_agent("worker", "x", "go")
        await control.wait_agent([path], timeout_ms=3000)
        await control.close_agent(path)
        info = control.get_info(path)
        assert info.status == "closed"
        # Idempotent — second call must not raise.
        await control.close_agent(path)
    finally:
        await root.shutdown()


@pytest.mark.asyncio
async def test_subagent_session_is_isolated_from_root():
    """L5 invariant: subagent has its own ContextManager / Mailbox."""
    provider = CannedProvider("alpha")
    root = _make_root_session(provider)
    try:
        path = await root.agent_control.spawn_agent("worker", "iso", "do thing")
        await root.agent_control.wait_agent([path], timeout_ms=3000)
        sub = root.agent_control.get_session(path)
        assert sub.context is not root.context
        assert sub.mailbox is not root.mailbox
        # Yet they share LLM / prompt_builder / workspace per L1.
        assert sub._llm is root._llm
        assert sub._prompt_builder is root._prompt_builder
    finally:
        await root.shutdown()


# ---- Tool handler smoke (spawn / list) ---------------------------------------


@pytest.mark.asyncio
async def test_spawn_agent_handler_description_lists_all_roles():
    handler = SpawnAgentHandler(load_builtin_roles())
    spec = handler.spec()
    assert "default" in spec.description
    assert "explorer" in spec.description
    assert "worker" in spec.description


@pytest.mark.asyncio
async def test_list_agents_handler_returns_no_subagents_initially():
    provider = CannedProvider("ok")
    root = _make_root_session(provider)
    try:
        # Touch agent_control so the property is initialised.
        _ = root.agent_control
        handler = ListAgentsHandler()
        inv = ToolInvocation(
            session=root,
            turn_context=_make_turn_context(),
            call_id="c1",
            tool_name="list_agents",
            arguments={},
            cancellation_token=CancellationToken(),
        )
        result = await handler.handle(inv)
        assert result.success
        assert "no subagents" in result.body
    finally:
        await root.shutdown()


@pytest.mark.asyncio
async def test_close_agent_handler_dispatches_close():
    provider = CannedProvider("ok")
    root = _make_root_session(provider)
    try:
        path = await root.agent_control.spawn_agent("worker", "z", "go")
        await root.agent_control.wait_agent([path], timeout_ms=3000)
        handler = CloseAgentHandler()
        inv = ToolInvocation(
            session=root,
            turn_context=_make_turn_context(),
            call_id="c2",
            tool_name="close_agent",
            arguments={"target": str(path)},
            cancellation_token=CancellationToken(),
        )
        res = await handler.handle(inv)
        assert res.success
        assert root.agent_control.get_info(path).status == "closed"
    finally:
        await root.shutdown()


@pytest.mark.asyncio
async def test_send_message_and_followup_handlers_route_to_mailbox():
    provider = CannedProvider("ok")
    root = _make_root_session(provider)
    try:
        path = await root.agent_control.spawn_agent("worker", "w", "go")
        await root.agent_control.wait_agent([path], timeout_ms=3000)

        sm = SendMessageHandler()
        inv1 = ToolInvocation(
            session=root,
            turn_context=_make_turn_context(),
            call_id="c1",
            tool_name="send_message",
            arguments={"target": str(path), "content": "hi"},
            cancellation_token=CancellationToken(),
        )
        res = await sm.handle(inv1)
        assert res.success
        assert root.agent_control.get_session(path).mailbox.has_pending()

        ft = FollowupTaskHandler()
        prev_turns = provider.turn_count
        inv2 = ToolInvocation(
            session=root,
            turn_context=_make_turn_context(),
            call_id="c2",
            tool_name="followup_task",
            arguments={"target": str(path), "content": "next step"},
            cancellation_token=CancellationToken(),
        )
        res2 = await ft.handle(inv2)
        assert res2.success
        await root.agent_control.wait_agent([path], timeout_ms=3000)
        assert provider.turn_count == prev_turns + 1
    finally:
        await root.shutdown()


@pytest.mark.asyncio
async def test_followup_task_handler_rejects_closed_agent():
    # Finding 1: followup on a closed agent would silently flip it back to
    # "running" on a dead loop. The handler must reject it instead.
    provider = CannedProvider("ok")
    root = _make_root_session(provider)
    try:
        path = await root.agent_control.spawn_agent("worker", "c1", "go")
        await root.agent_control.wait_agent([path], timeout_ms=3000)
        await root.agent_control.close_agent(path)
        ft = FollowupTaskHandler()
        inv = ToolInvocation(
            session=root,
            turn_context=_make_turn_context(),
            call_id="c",
            tool_name="followup_task",
            arguments={"target": str(path), "content": "more"},
            cancellation_token=CancellationToken(),
        )
        from codewright.tools.errors import RespondToModelError

        with pytest.raises(RespondToModelError, match="closed"):
            await ft.handle(inv)
        # Status must remain closed (not corrupted to running).
        assert root.agent_control.get_info(path).status == "closed"
    finally:
        await root.shutdown()


@pytest.mark.asyncio
async def test_send_message_handler_rejects_closed_agent():
    provider = CannedProvider("ok")
    root = _make_root_session(provider)
    try:
        path = await root.agent_control.spawn_agent("worker", "c2", "go")
        await root.agent_control.wait_agent([path], timeout_ms=3000)
        await root.agent_control.close_agent(path)
        sm = SendMessageHandler()
        inv = ToolInvocation(
            session=root,
            turn_context=_make_turn_context(),
            call_id="c",
            tool_name="send_message",
            arguments={"target": str(path), "content": "hi"},
            cancellation_token=CancellationToken(),
        )
        from codewright.tools.errors import RespondToModelError

        with pytest.raises(RespondToModelError, match="closed"):
            await sm.handle(inv)
    finally:
        await root.shutdown()


@pytest.mark.asyncio
async def test_wait_agent_handler_rejects_self_wait():
    # Finding 2: a subagent waiting on its own path can never resolve.
    provider = CannedProvider("ok")
    root = _make_root_session(provider)
    try:
        path = await root.agent_control.spawn_agent("worker", "self", "go")
        await root.agent_control.wait_agent([path], timeout_ms=3000)
        sub = root.agent_control.get_session(path)
        handler = WaitAgentHandler()
        inv = ToolInvocation(
            session=sub,  # the subagent itself is the caller
            turn_context=_make_turn_context(),
            call_id="c",
            tool_name="wait_agent",
            arguments={"targets": [str(path)]},
            cancellation_token=CancellationToken(),
        )
        from codewright.tools.errors import RespondToModelError

        with pytest.raises(RespondToModelError, match="yourself"):
            await handler.handle(inv)
    finally:
        await root.shutdown()


@pytest.mark.asyncio
async def test_wait_agent_handler_returns_terminal_info():
    provider = CannedProvider("hello")
    root = _make_root_session(provider)
    try:
        path = await root.agent_control.spawn_agent("worker", "z", "go")
        handler = WaitAgentHandler()
        inv = ToolInvocation(
            session=root,
            turn_context=_make_turn_context(),
            call_id="c3",
            tool_name="wait_agent",
            arguments={"targets": [str(path)], "timeout_ms": 3000},
            cancellation_token=CancellationToken(),
        )
        res = await handler.handle(inv)
        assert res.success
        assert "/root/worker_z" in res.body
        assert "completed" in res.body
    finally:
        await root.shutdown()


# ---- InterAgentCommunication via the submission_loop ------------------------


@pytest.mark.asyncio
async def test_inter_agent_communication_op_routes_into_target_mailbox():
    from codewright.protocol import OpInterAgentCommunication

    provider = CannedProvider("ok")
    root = _make_root_session(provider)
    try:
        path = await root.agent_control.spawn_agent("worker", "ab", "do x")
        await root.agent_control.wait_agent([path], timeout_ms=3000)
        await root.submit(
            OpInterAgentCommunication(
                author=AgentPath.root(),
                recipient=path,
                content="poke",
                trigger_turn=False,
            )
        )
        # Give the loop a tick to process the op.
        for _ in range(50):
            await asyncio.sleep(0.01)
            if root.agent_control.get_session(path).mailbox.has_pending():
                break
        assert root.agent_control.get_session(path).mailbox.has_pending()
    finally:
        await root.shutdown()
