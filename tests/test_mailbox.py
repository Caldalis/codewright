"""Mailbox queue + seq counter + wake event invariants."""

from __future__ import annotations

import asyncio

import pytest

from codewright.agent.cancellation import CancellationToken
from codewright.agent.mailbox import Mailbox
from codewright.protocol.agent_messages import AgentPath, InterAgentMessage


def _mail(content: str = "hi", *, trigger_turn: bool = False) -> InterAgentMessage:
    return InterAgentMessage(
        author=AgentPath.root(),
        recipient=AgentPath.parse("/root/worker_api"),
        content=content,
        trigger_turn=trigger_turn,
    )


def test_push_queues_without_waking():
    mb = Mailbox()
    mb.push(_mail("one"))
    assert mb.has_pending()
    assert mb.seq == 0  # plain push does NOT bump seq


def test_push_with_wake_bumps_seq_and_sets_event():
    mb = Mailbox()
    mb.push(_mail("a"))
    mb.push_with_wake(_mail("b", trigger_turn=True))
    assert mb.seq == 1


def test_drain_pending_returns_fifo_order_and_clears_event():
    mb = Mailbox()
    mb.push(_mail("a"))
    mb.push_with_wake(_mail("b", trigger_turn=True))
    mb.push(_mail("c"))
    drained = mb.drain_pending()
    assert [m.content for m in drained] == ["a", "b", "c"]
    assert not mb.has_pending()
    # Second push_with_wake after drain re-sets the event so wait_for_wake
    # can be re-entered without a stale signal.
    new_wait_signal_set = asyncio.Event()

    async def _check() -> None:
        wait = asyncio.create_task(mb.wait_for_wake())
        await asyncio.sleep(0.01)
        assert not wait.done()  # nothing pushed yet
        mb.push_with_wake(_mail("d", trigger_turn=True))
        await asyncio.wait_for(wait, timeout=1.0)
        new_wait_signal_set.set()

    asyncio.run(_check())
    assert new_wait_signal_set.is_set()


@pytest.mark.asyncio
async def test_wait_for_wake_returns_on_push_with_wake():
    mb = Mailbox()
    wait = asyncio.create_task(mb.wait_for_wake())
    await asyncio.sleep(0)
    assert not wait.done()
    mb.push_with_wake(_mail(trigger_turn=True))
    await asyncio.wait_for(wait, timeout=1.0)


@pytest.mark.asyncio
async def test_wait_for_wake_cancellable_via_token():
    mb = Mailbox()
    token = CancellationToken()
    wait = asyncio.create_task(mb.wait_for_wake(token))
    await asyncio.sleep(0)
    token.cancel()
    with pytest.raises(asyncio.CancelledError):
        await wait


@pytest.mark.asyncio
async def test_send_message_does_not_wake_recipient():
    """A pure ``push`` must NOT cause a ``wait_for_wake`` to return."""
    mb = Mailbox()
    wait = asyncio.create_task(mb.wait_for_wake())
    await asyncio.sleep(0)
    mb.push(_mail("note"))
    await asyncio.sleep(0.01)
    assert not wait.done()
    wait.cancel()
    with pytest.raises(asyncio.CancelledError):
        await wait
