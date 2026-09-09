"""CancellationToken: parent→child cascade + already-cancelled-parent behavior."""

from __future__ import annotations

import asyncio

import pytest

from codewright.agent.cancellation import CancellationToken


async def test_initial_state_not_cancelled():
    t = CancellationToken()
    assert t.is_cancelled() is False


async def test_cancel_is_idempotent():
    t = CancellationToken()
    t.cancel()
    t.cancel()  # should not raise
    assert t.is_cancelled() is True


async def test_wait_returns_after_cancel():
    t = CancellationToken()
    task = asyncio.create_task(t.wait())
    await asyncio.sleep(0)  # let the task start
    assert not task.done()
    t.cancel()
    await asyncio.wait_for(task, timeout=1.0)


async def test_parent_cancel_cascades_to_child():
    parent = CancellationToken()
    child = parent.child()
    assert child.is_cancelled() is False
    parent.cancel()
    assert child.is_cancelled() is True


async def test_parent_cancel_cascades_to_grandchild():
    parent = CancellationToken()
    child = parent.child()
    grandchild = child.child()
    parent.cancel()
    assert child.is_cancelled() is True
    assert grandchild.is_cancelled() is True


async def test_child_built_on_already_cancelled_parent_is_cancelled():
    parent = CancellationToken()
    parent.cancel()
    child = parent.child()
    assert child.is_cancelled() is True
    # And wait() should not hang.
    await asyncio.wait_for(child.wait(), timeout=1.0)


async def test_child_cancel_does_not_propagate_to_parent():
    parent = CancellationToken()
    child = parent.child()
    child.cancel()
    assert child.is_cancelled() is True
    assert parent.is_cancelled() is False


async def test_multiple_children_all_cancel():
    parent = CancellationToken()
    kids = [parent.child() for _ in range(5)]
    parent.cancel()
    for k in kids:
        assert k.is_cancelled() is True


async def test_pending_wait_unblocks_via_parent_cascade():
    parent = CancellationToken()
    child = parent.child()
    waiter = asyncio.create_task(child.wait())
    await asyncio.sleep(0)
    parent.cancel()
    await asyncio.wait_for(waiter, timeout=1.0)


@pytest.mark.parametrize("depth", [1, 3, 10])
async def test_deep_chain_cancel(depth: int):
    root = CancellationToken()
    cursor = root
    chain = [root]
    for _ in range(depth):
        cursor = cursor.child()
        chain.append(cursor)
    root.cancel()
    for node in chain:
        assert node.is_cancelled() is True
