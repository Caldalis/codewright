"""AsyncRwLock: multi-reader / single-writer / writer-preference."""

from __future__ import annotations

import asyncio

import pytest

from codewright.agent.rwlock import AsyncRwLock


async def test_multiple_readers_concurrent():
    """Two readers can hold the lock at the same time."""
    lock = AsyncRwLock()
    r1_in = asyncio.Event()
    r2_in = asyncio.Event()
    release = asyncio.Event()

    async def reader(in_event: asyncio.Event) -> None:
        async with lock.read():
            in_event.set()
            await release.wait()

    t1 = asyncio.create_task(reader(r1_in))
    t2 = asyncio.create_task(reader(r2_in))

    # Both should be able to enter without blocking on each other.
    await asyncio.wait_for(r1_in.wait(), timeout=1.0)
    await asyncio.wait_for(r2_in.wait(), timeout=1.0)

    release.set()
    await asyncio.gather(t1, t2)


async def test_writer_excludes_readers():
    """A writer holding the lock blocks new readers until release."""
    lock = AsyncRwLock()
    writer_in = asyncio.Event()
    writer_release = asyncio.Event()
    reader_in = asyncio.Event()

    async def writer() -> None:
        async with lock.write():
            writer_in.set()
            await writer_release.wait()

    async def reader() -> None:
        async with lock.read():
            reader_in.set()

    w = asyncio.create_task(writer())
    await asyncio.wait_for(writer_in.wait(), timeout=1.0)

    r = asyncio.create_task(reader())
    # Give the reader a chance to (try and) acquire; it must remain blocked.
    await asyncio.sleep(0.05)
    assert not reader_in.is_set(), "reader entered while writer held lock"

    writer_release.set()
    await asyncio.wait_for(reader_in.wait(), timeout=1.0)
    await asyncio.gather(w, r)


async def test_writer_excludes_other_writer():
    """Two writers serialize."""
    lock = AsyncRwLock()
    w1_in = asyncio.Event()
    w1_release = asyncio.Event()
    w2_in = asyncio.Event()

    async def writer_one() -> None:
        async with lock.write():
            w1_in.set()
            await w1_release.wait()

    async def writer_two() -> None:
        async with lock.write():
            w2_in.set()

    t1 = asyncio.create_task(writer_one())
    await asyncio.wait_for(w1_in.wait(), timeout=1.0)

    t2 = asyncio.create_task(writer_two())
    await asyncio.sleep(0.05)
    assert not w2_in.is_set(), "second writer entered while first held lock"

    w1_release.set()
    await asyncio.wait_for(w2_in.wait(), timeout=1.0)
    await asyncio.gather(t1, t2)


async def test_writer_preference_new_readers_block_when_writer_waits():
    """Once a writer queues, additional readers must wait too (no starvation).

    Sequence:
      1. R1 holds the read lock.
      2. W1 calls write() and blocks on R1.
      3. R2 calls read() and must block on W1's pending status.
      4. Release R1 → W1 acquires (not R2).
      5. Release W1 → R2 finally acquires.
    """
    lock = AsyncRwLock()
    r1_in = asyncio.Event()
    r1_release = asyncio.Event()
    w1_in = asyncio.Event()
    w1_release = asyncio.Event()
    r2_in = asyncio.Event()

    async def reader_one() -> None:
        async with lock.read():
            r1_in.set()
            await r1_release.wait()

    async def writer_one() -> None:
        async with lock.write():
            w1_in.set()
            await w1_release.wait()

    async def reader_two() -> None:
        async with lock.read():
            r2_in.set()

    t_r1 = asyncio.create_task(reader_one())
    await asyncio.wait_for(r1_in.wait(), timeout=1.0)

    t_w1 = asyncio.create_task(writer_one())
    # Let the writer enter the queue (writer_waiters += 1) but not acquire.
    await asyncio.sleep(0.05)
    assert not w1_in.is_set()

    t_r2 = asyncio.create_task(reader_two())
    # Reader-2 must NOT skip the queued writer.
    await asyncio.sleep(0.05)
    assert not r2_in.is_set(), "reader-2 entered before queued writer (no writer preference)"

    # Release reader-1; writer-1 should win the lock next.
    r1_release.set()
    await asyncio.wait_for(w1_in.wait(), timeout=1.0)
    # Reader-2 is still blocked while writer-1 holds.
    assert not r2_in.is_set()

    # Release writer-1; reader-2 finally proceeds.
    w1_release.set()
    await asyncio.wait_for(r2_in.wait(), timeout=1.0)

    await asyncio.gather(t_r1, t_w1, t_r2)


async def test_lock_recovers_after_handler_exception():
    """Exception inside read()/write() body still releases the lock."""
    lock = AsyncRwLock()

    with pytest.raises(RuntimeError):
        async with lock.write():
            raise RuntimeError("boom")

    # Should be acquirable again.
    async with lock.read():
        pass
    async with lock.write():
        pass
