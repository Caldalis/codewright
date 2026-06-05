from __future__ import annotations

import asyncio

from codewright.agent.cancellation import CancellationToken
from codewright.protocol.agent_messages import InterAgentMessage


class Mailbox:

    def __init__(self) -> None:
        self._queue: asyncio.Queue[InterAgentMessage] = asyncio.Queue()
        self._seq: int = 0
        self._seq_changed: asyncio.Event = asyncio.Event()

    def push(self, msg: InterAgentMessage) -> None:

        self._queue.put_nowait(msg)

    def push_with_wake(self, msg: InterAgentMessage) -> None:

        self._queue.put_nowait(msg)
        self._seq += 1
        self._seq_changed.set()

    def has_pending(self) -> bool:
        return not self._queue.empty()

    def drain_pending(self) -> list[InterAgentMessage]:
        out: list[InterAgentMessage] = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        self._seq_changed.clear()
        return out

    async def wait_for_wake(self, cancel_token: CancellationToken | None = None) -> None:

        if cancel_token is None:
            await self._seq_changed.wait()
            return

        wake = asyncio.create_task(self._seq_changed.wait())
        cancel_wait = asyncio.create_task(cancel_token.wait())
        try:
            done, _pending = await asyncio.wait(
                [wake, cancel_wait], return_when=asyncio.FIRST_COMPLETED
            )
            if cancel_wait in done and wake not in done:
                raise asyncio.CancelledError("mailbox wait cancelled")
        finally:
            if not wake.done():
                wake.cancel()
            if not cancel_wait.done():
                cancel_wait.cancel()

    @property
    def seq(self) -> int:
        return self._seq


__all__ = ["Mailbox"]
