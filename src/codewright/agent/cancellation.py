from __future__ import annotations

import asyncio
import weakref
from typing import Self


class CancellationToken:

    def __init__(self) -> None:
        self._event = asyncio.Event()

        self._children: weakref.WeakSet[CancellationToken] = weakref.WeakSet()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        if self._event.is_set():
            return
        self._event.set()
        for child in list(self._children):
            child.cancel()

    async def wait(self) -> None:
        await self._event.wait()

    def child(self) -> Self:
        c = type(self)()
        if self._event.is_set():

            c._event.set()
            return c
        self._children.add(c)
        return c
