from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class AsyncRwLock:


    def __init__(self) -> None:
        self._cond = asyncio.Condition()
        self._readers = 0
        self._writer_active = False

        self._writer_waiters = 0


    async def _acquire_read(self) -> None:
        async with self._cond:
            await self._cond.wait_for(
                lambda: not self._writer_active and self._writer_waiters == 0
            )
            self._readers += 1

    async def _release_read(self) -> None:
        async with self._cond:
            self._readers -= 1
  
            if self._readers == 0:
                self._cond.notify_all()

    async def _acquire_write(self) -> None:
        async with self._cond:
            self._writer_waiters += 1
            try:
                await self._cond.wait_for(
                    lambda: not self._writer_active and self._readers == 0
                )
            finally:

                self._writer_waiters -= 1
            self._writer_active = True

    async def _release_write(self) -> None:
        async with self._cond:
            self._writer_active = False
            self._cond.notify_all()


    @asynccontextmanager
    async def read(self) -> AsyncIterator[None]:
        await self._acquire_read()
        try:
            yield
        finally:
            await self._release_read()

    @asynccontextmanager
    async def write(self) -> AsyncIterator[None]:
        await self._acquire_write()
        try:
            yield
        finally:
            await self._release_write()
