from __future__ import annotations

import asyncio

from codewright.agent.rwlock import AsyncRwLock
from codewright.tools.errors import FatalToolError, RespondToModelError
from codewright.tools.invocation import ToolInvocation
from codewright.tools.registry import ToolRegistry
from codewright.tools.result import ToolResult


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, rwlock: AsyncRwLock | None = None) -> None:
        self._registry = registry
        self._rwlock = rwlock or AsyncRwLock()

    @property
    def rwlock(self) -> AsyncRwLock:
        return self._rwlock

    async def dispatch_batch(
        self, invocations: list[ToolInvocation]
    ) -> list[ToolResult]:
        if not invocations:
            return []
        tasks = [asyncio.create_task(self._dispatch_one(inv)) for inv in invocations]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return list(results)

    async def _dispatch_one(self, inv: ToolInvocation) -> ToolResult:
        try:
            handler = self._registry.get(inv.tool_name)
        except FatalToolError:

            return ToolResult(success=False, body=f"unknown tool: {inv.tool_name}")

        spec = handler.spec()
        ctx = self._rwlock.read() if spec.supports_parallel else self._rwlock.write()
        async with ctx:
            try:
                return await handler.handle(inv)
            except RespondToModelError as exc:
                return ToolResult(success=False, body=str(exc))
            except FatalToolError:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as exc:

                return ToolResult(
                    success=False,
                    body=(
                        f"{inv.tool_name} failed unexpectedly: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )

