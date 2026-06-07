from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Iterable

from codewright.agent.cancellation import CancellationToken
from codewright.protocol import (
    EventMsg,
    EvExecApprovalRequest,
    EvPatchApprovalRequest,
    EvWarning,
    PendingAction,
    ReviewDecision,
)

EmitEvent = Callable[[EventMsg, str], Awaitable[None]]
CancellationTokens = Callable[[], Iterable[CancellationToken]]


class ApprovalBroker:
    def __init__(
        self,
        *,
        emit_event: EmitEvent,
        cancellation_tokens: CancellationTokens,
    ) -> None:
        self._emit_event = emit_event
        self._cancellation_tokens = cancellation_tokens
        self._pending: dict[str, asyncio.Future[ReviewDecision]] = {}

    async def request(self, action: PendingAction) -> ReviewDecision:
        request_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ReviewDecision] = loop.create_future()
        self._pending[request_id] = future

        if action.kind == "exec":
            event: EventMsg = EvExecApprovalRequest(request_id=request_id, action=action)
        elif action.kind == "patch":
            event = EvPatchApprovalRequest(request_id=request_id, action=action)
        else:
            self._pending.pop(request_id, None)
            raise NotImplementedError(
                f"approval kind {action.kind!r} not in MVP - only 'exec' and 'patch'"
            )

        await self._emit_event(event, "")

        cancel_waits = [
            asyncio.create_task(token.wait())
            for token in self._cancellation_tokens()
        ]
        try:
            done, _pending = await asyncio.wait(
                [future, *cancel_waits], return_when=asyncio.FIRST_COMPLETED
            )
            if future in done:
                return future.result()
            self._pending.pop(request_id, None)
            raise asyncio.CancelledError("turn interrupted while awaiting approval")
        except asyncio.CancelledError:
            self._pending.pop(request_id, None)
            if not future.done():
                future.cancel()
            raise
        finally:
            for task in cancel_waits:
                if not task.done():
                    task.cancel()

    async def resolve_exec_response(
        self, request_id: str, decision: ReviewDecision, sub_id: str
    ) -> None:
        await self._resolve_response(
            label="exec",
            request_id=request_id,
            decision=decision,
            sub_id=sub_id,
        )

    async def resolve_patch_response(
        self, request_id: str, decision: ReviewDecision, sub_id: str
    ) -> None:
        await self._resolve_response(
            label="patch",
            request_id=request_id,
            decision=decision,
            sub_id=sub_id,
        )

    async def _resolve_response(
        self,
        *,
        label: str,
        request_id: str,
        decision: ReviewDecision,
        sub_id: str,
    ) -> None:
        if self._resolve_pending(request_id, decision):
            return
        await self._emit_event(
            EvWarning(
                message=(
                    f"{label} approval response for unknown "
                    f"request_id={request_id!r}"
                )
            ),
            sub_id,
        )

    def _resolve_pending(
        self, request_id: str, decision: ReviewDecision
    ) -> bool:
        future = self._pending.pop(request_id, None)
        if future is None or future.done():
            return False
        future.set_result(decision)
        return True
