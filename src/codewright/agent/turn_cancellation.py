from __future__ import annotations

from codewright.agent.cancellation import CancellationToken


class TurnCancellationManager:
    def __init__(self, session_token: CancellationToken) -> None:
        self._session_token = session_token
        self._queued: dict[str, CancellationToken] = {}
        self._active: CancellationToken | None = None

    def prepare(self, sub_id: str) -> CancellationToken:
        token = self._session_token.child()
        self._queued[sub_id] = token
        return token

    def activate(self, sub_id: str) -> CancellationToken:
        token = self._queued.pop(sub_id, None)
        if token is None:
            token = self._session_token.child()
        self._active = token
        return token

    def clear(self, token: CancellationToken) -> None:
        if self._active is token:
            self._active = None

    def cancel_current_or_queued(self) -> bool:
        if self._active is not None:
            self._active.cancel()
            return True

        for token in self._queued.values():
            if not token.is_cancelled():
                token.cancel()
                return True
        return False

    def approval_tokens(self) -> list[CancellationToken]:
        tokens = [self._session_token]
        if self._active is not None and self._active is not self._session_token:
            tokens.append(self._active)
        return tokens
