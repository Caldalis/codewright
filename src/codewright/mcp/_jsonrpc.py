from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from typing import Any

JSONRPC_VERSION = "2.0"


@dataclass(frozen=True)
class JsonRpcError(Exception):


    code: int
    message: str
    data: Any = None

    def __str__(self) -> str:  # pragma: no cover - tiny
        suffix = f" data={self.data!r}" if self.data is not None else ""
        return f"JSON-RPC error {self.code}: {self.message}{suffix}"


class IdAllocator:

    def __init__(self, start: int = 1) -> None:
        self._counter = itertools.count(start)

    def next(self) -> int:
        return next(self._counter)


def encode_request(rpc_id: int, method: str, params: dict[str, Any] | None) -> str:
    payload: dict[str, Any] = {
        "jsonrpc": JSONRPC_VERSION,
        "id": rpc_id,
        "method": method,
    }
    if params is not None:
        payload["params"] = params
    return json.dumps(payload, ensure_ascii=False) + "\n"


def encode_notification(method: str, params: dict[str, Any] | None) -> str:
    payload: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "method": method}
    if params is not None:
        payload["params"] = params
    return json.dumps(payload, ensure_ascii=False) + "\n"


def parse_response(line: str) -> dict[str, Any]:

    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON-RPC line: {line!r}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON-RPC envelope must be an object, got {type(payload).__name__}")
    return payload


def extract_result(envelope: dict[str, Any]) -> Any:
    if "error" in envelope:
        err = envelope["error"] or {}
        raise JsonRpcError(
            code=int(err.get("code", -32000)),
            message=str(err.get("message", "unknown error")),
            data=err.get("data"),
        )
    return envelope.get("result")


__all__ = [
    "JSONRPC_VERSION",
    "IdAllocator",
    "JsonRpcError",
    "encode_notification",
    "encode_request",
    "extract_result",
    "parse_response",
]
