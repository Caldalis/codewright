"""WebSocket bridge between browsers and the Session engine.

The engine already speaks a wire-ready protocol: ``Op`` and ``Event`` are
frozen pydantic tagged unions (``protocol/``). This module only *transports*
them — it never interprets or transforms engine state:

    browser ──(JSON text frame)──→ deserialize to Op ──→ session.submit()
    session.next_event() ──→ Event.model_dump() ──(JSON text frame)──→ browser

One WebSocket connection owns exactly one Session: connect spawns (or, with
``?session_id=``, resumes) it, disconnect shuts it down. Static conversation
history is REST (``/api/sessions``, ``/api/sessions/{id}/history``); only the
live stream goes over the socket. Approval round-trips need no special
handling here — ``OpExecApprovalResponse`` goes through ``session.submit()``,
which resolves it out-of-band exactly like the TUI does.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from codewright.agent.session import Session
from codewright.persistence.session_store import SessionStore
from codewright.protocol import EvShutdownComplete, Op

# None -> bootstrap a fresh session; a session_id -> resume that session.
SessionFactory = Callable[[str | None], Awaitable[Session]]

_OP_ADAPTER: TypeAdapter[Op] = TypeAdapter(Op)

# Ops a browser may submit. Everything else (shutdown, inter-agent plumbing,
# turn-context overrides) is engine/front-end internal; a web client asking
# for them is either a bug or someone poking the socket by hand.
_CLIENT_ALLOWED_OPS = frozenset(
    {
        "user_turn",
        "interrupt",
        "exec_approval_response",
        "patch_approval_response",
    }
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_TITLE_MAX_CHARS = 80
_TITLE_SCAN_LINES = 200
_HISTORY_BODY_CAP = 4_000

# WebSocket close codes (application range 4000-4999).
_WS_CLOSE_NOT_FOUND = 4404
_WS_CLOSE_CONFLICT = 4409


def create_app(session_factory: SessionFactory, workspace_root: Path) -> Starlette:
    """Build the bridge app. The factory is injected so tests can supply
    Sessions without a real LLM provider."""

    store = SessionStore(workspace_root)
    # session_ids currently attached to a live WebSocket. Two connections
    # resuming the same rollout file would race its append-only log.
    active_session_ids: set[str] = set()

    async def index(_request: Any) -> HTMLResponse:
        return HTMLResponse((_STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    async def list_sessions(_request: Any) -> JSONResponse:
        metas = await store.list_sessions()
        sessions = [
            {
                "session_id": meta.session_id,
                "title": _session_title(store.session_path(meta.session_id)),
                "model": meta.model,
                "start_time": meta.start_time,
                "active": meta.session_id in active_session_ids,
            }
            for meta in metas
        ]
        sessions.sort(key=lambda s: s["start_time"], reverse=True)
        return JSONResponse({"sessions": sessions})

    async def session_history(request: Any) -> JSONResponse:
        session_id = request.path_params["session_id"]
        try:
            _meta, lines = await store.load_session(session_id)
        except (FileNotFoundError, ValueError):
            return JSONResponse({"error": f"unknown session: {session_id}"}, status_code=404)
        return JSONResponse({"blocks": _history_blocks(lines)})

    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        requested_id = websocket.query_params.get("session_id")

        if requested_id and requested_id in active_session_ids:
            await _send_bridge_error(
                websocket, f"session {requested_id} is already open in another connection"
            )
            await websocket.close(code=_WS_CLOSE_CONFLICT)
            return

        try:
            session = await session_factory(requested_id)
        except (FileNotFoundError, ValueError) as exc:
            await _send_bridge_error(websocket, f"cannot resume session: {exc}")
            await websocket.close(code=_WS_CLOSE_NOT_FOUND)
            return

        active_session_ids.add(session.session_id)
        try:
            pump_events = asyncio.create_task(_pump_events(session, websocket))
            pump_ops = asyncio.create_task(_pump_ops(session, websocket))
            done, pending = await asyncio.wait(
                [pump_events, pump_ops], return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            for task in done:
                exc = task.exception()
                if exc is not None and not isinstance(exc, WebSocketDisconnect):
                    raise exc
        finally:
            active_session_ids.discard(session.session_id)
            await session.shutdown()

    routes = [
        Route("/", index),
        Route("/api/sessions", list_sessions),
        Route("/api/sessions/{session_id}/history", session_history),
        WebSocketRoute("/ws", ws_endpoint),
    ]
    return Starlette(routes=routes)


async def _pump_events(session: Session, websocket: WebSocket) -> None:
    """engine → browser: forward every Event as one JSON text frame."""
    while True:
        event = await session.next_event()
        await websocket.send_json(event.model_dump(mode="json"))
        if isinstance(event.msg, EvShutdownComplete):
            return


async def _pump_ops(session: Session, websocket: WebSocket) -> None:
    """browser → engine: validate each JSON text frame into an Op and submit it.

    A malformed frame answers with an ``{"type": "bridge_error"}`` frame and
    keeps the connection alive — the browser fixing its payload should not
    cost the user their session.
    """
    while True:
        raw = await websocket.receive_text()  # raises WebSocketDisconnect on close
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            await _send_bridge_error(websocket, f"invalid JSON: {exc}")
            continue

        op_type = payload.get("type") if isinstance(payload, dict) else None
        if op_type not in _CLIENT_ALLOWED_OPS:
            await _send_bridge_error(
                websocket,
                f"op type {op_type!r} not allowed over the web bridge; "
                f"allowed: {sorted(_CLIENT_ALLOWED_OPS)}",
            )
            continue

        try:
            op = _OP_ADAPTER.validate_python(payload)
        except ValidationError as exc:
            await _send_bridge_error(websocket, f"invalid op: {exc}")
            continue

        await session.submit(op)


async def _send_bridge_error(websocket: WebSocket, message: str) -> None:
    await websocket.send_json({"id": "", "msg": {"type": "bridge_error", "message": message}})


def _session_title(path: Path) -> str:
    """First line of the first user message, or '' if the session has none."""
    try:
        with open(path, encoding="utf-8") as fh:
            for _ in range(_TITLE_SCAN_LINES):
                raw = fh.readline()
                if not raw:
                    break
                try:
                    line = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if line.get("type") != "user_msg":
                    continue
                content = str((line.get("payload") or {}).get("content", "")).strip()
                if not content:
                    continue
                first = content.splitlines()[0].strip()
                if len(first) > _TITLE_MAX_CHARS:
                    first = first[: _TITLE_MAX_CHARS - 1] + "…"
                return first
    except OSError:
        pass
    return ""


def _history_blocks(lines: list[Any]) -> list[dict[str, Any]]:
    """Project rollout lines onto the display blocks the front end renders.

    Only user/assistant/tool lines become blocks; compaction, plan updates and
    meta lines are engine bookkeeping the history view does not show.
    """
    blocks: list[dict[str, Any]] = []
    for line in lines:
        payload = line.payload or {}
        if line.type == "user_msg":
            text = str(payload.get("content", "")).strip()
            if text:
                blocks.append({"kind": "user", "text": text})
        elif line.type == "assistant_msg":
            text = str(payload.get("content") or "").strip()
            if text:
                blocks.append({"kind": "agent", "text": text})
        elif line.type == "tool_result":
            body = str(payload.get("body") or "")
            if len(body) > _HISTORY_BODY_CAP:
                body = body[:_HISTORY_BODY_CAP] + "\n…[truncated for history view]"
            blocks.append(
                {
                    "kind": "tool",
                    "name": str(payload.get("tool_name") or ""),
                    "success": bool(payload.get("success", True)),
                    "body": body,
                }
            )
    return blocks
