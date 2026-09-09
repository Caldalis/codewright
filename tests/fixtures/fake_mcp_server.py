"""A minimal stdio MCP server fixture.

Exposes two tools:
- ``echo(text: str) -> str`` — returns the text verbatim.
- ``fail(reason: str) -> never`` — returns isError=true with ``reason``.

Used by ``tests/test_mcp_mock.py``. Run as a subprocess; reads JSON-RPC
requests from stdin (one per line), writes responses to stdout.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any


def _emit_pid() -> None:
    # Opt-in (via env) PID emission so tests can assert the subprocess is
    # actually reaped on shutdown. No effect when the env var is unset.
    path = os.environ.get("CW_MCP_PIDFILE")
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(str(os.getpid()))


def handle(req: dict[str, Any]) -> dict[str, Any] | None:
    method = req.get("method") or ""
    rpc_id = req.get("id")

    # Notifications carry no id; we drop them silently (no response expected).
    if rpc_id is None or method.startswith("notifications/"):
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "fake", "version": "0.0.1"},
                "capabilities": {"tools": {}},
            },
        }
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "tools": [
                    {
                        "name": "echo",
                        "description": "echo back the text",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    },
                    {
                        "name": "fail",
                        "description": "always fail",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"reason": {"type": "string"}},
                            "required": ["reason"],
                        },
                    },
                ]
            },
        }
    if method == "tools/call":
        params = req.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "echo":
            text = args.get("text", "")
            return {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {
                    "content": [{"type": "text", "text": str(text)}],
                    "isError": False,
                },
            }
        if name == "fail":
            return {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {
                    "content": [
                        {"type": "text", "text": str(args.get("reason", ""))}
                    ],
                    "isError": True,
                },
            }
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "error": {"code": -32601, "message": f"unknown tool: {name!r}"},
        }
    return {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "error": {"code": -32601, "message": f"unknown method: {method!r}"},
    }


def main() -> None:
    _emit_pid()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle(req)
        if resp is None:
            continue
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
