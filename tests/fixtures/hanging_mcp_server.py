"""Hanging MCP server fixture: reads stdin but never responds.

Used to exercise the per-server startup timeout in ``McpConnectionManager``.
"""

from __future__ import annotations

import os
import sys
import time


def _emit_pid() -> None:
    # Let tests assert the subprocess is actually reaped on shutdown by writing
    # our PID where they can find it. Opt-in via env so default behavior is
    # unchanged.
    path = os.environ.get("CW_MCP_PIDFILE")
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(str(os.getpid()))


def main() -> None:
    _emit_pid()
    # Block forever, occasionally consuming input so the parent's stdin
    # buffer doesn't fill. We never write anything back.
    while True:
        line = sys.stdin.readline()
        if not line:
            time.sleep(60)
            continue


if __name__ == "__main__":
    main()
