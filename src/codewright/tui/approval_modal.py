from __future__ import annotations

from codewright.protocol import PendingAction


def render_action(action: PendingAction) -> str:
    head = f"[{action.kind}] {action.summary}"
    if not action.details:
        return head
    detail_lines = "\n".join(f"  {k}: {v}" for k, v in sorted(action.details.items()))
    return f"{head}\n{detail_lines}"


__all__ = ["render_action"]
