from __future__ import annotations

from dataclasses import dataclass

from rich.text import Text


@dataclass
class StatusBarState:


    model: str = ""
    cwd: str = ""
    turn_state: str = "idle"
    input_tokens: int = 0
    output_tokens: int = 0
    pending_approvals: int = 0
    subagent_count: int = 0

    def render(self) -> Text:
        parts = [
            f"model={self.model}",
            f"cwd={self.cwd}",
            f"state={self.turn_state}",
            f"tokens={self.input_tokens}/{self.output_tokens}",
        ]
        if self.subagent_count:
            parts.append(f"agents={self.subagent_count}")
        if self.pending_approvals:
            parts.append(f"approvals={self.pending_approvals}")
        text = Text(" | ".join(parts), style="bold reverse")
        return text


__all__ = ["StatusBarState"]
