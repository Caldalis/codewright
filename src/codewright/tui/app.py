from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from codewright.protocol import (
    EvAgentMessage,
    EvAgentMessageDelta,
    EvError,
    EvExecApprovalRequest,
    EvPatchApprovalRequest,
    EvPlanUpdate,
    EvSessionConfigured,
    EvShutdownComplete,
    EvTokenCount,
    EvToolCallCompleted,
    EvToolCallStarted,
    EvTurnAborted,
    EvTurnCompleted,
    EvTurnStarted,
    EvWarning,
    OpShutdown,
    OpUserTurn,
    UserInputText,
)
from codewright.tui.approval_modal import request_decision
from codewright.tui.keybinds import install_keybindings
from codewright.tui.status_bar import StatusBarState

if TYPE_CHECKING:
    from codewright.agent.session import Session


_MAX_HISTORY_LINES = 500

class TuiApp:

    def __init__(self, session: Session, console: Console | None = None) -> None:
        self.session = session
        self._console = console or Console()
        self._history: list[Text] = []
        self._delta_buf: str = ""
        self._status = StatusBarState(
            model=session.model,
            cwd=str(session.cwd),
        )
        self._live: Live | None = None
        self._shutting_down: bool = False


    async def run(self) -> None:
        with Live(
            self._render(),
            console=self._console,
            refresh_per_second=8,
            transient=False,
        ) as live:
            self._live = live
            try:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self._event_consumer(), name="tui_events")
                    tg.create_task(self._input_handler(), name="tui_input")
            except* asyncio.CancelledError:
                pass
        self._live = None


    def _append_history(self, text: Text) -> None:
        self._history.append(text)
        if len(self._history) > _MAX_HISTORY_LINES:

            del self._history[: len(self._history) - _MAX_HISTORY_LINES]
        if self._live is not None:
            self._live.update(self._render())

    def _render(self) -> Group:
        body = Group(*self._history) if self._history else Text("(no history yet)")
        return Group(
            Panel(body, title="Codewright", border_style="cyan"),
            self._status.render(),
        )

    async def _event_consumer(self) -> None:
        try:
            while not self._shutting_down:
                ev = await self.session.next_event()
                await self._handle_event(ev.msg)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._append_history(Text(f"[event consumer error] {exc}", style="red"))

    async def _handle_event(self, msg) -> None:  # type: ignore[no-untyped-def]
        if isinstance(msg, EvSessionConfigured):
            self._append_history(
                Text(
                    f"session {msg.session_id} (model={msg.model}, cwd={msg.cwd})",
                    style="dim",
                )
            )
        elif isinstance(msg, EvTurnStarted):
            self._status.turn_state = "running"
            self._delta_buf = ""
        elif isinstance(msg, EvAgentMessageDelta):
            self._delta_buf += msg.delta
        elif isinstance(msg, EvAgentMessage):
            self._delta_buf = ""
            self._append_history(Text(msg.content))
        elif isinstance(msg, EvToolCallStarted):
            self._append_history(
                Text(f"  → {msg.tool_name}(call_id={msg.call_id})", style="yellow")
            )
        elif isinstance(msg, EvToolCallCompleted):
            color = "green" if msg.success else "red"
            tag = "ok" if msg.success else "fail"
            preview = msg.body[:200].replace("\n", " ")
            self._append_history(
                Text(f"    [{tag}] {preview}", style=color)
            )
        elif isinstance(msg, EvPlanUpdate):
            lines = "\n".join(f"  - [{p.status.value}] {p.step}" for p in msg.plan)
            self._append_history(
                Text(f"plan:\n{lines}", style="cyan")
            )
        elif isinstance(msg, EvTokenCount):
            self._status.input_tokens = msg.input
            self._status.output_tokens = msg.output
        elif isinstance(msg, EvExecApprovalRequest):
            self._status.pending_approvals += 1
            await request_decision(
                self.session,
                self._live,
                msg.request_id,
                msg.action,
                is_exec=True,
            )
            self._status.pending_approvals = max(0, self._status.pending_approvals - 1)
        elif isinstance(msg, EvPatchApprovalRequest):
            self._status.pending_approvals += 1
            await request_decision(
                self.session,
                self._live,
                msg.request_id,
                msg.action,
                is_exec=False,
            )
            self._status.pending_approvals = max(0, self._status.pending_approvals - 1)
        elif isinstance(msg, EvTurnCompleted):
            self._status.turn_state = "idle"
        elif isinstance(msg, EvTurnAborted):
            self._status.turn_state = "idle"
            self._append_history(
                Text(f"[turn aborted: {msg.reason}]", style="yellow")
            )
        elif isinstance(msg, EvWarning):
            self._append_history(Text(f"[warn] {msg.message}", style="yellow"))
        elif isinstance(msg, EvError):
            self._append_history(Text(f"[error] {msg.message}", style="red"))
        elif isinstance(msg, EvShutdownComplete):
            self._shutting_down = True
        if self._live is not None:
            self._live.update(self._render())

    async def _input_handler(self) -> None:
        kb = install_keybindings(self)
        prompt_session = PromptSession[str](key_bindings=kb)
        try:
            while not self._shutting_down:
                with patch_stdout():
                    try:
                        line = await prompt_session.prompt_async("> ")
                    except (EOFError, KeyboardInterrupt):
                        await self.session.submit(OpShutdown())
                        return
                if line is None:
                    await self.session.submit(OpShutdown())
                    return
                stripped = line.strip()
                if not stripped:
                    continue
                self._append_history(Text(f"> {stripped}", style="bold"))
                await self.session.submit(
                    OpUserTurn(items=[UserInputText(text=stripped)])
                )
        except asyncio.CancelledError:
            raise


__all__ = ["TuiApp"]
