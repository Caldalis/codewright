from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.input.base import Input
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.output.base import Output
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea
from rich.console import Console
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
    Op,
    OpExecApprovalResponse,
    OpInterrupt,
    OpPatchApprovalResponse,
    OpShutdown,
    OpUserTurn,
    PendingAction,
    ReviewDecision,
    UserInputText,
)
from codewright.tui.approval_modal import render_action
from codewright.tui.keybinds import install_keybindings
from codewright.tui.status_bar import StatusBarState

if TYPE_CHECKING:
    from codewright.agent.session import Session


_MAX_HISTORY_LINES = 500
_MAX_APPROVAL_PREVIEW_CHARS = 600


@dataclass
class _PendingApproval:
    request_id: str
    action: PendingAction
    is_exec: bool


class TuiApp:
    def __init__(
        self,
        session: Session,
        console: Console | None = None,
        *,
        pt_input: Input | None = None,
        pt_output: Output | None = None,
        full_screen: bool = False,
    ) -> None:
        self.session = session
        self._console = console or Console()
        self._pt_input = pt_input
        self._pt_output = pt_output
        self._full_screen = full_screen
        self._history: list[Text] = []
        self._delta_buf: str = ""
        self._status = StatusBarState(
            model=session.model,
            cwd=str(session.cwd),
        )
        self._shutting_down: bool = False

        self._application: Application | None = None
        self._event_task: asyncio.Task[None] | None = None
        self._bg_tasks: set[asyncio.Task[Any]] = set()
        self._history_view: TextArea | None = None
        self._input_view: TextArea | None = None
        self._pending_approval: _PendingApproval | None = None
        self._approval_queue: list[_PendingApproval] = []

    async def run(self) -> None:
        self._application = self._build_application()
        self._event_task = asyncio.create_task(
            self._event_consumer(), name="tui_events"
        )
        try:
            await self._application.run_async()
        finally:
            self._shutting_down = True
            if self._event_task is not None:
                self._event_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._event_task
        self._event_task = None
        self._application = None

    def request_interrupt(self) -> None:
        self._submit_op(OpInterrupt())

    def request_shutdown(self) -> None:
        self._shutting_down = True
        self._submit_op(OpShutdown())
        self._exit_application()

    @property
    def has_pending_approval(self) -> bool:
        return self._pending_approval is not None

    def submit_approval(self, decision: ReviewDecision) -> None:
        pending = self._pending_approval
        if pending is None:
            return

        self._pending_approval = None
        self._status.pending_approvals = max(0, self._status.pending_approvals - 1)
        self._append_history(Text(f"[approval] {decision.value}", style="dim"))

        if pending.is_exec:
            self._submit_op(
                OpExecApprovalResponse(
                    request_id=pending.request_id,
                    decision=decision,
                )
            )
        else:
            self._submit_op(
                OpPatchApprovalResponse(
                    request_id=pending.request_id,
                    decision=decision,
                )
            )
        self._pending_approval = (
            self._approval_queue.pop(0) if self._approval_queue else None
        )
        self._invalidate()

    def _build_application(self) -> Application:
        self._history_view = TextArea(
            text=self._history_text(),
            read_only=True,
            focusable=False,
            wrap_lines=True,
            scrollbar=True,
            style="class:history",
        )
        self._input_view = TextArea(
            height=1,
            multiline=False,
            prompt="> ",
            accept_handler=self._accept_input,
            style="class:input",
        )

        root = HSplit(
            [
                Frame(self._history_view, title="Codewright", style="class:frame"),
                Window(
                    height=1,
                    content=FormattedTextControl(self._status_fragments),
                    style="class:status",
                ),
                Window(
                    height=3,
                    content=FormattedTextControl(self._approval_fragments),
                    style="class:approval",
                    wrap_lines=True,
                ),
                self._input_view,
            ]
        )
        app = Application(
            layout=Layout(root, focused_element=self._input_view),
            key_bindings=install_keybindings(self),
            full_screen=self._full_screen,
            erase_when_done=False,
            input=self._pt_input,
            output=self._pt_output,
            mouse_support=False,
            style=Style.from_dict(
                {
                    "frame": "ansicyan",
                    "history": "",
                    "input": "",
                    "status": "reverse bold",
                    "approval": "ansiyellow",
                    "approval.key": "ansigreen bold",
                    "approval.danger": "ansired bold",
                    "warning": "ansiyellow",
                    "error": "ansired",
                }
            ),
        )
        return app

    def _accept_input(self, buffer: Buffer) -> bool:
        stripped = buffer.text.strip()
        buffer.reset()
        if not stripped:
            return True
        if self._pending_approval is not None:
            self._append_history(
                Text(
                    "[approval pending] press y/s/n/a before sending a new prompt",
                    style="yellow",
                )
            )
            return True

        self._append_history(Text(f"> {stripped}", style="bold"))
        self._submit_op(OpUserTurn(items=[UserInputText(text=stripped)]))
        self._invalidate()
        return True

    def _append_history(self, text: Text) -> None:
        self._history.append(text)
        if len(self._history) > _MAX_HISTORY_LINES:
            del self._history[: len(self._history) - _MAX_HISTORY_LINES]
        self._refresh_history_view()
        self._invalidate()

    def _history_text(self) -> str:
        if not self._history:
            return "(no history yet)"
        return "\n".join(t.plain for t in self._history)

    def _refresh_history_view(self) -> None:
        if self._history_view is None:
            return
        self._history_view.text = self._history_text()
        self._history_view.buffer.cursor_position = len(self._history_view.text)

    def _status_fragments(self) -> list[tuple[str, str]]:
        text = self._status.render().plain
        if self._delta_buf:
            text += " | streaming"
        return [("class:status", text)]

    def _approval_fragments(self) -> list[tuple[str, str]]:
        pending = self._pending_approval
        if pending is None:
            return [
                ("class:approval", "Ctrl-C interrupt | Ctrl-D exit"),
            ]

        raw_details = render_action(pending.action)
        details = raw_details.replace("\r\n", "\n")
        details = details[:_MAX_APPROVAL_PREVIEW_CHARS]
        if len(raw_details) > _MAX_APPROVAL_PREVIEW_CHARS:
            details += "..."
        details = details.replace("\n", " | ")
        return [
            ("class:approval.danger", "Approval required: "),
            ("class:approval", details),
            ("", "\n"),
            ("class:approval.key", "y"),
            ("class:approval", "=yes "),
            ("class:approval.key", "s"),
            ("class:approval", "=session "),
            ("class:approval.danger", "n"),
            ("class:approval", "=no "),
            ("class:approval.danger", "a"),
            ("class:approval", "=abort"),
        ]

    def _invalidate(self) -> None:
        if self._application is not None:
            self._application.invalidate()

    def _submit_op(self, op: Op) -> None:
        self._spawn(self.session.submit(op))

    def _spawn(self, coro: Awaitable[Any]) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._on_bg_task_done)
        return task

    def _on_bg_task_done(self, task: asyncio.Task[Any]) -> None:
        self._bg_tasks.discard(task)
        if task.cancelled():
            return
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            self._append_history(Text(f"[background error] {exc}", style="red"))

    async def _event_consumer(self) -> None:
        try:
            while not self._shutting_down:
                ev = await self.session.next_event()
                await self._handle_event(ev.msg)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._append_history(Text(f"[event consumer error] {exc}", style="red"))

    async def _handle_event(self, msg: Any) -> None:
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
                Text(f"  -> {msg.tool_name}(call_id={msg.call_id})", style="yellow")
            )
        elif isinstance(msg, EvToolCallCompleted):
            color = "green" if msg.success else "red"
            tag = "ok" if msg.success else "fail"
            preview = msg.body[:200].replace("\n", " ")
            self._append_history(Text(f"    [{tag}] {preview}", style=color))
        elif isinstance(msg, EvPlanUpdate):
            lines = "\n".join(f"  - [{p.status.value}] {p.step}" for p in msg.plan)
            self._append_history(Text(f"plan:\n{lines}", style="cyan"))
        elif isinstance(msg, EvTokenCount):
            self._status.input_tokens = msg.input
            self._status.output_tokens = msg.output
        elif isinstance(msg, EvExecApprovalRequest):
            self._start_approval(msg.request_id, msg.action, is_exec=True)
        elif isinstance(msg, EvPatchApprovalRequest):
            self._start_approval(msg.request_id, msg.action, is_exec=False)
        elif isinstance(msg, EvTurnCompleted):
            self._status.turn_state = "idle"
        elif isinstance(msg, EvTurnAborted):
            self._status.turn_state = "idle"
            self._append_history(Text(f"[turn aborted: {msg.reason}]", style="yellow"))
        elif isinstance(msg, EvWarning):
            self._append_history(Text(f"[warn] {msg.message}", style="yellow"))
        elif isinstance(msg, EvError):
            self._append_history(Text(f"[error] {msg.message}", style="red"))
        elif isinstance(msg, EvShutdownComplete):
            self._shutting_down = True
            self._exit_application()
        self._invalidate()

    def _exit_application(self) -> None:
        if self._application is None or not self._application.is_running:
            return
        self._application.exit()

    def _start_approval(
        self,
        request_id: str,
        action: PendingAction,
        *,
        is_exec: bool,
    ) -> None:
        pending = _PendingApproval(
            request_id=request_id,
            action=action,
            is_exec=is_exec,
        )
        self._status.pending_approvals += 1
        if self._pending_approval is None:
            self._pending_approval = pending
        else:
            self._approval_queue.append(pending)
        self._append_history(Text(f"[approval required] {action.summary}", style="yellow"))


__all__ = ["TuiApp"]
