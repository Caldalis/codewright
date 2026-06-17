from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from contextlib import suppress
from dataclasses import dataclass
from io import StringIO
from typing import TYPE_CHECKING, Any

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.formatted_text.utils import split_lines
from prompt_toolkit.input.base import Input
from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl, UIContent, UIControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.margins import Margin
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.output.base import Output
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import TextArea
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


_MAX_HISTORY_ITEMS = 500
_MAX_APPROVAL_PREVIEW_CHARS = 600
_MAX_STREAM_PREVIEW_CHARS = 240
_MOUSE_SCROLL_LINES = 4

_BLOCK_HEADINGS = {
    "approval": "! Approval",
    "assistant": "< Codewright",
    "error": "x Error",
    "plan": "# Plan",
    "session": "# Session",
    "system": "# System",
    "tool": "# Tool",
    "warning": "! Warning",
    "you": "> You",
}

_BLOCK_HEADING_STYLES = {
    "approval": "bold yellow",
    "assistant": "bold green",
    "error": "bold red",
    "plan": "bold cyan",
    "session": "dim",
    "system": "dim",
    "tool": "bold magenta",
    "warning": "bold yellow",
    "you": "bold cyan",
}

_BLOCK_BODY_STYLES = {
    "approval": "yellow",
    "error": "red",
    "session": "dim",
    "system": "dim",
    "warning": "yellow",
}


@dataclass
class _PendingApproval:
    request_id: str
    action: PendingAction
    is_exec: bool


class _TranscriptControl(UIControl):
    def __init__(self, owner: TuiApp) -> None:
        self._owner = owner

    def create_content(self, width: int, height: int) -> UIContent:
        self._owner._set_terminal_width(width)
        lines = self._owner._render_history_lines(width)
        visible_height = max(1, height or len(lines) or 1)
        self._owner._set_history_layout(
            line_count=len(lines),
            visible_height=visible_height,
        )
        top = self._owner._history_top_line()
        visible = lines[top : top + visible_height] or [[]]

        return UIContent(
            get_line=lambda index: visible[index] if index < len(visible) else [],
            line_count=len(visible),
            show_cursor=False,
        )

    def mouse_handler(self, mouse_event: MouseEvent) -> object | None:
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self._owner.scroll_history_up(_MOUSE_SCROLL_LINES)
            return None
        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self._owner.scroll_history_down(_MOUSE_SCROLL_LINES)
            return None
        return NotImplemented

class _HistoryScrollbarMargin(Margin):

    def __init__(self, owner: TuiApp) -> None:
        self._owner = owner

    def get_width(self, get_ui_content: Any) -> int:
        return 1

    def create_margin(
        self, window_render_info: Any, width: int, height: int
    ) -> list[tuple[str, str]]:
        if height <= 0:
            return []
        owner = self._owner
        total = max(1, owner._history_content_line_count)
        view = max(1, owner._history_view_height)

        fragments: list[tuple[str, str]] = []
        if total <= view:
            for i in range(height):
                fragments.append(("class:scrollbar.track", " "))
                if i < height - 1:
                    fragments.append(("", "\n"))
            return fragments

        top = owner._history_top_line()
        max_top = max(1, total - view)
        thumb = max(1, min(height, round(height * view / total)))
        pos = round((height - thumb) * top / max_top)
        pos = max(0, min(pos, height - thumb))
        for i in range(height):
            if pos <= i < pos + thumb:
                fragments.append(("class:scrollbar.button", "█"))
            else:
                fragments.append(("class:scrollbar.track", "│"))
            if i < height - 1:
                fragments.append(("", "\n"))
        return fragments

class TuiApp:
    def __init__(
        self,
        session: Session,
        console: Console | None = None,
        *,
        pt_input: Input | None = None,
        pt_output: Output | None = None,
        full_screen: bool = True,
    ) -> None:
        self.session = session
        self._console = console or Console()
        self._pt_input = pt_input
        self._pt_output = pt_output
        self._full_screen = full_screen
        self._history: list[Text] = []
        self._history_rich: Text = Text()
        self._history_plain: str = ""
        self._history_version: int = 0
        self._history_render_cache_key: tuple[int, int] | None = None
        self._history_rendered_lines: list[list[tuple[str, str]]] = []
        self._history_content_line_count: int = 1
        self._history_view_height: int = 1
        self._history_top_line_target: int = 0
        self._history_follow_tail: bool = True
        self._terminal_width: int = 100
        self._status_details_visible: bool = False
        self._delta_buf: str = ""
        self._status = StatusBarState(
            model=session.model,
            cwd=str(session.cwd),
        )
        self._shutting_down: bool = False

        self._application: Application | None = None
        self._event_task: asyncio.Task[None] | None = None
        self._bg_tasks: set[asyncio.Task[Any]] = set()
        self._history_view: Window | None = None
        self._input_view: TextArea | None = None
        self._pending_approval: _PendingApproval | None = None
        self._approval_queue: list[_PendingApproval] = []
        self._rebuild_history_cache()

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
        self._append_history(
            self._block("approval", f"decision: {decision.value}", body_style="dim")
        )

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
        self._history_view = Window(
            content=_TranscriptControl(self),
            always_hide_cursor=True,
            style="class:history",
            right_margins=[_HistoryScrollbarMargin(self)],
        )
        self._input_view = TextArea(
            height=Dimension(min=1, max=6),
            multiline=True,
            wrap_lines=True,
            dont_extend_height=True,
            prompt="> ",
            accept_handler=self._accept_input,
            style="class:input",
        )

        root = HSplit(
            [
                self._history_view,
                ConditionalContainer(
                    Window(
                        height=3,
                        content=FormattedTextControl(self._approval_fragments),
                        style="class:approval",
                        wrap_lines=True,
                    ),
                    filter=Condition(lambda: self._pending_approval is not None),
                ),
                self._input_view,
                ConditionalContainer(
                    Window(
                        height=1,
                        content=FormattedTextControl(self._status_detail_fragments),
                        style="class:status.detail",
                    ),
                    filter=Condition(
                        lambda: self._status_details_visible
                        and self._pending_approval is None
                    ),
                ),
                Window(
                    height=1,
                    content=FormattedTextControl(self._status_fragments),
                    style="class:status",
                ),
            ]
        )
        app = Application(
            layout=Layout(root, focused_element=self._input_view),
            key_bindings=install_keybindings(self),
            full_screen=self._full_screen,
            erase_when_done=False,
            input=self._pt_input,
            output=self._pt_output,
            mouse_support=True,
            style=Style.from_dict(
                {
                    "approval": "bg:#3a2f00 #facc15",
                    "approval.danger": "bold #f87171 bg:#3a2f00",
                    "approval.key": "bold #86efac bg:#3a2f00",
                    "history": "",
                    "input": "#d7d7d7",
                    "scrollbar.track": "#3a3a3a",
                    "scrollbar.button": "#7dd3fc",
                    "status": "bg:#202124 #a8a8a8",
                    "status.brand": "bold #7dd3fc bg:#202124",
                    "status.detail": "bg:#181818 #a8a8a8",
                    "status.detail.key": "bold #d7d7d7 bg:#181818",
                    "status.key": "bold #d7d7d7 bg:#202124",
                    "status.warn": "bold #facc15 bg:#202124",
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
                self._block(
                    "system",
                    "Approval is waiting. Press y, s, n, or a before sending a new prompt.",
                )
            )
            return True

        self.scroll_history_to_bottom()
        self._append_history(self._block("you", stripped))
        self._submit_op(OpUserTurn(items=[UserInputText(text=stripped)]))
        self._invalidate()
        return True

    def _append_history(self, text: Text) -> None:
        self._history.append(text)
        if len(self._history) > _MAX_HISTORY_ITEMS:
            del self._history[: len(self._history) - _MAX_HISTORY_ITEMS]
        self._rebuild_history_cache()
        self._invalidate()

    def _history_text(self) -> str:
        return self._history_plain

    def _refresh_history_view(self) -> None:
        self._rebuild_history_cache()

    def scroll_history_up(self, amount: int) -> None:
        current = self._history_top_line()
        self._history_follow_tail = False
        self._history_top_line_target = max(0, current - max(1, amount))
        self._invalidate()

    def scroll_history_down(self, amount: int) -> None:
        current = self._history_top_line()
        max_top = self._history_max_top_line()
        next_top = min(max_top, current + max(1, amount))
        if next_top >= max_top:
            self.scroll_history_to_bottom()
            return
        self._history_follow_tail = False
        self._history_top_line_target = next_top
        self._invalidate()

    def scroll_history_page_up(self) -> None:
        self.scroll_history_up(max(1, self._history_view_height - 2))

    def scroll_history_page_down(self) -> None:
        self.scroll_history_down(max(1, self._history_view_height - 2))

    def scroll_history_to_top(self) -> None:
        self._history_follow_tail = False
        self._history_top_line_target = 0
        self._invalidate()

    def scroll_history_to_bottom(self) -> None:
        self._history_follow_tail = True
        self._history_top_line_target = self._history_max_top_line()
        self._invalidate()

    @property
    def input_is_empty(self) -> bool:
        if self._input_view is None:
            return True
        return not self._input_view.text

    def toggle_status_details(self) -> None:
        self._status_details_visible = not self._status_details_visible
        self._invalidate()

    def _status_fragments(self) -> list[tuple[str, str]]:
        width = self._status_width()
        if self._pending_approval is not None:
            return self._approval_status_fragments(width)

        parts: list[list[tuple[str, str]]] = [
            [("class:status.brand", " Codewright")],
            [("class:status", self._status.turn_state)],
            [("class:status.key", "tok "), ("class:status", self._token_text())],
        ]
        if width >= 52:
            parts.insert(1, [("class:status", self._compact_model(width))])
        if self._status.pending_approvals:
            parts.append(
                [
                    ("class:status.warn", "approval "),
                    ("class:status", str(self._status.pending_approvals)),
                ]
            )
        if self._status.subagent_count:
            parts.append(
                [
                    ("class:status.key", "agents "),
                    ("class:status", str(self._status.subagent_count)),
                ]
            )
        scroll_text = self._history_scroll_status()
        if scroll_text:
            parts.append([("class:status.warn", scroll_text)])
        if self._delta_buf:
            preview = self._single_line(
                self._delta_buf,
                max_chars=min(_MAX_STREAM_PREVIEW_CHARS, max(24, width // 3)),
            )
            parts.append(
                [
                    ("class:status.key", "stream "),
                    ("class:status", preview),
                ]
            )
        if width >= 68:
            parts.append([("class:status.key", "F1 "), ("class:status", "details")])
        if width >= 84:
            parts.append([("class:status.key", "Ctrl-C "), ("class:status", "interrupt")])
            parts.append([("class:status.key", "Ctrl-D "), ("class:status", "exit")])

        return self._fit_status_parts(parts, width)

    def _status_detail_fragments(self) -> list[tuple[str, str]]:
        width = self._status_width()
        model_chars = max(12, min(32, width // 3))
        cwd_chars = max(12, min(72, width // 3))
        scroll_text = self._history_scroll_status() or "bottom"
        parts: list[list[tuple[str, str]]] = [
            [
                ("class:status.detail.key", "model "),
                (
                    "class:status.detail",
                    self._single_line(self._status.model, max_chars=model_chars),
                ),
            ],
            [
                ("class:status.detail.key", "cwd "),
                ("class:status.detail", self._compact_path(self._status.cwd, max_chars=cwd_chars)),
            ],
            [
                ("class:status.detail.key", "tokens "),
                ("class:status.detail", self._token_text()),
            ],
            [
                ("class:status.detail.key", "scroll "),
                ("class:status.detail", scroll_text),
            ],
            [
                ("class:status.detail.key", "PgUp/PgDn "),
                ("class:status.detail", "scroll"),
            ],
            [
                ("class:status.detail.key", "Ctrl-End "),
                ("class:status.detail", "bottom"),
            ],
            [
                ("class:status.detail.key", "F1 "),
                ("class:status.detail", "hide"),
            ],
        ]
        return self._fit_status_parts(
            parts,
            width,
            base_style="class:status.detail",
        )

    def _approval_status_fragments(self, width: int) -> list[tuple[str, str]]:
        parts: list[list[tuple[str, str]]] = [
            [("class:status.warn", "Approval required")],
            [("class:status.key", "y "), ("class:status", "yes")],
            [("class:status.warn", "n "), ("class:status", "no")],
            [("class:status.warn", "a "), ("class:status", "abort")],
            [("class:status.key", "s "), ("class:status", "session")],
        ]
        return self._fit_status_parts(parts, width)

    def _fit_status_parts(
        self,
        parts: list[list[tuple[str, str]]],
        width: int,
        *,
        base_style: str = "class:status",
    ) -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        used = 0
        for part in parts:
            chunk = ([(base_style, " | ")] if result else []) + part
            chunk_width = self._fragments_width(chunk)
            if used + chunk_width <= width:
                result.extend(chunk)
                used += chunk_width
        if result:
            return result
        if not parts:
            return [(base_style, "")]
        return self._truncate_fragments(parts[0], width)

    @staticmethod
    def _fragments_width(fragments: list[tuple[str, str]]) -> int:
        return sum(get_cwidth(text) for _, text in fragments)

    @staticmethod
    def _truncate_fragments(
        fragments: list[tuple[str, str]], width: int
    ) -> list[tuple[str, str]]:
        if width <= 0:
            return []
        text = "".join(fragment for _, fragment in fragments)
        if get_cwidth(text) <= width:
            return fragments
        style = fragments[0][0] if fragments else ""
        if width <= 3:
            return [(style, "." * width)]
        return [(style, text[: width - 3] + "...")]

    def _set_terminal_width(self, width: int) -> None:
        self._terminal_width = max(20, width)

    def _status_width(self) -> int:
        if self._application is not None:
            with suppress(Exception):
                return max(20, self._application.output.get_size().columns)
        return max(20, self._terminal_width)

    def _compact_model(self, width: int) -> str:
        max_chars = 20 if width < 76 else 32
        return self._single_line(self._status.model, max_chars=max_chars)

    def _token_text(self) -> str:
        return (
            f"{self._compact_number(self._status.input_tokens)}/"
            f"{self._compact_number(self._status.output_tokens)}"
        )

    @staticmethod
    def _compact_number(value: int) -> str:
        if value >= 1_000_000:
            return f"{value // 1_000_000}m"
        if value >= 1_000:
            return f"{value // 1_000}k"
        return str(value)

    def _approval_fragments(self) -> list[tuple[str, str]]:
        pending = self._pending_approval
        if pending is None:
            return []

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
            self._append_history(
                self._block("error", f"background task failed: {exc}")
            )

    async def _event_consumer(self) -> None:
        try:
            while not self._shutting_down:
                ev = await self.session.next_event()
                await self._handle_event(ev.msg)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._append_history(self._block("error", f"event consumer failed: {exc}"))

    async def _handle_event(self, msg: Any) -> None:
        if isinstance(msg, EvSessionConfigured):
            self._append_history(
                self._block(
                    "session",
                    f"session {msg.session_id} (model={msg.model}, cwd={msg.cwd})",
                )
            )
        elif isinstance(msg, EvTurnStarted):
            self._status.turn_state = "running"
            self._delta_buf = ""
        elif isinstance(msg, EvAgentMessageDelta):
            self._delta_buf += msg.delta
        elif isinstance(msg, EvAgentMessage):
            self._delta_buf = ""
            self._append_history(self._block("assistant", msg.content))
        elif isinstance(msg, EvToolCallStarted):
            detail = f"{msg.tool_name} started (call_id={msg.call_id})"
            if msg.arguments:
                detail += " " + self._single_line(
                    self._format_tool_args(msg.arguments), max_chars=200
                )
            self._append_history(self._block("tool", detail))
        elif isinstance(msg, EvToolCallCompleted):
            tag = "ok" if msg.success else "fail"
            preview = self._single_line(msg.body, max_chars=240)
            style = "green" if msg.success else "red"
            self._append_history(
                self._block("tool", f"[{tag}] {preview}", body_style=style)
            )
        elif isinstance(msg, EvPlanUpdate):
            lines = "\n".join(
                f"{self._plan_marker(p.status.value)} {p.step}" for p in msg.plan
            )
            if msg.explanation:
                lines = f"{msg.explanation}\n{lines}"
            self._append_history(self._block("plan", lines))
        elif isinstance(msg, EvTokenCount):
            self._status.input_tokens = msg.input
            self._status.output_tokens = msg.output
        elif isinstance(msg, EvExecApprovalRequest):
            self._start_approval(msg.request_id, msg.action, is_exec=True)
        elif isinstance(msg, EvPatchApprovalRequest):
            self._start_approval(msg.request_id, msg.action, is_exec=False)
        elif isinstance(msg, EvTurnCompleted):
            self._status.turn_state = "idle"
            self._delta_buf = ""
        elif isinstance(msg, EvTurnAborted):
            self._status.turn_state = "idle"
            self._delta_buf = ""
            self._clear_pending_approvals()
            self._append_history(self._block("system", f"turn aborted: {msg.reason}"))
        elif isinstance(msg, EvWarning):
            self._append_history(self._block("warning", msg.message))
        elif isinstance(msg, EvError):
            self._append_history(self._block("error", msg.message))
        elif isinstance(msg, EvShutdownComplete):
            self._shutting_down = True
            self._clear_pending_approvals()
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
        self._append_history(self._block("approval", action.summary))

    def _clear_pending_approvals(self) -> None:
        self._pending_approval = None
        self._approval_queue.clear()
        self._status.pending_approvals = 0

    def _block(self, label: str, body: str, *, body_style: str = "") -> Text:
        body = body.strip() or "<empty>"
        heading = _BLOCK_HEADINGS.get(label, label.upper())
        heading_style = _BLOCK_HEADING_STYLES.get(label, "bold")
        resolved_body_style = body_style or _BLOCK_BODY_STYLES.get(label, "")

        text = Text()
        text.append(f"{heading}\n", style=heading_style)
        for line in body.splitlines():
            text.append("  ", style="dim")
            text.append(f"{line}\n", style=resolved_body_style)
        return text

    def _rebuild_history_cache(self) -> None:
        merged = Text()
        if not self._history:
            merged.append("Codewright", style="bold cyan")
            merged.append("\n  session ready", style="dim")
        else:
            for index, block in enumerate(self._history):
                if index:
                    merged.append("\n")
                merged.append_text(block)

        self._history_rich = merged
        self._history_plain = merged.plain
        self._history_version += 1
        self._history_render_cache_key = None

    def _render_history_lines(self, width: int) -> list[list[tuple[str, str]]]:
        render_width = max(20, width)
        cache_key = (self._history_version, render_width)
        if self._history_render_cache_key == cache_key:
            return self._history_rendered_lines

        output = StringIO()
        console = Console(
            file=output,
            force_terminal=True,
            color_system="standard",
            legacy_windows=False,
            width=render_width,
        )
        console.print(
            self._history_rich,
            end="",
            highlight=False,
            markup=False,
            soft_wrap=False,
        )
        fragments = ANSI(output.getvalue()).__pt_formatted_text__()
        lines = [list(line) for line in split_lines(fragments)] or [[("", "")]]

        self._history_render_cache_key = cache_key
        self._history_rendered_lines = lines
        return lines

    def _set_history_layout(self, *, line_count: int, visible_height: int) -> None:
        self._history_content_line_count = max(1, line_count)
        self._history_view_height = max(1, visible_height)
        max_top = self._history_max_top_line()
        if self._history_follow_tail or max_top == 0:
            self._history_follow_tail = True
            self._history_top_line_target = max_top
        else:
            self._history_top_line_target = min(
                max(0, self._history_top_line_target),
                max_top,
            )

    def _history_top_line(self) -> int:
        max_top = self._history_max_top_line()
        if self._history_follow_tail:
            self._history_top_line_target = max_top
        else:
            self._history_top_line_target = min(
                max(0, self._history_top_line_target),
                max_top,
            )
        return self._history_top_line_target

    def _history_max_top_line(self) -> int:
        return max(0, self._history_content_line_count - self._history_view_height)

    def _history_scroll_status(self) -> str:
        max_top = self._history_max_top_line()
        if max_top == 0:
            return ""
        top = self._history_top_line()
        if self._history_follow_tail:
            return ""
        return f"history {top + 1}/{max_top + 1}"

    @staticmethod
    def _single_line(text: str, *, max_chars: int) -> str:
        line = " ".join(text.split())
        if len(line) <= max_chars:
            return line
        return line[: max(0, max_chars - 3)] + "..."

    @staticmethod
    def _format_tool_args(arguments: dict[str, Any]) -> str:
        # Compact key=value for the start line; _single_line flattens newlines
        # and caps total length so a large patch/command can't flood the view.
        return ", ".join(f"{k}={v}" for k, v in arguments.items())

    @staticmethod
    def _compact_path(path: str, *, max_chars: int) -> str:
        if len(path) <= max_chars:
            return path
        keep = max(8, max_chars - 3)
        return "..." + path[-keep:]

    @staticmethod
    def _plan_marker(status: str) -> str:
        if status == "completed":
            return "[x]"
        if status == "in_progress":
            return "[>]"
        return "[ ]"


__all__ = ["TuiApp"]
