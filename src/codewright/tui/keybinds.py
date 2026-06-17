from __future__ import annotations

from typing import TYPE_CHECKING

from prompt_toolkit.filters import Condition, has_focus
from prompt_toolkit.key_binding import KeyBindings

from codewright.protocol import ReviewDecision

if TYPE_CHECKING:  # pragma: no cover
    from codewright.tui.app import TuiApp


def install_keybindings(app: TuiApp) -> KeyBindings:
    kb = KeyBindings()
    approval_active = Condition(lambda: app.has_pending_approval)
    input_empty = Condition(lambda: app.input_is_empty)
    input_focused = has_focus(app._input_view.buffer)
    @kb.add("enter", filter=input_focused, eager=True)
    def _(event) -> None:
        if event.current_buffer.text.strip():
            event.current_buffer.validate_and_handle()

    @kb.add("c-j", filter=input_focused)
    def _(event) -> None:
        event.current_buffer.insert_text("\n")

    @kb.add("c-c")
    def _(event) -> None:
        app.request_interrupt()

    @kb.add("c-d")
    def _(event) -> None:
        app.request_shutdown()

    @kb.add("pageup")
    def _(event) -> None:
        app.scroll_history_page_up()

    @kb.add("pagedown")
    def _(event) -> None:
        app.scroll_history_page_down()

    @kb.add("c-home")
    def _(event) -> None:
        app.scroll_history_to_top()

    @kb.add("c-end")
    def _(event) -> None:
        app.scroll_history_to_bottom()

    @kb.add("f1")
    def _(event) -> None:
        app.toggle_status_details()

    @kb.add("?", filter=input_empty)
    def _(event) -> None:
        app.toggle_status_details()

    @kb.add("y", filter=approval_active)
    def _(event) -> None:
        app.submit_approval(ReviewDecision.APPROVED)

    @kb.add("s", filter=approval_active)
    def _(event) -> None:
        app.submit_approval(ReviewDecision.APPROVED_FOR_SESSION)

    @kb.add("n", filter=approval_active)
    def _(event) -> None:
        app.submit_approval(ReviewDecision.DENIED)

    @kb.add("a", filter=approval_active)
    def _(event) -> None:
        app.submit_approval(ReviewDecision.ABORT)

    return kb


__all__ = ["install_keybindings"]
