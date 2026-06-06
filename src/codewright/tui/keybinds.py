from __future__ import annotations

from typing import TYPE_CHECKING

from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings

from codewright.protocol import ReviewDecision

if TYPE_CHECKING:  # pragma: no cover
    from codewright.tui.app import TuiApp


def install_keybindings(app: TuiApp) -> KeyBindings:

    kb = KeyBindings()
    approval_active = Condition(lambda: app.has_pending_approval)

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
