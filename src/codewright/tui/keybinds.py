from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from prompt_toolkit.key_binding import KeyBindings

from codewright.protocol import OpInterrupt, OpShutdown

if TYPE_CHECKING:  # pragma: no cover
    from codewright.tui.app import TuiApp


_BG_TASKS: set[asyncio.Task[str]] = set()


def install_keybindings(app: TuiApp) -> KeyBindings:

    kb = KeyBindings()

    def _spawn(coro):  
        task = asyncio.create_task(coro)
        _BG_TASKS.add(task)
        task.add_done_callback(_BG_TASKS.discard)
        return task

    @kb.add("c-c")
    def _(event) -> None:  
        _spawn(app.session.submit(OpInterrupt()))

    @kb.add("c-d")
    def _(event) -> None:  
        _spawn(app.session.submit(OpShutdown()))
        event.app.exit()

    return kb


__all__ = ["install_keybindings"]
