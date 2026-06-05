from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


RolloutLineType = Literal[
    "session_meta",
    "user_msg",
    "assistant_msg",
    "tool_call",
    "tool_result",
    "reasoning",
    "compaction",
    "agent_event",
    "approval_request",
    "approval_response",
    "turn_context",
    "plan_update",
]


class SessionMeta(BaseModel):

    model_config = ConfigDict(frozen=True)

    session_id: str
    cwd: str
    model: str
    permission_profile: str
    start_time: float
    git_root: str | None = None


class RolloutLine(BaseModel):

    model_config = ConfigDict(frozen=True)

    ts: float = Field(default_factory=time.time)
    type: RolloutLineType
    payload: dict[str, Any] = Field(default_factory=dict)


class RolloutWriteError(RuntimeError):
    """Terminal writer task failure — surfaced from `append` / `flush`."""


_AddItems = tuple[Literal["add"], list[RolloutLine]]
_Flush = tuple[Literal["flush"], asyncio.Future[None]]
_Shutdown = tuple[Literal["shutdown"], asyncio.Future[None]]
_RolloutCmd = _AddItems | _Flush | _Shutdown


class RolloutRecorder:

    _SHUTDOWN_TIMEOUT = 5.0

    def __init__(
        self,
        path: Path,
        queue: asyncio.Queue[_RolloutCmd],
        task: asyncio.Task[None],
        terminal_failure: list[BaseException | None],
    ) -> None:

        self._path = path
        self._queue = queue
        self._task = task
        self._terminal_failure = terminal_failure  # one-element mailbox
        self._shutdown_called = False

    @classmethod
    async def create(cls, path: Path, meta: SessionMeta) -> RolloutRecorder:

        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():  # noqa: ASYNC240
            raise FileExistsError(f"rollout already exists: {path}")
        recorder = cls._spawn(path)
        meta_line = RolloutLine(type="session_meta", payload=meta.model_dump())
        await recorder.append(meta_line)
        return recorder

    @classmethod
    async def resume(cls, path: Path) -> RolloutRecorder:
        if not path.exists():  # noqa: ASYNC240
            raise FileNotFoundError(f"rollout not found: {path}")
        return cls._spawn(path)

    @classmethod
    def _spawn(cls, path: Path) -> RolloutRecorder:
        queue: asyncio.Queue[_RolloutCmd] = asyncio.Queue()
        terminal_failure: list[BaseException | None] = [None]
        task = asyncio.create_task(
            _writer_loop(path, queue, terminal_failure),
            name=f"rollout_writer[{path.name}]",
        )
        return cls(path, queue, task, terminal_failure)

    @property
    def path(self) -> Path:
        return self._path

    async def append(self, line: RolloutLine) -> None:
        self._check_terminal()
        await self._queue.put(("add", [line]))

    async def append_many(self, lines: list[RolloutLine]) -> None:
        if not lines:
            return
        self._check_terminal()
        await self._queue.put(("add", list(lines)))

    async def flush(self) -> None:
        self._check_terminal()
        loop = asyncio.get_running_loop()
        ack: asyncio.Future[None] = loop.create_future()
        await self._queue.put(("flush", ack))
        await ack

    async def shutdown(self) -> None:
        if self._shutdown_called:
            return
        self._shutdown_called = True
        loop = asyncio.get_running_loop()
        ack: asyncio.Future[None] = loop.create_future()
        try:
            await self._queue.put(("shutdown", ack))
        except RuntimeError:
            return
        try:
            await asyncio.wait_for(ack, timeout=self._SHUTDOWN_TIMEOUT)
        except (TimeoutError, asyncio.CancelledError):
            self._task.cancel()
        finally:
            if not self._task.done():
                try:
                    await asyncio.wait_for(
                        self._task, timeout=self._SHUTDOWN_TIMEOUT
                    )
                except (TimeoutError, asyncio.CancelledError):
                    pass

    def _check_terminal(self) -> None:
        err = self._terminal_failure[0]
        if err is not None:
            raise RolloutWriteError(
                f"rollout writer terminated: {err!r}"
            ) from err


async def _writer_loop(
    path: Path,
    queue: asyncio.Queue[_RolloutCmd],
    terminal_failure: list[BaseException | None],
) -> None:

    try:
        fh = open(path, "a", encoding="utf-8")  # noqa: ASYNC230
    except OSError as exc:
        terminal_failure[0] = exc
        return

    try:
        while True:
            cmd = await queue.get()
            kind = cmd[0]
            if kind == "add":
                lines = cmd[1]
                try:
                    for ln in lines:

                        fh.write(ln.model_dump_json())
                        fh.write("\n")
                    fh.flush()
                except OSError as exc:
                    terminal_failure[0] = exc
                    return
            elif kind == "flush":
                ack = cmd[1]
                try:
                    fh.flush()
                    if not ack.done():
                        ack.set_result(None)
                except OSError as exc:
                    if not ack.done():
                        ack.set_exception(exc)
                    terminal_failure[0] = exc
                    return
            elif kind == "shutdown":
                ack = cmd[1]
                try:
                    fh.flush()
                    if not ack.done():
                        ack.set_result(None)
                except OSError as exc:
                    if not ack.done():
                        ack.set_exception(exc)
                    terminal_failure[0] = exc
                return
    finally:
        try:
            fh.close()
        except OSError:
            pass


def write_line_sync(path: Path, line: RolloutLine) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line.model_dump_json())
        fh.write("\n")


__all__ = [
    "RolloutLine",
    "RolloutLineType",
    "RolloutRecorder",
    "RolloutWriteError",
    "SessionMeta",
    "write_line_sync",
]
