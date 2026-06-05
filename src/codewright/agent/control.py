from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from codewright.context.manager import ContextManager
from codewright.protocol import (
    EvAgentMessage,
    EvShutdownComplete,
    EvTurnAborted,
    EvTurnCompleted,
    EvTurnStarted,
    InterAgentMessage,
    OpUserTurn,
    UserInputText,
)
from codewright.protocol.agent_messages import AgentPath

if TYPE_CHECKING:  # pragma: no cover
    from codewright.agent.session import Session


AgentStatus = Literal["running", "completed", "failed", "closed"]


@dataclass
class AgentInfo:

    path: AgentPath
    role: str
    task_name: str
    status: AgentStatus = "running"
    last_message: str | None = None


@dataclass
class _AgentEntry:
    info: AgentInfo
    session: Session
    watcher: asyncio.Task[None] | None = None

    terminal_event: asyncio.Event = field(default_factory=asyncio.Event)


class AgentControl:

    def __init__(self, root_session: Session) -> None:
        self._root = root_session
        self._agents: dict[str, _AgentEntry] = {}

    @property
    def root_session(self) -> Session:
        return self._root

    def list_agents(self) -> list[AgentInfo]:
        return [entry.info for entry in self._agents.values()]

    def has(self, path: AgentPath) -> bool:
        return str(path) in self._agents

    def get_info(self, path: AgentPath) -> AgentInfo:
        try:
            return self._agents[str(path)].info
        except KeyError as exc:
            raise KeyError(f"unknown agent: {path}") from exc

    def get_session(self, path: AgentPath) -> Session:
        try:
            return self._agents[str(path)].session
        except KeyError as exc:
            raise KeyError(f"unknown agent: {path}") from exc

    async def spawn_agent(
        self,
        role: str,
        task_name: str,
        initial_message: str,
        parent_path: AgentPath | None = None,
    ) -> AgentPath:

        from codewright.agent.session import Session

        parent = parent_path or AgentPath.root()
        clean_role = _normalize_segment_part(role)
        clean_task = _normalize_segment_part(task_name)
        path = parent.child(f"{clean_role}_{clean_task}")
        if str(path) in self._agents:
            raise ValueError(f"agent path already exists: {path}")

        sub_session = Session(
            session_id=f"{self._root.session_id}:{path}",
            cwd=self._root.cwd,
            permission_profile=self._root.permission_profile,
            model=self._root.model,
            llm=self._root._llm,
            context_manager=ContextManager(
                max_context_tokens=self._root.context.max_context_tokens,
                compact_threshold=self._root.context.compact_threshold,
            )
            if self._root._context_manager is not None
            else None,
            prompt_builder=self._root._prompt_builder,
            approval_policy=self._root.approval_policy,
            workspace=self._root._workspace,
            tool_registry=self._root.tool_registry,
            agent_path=path,
            agent_control=self,
            role=clean_role,
        )
        entry = _AgentEntry(
            info=AgentInfo(path=path, role=clean_role, task_name=clean_task),
            session=sub_session,
        )
        self._agents[str(path)] = entry
        entry.watcher = asyncio.create_task(
            self._watch(entry), name=f"agent_watch[{path}]"
        )
        await sub_session.submit(OpUserTurn(items=[UserInputText(text=initial_message)]))
        return path

    async def send_message(
        self,
        target: AgentPath,
        content: str,
        author: AgentPath | None = None,
    ) -> None:

        session = self.get_session(target)
        author_path = author or AgentPath.root()
        session.mailbox.push(
            InterAgentMessage(
                author=author_path,
                recipient=target,
                content=content,
                trigger_turn=False,
            )
        )

    async def followup_task(
        self,
        target: AgentPath,
        content: str,
        author: AgentPath | None = None,
    ) -> None:

        session = self.get_session(target)
        entry = self._agents[str(target)]
        author_path = author or AgentPath.root()
        session.mailbox.push_with_wake(
            InterAgentMessage(
                author=author_path,
                recipient=target,
                content=content,
                trigger_turn=True,
            )
        )

        entry.info.status = "running"
        entry.terminal_event.clear()
        await session.submit(OpUserTurn(items=[UserInputText(text="")]))

    async def wait_agent(
        self,
        targets: list[AgentPath],
        timeout_ms: int | None = None,
    ) -> list[AgentInfo]:

        if not targets:
            return []
        entries = [self._agents[str(t)] for t in targets]
        ready_now = [e.info for e in entries if e.info.status != "running"]
        if ready_now:
            return ready_now

        waits = [asyncio.create_task(e.terminal_event.wait()) for e in entries]
        try:
            if timeout_ms is None:
                await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
            else:
                await asyncio.wait(
                    waits,
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=timeout_ms / 1000.0,
                )
        finally:
            for w in waits:
                if not w.done():
                    w.cancel()
        return [e.info for e in entries if e.info.status != "running"]

    async def close_agent(self, target: AgentPath) -> None:
        entry = self._agents.get(str(target))
        if entry is None:
            return
        if entry.info.status == "closed":
            return
        try:
            await entry.session.shutdown()
        except Exception:
            pass
        entry.info.status = "closed"
        entry.terminal_event.set()
        if entry.watcher is not None and not entry.watcher.done():
            entry.watcher.cancel()

    async def shutdown_all(self) -> None:
        for path_str in list(self._agents.keys()):
            try:
                await self.close_agent(AgentPath.parse(path_str))
            except Exception:
                pass

    async def _watch(self, entry: _AgentEntry) -> None:
        sess = entry.session
        try:
            while True:
                ev = await sess.next_event()
                msg = ev.msg
                if isinstance(msg, EvTurnStarted):
                    entry.info.status = "running"
                    entry.terminal_event.clear()
                elif isinstance(msg, EvAgentMessage):
                    entry.info.last_message = msg.content
                elif isinstance(msg, EvTurnCompleted):
                    if msg.last_agent_message:
                        entry.info.last_message = msg.last_agent_message
                    entry.info.status = "completed"
                    entry.terminal_event.set()
                elif isinstance(msg, EvTurnAborted):
                    entry.info.status = "failed" if msg.reason == "error" else "completed"
                    entry.terminal_event.set()
                elif isinstance(msg, EvShutdownComplete):
                    entry.info.status = "closed"
                    entry.terminal_event.set()
                    return
        except asyncio.CancelledError:
            return
        except Exception:
            entry.info.status = "failed"
            entry.terminal_event.set()


def _normalize_segment_part(text: str) -> str:

    out: list[str] = []
    last_underscore = False
    for ch in text.lower():
        if ch.isalnum() or ch == "_":
            out.append(ch)
            last_underscore = ch == "_"
        elif not last_underscore:
            out.append("_")
            last_underscore = True
    cleaned = "".join(out).strip("_")
    return cleaned or "task"


__all__ = ["AgentControl", "AgentInfo", "AgentStatus"]
