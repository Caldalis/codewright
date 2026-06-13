from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codewright.agent.approval import ApprovalBroker
from codewright.agent.cancellation import CancellationToken
from codewright.agent.mailbox import Mailbox
from codewright.agent.rwlock import AsyncRwLock
from codewright.agent.turn_cancellation import TurnCancellationManager
from codewright.protocol import (
    AgentPath,
    AskForApproval,
    Event,
    EventMsg,
    EvSessionConfigured,
    EvWarning,
    Op,
    OpExecApprovalResponse,
    OpInterrupt,
    OpPatchApprovalResponse,
    OpUserTurn,
    PendingAction,
    PermissionProfile,
    PlanItem,
    ReviewDecision,
    Submission,
)
from codewright.tools.executor import ToolExecutor
from codewright.tools.registry import ToolRegistry
from codewright.tools.router import ToolRouter
from codewright.tools.spec import ToolSpec

if TYPE_CHECKING:  # pragma: no cover
    from codewright.agent.control import AgentControl
    from codewright.context.manager import ContextManager
    from codewright.context.summarizer import Summarizer
    from codewright.llm.base import LLMProvider
    from codewright.persistence.rollout import RolloutRecorder
    from codewright.prompts.builder import PromptBuilder
    from codewright.workspace.manager import WorkspaceManager


_DEFAULT_QUEUE_MAXSIZE = 0


class Session:


    def __init__(
        self,
        session_id: str,
        cwd: Path,
        permission_profile: PermissionProfile,
        model: str = "mock",
        queue_maxsize: int = _DEFAULT_QUEUE_MAXSIZE,
        *,
        llm: LLMProvider | None = None,
        context_manager: ContextManager | None = None,
        prompt_builder: PromptBuilder | None = None,
        approval_policy: AskForApproval = AskForApproval.ON_REQUEST,
        workspace: WorkspaceManager | None = None,
        tool_registry: ToolRegistry | None = None,
        summarizer: Summarizer | None = None,
        rollout: RolloutRecorder | None = None,
        mailbox: Mailbox | None = None,
        agent_path: AgentPath | None = None,
        agent_control: AgentControl | None = None,
        role: str = "default",
    ) -> None:
        from codewright.agent.submission_loop import submission_loop

        self.session_id = session_id
        self.cwd = cwd
        self.permission_profile = permission_profile
        self.model = model
        self.approval_policy = approval_policy

        self._llm = llm
        self._context_manager = context_manager
        self._prompt_builder = prompt_builder
        self._workspace = workspace
        self._summarizer = summarizer
        self._rollout = rollout

        self._mcp: Any | None = None


        self._tool_registry = tool_registry or ToolRegistry()
        self._tool_router = ToolRouter(self._tool_registry)
        self._tool_executor = ToolExecutor(self._tool_registry, AsyncRwLock())


        self.plan: list[PlanItem] = []

        self._mailbox: Mailbox = mailbox or Mailbox()
        self._agent_path: AgentPath = agent_path or AgentPath.root()
        self._agent_control: AgentControl | None = agent_control
        self.role: str = role

        self._tx_sub: asyncio.Queue[Submission] = asyncio.Queue(maxsize=queue_maxsize)
        self._tx_event: asyncio.Queue[Event] = asyncio.Queue()
        self._cancellation_token = CancellationToken()
        self._turn_cancellation = TurnCancellationManager(self._cancellation_token)
        self._approvals = ApprovalBroker(
            emit_event=self.emit_event,
            cancellation_tokens=self._turn_cancellation.approval_tokens,
        )
        self._shutdown_complete = asyncio.Event()

        self._loop_task: asyncio.Task[None] = asyncio.create_task(
            submission_loop(self), name=f"submission_loop[{session_id}]"
        )

        self._tx_event.put_nowait(
            Event(
                id="",
                msg=EvSessionConfigured(
                    session_id=session_id,
                    model=model,
                    cwd=str(cwd),
                    permission_profile=permission_profile.value,
                ),
            )
        )


    async def submit(self, op: Op) -> str:
        sub_id = uuid.uuid4().hex
        submission = Submission(id=sub_id, op=op)
        await self.submit_with_id(submission)
        return sub_id

    async def submit_with_id(self, submission: Submission) -> None:
        if await self._handle_out_of_band_op(submission):
            return
        if isinstance(submission.op, OpUserTurn):
            self._turn_cancellation.prepare(submission.id)
        await self._tx_sub.put(submission)

    async def _handle_out_of_band_op(self, sub: Submission) -> bool:
        op = sub.op

        if isinstance(op, OpInterrupt):
            self._turn_cancellation.cancel_current_or_queued()
            return True
        if isinstance(op, OpExecApprovalResponse):
            await self._approvals.resolve_exec_response(
                op.request_id, op.decision, sub.id
            )
            return True
        if isinstance(op, OpPatchApprovalResponse):
            await self._approvals.resolve_patch_response(
                op.request_id, op.decision, sub.id
            )
            return True
        return False

    async def next_event(self) -> Event:
        return await self._tx_event.get()

    async def emit_event(self, msg: EventMsg, sub_id: str = "") -> None:
        await self._tx_event.put(Event(id=sub_id, msg=msg))


    @property
    def llm(self) -> LLMProvider:
        if self._llm is None:
            raise RuntimeError(
                "Session was constructed without an LLMProvider; "
                "OpUserTurn cannot drive run_turn. Provide llm= to enable."
            )
        return self._llm

    @property
    def context(self) -> ContextManager:
        if self._context_manager is None:
            raise RuntimeError(
                "Session was constructed without a ContextManager; "
                "Provide context_manager= to enable."
            )
        return self._context_manager

    @property
    def prompt_builder(self) -> PromptBuilder:
        if self._prompt_builder is None:
            raise RuntimeError(
                "Session was constructed without a PromptBuilder; "
                "Provide prompt_builder= to enable."
            )
        return self._prompt_builder

    @property
    def llm_enabled(self) -> bool:
        return (
            self._llm is not None
            and self._context_manager is not None
            and self._prompt_builder is not None
        )

    @property
    def summarizer(self) -> Summarizer:

        if self._summarizer is None:
            from codewright.context.compact import LLMSummarizer

            self._summarizer = LLMSummarizer(self.llm)
        return self._summarizer

    @property
    def rollout(self) -> RolloutRecorder | None:
        return self._rollout

    @property
    def mcp(self) -> Any | None:
        return self._mcp

    @property
    def mailbox(self) -> Mailbox:
        return self._mailbox

    @property
    def agent_path(self) -> AgentPath:
        return self._agent_path

    @property
    def agent_control(self) -> AgentControl:

        if self._agent_control is None:
            from codewright.agent.control import AgentControl as _AC

            self._agent_control = _AC(self)
        return self._agent_control

    @property
    def has_agent_control(self) -> bool:
        return self._agent_control is not None

    async def start_mcp(self, configs: list[Any]) -> None:

        if not configs:
            return
        from codewright.mcp import McpConnectionManager
        from codewright.tools.handlers.mcp_handler import McpToolHandler

        async def _warn(message: str) -> None:
            await self.emit_event(EvWarning(message=message))

        manager = McpConnectionManager(configs)
        await manager.start(on_warning=_warn)
        for info in manager.all_tools():
            self._tool_registry.register(McpToolHandler(info, manager))
        self._mcp = manager


    @property
    def workspace(self) -> WorkspaceManager:
        if self._workspace is None:
            raise RuntimeError(
                "Session was constructed without a WorkspaceManager; "
                "Provide workspace= to enable apply_patch / run_shell."
            )
        return self._workspace

    @property
    def tool_registry(self) -> ToolRegistry:
        return self._tool_registry

    @property
    def tool_router(self) -> ToolRouter:
        return self._tool_router

    @property
    def tool_executor(self) -> ToolExecutor:
        return self._tool_executor

    def tool_specs(self) -> list[ToolSpec]:
        return self._tool_registry.specs()



    async def request_approval(self, action: PendingAction) -> ReviewDecision:
        return await self._approvals.request(action)

    @property
    def cancellation_token(self) -> CancellationToken:
        return self._cancellation_token

    @property
    def turn_cancellation(self) -> TurnCancellationManager:
        return self._turn_cancellation

    async def shutdown(self) -> None:
        self._cancellation_token.cancel()
        self._turn_cancellation.cancel_current_or_queued()
        if not self._loop_task.done():
            from codewright.protocol import OpShutdown

            try:
                await self.submit(OpShutdown())
            except RuntimeError:
                pass

        try:
            await asyncio.wait_for(self._loop_task, timeout=5.0)
        except TimeoutError:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except (asyncio.CancelledError, BaseException):
                pass
        finally:
            self._shutdown_complete.set()


        if self._agent_control is not None and self._agent_path.is_root():
            try:
                await self._agent_control.shutdown_all()
            except Exception:
                pass

        if self._agent_path.is_root():
            for handler in self._tool_registry.handlers():
                aclose = getattr(handler, "aclose", None)
                if aclose is None:
                    continue
                try:
                    await aclose()
                except Exception:
                    pass


        if self._mcp is not None:
            try:
                await self._mcp.shutdown()
            except Exception:
                pass
            self._mcp = None


        if self._rollout is not None:
            try:
                await self._rollout.shutdown()
            except Exception:
                pass


    async def record_rollout(self, line: Any) -> None:

        if self._rollout is None:
            return
        await self._rollout.append(line)

    def replay(self, lines: list[Any]) -> None:

        from codewright.llm.base import CanonicalMessage, ToolCallBlock
        from codewright.protocol import PlanItem

        for line in lines:
            kind = getattr(line, "type", None)
            payload = getattr(line, "payload", {}) or {}
            if kind == "user_msg":
                content = payload.get("content")
                if isinstance(content, str):
                    self.context.append(
                        CanonicalMessage(role="user", content=content)
                    )
            elif kind == "assistant_msg":
                content = payload.get("content") or ""
                tool_calls_raw = payload.get("tool_calls") or []
                if tool_calls_raw:
                    tool_calls = tuple(
                        ToolCallBlock(
                            call_id=tc.get("call_id", ""),
                            tool_name=tc.get("tool_name", ""),
                            arguments_json=tc.get("arguments_json", ""),
                        )
                        for tc in tool_calls_raw
                    )
                    self.context.append(
                        CanonicalMessage(
                            role="assistant",
                            content=content,
                            tool_calls=tool_calls,
                        )
                    )
                else:
                    self.context.append(
                        CanonicalMessage(role="assistant", content=content)
                    )
            elif kind == "tool_result":
                self.context.append(
                    CanonicalMessage(
                        role="tool",
                        content=payload.get("body", ""),
                        name=payload.get("tool_name"),
                        tool_call_id=payload.get("call_id"),
                    )
                )
            elif kind == "compaction":
                replacement = payload.get("replacement") or []
                rebuilt: list[CanonicalMessage] = []
                for m in replacement:
                    rebuilt.append(
                        CanonicalMessage(
                            role=m.get("role", "user"),
                            content=m.get("content", ""),
                        )
                    )
                self.context.replace_all(rebuilt)
            elif kind == "plan_update":
                items_raw = payload.get("plan") or []
                try:
                    self.plan = [PlanItem(**it) for it in items_raw]
                except Exception:
                    pass

_ = Any
