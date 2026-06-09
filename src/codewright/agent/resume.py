from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from codewright.agent.session import Session
from codewright.persistence.session_store import SessionStore
from codewright.protocol import AskForApproval, PermissionProfile

if TYPE_CHECKING:  # pragma: no cover
    from codewright.context.manager import ContextManager
    from codewright.context.summarizer import Summarizer
    from codewright.llm.base import LLMProvider
    from codewright.prompts.builder import PromptBuilder
    from codewright.workspace.manager import WorkspaceManager


async def resume_session(
    workspace_root: Path,
    session_id: str,
    *,
    llm: LLMProvider | None = None,
    context_manager: ContextManager | None = None,
    prompt_builder: PromptBuilder | None = None,
    workspace: WorkspaceManager | None = None,
    summarizer: Summarizer | None = None,
    approval_policy: AskForApproval | None = None,
    model: str | None = None,
    permission_profile: PermissionProfile | None = None,
    role: str = "default",
    mcp_configs: list[Any] | None = None,
) -> Session:

    if context_manager is None:
        from codewright.context.manager import ContextManager as _CM

        context_manager = _CM()

    store = SessionStore(workspace_root)
    meta, lines = await store.load_session(session_id)
    recorder = await store.resume_recorder(session_id)
    effective_permission_profile = permission_profile or PermissionProfile(
        meta.permission_profile
    )
    effective_model = model or meta.model

    if workspace is None:
        from codewright.workspace.manager import WorkspaceManager

        workspace = WorkspaceManager(workspace_root, effective_permission_profile)

    session = Session(
        session_id=meta.session_id,
        cwd=Path(meta.cwd),
        permission_profile=effective_permission_profile,
        model=effective_model,
        llm=llm,
        context_manager=context_manager,
        prompt_builder=prompt_builder,
        approval_policy=approval_policy
        if approval_policy is not None
        else AskForApproval.ON_REQUEST,
        workspace=workspace,
        summarizer=summarizer,
        rollout=recorder,
        role=role,
    )

    session.replay([ln for ln in lines if ln.type != "session_meta"])

    if mcp_configs:
        await session.start_mcp(mcp_configs)

    return session


__all__ = ["resume_session"]
