from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from codewright.agent.cancellation import CancellationToken
from codewright.agent.resume import resume_session
from codewright.agent.session import Session
from codewright.context.manager import ContextManager
from codewright.llm import create_llm_provider
from codewright.persistence.rollout import SessionMeta
from codewright.persistence.session_store import SessionStore
from codewright.prompts import PromptBuilder, load_default_system_prompt
from codewright.protocol import (
    EvAgentMessage,
    EvError,
    EvShutdownComplete,
    EvTurnAborted,
    EvTurnCompleted,
    OpUserTurn,
    PermissionProfile,
    UserInputText,
)
from codewright.tools.handlers import (
    ApplyPatchHandler,
    CloseAgentHandler,
    FollowupTaskHandler,
    ListAgentsHandler,
    RunShellHandler,
    SendMessageHandler,
    SpawnAgentHandler,
    UpdatePlanHandler,
    WaitAgentHandler,
)
from codewright.workspace.manager import WorkspaceManager


def _resolve_workspace(path_str: str) -> Path:

    return Path(path_str).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(prog="codewright")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="non-interactive single user turn")
    p_run.add_argument("prompt")
    p_run.add_argument("--workspace", default=".")
    p_run.add_argument("--model", default=None)
    p_run.add_argument("--provider-base-url", default=None)
    p_run.add_argument("--api-style", choices=("chat_completions", "responses"), default="chat_completions")
    p_run.add_argument("--max-context-tokens", type=int, default=None)
    p_run.add_argument(
        "--permission-profile",
        choices=tuple(p.value for p in PermissionProfile),
        default=PermissionProfile.WORKSPACE_WRITE.value,
    )
    p_run.add_argument("--print-session-id", action="store_true")
    p_run.add_argument("--no-persist", action="store_true",
                       help="do not write a rollout file under .codewright/sessions/")

    p_resume = sub.add_parser("resume", help="resume a previous session")
    p_resume.add_argument("session_id")
    p_resume.add_argument("--workspace", default=".")
    p_resume.add_argument("--model", default=None)
    p_resume.add_argument("--provider-base-url", default=None)
    p_resume.add_argument("--api-style", choices=("chat_completions", "responses"), default="chat_completions")
    p_resume.add_argument("--message", default=None,
                          help="if given, run one turn with this prompt and exit (else launch the TUI)")

    p_list = sub.add_parser("list-sessions", help="list saved sessions in this workspace")
    p_list.add_argument("--workspace", default=".")

    p_tui = sub.add_parser("tui", help="launch the interactive TUI")
    p_tui.add_argument("--workspace", default=".")
    p_tui.add_argument("--model", default=None)
    p_tui.add_argument("--provider-base-url", default=None)
    p_tui.add_argument("--api-style", choices=("chat_completions", "responses"), default="chat_completions")
    p_tui.add_argument(
        "--permission-profile",
        choices=tuple(p.value for p in PermissionProfile),
        default=PermissionProfile.WORKSPACE_WRITE.value,
    )

    args = parser.parse_args()
    asyncio.run(_dispatch(args))


async def _dispatch(args: argparse.Namespace) -> None:
    if args.cmd == "list-sessions":
        await _cmd_list(args)
        return
    if args.cmd == "run":
        await _cmd_run(args)
        return
    if args.cmd == "resume":
        await _cmd_resume(args)
        return
    if args.cmd == "tui":
        await _cmd_tui(args)
        return
    raise SystemExit(f"unknown subcommand: {args.cmd}")


async def _cmd_list(args: argparse.Namespace) -> None:
    store = SessionStore(_resolve_workspace(args.workspace))
    metas = await store.list_sessions()
    if not metas:
        print("(no sessions)")
        return
    for meta in metas:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(meta.start_time))
        print(f"{meta.session_id}\t{ts}\tmodel={meta.model}\tcwd={meta.cwd}")


async def _cmd_run(args: argparse.Namespace) -> None:
    sess = await _bootstrap_session(
        workspace=_resolve_workspace(args.workspace),
        model=args.model,
        provider_base_url=args.provider_base_url,
        api_style=args.api_style,
        permission_profile=PermissionProfile(args.permission_profile),
        max_context_tokens=args.max_context_tokens,
        persist=not args.no_persist,
    )
    try:
        if args.print_session_id:
            print(sess.session_id, file=sys.stderr)
        await sess.submit(OpUserTurn(items=[UserInputText(text=args.prompt)]))
        final_text = await _consume_one_turn(sess)
        if final_text:
            print(final_text)
    finally:
        await sess.shutdown()


async def _cmd_resume(args: argparse.Namespace) -> None:
    workspace = _resolve_workspace(args.workspace)
    llm = _build_llm(args.api_style, args.model or _default_model(), args.provider_base_url)
    prompt_builder = PromptBuilder(load_default_system_prompt())
    session = await resume_session(
        workspace_root=workspace,
        session_id=args.session_id,
        llm=llm,
        prompt_builder=prompt_builder,
        workspace=WorkspaceManager(workspace, PermissionProfile.WORKSPACE_WRITE),
    )
    _register_builtin_tools(session)
    try:
        if args.message:
            await session.submit(OpUserTurn(items=[UserInputText(text=args.message)]))
            final_text = await _consume_one_turn(session)
            if final_text:
                print(final_text)
        else:
            from codewright.tui import TuiApp

            await TuiApp(session).run()
    finally:
        await session.shutdown()


async def _cmd_tui(args: argparse.Namespace) -> None:
    sess = await _bootstrap_session(
        workspace=_resolve_workspace(args.workspace),
        model=args.model,
        provider_base_url=args.provider_base_url,
        api_style=args.api_style,
        permission_profile=PermissionProfile(args.permission_profile),
        max_context_tokens=None,
        persist=True,
    )
    try:
        from codewright.tui import TuiApp

        await TuiApp(sess).run()
    finally:
        await sess.shutdown()




def _default_model() -> str:
    return os.environ.get("CODEWRIGHT_MODEL") or "gpt-4o-mini"


def _api_key() -> str:

    return (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("CODEWRIGHT_API_KEY")
        or ""
    )


def _build_llm(api_style: str, model: str, base_url: str | None) -> Any:
    return create_llm_provider(
        api_style=api_style,  # type: ignore[arg-type]
        api_key=_api_key(),
        model=model,
        base_url=base_url,
    )


async def _bootstrap_session(
    *,
    workspace: Path,
    model: str | None,
    provider_base_url: str | None,
    api_style: str,
    permission_profile: PermissionProfile,
    max_context_tokens: int | None,
    persist: bool,
) -> Session:
    model = model or _default_model()
    llm = _build_llm(api_style, model, provider_base_url)
    cm = ContextManager(max_context_tokens=max_context_tokens or 128_000)
    prompt_builder = PromptBuilder(load_default_system_prompt())
    wm = WorkspaceManager(workspace, permission_profile)
    rollout = None
    session_id = uuid.uuid4().hex
    if persist:
        store = SessionStore(workspace)
        meta = SessionMeta(
            session_id=session_id,
            cwd=str(workspace),
            model=model,
            permission_profile=permission_profile.value,
            start_time=time.time(),
        )
        rollout = await store.create_recorder(meta)
    session = Session(
        session_id=session_id,
        cwd=workspace,
        permission_profile=permission_profile,
        model=model,
        llm=llm,
        context_manager=cm,
        prompt_builder=prompt_builder,
        workspace=wm,
        rollout=rollout,
    )
    _register_builtin_tools(session)
    return session


def _register_builtin_tools(session: Session) -> None:
    reg = session.tool_registry

    if not reg.has("run_shell"):
        reg.register(RunShellHandler())
    if not reg.has("apply_patch"):
        reg.register(ApplyPatchHandler())
    if not reg.has("update_plan"):
        reg.register(UpdatePlanHandler())
    for handler in (
        SpawnAgentHandler(),
        SendMessageHandler(),
        FollowupTaskHandler(),
        WaitAgentHandler(),
        CloseAgentHandler(),
        ListAgentsHandler(),
    ):
        if not reg.has(handler.tool_name):
            reg.register(handler)


async def _consume_one_turn(session: Session) -> str | None:

    cancel = CancellationToken()
    last: str | None = None
    while not cancel.is_cancelled():
        ev = await session.next_event()
        msg = ev.msg
        if isinstance(msg, EvAgentMessage):
            last = msg.content
        elif isinstance(msg, EvTurnCompleted):
            return msg.last_agent_message or last
        elif isinstance(msg, EvTurnAborted):
            return last
        elif isinstance(msg, EvError):
            print(f"[error] {msg.message}", file=sys.stderr)
        elif isinstance(msg, EvShutdownComplete):
            return last
    return last
