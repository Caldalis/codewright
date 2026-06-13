from __future__ import annotations

import argparse
import asyncio
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from codewright.agent.cancellation import CancellationToken
from codewright.agent.resume import resume_session
from codewright.agent.session import Session
from codewright.config import CliOverrides, CodewrightConfig, load_config
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
    EvWarning,
    OpUserTurn,
    PermissionProfile,
    UserInputText,
)
from codewright.tools.handlers import (
    ApplyPatchHandler,
    CloseAgentHandler,
    FindFilesHandler,
    FollowupTaskHandler,
    ListAgentsHandler,
    ListDirHandler,
    ReadFileHandler,
    SearchTextHandler,
    SendMessageHandler,
    ShellHandler,
    ShellKillHandler,
    ShellManager,
    ShellOutputHandler,
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
    p_run.add_argument(
        "--api-style",
        choices=("chat_completions", "responses"),
        default=None,
    )
    p_run.add_argument("--max-context-tokens", type=int, default=None)
    p_run.add_argument(
        "--permission-profile",
        choices=tuple(p.value for p in PermissionProfile),
        default=None,
    )
    p_run.add_argument("--print-session-id", action="store_true")
    p_run.add_argument("--no-persist", action="store_true",
                       help="do not write a rollout file under .codewright/sessions/")

    p_resume = sub.add_parser("resume", help="resume a previous session")
    p_resume.add_argument("session_id")
    p_resume.add_argument("--workspace", default=".")
    p_resume.add_argument("--model", default=None)
    p_resume.add_argument("--provider-base-url", default=None)
    p_resume.add_argument(
        "--api-style",
        choices=("chat_completions", "responses"),
        default=None,
    )
    p_resume.add_argument("--message", default=None,
                          help="if given, run one turn with this prompt and exit (else launch the TUI)")

    p_list = sub.add_parser("list-sessions", help="list saved sessions in this workspace")
    p_list.add_argument("--workspace", default=".")

    p_tui = sub.add_parser("tui", help="launch the interactive TUI")
    p_tui.add_argument("--workspace", default=".")
    p_tui.add_argument("--model", default=None)
    p_tui.add_argument("--provider-base-url", default=None)
    p_tui.add_argument(
        "--api-style",
        choices=("chat_completions", "responses"),
        default=None,
    )
    p_tui.add_argument(
        "--permission-profile",
        choices=tuple(p.value for p in PermissionProfile),
        default=None,
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
    config = _effective_config(args)
    sess = await _bootstrap_session(
        workspace=_resolve_workspace(args.workspace),
        config=config,
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
    config = _effective_config(args)
    workspace = _resolve_workspace(args.workspace)
    session = await _resume_session_from_config(
        workspace=workspace,
        session_id=args.session_id,
        config=config,
    )
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
    config = _effective_config(args)
    sess = await _bootstrap_session(
        workspace=_resolve_workspace(args.workspace),
        config=config,
        persist=True,
    )
    try:
        from codewright.tui import TuiApp

        await TuiApp(sess).run()
    finally:
        await sess.shutdown()


def _effective_config(args: argparse.Namespace) -> CodewrightConfig:
    return load_config(
        CliOverrides(
            api_style=getattr(args, "api_style", None),
            model=getattr(args, "model", None),
            provider_base_url=getattr(args, "provider_base_url", None),
            max_context_tokens=getattr(args, "max_context_tokens", None),
            permission_profile=getattr(args, "permission_profile", None),
        )
    )


def _build_llm(config: CodewrightConfig) -> Any:
    return create_llm_provider(
        api_style=config.api_style,
        api_key=config.api_key,
        model=config.model,
        base_url=config.provider_base_url,
    )


def _context_manager_from_config(config: CodewrightConfig) -> ContextManager:
    return ContextManager(
        max_context_tokens=config.max_context_tokens,
        compact_threshold=config.compact_threshold,
    )


async def _resume_session_from_config(
    *,
    workspace: Path,
    session_id: str,
    config: CodewrightConfig,
) -> Session:
    session = await resume_session(
        workspace_root=workspace,
        session_id=session_id,
        llm=_build_llm(config),
        context_manager=_context_manager_from_config(config),
        prompt_builder=PromptBuilder(load_default_system_prompt()),
        model=config.model,
        permission_profile=config.permission_profile,
        role=config.default_role,
        mcp_configs=config.mcp_servers,
    )
    await _register_builtin_tools(session, shell_path=config.shell_path)
    return session


async def _bootstrap_session(
    *,
    workspace: Path,
    config: CodewrightConfig,
    persist: bool,
) -> Session:
    llm = _build_llm(config)
    cm = _context_manager_from_config(config)
    prompt_builder = PromptBuilder(load_default_system_prompt())
    wm = WorkspaceManager(workspace, config.permission_profile)
    rollout = None
    session_id = uuid.uuid4().hex
    if persist:
        store = SessionStore(workspace)
        meta = SessionMeta(
            session_id=session_id,
            cwd=str(workspace),
            model=config.model,
            permission_profile=config.permission_profile.value,
            start_time=time.time(),
        )
        rollout = await store.create_recorder(meta)
    session = Session(
        session_id=session_id,
        cwd=workspace,
        permission_profile=config.permission_profile,
        model=config.model,
        llm=llm,
        context_manager=cm,
        prompt_builder=prompt_builder,
        workspace=wm,
        rollout=rollout,
        role=config.default_role,
    )
    await _register_builtin_tools(session, shell_path=config.shell_path)
    await session.start_mcp(config.mcp_servers)
    return session


async def _register_builtin_tools(
    session: Session, shell_path: str | None = None
) -> None:
    reg = session.tool_registry

    for handler in (
        ReadFileHandler(),
        ListDirHandler(),
        FindFilesHandler(),
        SearchTextHandler(),
    ):
        if not reg.has(handler.tool_name):
            reg.register(handler)
    if not reg.has("shell"):
        shell_manager = ShellManager(shell_path=shell_path)
        reg.register(ShellHandler(shell_manager))
        reg.register(ShellOutputHandler(shell_manager))
        reg.register(ShellKillHandler(shell_manager))
        if not shell_manager.dialect.persistent:
            await session.emit_event(
                EvWarning(
                    message=(
                        "shell tool degraded to cmd.exe (bash not found): session "
                        "state (cd/export) will NOT persist and command safety "
                        "checks are limited. Install Git Bash or set "
                        "CODEWRIGHT_SHELL_PATH to a bash executable."
                    )
                )
            )
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
