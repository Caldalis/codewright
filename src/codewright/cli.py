from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from codewright.agent.resume import resume_session
from codewright.agent.session import Session
from codewright.config import CliOverrides, CodewrightConfig, load_config
from codewright.context.manager import ContextManager
from codewright.llm import create_llm_provider
from codewright.persistence.rollout import SessionMeta
from codewright.persistence.session_store import SessionStore
from codewright.prompts import PromptBuilder, load_default_system_prompt
from codewright.protocol import (
    AskForApproval,
    EvAgentMessage,
    EvCompactionCompleted,
    EvError,
    EvExecApprovalRequest,
    EvPatchApprovalRequest,
    EvShutdownComplete,
    EvTokenCount,
    EvToolCallStarted,
    EvTurnAborted,
    EvTurnCompleted,
    EvWarning,
    OpExecApprovalResponse,
    OpPatchApprovalResponse,
    OpUserTurn,
    PermissionProfile,
    ReviewDecision,
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
    SkillHandler,
    SpawnAgentHandler,
    UpdatePlanHandler,
    WaitAgentHandler,
)
from codewright.workspace.manager import WorkspaceManager


def _resolve_workspace(path_str: str) -> Path:

    return Path(path_str).resolve()


def _positive_int(raw: str) -> int:
    """`--max-steps 0` would abort before the first model call, burning a whole
    sweep on zero work. Reject it at parse time rather than at instance 500."""
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def _add_unattended_flags(parser: argparse.ArgumentParser) -> None:
    """Flags every non-interactive front end needs to run without a human."""

    parser.add_argument(
        "--full-auto",
        action="store_true",
        help=(
            "never prompt. Actions the profile would ask about are resolved "
            "without a human: recoverable work inside the workspace is allowed, "
            "while leaving the workspace root and privileged commands (sudo, "
            "curl|bash, git push --force) are denied. Pair with "
            "--permission-profile dangerous to allow everything instead, and "
            "only inside a throwaway container"
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=_positive_int,
        default=None,
        metavar="N",
        help=(
            "abort the turn after N model round-trips (default: unlimited). "
            "The budget is per agent per turn, not per run: a spawned "
            "sub-agent gets its own N, and each follow-up task starts a new "
            "turn with a fresh N. Use a wall-clock timeout for a global cap"
        ),
    )
    parser.add_argument(
        "--no-distill",
        action="store_true",
        help=(
            "disable self-improving skill distillation. It fires an extra LLM "
            "call on every red->green test transition, so benchmark runs want "
            "it off for reproducible cost and behavior"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="codewright")
    sub = parser.add_subparsers(dest="cmd")

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
    _add_unattended_flags(p_run)
    p_run.add_argument(
        "--output-json",
        default=None,
        metavar="PATH",
        help="write a machine-readable run summary to PATH ('-' for stdout)",
    )
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
    _add_unattended_flags(p_resume)
    p_resume.add_argument(
        "--output-json",
        default=None,
        metavar="PATH",
        help="with --message: write a run summary to PATH ('-' for stderr)",
    )

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

    p_serve = sub.add_parser("serve", help="run the WebSocket bridge for web front ends")
    p_serve.add_argument("--workspace", default=".")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8765)
    p_serve.add_argument("--model", default=None)
    p_serve.add_argument("--provider-base-url", default=None)
    p_serve.add_argument(
        "--api-style",
        choices=("chat_completions", "responses"),
        default=None,
    )
    p_serve.add_argument(
        "--permission-profile",
        choices=tuple(p.value for p in PermissionProfile),
        default=None,
    )

    argv = sys.argv[1:]
    subcommands = {"run", "resume", "list-sessions", "tui", "serve"}
    if not argv or (argv[0] not in subcommands and argv[0] not in ("-h", "--help")):
        argv = ["tui", *argv]
    args = parser.parse_args(argv)
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
    if args.cmd == "serve":
        await _cmd_serve(args)
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
    workspace = _resolve_workspace(args.workspace)
    started = time.monotonic()
    outcome = TurnOutcome()
    sess = await _bootstrap_session(
        workspace=workspace,
        config=config,
        persist=not args.no_persist,
        full_auto=args.full_auto,
        max_steps=args.max_steps,
        distill=not args.no_distill,
    )
    try:
        if args.print_session_id:
            print(sess.session_id, file=sys.stderr)
        await sess.submit(OpUserTurn(items=[UserInputText(text=args.prompt)]))
        outcome = await _consume_one_turn(sess, auto_approve=args.full_auto)
        outcome.auto_approved = sess.auto_approved
        if outcome.final_text:
            print(outcome.final_text)
    finally:
        await sess.shutdown()

    if args.output_json:
        _write_summary(
            args.output_json,
            {
                "session_id": sess.session_id,
                "model": config.model,
                "workspace": str(workspace),
                "full_auto": args.full_auto,
                "max_steps": args.max_steps,
                "wall_ms": int((time.monotonic() - started) * 1000),
                **asdict(outcome),
            },
        )

    if outcome.status != "completed":
        raise SystemExit(1)


async def _cmd_resume(args: argparse.Namespace) -> None:
    config = _effective_config(args)
    workspace = _resolve_workspace(args.workspace)
    session = await _resume_session_from_config(
        workspace=workspace,
        session_id=args.session_id,
        config=config,
        full_auto=args.full_auto,
        max_steps=args.max_steps,
        distill=not args.no_distill,
    )
    started = time.monotonic()
    outcome: TurnOutcome | None = None
    try:
        if args.message:
            await session.submit(OpUserTurn(items=[UserInputText(text=args.message)]))
            outcome = await _consume_one_turn(session, auto_approve=args.full_auto)
            outcome.auto_approved = session.auto_approved
            if outcome.final_text:
                print(outcome.final_text)
        else:
            from codewright.tui import TuiApp

            await TuiApp(session).run()
    finally:
        await session.shutdown()

    # `resume --message` is the same unattended entry point as `run`, so it
    # reports the same way. Without this the two look alike and behave
    # differently, which is worse than either choice on its own.
    if outcome is not None:
        if args.output_json:
            _write_summary(
                args.output_json,
                {
                    "session_id": session.session_id,
                    "model": config.model,
                    "workspace": str(workspace),
                    "full_auto": args.full_auto,
                    "max_steps": args.max_steps,
                    "wall_ms": int((time.monotonic() - started) * 1000),
                    **asdict(outcome),
                },
            )
        if outcome.status != "completed":
            raise SystemExit(1)


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


async def _cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from codewright.web import create_app

    config = _effective_config(args)
    workspace = _resolve_workspace(args.workspace)

    async def session_factory(session_id: str | None) -> Session:
        if session_id:
            return await _resume_session_from_config(
                workspace=workspace, session_id=session_id, config=config
            )
        return await _bootstrap_session(
            workspace=workspace, config=config, persist=True
        )

    app = create_app(session_factory, workspace_root=workspace)
    server = uvicorn.Server(
        uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    )
    print(f"codewright bridge: http://{args.host}:{args.port}  (test page at /)")
    await server.serve()


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
    full_auto: bool = False,
    max_steps: int | None = None,
    distill: bool = True,
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
        approval_policy=_approval_policy(full_auto),
        max_steps=max_steps,
        distill=distill,
    )
    await _register_builtin_tools(session, shell_path=config.shell_path)
    return session


async def _bootstrap_session(
    *,
    workspace: Path,
    config: CodewrightConfig,
    persist: bool,
    full_auto: bool = False,
    max_steps: int | None = None,
    distill: bool = True,
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
        extra_test_commands=config.test_commands,
        approval_policy=_approval_policy(full_auto),
        max_steps=max_steps,
        distill=distill,
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
    if not reg.has("skill"):
        reg.register(SkillHandler(session.skill_registry))
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


def _approval_policy(full_auto: bool) -> AskForApproval:
    return AskForApproval.NEVER if full_auto else AskForApproval.ON_REQUEST


def _write_summary(path_str: str, summary: dict[str, Any]) -> None:
    blob = json.dumps(summary, ensure_ascii=False, indent=2)
    if path_str == "-":
        # stderr, not stdout: stdout already carries the agent's final message,
        # and a consumer piping this wants JSON, not prose followed by JSON.
        print(blob, file=sys.stderr)
        return
    out = Path(path_str)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(blob + "\n", encoding="utf-8")


@dataclass
class TurnOutcome:
    """Machine-readable result of one non-interactive turn."""

    status: str = "unknown"  # completed | interrupted | error | shutdown
    final_text: str | None = None
    model_calls: int = 0
    tool_calls: int = 0
    # Set from the session after the turn: approvals are resolved in
    # WorkspaceManager and never surface as events under full-auto.
    auto_approved: int = 0
    # Counts the CLI-side fallback below, which only fires for an approval the
    # workspace layer did not already resolve.
    denied_for_approval: int = 0
    compactions: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # Sum of per-response totals, so the context resent on every round-trip is
    # counted each time. This is the billed figure, not the context size.
    billed_tokens: int = 0
    errors: list[str] = field(default_factory=list)


async def _answer_approval(
    session: Session,
    msg: EvExecApprovalRequest | EvPatchApprovalRequest,
    *,
    auto_approve: bool,
    outcome: TurnOutcome,
) -> None:
    if auto_approve:
        decision = ReviewDecision.APPROVED
    else:
        decision = ReviewDecision.DENIED
        outcome.denied_for_approval += 1
        print(
            f"[denied] {msg.action.summary!r} needs approval; "
            "re-run with --full-auto to allow it unattended",
            file=sys.stderr,
        )
    op = (
        OpExecApprovalResponse(request_id=msg.request_id, decision=decision)
        if isinstance(msg, EvExecApprovalRequest)
        else OpPatchApprovalResponse(request_id=msg.request_id, decision=decision)
    )
    await session.submit(op)


async def _consume_one_turn(
    session: Session, *, auto_approve: bool = False
) -> TurnOutcome:
    """Drive one turn to completion, answering approval requests inline.

    Only a front end can resolve an approval: the broker parks on a future until
    an OpExec/PatchApprovalResponse arrives, and `submit_with_id` handles those
    out of band, so replying from here mid-turn is safe. Without this branch any
    flagged action -- `$(...)` in a shell command being the common one -- would
    hang a headless run forever.
    """

    outcome = TurnOutcome()
    while True:
        ev = await session.next_event()
        msg = ev.msg
        if isinstance(msg, EvAgentMessage):
            outcome.final_text = msg.content
        elif isinstance(msg, EvToolCallStarted):
            outcome.tool_calls += 1
        elif isinstance(msg, EvTokenCount):
            outcome.model_calls += 1
            outcome.input_tokens += msg.input
            outcome.output_tokens += msg.output
            outcome.billed_tokens += msg.total
        elif isinstance(msg, EvCompactionCompleted):
            outcome.compactions += 1
        elif isinstance(msg, EvExecApprovalRequest | EvPatchApprovalRequest):
            await _answer_approval(
                session, msg, auto_approve=auto_approve, outcome=outcome
            )
        elif isinstance(msg, EvTurnCompleted):
            outcome.status = "completed"
            outcome.final_text = msg.last_agent_message or outcome.final_text
            return outcome
        elif isinstance(msg, EvTurnAborted):
            outcome.status = (
                "interrupted" if msg.reason == "interrupted" else "error"
            )
            return outcome
        elif isinstance(msg, EvError):
            outcome.errors.append(msg.message)
            print(f"[error] {msg.message}", file=sys.stderr)
        elif isinstance(msg, EvShutdownComplete):
            outcome.status = "shutdown"
            return outcome
