"""
Structural assertions tied to _SHARED.md §2 (15 architectural conditions).
Each phase appends its own section. Do not remove prior sections without
recording a corresponding ADR in DECISIONS.md.

Run: uv run pytest tests/test_architecture.py -v
"""
from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "codewright"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _iter_py(package: str) -> list[Path]:
    return sorted((SRC / package).rglob("*.py"))


# ============================================================
# === P1: protocol + agent.cancellation + agent.rwlock + agent.session skeleton + agent.submission_loop
# ============================================================


class TestP1Protocol:
    def test_protocol_files_exist(self):
        for name in ("op.py", "event.py", "messages.py", "agent_messages.py", "approval.py"):
            assert (SRC / "protocol" / name).exists(), f"missing protocol/{name}"

    def test_op_is_tagged_union(self):
        """A1: Op must be a tagged union (Literal field + discriminator), not enum+dict."""
        op = importlib.import_module("codewright.protocol.op")
        assert hasattr(op, "Op"), "protocol.op must export Op"
        src = _read(SRC / "protocol" / "op.py")
        # Forbid: a generic dict payload typed as Any/dict — common shortcut
        assert "payload: dict" not in src.lower(), "Op variants must not carry generic dict payloads"

    def test_event_has_correlation_id(self):
        """A1: Event{id, msg} carries Submission id for correlation."""
        src = _read(SRC / "protocol" / "event.py")
        assert "id" in src and "msg" in src, "Event must have id + msg fields"

    def test_approval_review_decision(self):
        """I2: ReviewDecision has 4 variants."""
        approval = importlib.import_module("codewright.protocol.approval")
        decisions = {m for m in dir(approval.ReviewDecision) if not m.startswith("_")}
        for name in ("APPROVED", "APPROVED_FOR_SESSION", "DENIED", "ABORT"):
            assert name in decisions, f"ReviewDecision missing {name}"


class TestP1Concurrency:
    def test_async_rwlock_module(self):
        rwlock = importlib.import_module("codewright.agent.rwlock")
        assert hasattr(rwlock, "AsyncRwLock"), "rwlock must export AsyncRwLock"
        assert not _imports("codewright.agent.rwlock", "aiorwlock"), (
            "must self-implement, not use aiorwlock"
        )

    def test_cancellation_token_child(self):
        cancel = importlib.import_module("codewright.agent.cancellation")
        assert hasattr(cancel, "CancellationToken"), "must export CancellationToken"
        assert hasattr(cancel.CancellationToken, "child"), "CancellationToken needs .child()"

    def test_no_time_sleep(self):
        """blacklist: time.sleep in event-loop code."""
        for path in (SRC / "agent").rglob("*.py"):
            src = _read(path)
            assert "time.sleep(" not in src, f"time.sleep forbidden in {path}"


class TestP1Session:
    def test_session_class_exists(self):
        session = importlib.import_module("codewright.agent.session")
        assert hasattr(session, "Session")

    def test_submission_loop_present(self):
        loop = importlib.import_module("codewright.agent.submission_loop")
        fn = getattr(loop, "submission_loop", None)
        assert fn is not None, "submission_loop function required"
        assert inspect.iscoroutinefunction(fn), "submission_loop must be async"


# ============================================================
# === P2: llm + agent.turn_context + agent.turn + context + prompts
# ============================================================


class TestP2LLM:
    def test_canonical_message_frozen(self):
        base = importlib.import_module("codewright.llm.base")
        cm = base.CanonicalMessage
        assert dataclasses.is_dataclass(cm), "CanonicalMessage must be a dataclass"
        assert cm.__dataclass_params__.frozen, "CanonicalMessage must be frozen"

    def test_provider_abc(self):
        base = importlib.import_module("codewright.llm.base")
        provider = base.LLMProvider
        assert inspect.isabstract(provider), "LLMProvider must be ABC"
        assert "stream" in {m for m in dir(provider)}, "LLMProvider needs stream()"

    def test_no_provider_dict_outside_llm(self):
        """F4: provider-specific message dict only allowed in llm/."""
        bad_patterns = [
            re.compile(r'"role"\s*:\s*"(system|user|assistant|tool)"'),
            re.compile(r"'role'\s*:\s*'(system|user|assistant|tool)'"),
        ]
        offenders = []
        for path in SRC.rglob("*.py"):
            if "llm" in path.parts:
                continue
            src = _read(path)
            for pat in bad_patterns:
                if pat.search(src):
                    offenders.append((path, pat.pattern))
        assert not offenders, f"provider-shape dict leaked outside llm/: {offenders}"


class TestP2Turn:
    def test_turn_context_frozen(self):
        tc_mod = importlib.import_module("codewright.agent.turn_context")
        tc = tc_mod.TurnContext
        assert dataclasses.is_dataclass(tc), "TurnContext must be a dataclass"
        assert tc.__dataclass_params__.frozen, "TurnContext must be frozen=True"

    def test_run_turn_is_not_recursive(self):
        src = _read(SRC / "agent" / "turn.py")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_turn":
                body_src = ast.unparse(node)
                assert "run_turn(" not in body_src.replace("def run_turn", ""), (
                    "run_turn must not call itself"
                )


class TestP2Context:
    def test_approx_token_count_chars(self):
        cm = importlib.import_module("codewright.context.manager")
        approx = cm.approx_token_count
        assert approx("a" * 100) == 25
        assert approx("") == 0

    def test_no_tokenizer_dep(self):
        py = (ROOT / "pyproject.toml").read_text()
        for forbidden in ("tiktoken", "transformers", "sentencepiece"):
            assert forbidden not in py, f"forbidden tokenizer dep: {forbidden}"


# ============================================================
# === P3: tools + workspace + handlers (shell, apply_patch, update_plan)
# ============================================================


class TestP3ToolAbstraction:
    def test_tool_handler_abc(self):
        h = importlib.import_module("codewright.tools.handler")
        assert inspect.isabstract(h.ToolHandler)
        method = h.ToolHandler.__abstractmethods__
        assert "handle" in method

    def test_tool_spec_uses_pydantic(self):
        spec = importlib.import_module("codewright.tools.spec")
        from pydantic import BaseModel
        assert issubclass(spec.ToolSpec, BaseModel) or hasattr(spec.ToolSpec, "model_json_schema"), (
            "ToolSpec must be pydantic-based"
        )

    def test_no_handwritten_schema_in_tools(self):
        for path in (SRC / "tools").rglob("*.py"):
            src = _read(path)
            assert '"type": "object", "properties"' not in src and (
                "'type': 'object', 'properties'" not in src
            ), f"hand-written JSON Schema in {path}"

    def test_no_big_dispatch_in_tools(self):
        for path in (SRC / "tools").rglob("*.py"):
            src = _read(path)
            assert not re.search(
                r"if\s+(tool_name|name|call\.name|payload\.tool_name)\s*==\s*['\"]shell['\"]",
                src,
            ), f"big-dispatch pattern in {path}"


class TestP3Errors:
    def test_respond_to_model_error_defined(self):
        errs = importlib.import_module("codewright.tools.errors")
        assert hasattr(errs, "RespondToModelError")
        assert hasattr(errs, "FatalToolError")

    def test_handlers_raise_respond_to_model_for_recoverable(self):
        for path in (SRC / "tools" / "handlers").rglob("*.py"):
            if path.name in ("__init__.py", "mcp_handler.py"):
                continue
            src = _read(path)
            for forbidden in ("raise Exception(", "raise RuntimeError(", "raise ValueError("):
                assert forbidden not in src, (
                    f"{path} raises generic exception (use RespondToModelError)"
                )


class TestP3ShellFamily:
    """D-3-008: the shell tool family (shell / shell_output / shell_kill)."""

    def test_shell_trio_handler_files_exist(self):
        for name in ("shell", "shell_output", "shell_kill"):
            assert (SRC / "tools" / "handlers" / f"{name}.py").exists(), (
                f"missing handlers/{name}.py"
            )

    def test_shell_spec_declares_dialect(self):
        # The model must never guess the dialect: the discovered shell label
        # (bash version + path, or the cmd fallback) is embedded in the spec.
        from codewright.tools.handlers._shell import ShellManager
        from codewright.tools.handlers.shell import ShellHandler

        manager = ShellManager()
        description = ShellHandler(manager).spec().description
        assert manager.dialect.label in description

    def test_cli_registers_shell_family_not_run_shell(self):
        src = _read(SRC / "cli.py")
        assert "ShellHandler" in src and "ShellOutputHandler" in src, (
            "cli must register the shell family"
        )
        assert "RunShellHandler()" not in src, (
            "run_shell is retired (D-3-008); do not re-register it"
        )

    def test_shell_output_and_kill_are_parallel_safe(self):
        # Polling a background job must not be blocked by the executor write
        # lock of a foreground command.
        from codewright.tools.handlers._shell import ShellManager
        from codewright.tools.handlers.shell_kill import ShellKillHandler
        from codewright.tools.handlers.shell_output import ShellOutputHandler

        manager = ShellManager()
        assert ShellOutputHandler(manager).spec().supports_parallel is True
        assert ShellKillHandler(manager).spec().supports_parallel is True


class TestP3ApplyPatch:
    def test_apply_patch_handler_independent(self):
        ap = importlib.import_module("codewright.tools.handlers.apply_patch")
        assert hasattr(ap, "ApplyPatchHandler") or any(
            inspect.isclass(o) and "ApplyPatch" in o.__name__ for o in vars(ap).values()
        )

    def test_workspace_canonicalize(self):
        wm = importlib.import_module("codewright.workspace.manager")
        assert hasattr(wm.WorkspaceManager, "canonicalize")
        assert hasattr(wm.WorkspaceManager, "resolve_agents_md")


# ============================================================
# === P4: compact + persistence + mcp + responses adapter
# ============================================================


class TestP4Compact:
    def test_compact_is_separate_call(self):
        compact = importlib.import_module("codewright.context.compact")
        assert hasattr(compact, "run_compact") or hasattr(compact, "compact_history"), (
            "context.compact must expose a top-level compaction entry"
        )


class TestP4Persistence:
    def test_rollout_append_only(self):
        src = _read(SRC / "persistence" / "rollout.py")
        assert re.search(r"open\([^)]*['\"]a['\"]", src) or "'a'" in src, (
            "rollout writer must open file in append mode"
        )


class TestP4MCP:
    def test_mcp_naming(self):
        importlib.import_module("codewright.mcp.handler")
        src = _read(SRC / "mcp" / "handler.py")
        assert "__" in src, "MCP handler must namespace tools with double underscore"

    def test_responses_adapter_exists(self):
        ra = importlib.import_module("codewright.llm.responses_api")
        assert hasattr(ra, "ResponsesApiAdapter")


# ============================================================
# === P5: multi-agent + tui + final docs
# ============================================================


class TestP5Mailbox:
    def test_mailbox_uses_queue_and_event(self):
        src = _read(SRC / "agent" / "mailbox.py")
        assert "asyncio.Queue" in src, "Mailbox needs asyncio.Queue"
        assert "Event" in src, "Mailbox needs asyncio.Event for seq wakeup"

    def test_agent_path_format(self):
        am = importlib.import_module("codewright.protocol.agent_messages")
        assert hasattr(am, "AgentPath")


class TestP5Subagent:
    def test_six_subagent_tools(self):
        required = {
            "spawn_agent",
            "send_message",
            "followup_task",
            "wait_agent",
            "close_agent",
            "list_agents",
        }
        existing = {
            p.stem for p in (SRC / "tools" / "handlers").glob("*.py") if p.stem not in ("__init__",)
        }
        missing = required - existing
        assert not missing, f"missing subagent handlers: {missing}"

    def test_spawn_agent_description_lists_roles(self):
        # Behavioral check: the tool description must enumerate the available
        # roles so the model can pick one. The description is rendered from the
        # RoleRegistry (the single source of truth in agent/roles.py), so assert
        # against the generated spec rather than grepping the source text.
        from codewright.agent.roles import load_builtin_roles
        from codewright.tools.handlers.spawn_agent import SpawnAgentHandler

        description = SpawnAgentHandler().spec().description.lower()
        for role in load_builtin_roles().all_roles():
            assert role.name.lower() in description, (
                f"spawn_agent description must mention role: {role.name}"
            )


class TestP5Docs:
    def test_readme_and_architecture_exist(self):
        assert (ROOT / "README.md").exists()
        assert (ROOT / "ARCHITECTURE.md").exists()

    def test_architecture_has_mermaid(self):
        src = (ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
        # _SHARED.md raises the floor to 3 Mermaid diagrams for P5 final docs.
        assert src.count("```mermaid") >= 3, (
            "ARCHITECTURE.md must contain at least 3 Mermaid diagrams"
        )


# ============================================================
# === Cross-phase invariants (always-on)
# ============================================================


class TestBlacklist:
    def test_no_forbidden_deps(self):
        py = (ROOT / "pyproject.toml").read_text()
        for forbidden in (
            "langchain",
            "autogen",
            "instructor",
            "crewai",
            "pyyaml",
            "aiorwlock",
            "tiktoken",
        ):
            assert forbidden not in py, f"forbidden dependency: {forbidden}"

    def test_no_yaml_config(self):
        # node_modules/dist 是前端第三方包与构建产物,其内部自带的 CI 配置
        # 不属于"本项目的配置文件",不在黑名单管辖范围内。
        ignored_dirs = {".venv", ".git", "node_modules", "dist"}
        for ext in ("*.yaml", "*.yml"):
            offenders = list(ROOT.rglob(ext))
            offenders = [p for p in offenders if not ignored_dirs.intersection(p.parts)]
            assert not offenders, f"YAML files forbidden: {offenders}"


# ============================================================
# Helpers
# ============================================================


def _imports(module: str, target: str) -> bool:
    """Check if a module imports another by name (rough AST/source check)."""
    try:
        mod = importlib.import_module(module)
    except Exception:
        return False
    src = inspect.getsource(mod) if inspect.ismodule(mod) else ""
    return target in src
