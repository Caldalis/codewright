from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codewright.agent.skills import SkillRegistry
from codewright.llm.base import CanonicalMessage
from codewright.prompts.builder import load_role_text
from codewright.protocol import EvWarning

if TYPE_CHECKING:
    from codewright.agent.turn_context import TurnContext


_TEST_RUNNER_RE = re.compile(
    r"(?:^|[\s;&|(])("
    r"pytest|py\.test|python\s+-m\s+pytest|python\s+-m\s+unittest|"
    r"tox|nox|jest|vitest|mocha|"
    r"npm\s+(?:run\s+)?test|yarn\s+test|pnpm\s+(?:run\s+)?test|bun\s+test|"
    r"go\s+test|cargo\s+(?:test|nextest)|rspec|rails\s+test|"
    r"mvn\s+test|gradle\s+test|dotnet\s+test|phpunit|gotestsum"
    r")\b",
    re.IGNORECASE,
)

_TEST_PATH_RE = re.compile(
    r"(tests?/|test_[\w.-]+|_test\.|\.test\.|\.spec\.|/spec/)", re.IGNORECASE
)

_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_VALID_TYPES = ("fact", "skill", "lesson")

def is_test_command(command: str, extra: tuple[str, ...] = ()) -> bool:
    if not command:
        return False
    if _TEST_RUNNER_RE.search(command):
        return True
    cmd = command.strip()
    return any(prefix and cmd.startswith(prefix) for prefix in extra)

def _norm_cmd(command: str) -> str:
    return " ".join(command.split())

@dataclass(frozen=True)
class TestCounts:
    passed: int
    failed: int
    errors: int
    @property
    def total_run(self) -> int:
        return self.passed + self.failed + self.errors

def extract_test_counts(output: str) -> TestCounts | None:
    passed = _grab_int(re.search(r"(\d+)\s+passed", output))
    failed = _grab_int(re.search(r"(\d+)\s+failed", output))
    errors = _grab_int(re.search(r"(\d+)\s+errors?", output))
    if passed is None and failed is None and errors is None:
        return None
    return TestCounts(passed or 0, failed or 0, errors or 0)

def _grab_int(match: re.Match[str] | None) -> int | None:
    return int(match.group(1)) if match else None

def anti_cheat_ok(red_output: str, green_output: str, edited_test_files: bool) -> bool:
    red = extract_test_counts(red_output)
    green = extract_test_counts(green_output)
    if red is not None and green is not None:
        return green.total_run >= red.total_run
    return not edited_test_files

def parse_candidates(text: str) -> list[dict[str, str]]:
    obj = _loads_lenient(text)
    if not isinstance(obj, dict):
        return []
    raw = obj.get("candidates")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for c in raw:
        if not isinstance(c, dict) or not c.get("name") or not c.get("body"):
            continue
        out.append(
            {
                "type": str(c.get("type", "skill")).strip() or "skill",
                "name": str(c.get("name", "")).strip(),
                "description": str(c.get("description", "")).strip(),
                "body": str(c.get("body", "")).strip(),
            }
        )
    return out

def _loads_lenient(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            return None
    return None

def _valid_name(name: str) -> bool:
    return bool(name) and len(name) <= 64 and bool(_NAME_RE.match(name))

def gate_candidate(candidate: dict[str, str], existing_names: set[str]) -> bool:
    name = candidate.get("name", "")
    if not _valid_name(name) or name in existing_names:
        return False  # invalid name, or would collide with / overwrite an existing skill
    description = (candidate.get("description") or "").strip()
    body = (candidate.get("body") or "").strip()
    if not description or len(description) > 1024 or not body:
        return False
    return candidate.get("type", "skill") in _VALID_TYPES

def render_skill_md(candidate: dict[str, str], evidence: str) -> str:
    return "\n".join(
        [
            "---",
            f"name: {candidate['name']}",
            f"description: {_yaml_quote(candidate.get('description', ''))}",
            "metadata:",
            "  cw-source: learned",
            "  cw-status: provisional",
            f"  cw-type: {candidate.get('type', 'skill')}",
            f"  cw-evidence: {_yaml_quote(evidence)}",
            "---",
            "",
            candidate.get("body", "").strip(),
            "",
        ]
    )

def _yaml_quote(text: str) -> str:
    flat = " ".join(text.split()).replace('"', "'")
    return f'"{flat}"'

def write_provisional_skill(
    workspace_root: Path, candidate: dict[str, str], evidence: str
) -> Path | None:
    name = candidate.get("name", "")
    if not _valid_name(name):
        return None
    skills_root = Path(workspace_root) / "skills"
    skill_dir = skills_root / name
    try:
        skill_dir.resolve().relative_to(skills_root.resolve())
    except ValueError:
        return None
    try:
        skill_dir.mkdir(parents=True, exist_ok=True)
        target = skill_dir / "SKILL.md"
        tmp = target.with_name("SKILL.md.tmp")
        tmp.write_text(render_skill_md(candidate, evidence), encoding="utf-8")
        tmp.replace(target)
    except OSError:
        return None
    return target

PROMOTE_MIN_SUCCESSES = 2
QUARANTINE_MIN_ATTRIBUTIONS = 2
QUARANTINE_FAILURE_RATE = 0.5
STALE_AGE_SECONDS = 30 * 24 * 3600

def decide_status(
    status: str, source: str, success: int, failure: int, age_seconds: float | None
) -> str | None:

    if source != "learned" or status == "quarantined":
        return None
    attributions = success + failure
    if (
        attributions >= QUARANTINE_MIN_ATTRIBUTIONS
        and failure / attributions >= QUARANTINE_FAILURE_RATE
    ):
        return "quarantined"
    if status == "provisional" and success >= PROMOTE_MIN_SUCCESSES and failure == 0:
        return "trusted"
    if status == "provisional" and age_seconds is not None and age_seconds > STALE_AGE_SECONDS:
        return "quarantined"
    return None

def set_skill_status(skill_md_path: Path, new_status: str) -> bool:
    try:
        text = skill_md_path.read_text(encoding="utf-8")
    except OSError:
        return False
    new_text = _rewrite_status(text, new_status)
    if new_text == text:
        new_text = _ensure_status(text, new_status)
    if new_text == text:
        return False
    try:
        tmp = skill_md_path.with_name(skill_md_path.name + ".tmp")
        tmp.write_text(new_text, encoding="utf-8")
        tmp.replace(skill_md_path)
    except OSError:
        return False
    return True

def _rewrite_status(text: str, new_status: str) -> str:
    return re.sub(r"(?m)^(\s*cw-status:\s*).*$", rf"\g<1>{new_status}", text, count=1)

def _ensure_status(text: str, new_status: str) -> str:
    if re.search(r"(?m)^metadata:\s*$", text):
        return re.sub(
            r"(?m)^(metadata:\s*\n)", rf"\g<1>  cw-status: {new_status}\n", text, count=1
        )
    return text

@dataclass(frozen=True)
class _Transition:
    command: str
    kind: str  # "red_green" (verified success) | "green_red" (regression)
    prev_output: str
    cur_output: str

class VerificationTracker:
    """Remembers each test command's last outcome and flags red->green flips."""
    def __init__(self, extra_commands: tuple[str, ...] = ()) -> None:
        self._last: dict[str, tuple[bool, str]] = {}
        self._extra = tuple(extra_commands)

    def observe(self, command: str, exit_code: int | None, output: str) -> _Transition | None:
        if exit_code is None or not is_test_command(command, self._extra):
            return None
        key = _norm_cmd(command)
        passed = exit_code == 0
        prev = self._last.get(key)
        self._last[key] = (passed, output)
        if prev is None:
            return None
        if prev[0] is False and passed:
            return _Transition(command, "red_green", prev[1], output)
        if prev[0] is True and not passed:
            return _Transition(command, "green_red", prev[1], output)
        return None

@dataclass
class _Event:
    kind: str  # "task" | "edit" | "shell"
    text: str

class DistillationCoordinator:
    def __init__(self, workspace_root: Path, *, extra_test_commands: tuple[str, ...] = ()) -> None:
        self._root = Path(workspace_root)
        self._tracker = VerificationTracker(extra_test_commands)
        self._events: list[_Event] = []
        self._pending: list[_Transition] = []
        self._edited_test_files = False
        self._failure_pending = False
        self._retrieved: set[str] = set()
        self._tasks: set[asyncio.Task[None]] = set()
    def note_user(self, text: str) -> None:
        if text and text.strip():
            self._events.append(_Event("task", text.strip()))

    def note_retrieval(self, name: str) -> None:
        self._retrieved.add(name)

    def note_tool(self, tool_name: str, arguments: dict[str, Any], result: Any) -> None:
        if tool_name == "apply_patch" and getattr(result, "success", False):
            patch = _patch_text(arguments)
            if patch:
                self._events.append(_Event("edit", patch))
                if _touches_test_files(patch):
                    self._edited_test_files = True
        elif tool_name == "shell":
            command = str((arguments or {}).get("command", "")).strip()
            if not command:
                return
            data = getattr(result, "structured_data", None) or {}
            exit_code = data.get("exit_code")
            body = getattr(result, "body", "") or ""
            self._events.append(
                _Event("shell", f"$ {command}\nexit={exit_code}\n{_short(body, 1500)}")
            )
            transition = self._tracker.observe(command, exit_code, body)
            if transition is None:
                return
            if transition.kind == "red_green":
                self._pending.append(transition)
            elif transition.kind == "green_red":
                self._failure_pending = True

    async def on_batch_end(self, session: Any, turn_context: TurnContext) -> None:
        registry: SkillRegistry = session.skill_registry
        changed = False
        if self._pending:
            changed = await self._handle_green(session, turn_context, registry) or changed
        if self._failure_pending:
            self._failure_pending = False
            for name in self._retrieved:
                registry.metrics.record_failure(name)
            self._retrieved.clear()
            changed = True
        if changed:
            await self._apply_lifecycle(session, registry)

    async def _handle_green(
        self, session: Any, turn_context: TurnContext, registry: SkillRegistry
    ) -> bool:
        transition = self._pending[-1]
        edited = self._edited_test_files
        legit = anti_cheat_ok(transition.prev_output, transition.cur_output, edited)
        slice_text = self._render_slice(transition)
        evidence = _evidence_line(transition)
        self._reset_episode()
        if not legit:
            return False
        for name in self._retrieved:
            registry.metrics.record_success(name)
        self._retrieved.clear()
        task = asyncio.create_task(self._run(session, turn_context, slice_text, evidence))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    async def _run(self, session: Any, turn_context: TurnContext, slice_text: str, evidence: str) -> None:
        try:
            registry: SkillRegistry = session.skill_registry
            existing = {s.name for s in registry.all_skills(include_quarantined=True)}
            menu = _render_menu(registry)
            candidates = await _distill_candidates(session.llm, turn_context, slice_text, menu)
            for cand in candidates:
                if not gate_candidate(cand, existing):
                    continue
                if write_provisional_skill(session.cwd, cand, evidence) is not None:
                    existing.add(cand["name"])
                    await session.emit_event(
                        EvWarning(
                            message=(
                                f"[skill] learned '{cand['name']}' (provisional) — "
                                "review under ./skills"
                            )
                        )
                    )
        except Exception:
            return  # distillation is best-effort; it must never break a turn

    async def drain(self) -> None:
        """used by tests and shutdown"""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*list(self._tasks), return_exceptions=True)
        self._tasks.clear()

    def _reset_episode(self) -> None:
        self._events = []
        self._pending = []
        self._edited_test_files = False

    def _render_slice(self, transition: _Transition) -> str:
        tasks = [e.text for e in self._events if e.kind == "task"]
        edits = [e.text for e in self._events if e.kind == "edit"]
        parts: list[str] = []
        if tasks:
            parts.append("## Task\n" + "\n\n".join(_short(t, 1000) for t in tasks))
        if edits:
            joined = "\n\n".join(_short(e, 4000) for e in edits)
            parts.append("## Code changes during this episode\n" + _short(joined, 12000))
        parts.append(
            "## Verification (objective success signal)\n"
            f"Test command: {transition.command}\n\n"
            f"BEFORE (failing):\n{_short(transition.prev_output, 2000)}\n\n"
            f"AFTER (passing):\n{_short(transition.cur_output, 2000)}"
        )
        return "\n\n".join(parts)

    async def _apply_lifecycle(self, session: Any, registry: SkillRegistry) -> None:
        now = time.time()
        metrics = registry.metrics
        for skill in registry.all_skills(include_quarantined=False):
            if skill.source != "learned":
                continue
            entry = metrics.get(skill.name)
            success = int(entry.get("success_after", 0))
            failure = int(entry.get("failure_after", 0))
            age = _age_seconds(entry.get("last_retrieved_ts"), skill.body_path, now)
            new_status = decide_status(skill.status, skill.source, success, failure, age)
            if (
                new_status
                and new_status != skill.status
                and set_skill_status(skill.body_path, new_status)
            ):
                await session.emit_event(
                    EvWarning(message=f"[skill] '{skill.name}' -> {new_status}")
                )

def _render_menu(registry: SkillRegistry) -> str:
    skills = registry.all_skills(include_quarantined=True)
    if not skills:
        return "(none)"
    return "\n".join(f"- {s.name}: {s.description}" for s in skills)

async def _distill_candidates(
    llm: Any, turn_context: TurnContext, slice_text: str, menu: str
) -> list[dict[str, str]]:
    system = load_role_text("distiller") or _DEFAULT_DISTILLER_PROMPT
    user = (
        f"{slice_text}\n\n"
        f"## Existing skills (do not duplicate these)\n{menu}\n\n"
        "Now output the JSON object of candidates."
    )
    messages = [
        CanonicalMessage(role="system", content=system),
        CanonicalMessage(role="user", content=user),
    ]
    text = await _collect_stream_text(llm, messages, turn_context)
    return parse_candidates(text)

async def _collect_stream_text(llm: Any, messages: list[CanonicalMessage], turn_context: TurnContext) -> str:
    result = llm.stream(messages, [], turn_context)
    iterator = result if hasattr(result, "__aiter__") else await result
    chunks: list[str] = []
    async for ev in iterator:
        kind = getattr(ev, "kind", None)
        if kind == "text_delta" and ev.text:
            chunks.append(ev.text)
        elif kind == "error":
            break
    return "".join(chunks)

def _patch_text(arguments: dict[str, Any]) -> str:
    if not isinstance(arguments, dict):
        return ""
    for key in ("patch", "input", "content", "envelope"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""

def _touches_test_files(patch: str) -> bool:
    for line in patch.splitlines():
        if line.startswith("*** ") and _TEST_PATH_RE.search(line):
            return True
    return False

def _evidence_line(transition: _Transition) -> str:
    red = extract_test_counts(transition.prev_output)
    green = extract_test_counts(transition.cur_output)
    detail = f" ({red.failed} failed -> {green.passed} passed)" if red and green else ""
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return f"{_norm_cmd(transition.command)}: red -> green{detail} @ {ts}"

def _short(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    keep = max(0, limit - 24)
    return text[:keep] + f"\n…[{len(text) - keep} chars truncated]"

def _age_seconds(last_retrieved_ts: Any, path: Path, now: float) -> float | None:
    try:
        if last_retrieved_ts is not None:
            return now - float(last_retrieved_ts)
        return now - path.stat().st_mtime
    except (OSError, TypeError, ValueError):
        return None

_DEFAULT_DISTILLER_PROMPT = (
    "You are the Codewright skill distiller. Review the episode and output a "
    "STRICT JSON object {\"candidates\": [{\"type\",\"name\",\"description\","
    "\"body\"}]} of 0-3 reusable, durable notes worth remembering for future "
    "tasks. Use type fact|skill|lesson. name is lowercase letters/digits/hyphens "
    "(<=64). If nothing is worth saving, output {\"candidates\": []}."
)

__all__ = [
    "DistillationCoordinator",
    "VerificationTracker",
    "anti_cheat_ok",
    "decide_status",
    "extract_test_counts",
    "gate_candidate",
    "is_test_command",
    "parse_candidates",
    "render_skill_md",
    "set_skill_status",
    "write_provisional_skill",
]

