from __future__ import annotations

import asyncio
import atexit
import locale
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
import weakref
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from codewright.tools.truncate import truncate_middle

_IS_WINDOWS = sys.platform == "win32"

DEFAULT_TIMEOUT_MS = 120_000
BODY_BUDGET_CHARS = 30_000
PAGE_BYTES = 24_000
_HEAD_CAP_BYTES = 128 * 1024
_TAIL_CAP_BYTES = 128 * 1024
_SPILL_CAP_BYTES = 8 * 1024 * 1024
_DRAIN_GRACE_SECONDS = 2.0
_REPEAT_FOLD_THRESHOLD = 3

_SHELL_DIR_REL = Path(".codewright") / "shell"

ENV_ARMOR: dict[str, str] = {
    "GIT_TERMINAL_PROMPT": "0",
    "GCM_INTERACTIVE": "never",
    "GIT_PAGER": "cat",
    "PAGER": "cat",
    "EDITOR": "true",
    "VISUAL": "true",
    "DEBIAN_FRONTEND": "noninteractive",
    "CI": "true",
    "NO_COLOR": "1",
    "CLICOLOR": "0",
    "FORCE_COLOR": "0",
    "TERM": "dumb",
    "PYTHONUNBUFFERED": "1",
}

@dataclass(frozen=True)
class ShellDialect:
    flavor: str  # "bash" | "cmd"
    path: str  # executable path; "" for the cmd fallback
    label: str  # one-liner surfaced in the tool spec so the model never guesses
    persistent: bool  # state capture/replay available (bash only)

def discover_dialect(explicit: str | None = None) -> ShellDialect:
    path = _find_bash(explicit)
    if path is None:
        return ShellDialect(
            flavor="cmd",
            path="",
            label=(
                "cmd.exe on Windows (bash not found — session state persistence "
                "is DISABLED; cd/export do not survive across calls)"
            ),
            persistent=False,
        )
    version = _probe_bash_version(path)
    platform = "Windows (Git Bash)" if _IS_WINDOWS else sys.platform
    return ShellDialect(
        flavor="bash",
        path=path,
        label=f"{version} at {path} on {platform}",
        persistent=True,
    )

def _find_bash(explicit: str | None) -> str | None:
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return str(p)
        return None
    if not _IS_WINDOWS:
        return shutil.which("bash") or shutil.which("sh")

    candidates: list[Path] = []
    git = shutil.which("git")
    if git:
        git_path = Path(git)
        for up in (2, 3):
            if len(git_path.parents) >= up:
                root = git_path.parents[up - 1]
                candidates.append(root / "usr" / "bin" / "bash.exe")
                candidates.append(root / "bin" / "bash.exe")
    for base in (
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
        os.environ.get("ProgramW6432"),
    ):
        if base:
            candidates.append(Path(base) / "Git" / "usr" / "bin" / "bash.exe")
            candidates.append(Path(base) / "Git" / "bin" / "bash.exe")
    local = os.environ.get("LocalAppData")
    if local:
        candidates.append(Path(local) / "Programs" / "Git" / "usr" / "bin" / "bash.exe")
    which_bash = shutil.which("bash")
    if which_bash and not _is_wsl_shim(which_bash):
        candidates.append(Path(which_bash))

    for candidate in candidates:
        if candidate.is_file() and not _is_wsl_shim(str(candidate)):
            return str(candidate)
    return None


def _is_wsl_shim(path: str) -> bool:
    """System32 bash.exe is the WSL launcher."""
    lowered = path.lower()
    return "\\windows\\system32" in lowered or "\\windows\\sysnative" in lowered
def _probe_bash_version(path: str) -> str:
    try:
        out = subprocess.run(
            [path, "--version"], capture_output=True, timeout=5, check=False
        )
        first = out.stdout.decode("utf-8", errors="replace").splitlines()
        if first:
            return first[0].strip()[:80]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "bash (version unknown)"



_REPLAY = """__cw_state="$CW_SHELL_STATE"
if [ -f "$__cw_state/env" ]; then source "$__cw_state/env" >/dev/null 2>&1 || true; fi
if [ -f "$__cw_state/funcs" ]; then source "$__cw_state/funcs" >/dev/null 2>&1 || true; fi
if [ -f "$__cw_state/cwd" ]; then cd -- "$(cat "$__cw_state/cwd")" >/dev/null 2>&1 || true; fi
unset CW_SHELL_STATE
"""
_CAPTURE_TRAP = (
    "trap '__cw_rc=$?; "
    'pwd > "$__cw_state/cwd.tmp" 2>/dev/null '
    '&& mv -f "$__cw_state/cwd.tmp" "$__cw_state/cwd" 2>/dev/null; '
    '{ pwd -W 2>/dev/null || pwd; } > "$__cw_state/cwdw.tmp" 2>/dev/null '
    '&& mv -f "$__cw_state/cwdw.tmp" "$__cw_state/cwd_w" 2>/dev/null; '
    'export -p > "$__cw_state/env.tmp" 2>/dev/null '
    '&& mv -f "$__cw_state/env.tmp" "$__cw_state/env" 2>/dev/null; '
    'declare -f > "$__cw_state/funcs.tmp" 2>/dev/null '
    '&& mv -f "$__cw_state/funcs.tmp" "$__cw_state/funcs" 2>/dev/null; '
    "exit $__cw_rc' EXIT\n"
)

def build_wrapper(command: str, capture_state: bool) -> str:
    script = _REPLAY
    if capture_state:
        script += _CAPTURE_TRAP
    return script + command.replace("\r\n", "\n") + "\n"

_ANSI_RE = re.compile(
    r"\x1b(?:\[[0-9;?]*[ -/]*[@-~]"  # CSI sequences (colors, cursor moves)
    r"|\][^\x07\x1b]*(?:\x07|\x1b\\)?"  # OSC sequences (titles, hyperlinks)
    r"|[@-Z\\-_])"  # two-byte escapes
)


def decode_output(data: bytes) -> str:
    if not _IS_WINDOWS or not data:
        return data.decode("utf-8", errors="replace")
    text = data.decode("utf-8", errors="replace")
    if text.count("�") <= max(1, len(text) // 100):
        return text
    enc = locale.getpreferredencoding(False) or "utf-8"
    return data.decode(enc, errors="replace")


def clean_output(text: str) -> str:
    text = _ANSI_RE.sub("", text)
    text = text.replace("\r\n", "\n")
    if "\r" in text:
        # A lone \r overwrites the line (progress bars); keep the final segment.
        text = "\n".join(seg.rsplit("\r", 1)[-1] for seg in text.split("\n"))
    return _fold_repeats(text)

def _fold_repeats(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        j = i + 1
        while j < len(lines) and lines[j] == line:
            j += 1
        run = j - i
        out.append(line)
        if run > _REPEAT_FOLD_THRESHOLD:
            out.append(f"[previous line repeated {run - 1} more times]")
        elif run > 1:
            out.extend(lines[i + 1 : j])
        i = j
    return "\n".join(out)

def render_output(head: bytes, tail: bytes, dropped: bool) -> str:
    text = decode_output(head)
    if dropped:
        text += "\n…[middle of output dropped]…\n"
    if tail:
        text += decode_output(tail)
    return truncate_middle(clean_output(text), BODY_BUDGET_CHARS)


@dataclass
class ShellJob:
    job_id: str
    session_key: str
    session_name: str
    command: str
    spill_path: Path
    wrapper_path: Path | None
    background: bool
    started: float
    proc: asyncio.subprocess.Process | None = None
    pgid: int | None = None
    status: str = "running"  # running | done | killed | timeout | spawn_error
    exit_code: int | None = None
    wall_time_s: float | None = None
    spill_bytes: int = 0
    spill_truncated: bool = False
    head_buf: bytes = b""
    tail_buf: bytes = b""
    dropped_middle: bool = False
    babysitter: asyncio.Task | None = None

    @property
    def running(self) -> bool:
        return self.status == "running"


@dataclass(frozen=True)
class ForegroundResult:
    job: ShellJob
    cwd: str

class SpawnFailure(Exception):
    """Raised internally when the child process cannot be created."""


_LIVE_MANAGERS: weakref.WeakSet[ShellManager] = weakref.WeakSet()
_ATEXIT_REGISTERED = False


def _kill_leaked_processes() -> None:
    for manager in list(_LIVE_MANAGERS):
        manager.kill_all_sync()


class ShellManager:
    def __init__(self, shell_path: str | None = None) -> None:
        self._dialect = discover_dialect(
            shell_path or os.environ.get("CODEWRIGHT_SHELL_PATH")
        )
        self._jobs: dict[str, ShellJob] = {}
        global _ATEXIT_REGISTERED
        _LIVE_MANAGERS.add(self)
        if not _ATEXIT_REGISTERED:
            atexit.register(_kill_leaked_processes)
            _ATEXIT_REGISTERED = True

    @property
    def dialect(self) -> ShellDialect:
        return self._dialect


    @staticmethod
    def session_key(session_id: str, name: str) -> str:
        raw = f"{session_id}.{name}"
        return re.sub(r"[^A-Za-z0-9._-]", "_", raw)[:80]

    def state_dir(self, root: Path, session_key: str) -> Path:
        d = root / _SHELL_DIR_REL / "state" / session_key
        d.mkdir(parents=True, exist_ok=True)
        return d

    def jobs_dir(self, root: Path) -> Path:
        d = root / _SHELL_DIR_REL / "jobs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def reset_session(self, root: Path, session_key: str) -> None:
        d = self.state_dir(root, session_key)
        for name in ("cwd", "cwd_w", "env", "funcs"):
            try:
                (d / name).unlink(missing_ok=True)
            except OSError:
                pass

    def session_cwd(self, root: Path, session_key: str) -> str | None:
        """Last captured cwd in OS-native (Windows drive-letter) form."""
        path = root / _SHELL_DIR_REL / "state" / session_key / "cwd_w"
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return text or None

    def get_job(self, job_id: str) -> ShellJob | None:
        return self._jobs.get(job_id)

    def jobs_for_session(self, session_key: str) -> list[ShellJob]:
        return [j for j in self._jobs.values() if j.session_key == session_key]

    async def run_foreground(
        self,
        *,
        root: Path,
        session_key: str,
        session_name: str,
        command: str,
        start_cwd: Path,
        timeout_ms: int,
        cancellation_token,
    ) -> ForegroundResult:
        job = await self._spawn(
            root=root,
            session_key=session_key,
            session_name=session_name,
            command=command,
            start_cwd=start_cwd,
            background=False,
        )
        if job.status == "spawn_error":
            return ForegroundResult(job=job, cwd=str(start_cwd))

        drain = asyncio.create_task(self._drain(job))
        wait_proc = asyncio.create_task(job.proc.wait())
        wait_cancel = asyncio.create_task(cancellation_token.wait())
        wait_timeout = asyncio.create_task(asyncio.sleep(timeout_ms / 1000.0))
        try:
            done, _ = await asyncio.wait(
                [wait_proc, wait_cancel, wait_timeout],
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            _terminate_tree(job.proc, job.pgid)
            job.status = "killed"
            job.exit_code = -1
            job.wall_time_s = round(time.monotonic() - job.started, 3)
            for t in (wait_cancel, wait_timeout, drain):
                if not t.done():
                    t.cancel()
            self._cleanup_wrapper(job)
            raise
        finally:
            for t in (wait_cancel, wait_timeout):
                if not t.done():
                    t.cancel()

        cancelled = wait_cancel in done and job.proc.returncode is None
        timed_out = wait_timeout in done and job.proc.returncode is None
        if job.proc.returncode is None:
            _terminate_tree(job.proc, job.pgid)
            try:
                await asyncio.wait_for(job.proc.wait(), timeout=5.0)
            except TimeoutError:
                pass
        try:
            await asyncio.wait_for(drain, timeout=_DRAIN_GRACE_SECONDS)
        except TimeoutError:
            drain.cancel()

        job.wall_time_s = round(time.monotonic() - job.started, 3)
        if cancelled:
            job.status = "killed"
            job.exit_code = -1
        elif timed_out:
            job.status = "timeout"
            job.exit_code = -1
        else:
            job.status = "done"
            job.exit_code = (
                job.proc.returncode if job.proc.returncode is not None else -1
            )
        self._cleanup_wrapper(job)
        cwd = self.session_cwd(root, session_key) or str(start_cwd)
        return ForegroundResult(job=job, cwd=cwd)

    async def spawn_background(
        self,
        *,
        root: Path,
        session_key: str,
        session_name: str,
        command: str,
        start_cwd: Path,
    ) -> ShellJob:
        job = await self._spawn(
            root=root,
            session_key=session_key,
            session_name=session_name,
            command=command,
            start_cwd=start_cwd,
            background=True,
        )
        if job.status != "spawn_error":
            job.babysitter = asyncio.create_task(self._babysit(job))
        return job

    async def _babysit(self, job: ShellJob) -> None:
        drain = asyncio.create_task(self._drain(job))
        await job.proc.wait()
        try:
            await asyncio.wait_for(drain, timeout=_DRAIN_GRACE_SECONDS)
        except TimeoutError:
            drain.cancel()
        if job.status == "running":
            job.status = "done"
            job.exit_code = (
                job.proc.returncode if job.proc.returncode is not None else -1
            )
        job.wall_time_s = round(time.monotonic() - job.started, 3)
        self._cleanup_wrapper(job)

    async def _spawn(
        self,
        *,
        root: Path,
        session_key: str,
        session_name: str,
        command: str,
        start_cwd: Path,
        background: bool,
    ) -> ShellJob:
        job_id = uuid.uuid4().hex[:8]
        wrapper_path: Path | None = None
        env = dict(os.environ)
        env.update(ENV_ARMOR)
        cwd = _existing_dir_or(start_cwd, root)

        # Provisional path (no mkdir yet); the directory is created inside the
        # try below so a filesystem failure becomes a clean spawn_error rather
        # than an unhandled exception escaping the handler.
        job = ShellJob(
            job_id=job_id,
            session_key=session_key,
            session_name=session_name,
            command=command,
            spill_path=root / _SHELL_DIR_REL / "jobs" / f"{job_id}.log",
            wrapper_path=None,
            background=background,
            started=time.monotonic(),
        )
        try:
            jobs_dir = self.jobs_dir(root)
            job.spill_path = jobs_dir / f"{job_id}.log"
            if self._dialect.flavor == "bash":
                # Make bash's own bin dir (coreutils on Git Bash: mv, cat, seq,
                # …) resolvable even when the host PATH lacks it — the state
                # capture trap depends on mv, and the model benefits too.
                bash_bin = str(Path(self._dialect.path).parent)
                env["PATH"] = bash_bin + os.pathsep + env.get("PATH", "")
                state_dir = self.state_dir(root, session_key)
                env["CW_SHELL_STATE"] = str(state_dir).replace("\\", "/")
                wrapper_path = jobs_dir / f"{job_id}.sh"
                wrapper_path.write_text(
                    build_wrapper(command, capture_state=not background),
                    encoding="utf-8",
                    newline="\n",
                )
                job.wrapper_path = wrapper_path
                job.proc, job.pgid = await _spawn_exec(
                    [self._dialect.path, str(wrapper_path)], cwd, env
                )
            else:
                job.proc, job.pgid = await _spawn_cmd(command, cwd, env)
        except OSError as exc:
            job.status = "spawn_error"
            job.exit_code = None
            job.wall_time_s = round(time.monotonic() - job.started, 3)
            job.head_buf = f"{type(exc).__name__}: {exc}".encode()
            self._cleanup_wrapper(job)
            self._jobs[job_id] = job
            return job

        self._jobs[job_id] = job
        return job

    async def _drain(self, job: ShellJob) -> None:
        stream = job.proc.stdout
        if stream is None:
            return
        head: list[bytes] = []
        head_len = 0
        tail: deque[bytes] = deque()
        tail_len = 0
        spill_fh = None
        try:
            spill_fh = job.spill_path.open("wb")
        except OSError:
            spill_fh = None
        try:
            while True:
                try:
                    chunk = await stream.read(4096)
                except (asyncio.CancelledError, BrokenPipeError, ConnectionResetError):
                    return
                if not chunk:
                    return
                if spill_fh is not None:
                    if job.spill_bytes < _SPILL_CAP_BYTES:
                        writable = chunk[: _SPILL_CAP_BYTES - job.spill_bytes]
                        try:
                            spill_fh.write(writable)
                            spill_fh.flush()
                        except OSError:
                            pass
                    elif not job.spill_truncated:
                        job.spill_truncated = True
                job.spill_bytes += len(chunk)
                if head_len < _HEAD_CAP_BYTES:
                    take = chunk[: _HEAD_CAP_BYTES - head_len]
                    head.append(take)
                    head_len += len(take)
                    chunk = chunk[len(take) :]
                    if not chunk:
                        continue
                tail.append(chunk)
                tail_len += len(chunk)
                while tail_len > _TAIL_CAP_BYTES and tail:
                    removed = tail.popleft()
                    tail_len -= len(removed)
                    job.dropped_middle = True
        finally:
            job.head_buf = b"".join(head)
            job.tail_buf = b"".join(tail)
            if spill_fh is not None:
                try:
                    spill_fh.close()
                except OSError:
                    pass

    def kill_job(self, job: ShellJob) -> bool:
        if not job.running or job.proc is None:
            return False
        _terminate_tree(job.proc, job.pgid)
        job.status = "killed"
        job.exit_code = -1
        job.wall_time_s = round(time.monotonic() - job.started, 3)
        return True

    def kill_all_sync(self) -> None:
        for job in list(self._jobs.values()):
            if job.running and job.proc is not None:
                _terminate_tree(job.proc, job.pgid)
                job.status = "killed"
                job.exit_code = -1

    def _cleanup_wrapper(self, job: ShellJob) -> None:
        if job.wrapper_path is not None:
            try:
                job.wrapper_path.unlink(missing_ok=True)
            except OSError:
                pass
            job.wrapper_path = None

    async def aclose(self) -> None:
        for job in list(self._jobs.values()):
            if job.running:
                self.kill_job(job)
            if job.babysitter is not None and not job.babysitter.done():
                job.babysitter.cancel()
                try:
                    await job.babysitter
                except (asyncio.CancelledError, Exception):
                    pass
                job.babysitter = None
        # Close subprocess transports now: on Windows proactor they otherwise
        # finalize via __del__ after the loop is gone and warn loudly.
        for job in list(self._jobs.values()):
            transport = getattr(job.proc, "_transport", None)
            if transport is not None:
                try:
                    transport.close()
                except Exception:
                    pass
        await asyncio.sleep(0)


def _existing_dir_or(candidate: Path, fallback: Path) -> Path:
    """The session's saved cwd may have been deleted since the last call."""
    try:
        return candidate if candidate.is_dir() else fallback
    except OSError:
        return fallback
async def _spawn_exec(
    argv: list[str], cwd: Path, env: dict[str, str]
) -> tuple[asyncio.subprocess.Process, int | None]:
    kwargs: dict = {
        "cwd": str(cwd),
        "env": env,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
    }
    if _IS_WINDOWS:
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        )
        proc = await asyncio.create_subprocess_exec(*argv, **kwargs)
        return proc, None
    kwargs["preexec_fn"] = os.setsid
    proc = await asyncio.create_subprocess_exec(*argv, **kwargs)
    return proc, proc.pid

async def _spawn_cmd(
    command: str, cwd: Path, env: dict[str, str]
) -> tuple[asyncio.subprocess.Process, int | None]:
    """cmd.exe fallback: create_subprocess_shell hands the line to cmd /c verbatim,
    avoiding the list2cmdline re-quoting that mangles quoted arguments."""
    kwargs: dict = {
        "cwd": str(cwd),
        "env": env,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
    }
    if _IS_WINDOWS:
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        )
        proc = await asyncio.create_subprocess_shell(command, **kwargs)
        return proc, None
    kwargs["preexec_fn"] = os.setsid
    proc = await asyncio.create_subprocess_shell(command, **kwargs)
    return proc, proc.pid


def _terminate_tree(proc: asyncio.subprocess.Process, pgid: int | None) -> None:
    if proc.returncode is not None:
        return
    if _IS_WINDOWS:
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                check=False,
                capture_output=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            try:
                proc.kill()
            except ProcessLookupError:
                return
        return
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
            return
        except ProcessLookupError:
            return
    try:
        proc.kill()
    except ProcessLookupError:
        return
