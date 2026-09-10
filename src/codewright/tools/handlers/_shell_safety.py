from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

_KEYWORDS = frozenset(
    {
        "if", "then", "elif", "else", "fi",
        "for", "while", "until", "do", "done",
        "case", "esac", "in",
        "{", "}", "!", "time", "function",
    }
)
_RO_HEADS = frozenset(
    {
        "cat", "ls", "pwd", "echo", "printf", "head", "tail", "wc", "nl",
        "which", "whereis", "env", "printenv", "uname", "whoami", "id",
        "date", "stat", "file", "du", "df", "tree",
        "grep", "rg", "egrep", "fgrep", "sort", "uniq", "cut", "tr",
        "column", "basename", "dirname", "realpath", "readlink", "type",
        "command", "test", "[", "true", "false", "sleep", "diff", "cmp",
        "md5sum", "sha1sum", "sha256sum",
    }
)

_GIT_RO_SUBCOMMANDS = frozenset(
    {
        "status", "diff", "log", "show", "blame", "shortlog", "describe",
        "rev-parse", "ls-files", "grep", "var", "help",
    }
)

_SHELL_INTERPRETERS = frozenset({"sh", "bash", "zsh", "dash", "ksh", "fish"})
_DOWNLOADERS = frozenset({"curl", "wget"})
_HARD_FLAG_HEADS = frozenset(
    {
        "mkfs", "dd", "format", "shutdown", "reboot", "poweroff", "halt",
        "sudo", "su", "runas", "eval",
    }
)

_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_RM_DANGEROUS_FLAG_RE = re.compile(r"^-[A-Za-z]*[rRf]")
_SUBSTITUTION_RE = re.compile(r"\$\(|`|<\(|>\(")
_SEPARATOR_RE = re.compile(r"^[;&|]+$")


@dataclass(frozen=True)
class CommandAnalysis:
    heads: tuple[str, ...]
    flagged: tuple[str, ...]
    readonly: bool
    backgrounded: bool
    # Subset of `flagged` that must never resolve to "allowed" without a human.
    # Everything else in `flagged` is destructive-but-recoverable (rm -rf, git
    # reset --hard) and an unattended run needs it to make progress.
    hard_flagged: tuple[str, ...] = ()


def has_background_operator(command: str) -> bool:
    adjacent = {"&", ">", "<", "|"}
    in_single = in_double = False
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if in_single:
            if ch == "'":
                in_single = False
            i += 1
            continue
        if in_double:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == '"':
                in_double = False
            i += 1
            continue
        if ch == "\\":
            i += 2  # backslash-escaped char is literal
            continue
        if ch == "'":
            in_single = True
            i += 1
            continue
        if ch == '"':
            in_double = True
            i += 1
            continue
        if ch == "&":
            prev = command[i - 1] if i > 0 else ""
            nxt = command[i + 1] if i + 1 < n else ""
            if prev not in adjacent and nxt not in adjacent:
                return True
        i += 1
    return False

_MAX_SUBSTITUTION_DEPTH = 4


def analyze_command(command: str, _depth: int = 0) -> CommandAnalysis:
    flagged: list[str] = []
    backgrounded = has_background_operator(command)
    if _SUBSTITUTION_RE.search(command):
        flagged.append("command/process substitution ($(...), `...`)")
        # Analyse what is *inside* the substitution too. The outer pass reads
        # `echo $(sudo rm -rf /etc)` as a plain `echo`, so a hard flag hidden in
        # the body would otherwise resolve to "allow".
        if _depth >= _MAX_SUBSTITUTION_DEPTH:
            flagged.append(
                "unanalysable substitution (nested deeper than "
                f"{_MAX_SUBSTITUTION_DEPTH})"
            )
        else:
            for body in _substitution_bodies(command):
                if not body.strip():
                    continue
                inner = analyze_command(body, _depth + 1)
                flagged.extend(f"inside substitution: {f}" for f in inner.flagged)

    # Newlines separate statements just like `;` for segmentation purposes.
    normalized = command.replace("\r\n", "\n").replace("\n", " ; ")
    try:
        lex = shlex.shlex(normalized, posix=True, punctuation_chars=";&|<>")
        lex.whitespace_split = True
        tokens = list(lex)
    except ValueError:
        flagged.append("unparseable quoting (unbalanced quote?)")
        # Fail closed: this early return used to leave `hard_flagged` empty, so
        # a command the analyser could not parse resolved to "allow" -- the one
        # case where knowing nothing was treated as knowing it was safe.
        return CommandAnalysis(
            heads=(),
            flagged=tuple(flagged),
            readonly=False,
            backgrounded=backgrounded,
            hard_flagged=tuple(f for f in flagged if is_hard_flag(f)),
        )

    segments: list[list[str]] = []
    separators: list[str] = []  # separator *before* segments[i] ("" for the first)
    current: list[str] = []
    has_write_redirect = False
    last_sep = ""
    for token in tokens:
        if _SEPARATOR_RE.match(token):
            if current:
                segments.append(current)
                separators.append(last_sep)
                current = []
            last_sep = token
            continue
        if ">" in token or "<" in token:
            if ">" in token:
                has_write_redirect = True
            continue  # redirect operators and their fd prefixes are not words
        current.append(token)
    if current:
        segments.append(current)
        separators.append(last_sep)

    heads: list[str] = []
    readonly = bool(segments) and not has_write_redirect and not flagged
    for words, sep in zip(segments, separators, strict=True):
        head, args = _head_and_args(words)
        if head is None:
            continue
        heads.append(head)
        flagged.extend(_segment_flags(head, args, sep))
        if not _is_readonly_segment(head, args):
            readonly = False
        if "&" in sep:
            readonly = False

    if flagged:
        readonly = False
    return CommandAnalysis(
        heads=tuple(heads),
        flagged=tuple(flagged),
        readonly=readonly,
        backgrounded=backgrounded,
        hard_flagged=tuple(f for f in flagged if is_hard_flag(f)),
    )

def _substitution_bodies(command: str) -> list[str]:
    """Inner text of every `$(...)`, `<(...)`, `>(...)` and backtick span.

    Segment analysis only ever saw the *outer* command, so `echo $(sudo rm -rf
    /etc)` was read as a plain `echo`. The bodies are handed back here to be
    analysed on their own.
    """
    bodies: list[str] = []

    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if ch == "`":
            end = command.find("`", i + 1)
            if end == -1:
                break
            bodies.append(command[i + 1 : end])
            i = end + 1
            continue
        if ch in "$<>" and i + 1 < n and command[i + 1] == "(":
            depth = 0
            j = i + 1
            while j < n:
                if command[j] == "(":
                    depth += 1
                elif command[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if depth != 0:
                break  # unbalanced; the shlex pass flags this separately
            bodies.append(command[i + 2 : j])
            i = j + 1
            continue
        i += 1
    return bodies


def _head_and_args(words: list[str]) -> tuple[str | None, list[str]]:
    idx = 0
    while idx < len(words):
        word = words[idx]
        if word in _KEYWORDS or _ASSIGNMENT_RE.match(word):
            idx += 1
            continue
        return _basename(word), words[idx + 1 :]
    return None, []

def _basename(word: str) -> str:
    return word.replace("\\", "/").rsplit("/", 1)[-1].lower()

_SUBSTITUTION_FLAG_PREFIX = "inside substitution: "


def is_hard_flag(flag: str) -> bool:
    """Flags an unattended run must refuse rather than auto-approve.

    A hard flag stays hard however deeply it is nested: `$(sudo ...)` is exactly
    as privileged as a bare `sudo`.
    """
    while flag.startswith(_SUBSTITUTION_FLAG_PREFIX):
        flag = flag[len(_SUBSTITUTION_FLAG_PREFIX) :]
    return flag.startswith(_HARD_FLAG_PREFIXES)


# Kept next to the strings they match so the two cannot drift apart.
_HARD_FLAG_PREFIXES = (
    "privileged or destructive command:",
    "pipeline into shell interpreter:",
    "git push --force",
    # A safety analysis that could not run is not a safe command. Without this
    # an unbalanced quote takes the early return below, leaves `hard_flagged`
    # empty, and a command nothing understood resolves to "allow".
    "unparseable quoting",
    "unanalysable substitution",
)


def _segment_flags(head: str, args: list[str], separator: str) -> list[str]:
    flags: list[str] = []
    if head in _HARD_FLAG_HEADS:
        flags.append(f"privileged or destructive command: {head}")
    if head == "rm" and any(
        _RM_DANGEROUS_FLAG_RE.match(a) or a in ("--recursive", "--force")
        for a in args
    ):
        flags.append("rm with recursive/force flag")
    if head == "git":
        flags.extend(_git_flags(args))
    if head in _SHELL_INTERPRETERS and "|" in separator:
        flags.append(f"pipeline into shell interpreter: | {head}")
    return flags
def _git_flags(args: list[str]) -> list[str]:
    flags: list[str] = []
    arg_set = set(args)
    if "push" in arg_set and arg_set & {"--force", "-f", "--force-with-lease"}:
        flags.append("git push --force")
    if "reset" in arg_set and "--hard" in arg_set:
        flags.append("git reset --hard")
    if "clean" in arg_set and any(
        a.startswith("-") and "f" in a.lstrip("-") for a in args
    ):
        flags.append("git clean -f")
    return flags
def _is_readonly_segment(head: str, args: list[str]) -> bool:
    if head == "git":
        sub = next((a for a in args if not a.startswith("-")), None)
        return sub in _GIT_RO_SUBCOMMANDS
    return head in _RO_HEADS
