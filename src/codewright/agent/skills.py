from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

SkillStatus = Literal["provisional", "trusted", "quarantined"]
SkillSource = Literal["learned", "authored"]
SkillType = Literal["fact", "skill", "lesson"]  # A / B / C in the design notes

_SKILLS_DIRNAME = "skills"
_SKILL_FILE = "SKILL.md"
_METRICS_FILE = ".cw-metrics.json"

_MD_SOURCE = "cw-source"
_MD_STATUS = "cw-status"
_MD_TYPE = "cw-type"
_MD_EVIDENCE = "cw-evidence"


@dataclass(frozen=True)
class SkillConfig:
    name: str
    description: str
    body_path: Path
    base_dir: Path
    source: SkillSource = "authored"
    status: SkillStatus = "trusted"
    type: SkillType = "skill"
    evidence: str | None = None


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end == -1:
        return {}, text
    fm = _parse_block(lines[1:end])
    body = "\n".join(lines[end + 1 :])
    return fm, body

def _is_indented(line: str) -> bool:
    return line[:1] in (" ", "\t")

def _parse_block(lines: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or _is_indented(line):
            i += 1
            continue
        key, sep, val = line.partition(":")
        if not sep:
            i += 1
            continue
        key = key.strip()
        val = val.strip()
        block: list[str] = []
        j = i + 1
        while j < n and (_is_indented(lines[j]) or not lines[j].strip()):
            if lines[j].strip():
                block.append(lines[j])
            j += 1
        if val:
            result[key] = _unquote(val)
            i += 1  # inline value: do not consume the following block
        elif key.lower() == "metadata":
            nested: dict[str, str] = {}
            for bl in block:
                k2, s2, v2 = bl.strip().partition(":")
                if s2:
                    nested[k2.strip()] = _unquote(v2.strip())
            result[key] = nested
            i = j
        elif block:
            result[key] = _unquote(" ".join(b.strip() for b in block))
            i = j
        else:
            result[key] = ""
            i += 1
    return result

def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s

def _coerce_source(value: Any) -> SkillSource:
    return "learned" if value == "learned" else "authored"

def _coerce_status(value: Any, source: SkillSource) -> SkillStatus:
    if value in ("provisional", "trusted", "quarantined"):
        return value
    return "trusted" if source == "authored" else "provisional"

def _coerce_type(value: Any) -> SkillType:
    if value in ("fact", "skill", "lesson"):
        return value
    return "skill"

class SkillMetrics:
    def __init__(self, root: Path) -> None:
        self._path = Path(root) / _METRICS_FILE

    @property
    def path(self) -> Path:
        return self._path

    def _read(self) -> dict[str, dict[str, Any]]:
        if not self._path.is_file():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, data: dict[str, dict[str, Any]]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_name(self._path.name + ".tmp")
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            tmp.replace(self._path)
        except OSError:
            return

    def get(self, name: str) -> dict[str, Any]:
        return self._read().get(name, {})

    def record_retrieval(self, name: str) -> None:
        data = self._read()
        entry = data.setdefault(name, {})
        now = time.time()
        entry["retrieved"] = int(entry.get("retrieved", 0)) + 1
        entry["last_retrieved_ts"] = now
        entry.setdefault("created_ts", now)
        entry.setdefault("success_after", 0)
        entry.setdefault("failure_after", 0)
        self._write(data)

    def record_success(self, name: str) -> None:
        self._bump(name, "success_after")

    def record_failure(self, name: str) -> None:
        self._bump(name, "failure_after")

    def _bump(self, name: str, field_name: str) -> None:
        data = self._read()
        entry = data.setdefault(name, {})
        entry[field_name] = int(entry.get(field_name, 0)) + 1
        entry.setdefault("created_ts", time.time())
        entry.setdefault("retrieved", 0)
        self._write(data)

class SkillRegistry:

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._metrics = SkillMetrics(self._root)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def metrics(self) -> SkillMetrics:
        return self._metrics

    def _scan(self) -> list[SkillConfig]:
        out: list[SkillConfig] = []
        if not self._root.is_dir():
            return out
        for child in sorted(self._root.iterdir()):
            if not child.is_dir():
                continue
            skill_file = child / _SKILL_FILE
            if not skill_file.is_file():
                continue
            try:
                text = skill_file.read_text(encoding="utf-8")
            except OSError:
                continue
            fm, _body = parse_frontmatter(text)
            name = fm.get("name")
            if not isinstance(name, str) or not name.strip():
                name = child.name  # spec: name == dir; fall back to dir name
            description = fm.get("description")
            if not isinstance(description, str):
                description = ""
            metadata = fm.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            source = _coerce_source(metadata.get(_MD_SOURCE))
            evidence = metadata.get(_MD_EVIDENCE)
            out.append(
                SkillConfig(
                    name=name.strip(),
                    description=description.strip(),
                    body_path=skill_file,
                    base_dir=child,
                    source=source,
                    status=_coerce_status(metadata.get(_MD_STATUS), source),
                    type=_coerce_type(metadata.get(_MD_TYPE)),
                    evidence=evidence if isinstance(evidence, str) else None,
                )
            )
        return out

    def all_skills(self, *, include_quarantined: bool = False) -> list[SkillConfig]:
        skills = self._scan()
        if include_quarantined:
            return skills
        return [s for s in skills if s.status != "quarantined"]

    def get(self, name: str) -> SkillConfig | None:
        for skill in self._scan():
            if skill.name == name:
                return skill
        return None

    def has(self, name: str) -> bool:
        return self.get(name) is not None

    def load_body(self, skill: SkillConfig, *, include_dir_header: bool = True) -> str:
        try:
            text = skill.body_path.read_text(encoding="utf-8")
        except OSError:
            return ""
        _fm, body = parse_frontmatter(text)
        body = body.strip()
        if not include_dir_header:
            return body
        header = f"Skill directory: {skill.base_dir}"
        return f"{header}\n\n{body}" if body else header

    def facts(self) -> list[SkillConfig]:
        return [s for s in self._scan() if s.type == "fact" and s.status == "trusted"]

    def always_on_text(self) -> str | None:
        facts = self.facts()
        if not facts:
            return None
        blocks: list[str] = []
        for skill in facts:
            body = self.load_body(skill, include_dir_header=False)
            if body:
                blocks.append(f"### {skill.name}\n{body}")
        return "\n\n".join(blocks) if blocks else None

    def record_retrieval(self, name: str) -> None:
        self._metrics.record_retrieval(name)

def load_skills(workspace_root: Path) -> SkillRegistry:
    return SkillRegistry(Path(workspace_root) / _SKILLS_DIRNAME)

__all__ = [
    "SkillConfig",
    "SkillMetrics",
    "SkillRegistry",
    "SkillSource",
    "SkillStatus",
    "SkillType",
    "load_skills",
    "parse_frontmatter",
]
