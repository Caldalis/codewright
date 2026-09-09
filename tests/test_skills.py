"""Phase 1 (READ half) of self-improving skills: discovery, frontmatter parsing,
the model-invoked `skill` tool (menu + retrieval + [未验证] marking), A-type fact
always-on injection, and the metrics sidecar."""

from __future__ import annotations

from pathlib import Path

import pytest

from codewright.agent.cancellation import CancellationToken
from codewright.agent.skills import SkillRegistry, load_skills, parse_frontmatter
from codewright.agent.turn_context import TurnContext
from codewright.prompts.builder import PromptBuilder
from codewright.protocol import AskForApproval, PermissionProfile
from codewright.tools.errors import RespondToModelError
from codewright.tools.handlers.skill import SkillHandler
from codewright.tools.invocation import ToolInvocation

PROVISIONAL_MARK = "[未验证]"


def _write_skill(
    workspace: Path,
    name: str,
    *,
    description: str = "what it does and when to use it",
    status: str | None = None,
    source: str | None = None,
    skill_type: str | None = None,
    body: str = "BODY",
) -> Path:
    d = workspace / "skills" / name
    d.mkdir(parents=True, exist_ok=True)
    lines = ["---", f"name: {name}", f"description: {description}"]
    meta: dict[str, str] = {}
    if source:
        meta["cw-source"] = source
    if status:
        meta["cw-status"] = status
    if skill_type:
        meta["cw-type"] = skill_type
    if meta:
        lines.append("metadata:")
        for k, v in meta.items():
            lines.append(f"  {k}: {v}")
    lines += ["---", "", body, ""]
    (d / "SKILL.md").write_text("\n".join(lines), encoding="utf-8")
    return d


def _inv(args: dict) -> ToolInvocation:
    return ToolInvocation(
        session=None,
        turn_context=TurnContext(
            turn_id="t",
            cwd=Path("."),
            model="m",
            permission_profile=PermissionProfile.WORKSPACE_WRITE,
            approval_policy=AskForApproval.NEVER,
            cancellation_token=CancellationToken(),
        ),
        call_id="c1",
        tool_name="skill",
        arguments=args,
        cancellation_token=CancellationToken(),
    )


def _ctx() -> TurnContext:
    return TurnContext(
        turn_id="t1",
        cwd=Path("/tmp/work"),
        model="gpt-x",
        permission_profile=PermissionProfile.WORKSPACE_WRITE,
        approval_policy=AskForApproval.ON_REQUEST,
        cancellation_token=CancellationToken(),
    )


# --------------------------------------------------------------------------- #
# frontmatter parser
# --------------------------------------------------------------------------- #


def test_parse_frontmatter_metadata_and_quotes():
    text = (
        "---\n"
        "name: x\n"
        'description: "Hello: world"\n'
        "metadata:\n"
        "  cw-status: provisional\n"
        "  cw-source: learned\n"
        "---\n"
        "BODY\n"
    )
    fm, body = parse_frontmatter(text)
    assert fm["name"] == "x"
    assert fm["description"] == "Hello: world"  # colon inside value, quotes stripped
    assert fm["metadata"]["cw-status"] == "provisional"
    assert fm["metadata"]["cw-source"] == "learned"
    assert body.strip() == "BODY"


def test_parse_frontmatter_absent_returns_whole_text():
    fm, body = parse_frontmatter("no frontmatter here\nsecond line")
    assert fm == {}
    assert body == "no frontmatter here\nsecond line"


def test_parse_frontmatter_real_repo_example_skills():
    repo = Path(__file__).resolve().parent.parent
    skill_dir = repo / ".agents" / "skills"
    if not skill_dir.is_dir():
        pytest.skip("repo .agents/skills not present")
    files = list(skill_dir.glob("*/SKILL.md"))
    assert files, "expected example SKILL.md files under .agents/skills"
    for path in files:
        fm, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        assert isinstance(fm.get("name"), str) and fm["name"].strip()
        assert isinstance(fm.get("description"), str) and fm["description"].strip()
        assert body.strip()


# --------------------------------------------------------------------------- #
# registry discovery + defaults
# --------------------------------------------------------------------------- #


def test_missing_skills_dir_is_empty(tmp_path: Path):
    reg = load_skills(tmp_path)
    assert reg.all_skills() == []
    assert reg.always_on_text() is None
    assert reg.get("anything") is None


def test_status_and_source_defaults(tmp_path: Path):
    _write_skill(tmp_path, "authored-one")  # no metadata
    _write_skill(tmp_path, "learned-one", source="learned")  # learned, no status
    reg = load_skills(tmp_path)
    by = {s.name: s for s in reg.all_skills()}
    assert by["authored-one"].source == "authored"
    assert by["authored-one"].status == "trusted"  # human-authored is vouched for
    assert by["learned-one"].source == "learned"
    assert by["learned-one"].status == "provisional"  # learned default is conservative


def test_name_falls_back_to_dir_when_frontmatter_missing(tmp_path: Path):
    d = tmp_path / "skills" / "dir-name"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("no frontmatter, just a body", encoding="utf-8")
    reg = load_skills(tmp_path)
    skills = reg.all_skills()
    assert len(skills) == 1
    assert skills[0].name == "dir-name"


# --------------------------------------------------------------------------- #
# skill tool: menu rendering
# --------------------------------------------------------------------------- #


def test_menu_lists_skills_and_marks_provisional(tmp_path: Path):
    _write_skill(tmp_path, "trusted-skill", source="authored")
    _write_skill(tmp_path, "learned-skill", source="learned", status="provisional")
    desc = SkillHandler(load_skills(tmp_path)).spec().description
    assert "trusted-skill" in desc
    assert "learned-skill" in desc
    prov_line = next(ln for ln in desc.splitlines() if "learned-skill" in ln)
    assert PROVISIONAL_MARK in prov_line
    trusted_line = next(ln for ln in desc.splitlines() if "trusted-skill" in ln)
    assert PROVISIONAL_MARK not in trusted_line
    # the description must explain what the [未验证] label means
    assert PROVISIONAL_MARK in desc
    assert "provisional" in desc.lower()


def test_empty_menu_when_no_skills(tmp_path: Path):
    desc = SkillHandler(load_skills(tmp_path)).spec().description
    assert "no skills available yet" in desc


def test_skill_spec_schema_is_generated_not_handwritten(tmp_path: Path):
    spec = SkillHandler(load_skills(tmp_path)).spec()
    # comes from ParameterModel.to_json_schema()
    assert spec.parameters["type"] == "object"
    assert "name" in spec.parameters["properties"]
    assert spec.supports_parallel is True


# --------------------------------------------------------------------------- #
# skill tool: retrieval
# --------------------------------------------------------------------------- #


async def test_retrieval_returns_body_with_dir_header_and_records_metric(tmp_path: Path):
    _write_skill(tmp_path, "add-migration", body="Run scripts/migrate.py to add one.")
    reg = load_skills(tmp_path)
    result = await SkillHandler(reg).handle(_inv({"name": "add-migration"}))
    assert result.success is True
    assert "Run scripts/migrate.py to add one." in result.body
    assert "Skill directory:" in result.body  # lets the model read bundled files
    assert result.structured_data["status"] == "trusted"
    assert reg.metrics.get("add-migration")["retrieved"] == 1


async def test_retrieval_of_provisional_appends_warning_note(tmp_path: Path):
    _write_skill(tmp_path, "maybe", source="learned", status="provisional", body="DO X")
    result = await SkillHandler(load_skills(tmp_path)).handle(_inv({"name": "maybe"}))
    assert "DO X" in result.body
    assert PROVISIONAL_MARK in result.body
    assert result.structured_data["status"] == "provisional"


async def test_unknown_skill_raises_respond_to_model(tmp_path: Path):
    with pytest.raises(RespondToModelError):
        await SkillHandler(load_skills(tmp_path)).handle(_inv({"name": "nope"}))


async def test_invalid_arguments_raise_respond_to_model(tmp_path: Path):
    with pytest.raises(RespondToModelError):
        await SkillHandler(load_skills(tmp_path)).handle(_inv({}))


# --------------------------------------------------------------------------- #
# quarantine + facts
# --------------------------------------------------------------------------- #


async def test_quarantined_hidden_from_menu_and_not_loadable(tmp_path: Path):
    _write_skill(tmp_path, "bad-skill", source="learned", status="quarantined")
    reg = load_skills(tmp_path)
    assert reg.all_skills() == []  # excluded unless include_quarantined
    assert len(reg.all_skills(include_quarantined=True)) == 1
    assert "bad-skill" not in SkillHandler(reg).spec().description
    with pytest.raises(RespondToModelError):
        await SkillHandler(reg).handle(_inv({"name": "bad-skill"}))


def test_trusted_fact_is_always_on_and_excluded_from_menu(tmp_path: Path):
    _write_skill(tmp_path, "use-uv", skill_type="fact", body="Use `uv run pytest`.")
    reg = load_skills(tmp_path)
    text = reg.always_on_text()
    assert text and "Use `uv run pytest`." in text
    # always-on facts are injected every turn, so they are not in the pull-menu
    assert "use-uv" not in SkillHandler(reg).spec().description


def test_provisional_fact_is_in_menu_not_always_on(tmp_path: Path):
    _write_skill(tmp_path, "maybe-fact", skill_type="fact", source="learned")
    reg = load_skills(tmp_path)
    assert reg.always_on_text() is None  # provisional facts are not yet always-on
    assert "maybe-fact" in SkillHandler(reg).spec().description


# --------------------------------------------------------------------------- #
# prompt builder integration
# --------------------------------------------------------------------------- #


def test_prompt_builder_injects_learned_facts_block():
    pb = PromptBuilder("SYS")
    msgs = pb.build(_ctx(), history=[], user_input="hi", learned_facts="Use uv.")
    dev = msgs[1].content
    assert "<learned_facts>" in dev
    assert "Use uv." in dev


def test_prompt_builder_omits_block_when_no_facts():
    pb = PromptBuilder("SYS")
    msgs = pb.build(_ctx(), history=[], user_input="hi")
    assert "<learned_facts>" not in msgs[1].content


def test_registry_is_a_skillregistry(tmp_path: Path):
    assert isinstance(load_skills(tmp_path), SkillRegistry)


async def test_skill_load_survives_corrupt_metrics(tmp_path: Path):
    # A corrupt .cw-metrics.json must not fail a skill load (bug 1 / fix d).
    _write_skill(tmp_path, "x", source="learned", status="provisional", body="THE BODY")
    (tmp_path / "skills" / ".cw-metrics.json").write_text('{"x": "not-a-dict"}', encoding="utf-8")
    result = await SkillHandler(load_skills(tmp_path)).handle(_inv({"name": "x"}))
    assert result.success is True  # bookkeeping failure is swallowed, the load still succeeds
    assert "THE BODY" in result.body
