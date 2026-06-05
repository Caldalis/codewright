from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolSpec(BaseModel):

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    supports_parallel: bool = False
    requires_approval: bool = False


class ParameterModel(BaseModel):

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def to_json_schema(cls) -> dict[str, Any]:
        schema = cls.model_json_schema()
        schema.pop("title", None)

        if "$defs" in schema and not _has_external_refs(schema, schema.get("$defs", {})):
            schema.pop("$defs", None)
        return schema


def _has_external_refs(node: Any, defs: dict[str, Any]) -> bool:
    if isinstance(node, dict):
        if "$ref" in node:
            ref = node["$ref"]
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                key = ref[len("#/$defs/"):]
                if key in defs:
                    return True
        return any(_has_external_refs(v, defs) for v in node.values())
    if isinstance(node, list):
        return any(_has_external_refs(v, defs) for v in node)
    return False
