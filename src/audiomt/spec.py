"""MR specification: what an LLM (or a person) is allowed to write, and how it is checked and run."""
from __future__ import annotations

import json
import re
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from audiomt.ops import OPS


class OpCall(BaseModel):
    model_config = ConfigDict(extra="allow")  # parameters arrive as extra keys: {"op": "gain", "db": -6}
    op: str

    def params(self) -> dict:
        return dict(self.model_extra or {})


class MRSpec(BaseModel):
    name: str = Field(pattern=r"^[a-z0-9_]{3,40}$")
    ops: list[OpCall] = Field(min_length=1, max_length=3)
    expected: Literal["same_label"] = "same_label"
    applies_to: Literal["all"] | list[str] = "all"
    rationale: str = Field(max_length=300)

    @field_validator("ops")
    @classmethod
    def check_ops(cls, ops: list[OpCall]) -> list[OpCall]:
        for call in ops:
            if call.op not in OPS:
                raise ValueError(f"unknown op '{call.op}'")
            spec, given = OPS[call.op].params, call.params()
            if set(given) != set(spec):
                raise ValueError(f"{call.op}: expected params {sorted(spec)}, got {sorted(given)}")
            for key, p in spec.items():
                v = given[key]
                if p.choices:
                    if v not in p.choices:
                        raise ValueError(f"{call.op}.{key}={v!r} not in {p.choices}")
                elif not isinstance(v, (int, float)) or not p.lo <= v <= p.hi:
                    raise ValueError(f"{call.op}.{key}={v!r} outside [{p.lo}, {p.hi}]")
        return ops

    def apply(self, y: np.ndarray, sr: int) -> np.ndarray:
        out = y
        for call in self.ops:
            out = OPS[call.op].fn(out, sr, **call.params())
        return out.astype(np.float32)


def parse_llm_reply(text: str) -> tuple[list[MRSpec], list[dict]]:
    """Return (valid specs, rejections with reasons). Never executes anything the LLM wrote."""
    body = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        items = json.loads(body)
    except json.JSONDecodeError as e:
        return [], [{"stage": "json", "reason": str(e)}]
    valid, rejected = [], []
    for i, item in enumerate(items if isinstance(items, list) else [items]):
        try:
            valid.append(MRSpec.model_validate(item))
        except ValidationError as e:
            rejected.append({"stage": "schema", "index": i, "reason": e.errors()[0]["msg"]})
    return valid, rejected


def operation_catalog() -> str:
    """Prompt text generated from OPS, so the prompt and the code can never disagree."""
    lines = []
    for name, d in OPS.items():
        ps = ", ".join(f"{k} in {list(p.choices)}" if p.choices else f"{k} in [{p.lo}, {p.hi}]"
                       for k, p in d.params.items())
        lines.append(f"- {name}: {d.doc}; params: {ps}")
    return "\n".join(lines)
