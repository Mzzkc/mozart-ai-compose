"""Tests for technique system configuration models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from marianne.core.config.techniques import TechniqueConfig, TechniqueKind


class TestTechniqueConfig:
    def test_kind_enum_values(self) -> None:
        assert TechniqueKind.SKILL == "skill"
        assert TechniqueKind.MCP == "mcp"
        assert TechniqueKind.PROTOCOL == "protocol"

    def test_skill_kind(self) -> None:
        tc = TechniqueConfig(kind="skill", phases=["work", "inspect"])  # type: ignore[arg-type]
        assert tc.kind == TechniqueKind.SKILL

    def test_mcp_kind(self) -> None:
        tc = TechniqueConfig(kind="mcp", phases=["recon"], config={"server": "github"})  # type: ignore[arg-type]
        assert tc.kind == TechniqueKind.MCP

    def test_protocol_kind(self) -> None:
        tc = TechniqueConfig(kind="protocol", phases=["all"])  # type: ignore[arg-type]
        assert tc.kind == TechniqueKind.PROTOCOL

    def test_string_kind_values(self) -> None:
        for kind_str in ["skill", "mcp", "protocol"]:
            tc = TechniqueConfig(kind=kind_str, phases=[])  # type: ignore[arg-type]
            assert tc.kind.value == kind_str

    def test_invalid_kind_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TechniqueConfig(kind="invalid", phases=[])  # type: ignore[arg-type]

    def test_from_dict(self) -> None:
        tc = TechniqueConfig.model_validate(
            {"kind": "mcp", "phases": ["work"], "config": {"server": "fs"}}
        )
        assert tc.kind == TechniqueKind.MCP

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TechniqueConfig(kind="skill", phases=[], unknown_field="bad")  # type: ignore[call-arg]

    def test_empty_phases_allowed(self) -> None:
        tc = TechniqueConfig(kind="skill", phases=[])  # type: ignore[arg-type]
        assert tc.phases == []


class TestJobConfigTechniques:
    def _minimal(self, **kw: object) -> dict:
        d: dict = {
            "name": "t",
            "workspace": "/tmp/t",
            "sheet": {"size": 1, "total_items": 1},
            "prompt": {"template": "x"},
        }
        d.update(kw)
        return d

    def test_no_techniques_default_empty(self) -> None:
        from marianne.core.config.job import JobConfig

        assert JobConfig.model_validate(self._minimal()).techniques == {}

    def test_backward_compat_no_techniques(self) -> None:
        from marianne.core.config.job import JobConfig

        assert isinstance(JobConfig.model_validate(self._minimal()).techniques, dict)

    def test_techniques_from_dict(self) -> None:
        from marianne.core.config.job import JobConfig

        c = JobConfig.model_validate(
            self._minimal(techniques={"github": {"kind": "mcp", "phases": ["work"]}})
        )
        assert "github" in c.techniques and c.techniques["github"].kind == TechniqueKind.MCP

    def test_techniques_from_yaml(self) -> None:
        from marianne.core.config.job import JobConfig

        c = JobConfig.model_validate(
            self._minimal(
                techniques={
                    "a2a": {"kind": "protocol", "phases": ["all"]},
                    "m": {"kind": "skill", "phases": ["c"]},
                }
            )
        )
        assert len(c.techniques) == 2

    def test_multiple_technique_kinds_coexist(self) -> None:
        from marianne.core.config.job import JobConfig

        c = JobConfig.model_validate(
            self._minimal(
                techniques={
                    "a": {"kind": "mcp", "phases": ["w"]},
                    "b": {"kind": "skill", "phases": ["r"]},
                    "c": {"kind": "protocol", "phases": ["all"]},
                }
            )
        )
        assert len(c.techniques) == 3

    def test_technique_with_empty_config(self) -> None:
        from marianne.core.config.job import JobConfig

        c = JobConfig.model_validate(
            self._minimal(techniques={"s": {"kind": "skill", "phases": ["w"], "config": {}}})
        )
        assert c.techniques["s"].config == {}

    def test_invalid_technique_kind_rejected(self) -> None:
        from marianne.core.config.job import JobConfig

        with pytest.raises(ValidationError):
            JobConfig.model_validate(
                self._minimal(techniques={"bad": {"kind": "invalid_kind", "phases": []}})
            )
