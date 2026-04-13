"""Fleet configuration models.

A fleet is a concert-of-concerts: multiple agent scores launched and managed
as a unit. Fleet configs define score membership, group dependencies, and
fleet-level operations.

Fleets are one level of nesting only: fleet → score → sheet. No
fleet-of-fleets — sane limits to prevent recursive explosion.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FleetScoreEntry(BaseModel):
    """One score in a fleet roster.

    Each entry references a score YAML path and optionally assigns
    it to a group for dependency ordering.
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        description="Path to the score YAML file, relative to fleet config location",
    )
    group: str | None = Field(
        default=None,
        description="Group name for dependency ordering. "
        "Scores in the same group start concurrently.",
    )


class FleetGroupConfig(BaseModel):
    """Dependency declaration for a fleet group.

    Groups without depends_on start immediately. Groups with depends_on
    wait for all named groups to complete their first cycle before starting.
    """

    model_config = ConfigDict(extra="forbid")

    depends_on: list[str] = Field(
        default_factory=list,
        description="Group names this group depends on. "
        "All listed groups must complete before this group starts.",
    )


class FleetConfig(BaseModel):
    """Top-level fleet configuration.

    A fleet launches and manages multiple agent scores as a unit.
    Run like any score: ``mzt run fleet.yaml``. Fleet-level operations
    act on all members.

    Example YAML::

        name: marianne-dev-fleet
        type: fleet

        scores:
          - path: scores/agents/canyon.yaml
            group: architects
          - path: scores/agents/forge.yaml
            group: builders

        groups:
          architects:
            depends_on: []
          builders:
            depends_on: [architects]
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description="Fleet name, used for fleet-level operations",
    )
    type: Literal["fleet"] = Field(
        default="fleet",
        description="Config type discriminator. Must be 'fleet'.",
    )
    scores: list[FleetScoreEntry] = Field(
        description="Score entries in this fleet",
    )
    groups: dict[str, FleetGroupConfig] = Field(
        default_factory=dict,
        description="Group definitions with dependency ordering",
    )

    @model_validator(mode="after")
    def _validate_group_dependencies(self) -> FleetConfig:
        """Validate group dependency graph: no undefined refs, no cycles."""
        defined_groups = set(self.groups.keys())

        # Check for undefined group references in depends_on
        for group_name, group_config in self.groups.items():
            for dep in group_config.depends_on:
                if dep not in defined_groups:
                    raise ValueError(
                        f"Group '{group_name}' depends on undefined group '{dep}'. "
                        f"Defined groups: {sorted(defined_groups)}"
                    )

        # Check for circular dependencies via topological traversal
        visited: set[str] = set()
        in_stack: set[str] = set()

        def _visit(name: str) -> None:
            if name in in_stack:
                raise ValueError(
                    f"Circular dependency detected involving group '{name}'"
                )
            if name in visited:
                return
            in_stack.add(name)
            group_cfg = self.groups.get(name)
            if group_cfg:
                for dep in group_cfg.depends_on:
                    _visit(dep)
            in_stack.discard(name)
            visited.add(name)

        for group_name in self.groups:
            _visit(group_name)

        return self
