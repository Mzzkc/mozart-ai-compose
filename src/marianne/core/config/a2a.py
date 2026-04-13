"""Agent-to-Agent (A2A) protocol configuration models.

Defines agent cards and skill declarations for A2A discovery and
task delegation between running agents.

An agent card is registered with the conductor when a job starts.
Other agents can query "who's running and what can they do?" to
discover available services for delegation.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class A2ASkill(BaseModel):
    """A skill declaration on an agent card.

    Skills describe what an agent can do for other agents. They're
    used for discovery — an agent looking for help can query the
    registry and find agents with matching skills.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        description="Unique skill identifier, e.g. 'architecture-review'",
    )
    description: str = Field(
        description="Human-readable description of what this skill provides",
    )


class AgentCard(BaseModel):
    """Agent identity card for A2A protocol discovery.

    When a score runs, its agent card is registered with the conductor.
    The card describes the agent's capabilities so other agents can
    discover and delegate tasks.

    Example YAML::

        agent_card:
          name: canyon
          description: "Systems architect — traces boundaries"
          skills:
            - id: architecture-review
              description: "Review system architecture"
            - id: boundary-analysis
              description: "Trace and analyze system boundaries"
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description="Agent name, matches the score's agent identity",
    )
    description: str = Field(
        description="Brief description of the agent's role and capabilities",
    )
    skills: list[A2ASkill] = Field(
        default_factory=list,
        description="Skills this agent offers for A2A delegation",
    )
