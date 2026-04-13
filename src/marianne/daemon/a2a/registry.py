"""Agent card registry — tracks running agents for A2A discovery.

When a score starts, the conductor registers its agent card here.
Other agents can query the registry to discover who's running and
what skills they offer. The registry is the A2A discovery mechanism.

Lifecycle:
- ``register(job_id, card)`` — called on job start
- ``deregister(job_id)`` — called on job completion/cancellation
- ``query()`` / ``query_by_skill()`` — called by agents during execution
- ``get_job_id_for_agent(name)`` — resolve agent name to job_id for routing

The registry is in-memory only — it reflects the current state of running
jobs. On conductor restart, cards are re-registered as jobs resume.
"""

from __future__ import annotations

from marianne.core.config.a2a import AgentCard
from marianne.core.logging import get_logger

_logger = get_logger("daemon.a2a.registry")


class AgentCardRegistry:
    """In-memory registry of agent cards for running jobs.

    Thread-safe for single-threaded asyncio use (no locking needed).
    The baton's event loop is the sole writer; query methods are read-only.

    Usage::

        registry = AgentCardRegistry()

        # On job start
        card = AgentCard(name="canyon", description="...", skills=[...])
        registry.register("job-123", card)

        # Discovery
        agents = registry.query()  # all running agents
        architects = registry.query_by_skill("architecture-review")

        # On job end
        registry.deregister("job-123")
    """

    def __init__(self) -> None:
        # job_id → AgentCard
        self._cards: dict[str, AgentCard] = {}
        # agent_name → job_id (reverse index for routing)
        self._name_to_job: dict[str, str] = {}

    @property
    def count(self) -> int:
        """Number of registered agent cards."""
        return len(self._cards)

    def register(self, job_id: str, card: AgentCard) -> None:
        """Register an agent card for a running job.

        If the job_id is already registered, the card is replaced.
        If an agent with the same name is registered under a different
        job, the old registration is removed (agent name must be unique).

        Args:
            job_id: The job this agent card belongs to.
            card: The agent's identity card.
        """
        if not job_id:
            raise ValueError("job_id must not be empty")
        if not card.name:
            raise ValueError("agent card name must not be empty")

        # If this agent name is already registered under a different job,
        # remove the stale entry first (name uniqueness invariant).
        existing_job = self._name_to_job.get(card.name)
        if existing_job is not None and existing_job != job_id:
            _logger.warning(
                "a2a.registry.name_conflict",
                extra={
                    "agent_name": card.name,
                    "old_job_id": existing_job,
                    "new_job_id": job_id,
                },
            )
            self._cards.pop(existing_job, None)

        # If this job_id had a different card, clean up the old name index
        old_card = self._cards.get(job_id)
        if old_card is not None and old_card.name != card.name:
            self._name_to_job.pop(old_card.name, None)

        self._cards[job_id] = card
        self._name_to_job[card.name] = job_id

        _logger.info(
            "a2a.registry.registered",
            extra={
                "job_id": job_id,
                "agent_name": card.name,
                "skill_count": len(card.skills),
            },
        )

    def deregister(self, job_id: str) -> AgentCard | None:
        """Remove an agent card when a job completes or is cancelled.

        Args:
            job_id: The job to deregister.

        Returns:
            The removed card, or None if the job wasn't registered.
        """
        card = self._cards.pop(job_id, None)
        if card is not None:
            self._name_to_job.pop(card.name, None)
            _logger.info(
                "a2a.registry.deregistered",
                extra={"job_id": job_id, "agent_name": card.name},
            )
        return card

    def get(self, job_id: str) -> AgentCard | None:
        """Get the agent card for a specific job.

        Args:
            job_id: The job to look up.

        Returns:
            The agent card, or None if not registered.
        """
        return self._cards.get(job_id)

    def get_job_id_for_agent(self, agent_name: str) -> str | None:
        """Resolve an agent name to its job_id.

        Used by the conductor to route A2A tasks — when an agent
        sends a task to "canyon", this resolves to canyon's job_id
        so the task can be persisted in the correct inbox.

        Args:
            agent_name: The target agent's name.

        Returns:
            The job_id, or None if the agent isn't running.
        """
        return self._name_to_job.get(agent_name)

    def query(self) -> list[AgentCard]:
        """List all registered agent cards.

        Returns:
            List of all currently registered cards (snapshot).
        """
        return list(self._cards.values())

    def query_by_skill(self, skill_id: str) -> list[AgentCard]:
        """Find agents that offer a specific skill.

        Args:
            skill_id: The skill identifier to search for.

        Returns:
            List of agent cards that declare the given skill.
        """
        return [
            card for card in self._cards.values()
            if any(s.id == skill_id for s in card.skills)
        ]

    def clear(self) -> None:
        """Remove all registrations. Used on conductor shutdown."""
        count = len(self._cards)
        self._cards.clear()
        self._name_to_job.clear()
        if count > 0:
            _logger.info(
                "a2a.registry.cleared",
                extra={"deregistered_count": count},
            )
