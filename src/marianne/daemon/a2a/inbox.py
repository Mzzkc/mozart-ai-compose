"""A2A task inbox — per-job persistent task queue.

Each running job has an inbox where other agents can deposit tasks.
Tasks persist across sheet boundaries — the conductor saves them
atomically with job state. When an A2A-enabled sheet starts, pending
tasks are injected as context for the musician.

Task lifecycle:
1. Agent A submits a task targeting agent B (A2ATaskSubmitted event)
2. Conductor routes it to B's inbox (persisted, A2ATaskRouted event)
3. B's next A2A-enabled sheet receives the task as context
4. B processes the task and produces artifacts
5. B completes the task (A2ATaskCompleted event) or fails it (A2ATaskFailed)
6. Results route back to A's inbox

Tasks are identified by a UUID. Each task tracks its lifecycle state
for observability and routing.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from marianne.core.logging import get_logger

_logger = get_logger("daemon.a2a.inbox")


class A2ATaskStatus(str, Enum):
    """Lifecycle state of an A2A task."""

    PENDING = "pending"
    """Task is waiting in the inbox for the target agent to pick up."""

    ACCEPTED = "accepted"
    """Target agent has picked up the task (injected into a sheet)."""

    COMPLETED = "completed"
    """Task was successfully completed with artifacts."""

    FAILED = "failed"
    """Task could not be fulfilled."""


class A2ATask(BaseModel):
    """A task in an agent's A2A inbox.

    Immutable once created — status transitions produce new snapshots
    saved atomically with job state.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(
        description="Unique task identifier (UUID)",
    )
    source_job_id: str = Field(
        description="Job ID of the requesting agent",
    )
    source_agent: str = Field(
        description="Name of the requesting agent",
    )
    target_agent: str = Field(
        description="Name of the target agent (this inbox's owner)",
    )
    description: str = Field(
        description="What the requesting agent needs done",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context for the task (files, references, etc.)",
    )
    status: A2ATaskStatus = Field(
        default=A2ATaskStatus.PENDING,
        description="Current task lifecycle state",
    )
    artifacts: dict[str, Any] = Field(
        default_factory=dict,
        description="Artifacts produced when the task is completed",
    )
    failure_reason: str | None = Field(
        default=None,
        description="Reason for failure, set when status is FAILED",
    )


class A2AInbox:
    """Per-job task inbox for A2A protocol.

    The conductor maintains one inbox per running job. Tasks are
    added when other agents submit work, and consumed when the
    owning agent's A2A-enabled sheets execute.

    Serialization: ``to_dict()`` / ``from_dict()`` for atomic
    persistence with job state. The inbox is saved alongside
    CheckpointState — same atomicity guarantees.

    Usage::

        inbox = A2AInbox(job_id="j1", agent_name="canyon")

        # Route a task
        task = inbox.submit_task(
            source_job_id="j2",
            source_agent="forge",
            description="Review architecture for module X",
        )

        # Inject pending tasks into sheet context
        context_text = inbox.render_pending_context()

        # Mark tasks as accepted when injected
        inbox.mark_accepted(task.task_id)

        # Complete a task with results
        inbox.complete_task(task.task_id, artifacts={"review": "..."})
    """

    def __init__(self, *, job_id: str, agent_name: str) -> None:
        if not job_id:
            raise ValueError("job_id must not be empty")
        if not agent_name:
            raise ValueError("agent_name must not be empty")

        self._job_id = job_id
        self._agent_name = agent_name
        self._tasks: dict[str, A2ATask] = {}

    @property
    def job_id(self) -> str:
        """The job this inbox belongs to."""
        return self._job_id

    @property
    def agent_name(self) -> str:
        """The agent this inbox belongs to."""
        return self._agent_name

    @property
    def task_count(self) -> int:
        """Total number of tasks in the inbox."""
        return len(self._tasks)

    @property
    def pending_count(self) -> int:
        """Number of pending tasks waiting for the agent."""
        return sum(
            1 for t in self._tasks.values()
            if t.status == A2ATaskStatus.PENDING
        )

    def submit_task(
        self,
        *,
        source_job_id: str,
        source_agent: str,
        description: str,
        context: dict[str, Any] | None = None,
    ) -> A2ATask:
        """Add a new task to the inbox.

        Called by the conductor when routing an A2ATaskSubmitted event.

        Args:
            source_job_id: Job ID of the requesting agent.
            source_agent: Name of the requesting agent.
            description: What needs to be done.
            context: Optional additional context.

        Returns:
            The created task with a unique ID.
        """
        task_id = str(uuid.uuid4())
        task = A2ATask(
            task_id=task_id,
            source_job_id=source_job_id,
            source_agent=source_agent,
            target_agent=self._agent_name,
            description=description,
            context=context or {},
        )
        self._tasks[task_id] = task

        _logger.info(
            "a2a.inbox.task_submitted",
            extra={
                "job_id": self._job_id,
                "task_id": task_id,
                "source_agent": source_agent,
                "target_agent": self._agent_name,
            },
        )

        return task

    def get_task(self, task_id: str) -> A2ATask | None:
        """Get a specific task by ID."""
        return self._tasks.get(task_id)

    def get_pending_tasks(self) -> list[A2ATask]:
        """Get all tasks in PENDING status.

        Used to inject pending work into the agent's next sheet.
        """
        return [
            t for t in self._tasks.values()
            if t.status == A2ATaskStatus.PENDING
        ]

    def mark_accepted(self, task_id: str) -> bool:
        """Mark a task as accepted (injected into a sheet).

        Args:
            task_id: The task to accept.

        Returns:
            True if the task was found and transitioned, False otherwise.
        """
        task = self._tasks.get(task_id)
        if task is None or task.status != A2ATaskStatus.PENDING:
            return False

        # Pydantic model is not frozen, so we can update in place
        self._tasks[task_id] = task.model_copy(
            update={"status": A2ATaskStatus.ACCEPTED}
        )

        _logger.debug(
            "a2a.inbox.task_accepted",
            extra={
                "job_id": self._job_id,
                "task_id": task_id,
                "agent_name": self._agent_name,
            },
        )
        return True

    def complete_task(
        self,
        task_id: str,
        *,
        artifacts: dict[str, Any] | None = None,
    ) -> bool:
        """Mark a task as completed with optional artifacts.

        Args:
            task_id: The task to complete.
            artifacts: Output artifacts from the completed work.

        Returns:
            True if the task was found and completed, False otherwise.
        """
        task = self._tasks.get(task_id)
        if task is None:
            return False
        if task.status not in (A2ATaskStatus.PENDING, A2ATaskStatus.ACCEPTED):
            return False

        self._tasks[task_id] = task.model_copy(
            update={
                "status": A2ATaskStatus.COMPLETED,
                "artifacts": artifacts or {},
            }
        )

        _logger.info(
            "a2a.inbox.task_completed",
            extra={
                "job_id": self._job_id,
                "task_id": task_id,
                "agent_name": self._agent_name,
                "has_artifacts": bool(artifacts),
            },
        )
        return True

    def fail_task(self, task_id: str, *, reason: str) -> bool:
        """Mark a task as failed.

        Args:
            task_id: The task that failed.
            reason: Why the task could not be fulfilled.

        Returns:
            True if the task was found and failed, False otherwise.
        """
        task = self._tasks.get(task_id)
        if task is None:
            return False
        if task.status not in (A2ATaskStatus.PENDING, A2ATaskStatus.ACCEPTED):
            return False

        self._tasks[task_id] = task.model_copy(
            update={
                "status": A2ATaskStatus.FAILED,
                "failure_reason": reason,
            }
        )

        _logger.info(
            "a2a.inbox.task_failed",
            extra={
                "job_id": self._job_id,
                "task_id": task_id,
                "agent_name": self._agent_name,
                "reason": reason,
            },
        )
        return True

    def render_pending_context(self) -> str:
        """Render pending tasks as markdown context for sheet injection.

        Produces a section that the musician reads to understand
        incoming A2A tasks. Injected as cadenza context on A2A-enabled
        sheets.

        Returns:
            Markdown string, or empty string if no pending tasks.
        """
        pending = self.get_pending_tasks()
        if not pending:
            return ""

        lines = [
            "## A2A Inbox — Pending Tasks",
            "",
            f"You have {len(pending)} task(s) from other agents:",
            "",
        ]

        for i, task in enumerate(pending, 1):
            lines.append(f"### Task {i}: from {task.source_agent}")
            lines.append(f"**Task ID:** `{task.task_id}`")
            lines.append(f"**Description:** {task.description}")
            if task.context:
                lines.append("**Context:**")
                for key, value in task.context.items():
                    lines.append(f"  - {key}: {value}")
            lines.append("")

        lines.append(
            "To complete a task, include its task_id in your output "
            "with the results. To decline, explain why."
        )

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for atomic persistence with job state.

        Returns:
            Dict representation suitable for JSON serialization.
        """
        return {
            "job_id": self._job_id,
            "agent_name": self._agent_name,
            "tasks": {
                tid: task.model_dump(mode="json")
                for tid, task in self._tasks.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> A2AInbox:
        """Restore from serialized state.

        Args:
            data: Dict from ``to_dict()``.

        Returns:
            Reconstructed inbox with all tasks.
        """
        inbox = cls(
            job_id=data["job_id"],
            agent_name=data["agent_name"],
        )
        for tid, task_data in data.get("tasks", {}).items():
            inbox._tasks[tid] = A2ATask.model_validate(task_data)
        return inbox
