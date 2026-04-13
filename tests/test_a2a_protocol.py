"""Tests for A2A protocol — registry, inbox, event routing.

Tests cover:
- AgentCardRegistry: register, deregister, query, name uniqueness
- A2AInbox: submit, accept, complete, fail, serialization
- Integration: inbox context rendering, edge cases
"""

from __future__ import annotations

import pytest

from marianne.core.config.a2a import A2ASkill, AgentCard
from marianne.daemon.a2a.inbox import A2AInbox, A2ATask, A2ATaskStatus
from marianne.daemon.a2a.registry import AgentCardRegistry


# =============================================================================
# AgentCardRegistry tests
# =============================================================================


class TestAgentCardRegistry:
    """Test the agent card registry for A2A discovery."""

    def test_register_and_query(self) -> None:
        """Basic register and query flow."""
        registry = AgentCardRegistry()
        card = AgentCard(
            name="canyon",
            description="Systems architect",
            skills=[A2ASkill(id="arch-review", description="Review architecture")],
        )
        registry.register("job-1", card)

        assert registry.count == 1
        assert registry.get("job-1") is card
        assert registry.query() == [card]

    def test_register_multiple_agents(self) -> None:
        """Multiple agents can be registered."""
        registry = AgentCardRegistry()
        cards = [
            AgentCard(name="canyon", description="Architect"),
            AgentCard(name="forge", description="Builder"),
            AgentCard(name="sentinel", description="Auditor"),
        ]
        for i, card in enumerate(cards):
            registry.register(f"job-{i}", card)

        assert registry.count == 3
        assert len(registry.query()) == 3

    def test_deregister(self) -> None:
        """Deregistering removes the card and cleans up indices."""
        registry = AgentCardRegistry()
        card = AgentCard(name="canyon", description="Architect")
        registry.register("job-1", card)

        removed = registry.deregister("job-1")
        assert removed is card
        assert registry.count == 0
        assert registry.get("job-1") is None
        assert registry.get_job_id_for_agent("canyon") is None

    def test_deregister_nonexistent(self) -> None:
        """Deregistering a non-existent job returns None."""
        registry = AgentCardRegistry()
        assert registry.deregister("nonexistent") is None

    def test_query_by_skill(self) -> None:
        """Query agents by skill ID."""
        registry = AgentCardRegistry()
        canyon = AgentCard(
            name="canyon",
            description="Architect",
            skills=[
                A2ASkill(id="arch-review", description="Review architecture"),
                A2ASkill(id="boundary", description="Trace boundaries"),
            ],
        )
        forge = AgentCard(
            name="forge",
            description="Builder",
            skills=[
                A2ASkill(id="impl", description="Implement code"),
            ],
        )
        registry.register("j1", canyon)
        registry.register("j2", forge)

        arch = registry.query_by_skill("arch-review")
        assert len(arch) == 1
        assert arch[0].name == "canyon"

        impl = registry.query_by_skill("impl")
        assert len(impl) == 1
        assert impl[0].name == "forge"

        none = registry.query_by_skill("nonexistent")
        assert len(none) == 0

    def test_get_job_id_for_agent(self) -> None:
        """Resolve agent name to job_id."""
        registry = AgentCardRegistry()
        card = AgentCard(name="canyon", description="Architect")
        registry.register("job-42", card)

        assert registry.get_job_id_for_agent("canyon") == "job-42"
        assert registry.get_job_id_for_agent("nonexistent") is None

    def test_name_uniqueness(self) -> None:
        """Agent names must be unique — re-registering same name replaces."""
        registry = AgentCardRegistry()
        card1 = AgentCard(name="canyon", description="First")
        card2 = AgentCard(name="canyon", description="Second")

        registry.register("job-1", card1)
        registry.register("job-2", card2)

        # The old job-1 registration should be gone
        assert registry.count == 1
        assert registry.get("job-1") is None
        assert registry.get("job-2") is card2
        assert registry.get_job_id_for_agent("canyon") == "job-2"

    def test_replace_card_for_same_job(self) -> None:
        """Re-registering the same job replaces the card."""
        registry = AgentCardRegistry()
        card1 = AgentCard(name="canyon", description="First")
        card2 = AgentCard(name="canyon", description="Second")

        registry.register("job-1", card1)
        registry.register("job-1", card2)

        assert registry.count == 1
        assert registry.get("job-1") is card2

    def test_replace_card_different_name(self) -> None:
        """Replacing a job's card with a different agent name cleans up."""
        registry = AgentCardRegistry()
        card1 = AgentCard(name="canyon", description="Was canyon")
        card2 = AgentCard(name="forge", description="Now forge")

        registry.register("job-1", card1)
        assert registry.get_job_id_for_agent("canyon") == "job-1"

        registry.register("job-1", card2)
        assert registry.get_job_id_for_agent("canyon") is None
        assert registry.get_job_id_for_agent("forge") == "job-1"
        assert registry.count == 1

    def test_clear(self) -> None:
        """Clear removes all registrations."""
        registry = AgentCardRegistry()
        for i in range(5):
            registry.register(
                f"job-{i}",
                AgentCard(name=f"agent-{i}", description=f"Agent {i}"),
            )
        assert registry.count == 5

        registry.clear()
        assert registry.count == 0
        assert registry.query() == []

    def test_register_empty_job_id_raises(self) -> None:
        """Empty job_id raises ValueError."""
        registry = AgentCardRegistry()
        card = AgentCard(name="canyon", description="Arch")
        with pytest.raises(ValueError, match="job_id must not be empty"):
            registry.register("", card)

    def test_register_empty_name_raises(self) -> None:
        """Empty agent name raises ValueError."""
        registry = AgentCardRegistry()
        card = AgentCard(name="", description="Empty")
        with pytest.raises(ValueError, match="agent card name must not be empty"):
            registry.register("job-1", card)


# =============================================================================
# A2AInbox tests
# =============================================================================


class TestA2AInbox:
    """Test the per-job A2A task inbox."""

    def test_submit_task(self) -> None:
        """Submit a task and verify it's in the inbox."""
        inbox = A2AInbox(job_id="j1", agent_name="canyon")
        task = inbox.submit_task(
            source_job_id="j2",
            source_agent="forge",
            description="Review module X architecture",
        )

        assert task.task_id  # UUID generated
        assert task.source_agent == "forge"
        assert task.target_agent == "canyon"
        assert task.status == A2ATaskStatus.PENDING
        assert inbox.task_count == 1
        assert inbox.pending_count == 1

    def test_get_pending_tasks(self) -> None:
        """Get only pending tasks."""
        inbox = A2AInbox(job_id="j1", agent_name="canyon")
        t1 = inbox.submit_task(
            source_job_id="j2", source_agent="forge", description="Task 1",
        )
        inbox.submit_task(
            source_job_id="j3", source_agent="sentinel", description="Task 2",
        )

        pending = inbox.get_pending_tasks()
        assert len(pending) == 2

        # Accept one — pending should drop to 1
        inbox.mark_accepted(t1.task_id)
        assert inbox.pending_count == 1
        assert len(inbox.get_pending_tasks()) == 1

    def test_task_lifecycle_complete(self) -> None:
        """Full lifecycle: submit → accept → complete."""
        inbox = A2AInbox(job_id="j1", agent_name="canyon")
        task = inbox.submit_task(
            source_job_id="j2", source_agent="forge", description="Review",
        )

        assert inbox.mark_accepted(task.task_id)
        updated = inbox.get_task(task.task_id)
        assert updated is not None
        assert updated.status == A2ATaskStatus.ACCEPTED

        assert inbox.complete_task(
            task.task_id, artifacts={"review": "LGTM"},
        )
        completed = inbox.get_task(task.task_id)
        assert completed is not None
        assert completed.status == A2ATaskStatus.COMPLETED
        assert completed.artifacts == {"review": "LGTM"}

    def test_task_lifecycle_fail(self) -> None:
        """Full lifecycle: submit → fail."""
        inbox = A2AInbox(job_id="j1", agent_name="canyon")
        task = inbox.submit_task(
            source_job_id="j2", source_agent="forge", description="Review",
        )

        assert inbox.fail_task(task.task_id, reason="Not my domain")
        failed = inbox.get_task(task.task_id)
        assert failed is not None
        assert failed.status == A2ATaskStatus.FAILED
        assert failed.failure_reason == "Not my domain"

    def test_complete_already_completed_returns_false(self) -> None:
        """Completing an already-completed task returns False."""
        inbox = A2AInbox(job_id="j1", agent_name="canyon")
        task = inbox.submit_task(
            source_job_id="j2", source_agent="forge", description="Review",
        )
        inbox.complete_task(task.task_id, artifacts={"r": "done"})

        # Try completing again
        assert not inbox.complete_task(task.task_id, artifacts={"r": "again"})

    def test_fail_already_failed_returns_false(self) -> None:
        """Failing an already-failed task returns False."""
        inbox = A2AInbox(job_id="j1", agent_name="canyon")
        task = inbox.submit_task(
            source_job_id="j2", source_agent="forge", description="Review",
        )
        inbox.fail_task(task.task_id, reason="Nope")

        assert not inbox.fail_task(task.task_id, reason="Double nope")

    def test_accept_nonexistent_returns_false(self) -> None:
        """Accepting a nonexistent task returns False."""
        inbox = A2AInbox(job_id="j1", agent_name="canyon")
        assert not inbox.mark_accepted("nonexistent-id")

    def test_context_with_extra_data(self) -> None:
        """Tasks can carry additional context."""
        inbox = A2AInbox(job_id="j1", agent_name="canyon")
        task = inbox.submit_task(
            source_job_id="j2",
            source_agent="forge",
            description="Review",
            context={"file": "src/main.py", "priority": "P0"},
        )
        assert task.context == {"file": "src/main.py", "priority": "P0"}

    def test_render_pending_context_empty(self) -> None:
        """Empty inbox renders empty string."""
        inbox = A2AInbox(job_id="j1", agent_name="canyon")
        assert inbox.render_pending_context() == ""

    def test_render_pending_context(self) -> None:
        """Pending tasks render as markdown context."""
        inbox = A2AInbox(job_id="j1", agent_name="canyon")
        inbox.submit_task(
            source_job_id="j2",
            source_agent="forge",
            description="Review architecture for module X",
            context={"file": "src/arch.py"},
        )

        rendered = inbox.render_pending_context()
        assert "## A2A Inbox" in rendered
        assert "forge" in rendered
        assert "Review architecture for module X" in rendered
        assert "src/arch.py" in rendered

    def test_serialization_round_trip(self) -> None:
        """to_dict / from_dict preserves all state."""
        inbox = A2AInbox(job_id="j1", agent_name="canyon")
        t1 = inbox.submit_task(
            source_job_id="j2", source_agent="forge", description="Task 1",
            context={"key": "val"},
        )
        inbox.submit_task(
            source_job_id="j3", source_agent="sentinel", description="Task 2",
        )
        inbox.mark_accepted(t1.task_id)
        inbox.complete_task(t1.task_id, artifacts={"result": "done"})

        # Round-trip
        data = inbox.to_dict()
        restored = A2AInbox.from_dict(data)

        assert restored.job_id == "j1"
        assert restored.agent_name == "canyon"
        assert restored.task_count == 2

        # Verify completed task
        rt1 = restored.get_task(t1.task_id)
        assert rt1 is not None
        assert rt1.status == A2ATaskStatus.COMPLETED
        assert rt1.artifacts == {"result": "done"}
        assert rt1.context == {"key": "val"}

    def test_empty_inbox_serialization(self) -> None:
        """Empty inbox round-trips cleanly."""
        inbox = A2AInbox(job_id="j1", agent_name="canyon")
        data = inbox.to_dict()
        restored = A2AInbox.from_dict(data)
        assert restored.task_count == 0
        assert restored.job_id == "j1"

    def test_empty_job_id_raises(self) -> None:
        """Empty job_id raises ValueError."""
        with pytest.raises(ValueError, match="job_id must not be empty"):
            A2AInbox(job_id="", agent_name="canyon")

    def test_empty_agent_name_raises(self) -> None:
        """Empty agent_name raises ValueError."""
        with pytest.raises(ValueError, match="agent_name must not be empty"):
            A2AInbox(job_id="j1", agent_name="")


# =============================================================================
# A2ATask model tests
# =============================================================================


class TestA2ATask:
    """Test the A2ATask Pydantic model."""

    def test_default_status(self) -> None:
        """Default status is PENDING."""
        task = A2ATask(
            task_id="abc",
            source_job_id="j1",
            source_agent="forge",
            target_agent="canyon",
            description="Test",
        )
        assert task.status == A2ATaskStatus.PENDING
        assert task.artifacts == {}
        assert task.failure_reason is None

    def test_model_dump(self) -> None:
        """Model serializes cleanly."""
        task = A2ATask(
            task_id="abc",
            source_job_id="j1",
            source_agent="forge",
            target_agent="canyon",
            description="Test",
            context={"key": "val"},
        )
        data = task.model_dump(mode="json")
        assert data["task_id"] == "abc"
        assert data["context"] == {"key": "val"}

    def test_model_validate(self) -> None:
        """Model validates from dict."""
        data = {
            "task_id": "abc",
            "source_job_id": "j1",
            "source_agent": "forge",
            "target_agent": "canyon",
            "description": "Test",
            "status": "completed",
            "artifacts": {"review": "LGTM"},
        }
        task = A2ATask.model_validate(data)
        assert task.status == A2ATaskStatus.COMPLETED
        assert task.artifacts == {"review": "LGTM"}
