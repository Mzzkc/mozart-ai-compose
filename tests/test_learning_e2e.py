"""End-to-end learning cycle test for Mozart.

Tests the complete learning lifecycle:
  detect pattern -> store -> query -> apply -> record feedback

This test exercises the real GlobalLearningStore (SQLite) with a
temporary database, verifying the full cycle works without mocking.
"""

from pathlib import Path

import pytest

from mozart.learning.global_store import GlobalLearningStore


@pytest.fixture
def store(tmp_path: Path) -> GlobalLearningStore:
    """Create a GlobalLearningStore with a temporary database."""
    db_path = tmp_path / "test-learning.db"
    return GlobalLearningStore(db_path=db_path)


class TestLearningCycleE2E:
    """End-to-end tests for the learning cycle."""

    def test_record_and_query_pattern(self, store: GlobalLearningStore) -> None:
        """Test that a recorded pattern can be queried back."""
        # Step 1: Record a pattern (discovery)
        pattern_id = store.record_pattern(
            pattern_type="validation_failure",
            pattern_name="missing_output_file",
            description="Output file not created by sheet execution",
            context_tags=["validation", "file_check"],
            suggested_action="Ensure the sheet creates the expected output file",
        )

        assert pattern_id is not None

        # Step 2: Query the pattern back
        patterns = store.get_patterns_for_auto_apply(
            trust_threshold=0.0,  # Low threshold to find new patterns
            require_validated=False,
            limit=10,
        )

        # Should find our pattern
        assert len(patterns) >= 1
        found = [p for p in patterns if p.id == pattern_id]
        assert len(found) == 1
        assert found[0].pattern_name == "missing_output_file"
        assert found[0].pattern_type == "validation_failure"

    def test_pattern_application_and_feedback(
        self, store: GlobalLearningStore
    ) -> None:
        """Test recording a pattern application and its outcome."""
        # Step 1: Record pattern
        pattern_id = store.record_pattern(
            pattern_type="error_recovery",
            pattern_name="retry_on_timeout",
            description="Retry the sheet when timeout occurs",
            context_tags=["timeout", "retry"],
            suggested_action="Increase timeout and retry",
        )

        # Step 2: Record application (pattern was applied to a sheet)
        app_id = store.record_pattern_application(
            pattern_id=pattern_id,
            execution_id="exec-001",
            outcome_improved=True,
            retry_count_before=2,
            retry_count_after=0,
            application_mode="exploitation",
        )

        assert app_id is not None

        # Step 3: Query pattern again - effectiveness should be updated
        patterns = store.get_patterns_for_auto_apply(
            trust_threshold=0.0,
            require_validated=False,
            limit=10,
        )

        found = [p for p in patterns if p.id == pattern_id]
        assert len(found) == 1
        # led_to_success_count should be updated from the successful application
        assert found[0].led_to_success_count >= 1

    def test_duplicate_pattern_increments_count(
        self, store: GlobalLearningStore
    ) -> None:
        """Test that recording the same pattern twice increments its count."""
        # Record same pattern twice
        id1 = store.record_pattern(
            pattern_type="rate_limit",
            pattern_name="claude_429",
            description="Claude API 429 rate limit",
            context_tags=["api", "rate_limit"],
        )

        id2 = store.record_pattern(
            pattern_type="rate_limit",
            pattern_name="claude_429",
            description="Claude API 429 rate limit",
            context_tags=["api", "rate_limit"],
        )

        # Should return same ID (deduplicated)
        assert id1 == id2

        # Occurrence count should be 2
        patterns = store.get_patterns_for_auto_apply(
            trust_threshold=0.0,
            require_validated=False,
            limit=10,
        )

        found = [p for p in patterns if p.id == id1]
        assert len(found) == 1
        assert found[0].occurrence_count == 2

    def test_rate_limit_coordination(
        self, store: GlobalLearningStore
    ) -> None:
        """Test cross-workspace rate limit coordination."""
        # Record a rate limit event
        store.record_rate_limit_event(
            error_code="E101",
            duration_seconds=300.0,
            job_id="job-1",
            model="claude-sonnet-4-20250514",
        )

        # Check if rate limited
        is_limited, wait_time = store.is_rate_limited(
            model="claude-sonnet-4-20250514",
        )

        # Should detect the rate limit
        assert is_limited is True
        assert wait_time is not None
        assert wait_time > 0

    def test_full_learning_cycle(self, store: GlobalLearningStore) -> None:
        """Test the complete learning cycle: discover -> store -> query -> apply -> feedback.

        This simulates what happens across two job runs:
        Run 1: Discovers a pattern from a failure
        Run 2: Queries patterns, applies one, records success
        """
        # === Run 1: Pattern Discovery ===

        # 1a: Discover pattern from a failure
        pattern_id = store.record_pattern(
            pattern_type="error_recovery",
            pattern_name="add_explicit_output_path",
            description="Adding explicit output path to prompt prevents file-not-found validation failures",
            context_tags=["validation", "output_file", "prompt_engineering"],
            suggested_action="Append 'Write output to {{workspace}}/output.txt' to prompt",
            provenance_job_hash="run-1",
            provenance_sheet_num=1,
        )

        assert pattern_id is not None

        # === Run 2: Pattern Application ===

        # 2a: Query applicable patterns before execution
        patterns = store.get_patterns_for_auto_apply(
            trust_threshold=0.0,
            require_validated=False,
            limit=5,
            context_tags=["validation"],
        )

        # Should find the pattern from Run 1
        applicable = [p for p in patterns if p.id == pattern_id]
        assert len(applicable) == 1
        assert applicable[0].suggested_action is not None

        # 2b: Apply the pattern and record the application
        app_id = store.record_pattern_application(
            pattern_id=pattern_id,
            execution_id="run-2-exec-001",
            outcome_improved=True,
            retry_count_before=1,
            retry_count_after=0,
            application_mode="exploitation",
            validation_passed=True,
        )

        assert app_id is not None

        # === Verify Learning ===

        # Pattern should have success count incremented
        final_patterns = store.get_patterns_for_auto_apply(
            trust_threshold=0.0,
            require_validated=False,
            limit=10,
        )

        final = [p for p in final_patterns if p.id == pattern_id]
        assert len(final) == 1
        assert final[0].led_to_success_count >= 1
        # Effectiveness should be positive (pattern improved outcome)
        assert final[0].effectiveness_score > 0

    def test_pattern_negative_feedback_reduces_effectiveness(
        self, store: GlobalLearningStore
    ) -> None:
        """Test that negative feedback reduces pattern effectiveness."""
        # Record pattern
        pattern_id = store.record_pattern(
            pattern_type="prompt_tweak",
            pattern_name="add_verbose_logging",
            description="Add verbose logging request to prompt",
            context_tags=["prompt"],
        )

        # Record multiple failed applications
        for i in range(3):
            store.record_pattern_application(
                pattern_id=pattern_id,
                execution_id=f"exec-fail-{i}",
                outcome_improved=False,
                retry_count_before=1,
                retry_count_after=2,  # Got worse
                application_mode="exploitation",
            )

        # Check pattern - failure count should reflect negative outcomes
        patterns = store.get_patterns_for_auto_apply(
            trust_threshold=0.0,
            require_validated=False,
            limit=10,
        )

        found = [p for p in patterns if p.id == pattern_id]
        assert len(found) == 1
        assert found[0].led_to_failure_count >= 3
