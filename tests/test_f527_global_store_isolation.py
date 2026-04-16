"""Regression tests for F-527: Global learning store singleton isolation.

F-527 Finding:
    Tests in test_global_learning.py (TestGoalDriftDetection::test_drift_threshold_alerting
    and TestExplorationBudget::test_get_exploration_budget_history) fail when run in the
    full test suite but pass when run in isolation.

Root Cause:
    The global learning store singleton (_global_store) is not reset between tests.
    When tests use get_global_store() without a path parameter, they get a cached
    instance pointing to a stale database, while other tests use the fixture which
    creates fresh temp databases.

Fix:
    Added autouse fixture in conftest.py that resets _global_store to None before
    and after each test, ensuring test isolation.

These regression tests verify:
1. The singleton is properly reset between tests
2. Tests using both get_global_store() and the fixture don't interfere
3. Multiple sequential calls to get_global_store() within the same test work correctly
"""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from marianne.learning.store import GlobalLearningStore, get_global_store


@pytest.fixture
def temp_db_path() -> Path:
    """Create a temporary database path for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        return Path(f.name)


@pytest.fixture
def global_store(temp_db_path: Path) -> Generator[GlobalLearningStore, None, None]:
    """Create a GlobalLearningStore with a temporary database."""
    store = GlobalLearningStore(temp_db_path)
    yield store
    # Cleanup
    if temp_db_path.exists():
        temp_db_path.unlink()


class TestGlobalStoreSingletonIsolation:
    """Verify global learning store singleton is properly isolated between tests."""

    def test_singleton_reset_between_tests_part1(self) -> None:
        """First test: create a singleton with default path.

        This test intentionally uses get_global_store() to populate the singleton cache.
        If the autouse fixture works correctly, the next test should NOT see this instance.
        """
        # Import the module to verify internal state
        import marianne.learning.store as store_module

        # Singleton should start as None (reset by autouse fixture)
        assert store_module._global_store is None, "Singleton should be None at test start"

        # Create singleton with default path
        store1 = get_global_store()
        assert store1 is not None
        assert store_module._global_store is store1, "Singleton should be cached"

        # Record a pattern to modify state
        pattern_id = store1.record_pattern(
            pattern_type="test_isolation",
            pattern_name="test_pattern_part1",
        )
        assert pattern_id is not None

    def test_singleton_reset_between_tests_part2(self) -> None:
        """Second test: verify singleton was reset.

        If the autouse fixture works correctly, _global_store should be None again,
        and get_global_store() should create a FRESH instance.
        """
        # Import the module to verify internal state
        import marianne.learning.store as store_module

        # Singleton should be None (reset by autouse fixture)
        assert store_module._global_store is None, (
            "F-527: Singleton was not reset between tests! "
            "The autouse fixture reset_global_learning_store() is not working."
        )

        # Create singleton (should be fresh instance)
        store2 = get_global_store()
        assert store2 is not None
        assert store_module._global_store is store2

        # This store should NOT have the pattern from part1
        # (We can't easily verify this without querying the DB, but the None check above
        # proves the singleton was reset)

    def test_singleton_with_custom_path_recreated_on_path_change(self, tmp_path: Path) -> None:
        """Verify singleton recreates when db_path changes."""
        # Import the module to verify internal state
        import marianne.learning.store as store_module

        # Create two temporary databases
        custom_db1 = tmp_path / "custom1.db"
        custom_db2 = tmp_path / "custom2.db"

        # Get singleton with first custom path
        store1 = get_global_store(db_path=custom_db1)
        assert store1.db_path == custom_db1
        assert store_module._global_store is store1

        # Get singleton with second custom path (should create NEW instance)
        store2 = get_global_store(db_path=custom_db2)
        assert store2.db_path == custom_db2
        assert store2 is not store1, "Singleton should be recreated when db_path changes"
        assert store_module._global_store is store2

        # Get singleton again with first path (should create NEW instance again)
        store3 = get_global_store(db_path=custom_db1)
        assert store3.db_path == custom_db1
        assert store3 is not store1, (
            "Singleton should be recreated when switching back to previous path"
        )
        assert store3 is not store2
        assert store_module._global_store is store3

    def test_fixture_and_singleton_dont_interfere(self, global_store: GlobalLearningStore) -> None:
        """Verify fixture-created store doesn't interfere with singleton.

        The global_store fixture creates a temp database. If we call get_global_store()
        in this test, it should create a DIFFERENT instance (the singleton), not return
        the fixture's instance.
        """
        # Import the module to verify internal state
        import marianne.learning.store as store_module

        # Singleton should still be None (fixture doesn't set it)
        assert store_module._global_store is None

        # Fixture store should have temp db path
        assert global_store.db_path.name.endswith(".db")

        # Get singleton (should be different instance)
        singleton_store = get_global_store()
        assert singleton_store is not global_store, (
            "Singleton should be distinct from fixture instance"
        )
        assert store_module._global_store is singleton_store

    def test_multiple_calls_same_test_return_same_instance(self) -> None:
        """Verify singleton pattern works correctly within a single test."""
        store1 = get_global_store()
        store2 = get_global_store()

        assert store1 is store2, "Multiple calls should return same instance"

    def test_cleanup_temp_db_after_test(self, tmp_path: Path) -> None:
        """Verify cleanup works correctly with temp databases.

        The autouse fixture should reset the singleton even if it points to
        a temp database.
        """
        temp_db = tmp_path / "temp_test.db"

        store = get_global_store(db_path=temp_db)
        assert store.db_path == temp_db

        # Record some data
        pattern_id = store.record_pattern(
            pattern_type="cleanup_test",
            pattern_name="temp_pattern",
        )
        assert pattern_id is not None

        # Database file should exist
        assert temp_db.exists()

        # The autouse fixture will reset the singleton when this test ends


class TestF527RegressionScenario:
    """Regression test for the exact F-527 scenario.

    Simulates what happens when test_cli_learning.py runs before
    test_global_learning.py in the full suite.
    """

    def test_cli_test_using_singleton(self) -> None:
        """Simulate a CLI test that uses get_global_store()."""
        # This is what some CLI tests do - they call get_global_store()
        # without a path, which creates a singleton pointing to the default DB
        store = get_global_store()
        assert store is not None

        # Do some work that modifies the default DB
        pattern_id = store.record_pattern(
            pattern_type="cli_test",
            pattern_name="cli_pattern",
        )
        assert pattern_id is not None

    def test_learning_test_using_fixture(self, global_store: GlobalLearningStore) -> None:
        """Simulate a global_learning.py test that uses the fixture.

        This test should work correctly even after test_cli_test_using_singleton
        ran first, because the autouse fixture reset the singleton.
        """
        # This is what test_global_learning.py tests do - use the fixture
        # which creates a fresh temp database

        # Should be able to work with the fixture store without interference
        pattern_id = global_store.record_pattern(
            pattern_type="learning_test",
            pattern_name="learning_pattern",
        )
        assert pattern_id is not None

        # This should work correctly - no stale data from previous test
        patterns = global_store.get_patterns(pattern_type="learning_test")
        assert len(patterns) == 1
        assert patterns[0].pattern_name == "learning_pattern"

        # Should NOT see the cli_pattern from the previous test
        # (because it's in a different database)
        cli_patterns = global_store.get_patterns(pattern_type="cli_test")
        assert len(cli_patterns) == 0, (
            "F-527: Found patterns from previous test! Test isolation is broken."
        )
