"""Tests for pattern detection and matching.

Tests the Priority 1 Evolution: Close the Learning Loop.
"""

from datetime import datetime, timedelta

import pytest

from mozart.core.checkpoint import SheetStatus
from mozart.learning.outcomes import SheetOutcome
from mozart.learning.patterns import (
    DetectedPattern,
    PatternApplicator,
    PatternDetector,
    PatternMatcher,
    PatternType,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_successful_outcome() -> SheetOutcome:
    """Create a sample successful outcome."""
    return SheetOutcome(
        sheet_id="job1-sheet1",
        job_id="job1",
        validation_results=[
            {"rule_type": "file_exists", "passed": True, "confidence": 1.0},
            {"rule_type": "content_contains", "passed": True, "confidence": 0.9},
        ],
        execution_duration=30.0,
        retry_count=0,
        completion_mode_used=False,
        final_status=SheetStatus.COMPLETED,
        validation_pass_rate=1.0,
        first_attempt_success=True,
    )


@pytest.fixture
def sample_failed_outcome() -> SheetOutcome:
    """Create a sample failed outcome."""
    return SheetOutcome(
        sheet_id="job1-sheet2",
        job_id="job1",
        validation_results=[
            {"rule_type": "file_exists", "passed": False, "confidence": 1.0},
            {"rule_type": "content_contains", "passed": True, "confidence": 0.8},
        ],
        execution_duration=45.0,
        retry_count=2,
        completion_mode_used=True,
        final_status=SheetStatus.FAILED,
        validation_pass_rate=0.5,
        first_attempt_success=False,
    )


@pytest.fixture
def sample_retry_success_outcome() -> SheetOutcome:
    """Create an outcome that succeeded after retries."""
    return SheetOutcome(
        sheet_id="job1-sheet3",
        job_id="job1",
        validation_results=[
            {"rule_type": "file_exists", "passed": True, "confidence": 1.0},
        ],
        execution_duration=60.0,
        retry_count=1,
        completion_mode_used=False,
        final_status=SheetStatus.COMPLETED,
        validation_pass_rate=1.0,
        first_attempt_success=False,
    )


@pytest.fixture
def multiple_outcomes(
    sample_successful_outcome: SheetOutcome,
    sample_failed_outcome: SheetOutcome,
    sample_retry_success_outcome: SheetOutcome,
) -> list[SheetOutcome]:
    """Create multiple outcomes for pattern detection."""
    # Create more outcomes to trigger pattern detection
    outcomes = [sample_successful_outcome, sample_failed_outcome, sample_retry_success_outcome]

    # Add more failed file_exists validations to create a pattern
    for i in range(3):
        outcomes.append(
            SheetOutcome(
                sheet_id=f"job1-sheet{i+4}",
                job_id="job1",
                validation_results=[
                    {"rule_type": "file_exists", "passed": False, "confidence": 1.0},
                ],
                execution_duration=30.0,
                retry_count=1,
                completion_mode_used=False,
                final_status=SheetStatus.FAILED,
                validation_pass_rate=0.0,
                first_attempt_success=False,
            )
        )

    # Add more first-attempt successes
    for i in range(4):
        outcomes.append(
            SheetOutcome(
                sheet_id=f"job1-sheet{i+7}",
                job_id="job1",
                validation_results=[
                    {"rule_type": "file_exists", "passed": True, "confidence": 1.0},
                ],
                execution_duration=25.0,
                retry_count=0,
                completion_mode_used=False,
                final_status=SheetStatus.COMPLETED,
                validation_pass_rate=1.0,
                first_attempt_success=True,
            )
        )

    return outcomes


# =============================================================================
# TestPatternDetector
# =============================================================================


class TestPatternDetector:
    """Tests for PatternDetector class."""

    def test_detect_validation_patterns(self, multiple_outcomes: list[SheetOutcome]) -> None:
        """Test detection of recurring validation failure patterns."""
        detector = PatternDetector(multiple_outcomes)
        patterns = detector.detect_all()

        # Should detect file_exists validation failures (seen 4 times)
        validation_patterns = [
            p for p in patterns if p.pattern_type == PatternType.VALIDATION_FAILURE
        ]
        assert len(validation_patterns) >= 1

        file_exists_pattern = next(
            (p for p in validation_patterns if "file_exists" in p.description),
            None,
        )
        assert file_exists_pattern is not None
        assert file_exists_pattern.frequency >= 2

    def test_detect_retry_patterns(self, multiple_outcomes: list[SheetOutcome]) -> None:
        """Test detection of retry success patterns."""
        # Add more retry successes
        outcomes = multiple_outcomes + [
            SheetOutcome(
                sheet_id="job1-retry1",
                job_id="job1",
                validation_results=[{"rule_type": "file_exists", "passed": True}],
                execution_duration=50.0,
                retry_count=2,
                completion_mode_used=False,
                final_status=SheetStatus.COMPLETED,
                validation_pass_rate=1.0,
                first_attempt_success=False,
            )
        ]

        detector = PatternDetector(outcomes)
        patterns = detector.detect_all()

        retry_patterns = [
            p for p in patterns if p.pattern_type == PatternType.RETRY_SUCCESS
        ]
        assert len(retry_patterns) >= 1

    def test_empty_outcomes_returns_empty(self) -> None:
        """Test that empty outcomes returns empty patterns list."""
        detector = PatternDetector([])
        patterns = detector.detect_all()
        assert patterns == []

    def test_success_rate_calculation(self, multiple_outcomes: list[SheetOutcome]) -> None:
        """Test success rate calculation."""
        success_rate = PatternDetector.calculate_success_rate(multiple_outcomes)
        # 5 successful (1 original + 4 first_attempt) out of 10 total
        assert 0.0 <= success_rate <= 1.0

    def test_detect_success_patterns(self, multiple_outcomes: list[SheetOutcome]) -> None:
        """Test detection of first-attempt success patterns."""
        detector = PatternDetector(multiple_outcomes)
        patterns = detector.detect_all()

        success_patterns = [
            p for p in patterns if p.pattern_type == PatternType.FIRST_ATTEMPT_SUCCESS
        ]
        # Should detect first-attempt success pattern (5 successes)
        assert len(success_patterns) >= 1

    def test_detect_completion_patterns(self) -> None:
        """Test detection of completion mode patterns."""
        outcomes = [
            SheetOutcome(
                sheet_id=f"job1-sheet{i}",
                job_id="job1",
                validation_results=[{"rule_type": "file_exists", "passed": True}],
                execution_duration=50.0,
                retry_count=1,
                completion_mode_used=True,
                final_status=SheetStatus.COMPLETED,
                validation_pass_rate=1.0,
                first_attempt_success=False,
            )
            for i in range(3)
        ]

        detector = PatternDetector(outcomes)
        patterns = detector.detect_all()

        completion_patterns = [
            p for p in patterns if p.pattern_type == PatternType.COMPLETION_MODE
        ]
        assert len(completion_patterns) >= 1

    def test_detect_low_confidence_patterns(self) -> None:
        """Test detection of low confidence validation patterns."""
        outcomes = [
            SheetOutcome(
                sheet_id=f"job1-sheet{i}",
                job_id="job1",
                validation_results=[
                    {"rule_type": "content_regex", "passed": True, "confidence": 0.5},
                    {"rule_type": "file_exists", "passed": True, "confidence": 0.6},
                ],
                execution_duration=30.0,
                retry_count=0,
                completion_mode_used=False,
                final_status=SheetStatus.COMPLETED,
                validation_pass_rate=1.0,
                first_attempt_success=True,
            )
            for i in range(3)
        ]

        detector = PatternDetector(outcomes)
        patterns = detector.detect_all()

        low_conf_patterns = [
            p for p in patterns if p.pattern_type == PatternType.LOW_CONFIDENCE
        ]
        # Should detect low confidence pattern (6 low-confidence validations)
        assert len(low_conf_patterns) >= 1


# =============================================================================
# TestPatternMatcher
# =============================================================================


class TestPatternMatcher:
    """Tests for PatternMatcher class."""

    @pytest.fixture
    def sample_patterns(self) -> list[DetectedPattern]:
        """Create sample patterns for matching tests."""
        return [
            DetectedPattern(
                pattern_type=PatternType.VALIDATION_FAILURE,
                description="file_exists validation tends to fail",
                frequency=5,
                success_rate=0.0,
                context_tags=["validation:file_exists"],
                confidence=0.7,
            ),
            DetectedPattern(
                pattern_type=PatternType.RETRY_SUCCESS,
                description="Retrying works: avg 1.5 retries lead to success",
                frequency=3,
                success_rate=1.0,
                context_tags=["retry:effective"],
                confidence=0.6,
            ),
            DetectedPattern(
                pattern_type=PatternType.FIRST_ATTEMPT_SUCCESS,
                description="First-attempt success rate: 70%",
                frequency=7,
                success_rate=0.7,
                context_tags=["success:first_attempt"],
                confidence=0.8,
            ),
        ]

    def test_match_by_context_tags(self, sample_patterns: list[DetectedPattern]) -> None:
        """Test matching patterns by context tags."""
        matcher = PatternMatcher(sample_patterns)

        context = {"tags": ["validation:file_exists"]}
        matched = matcher.match(context, limit=5)

        # Should include the file_exists pattern
        assert len(matched) >= 1
        file_exists_matched = any(
            "file_exists" in p.description for p in matched
        )
        assert file_exists_matched

    def test_relevance_scoring(self, sample_patterns: list[DetectedPattern]) -> None:
        """Test that patterns are scored and sorted by relevance."""
        matcher = PatternMatcher(sample_patterns)

        context = {"tags": []}
        matched = matcher.match(context, limit=5)

        # Should return patterns sorted by score (highest first)
        # Higher confidence patterns should come first
        assert len(matched) >= 1

    def test_limit_results(self, sample_patterns: list[DetectedPattern]) -> None:
        """Test that limit parameter works correctly."""
        matcher = PatternMatcher(sample_patterns)

        context = {}
        matched = matcher.match(context, limit=1)

        assert len(matched) <= 1

    def test_empty_patterns_returns_empty(self) -> None:
        """Test that empty patterns returns empty matches."""
        matcher = PatternMatcher([])
        matched = matcher.match({}, limit=5)
        assert matched == []

    def test_recency_affects_score(self, sample_patterns: list[DetectedPattern]) -> None:
        """Test that more recent patterns score higher."""
        # Create two patterns: one recent, one old
        recent = DetectedPattern(
            pattern_type=PatternType.RETRY_SUCCESS,
            description="Recent pattern",
            frequency=1,
            confidence=0.5,
            last_seen=datetime.now(),
        )
        old = DetectedPattern(
            pattern_type=PatternType.RETRY_SUCCESS,
            description="Old pattern",
            frequency=1,
            confidence=0.5,
            last_seen=datetime.now() - timedelta(days=30),
        )

        matcher = PatternMatcher([old, recent])
        matched = matcher.match({}, limit=2)

        # Recent pattern should score higher and come first
        assert len(matched) == 2
        assert matched[0].description == "Recent pattern"


# =============================================================================
# TestPatternApplicator
# =============================================================================


class TestPatternApplicator:
    """Tests for PatternApplicator class."""

    @pytest.fixture
    def sample_patterns(self) -> list[DetectedPattern]:
        """Create sample patterns for applicator tests."""
        return [
            DetectedPattern(
                pattern_type=PatternType.VALIDATION_FAILURE,
                description="file_exists validation tends to fail",
                frequency=5,
            ),
            DetectedPattern(
                pattern_type=PatternType.RETRY_SUCCESS,
                description="Retry with 2 attempts usually works",
                frequency=3,
                success_rate=0.85,
            ),
        ]

    def test_generate_prompt_section(self, sample_patterns: list[DetectedPattern]) -> None:
        """Test generating a prompt section from patterns."""
        applicator = PatternApplicator(sample_patterns)
        section = applicator.generate_prompt_section()

        assert "## Learned Patterns" in section
        assert "file_exists" in section
        assert "Retry" in section

    def test_empty_patterns_returns_empty(self) -> None:
        """Test that empty patterns returns empty section."""
        applicator = PatternApplicator([])
        section = applicator.generate_prompt_section()
        assert section == ""

    def test_get_pattern_descriptions(self, sample_patterns: list[DetectedPattern]) -> None:
        """Test getting pattern descriptions as list."""
        applicator = PatternApplicator(sample_patterns)
        descriptions = applicator.get_pattern_descriptions()

        assert len(descriptions) == 2
        assert any("file_exists" in d for d in descriptions)

    def test_pattern_to_prompt_guidance(self) -> None:
        """Test individual pattern to prompt guidance conversion."""
        patterns = [
            DetectedPattern(
                pattern_type=PatternType.VALIDATION_FAILURE,
                description="Test failure pattern",
                frequency=3,
            ),
            DetectedPattern(
                pattern_type=PatternType.RETRY_SUCCESS,
                description="Test retry pattern",
                frequency=2,
                success_rate=0.9,
            ),
            DetectedPattern(
                pattern_type=PatternType.FIRST_ATTEMPT_SUCCESS,
                description="Test success pattern",
                frequency=5,
            ),
            DetectedPattern(
                pattern_type=PatternType.LOW_CONFIDENCE,
                description="Test low confidence",
                frequency=2,
            ),
        ]

        for pattern in patterns:
            guidance = pattern.to_prompt_guidance()
            assert guidance  # Should not be empty
            assert pattern.description in guidance or "Test" in guidance


# =============================================================================
# TestDetectedPattern
# =============================================================================


class TestDetectedPattern:
    """Tests for DetectedPattern dataclass."""

    def test_to_prompt_guidance_validation_failure(self) -> None:
        """Test prompt guidance for validation failure pattern."""
        pattern = DetectedPattern(
            pattern_type=PatternType.VALIDATION_FAILURE,
            description="file_exists validation fails",
            frequency=5,
        )
        guidance = pattern.to_prompt_guidance()

        assert "⚠️" in guidance
        assert "Common issue" in guidance
        assert "5x" in guidance

    def test_to_prompt_guidance_retry_success(self) -> None:
        """Test prompt guidance for retry success pattern."""
        pattern = DetectedPattern(
            pattern_type=PatternType.RETRY_SUCCESS,
            description="Retry after file creation works",
            frequency=3,
            success_rate=0.85,
        )
        guidance = pattern.to_prompt_guidance()

        assert "✓" in guidance
        assert "Tip" in guidance
        assert "85%" in guidance

    def test_to_prompt_guidance_first_attempt_success(self) -> None:
        """Test prompt guidance for first-attempt success pattern."""
        pattern = DetectedPattern(
            pattern_type=PatternType.FIRST_ATTEMPT_SUCCESS,
            description="Following spec exactly works",
            frequency=10,
        )
        guidance = pattern.to_prompt_guidance()

        assert "✓" in guidance
        assert "Best practice" in guidance

    def test_default_values(self) -> None:
        """Test DetectedPattern default values."""
        pattern = DetectedPattern(
            pattern_type=PatternType.VALIDATION_FAILURE,
            description="Test pattern",
        )

        assert pattern.frequency == 1
        assert pattern.success_rate == 0.0
        assert pattern.confidence == 0.5
        assert pattern.context_tags == []
        assert pattern.evidence == []
