"""Tests for rate limit intelligence — duration parsing, per-model hold, fallback.

Adversarial TDD: tests written before implementation. Red first, then green.
"""

from __future__ import annotations

import time

from marianne.core.errors import ErrorClassifier
from marianne.daemon.baton.core import BatonCore
from marianne.daemon.baton.events import RateLimitHit
from marianne.daemon.baton.state import BatonSheetStatus, SheetExecutionState

# =============================================================================
# Task 1: Duration Parsing
# =============================================================================


class TestRateLimitDurationParsing:
    """Extract actual wait durations from rate limit error text."""

    def test_anthropic_retry_after_seconds(self) -> None:
        """Parse 'Please retry after X seconds' — clamped to minimum."""
        classifier = ErrorClassifier()
        duration = classifier.extract_rate_limit_wait(
            "Rate limit exceeded. Please retry after 120 seconds."
        )
        # 120 < RESET_TIME_MINIMUM_WAIT_SECONDS (300), so clamped
        assert duration == 300.0

    def test_anthropic_retry_after_minutes(self) -> None:
        """Parse 'retry after N minutes' above clamp threshold."""
        classifier = ErrorClassifier()
        duration = classifier.extract_rate_limit_wait(
            "Rate limit exceeded. Please retry after 10 minutes."
        )
        assert duration == 600.0

    def test_claude_code_rate_limit_minutes(self) -> None:
        """Parse 'try again in N minutes' from Claude Code output."""
        classifier = ErrorClassifier()
        duration = classifier.extract_rate_limit_wait(
            "You've hit a rate limit. Please try again in 5 minutes."
        )
        assert duration == 300.0

    def test_quota_exhaustion_hours(self) -> None:
        """Parse quota reset duration."""
        classifier = ErrorClassifier()
        duration = classifier.extract_rate_limit_wait(
            "Daily token quota exhausted. Resets in 3 hours."
        )
        assert duration == 10800.0

    def test_no_duration_returns_none(self) -> None:
        """When no parseable duration, return None (caller uses default)."""
        classifier = ErrorClassifier()
        duration = classifier.extract_rate_limit_wait("Rate limit exceeded.")
        assert duration is None

    def test_retry_after_header_value(self) -> None:
        """Parse raw Retry-After header value (integer seconds)."""
        classifier = ErrorClassifier()
        duration = classifier.extract_rate_limit_wait("Retry-After: 600")
        assert duration == 600.0

    def test_empty_string_returns_none(self) -> None:
        """Empty input returns None."""
        classifier = ErrorClassifier()
        assert classifier.extract_rate_limit_wait("") is None

    def test_wait_600_seconds(self) -> None:
        """Parse 'wait N seconds' variant above clamp."""
        classifier = ErrorClassifier()
        duration = classifier.extract_rate_limit_wait("Please wait 600 seconds before retrying.")
        assert duration == 600.0

    def test_please_try_again_in_10_minutes(self) -> None:
        """Parse 'try again in N minutes' above clamp."""
        classifier = ErrorClassifier()
        duration = classifier.extract_rate_limit_wait("try again in 10 minutes")
        assert duration == 600.0

    def test_resets_in_1_hour(self) -> None:
        """Parse 'resets in 1 hour' (singular)."""
        classifier = ErrorClassifier()
        duration = classifier.extract_rate_limit_wait("Usage limit reached. Resets in 1 hour.")
        assert duration == 3600.0


# =============================================================================
# Task 2: Per-Model Rate Limit Hold
# =============================================================================


class TestPerModelRateLimitHold:
    """Rate limits should hold per-model, not per-instrument."""

    def test_rate_limit_holds_only_matching_model(self) -> None:
        """When claude-sonnet is rate limited, claude-opus sheets continue."""

        baton = BatonCore()
        sheets = {
            1: SheetExecutionState(
                sheet_num=1,
                instrument_name="claude-code",
                model="claude-sonnet-4-5",
                max_retries=1,
            ),
            2: SheetExecutionState(
                sheet_num=2,
                instrument_name="claude-code",
                model="claude-opus-4-6",
                max_retries=1,
            ),
        }
        baton.register_job("j1", sheets, {})

        # Dispatch both
        sheets[1].status = BatonSheetStatus.DISPATCHED
        sheets[2].status = BatonSheetStatus.DISPATCHED

        # Rate limit hits claude-sonnet only
        baton._handle_rate_limit_hit(
            RateLimitHit(
                instrument="claude-code",
                model="claude-sonnet-4-5",
                wait_seconds=120.0,
                job_id="j1",
                sheet_num=1,
            )
        )

        # Sheet 1 (sonnet) should be WAITING
        assert sheets[1].status == BatonSheetStatus.WAITING
        # Sheet 2 (opus) should still be DISPATCHED
        assert sheets[2].status == BatonSheetStatus.DISPATCHED

    def test_rate_limit_without_model_holds_all_on_instrument(self) -> None:
        """When no model specified, all sheets on instrument wait (backward compat)."""

        baton = BatonCore()
        sheets = {
            1: SheetExecutionState(
                sheet_num=1,
                instrument_name="claude-code",
                model="claude-sonnet-4-5",
                max_retries=1,
            ),
            2: SheetExecutionState(
                sheet_num=2,
                instrument_name="claude-code",
                model="claude-opus-4-6",
                max_retries=1,
            ),
        }
        baton.register_job("j1", sheets, {})
        sheets[1].status = BatonSheetStatus.DISPATCHED
        sheets[2].status = BatonSheetStatus.DISPATCHED

        # No model specified — old behavior: all sheets on instrument wait
        baton._handle_rate_limit_hit(
            RateLimitHit(
                instrument="claude-code",
                wait_seconds=60.0,
                job_id="j1",
                sheet_num=1,
            )
        )

        assert sheets[1].status == BatonSheetStatus.WAITING
        assert sheets[2].status == BatonSheetStatus.WAITING


# =============================================================================
# Task 4: Fallback on Rate Limit
# =============================================================================


class TestFallbackOnRateLimit:
    """Sheets with fallback instruments should try the fallback instead of waiting."""

    def test_sheet_with_fallback_advances_on_rate_limit(self) -> None:
        """When rate limited, a sheet with fallback instruments advances."""

        baton = BatonCore()
        sheets = {
            1: SheetExecutionState(
                sheet_num=1,
                instrument_name="claude-code",
                model="claude-sonnet-4-5",
                max_retries=1,
                fallback_chain=["gemini-cli"],
            ),
        }
        baton.register_job("j1", sheets, {})
        sheets[1].status = BatonSheetStatus.DISPATCHED

        baton._handle_rate_limit_hit(
            RateLimitHit(
                instrument="claude-code",
                model="claude-sonnet-4-5",
                wait_seconds=3600.0,
                job_id="j1",
                sheet_num=1,
            )
        )

        # Sheet should advance to fallback, not wait
        assert sheets[1].instrument_name == "gemini-cli"
        assert sheets[1].status == BatonSheetStatus.PENDING

    def test_sheet_without_fallback_waits(self) -> None:
        """When rate limited with no fallback, sheet waits for timer."""

        baton = BatonCore()
        sheets = {
            1: SheetExecutionState(
                sheet_num=1,
                instrument_name="claude-code",
                model="claude-sonnet-4-5",
                max_retries=1,
            ),
        }
        baton.register_job("j1", sheets, {})
        sheets[1].status = BatonSheetStatus.DISPATCHED

        baton._handle_rate_limit_hit(
            RateLimitHit(
                instrument="claude-code",
                model="claude-sonnet-4-5",
                wait_seconds=120.0,
                job_id="j1",
                sheet_num=1,
            )
        )

        # No fallback — sheet waits
        assert sheets[1].status == BatonSheetStatus.WAITING


# =============================================================================
# Task 5: Parsed Duration End-to-End
# =============================================================================


class TestParsedDurationEndToEnd:
    """Parsed wait duration flows from backend through to baton timer."""

    def test_parsed_duration_used_for_timer(self) -> None:
        """RateLimitHit with parsed wait_seconds schedules timer for that duration."""

        baton = BatonCore()
        sheets = {1: SheetExecutionState(sheet_num=1, instrument_name="claude-code", max_retries=1)}
        baton.register_job("j1", sheets, {})
        sheets[1].status = BatonSheetStatus.DISPATCHED

        # Rate limit with parsed 300s wait (not default 60s)
        baton._handle_rate_limit_hit(
            RateLimitHit(
                instrument="claude-code",
                wait_seconds=300.0,
                job_id="j1",
                sheet_num=1,
            )
        )

        # Verify the instrument's rate_limit_expires_at uses the parsed duration
        inst = baton._instruments.get("claude-code")
        assert inst is not None
        assert inst.rate_limited is True
        remaining = inst.rate_limit_expires_at - time.monotonic()
        assert remaining > 250.0, f"Expected ~300s remaining, got {remaining:.0f}s"
