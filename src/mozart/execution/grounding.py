"""External grounding hooks for external validation of sheet outputs.

Provides a mechanism for validating sheet outputs against external sources
(APIs, databases, file checksums, etc.) to prevent model drift and ensure
output quality beyond internal validation.

This addresses the mathematical necessity of external validators documented
in arXiv 2601.05280 (entropy decay in self-training) and DGM objective hacking.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from mozart.core.checkpoint import ValidationDetailDict


class GroundingPhase(str, Enum):
    """When the grounding hook should execute relative to validation."""

    PRE_VALIDATION = "pre_validation"
    """Run before internal validation engine."""

    POST_VALIDATION = "post_validation"
    """Run after internal validation engine."""

    BOTH = "both"
    """Run both before and after internal validation."""


@dataclass
class GroundingContext:
    """Context provided to grounding hooks for validation.

    Contains all relevant information about the sheet execution that
    the hook can use to perform external validation.
    """

    job_id: str
    """Unique identifier for the job."""

    sheet_num: int
    """Sheet number being validated."""

    prompt: str
    """The prompt that was used for sheet execution."""

    output: str
    """The raw output from sheet execution."""

    output_files: list[str] = field(default_factory=list)
    """List of file paths created/modified by the sheet."""

    validation_passed: bool | None = None
    """Result of internal validation (None if pre_validation phase)."""

    validation_details: list[ValidationDetailDict] = field(default_factory=list)
    """Detailed validation results from internal engine."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional context metadata (e.g., config values, timing)."""


@dataclass
class GroundingResult:
    """Result from a grounding hook execution.

    Contains the validation outcome and optional guidance for recovery.
    """

    passed: bool
    """Whether the grounding validation passed."""

    hook_name: str
    """Name/identifier of the hook that produced this result."""

    message: str = ""
    """Human-readable description of the result."""

    confidence: float = 1.0
    """Confidence in the grounding result (0.0-1.0)."""

    details: dict[str, Any] = field(default_factory=dict)
    """Additional details about the validation."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    """When the grounding check was performed."""

    recovery_guidance: str | None = None
    """Optional guidance for what to do on failure."""

    should_escalate: bool = False
    """Whether this failure should trigger escalation (if available)."""


@runtime_checkable
class GroundingHook(Protocol):
    """Protocol for external grounding hooks.

    Implementations can check outputs against external sources like:
    - API endpoints for data freshness
    - File checksums for artifact integrity
    - External validators for output quality
    - Database queries for data consistency
    """

    @property
    def name(self) -> str:
        """Unique name for this grounding hook."""
        ...

    @property
    def phase(self) -> GroundingPhase:
        """When this hook should run relative to validation."""
        ...

    async def validate(self, context: GroundingContext) -> GroundingResult:
        """Perform external validation on the sheet output.

        Args:
            context: Full context about the sheet execution.

        Returns:
            GroundingResult with pass/fail status and details.
        """
        ...


# Import GroundingConfig and GroundingHookConfig from core.config to avoid duplication
# Type-only import to avoid circular dependency at runtime
if TYPE_CHECKING:
    from mozart.core.config import GroundingConfig, GroundingHookConfig


class FileChecksumGroundingHook:
    """Example grounding hook that validates file checksums.

    Checks that output files have expected checksums, preventing
    model from overwriting important files incorrectly.
    """

    def __init__(
        self,
        expected_checksums: dict[str, str] | None = None,
        checksum_algorithm: Literal["md5", "sha256"] = "sha256",
        name: str | None = None,
    ) -> None:
        """Initialize the file checksum hook.

        Args:
            expected_checksums: Map of file path to expected checksum.
            checksum_algorithm: Algorithm to use for checksums.
            name: Custom name for this hook (defaults to "file_checksum").
        """
        self._expected_checksums = expected_checksums or {}
        self._algorithm = checksum_algorithm
        self._name = name or "file_checksum"

    @property
    def name(self) -> str:
        return self._name

    @property
    def phase(self) -> GroundingPhase:
        return GroundingPhase.POST_VALIDATION

    async def validate(self, context: GroundingContext) -> GroundingResult:
        """Validate file checksums match expected values."""
        import hashlib
        from pathlib import Path

        if not self._expected_checksums:
            return GroundingResult(
                passed=True,
                hook_name=self.name,
                message="No checksums configured",
            )

        mismatches: list[str] = []
        checked = 0

        for file_path, expected_hash in self._expected_checksums.items():
            path = Path(file_path)
            if not path.exists():
                mismatches.append(f"{file_path}: file not found")
                continue

            hasher = (
                hashlib.md5() if self._algorithm == "md5" else hashlib.sha256()
            )
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)

            actual_hash = hasher.hexdigest()
            if actual_hash != expected_hash:
                mismatches.append(
                    f"{file_path}: expected {expected_hash[:16]}..., "
                    f"got {actual_hash[:16]}..."
                )
            checked += 1

        if mismatches:
            return GroundingResult(
                passed=False,
                hook_name=self.name,
                message=f"Checksum validation failed for {len(mismatches)} file(s)",
                details={"mismatches": mismatches, "files_checked": checked},
                recovery_guidance="Re-generate the affected files or verify expected checksums",
            )

        return GroundingResult(
            passed=True,
            hook_name=self.name,
            message=f"All {checked} file checksum(s) validated",
            details={"files_checked": checked},
        )


class GroundingEngine:
    """Engine for executing grounding hooks.

    Coordinates multiple hooks and aggregates their results.
    """

    def __init__(
        self,
        hooks: list[GroundingHook] | None = None,
        config: "GroundingConfig | None" = None,
    ) -> None:
        """Initialize the grounding engine.

        Args:
            hooks: List of grounding hooks to execute.
            config: Configuration for grounding behavior (from core.config).
        """
        self._hooks = hooks or []
        # Default to a basic config if none provided
        if config is None:
            from mozart.core.config import GroundingConfig as GC
            self._config = GC()
        else:
            self._config = config

    def add_hook(self, hook: GroundingHook) -> None:
        """Add a grounding hook to the engine."""
        self._hooks.append(hook)

    def get_hook_count(self) -> int:
        """Get the number of registered hooks."""
        return len(self._hooks)

    # Circuit breaker: abort after this many consecutive hook failures
    CIRCUIT_BREAKER_THRESHOLD = 3

    async def run_hooks(
        self,
        context: GroundingContext,
        phase: GroundingPhase,
    ) -> list[GroundingResult]:
        """Run all hooks matching the specified phase.

        Includes a circuit breaker: after CIRCUIT_BREAKER_THRESHOLD consecutive
        failures (timeout or exception), remaining hooks are skipped to avoid
        wasting O(hooks x timeout) when an external service is down.

        Args:
            context: Context for grounding validation.
            phase: Which phase to run hooks for.

        Returns:
            List of GroundingResult from all matching hooks.
        """
        import asyncio

        results: list[GroundingResult] = []
        consecutive_failures = 0

        for hook in self._hooks:
            # Check if hook should run in this phase
            if hook.phase not in (phase, GroundingPhase.BOTH):
                continue

            # Circuit breaker: skip remaining hooks after too many failures
            if consecutive_failures >= self.CIRCUIT_BREAKER_THRESHOLD:
                results.append(
                    GroundingResult(
                        passed=False,
                        hook_name=hook.name,
                        message=(
                            f"Skipped: circuit breaker open after "
                            f"{consecutive_failures} consecutive failures"
                        ),
                        should_escalate=self._config.escalate_on_failure,
                    )
                )
                continue

            try:
                # Run hook with timeout
                result = await asyncio.wait_for(
                    hook.validate(context),
                    timeout=self._config.timeout_seconds,
                )
                results.append(result)
                if result.passed:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
            except asyncio.TimeoutError:
                consecutive_failures += 1
                results.append(
                    GroundingResult(
                        passed=False,
                        hook_name=hook.name,
                        message=f"Hook timed out after {self._config.timeout_seconds}s",
                        should_escalate=self._config.escalate_on_failure,
                    )
                )
            except Exception as e:
                consecutive_failures += 1
                results.append(
                    GroundingResult(
                        passed=False,
                        hook_name=hook.name,
                        message=f"Hook error: {e!s}",
                        details={"error_type": type(e).__name__},
                        should_escalate=self._config.escalate_on_failure,
                    )
                )

        return results

    def aggregate_results(
        self,
        results: list[GroundingResult],
    ) -> tuple[bool, str]:
        """Aggregate multiple grounding results into overall status.

        Args:
            results: List of grounding results to aggregate.

        Returns:
            Tuple of (overall_passed, summary_message).
        """
        if not results:
            return True, "No grounding hooks executed"

        passed = all(r.passed for r in results)
        failed = [r for r in results if not r.passed]

        if passed:
            return True, f"All {len(results)} grounding check(s) passed"

        # Build failure summary
        failures = ", ".join(f"{r.hook_name}: {r.message}" for r in failed)
        return False, f"{len(failed)}/{len(results)} grounding check(s) failed: {failures}"


def create_hook_from_config(
    hook_config: "GroundingHookConfig",
) -> GroundingHook:
    """Factory function to create a grounding hook from configuration.

    This is the integration point for hook registration. The factory reads
    hook configuration from YAML and instantiates the appropriate hook class.

    Args:
        hook_config: Configuration for the hook (from GroundingConfig.hooks).

    Returns:
        An instantiated GroundingHook ready for registration.

    Raises:
        ValueError: If hook type is unknown.

    Example:
        from mozart.core.config import GroundingHookConfig
        config = GroundingHookConfig(
            type="file_checksum",
            expected_checksums={"output.txt": "abc123..."},
        )
        hook = create_hook_from_config(config)
        grounding_engine.add_hook(hook)
    """
    # Import here to avoid TYPE_CHECKING import issues
    from mozart.core.config import GroundingHookConfig as GHC

    if not isinstance(hook_config, GHC):
        raise TypeError(f"Expected GroundingHookConfig, got {type(hook_config)}")

    if hook_config.type == "file_checksum":
        return FileChecksumGroundingHook(
            expected_checksums=hook_config.expected_checksums,
            checksum_algorithm=hook_config.checksum_algorithm,
            name=hook_config.name,
        )

    # Future hook types can be added here:
    # elif hook_config.type == "api_validator":
    #     return ApiValidatorGroundingHook(...)

    raise ValueError(f"Unknown grounding hook type: {hook_config.type}")
