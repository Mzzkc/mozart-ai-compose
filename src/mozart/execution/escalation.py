"""Escalation protocol for low-confidence batch execution decisions.

Provides a mechanism for escalating to external decision-makers (human or AI)
when batch confidence is too low to proceed automatically.

Phase 2 of AGI Evolution: Confidence-Based Execution
"""

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from mozart.core.checkpoint import BatchState
from mozart.execution.validation import BatchValidationResult


@dataclass
class EscalationContext:
    """Context provided to escalation handlers for decision-making.

    Contains all relevant information about the batch execution state
    that led to escalation.
    """

    job_id: str
    """Unique identifier for the job."""

    batch_num: int
    """Batch number that triggered escalation."""

    validation_results: list[dict[str, Any]]
    """Serialized validation results from the batch."""

    confidence: float
    """Aggregate confidence score that triggered escalation (0.0-1.0)."""

    retry_count: int
    """Number of retry attempts already made."""

    error_history: list[str]
    """List of error messages from previous attempts."""

    prompt_used: str
    """The prompt that was used for the batch execution."""

    output_summary: str
    """Summary of the batch execution output."""


@dataclass
class EscalationResponse:
    """Response from an escalation handler specifying how to proceed.

    Determines the next action for a batch that triggered escalation.
    """

    action: Literal["retry", "skip", "abort", "modify_prompt"]
    """Action to take:
    - retry: Retry the batch with the same or modified prompt
    - skip: Skip this batch and continue to the next
    - abort: Stop the entire job
    - modify_prompt: Retry with a modified prompt (requires modified_prompt)
    """

    modified_prompt: str | None = None
    """Modified prompt to use if action is 'modify_prompt'."""

    guidance: str | None = None
    """Optional guidance or reasoning for the decision."""

    confidence_boost: float = 0.0
    """Amount to boost confidence threshold for next attempt (0.0-1.0).
    Allows temporary threshold adjustment based on escalation feedback.
    """


@runtime_checkable
class EscalationHandler(Protocol):
    """Protocol for escalation handlers.

    Implementations can be console-based (human), API-based (AI judgment),
    or any other decision-making mechanism.
    """

    async def should_escalate(
        self,
        batch_state: BatchState,
        validation_result: BatchValidationResult,
        confidence: float,
    ) -> bool:
        """Determine if escalation is needed for this batch.

        Args:
            batch_state: Current state of the batch.
            validation_result: Results from validation engine.
            confidence: Aggregate confidence score.

        Returns:
            True if escalation should be triggered, False to proceed normally.
        """
        ...

    async def escalate(self, context: EscalationContext) -> EscalationResponse:
        """Handle escalation and return the decision.

        Args:
            context: Full context about the escalation trigger.

        Returns:
            EscalationResponse with the action to take.
        """
        ...


class ConsoleEscalationHandler:
    """Console-based escalation handler that prompts the user.

    Implements the EscalationHandler protocol for interactive
    human-in-the-loop decision making.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        auto_retry_on_first_failure: bool = True,
    ) -> None:
        """Initialize the console escalation handler.

        Args:
            confidence_threshold: Confidence below which to escalate (0.0-1.0).
            auto_retry_on_first_failure: If True, auto-retry on first failure
                without escalating (reduces noise for transient issues).
        """
        self.confidence_threshold = confidence_threshold
        self.auto_retry_on_first_failure = auto_retry_on_first_failure

    async def should_escalate(
        self,
        batch_state: BatchState,
        validation_result: BatchValidationResult,
        confidence: float,
    ) -> bool:
        """Determine if escalation is needed.

        Escalates when:
        - Confidence is below threshold
        - AND (not first attempt OR auto_retry_on_first_failure is False)

        Args:
            batch_state: Current batch state.
            validation_result: Validation results.
            confidence: Aggregate confidence score.

        Returns:
            True if user should be prompted for a decision.
        """
        # Don't escalate if confidence is acceptable
        if confidence >= self.confidence_threshold:
            return False

        # Escalate if auto-retry is disabled OR this is not the first attempt
        return not (self.auto_retry_on_first_failure and batch_state.attempt_count <= 1)

    async def escalate(self, context: EscalationContext) -> EscalationResponse:
        """Prompt user for escalation decision via console.

        Args:
            context: Full escalation context.

        Returns:
            EscalationResponse based on user input.
        """
        self._print_context_summary(context)
        return await self._prompt_for_action(context)

    def _print_context_summary(self, context: EscalationContext) -> None:
        """Print a summary of the escalation context to console."""
        separator = "=" * 60
        print(f"\n{separator}")
        print("ESCALATION REQUIRED - Low Confidence Batch Execution")
        print(separator)
        print(f"Job ID:       {context.job_id}")
        print(f"Batch:        {context.batch_num}")
        print(f"Confidence:   {context.confidence:.1%}")
        print(f"Retry Count:  {context.retry_count}")
        print("-" * 60)

        # Validation summary
        passed = sum(1 for v in context.validation_results if v.get("passed", False))
        failed = len(context.validation_results) - passed
        print(f"Validations:  {passed} passed, {failed} failed")

        if failed > 0:
            print("\nFailed validations:")
            for v in context.validation_results:
                if not v.get("passed", False):
                    desc = v.get("description") or v.get("path") or "unnamed"
                    error = v.get("error_message", "no details")
                    print(f"  - {desc}: {error}")

        # Error history
        if context.error_history:
            print("\nRecent errors:")
            for i, error in enumerate(context.error_history[-3:], 1):
                # Truncate long error messages
                error_short = error[:100] + "..." if len(error) > 100 else error
                print(f"  {i}. {error_short}")

        # Output summary (truncated)
        if context.output_summary:
            summary = context.output_summary[:200]
            if len(context.output_summary) > 200:
                summary += "..."
            print(f"\nOutput summary: {summary}")

        print(separator)

    async def _prompt_for_action(
        self, context: EscalationContext
    ) -> EscalationResponse:
        """Prompt user for action and return response.

        Args:
            context: Escalation context (for modify_prompt option).

        Returns:
            EscalationResponse based on user choice.
        """
        print("\nActions:")
        print("  [r] Retry - Try the batch again with same prompt")
        print("  [s] Skip  - Skip this batch and continue")
        print("  [a] Abort - Stop the entire job")
        print("  [m] Modify - Retry with modified prompt")
        print()

        while True:
            try:
                choice = input("Choose action [r/s/a/m]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                # Handle non-interactive or interrupted input
                print("\nNo input available, defaulting to abort.")
                return EscalationResponse(
                    action="abort",
                    guidance="Non-interactive environment or interrupted",
                )

            if choice == "r":
                guidance = self._get_optional_guidance()
                return EscalationResponse(
                    action="retry",
                    guidance=guidance,
                )

            elif choice == "s":
                guidance = self._get_optional_guidance()
                return EscalationResponse(
                    action="skip",
                    guidance=guidance,
                )

            elif choice == "a":
                return EscalationResponse(
                    action="abort",
                    guidance="User aborted via console",
                )

            elif choice == "m":
                modified_prompt = self._get_modified_prompt(context.prompt_used)
                if modified_prompt is None:
                    print("Modification cancelled, please choose again.")
                    continue
                guidance = self._get_optional_guidance()
                return EscalationResponse(
                    action="modify_prompt",
                    modified_prompt=modified_prompt,
                    guidance=guidance,
                )

            else:
                print(f"Invalid choice: '{choice}'. Please enter r, s, a, or m.")

    def _get_optional_guidance(self) -> str | None:
        """Prompt for optional guidance/notes."""
        try:
            guidance = input("Add notes (optional, press Enter to skip): ").strip()
            return guidance if guidance else None
        except (EOFError, KeyboardInterrupt):
            return None

    def _get_modified_prompt(self, original_prompt: str) -> str | None:
        """Get modified prompt from user.

        Args:
            original_prompt: The original prompt to modify.

        Returns:
            Modified prompt string, or None if cancelled.
        """
        print("\nOriginal prompt (first 500 chars):")
        print("-" * 40)
        print(original_prompt[:500])
        if len(original_prompt) > 500:
            print("...")
        print("-" * 40)

        print("\nEnter modification instructions (or 'cancel' to go back):")
        print("You can add a prefix, suffix, or replacement instruction.")

        try:
            instruction = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if instruction.lower() == "cancel":
            return None

        # Simple modification: treat as suffix by default
        if instruction.startswith("PREFIX:"):
            return instruction[7:].strip() + "\n\n" + original_prompt
        elif instruction.startswith("REPLACE:"):
            return instruction[8:].strip()
        else:
            # Default: append as additional instruction
            return original_prompt + "\n\n---\nAdditional guidance: " + instruction
