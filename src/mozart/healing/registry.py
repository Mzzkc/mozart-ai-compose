"""Registry of available remedies.

Provides a central place to register and look up remedies.
The create_default_registry() factory returns a registry with
all built-in remedies pre-registered.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mozart.healing.context import ErrorContext
    from mozart.healing.diagnosis import Diagnosis
    from mozart.healing.remedies.base import Remedy


class RemedyRegistry:
    """Registry of available remedies.

    Maintains a list of remedy instances and provides methods
    to query them by name or find applicable remedies for a
    given error context.

    Example:
        registry = RemedyRegistry()
        registry.register(CreateMissingWorkspaceRemedy())

        # Find by name
        remedy = registry.get_by_name("create_missing_workspace")

        # Find all that apply to an error
        applicable = registry.find_applicable(context)
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._remedies: list["Remedy"] = []

    def register(self, remedy: "Remedy") -> None:
        """Register a remedy.

        Args:
            remedy: Remedy instance to register.
        """
        self._remedies.append(remedy)

    def all_remedies(self) -> list["Remedy"]:
        """Get all registered remedies.

        Returns:
            List of all registered remedy instances.
        """
        return list(self._remedies)

    def get_by_name(self, name: str) -> "Remedy | None":
        """Get remedy by name.

        Args:
            name: Unique remedy identifier.

        Returns:
            Remedy if found, None otherwise.
        """
        for remedy in self._remedies:
            if remedy.name == name:
                return remedy
        return None

    def find_applicable(
        self,
        context: "ErrorContext",
    ) -> list[tuple["Remedy", "Diagnosis"]]:
        """Find all remedies that apply to the error.

        Queries each registered remedy and collects those that
        return a diagnosis. Results are sorted by diagnosis
        confidence (highest first).

        Args:
            context: Error context with diagnostic information.

        Returns:
            List of (remedy, diagnosis) tuples sorted by confidence.
        """
        applicable: list[tuple["Remedy", "Diagnosis"]] = []

        for remedy in self._remedies:
            try:
                diagnosis = remedy.diagnose(context)
                if diagnosis is not None:
                    applicable.append((remedy, diagnosis))
            except Exception:
                # Individual remedy failures shouldn't block finding others
                pass

        # Sort by diagnosis confidence, highest first
        applicable.sort(key=lambda x: x[1].confidence, reverse=True)
        return applicable

    def count(self) -> int:
        """Get the number of registered remedies.

        Returns:
            Number of remedies in the registry.
        """
        return len(self._remedies)


def create_default_registry() -> RemedyRegistry:
    """Create registry with all built-in remedies.

    This is the primary factory for getting a fully-configured
    remedy registry. It registers:

    Automatic remedies (safe, apply without confirmation):
    - CreateMissingWorkspaceRemedy
    - CreateMissingParentDirsRemedy
    - FixPathSeparatorsRemedy

    Suggested remedies (require user confirmation):
    - SuggestJinjaFixRemedy

    Diagnostic remedies (provide guidance only):
    - DiagnoseAuthErrorRemedy
    - DiagnoseMissingCLIRemedy

    Returns:
        RemedyRegistry with all built-in remedies registered.
    """
    from mozart.healing.remedies.diagnostics import (
        DiagnoseAuthErrorRemedy,
        DiagnoseMissingCLIRemedy,
    )
    from mozart.healing.remedies.jinja import SuggestJinjaFixRemedy
    from mozart.healing.remedies.paths import (
        CreateMissingParentDirsRemedy,
        CreateMissingWorkspaceRemedy,
        FixPathSeparatorsRemedy,
    )

    registry = RemedyRegistry()

    # Automatic remedies (safe, apply without asking)
    registry.register(CreateMissingWorkspaceRemedy())
    registry.register(CreateMissingParentDirsRemedy())
    registry.register(FixPathSeparatorsRemedy())

    # Suggested remedies (ask user first)
    registry.register(SuggestJinjaFixRemedy())

    # Diagnostic-only (provide guidance)
    registry.register(DiagnoseAuthErrorRemedy())
    registry.register(DiagnoseMissingCLIRemedy())

    return registry
