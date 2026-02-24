"""Path validation checks.

Validates that paths referenced in the configuration exist and are accessible.
"""

from pathlib import Path

from mozart.core.config import JobConfig
from mozart.validation.base import ValidationIssue, ValidationSeverity
from mozart.validation.checks._helpers import find_line_in_yaml, resolve_path


class WorkspaceParentExistsCheck:
    """Check that workspace parent directory exists (V002).

    The workspace itself will be created, but its parent must exist.
    This is auto-fixable by creating the parent directories.
    """

    @property
    def check_id(self) -> str:
        return "V002"

    @property
    def severity(self) -> ValidationSeverity:
        return ValidationSeverity.ERROR

    @property
    def description(self) -> str:
        return "Checks that workspace parent directory exists"

    def check(
        self,
        config: JobConfig,
        config_path: Path,
        raw_yaml: str,
    ) -> list[ValidationIssue]:
        """Check workspace parent exists."""
        issues: list[ValidationIssue] = []

        workspace = resolve_path(config.workspace, config_path)
        parent = workspace.parent

        if not parent.exists():
            issues.append(
                ValidationIssue(
                    check_id=self.check_id,
                    severity=self.severity,
                    message=f"Workspace parent directory does not exist: {parent}",
                    line=find_line_in_yaml(raw_yaml, "workspace:"),
                    suggestion=f"Create parent directory: mkdir -p {parent}",
                    auto_fixable=True,
                    metadata={
                        "path": str(parent),
                        "workspace": str(workspace),
                    },
                )
            )

        return issues


class TemplateFileExistsCheck:
    """Check that template_file exists if specified (V003)."""

    @property
    def check_id(self) -> str:
        return "V003"

    @property
    def severity(self) -> ValidationSeverity:
        return ValidationSeverity.ERROR

    @property
    def description(self) -> str:
        return "Checks that template_file exists"

    def check(
        self,
        config: JobConfig,
        config_path: Path,
        raw_yaml: str,
    ) -> list[ValidationIssue]:
        """Check template_file exists."""
        issues: list[ValidationIssue] = []

        if config.prompt.template_file:
            template_path = resolve_path(config.prompt.template_file, config_path)

            if not template_path.exists():
                issues.append(
                    ValidationIssue(
                        check_id=self.check_id,
                        severity=self.severity,
                        message=f"Template file not found: {template_path}",
                        line=find_line_in_yaml(raw_yaml, "template_file:"),
                        suggestion="Create the template file or fix the path",
                        metadata={
                            "expected_path": str(template_path),
                        },
                    )
                )
            elif not template_path.is_file():
                issues.append(
                    ValidationIssue(
                        check_id=self.check_id,
                        severity=self.severity,
                        message=f"Template path is not a file: {template_path}",
                        line=find_line_in_yaml(raw_yaml, "template_file:"),
                        suggestion="Ensure template_file points to a file, not a directory",
                    )
                )

        return issues


class SystemPromptFileCheck:
    """Check that system_prompt_file exists if specified (V004)."""

    @property
    def check_id(self) -> str:
        return "V004"

    @property
    def severity(self) -> ValidationSeverity:
        return ValidationSeverity.ERROR

    @property
    def description(self) -> str:
        return "Checks that system_prompt_file exists"

    def check(
        self,
        config: JobConfig,
        config_path: Path,
        raw_yaml: str,
    ) -> list[ValidationIssue]:
        """Check system_prompt_file exists."""
        issues: list[ValidationIssue] = []

        if config.backend.system_prompt_file:
            sys_prompt_path = resolve_path(
                config.backend.system_prompt_file, config_path
            )

            if not sys_prompt_path.exists():
                issues.append(
                    ValidationIssue(
                        check_id=self.check_id,
                        severity=self.severity,
                        message=f"System prompt file not found: {sys_prompt_path}",
                        line=find_line_in_yaml(raw_yaml, "system_prompt_file:"),
                        suggestion="Create the system prompt file or fix the path",
                        metadata={
                            "expected_path": str(sys_prompt_path),
                        },
                    )
                )

        return issues


class WorkingDirectoryCheck:
    """Check that working_directory is valid if specified (V005)."""

    @property
    def check_id(self) -> str:
        return "V005"

    @property
    def severity(self) -> ValidationSeverity:
        return ValidationSeverity.ERROR

    @property
    def description(self) -> str:
        return "Checks that working_directory exists and is a directory"

    def check(
        self,
        config: JobConfig,
        config_path: Path,
        raw_yaml: str,
    ) -> list[ValidationIssue]:
        """Check working_directory is valid."""
        issues: list[ValidationIssue] = []

        if config.backend.working_directory:
            working_dir = resolve_path(
                config.backend.working_directory, config_path
            )

            if not working_dir.exists():
                issues.append(
                    ValidationIssue(
                        check_id=self.check_id,
                        severity=self.severity,
                        message=f"Working directory does not exist: {working_dir}",
                        line=find_line_in_yaml(raw_yaml, "working_directory:"),
                        suggestion=f"Create directory: mkdir -p {working_dir}",
                        auto_fixable=True,
                        metadata={
                            "path": str(working_dir),
                        },
                    )
                )
            elif not working_dir.is_dir():
                issues.append(
                    ValidationIssue(
                        check_id=self.check_id,
                        severity=self.severity,
                        message=f"Working directory path is not a directory: {working_dir}",
                        line=find_line_in_yaml(raw_yaml, "working_directory:"),
                        suggestion="Ensure path points to a directory, not a file",
                    )
                )

        return issues


class PreludeCadenzaFileCheck:
    """Check that prelude/cadenza files exist when paths are static (V108).

    Only checks non-templated paths (no Jinja ``{{``). Templated paths
    are resolved at execution time and cannot be validated statically.
    This is a WARNING because files might be created before execution.
    """

    @property
    def check_id(self) -> str:
        return "V108"

    @property
    def severity(self) -> ValidationSeverity:
        return ValidationSeverity.WARNING

    @property
    def description(self) -> str:
        return "Checks that prelude/cadenza files exist (static paths only)"

    def check(
        self,
        config: JobConfig,
        config_path: Path,
        raw_yaml: str,
    ) -> list[ValidationIssue]:
        """Check prelude and cadenza file paths."""
        issues: list[ValidationIssue] = []

        all_items: list[tuple[str, str]] = []
        for item in config.sheet.prelude:
            all_items.append((item.file, "prelude"))
        for sheet_num, items in config.sheet.cadenzas.items():
            for item in items:
                all_items.append((item.file, f"cadenza (sheet {sheet_num})"))

        for file_path, source in all_items:
            # Skip Jinja-templated paths — can't resolve statically
            if "{{" in file_path:
                continue

            path = resolve_path(Path(file_path), config_path)

            if not path.exists():
                issues.append(
                    ValidationIssue(
                        check_id=self.check_id,
                        severity=self.severity,
                        message=(
                            f"Injection file from {source} not found: {path}"
                        ),
                        line=find_line_in_yaml(raw_yaml, file_path),
                        suggestion="Create the file or fix the path",
                        metadata={
                            "source": source,
                            "path": str(path),
                        },
                    )
                )

        return issues


class SkillFilesExistCheck:
    """Check that files referenced in validation commands exist (V107).

    This is a WARNING because skill files might be optional.
    """

    @property
    def check_id(self) -> str:
        return "V107"

    @property
    def severity(self) -> ValidationSeverity:
        return ValidationSeverity.WARNING

    @property
    def description(self) -> str:
        return "Checks for referenced files in validation paths"

    def check(
        self,
        config: JobConfig,
        config_path: Path,
        raw_yaml: str,
    ) -> list[ValidationIssue]:
        """Check files referenced in validations."""
        issues: list[ValidationIssue] = []

        for i, validation in enumerate(config.validations):
            # Skip validations with template variables in path
            # Check file_exists validations - these are expected to be created
            # so we don't warn about them. Check other types.
            if (
                validation.path
                and "{" not in validation.path
                and validation.type in ("content_contains", "content_regex")
            ):
                file_path = resolve_path(Path(validation.path), config_path)

                # Only warn if it's an absolute path that doesn't exist
                # Relative paths might be created during execution
                if file_path.is_absolute() and not file_path.exists():
                    issues.append(
                        ValidationIssue(
                            check_id=self.check_id,
                            severity=self.severity,
                            message=(
                                f"File referenced in validation {i + 1}"
                                f" does not exist: {file_path}"
                            ),
                            suggestion="Ensure file will be created before this validation runs",
                            metadata={
                                "validation_index": str(i),
                                "path": str(file_path),
                            },
                        )
                    )

        return issues
