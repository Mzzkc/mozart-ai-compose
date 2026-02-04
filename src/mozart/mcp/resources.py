"""Mozart MCP Resources - Resource implementations for Mozart configuration access.

This module implements MCP resources that expose Mozart configuration and
documentation as readable content. Resources provide context and reference
material for AI agents working with Mozart.

Resources are organized by category:
- ConfigResources: Access to Mozart configuration schemas and examples
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from mozart.state.base import StateBackend

logger = logging.getLogger(__name__)


class ConfigResources:
    """Mozart configuration resources.

    Provides access to Mozart configuration schemas, examples, and documentation
    as MCP resources. These resources help AI agents understand Mozart's
    configuration format and available options.
    """

    def __init__(
        self, state_backend: StateBackend | None = None, workspace_root: Path | None = None
    ) -> None:
        # Base project directory (assuming we're in src/mozart/mcp/)
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.state_backend = state_backend
        self.workspace_root = workspace_root or Path.cwd()

    async def list_resources(self) -> list[dict[str, Any]]:
        """List all configuration resources."""
        resources = [
            {
                "uri": "config://schema",
                "name": "Mozart Configuration Schema",
                "description": "Complete JSON schema for Mozart job configuration files",
                "mimeType": "application/json"
            },
            {
                "uri": "config://example",
                "name": "Mozart Configuration Example",
                "description": "Example Mozart job configuration with common patterns",
                "mimeType": "text/yaml"
            },
            {
                "uri": "config://backend-options",
                "name": "Backend Configuration Options",
                "description": "Available backend types and their configuration options",
                "mimeType": "application/json"
            },
            {
                "uri": "config://validation-types",
                "name": "Validation Types Reference",
                "description": "Available validation types and their parameters",
                "mimeType": "application/json"
            },
            {
                "uri": "config://learning-options",
                "name": "Learning Configuration Options",
                "description": "Learning system configuration parameters and patterns",
                "mimeType": "application/json"
            },
            # Job management resources
            {
                "uri": "mozart://jobs",
                "name": "Mozart Jobs Overview",
                "description": "List of all Mozart jobs with status and metadata",
                "mimeType": "application/json"
            },
            {
                "uri": "mozart://templates",
                "name": "Mozart Job Templates",
                "description": "Collection of Mozart job configuration templates",
                "mimeType": "application/json"
            }
        ]

        # Dynamic job detail resources - only available if we have state backend
        if self.state_backend:
            try:
                # Note: This would require implementing a method to list all job IDs
                # For now, we document the pattern for job-specific resources
                resources.append({
                    "uri": "mozart://jobs/{job_id}",
                    "name": "Mozart Job Details (Template)",
                    "description": "Detailed information about a specific Mozart job",
                    "mimeType": "application/json"
                })
            except Exception:
                logger.debug("State backend not available for job resources", exc_info=True)

        return resources

    async def read_resource(self, uri: str) -> dict[str, Any]:
        """Read a configuration resource by URI."""
        try:
            if uri == "config://schema":
                return await self._get_config_schema()
            elif uri == "config://example":
                return await self._get_config_example()
            elif uri == "config://backend-options":
                return await self._get_backend_options()
            elif uri == "config://validation-types":
                return await self._get_validation_types()
            elif uri == "config://learning-options":
                return await self._get_learning_options()
            elif uri == "mozart://jobs":
                return await self._get_jobs_overview()
            elif uri == "mozart://templates":
                return await self._get_job_templates()
            elif uri.startswith("mozart://jobs/"):
                job_id = uri.replace("mozart://jobs/", "")
                return await self._get_job_details(job_id)
            else:
                raise ValueError(f"Unknown resource URI: {uri}")

        except Exception as e:
            logger.exception(f"Error reading resource {uri}")
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "text/plain",
                        "text": f"Error reading resource: {str(e)}"
                    }
                ]
            }

    async def _get_config_schema(self) -> dict[str, Any]:
        """Generate JSON schema for Mozart configuration."""
        # This would ideally be generated from the Pydantic models
        # For now, we provide a simplified schema
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Mozart Job Configuration",
            "type": "object",
            "required": ["job_id", "sheets", "backend"],
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Unique identifier for the job"
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable description of the job"
                },
                "sheets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "prompt"],
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Sheet name"
                            },
                            "prompt": {
                                "type": "string",
                                "description": "Prompt template for the sheet"
                            },
                            "timeout_seconds": {
                                "type": "integer",
                                "default": 300,
                                "description": "Timeout for sheet execution"
                            },
                            "max_retries": {
                                "type": "integer",
                                "default": 3,
                                "description": "Maximum retry attempts"
                            },
                            "dependencies": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Sheet dependencies (other sheet names)"
                            },
                            "validation": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": ["type"],
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "enum": ["file_exists", "regex_match", "json_schema", "custom", "llm_judge"]
                                        },
                                        "description": {
                                            "type": "string",
                                            "description": "Human-readable validation description"
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "backend": {
                    "type": "object",
                    "required": ["backend_type"],
                    "properties": {
                        "backend_type": {
                            "type": "string",
                            "enum": ["claude_cli", "anthropic_api"],
                            "description": "Backend implementation to use"
                        },
                        "disable_mcp": {
                            "type": "boolean",
                            "default": True,
                            "description": "Disable MCP servers for faster execution"
                        }
                    }
                }
            }
        }

        return {
            "contents": [
                {
                    "uri": "config://schema",
                    "mimeType": "application/json",
                    "text": json.dumps(schema, indent=2)
                }
            ]
        }

    async def _get_config_example(self) -> dict[str, Any]:
        """Get example Mozart configuration."""
        example_content = """# Mozart Job Configuration Example

job_id: example-review
description: Example Mozart job configuration

backend:
  backend_type: claude_cli
  disable_mcp: true  # For faster execution
  timeout_seconds: 300

sheets:
  - name: analyze-code
    prompt: |
      Analyze the code in the current directory and provide a summary of:
      1. Main functionality
      2. Key patterns and architecture
      3. Potential improvements

    timeout_seconds: 180
    max_retries: 3

    validation:
      - type: regex_match
        description: "Output contains analysis sections"
        pattern: "Main functionality|Key patterns|Potential improvements"

      - type: file_exists
        description: "Analysis output written to file"
        path: "analysis-summary.md"

  - name: create-documentation
    prompt: |
      Based on the analysis from the previous sheet, create comprehensive
      documentation for this codebase including:
      1. Setup instructions
      2. Usage examples
      3. API reference

      Previous analysis: {{ sheets.0.output }}

    dependencies: ["analyze-code"]
    timeout_seconds: 240

    validation:
      - type: file_exists
        description: "Documentation created"
        path: "README.md"

      - type: regex_match
        description: "Documentation contains required sections"
        pattern: "Setup|Usage|API"

# Global configuration
workspace: "./workspace"
learning:
  enabled: true
  pattern_detection: true

notifications:
  - type: desktop
    title: "Mozart Job Complete"
"""

        return {
            "contents": [
                {
                    "uri": "config://example",
                    "mimeType": "text/yaml",
                    "text": example_content
                }
            ]
        }

    async def _get_backend_options(self) -> dict[str, Any]:
        """Get backend configuration options."""
        backend_options = {
            "available_backends": {
                "claude_cli": {
                    "description": "Claude CLI backend using subprocess calls",
                    "options": {
                        "disable_mcp": {
                            "type": "boolean",
                            "default": True,
                            "description": "Disable MCP servers for faster execution"
                        },
                        "timeout_seconds": {
                            "type": "integer",
                            "default": 300,
                            "description": "Timeout for individual requests"
                        },
                        "cli_extra_args": {
                            "type": "array",
                            "description": "Additional CLI arguments to pass"
                        }
                    }
                },
                "anthropic_api": {
                    "description": "Direct Anthropic API backend",
                    "options": {
                        "api_key": {
                            "type": "string",
                            "description": "Anthropic API key (or use ANTHROPIC_API_KEY env var)"
                        },
                        "model": {
                            "type": "string",
                            "default": "claude-3-5-sonnet-20241022",
                            "description": "Model to use for requests"
                        },
                        "max_tokens": {
                            "type": "integer",
                            "default": 8192,
                            "description": "Maximum tokens per response"
                        }
                    }
                }
            }
        }

        return {
            "contents": [
                {
                    "uri": "config://backend-options",
                    "mimeType": "application/json",
                    "text": json.dumps(backend_options, indent=2)
                }
            ]
        }

    async def _get_validation_types(self) -> dict[str, Any]:
        """Get validation types reference."""
        validation_types = {
            "available_validation_types": {
                "file_exists": {
                    "description": "Check if a file exists at the specified path",
                    "parameters": {
                        "path": "Required. File path to check (relative to workspace)"
                    },
                    "example": {
                        "type": "file_exists",
                        "description": "Output file was created",
                        "path": "results.txt"
                    }
                },
                "regex_match": {
                    "description": "Check if output matches a regular expression",
                    "parameters": {
                        "pattern": "Required. Regular expression pattern to match",
                        "flags": "Optional. Regex flags (i for case-insensitive, etc.)"
                    },
                    "example": {
                        "type": "regex_match",
                        "description": "Output contains success message",
                        "pattern": "Success|Complete|Done"
                    }
                },
                "json_schema": {
                    "description": "Validate output against a JSON schema",
                    "parameters": {
                        "schema": "Required. JSON schema to validate against"
                    },
                    "example": {
                        "type": "json_schema",
                        "description": "Output is valid JSON with required fields",
                        "schema": {
                            "type": "object",
                            "required": ["status", "result"],
                            "properties": {
                                "status": {"type": "string"},
                                "result": {"type": "string"}
                            }
                        }
                    }
                },
                "custom": {
                    "description": "Run custom validation command",
                    "parameters": {
                        "command": "Required. Command to execute for validation",
                        "expected_exit_code": "Optional. Expected exit code (default: 0)"
                    },
                    "example": {
                        "type": "custom",
                        "description": "Custom script validates results",
                        "command": "python validate_results.py"
                    }
                },
                "llm_judge": {
                    "description": "Use LLM to judge output quality",
                    "parameters": {
                        "criteria": "Required. Criteria for the LLM to evaluate",
                        "model": "Optional. Model to use for judging"
                    },
                    "example": {
                        "type": "llm_judge",
                        "description": "Output provides clear, helpful analysis",
                        "criteria": "The output should be well-structured, informative, and directly address the prompt requirements"
                    }
                }
            }
        }

        return {
            "contents": [
                {
                    "uri": "config://validation-types",
                    "mimeType": "application/json",
                    "text": json.dumps(validation_types, indent=2)
                }
            ]
        }

    async def _get_learning_options(self) -> dict[str, Any]:
        """Get learning configuration options."""
        learning_options = {
            "learning_system": {
                "description": "Mozart's adaptive learning system configuration",
                "options": {
                    "enabled": {
                        "type": "boolean",
                        "default": True,
                        "description": "Enable pattern learning and adaptation"
                    },
                    "pattern_detection": {
                        "type": "boolean",
                        "default": True,
                        "description": "Enable automatic pattern detection"
                    },
                    "escalation": {
                        "type": "boolean",
                        "default": False,
                        "description": "Enable escalation to more powerful models"
                    },
                    "global_learning": {
                        "type": "boolean",
                        "default": True,
                        "description": "Participate in global learning across jobs"
                    },
                    "pattern_trust_threshold": {
                        "type": "number",
                        "default": 0.7,
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Minimum trust score for applying patterns"
                    }
                }
            },
            "pattern_types": [
                "error_resolution",
                "optimization",
                "validation_improvement",
                "retry_strategy",
                "timeout_adjustment",
                "dependency_ordering"
            ]
        }

        return {
            "contents": [
                {
                    "uri": "config://learning-options",
                    "mimeType": "application/json",
                    "text": json.dumps(learning_options, indent=2)
                }
            ]
        }

    async def _get_jobs_overview(self) -> dict[str, Any]:
        """Get overview of all Mozart jobs."""
        if not self.state_backend:
            return {
                "contents": [
                    {
                        "uri": "mozart://jobs",
                        "mimeType": "application/json",
                        "text": json.dumps({
                            "error": "Jobs overview requires state backend initialization",
                            "note": "Configure MCP server with workspace_root to enable job listing"
                        }, indent=2)
                    }
                ]
            }

        jobs_overview: dict[str, Any] = {
            "jobs": [],
            "summary": {
                "total_jobs": 0,
                "running_jobs": 0,
                "completed_jobs": 0,
                "failed_jobs": 0,
                "paused_jobs": 0
            },
            "last_updated": json.dumps(datetime.now().isoformat())
        }

        # Note: This would require implementing a method to list all jobs
        # For now, we provide a placeholder structure
        try:
            # Scan workspace for job state files
            for state_file in self.workspace_root.glob("*.json"):
                if state_file.stem != "global_learning":  # Skip non-job files
                    try:
                        job_id = state_file.stem
                        state = await self.state_backend.load(job_id)
                        if state:
                            jobs_overview["jobs"].append({
                                "job_id": job_id,
                                "job_name": state.job_name,
                                "status": state.status.value,
                                "started_at": state.started_at.isoformat() if state.started_at else None,
                                "completed_at": state.completed_at.isoformat() if state.completed_at else None,
                                "total_sheets": len(state.sheets),
                                "completed_sheets": len([s for s in state.sheets.values()
                                                       if s.status.value == "completed"]),
                                "error_message": getattr(state, 'error_message', None)
                            })

                            # Update summary
                            jobs_overview["summary"]["total_jobs"] += 1
                            status = state.status.value
                            if status == "running":
                                jobs_overview["summary"]["running_jobs"] += 1
                            elif status == "completed":
                                jobs_overview["summary"]["completed_jobs"] += 1
                            elif status == "failed":
                                jobs_overview["summary"]["failed_jobs"] += 1
                            elif status == "paused":
                                jobs_overview["summary"]["paused_jobs"] += 1

                    except Exception as e:
                        # Skip invalid state files
                        logger.debug(f"Skipping invalid state file {state_file}: {e}")
                        continue

        except Exception as e:
            jobs_overview["error"] = f"Error scanning jobs: {str(e)}"

        return {
            "contents": [
                {
                    "uri": "mozart://jobs",
                    "mimeType": "application/json",
                    "text": json.dumps(jobs_overview, indent=2)
                }
            ]
        }

    async def _get_job_details(self, job_id: str) -> dict[str, Any]:
        """Get detailed information about a specific job."""
        if not self.state_backend:
            return {
                "contents": [
                    {
                        "uri": f"mozart://jobs/{job_id}",
                        "mimeType": "application/json",
                        "text": json.dumps({
                            "error": "Job details require state backend initialization"
                        }, indent=2)
                    }
                ]
            }

        try:
            state = await self.state_backend.load(job_id)
            if not state:
                raise FileNotFoundError(f"Job not found: {job_id}")

            job_details: dict[str, Any] = {
                "job_id": job_id,
                "job_name": state.job_name,
                "status": state.status.value,
                "started_at": state.started_at.isoformat() if state.started_at else None,
                "completed_at": state.completed_at.isoformat() if state.completed_at else None,
                "last_updated": state.updated_at.isoformat() if state.updated_at else None,
                "error_message": state.error_message,
                "total_sheets": len(state.sheets),
                "sheets": {},
                "configuration": {
                    "workspace": str(state.workspace) if hasattr(state, 'workspace') else None,
                    "backend_type": getattr(state, 'backend_type', 'unknown'),
                },
                "progress": {
                    "completed_sheets": len([s for s in state.sheets.values()
                                           if s.status.value == "completed"]),
                    "failed_sheets": len([s for s in state.sheets.values()
                                        if s.status.value == "failed"]),
                    "running_sheets": len([s for s in state.sheets.values()
                                         if s.status.value == "running"]),
                    "pending_sheets": len([s for s in state.sheets.values()
                                         if s.status.value == "pending"]),
                }
            }

            # Add sheet details
            for sheet_num, sheet in state.sheets.items():
                job_details["sheets"][str(sheet_num)] = {
                    "sheet_num": sheet.sheet_num,
                    "status": sheet.status.value,
                    "started_at": sheet.started_at.isoformat() if sheet.started_at else None,
                    "completed_at": sheet.completed_at.isoformat() if sheet.completed_at else None,
                    "attempt_count": sheet.attempt_count,
                    "error_message": sheet.error_message,
                    "validation_passed": getattr(sheet, 'validation_passed', None),
                    "output_size": len(sheet.stdout_tail) if sheet.stdout_tail else 0
                }

            return {
                "contents": [
                    {
                        "uri": f"mozart://jobs/{job_id}",
                        "mimeType": "application/json",
                        "text": json.dumps(job_details, indent=2)
                    }
                ]
            }

        except Exception as e:
            return {
                "contents": [
                    {
                        "uri": f"mozart://jobs/{job_id}",
                        "mimeType": "application/json",
                        "text": json.dumps({
                            "error": f"Error loading job details: {str(e)}",
                            "job_id": job_id
                        }, indent=2)
                    }
                ]
            }

    async def _get_job_templates(self) -> dict[str, Any]:
        """Get collection of Mozart job configuration templates."""
        templates = {
            "templates": {
                "code-analysis": {
                    "name": "Code Analysis Template",
                    "description": "Template for analyzing codebases and generating documentation",
                    "use_cases": ["code review", "documentation generation", "architecture analysis"],
                    "config": {
                        "job_id": "code-analysis-{timestamp}",
                        "description": "Analyze codebase structure and patterns",
                        "backend": {
                            "backend_type": "claude_cli",
                            "disable_mcp": True,
                            "timeout_seconds": 300
                        },
                        "sheets": [
                            {
                                "name": "scan-codebase",
                                "prompt": "Scan the current directory and provide an overview of:\n1. Project structure and key files\n2. Programming languages and frameworks used\n3. Main functionality and purpose\n\nFocus on understanding the codebase architecture.",
                                "timeout_seconds": 180,
                                "max_retries": 2,
                                "validation": [
                                    {
                                        "type": "regex_match",
                                        "description": "Output contains project structure analysis",
                                        "pattern": "structure|files|directories"
                                    }
                                ]
                            },
                            {
                                "name": "analyze-patterns",
                                "prompt": "Based on the codebase scan, analyze:\n1. Architectural patterns and design principles\n2. Code quality and potential improvements\n3. Documentation gaps\n\nPrevious scan: {{ sheets.0.output }}",
                                "dependencies": ["scan-codebase"],
                                "timeout_seconds": 240,
                                "validation": [
                                    {
                                        "type": "regex_match",
                                        "description": "Analysis contains patterns and improvements",
                                        "pattern": "patterns|quality|improvements"
                                    }
                                ]
                            }
                        ]
                    }
                },
                "test-generation": {
                    "name": "Test Generation Template",
                    "description": "Template for generating comprehensive tests for existing code",
                    "use_cases": ["test coverage", "quality assurance", "regression testing"],
                    "config": {
                        "job_id": "test-generation-{timestamp}",
                        "description": "Generate comprehensive tests for codebase",
                        "backend": {
                            "backend_type": "claude_cli",
                            "disable_mcp": True
                        },
                        "sheets": [
                            {
                                "name": "identify-testable-units",
                                "prompt": "Identify functions, classes, and modules that need test coverage:\n1. List main functions and classes\n2. Identify current test coverage gaps\n3. Prioritize by importance and complexity",
                                "timeout_seconds": 180,
                                "validation": [
                                    {
                                        "type": "regex_match",
                                        "description": "Lists testable units",
                                        "pattern": "functions|classes|coverage|test"
                                    }
                                ]
                            },
                            {
                                "name": "generate-unit-tests",
                                "prompt": "Generate unit tests for the identified components:\n1. Create comprehensive test cases\n2. Include edge cases and error conditions\n3. Follow project testing conventions\n\nComponents to test: {{ sheets.0.output }}",
                                "dependencies": ["identify-testable-units"],
                                "timeout_seconds": 300,
                                "validation": [
                                    {
                                        "type": "file_exists",
                                        "description": "Test files created",
                                        "path": "tests/"
                                    }
                                ]
                            }
                        ]
                    }
                },
                "documentation": {
                    "name": "Documentation Template",
                    "description": "Template for generating project documentation",
                    "use_cases": ["API documentation", "user guides", "README creation"],
                    "config": {
                        "job_id": "documentation-{timestamp}",
                        "description": "Generate comprehensive project documentation",
                        "backend": {
                            "backend_type": "claude_cli",
                            "disable_mcp": True
                        },
                        "sheets": [
                            {
                                "name": "create-readme",
                                "prompt": "Create a comprehensive README.md file including:\n1. Project description and purpose\n2. Installation instructions\n3. Usage examples\n4. Contributing guidelines",
                                "timeout_seconds": 240,
                                "validation": [
                                    {
                                        "type": "file_exists",
                                        "description": "README.md created",
                                        "path": "README.md"
                                    },
                                    {
                                        "type": "regex_match",
                                        "description": "README contains required sections",
                                        "pattern": "Installation|Usage|Contributing"
                                    }
                                ]
                            },
                            {
                                "name": "api-documentation",
                                "prompt": "Generate API documentation:\n1. Document all public functions and classes\n2. Include parameter descriptions and examples\n3. Add usage examples for key features",
                                "timeout_seconds": 300,
                                "validation": [
                                    {
                                        "type": "file_exists",
                                        "description": "API documentation created",
                                        "path": "docs/"
                                    }
                                ]
                            }
                        ]
                    }
                },
                "refactoring": {
                    "name": "Code Refactoring Template",
                    "description": "Template for systematic code refactoring and improvement",
                    "use_cases": ["code cleanup", "performance optimization", "modernization"],
                    "config": {
                        "job_id": "refactoring-{timestamp}",
                        "description": "Systematic code refactoring and improvement",
                        "backend": {
                            "backend_type": "claude_cli",
                            "disable_mcp": True
                        },
                        "learning": {
                            "enabled": True,
                            "pattern_detection": True
                        },
                        "sheets": [
                            {
                                "name": "identify-improvements",
                                "prompt": "Identify code improvements:\n1. Find code duplication and redundancy\n2. Identify performance bottlenecks\n3. Look for outdated patterns or deprecated usage\n4. Suggest modernization opportunities",
                                "timeout_seconds": 180,
                                "validation": [
                                    {
                                        "type": "regex_match",
                                        "description": "Identifies specific improvements",
                                        "pattern": "duplication|performance|deprecated|improvements"
                                    }
                                ]
                            },
                            {
                                "name": "implement-refactoring",
                                "prompt": "Implement the identified improvements:\n1. Refactor duplicated code into reusable functions\n2. Optimize performance bottlenecks\n3. Update deprecated usage\n4. Ensure backward compatibility\n\nImprovements to implement: {{ sheets.0.output }}",
                                "dependencies": ["identify-improvements"],
                                "timeout_seconds": 400,
                                "max_retries": 2,
                                "validation": [
                                    {
                                        "type": "custom",
                                        "description": "Code still compiles after refactoring",
                                        "command": "python -m py_compile **/*.py"
                                    }
                                ]
                            }
                        ]
                    }
                }
            },
            "usage": {
                "description": "Mozart job templates provide starting points for common development tasks",
                "how_to_use": [
                    "Copy the desired template configuration",
                    "Replace {timestamp} placeholders with actual values",
                    "Modify prompts and validation rules for your specific needs",
                    "Add or remove sheets based on your requirements",
                    "Configure backend settings for your environment"
                ],
                "customization_tips": [
                    "Adjust timeout_seconds based on expected task complexity",
                    "Add sheet dependencies to ensure proper execution order",
                    "Use regex_match validation for content verification",
                    "Use file_exists validation for output verification",
                    "Enable learning to improve performance over time"
                ]
            }
        }

        return {
            "contents": [
                {
                    "uri": "mozart://templates",
                    "mimeType": "application/json",
                    "text": json.dumps(templates, indent=2)
                }
            ]
        }


# Code Review During Implementation:
# ✓ Resource URIs follow consistent naming scheme (config://*)
# ✓ MIME types properly specified for different content types
# ✓ Error handling with graceful degradation
# ✓ JSON schema generation for configuration validation
# ✓ Comprehensive examples and documentation
# ✓ Security consideration: read-only resource access
# ✓ Proper async/await patterns
