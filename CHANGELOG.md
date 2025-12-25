# Changelog

All notable changes to Mozart AI Compose will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-12-25

### Added

#### Core Features
- **YAML Configuration** - Declarative job definitions with Jinja2 templating
- **Batch Processing** - Split work into configurable batches with customizable sizes
- **Resumable Execution** - Checkpoint-based state management for fault tolerance
- **Smart Retry** - Exponential backoff with jitter and error classification
- **Rate Limit Handling** - Automatic detection and wait with configurable thresholds

#### CLI Commands
- `mozart run` - Execute batch jobs with progress tracking and ETA
- `mozart status` - View detailed job status with batch breakdown
- `mozart resume` - Continue paused or failed jobs from checkpoint
- `mozart list` - List all jobs with filtering by status
- `mozart validate` - Validate configuration files before running
- `mozart dashboard` - Start web dashboard for monitoring

#### State Management
- **JSON Backend** - File-based state storage for simplicity
- **SQLite Backend** - Database-backed state for production use
- **Config Snapshots** - Store configuration in state for resume without original file

#### Validation System
- `file_exists` - Check for expected output files
- `file_modified` - Verify file was updated during batch
- `content_contains` - Match patterns in file content
- `command_succeeds` - Execute shell commands as quality checks
- Confidence scoring for validation results

#### Backends
- **Claude CLI Backend** - Execute via Claude CLI with full options
- **Anthropic API Backend** - Direct API integration with model selection

#### Dashboard
- FastAPI-based REST API
- Job listing with status filtering
- Detailed job view with batch information
- Health check endpoint
- OpenAPI/Swagger documentation
- CORS support for frontend development

#### Notifications
- Desktop notifications via system notify
- Slack webhook integration
- Generic webhook support
- Configurable event triggers (job_complete, job_failed, batch_failed)

#### User Experience
- **Progress Bar** - Real-time progress with ETA during execution
- **Graceful Shutdown** - Ctrl+C saves state and shows resume command
- **Run Summary** - Comprehensive summary panel at job completion
- **Verbosity Control** - `--verbose` and `--quiet` flags
- **JSON Output** - Machine-readable output for scripting

#### Developer Experience
- Type hints throughout with mypy validation
- Comprehensive test suite with pytest
- Linting with ruff
- Pydantic v2 for configuration and state models
- Protocol-based backends for extensibility

### Architecture

```
mozart/
├── core/           # Domain models, config, errors
├── backends/       # Claude CLI and API backends
├── execution/      # Runner, validation, retry logic
├── state/          # JSON and SQLite state backends
├── prompts/        # Jinja2 templating
├── notifications/  # Desktop, Slack, webhook
├── dashboard/      # FastAPI web interface
└── learning/       # Outcome tracking (experimental)
```

### Self-Development Note

Mozart 0.1.0 was developed through an iterative self-improvement process where the AI system helped plan and implement its own features. The development was organized into 12 batches:

1. **Context Gathering** - Analyze existing codebase and gaps
2. **Brainstorming** - Generate feature ideas with TDF prioritization
3. **Planning** - Create detailed implementation plan
4. **SQLite + List** - Integrate SQLite, implement list command
5. **Status Command** - Implement detailed status display
6. **Config Storage** - Store config in state for resume
7. **Resume Command** - Implement job resumption
8. **Dashboard API** - Create FastAPI routes
9. **Dashboard CLI** - Connect dashboard to CLI
10. **Graceful Shutdown** - Ctrl+C handling and progress bars
11. **Run Summary** - Summary panel and verbosity flags
12. **Integration Tests** - End-to-end tests and documentation

This represents a milestone in AI-assisted software development where the system contributed to its own completion.

---

*Mozart AI Compose - Orchestration for the AI Age*
