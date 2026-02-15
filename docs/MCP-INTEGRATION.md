# Mozart MCP Integration Guide

This guide explains how to integrate Mozart AI Compose with Claude Desktop using the Model Context Protocol (MCP) server.

## Overview

Mozart's MCP server provides external AI agents with comprehensive access to:

- **Job Management**: Start, pause, resume, and cancel Mozart jobs
- **Progress Monitoring**: Real-time status, process health, and execution logs
- **Workspace Access**: Browse files, read artifacts, and access job outputs
- **Quality Assessment**: Code review scoring and validation tools
- **Configuration**: Access to job configurations and runtime settings

## Claude Desktop Configuration

### 1. Prerequisites

Ensure Mozart is properly installed and accessible via CLI:

```bash
# Verify Mozart installation
mozart --version

# Test basic functionality
mozart validate examples/sheet-review.yaml
```

### 2. Configure Claude Desktop MCP

Add the Mozart MCP server to your Claude Desktop configuration file.

**Location:**
- **macOS**: `~/Library/Application\ Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

**Configuration:**

```json
{
  "mcpServers": {
    "mozart": {
      "command": "python",
      "args": ["-m", "mozart.mcp.server"],
      "env": {
        "MOZART_WORKSPACE_ROOT": "/path/to/your/workspaces"
      }
    }
  }
}
```

### 3. Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `MOZART_WORKSPACE_ROOT` | Root directory for workspace operations | Current working directory |
| `MOZART_LOG_LEVEL` | Logging level for MCP server | `INFO` |

### 4. Workspace Security

The MCP server restricts file system access to the configured workspace root directory. This prevents:

- Access to files outside designated workspaces
- Execution of arbitrary system commands
- Unauthorized file modifications

## Available Tools

### Job Management Tools

#### `list_jobs`
List all Mozart jobs with optional status filtering.

**Parameters:**
- `status_filter` (optional): Filter by job status (`running`, `paused`, `completed`, `failed`, `cancelled`)
- `limit` (optional): Maximum number of jobs to return (default: 50)

#### `get_job`
Get detailed information about a specific Mozart job.

**Parameters:**
- `job_id` (required): Mozart job ID to retrieve

**Returns:**
- Job metadata and configuration
- Process information (PID, CPU, memory)
- Sheet-level progress details
- Recent execution status

#### `start_job`
Start a new Mozart job from a configuration file.

**Parameters:**
- `config_path` (required): Path to Mozart job configuration file
- `workspace` (optional): Workspace directory for job execution
- `start_sheet` (optional): Sheet number to start from (default: 1)
- `self_healing` (optional): Enable automatic error recovery (default: false)

### Job Control Tools

#### `pause_job`
Gracefully pause a running Mozart job at the next sheet boundary.

**Parameters:**
- `job_id` (required): Mozart job ID to pause

#### `resume_job`
Resume a paused Mozart job from where it left off.

**Parameters:**
- `job_id` (required): Mozart job ID to resume

#### `cancel_job`
Permanently cancel a running Mozart job.

**Parameters:**
- `job_id` (required): Mozart job ID to cancel

### Artifact Management Tools

#### `mozart_artifact_list`
List files in a Mozart workspace.

**Parameters:**
- `workspace` (required): Workspace directory to browse
- `path` (optional): Subdirectory path within workspace
- `include_hidden` (optional): Include hidden files (default: false)

#### `mozart_artifact_read`
Read content of a file in the workspace.

**Parameters:**
- `workspace` (required): Workspace directory
- `file_path` (required): Path to file within workspace
- `max_size` (optional): Maximum file size to read in bytes (default: 50,000)
- `encoding` (optional): Text encoding to use (default: UTF-8)

#### `mozart_artifact_get_logs`
Get execution logs from a Mozart job.

**Parameters:**
- `job_id` (required): Mozart job ID
- `workspace` (optional): Workspace directory (auto-detected if not provided)
- `lines` (optional): Number of recent lines to return (default: 100)
- `level` (optional): Log level filter (`debug`, `info`, `warning`, `error`, `all`)

#### `mozart_artifact_list_artifacts`
List all artifacts created by a Mozart job.

**Parameters:**
- `job_id` (required): Mozart job ID
- `workspace` (optional): Workspace directory (auto-detected if not provided)
- `sheet_filter` (optional): Filter artifacts by sheet number
- `artifact_type` (optional): Filter by type (`output`, `error`, `log`, `state`, `all`)

#### `mozart_artifact_get_artifact`
Get a specific artifact from a Mozart job.

**Parameters:**
- `job_id` (required): Mozart job ID
- `artifact_path` (required): Relative path to artifact within job workspace
- `workspace` (optional): Workspace directory (auto-detected if not provided)
- `max_size` (optional): Maximum artifact size to read (default: 100,000 bytes)

### Code Quality Score Tools

#### `validate_score`
Validate that code changes meet quality score thresholds.

**Parameters:**
- `workspace` (required): Workspace directory to validate
- `min_score` (optional): Minimum acceptable quality score 0-100 (default: 60)
- `target_score` (optional): Target score for high quality 0-100 (default: 80)
- `since_commit` (optional): Git commit hash to diff from

**Score Components:**
- **Code Quality (30%)**: Complexity, patterns, readability
- **Test Coverage (25%)**: New code tested, edge cases covered
- **Security (25%)**: No secrets, proper validation, safe error handling
- **Documentation (20%)**: APIs documented, complex logic explained

#### `generate_score`
Generate quality score for code changes with detailed feedback.

**Parameters:**
- `workspace` (required): Workspace directory to score
- `since_commit` (optional): Git commit hash to diff from
- `detailed` (optional): Include detailed feedback and suggestions (default: false)

## Usage Examples

### Starting a Job

```javascript
// In Claude Desktop conversation
await call_tool("start_job", {
  config_path: "/workspaces/my-project/job-config.yaml",
  workspace: "/workspaces/my-project",
  self_healing: true
});
```

### Monitoring Progress

```javascript
// Get detailed job status
const status = await call_tool("get_job", {
  job_id: "my-job-123"
});

// Get recent logs
const logs = await call_tool("mozart_artifact_get_logs", {
  job_id: "my-job-123",
  lines: 50,
  level: "error"
});
```

### Quality Assessment

```javascript
// Validate code quality
const validation = await call_tool("validate_score", {
  workspace: "/workspaces/my-project",
  min_score: 70,
  target_score: 85
});

// Generate detailed score report
const score_report = await call_tool("generate_score", {
  workspace: "/workspaces/my-project",
  detailed: true
});
```

### Workspace Management

```javascript
// Browse workspace files
const files = await call_tool("mozart_artifact_list", {
  workspace: "/workspaces/my-project",
  include_hidden: false
});

// Read specific artifact
const artifact = await call_tool("mozart_artifact_get_artifact", {
  job_id: "my-job-123",
  artifact_path: "output/results.json"
});
```

## Available Resources

### Configuration Resources

The MCP server exposes Mozart configurations as readable resources:

- `config://job/{job_id}` - Job configuration
- `mozart://config/global` - Global Mozart settings

## Security Considerations

### User Consent Required

All Mozart MCP tools require explicit user consent before execution due to:
- File system access and modification
- Process execution and management
- Network operations for job orchestration

### Access Restrictions

- File access limited to configured workspace root
- No arbitrary code execution beyond Mozart's built-in capabilities
- Process management restricted to Mozart job processes only

### Data Protection

- Sensitive configuration data (API keys, credentials) are masked in responses
- Log outputs are truncated to prevent information leakage
- All file operations include path validation to prevent directory traversal

## Troubleshooting

### Common Issues

**MCP Server Not Starting**
```bash
# Check Mozart installation
mozart --version

# Verify MCP server module
python -m mozart.mcp.server --help
```

**Permission Errors**
```bash
# Ensure workspace directory exists and is writable
mkdir -p /path/to/workspaces
chmod 755 /path/to/workspaces
```

**Tool Execution Failures**
```bash
# Check Mozart CLI functionality
mozart validate examples/sheet-review.yaml

# Verify workspace permissions
ls -la /path/to/workspace
```

### Debugging

Enable debug logging in Claude Desktop configuration:

```json
{
  "mcpServers": {
    "mozart": {
      "command": "python",
      "args": ["-m", "mozart.mcp.server", "--debug"],
      "env": {
        "MOZART_WORKSPACE_ROOT": "/path/to/workspaces",
        "MOZART_LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

## Daemon Integration

When the Mozart conductor (`mozart start`) is running, the MCP server automatically routes job operations through it. This enables:

- **Coordinated rate limiting** across multiple concurrent jobs
- **Centralized learning** — patterns learned by one job benefit others
- **Resource monitoring** — backpressure prevents resource exhaustion

The MCP server detects the daemon automatically. No additional configuration is required.

### Standalone vs Daemon Mode

| Mode | How to Start | When to Use |
|------|-------------|-------------|
| Standalone | `mozart mcp` | Single-job workflows, development |
| Through Conductor | `mozart start` then `mozart mcp` | Multi-job orchestration, production |

### Related Commands

- `mozart mcp` — Start the MCP server (see [CLI Reference](cli-reference.md#mozart-mcp))
- `mozart start` — Start the conductor (see [CLI Reference](cli-reference.md#mozart-start))
- `mozart config` — Manage conductor configuration (see [CLI Reference](cli-reference.md#mozart-config))

---

## Integration Patterns

### Automated Job Monitoring

Use MCP tools to create automated monitoring workflows:

1. Start job with `start_job`
2. Poll status with `get_job`
3. Retrieve logs on errors with `mozart_artifact_get_logs`
4. Validate results with `validate_score`

### Quality Gates

Implement quality gates in development workflows:

1. Run Mozart job for code analysis
2. Use `validate_score` to check quality thresholds
3. Block deployment if score below minimum
4. Generate detailed reports with `generate_score`

### Workspace Analysis

Perform comprehensive workspace analysis:

1. List all artifacts with `mozart_artifact_list_artifacts`
2. Read configuration files with `mozart_artifact_read`
3. Analyze execution logs with `mozart_artifact_get_logs`
4. Generate quality assessments with score tools

## Support

For issues with MCP integration:

1. Verify Mozart CLI functionality first
2. Check Claude Desktop MCP server logs
3. Validate workspace permissions and configuration
4. Refer to Mozart documentation for job configuration

---

*This integration guide reflects the current Mozart MCP capabilities. MCP tools and capabilities continue to evolve.*