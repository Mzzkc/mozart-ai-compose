"""Marianne MCP bridge — MCP server subprocess + tool-execution glue.

Originally written as "Marianne Ollama Bridge" during the Ollama
integration epic, this package is now a generic MCP tool-execution
proxy used by any instrument that needs MCP servers managed on its
behalf. Three ghost component references that were listed here but
never implemented were removed in Phase 5 of the backend atlas
migration.

Exposed symbols:

- ``MCPProxyService``: manages MCP server subprocesses for tool
  execution
- ``MCPConnection``, ``MCPTool``, ``ToolResult``, ``ContentBlock``,
  ``ServerCapabilities``: public data types returned by the proxy
- ``ToolExecutionTimeout``, ``ToolNotFoundError``: public exception
  types raised by tool execution
"""

from marianne.bridge.mcp_proxy import (
    ContentBlock,
    MCPConnection,
    MCPProxyService,
    MCPTool,
    ServerCapabilities,
    ToolExecutionTimeout,
    ToolNotFoundError,
    ToolResult,
)

__all__ = [
    "ContentBlock",
    "MCPConnection",
    "MCPProxyService",
    "MCPTool",
    "ServerCapabilities",
    "ToolExecutionTimeout",
    "ToolNotFoundError",
    "ToolResult",
]
