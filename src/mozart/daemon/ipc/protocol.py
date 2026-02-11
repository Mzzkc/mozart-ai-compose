"""JSON-RPC 2.0 wire protocol models for Mozart daemon IPC.

Defines Pydantic v2 models for the JSON-RPC 2.0 message types used over
the Unix domain socket. These models enforce the wire format at the
serialization boundary — business logic never touches raw dicts.

Wire format: newline-delimited JSON (NDJSON). Each message is a single
JSON object terminated by ``\\n``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# JSON-RPC 2.0 base types
# ---------------------------------------------------------------------------


class JsonRpcRequest(BaseModel):
    """Inbound JSON-RPC 2.0 request.

    When ``id`` is None the message is a *notification* — the server
    MUST NOT send a response.
    """

    jsonrpc: Literal["2.0"] = "2.0"
    method: str
    params: dict[str, Any] | None = None
    id: int | str | None = None


class ErrorDetail(BaseModel):
    """Error payload within a JSON-RPC error response."""

    code: int
    message: str
    data: dict[str, Any] | None = None


class JsonRpcResponse(BaseModel):
    """Outbound JSON-RPC 2.0 success response."""

    jsonrpc: Literal["2.0"] = "2.0"
    result: Any
    id: int | str


class JsonRpcError(BaseModel):
    """Outbound JSON-RPC 2.0 error response."""

    jsonrpc: Literal["2.0"] = "2.0"
    error: ErrorDetail
    id: int | str | None


class JsonRpcNotification(BaseModel):
    """Server-initiated notification (no ``id``, no response expected).

    Used for streaming log lines (``log.line``) and heartbeats
    (``log.heartbeat``) pushed over the same connection.
    """

    jsonrpc: Literal["2.0"] = "2.0"
    method: str
    params: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Job-specific parameter models
# ---------------------------------------------------------------------------


class JobStatusParams(BaseModel):
    """Parameters for ``job.status``, ``job.pause``, ``job.cancel``."""

    job_id: str
    workspace: Path


class JobResumeParams(BaseModel):
    """Parameters for ``job.resume``."""

    job_id: str
    workspace: Path
    config_path: Path | None = None
    reload_config: bool = False
    self_healing: bool = False
    self_healing_auto_confirm: bool = False


class JobListParams(BaseModel):
    """Parameters for ``job.list``."""

    workspace: Path | None = None


class JobLogsParams(BaseModel):
    """Parameters for ``job.logs``."""

    job_id: str
    workspace: Path
    follow: bool = False
    tail: int = 100


class ShutdownParams(BaseModel):
    """Parameters for ``daemon.shutdown``."""

    graceful: bool = True


class JobSummary(BaseModel):
    """Lightweight job info returned by ``job.list``."""

    job_id: str
    status: str
    total_sheets: int
    completed_sheets: int
    workspace: str
    started_at: str | None = None


__all__ = [
    "ErrorDetail",
    "JobListParams",
    "JobLogsParams",
    "JobResumeParams",
    "JobStatusParams",
    "JobSummary",
    "JsonRpcError",
    "JsonRpcNotification",
    "JsonRpcRequest",
    "JsonRpcResponse",
    "ShutdownParams",
]
