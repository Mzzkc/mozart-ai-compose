"""Tests for mozart.daemon.ipc.protocol — JSON-RPC 2.0 wire format models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from mozart.daemon.ipc.protocol import (
    ErrorDetail,
    JobListParams,
    JobLogsParams,
    JobResumeParams,
    JobStatusParams,
    JobSummary,
    JsonRpcError,
    JsonRpcNotification,
    JsonRpcRequest,
    JsonRpcResponse,
    ShutdownParams,
)


class TestJsonRpcRequest:
    """Tests for JsonRpcRequest model."""

    def test_minimal_request(self):
        """Method-only request with defaults."""
        req = JsonRpcRequest(method="daemon.status")
        assert req.jsonrpc == "2.0"
        assert req.method == "daemon.status"
        assert req.params is None
        assert req.id is None

    def test_request_with_params_and_id(self):
        """Full request with params and numeric ID."""
        req = JsonRpcRequest(
            method="job.submit",
            params={"config_path": "/tmp/test.yaml"},
            id=42,
        )
        assert req.method == "job.submit"
        assert req.params == {"config_path": "/tmp/test.yaml"}
        assert req.id == 42

    def test_request_with_string_id(self):
        """JSON-RPC allows string IDs."""
        req = JsonRpcRequest(method="job.status", id="abc-123")
        assert req.id == "abc-123"

    def test_notification_has_no_id(self):
        """A notification is a request where id is None."""
        req = JsonRpcRequest(method="log.heartbeat")
        assert req.id is None

    def test_serialization_roundtrip(self):
        """Request survives model_dump_json -> model_validate_json roundtrip."""
        original = JsonRpcRequest(
            method="job.submit",
            params={"config_path": "/tmp/test.yaml", "fresh": True},
            id=7,
        )
        json_bytes = original.model_dump_json()
        restored = JsonRpcRequest.model_validate_json(json_bytes)
        assert restored.method == original.method
        assert restored.params == original.params
        assert restored.id == original.id

    def test_jsonrpc_field_defaults_to_2_0(self):
        """The jsonrpc field should always default to '2.0'."""
        req = JsonRpcRequest(method="test")
        dumped = req.model_dump()
        assert dumped["jsonrpc"] == "2.0"

    def test_invalid_jsonrpc_version_rejected(self):
        """Constructing with wrong jsonrpc version fails Pydantic validation."""
        with pytest.raises(ValidationError):
            JsonRpcRequest(jsonrpc="1.0", method="test")  # type: ignore[arg-type]

    def test_from_dict(self):
        """Construct from raw dict (as received over wire)."""
        raw = {"jsonrpc": "2.0", "method": "daemon.shutdown", "id": 1}
        req = JsonRpcRequest.model_validate(raw)
        assert req.method == "daemon.shutdown"
        assert req.id == 1


class TestJsonRpcResponse:
    """Tests for JsonRpcResponse model."""

    def test_success_response(self):
        """Basic success response."""
        resp = JsonRpcResponse(result={"status": "ok"}, id=1)
        assert resp.jsonrpc == "2.0"
        assert resp.result == {"status": "ok"}
        assert resp.id == 1

    def test_null_result(self):
        """Result can be None/null for void methods."""
        resp = JsonRpcResponse(result=None, id=5)
        assert resp.result is None

    def test_serialization_roundtrip(self):
        """Response survives JSON roundtrip."""
        original = JsonRpcResponse(
            result={"jobs": [{"id": "j1", "status": "running"}]},
            id=99,
        )
        json_bytes = original.model_dump_json()
        restored = JsonRpcResponse.model_validate_json(json_bytes)
        assert restored.result == original.result
        assert restored.id == original.id

    def test_string_id(self):
        """Response with string ID."""
        resp = JsonRpcResponse(result="pong", id="req-42")
        assert resp.id == "req-42"


class TestJsonRpcError:
    """Tests for JsonRpcError model."""

    def test_error_response(self):
        """Basic error response."""
        error = JsonRpcError(
            error=ErrorDetail(code=-32600, message="Invalid request"),
            id=1,
        )
        assert error.jsonrpc == "2.0"
        assert error.error.code == -32600
        assert error.error.message == "Invalid request"
        assert error.id == 1

    def test_error_with_null_id(self):
        """Error response with null ID (e.g., parse error before ID is known)."""
        error = JsonRpcError(
            error=ErrorDetail(code=-32700, message="Parse error"),
            id=None,
        )
        assert error.id is None

    def test_error_with_data(self):
        """Error response carrying extra data."""
        error = JsonRpcError(
            error=ErrorDetail(
                code=-32601,
                message="Method not found",
                data={"method": "foo.bar"},
            ),
            id=3,
        )
        assert error.error.data == {"method": "foo.bar"}

    def test_serialization_roundtrip(self):
        """Error response survives JSON roundtrip."""
        original = JsonRpcError(
            error=ErrorDetail(code=-32603, message="Internal error", data={"trace": "..."}),
            id=10,
        )
        json_bytes = original.model_dump_json()
        restored = JsonRpcError.model_validate_json(json_bytes)
        assert restored.error.code == original.error.code
        assert restored.error.message == original.error.message
        assert restored.error.data == original.error.data


class TestErrorDetail:
    """Tests for ErrorDetail model."""

    def test_minimal(self):
        """Code and message only."""
        detail = ErrorDetail(code=-32600, message="Bad request")
        assert detail.code == -32600
        assert detail.message == "Bad request"
        assert detail.data is None

    def test_with_data(self):
        """Include extra data dict."""
        detail = ErrorDetail(
            code=-32602,
            message="Invalid params",
            data={"field": "config_path", "reason": "missing"},
        )
        assert detail.data["field"] == "config_path"


class TestJsonRpcNotification:
    """Tests for JsonRpcNotification model."""

    def test_notification(self):
        """Server-sent notification has method but no id."""
        notif = JsonRpcNotification(
            method="log.line",
            params={"line": "Processing sheet 3..."},
        )
        assert notif.jsonrpc == "2.0"
        assert notif.method == "log.line"
        assert notif.params == {"line": "Processing sheet 3..."}

    def test_notification_no_params(self):
        """Heartbeat notification with no params."""
        notif = JsonRpcNotification(method="log.heartbeat")
        assert notif.params is None

    def test_serialization_roundtrip(self):
        """Notification survives JSON roundtrip."""
        original = JsonRpcNotification(
            method="log.line",
            params={"line": "done", "sheet": 5},
        )
        json_bytes = original.model_dump_json()
        restored = JsonRpcNotification.model_validate_json(json_bytes)
        assert restored.method == original.method
        assert restored.params == original.params


class TestJobStatusParams:
    """Tests for JobStatusParams model."""

    def test_basic(self):
        req = JobStatusParams(job_id="test-job", workspace=Path("/tmp/ws"))
        assert req.job_id == "test-job"
        assert req.workspace == Path("/tmp/ws")


class TestJobResumeParams:
    """Tests for JobResumeParams model."""

    def test_minimal(self):
        req = JobResumeParams(job_id="j1", workspace=Path("/tmp/ws"))
        assert req.reload_config is False
        assert req.self_healing is False

    def test_full(self):
        req = JobResumeParams(
            job_id="j1",
            workspace=Path("/tmp/ws"),
            config_path=Path("/tmp/new.yaml"),
            reload_config=True,
            self_healing=True,
            self_healing_auto_confirm=True,
        )
        assert req.config_path == Path("/tmp/new.yaml")
        assert req.reload_config is True
        assert req.self_healing_auto_confirm is True


class TestJobListParams:
    """Tests for JobListParams model."""

    def test_no_filter(self):
        params = JobListParams()
        assert params.workspace is None

    def test_with_workspace_filter(self):
        params = JobListParams(workspace=Path("/tmp/ws"))
        assert params.workspace == Path("/tmp/ws")


class TestJobLogsParams:
    """Tests for JobLogsParams model."""

    def test_defaults(self):
        params = JobLogsParams(job_id="j1", workspace=Path("/tmp/ws"))
        assert params.follow is False
        assert params.tail == 100

    def test_custom(self):
        params = JobLogsParams(
            job_id="j1", workspace=Path("/tmp/ws"), follow=True, tail=50
        )
        assert params.follow is True
        assert params.tail == 50


class TestShutdownParams:
    """Tests for ShutdownParams model."""

    def test_default_graceful(self):
        params = ShutdownParams()
        assert params.graceful is True

    def test_force_shutdown(self):
        params = ShutdownParams(graceful=False)
        assert params.graceful is False


class TestJobSummary:
    """Tests for JobSummary model."""

    def test_full(self):
        summary = JobSummary(
            job_id="test-job",
            status="running",
            total_sheets=10,
            completed_sheets=3,
            workspace="/tmp/ws",
            started_at="2025-01-01T00:00:00Z",
        )
        assert summary.job_id == "test-job"
        assert summary.status == "running"
        assert summary.total_sheets == 10
        assert summary.completed_sheets == 3

    def test_minimal(self):
        summary = JobSummary(
            job_id="j1",
            status="pending",
            total_sheets=5,
            completed_sheets=0,
            workspace="/tmp/ws",
        )
        assert summary.started_at is None
