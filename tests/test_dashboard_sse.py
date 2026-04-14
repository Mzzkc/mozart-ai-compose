"""Tests for SSE wire-format event dataclass."""

import json

from marianne.dashboard.services.sse_manager import SSEEvent


class TestSSEEvent:
    """Test SSE event formatting."""

    def test_simple_event_format(self):
        event = SSEEvent(event="test", data="hello world")
        formatted = event.format()

        expected = "event: test\ndata: hello world\n\n"
        assert formatted == expected

    def test_event_with_id_and_retry(self):
        event = SSEEvent(event="status", data="job started", id="test-123", retry=5000)
        formatted = event.format()

        expected = "id: test-123\nretry: 5000\nevent: status\ndata: job started\n\n"
        assert formatted == expected

    def test_multiline_data(self):
        event = SSEEvent(event="log", data="line 1\nline 2\nline 3")
        formatted = event.format()

        expected = "event: log\ndata: line 1\ndata: line 2\ndata: line 3\n\n"
        assert formatted == expected

    def test_json_data(self):
        data = {"status": "running", "progress": 50}
        event = SSEEvent(event="update", data=json.dumps(data))
        formatted = event.format()

        expected = 'event: update\ndata: {"status": "running", "progress": 50}\n\n'
        assert formatted == expected
