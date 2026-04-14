"""Server-Sent Events wire-format dataclass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SSEEvent:
    """An SSE event to be sent to clients."""

    event: str
    data: str
    id: str | None = None
    retry: int | None = None

    def format(self) -> str:
        """Format as SSE wire format."""
        lines = []
        if self.id:
            lines.append(f"id: {self.id}")
        if self.retry:
            lines.append(f"retry: {self.retry}")
        lines.append(f"event: {self.event}")
        for line in self.data.split("\n"):
            lines.append(f"data: {line}")
        lines.append("")
        return "\n".join(lines) + "\n"
