"""Server-Sent Events manager for real-time updates."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from mozart.core.constants import SSE_QUEUE_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


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
        lines.append("")  # Final blank line
        return "\n".join(lines) + "\n"


@dataclass
class ClientConnection:
    """A connected SSE client."""
    client_id: str
    job_id: str
    queue: asyncio.Queue[SSEEvent] = field(default_factory=lambda: asyncio.Queue(maxsize=100))
    connected_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class SSEManager:
    """Manages SSE connections and broadcasts."""

    DEFAULT_MAX_CONNECTIONS = 100

    def __init__(self, max_connections: int = DEFAULT_MAX_CONNECTIONS) -> None:
        # job_id -> {client_id -> conn}
        self._connections: dict[str, dict[str, ClientConnection]] = {}
        self._lock = asyncio.Lock()
        self._max_connections = max_connections

    async def connect(self, job_id: str, client_id: str | None = None) -> AsyncIterator[str]:
        """Connect a client and yield SSE events."""
        if client_id is None:
            client_id = str(uuid.uuid4())

        logger.info("SSE client connecting", extra={"client_id": client_id, "job_id": job_id})

        # Create connection
        connection = ClientConnection(client_id=client_id, job_id=job_id)

        async with self._lock:
            # Enforce max connection limit to prevent resource exhaustion
            total = sum(len(clients) for clients in self._connections.values())
            if total >= self._max_connections:
                logger.warning(
                    "SSE connection rejected: max connections reached",
                    extra={"client_id": client_id, "job_id": job_id, "max": self._max_connections},
                )
                raise ConnectionError(
                    f"Maximum SSE connections ({self._max_connections}) reached"
                )
            if job_id not in self._connections:
                self._connections[job_id] = {}
            self._connections[job_id][client_id] = connection

        try:
            # Send initial connection event
            initial_event = SSEEvent(
                event="connected",
                data=json.dumps({"client_id": client_id, "job_id": job_id}),
                id=f"conn-{datetime.now(UTC).isoformat()}"
            )
            yield initial_event.format()

            # Yield events from the queue
            while True:
                try:
                    # Wait for event with timeout to allow graceful shutdown
                    event = await asyncio.wait_for(
                        connection.queue.get(), timeout=SSE_QUEUE_TIMEOUT_SECONDS
                    )
                    yield event.format()
                    connection.queue.task_done()
                except TimeoutError:
                    # Send heartbeat to keep connection alive
                    heartbeat = SSEEvent(
                        event="heartbeat",
                        data=json.dumps({"timestamp": datetime.now(UTC).isoformat()}),
                        retry=30000
                    )
                    yield heartbeat.format()
                except asyncio.CancelledError:
                    logger.info("SSE client connection cancelled", extra={"client_id": client_id})
                    break
                except Exception:
                    logger.exception("Error in SSE event loop", extra={"client_id": client_id})
                    break

        finally:
            await self.disconnect(job_id, client_id)

    async def disconnect(self, job_id: str, client_id: str) -> None:
        """Disconnect a client."""
        logger.info("SSE client disconnecting", extra={"client_id": client_id, "job_id": job_id})

        async with self._lock:
            if job_id in self._connections:
                self._connections[job_id].pop(client_id, None)
                # Clean up empty job entries
                if not self._connections[job_id]:
                    del self._connections[job_id]

    async def broadcast(self, job_id: str, event: SSEEvent) -> int:
        """Broadcast an event to all clients subscribed to a job.

        Returns the number of clients that received the event.
        """
        sent_count = 0

        async with self._lock:
            if job_id not in self._connections:
                logger.debug("No SSE clients connected", extra={"job_id": job_id})
                return 0

            # Copy connections to avoid holding lock during queue operations
            connections = dict(self._connections[job_id])

        for client_id, connection in connections.items():
            try:
                # Use put_nowait to avoid blocking; if queue is full, skip this client
                connection.queue.put_nowait(event)
                sent_count += 1
            except asyncio.QueueFull:
                logger.warning("SSE client queue full, skipping event", extra={"client_id": client_id})
                # Don't disconnect; client might catch up
            except Exception:
                logger.exception("Error sending SSE event", extra={"client_id": client_id})

        logger.debug("Broadcast SSE event", extra={"event": event.event, "sent_count": sent_count, "job_id": job_id})
        return sent_count

    async def send_job_update(self, job_id: str, status: str, data: dict[str, Any]) -> int:
        """Convenience method to send job status update."""
        event_data = {
            "status": status,
            "timestamp": datetime.now(UTC).isoformat(),
            **data
        }

        event = SSEEvent(
            event="job_status",
            data=json.dumps(event_data),
            id=f"status-{job_id}-{datetime.now(UTC).timestamp()}"
        )

        return await self.broadcast(job_id, event)

    async def send_log_line(self, job_id: str, line: str, level: str = "info") -> int:
        """Convenience method to send log line."""
        event_data = {
            "line": line.rstrip('\n'),
            "level": level,
            "timestamp": datetime.now(UTC).isoformat()
        }

        event = SSEEvent(
            event="log",
            data=json.dumps(event_data),
            id=f"log-{job_id}-{datetime.now(UTC).timestamp()}"
        )

        return await self.broadcast(job_id, event)

    def get_connection_count(self, job_id: str | None = None) -> int:
        """Get number of connected clients."""
        if job_id is None:
            # Return total across all jobs
            return sum(len(clients) for clients in self._connections.values())
        else:
            return len(self._connections.get(job_id, {}))

    async def get_connected_jobs(self) -> list[str]:
        """Get list of job IDs that have active connections."""
        async with self._lock:
            return list(self._connections.keys())

    async def close_all_connections(self) -> None:
        """Close all active connections (for shutdown)."""
        logger.info("Closing all SSE connections")

        # Snapshot and clear under lock, then notify outside the lock
        async with self._lock:
            all_connections = dict(self._connections)
            self._connections.clear()

        # Send close events without holding the lock
        close_event = SSEEvent(
            event="close",
            data=json.dumps({"reason": "server_shutdown"})
        )
        for clients in all_connections.values():
            for client_id, connection in clients.items():
                try:
                    connection.queue.put_nowait(close_event)
                except asyncio.QueueFull:
                    pass  # Ignore if queue is full during shutdown
                except Exception:
                    logger.debug("Error sending close event", extra={"client_id": client_id}, exc_info=True)
