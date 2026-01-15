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

    def __init__(self) -> None:
        self._connections: dict[str, dict[str, ClientConnection]] = {}  # job_id -> {client_id -> conn}
        self._lock = asyncio.Lock()

    async def connect(self, job_id: str, client_id: str | None = None) -> AsyncIterator[str]:
        """Connect a client and yield SSE events."""
        if client_id is None:
            client_id = str(uuid.uuid4())

        logger.info(f"SSE client {client_id} connecting to job {job_id}")

        # Create connection
        connection = ClientConnection(client_id=client_id, job_id=job_id)

        async with self._lock:
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
                    event = await asyncio.wait_for(connection.queue.get(), timeout=30.0)
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
                    logger.info(f"SSE client {client_id} connection cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in SSE client {client_id}: {e}")
                    break

        finally:
            await self.disconnect(job_id, client_id)

    async def disconnect(self, job_id: str, client_id: str) -> None:
        """Disconnect a client."""
        logger.info(f"SSE client {client_id} disconnecting from job {job_id}")

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
                logger.debug(f"No SSE clients connected to job {job_id}")
                return 0

            # Copy connections to avoid holding lock during queue operations
            connections = dict(self._connections[job_id])

        for client_id, connection in connections.items():
            try:
                # Use put_nowait to avoid blocking; if queue is full, skip this client
                connection.queue.put_nowait(event)
                sent_count += 1
            except asyncio.QueueFull:
                logger.warning(f"SSE client {client_id} queue is full, skipping event")
                # Don't disconnect; client might catch up
            except Exception as e:
                logger.error(f"Error sending event to SSE client {client_id}: {e}")

        logger.debug(f"Broadcast event '{event.event}' to {sent_count} clients for job {job_id}")
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

        async with self._lock:
            # Cancel all queues to trigger connection cleanup
            for clients in self._connections.values():
                for client_id, connection in clients.items():
                    try:
                        # Send close event to gracefully notify clients
                        close_event = SSEEvent(
                            event="close",
                            data=json.dumps({"reason": "server_shutdown"})
                        )
                        connection.queue.put_nowait(close_event)
                    except asyncio.QueueFull:
                        pass  # Ignore if queue is full during shutdown
                    except Exception as e:
                        logger.debug(f"Error sending close event to {client_id}: {e}")

            # Clear all connections
            self._connections.clear()
