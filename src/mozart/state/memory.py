"""In-memory state backend for testing.

Provides a state backend that stores all state in memory without filesystem I/O.
Useful for unit tests that need a real StateBackend implementation.
"""

from mozart.core.checkpoint import CheckpointState, SheetStatus
from mozart.state.base import StateBackend


class InMemoryStateBackend(StateBackend):
    """In-memory state backend for testing.

    Tracks state changes in a dict without filesystem I/O.
    """

    def __init__(self) -> None:
        self.states: dict[str, CheckpointState] = {}

    async def load(self, job_id: str) -> CheckpointState | None:
        return self.states.get(job_id)

    async def save(self, state: CheckpointState) -> None:
        self.states[state.job_id] = state

    async def delete(self, job_id: str) -> bool:
        if job_id in self.states:
            del self.states[job_id]
            return True
        return False

    async def list_jobs(self) -> list[CheckpointState]:
        return list(self.states.values())

    async def get_next_sheet(self, job_id: str) -> int | None:
        state = await self.load(job_id)
        return state.get_next_sheet() if state else None

    async def mark_sheet_status(
        self,
        job_id: str,
        sheet_num: int,
        status: SheetStatus,
        error_message: str | None = None,
    ) -> None:
        state = await self.load(job_id)
        if state is None:
            return
        if status == SheetStatus.COMPLETED:
            state.mark_sheet_completed(sheet_num)
        elif status == SheetStatus.FAILED:
            state.mark_sheet_failed(sheet_num, error_message or "Unknown error")
        elif status == SheetStatus.IN_PROGRESS:
            state.mark_sheet_started(sheet_num)
        await self.save(state)
