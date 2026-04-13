"""Agent-to-Agent (A2A) protocol implementation.

Provides agent card registration, task inbox persistence, and event handling
for inter-agent task delegation mediated by the conductor.

Components:
- ``AgentCardRegistry`` — tracks which agents are running and what they can do
- ``A2AInbox`` — per-job persistent task queue for incoming A2A tasks

The conductor routes A2A events through the baton's event bus. Tasks persist
across sheet boundaries — when an agent sends a task, the conductor persists
it in the target's inbox, and the target picks it up on their next A2A-enabled
sheet.
"""

from marianne.daemon.a2a.inbox import A2AInbox
from marianne.daemon.a2a.registry import AgentCardRegistry

__all__ = [
    "A2AInbox",
    "AgentCardRegistry",
]
