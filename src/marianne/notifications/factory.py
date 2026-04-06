"""Factory for creating notifiers from configuration.

Moved from marianne.cli.helpers to break the CLI dependency in the daemon package.
This is a notification-layer concern, not a CLI concern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from marianne.core.logging import get_logger

if TYPE_CHECKING:
    from marianne.core.config import NotificationConfig
    from marianne.notifications.base import Notifier

_logger = get_logger("notifications.factory")


def create_notifiers_from_config(
    notification_configs: list[NotificationConfig],
) -> list[Notifier]:
    """Create Notifier instances from notification configuration.

    Args:
        notification_configs: List of NotificationConfig from job config.

    Returns:
        List of configured Notifier instances.
    """
    from marianne.notifications.desktop import DesktopNotifier
    from marianne.notifications.slack import SlackNotifier
    from marianne.notifications.webhook import WebhookNotifier

    notifiers: list[Notifier] = []

    for config in notification_configs:
        notifier: Notifier | None = None
        # Cast Literal list to str list for from_config methods
        events: list[str] = list(config.on_events)

        if config.type == "desktop":
            notifier = DesktopNotifier.from_config(
                on_events=events,
                config=config.config,
            )
        elif config.type == "slack":
            notifier = SlackNotifier.from_config(
                on_events=events,
                config=config.config,
            )
        elif config.type == "webhook":
            notifier = WebhookNotifier.from_config(
                on_events=events,
                config=config.config,
            )
        else:
            _logger.warning("unknown_notification_type", type=config.type)
            continue

        if notifier:
            notifiers.append(notifier)

    return notifiers


__all__ = ["create_notifiers_from_config"]
