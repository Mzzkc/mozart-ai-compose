"""System summary header bar for the Mozart Monitor TUI.

Displays conductor status, memory bar, CPU bar, GPU indicator,
pressure level, and job/sheet counts in a compact single-line format.
"""

from __future__ import annotations

from textual.widgets import Static

from mozart.daemon.profiler.models import SystemSnapshot


def _bar(pct: float, width: int = 10) -> str:
    """Render a simple bar like ████░░░░░░ from a 0-100 percentage."""
    filled = int(round(pct / 100.0 * width))
    filled = max(0, min(width, filled))
    return "\u2588" * filled + "\u2591" * (width - filled)


def _pressure_color(level: str) -> str:
    """Return a Rich color name for the given pressure level."""
    level_lower = level.lower()
    if level_lower in ("critical", "high"):
        return "red"
    elif level_lower == "medium":
        return "yellow"
    return "green"


def _format_uptime(seconds: float) -> str:
    """Format seconds into a human-readable uptime string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h{mins:02d}m"


class HeaderPanel(Static):
    """Renders the system summary header with Rich markup.

    Layout matches the design doc header bar:
    ``● Conductor: UP 2h15m  Memory: ██░░ 30%  CPU: █░░░ 12%  GPU: —``
    ``Pressure: LOW  Jobs: 2/4  Sheets: 3 active  ⚠ 1 anomaly``
    """

    DEFAULT_CSS = """
    HeaderPanel {
        height: 2;
        dock: top;
        background: $surface;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._snapshot: SystemSnapshot | None = None
        self._conductor_up: bool = False
        self._uptime_seconds: float = 0.0
        self._anomaly_count: int = 0

    def update_data(
        self,
        snapshot: SystemSnapshot | None,
        conductor_up: bool = False,
        uptime_seconds: float = 0.0,
        anomaly_count: int = 0,
    ) -> None:
        """Update the header with new snapshot data and refresh display."""
        self._snapshot = snapshot
        self._conductor_up = conductor_up
        self._uptime_seconds = uptime_seconds
        self._anomaly_count = anomaly_count
        self._render_header()

    def _render_header(self) -> None:
        """Render the header content with Rich markup."""
        snap = self._snapshot

        # Line 1: Conductor status, memory, CPU, GPU
        if self._conductor_up:
            uptime = _format_uptime(self._uptime_seconds)
            status = f"[green]●[/] Conductor: [bold green]UP[/] {uptime}"
        else:
            status = "[red]●[/] Conductor: [bold red]DOWN[/]"

        if snap is not None:
            mem_total = snap.system_memory_total_mb
            mem_used = snap.system_memory_used_mb
            mem_pct = (mem_used / mem_total * 100.0) if mem_total > 0 else 0.0
            mem_str = f"Memory: {_bar(mem_pct, 4)} {mem_pct:.0f}%"

            # CPU: sum of per-process CPU (capped at 100 for display)
            total_cpu = sum(p.cpu_percent for p in snap.processes)
            cpu_str = f"CPU: {_bar(min(total_cpu, 100), 4)} {total_cpu:.0f}%"

            # GPU
            if snap.gpus:
                gpu_util = max(g.utilization_pct for g in snap.gpus)
                gpu_str = f"GPU: {gpu_util:.0f}%"
            else:
                gpu_str = "GPU: \u2014"
        else:
            mem_str = "Memory: \u2014"
            cpu_str = "CPU: \u2014"
            gpu_str = "GPU: \u2014"

        line1 = f"{status}  {mem_str}  {cpu_str}  {gpu_str}"

        # Line 2: Pressure, jobs, sheets, anomalies
        if snap is not None:
            pressure = snap.pressure_level.upper()
            p_color = _pressure_color(snap.pressure_level)
            pressure_str = f"Pressure: [{p_color}]{pressure}[/]"
            jobs_str = f"Jobs: {snap.running_jobs}"
            sheets_str = f"Sheets: {snap.active_sheets} active"
        else:
            pressure_str = "Pressure: \u2014"
            jobs_str = "Jobs: \u2014"
            sheets_str = "Sheets: \u2014"

        anomaly_str = ""
        if self._anomaly_count > 0:
            noun = "anomalies" if self._anomaly_count != 1 else "anomaly"
            anomaly_str = f"  [bold yellow]\u26a0 {self._anomaly_count} {noun}[/]"

        line2 = f"{pressure_str}  {jobs_str}  {sheets_str}{anomaly_str}"

        self.update(f"{line1}\n{line2}")


__all__ = ["HeaderPanel"]
