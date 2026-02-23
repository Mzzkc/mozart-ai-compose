"""Profiler package — system resource collection, storage, and anomaly detection.

Collects per-process metrics, GPU stats, strace summaries, and stores
time-series data in SQLite for consumption by ``mozart top`` and the
anomaly/correlation analyzers.
"""

from mozart.daemon.profiler.anomaly import AnomalyDetector
from mozart.daemon.profiler.collector import ProfilerCollector
from mozart.daemon.profiler.correlation import CorrelationAnalyzer
from mozart.daemon.profiler.gpu_probe import GpuMetric, GpuProbe
from mozart.daemon.profiler.models import (
    Anomaly,
    AnomalyConfig,
    AnomalySeverity,
    AnomalyType,
    CorrelationConfig,
    EventType,
    ProcessEvent,
    ProcessMetric,
    ProfilerConfig,
    ResourceEstimate,
    RetentionConfig,
    SystemSnapshot,
)
from mozart.daemon.profiler.storage import MonitorStorage, generate_resource_report
from mozart.daemon.profiler.strace_manager import StraceManager

__all__ = [
    "Anomaly",
    "AnomalyConfig",
    "AnomalyDetector",
    "CorrelationAnalyzer",
    "AnomalySeverity",
    "AnomalyType",
    "CorrelationConfig",
    "EventType",
    "GpuMetric",
    "GpuProbe",
    "MonitorStorage",
    "generate_resource_report",
    "ProcessEvent",
    "ProcessMetric",
    "ProfilerCollector",
    "ProfilerConfig",
    "ResourceEstimate",
    "RetentionConfig",
    "StraceManager",
    "SystemSnapshot",
]
