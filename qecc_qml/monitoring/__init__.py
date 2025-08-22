"""
Real-time monitoring and dashboard module.

Provides comprehensive monitoring capabilities for QECC-aware QML systems
including performance metrics, hardware status, and adaptive behavior tracking.
"""

try:
    from .dashboard import QECCDashboard, DashboardConfig
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    class QECCDashboard:
        def __init__(self, *args, **kwargs):
            raise ImportError("Dashboard requires 'dash' package: pip install dash")
    class DashboardConfig:
        pass

from .metrics_collector import MetricsCollector, MetricType
from .real_time_monitor import RealTimeMonitor, MonitoringEvent
from .alerts import AlertManager, AlertRule, AlertSeverity
from .health_monitor import HealthMonitor
from .comprehensive_health_monitor import ComprehensiveHealthMonitor, MonitoringConfig

__all__ = [
    "QECCDashboard",
    "DashboardConfig", 
    "MetricsCollector",
    "MetricType",
    "RealTimeMonitor",
    "MonitoringEvent",
    "AlertManager",
    "AlertRule",
    "AlertSeverity",
    "HealthMonitor",
    "ComprehensiveHealthMonitor",
    "MonitoringConfig",
]