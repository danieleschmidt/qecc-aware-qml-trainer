"""
Metrics collection system for QECC-aware QML monitoring.

Collects, processes, and manages metrics from training processes,
hardware monitors, and error correction systems.
"""

# Import with fallback support
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from qecc_qml.core.fallback_imports import create_fallback_implementations
    create_fallback_implementations()
except ImportError:
    pass
try:
    import numpy as np
except ImportError:
    import sys
    if 'numpy' in sys.modules:
        np = sys.modules['numpy']
    else:
        class MockNumPy:
            @staticmethod
            def array(x): return list(x) if isinstance(x, (list, tuple)) else x
            @staticmethod
            def zeros(shape): return [0] * (shape if isinstance(shape, int) else shape[0])
            @staticmethod  
            def ones(shape): return [1] * (shape if isinstance(shape, int) else shape[0])
            ndarray = list
        np = MockNumPy()
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
import logging
from collections import defaultdict, deque
import json


class MetricType(Enum):
    """Types of metrics that can be collected."""
    TRAINING = "training"
    HARDWARE = "hardware"
    ERROR_CORRECTION = "error_correction"
    PERFORMANCE = "performance"
    SYSTEM = "system"
    CUSTOM = "custom"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: Union[float, int, str]
    timestamp: float
    metric_type: MetricType
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricAggregate:
    """Aggregated metric statistics."""
    name: str
    count: int
    sum: float
    mean: float
    min: float
    max: float
    std: float
    last_value: float
    last_timestamp: float


class MetricsCollector:
    """
    Comprehensive metrics collection system.
    
    Collects metrics from various sources, performs real-time aggregation,
    and provides efficient access to current and historical data.
    """
    
    def __init__(
        self,
        max_history_per_metric: int = 10000,
        aggregation_window: int = 100,
        auto_compute_aggregates: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize metrics collector.
        
        Args:
            max_history_per_metric: Maximum number of data points per metric
            aggregation_window: Window size for rolling aggregations
            auto_compute_aggregates: Whether to automatically compute aggregates
            logger: Optional logger instance
        """
        self.max_history = max_history_per_metric
        self.aggregation_window = aggregation_window
        self.auto_compute_aggregates = auto_compute_aggregates
        self.logger = logger or logging.getLogger(__name__)
        
        # Data storage
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_per_metric))
        self.metric_metadata: Dict[str, Dict[str, Any]] = {}
        self.aggregates: Dict[str, MetricAggregate] = {}
        
        # Processing
        self.processors: Dict[str, List[Callable[[Metric], Metric]]] = defaultdict(list)
        self.filters: Dict[str, List[Callable[[Metric], bool]]] = defaultdict(list)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.total_metrics_collected = 0
        self.metrics_by_type: Dict[MetricType, int] = defaultdict(int)
        self.start_time = time.time()
    
    def add_metric(
        self,
        name: str,
        value: Union[float, int, str],
        metric_type: MetricType = MetricType.CUSTOM,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Add a single metric data point.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            timestamp: Timestamp (defaults to current time)
            metadata: Additional metadata
            tags: Tags for filtering/grouping
        """
        if timestamp is None:
            timestamp = time.time()
        
        metric = Metric(
            name=name,
            value=value,
            timestamp=timestamp,
            metric_type=metric_type,
            metadata=metadata or {},
            tags=tags or {}
        )
        
        self._process_metric(metric)
    
    def add_metrics(
        self,
        metrics_dict: Dict[str, Union[float, int, str]],
        metric_type: MetricType = MetricType.CUSTOM,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Add multiple metrics at once.
        
        Args:
            metrics_dict: Dictionary of metric name -> value
            metric_type: Type of metrics
            timestamp: Timestamp (defaults to current time)
            metadata: Additional metadata
            tags: Tags for filtering/grouping
        """
        if timestamp is None:
            timestamp = time.time()
        
        for name, value in metrics_dict.items():
            self.add_metric(name, value, metric_type, timestamp, metadata, tags)
    
    def _process_metric(self, metric: Metric):
        """Process a single metric through filters and processors."""
        with self.lock:
            # Apply filters
            if not self._should_accept_metric(metric):
                return
            
            # Apply processors
            processed_metric = self._apply_processors(metric)
            
            # Store metric
            self.metrics_history[metric.name].append(processed_metric)
            
            # Update metadata
            if metric.name not in self.metric_metadata:
                self.metric_metadata[metric.name] = {
                    'metric_type': metric.metric_type,
                    'first_seen': metric.timestamp,
                    'data_type': type(metric.value).__name__
                }
            
            self.metric_metadata[metric.name]['last_seen'] = metric.timestamp
            
            # Update statistics
            self.total_metrics_collected += 1
            self.metrics_by_type[metric.metric_type] += 1
            
            # Compute aggregates if enabled
            if self.auto_compute_aggregates:
                self._update_aggregate(processed_metric)
    
    def _should_accept_metric(self, metric: Metric) -> bool:
        """Check if metric passes all filters."""
        for filter_func in self.filters[metric.name] + self.filters['*']:
            if not filter_func(metric):
                return False
        return True
    
    def _apply_processors(self, metric: Metric) -> Metric:
        """Apply all processors to a metric."""
        processed = metric
        for processor in self.processors[metric.name] + self.processors['*']:
            processed = processor(processed)
        return processed
    
    def _update_aggregate(self, metric: Metric):
        """Update aggregate statistics for a metric."""
        if not isinstance(metric.value, (int, float)):
            return  # Can't aggregate non-numeric values
        
        name = metric.name
        
        if name not in self.aggregates:
            self.aggregates[name] = MetricAggregate(
                name=name,
                count=1,
                sum=float(metric.value),
                mean=float(metric.value),
                min=float(metric.value),
                max=float(metric.value),
                std=0.0,
                last_value=float(metric.value),
                last_timestamp=metric.timestamp
            )
        else:
            agg = self.aggregates[name]
            
            # Update statistics
            agg.count += 1
            agg.sum += metric.value
            agg.mean = agg.sum / agg.count
            agg.min = min(agg.min, metric.value)
            agg.max = max(agg.max, metric.value)
            agg.last_value = float(metric.value)
            agg.last_timestamp = metric.timestamp
            
            # Update standard deviation (online algorithm)
            if agg.count > 1:
                # Get recent values for std calculation
                recent_values = [m.value for m in list(self.metrics_history[name])[-self.aggregation_window:]]
                agg.std = float(np.std(recent_values)) if recent_values else 0.0
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current (most recent) values for all metrics."""
        with self.lock:
            current = {}
            for name, history in self.metrics_history.items():
                if history:
                    latest = history[-1]
                    current[name] = latest.value
            return current
    
    def get_metric_history(
        self,
        name: str,
        limit: Optional[int] = None,
        since: Optional[float] = None
    ) -> List[Metric]:
        """
        Get historical data for a specific metric.
        
        Args:
            name: Metric name
            limit: Maximum number of data points to return
            since: Only return data after this timestamp
            
        Returns:
            List of metric data points
        """
        with self.lock:
            if name not in self.metrics_history:
                return []
            
            history = list(self.metrics_history[name])
            
            # Filter by timestamp if specified
            if since is not None:
                history = [m for m in history if m.timestamp >= since]
            
            # Limit results if specified
            if limit is not None:
                history = history[-limit:]
            
            return history
    
    def get_metric_values(
        self,
        name: str,
        limit: Optional[int] = None,
        since: Optional[float] = None
    ) -> List[Union[float, int, str]]:
        """Get just the values for a metric (no metadata)."""
        history = self.get_metric_history(name, limit, since)
        return [m.value for m in history]
    
    def get_metric_aggregate(self, name: str) -> Optional[MetricAggregate]:
        """Get aggregate statistics for a metric."""
        with self.lock:
            return self.aggregates.get(name)
    
    def get_all_aggregates(self) -> Dict[str, MetricAggregate]:
        """Get aggregate statistics for all metrics."""
        with self.lock:
            return self.aggregates.copy()
    
    def add_processor(self, metric_name: str, processor: Callable[[Metric], Metric]):
        """
        Add a processor function for a metric.
        
        Args:
            metric_name: Name of metric to process (use '*' for all)
            processor: Function that takes a Metric and returns a Metric
        """
        self.processors[metric_name].append(processor)
        self.logger.info(f"Added processor for metric: {metric_name}")
    
    def add_filter(self, metric_name: str, filter_func: Callable[[Metric], bool]):
        """
        Add a filter function for a metric.
        
        Args:
            metric_name: Name of metric to filter (use '*' for all)
            filter_func: Function that takes a Metric and returns bool
        """
        self.filters[metric_name].append(filter_func)
        self.logger.info(f"Added filter for metric: {metric_name}")
    
    def get_metrics_by_type(self, metric_type: MetricType) -> Dict[str, List[Metric]]:
        """Get all metrics of a specific type."""
        with self.lock:
            result = {}
            for name, history in self.metrics_history.items():
                filtered_history = [m for m in history if m.metric_type == metric_type]
                if filtered_history:
                    result[name] = filtered_history
            return result
    
    def get_metrics_by_tags(self, tags: Dict[str, str]) -> Dict[str, List[Metric]]:
        """Get all metrics matching specific tags."""
        with self.lock:
            result = {}
            for name, history in self.metrics_history.items():
                filtered_history = []
                for metric in history:
                    if all(metric.tags.get(k) == v for k, v in tags.items()):
                        filtered_history.append(metric)
                if filtered_history:
                    result[name] = filtered_history
            return result
    
    def compute_correlation(self, metric1: str, metric2: str, window: Optional[int] = None) -> Optional[float]:
        """
        Compute correlation between two metrics.
        
        Args:
            metric1: First metric name
            metric2: Second metric name
            window: Number of recent points to use (None for all)
            
        Returns:
            Correlation coefficient or None if insufficient data
        """
        history1 = self.get_metric_values(metric1, limit=window)
        history2 = self.get_metric_values(metric2, limit=window)
        
        if len(history1) < 2 or len(history2) < 2 or len(history1) != len(history2):
            return None
        
        # Ensure we have numeric values
        try:
            values1 = [float(v) for v in history1]
            values2 = [float(v) for v in history2]
        except (ValueError, TypeError):
            return None
        
        return float(np.corrcoef(values1, values2)[0, 1])
    
    def detect_anomalies(
        self,
        metric_name: str,
        threshold: float = 3.0,
        window: int = 100
    ) -> List[Metric]:
        """
        Detect anomalies in a metric using z-score.
        
        Args:
            metric_name: Name of metric to analyze
            threshold: Z-score threshold for anomaly detection
            window: Window size for computing statistics
            
        Returns:
            List of anomalous data points
        """
        history = self.get_metric_history(metric_name, limit=window)
        
        if len(history) < 10:  # Need minimum data points
            return []
        
        # Extract numeric values
        try:
            values = [float(m.value) for m in history]
        except (ValueError, TypeError):
            return []  # Can't analyze non-numeric data
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return []  # No variation in data
        
        anomalies = []
        for metric, value in zip(history, values):
            z_score = abs((value - mean_val) / std_val)
            if z_score > threshold:
                anomalies.append(metric)
        
        return anomalies
    
    def export_data(
        self,
        format: str = "json",
        metrics: Optional[List[str]] = None,
        since: Optional[float] = None
    ) -> str:
        """
        Export metrics data.
        
        Args:
            format: Export format ("json", "csv")
            metrics: List of metrics to export (None for all)
            since: Only export data after this timestamp
            
        Returns:
            Exported data as string
        """
        with self.lock:
            export_data = {}
            
            metric_names = metrics or list(self.metrics_history.keys())
            
            for name in metric_names:
                if name in self.metrics_history:
                    history = self.get_metric_history(name, since=since)
                    export_data[name] = [
                        {
                            'timestamp': m.timestamp,
                            'value': m.value,
                            'type': m.metric_type.value,
                            'metadata': m.metadata,
                            'tags': m.tags
                        }
                        for m in history
                    ]
            
            if format.lower() == "json":
                return json.dumps(export_data, indent=2, default=str)
            elif format.lower() == "csv":
                # Simple CSV export (flattened)
                import csv
                import io
                
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Header
                writer.writerow(['metric_name', 'timestamp', 'value', 'type'])
                
                # Data
                for name, history in export_data.items():
                    for point in history:
                        writer.writerow([name, point['timestamp'], point['value'], point['type']])
                
                return output.getvalue()
            else:
                raise ValueError(f"Unsupported export format: {format}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collector statistics."""
        with self.lock:
            uptime = time.time() - self.start_time
            
            return {
                'total_metrics_collected': self.total_metrics_collected,
                'metrics_by_type': dict(self.metrics_by_type),
                'unique_metrics': len(self.metrics_history),
                'uptime_seconds': uptime,
                'average_metrics_per_second': self.total_metrics_collected / uptime if uptime > 0 else 0,
                'memory_usage': {
                    'total_data_points': sum(len(history) for history in self.metrics_history.values()),
                    'average_history_length': np.mean([len(history) for history in self.metrics_history.values()]) if self.metrics_history else 0
                }
            }
    
    def clear_history(self, metric_name: Optional[str] = None):
        """
        Clear metric history.
        
        Args:
            metric_name: Specific metric to clear (None for all)
        """
        with self.lock:
            if metric_name is None:
                # Clear all
                self.metrics_history.clear()
                self.aggregates.clear()
                self.metric_metadata.clear()
                self.total_metrics_collected = 0
                self.metrics_by_type.clear()
                self.start_time = time.time()
                self.logger.info("Cleared all metrics history")
            else:
                # Clear specific metric
                if metric_name in self.metrics_history:
                    del self.metrics_history[metric_name]
                if metric_name in self.aggregates:
                    del self.aggregates[metric_name]
                if metric_name in self.metric_metadata:
                    del self.metric_metadata[metric_name]
                self.logger.info(f"Cleared history for metric: {metric_name}")
    
    def __len__(self):
        """Return total number of unique metrics."""
        return len(self.metrics_history)
    
    def __contains__(self, metric_name):
        """Check if metric exists."""
        return metric_name in self.metrics_history