"""
Real-time dashboard for QECC-aware QML monitoring.

Provides interactive web-based dashboard for monitoring training progress,
hardware status, error correction performance, and adaptive behavior.
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
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import threading
import time
import logging
import json
from datetime import datetime, timedelta

import dash
from dash import html, dcc, Input, Output, State, callback
import plotly.graph_objs as go
import plotly.express as px
from dash.dependencies import ALL

from .metrics_collector import MetricsCollector, MetricType
from .real_time_monitor import RealTimeMonitor
from .alerts import AlertManager, AlertSeverity


@dataclass
class DashboardConfig:
    """Configuration for QECC dashboard."""
    port: int = 8050
    host: str = "0.0.0.0"
    debug: bool = False
    update_interval: float = 1.0  # seconds
    history_length: int = 1000
    auto_refresh: bool = True
    theme: str = "plotly_dark"
    max_metrics_display: int = 20


class QECCDashboard:
    """
    Real-time dashboard for QECC-aware QML systems.
    
    Provides interactive monitoring of training progress, hardware metrics,
    error correction performance, and system alerts.
    """
    
    def __init__(
        self,
        config: Optional[DashboardConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize QECC dashboard.
        
        Args:
            config: Dashboard configuration
            logger: Optional logger instance
        """
        self.config = config or DashboardConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize components
        self.metrics_collector = MetricsCollector()
        self.real_time_monitor = RealTimeMonitor()
        self.alert_manager = AlertManager()
        
        # Dash app
        self.app = dash.Dash(__name__)
        self.app.title = "QECC-Aware QML Monitor"
        
        # State management
        self.is_running = False
        self.update_thread: Optional[threading.Thread] = None
        self.tracked_experiments: Dict[str, Any] = {}
        
        # Data storage
        self.metrics_history: Dict[str, List[Any]] = {}
        self.hardware_data: Dict[str, List[Any]] = {}
        self.adaptation_events: List[Dict[str, Any]] = []
        
        # Initialize dashboard layout and callbacks
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("🚀 QECC-Aware QML Monitor", className="header-title"),
                html.Div([
                    html.Span(id="live-status", className="status-indicator"),
                    html.Span(id="last-update", className="last-update")
                ], className="header-status")
            ], className="header"),
            
            # Control panel
            html.Div([
                html.Div([
                    html.Button("▶️ Start", id="start-btn", n_clicks=0, className="control-btn start"),
                    html.Button("⏸️ Pause", id="pause-btn", n_clicks=0, className="control-btn pause"),
                    html.Button("🗑️ Clear", id="clear-btn", n_clicks=0, className="control-btn clear"),
                ], className="control-buttons"),
                
                html.Div([
                    html.Label("Update Interval (s):"),
                    dcc.Input(id="update-interval", type="number", value=self.config.update_interval, 
                             min=0.1, max=10.0, step=0.1, className="control-input"),
                    html.Label("Max Points:"),
                    dcc.Input(id="max-points", type="number", value=self.config.history_length,
                             min=100, max=5000, step=100, className="control-input")
                ], className="control-settings")
            ], className="control-panel"),
            
            # Main content tabs
            dcc.Tabs(id="main-tabs", value="training", children=[
                # Training monitoring tab
                dcc.Tab(label="📈 Training", value="training", children=[
                    html.Div([
                        # Training metrics
                        html.Div([
                            html.H3("Training Metrics"),
                            html.Div([
                                html.Div([
                                    dcc.Graph(id="loss-plot", config={'displayModeBar': False})
                                ], className="metric-plot", style={"width": "50%"}),
                                html.Div([
                                    dcc.Graph(id="accuracy-plot", config={'displayModeBar': False})
                                ], className="metric-plot", style={"width": "50%"})
                            ], style={"display": "flex"}),
                            
                            html.Div([
                                html.Div([
                                    dcc.Graph(id="fidelity-plot", config={'displayModeBar': False})
                                ], className="metric-plot", style={"width": "50%"}),
                                html.Div([
                                    dcc.Graph(id="error-rate-plot", config={'displayModeBar': False})
                                ], className="metric-plot", style={"width": "50%"})
                            ], style={"display": "flex"})
                        ], className="metrics-section"),
                        
                        # Training statistics
                        html.Div([
                            html.H3("Current Statistics"),
                            html.Div(id="training-stats", className="stats-grid")
                        ], className="stats-section")
                    ])
                ]),
                
                # Hardware monitoring tab
                dcc.Tab(label="🔧 Hardware", value="hardware", children=[
                    html.Div([
                        # Hardware metrics
                        html.Div([
                            html.H3("Hardware Status"),
                            html.Div([
                                html.Div([
                                    dcc.Graph(id="gate-error-plot", config={'displayModeBar': False})
                                ], className="metric-plot", style={"width": "50%"}),
                                html.Div([
                                    dcc.Graph(id="coherence-plot", config={'displayModeBar': False})
                                ], className="metric-plot", style={"width": "50%"})
                            ], style={"display": "flex"}),
                            
                            html.Div([
                                html.Div([
                                    dcc.Graph(id="readout-error-plot", config={'displayModeBar': False})
                                ], className="metric-plot", style={"width": "50%"}),
                                html.Div([
                                    dcc.Graph(id="temperature-plot", config={'displayModeBar': False})
                                ], className="metric-plot", style={"width": "50%"})
                            ], style={"display": "flex"})
                        ], className="metrics-section"),
                        
                        # Hardware statistics
                        html.Div([
                            html.H3("Hardware Statistics"),
                            html.Div(id="hardware-stats", className="stats-grid")
                        ], className="stats-section")
                    ])
                ]),
                
                # Error correction tab
                dcc.Tab(label="🛡️ Error Correction", value="qecc", children=[
                    html.Div([
                        # QECC metrics
                        html.Div([
                            html.H3("Error Correction Performance"),
                            html.Div([
                                html.Div([
                                    dcc.Graph(id="logical-error-plot", config={'displayModeBar': False})
                                ], className="metric-plot", style={"width": "50%"}),
                                html.Div([
                                    dcc.Graph(id="syndrome-plot", config={'displayModeBar': False})
                                ], className="metric-plot", style={"width": "50%"})
                            ], style={"display": "flex"}),
                            
                            html.Div([
                                html.Div([
                                    dcc.Graph(id="decoder-performance-plot", config={'displayModeBar': False})
                                ], className="metric-plot", style={"width": "50%"}),
                                html.Div([
                                    dcc.Graph(id="code-distance-plot", config={'displayModeBar': False})
                                ], className="metric-plot", style={"width": "50%"})
                            ], style={"display": "flex"})
                        ], className="metrics-section"),
                        
                        # Adaptation events
                        html.Div([
                            html.H3("Adaptation Events"),
                            html.Div(id="adaptation-events", className="events-list")
                        ], className="events-section")
                    ])
                ]),
                
                # Alerts and system tab
                dcc.Tab(label="⚠️ Alerts", value="alerts", children=[
                    html.Div([
                        # Active alerts
                        html.Div([
                            html.H3("Active Alerts"),
                            html.Div(id="active-alerts", className="alerts-list")
                        ], className="alerts-section"),
                        
                        # Alert history
                        html.Div([
                            html.H3("Alert History"),
                            html.Div(id="alert-history", className="alerts-list")
                        ], className="alerts-section")
                    ])
                ])
            ]),
            
            # Auto-refresh component
            dcc.Interval(
                id="interval-component",
                interval=self.config.update_interval * 1000,  # milliseconds
                n_intervals=0,
                disabled=not self.config.auto_refresh
            ),
            
            # Store components for data
            dcc.Store(id="metrics-store", data={}),
            dcc.Store(id="hardware-store", data={}),
            dcc.Store(id="alerts-store", data=[])
            
        ], className="dashboard-container")
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            [Output("metrics-store", "data"),
             Output("hardware-store", "data"),
             Output("alerts-store", "data"),
             Output("live-status", "children"),
             Output("last-update", "children")],
            [Input("interval-component", "n_intervals")],
            prevent_initial_call=False
        )
        def update_data(n_intervals):
            """Update all data stores."""
            if not self.is_running:
                return {}, {}, [], "⭕ Stopped", "Not updating"
            
            # Collect current metrics
            current_metrics = self.metrics_collector.get_current_metrics()
            hardware_data = self._get_current_hardware_data()
            alerts_data = self.alert_manager.get_active_alerts()
            
            # Update history
            timestamp = datetime.now().isoformat()
            self._update_metrics_history(current_metrics, timestamp)
            self._update_hardware_history(hardware_data, timestamp)
            
            status = "🟢 Running" if self.is_running else "🔴 Stopped"
            last_update = f"Last update: {datetime.now().strftime('%H:%M:%S')}"
            
            return (
                self._format_metrics_for_store(),
                self._format_hardware_for_store(),
                alerts_data,
                status,
                last_update
            )
        
        @self.app.callback(
            [Output("loss-plot", "figure"),
             Output("accuracy-plot", "figure"),
             Output("fidelity-plot", "figure"),
             Output("error-rate-plot", "figure")],
            [Input("metrics-store", "data")]
        )
        def update_training_plots(metrics_data):
            """Update training metric plots."""
            return (
                self._create_loss_plot(metrics_data),
                self._create_accuracy_plot(metrics_data),
                self._create_fidelity_plot(metrics_data),
                self._create_error_rate_plot(metrics_data)
            )
        
        @self.app.callback(
            [Output("gate-error-plot", "figure"),
             Output("coherence-plot", "figure"),
             Output("readout-error-plot", "figure"),
             Output("temperature-plot", "figure")],
            [Input("hardware-store", "data")]
        )
        def update_hardware_plots(hardware_data):
            """Update hardware metric plots."""
            return (
                self._create_gate_error_plot(hardware_data),
                self._create_coherence_plot(hardware_data),
                self._create_readout_error_plot(hardware_data),
                self._create_temperature_plot(hardware_data)
            )
        
        @self.app.callback(
            Output("training-stats", "children"),
            [Input("metrics-store", "data")]
        )
        def update_training_stats(metrics_data):
            """Update training statistics display."""
            return self._create_training_stats_display(metrics_data)
        
        @self.app.callback(
            Output("active-alerts", "children"),
            [Input("alerts-store", "data")]
        )
        def update_active_alerts(alerts_data):
            """Update active alerts display."""
            return self._create_alerts_display(alerts_data)
    
    def _create_loss_plot(self, metrics_data):
        """Create loss plot."""
        fig = go.Figure()
        
        if 'loss' in metrics_data:
            timestamps = metrics_data.get('timestamps', [])
            loss_values = metrics_data['loss']
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=loss_values,
                mode='lines+markers',
                name='Training Loss',
                line=dict(color='#FF6B6B', width=2),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title="Training Loss",
            xaxis_title="Time",
            yaxis_title="Loss",
            template=self.config.theme,
            height=300,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig
    
    def _create_accuracy_plot(self, metrics_data):
        """Create accuracy plot."""
        fig = go.Figure()
        
        if 'accuracy' in metrics_data:
            timestamps = metrics_data.get('timestamps', [])
            accuracy_values = metrics_data['accuracy']
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=accuracy_values,
                mode='lines+markers',
                name='Training Accuracy',
                line=dict(color='#4ECDC4', width=2),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title="Training Accuracy",
            xaxis_title="Time",
            yaxis_title="Accuracy",
            template=self.config.theme,
            height=300,
            margin=dict(l=40, r=40, t=40, b=40),
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def _create_fidelity_plot(self, metrics_data):
        """Create fidelity plot."""
        fig = go.Figure()
        
        if 'fidelity' in metrics_data:
            timestamps = metrics_data.get('timestamps', [])
            fidelity_values = metrics_data['fidelity']
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=fidelity_values,
                mode='lines+markers',
                name='Circuit Fidelity',
                line=dict(color='#45B7D1', width=2),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title="Circuit Fidelity",
            xaxis_title="Time",
            yaxis_title="Fidelity",
            template=self.config.theme,
            height=300,
            margin=dict(l=40, r=40, t=40, b=40),
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def _create_error_rate_plot(self, metrics_data):
        """Create error rate plot."""
        fig = go.Figure()
        
        if 'logical_error_rate' in metrics_data:
            timestamps = metrics_data.get('timestamps', [])
            error_rates = metrics_data['logical_error_rate']
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=error_rates,
                mode='lines+markers',
                name='Logical Error Rate',
                line=dict(color='#F7931E', width=2),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title="Logical Error Rate",
            xaxis_title="Time",
            yaxis_title="Error Rate",
            template=self.config.theme,
            height=300,
            margin=dict(l=40, r=40, t=40, b=40),
            yaxis_type="log"
        )
        
        return fig
    
    def _create_gate_error_plot(self, hardware_data):
        """Create gate error rate plot."""
        fig = go.Figure()
        
        if 'gate_error_rate' in hardware_data:
            timestamps = hardware_data.get('timestamps', [])
            error_rates = hardware_data['gate_error_rate']
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=error_rates,
                mode='lines+markers',
                name='Gate Error Rate',
                line=dict(color='#FF6B6B', width=2),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title="Gate Error Rate",
            xaxis_title="Time",
            yaxis_title="Error Rate",
            template=self.config.theme,
            height=300,
            margin=dict(l=40, r=40, t=40, b=40),
            yaxis_type="log"
        )
        
        return fig
    
    def _create_coherence_plot(self, hardware_data):
        """Create coherence time plot."""
        fig = go.Figure()
        
        if 'coherence_t1' in hardware_data and 'coherence_t2' in hardware_data:
            timestamps = hardware_data.get('timestamps', [])
            t1_values = [t * 1e6 for t in hardware_data['coherence_t1']]  # Convert to microseconds
            t2_values = [t * 1e6 for t in hardware_data['coherence_t2']]
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=t1_values,
                mode='lines+markers',
                name='T1 (μs)',
                line=dict(color='#4ECDC4', width=2),
                marker=dict(size=4)
            ))
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=t2_values,
                mode='lines+markers',
                name='T2 (μs)',
                line=dict(color='#45B7D1', width=2),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title="Coherence Times",
            xaxis_title="Time",
            yaxis_title="Coherence Time (μs)",
            template=self.config.theme,
            height=300,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig
    
    def _create_readout_error_plot(self, hardware_data):
        """Create readout error plot."""
        fig = go.Figure()
        
        if 'readout_error_rate' in hardware_data:
            timestamps = hardware_data.get('timestamps', [])
            readout_errors = hardware_data['readout_error_rate']
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=readout_errors,
                mode='lines+markers',
                name='Readout Error Rate',
                line=dict(color='#F7931E', width=2),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title="Readout Error Rate",
            xaxis_title="Time",
            yaxis_title="Error Rate",
            template=self.config.theme,
            height=300,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig
    
    def _create_temperature_plot(self, hardware_data):
        """Create temperature plot."""
        fig = go.Figure()
        
        if 'temperature' in hardware_data:
            timestamps = hardware_data.get('timestamps', [])
            temperatures = hardware_data['temperature']
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=temperatures,
                mode='lines+markers',
                name='Temperature (K)',
                line=dict(color='#96CEB4', width=2),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title="System Temperature",
            xaxis_title="Time",
            yaxis_title="Temperature (K)",
            template=self.config.theme,
            height=300,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig
    
    def _create_training_stats_display(self, metrics_data):
        """Create training statistics display."""
        if not metrics_data:
            return html.Div("No training data available", className="no-data")
        
        stats = []
        
        # Current values
        for metric, values in metrics_data.items():
            if metric == 'timestamps' or not values:
                continue
            
            current_value = values[-1] if values else 0
            
            if metric == 'loss':
                stats.append(html.Div([
                    html.H4("Current Loss"),
                    html.P(f"{current_value:.6f}", className="stat-value")
                ], className="stat-card"))
            
            elif metric == 'accuracy':
                stats.append(html.Div([
                    html.H4("Current Accuracy"),
                    html.P(f"{current_value:.3f}", className="stat-value")
                ], className="stat-card"))
            
            elif metric == 'fidelity':
                stats.append(html.Div([
                    html.H4("Circuit Fidelity"),
                    html.P(f"{current_value:.3f}", className="stat-value")
                ], className="stat-card"))
        
        return stats
    
    def _create_alerts_display(self, alerts_data):
        """Create alerts display."""
        if not alerts_data:
            return html.Div("No active alerts", className="no-alerts")
        
        alerts = []
        for alert in alerts_data[-10:]:  # Show last 10 alerts
            severity_class = f"alert-{alert.get('severity', 'info').lower()}"
            
            alerts.append(html.Div([
                html.Div([
                    html.Span(alert.get('severity', 'INFO'), className="alert-severity"),
                    html.Span(alert.get('timestamp', ''), className="alert-timestamp")
                ], className="alert-header"),
                html.Div(alert.get('message', ''), className="alert-message")
            ], className=f"alert-card {severity_class}"))
        
        return alerts
    
    def _get_current_hardware_data(self):
        """Get current hardware data."""
        # This would interface with the hardware monitor
        return {
            'gate_error_rate': 0.001 + 0.0001 * np.random.randn(),
            'readout_error_rate': 0.01 + 0.001 * np.random.randn(),
            'coherence_t1': 50e-6 + 5e-6 * np.random.randn(),
            'coherence_t2': 70e-6 + 7e-6 * np.random.randn(),
            'temperature': 0.015 + 0.001 * np.random.randn()
        }
    
    def _update_metrics_history(self, metrics, timestamp):
        """Update metrics history."""
        for metric, value in metrics.items():
            if metric not in self.metrics_history:
                self.metrics_history[metric] = []
            
            self.metrics_history[metric].append((timestamp, value))
            
            # Keep history bounded
            if len(self.metrics_history[metric]) > self.config.history_length:
                self.metrics_history[metric] = self.metrics_history[metric][-self.config.history_length:]
    
    def _update_hardware_history(self, hardware_data, timestamp):
        """Update hardware history."""
        for metric, value in hardware_data.items():
            if metric not in self.hardware_data:
                self.hardware_data[metric] = []
            
            self.hardware_data[metric].append((timestamp, value))
            
            # Keep history bounded
            if len(self.hardware_data[metric]) > self.config.history_length:
                self.hardware_data[metric] = self.hardware_data[metric][-self.config.history_length:]
    
    def _format_metrics_for_store(self):
        """Format metrics data for Dash store."""
        formatted = {}
        
        for metric, history in self.metrics_history.items():
            if history:
                timestamps, values = zip(*history)
                formatted[metric] = list(values)
                if 'timestamps' not in formatted:
                    formatted['timestamps'] = list(timestamps)
        
        return formatted
    
    def _format_hardware_for_store(self):
        """Format hardware data for Dash store."""
        formatted = {}
        
        for metric, history in self.hardware_data.items():
            if history:
                timestamps, values = zip(*history)
                formatted[metric] = list(values)
                if 'timestamps' not in formatted:
                    formatted['timestamps'] = list(timestamps)
        
        return formatted
    
    def track_experiment(
        self,
        trainer,
        experiment_name: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ):
        """
        Track a training experiment.
        
        Args:
            trainer: QECC trainer instance
            experiment_name: Name for the experiment
            metrics: List of metrics to track
        """
        if experiment_name is None:
            experiment_name = f"experiment_{len(self.tracked_experiments)}"
        
        if metrics is None:
            metrics = ['loss', 'accuracy', 'fidelity', 'logical_error_rate']
        
        self.tracked_experiments[experiment_name] = {
            'trainer': trainer,
            'metrics': metrics,
            'start_time': datetime.now(),
            'status': 'active'
        }
        
        # Register trainer callback
        def training_callback(epoch, metrics_dict):
            self.metrics_collector.add_metrics(metrics_dict)
        
        # This would need to be implemented in the trainer
        # trainer.add_callback(training_callback)
        
        self.logger.info(f"Started tracking experiment: {experiment_name}")
    
    def start_monitoring(self):
        """Start the monitoring system."""
        self.is_running = True
        self.real_time_monitor.start()
        self.logger.info("Dashboard monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.is_running = False
        self.real_time_monitor.stop()
        self.logger.info("Dashboard monitoring stopped")
    
    def launch(self, **kwargs):
        """
        Launch the dashboard.
        
        Args:
            **kwargs: Additional arguments for Dash.run_server()
        """
        self.start_monitoring()
        
        run_kwargs = {
            'host': self.config.host,
            'port': self.config.port,
            'debug': self.config.debug
        }
        run_kwargs.update(kwargs)
        
        self.logger.info(f"Launching dashboard at http://{self.config.host}:{self.config.port}")
        
        try:
            self.app.run_server(**run_kwargs)
        except KeyboardInterrupt:
            self.logger.info("Dashboard shutdown requested")
        finally:
            self.stop_monitoring()
    
    def get_dashboard_url(self):
        """Get the dashboard URL."""
        return f"http://{self.config.host}:{self.config.port}"