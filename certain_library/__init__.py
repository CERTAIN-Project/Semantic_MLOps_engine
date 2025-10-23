"""
Certain Library - MLflow Logging Utilities

A comprehensive Python library for logging machine learning experiments,
data analysis, and resource monitoring to MLflow.
"""

__version__ = "0.1.0"
__author__ = "dimitrioschristodoulou"
__email__ = "dimitrios.christodoulou@digital4planet.org"

# Import main modules for easy access
from . import data_analysis
from . import resource_monitor
from . import train_monitor

# Import commonly used functions
from .data_analysis.log_dataset import log_dataset
from .data_analysis.log_data_techniques import log_data_techniques
from .data_analysis.log_timeseries import timestamp_analysis

# Optional whylogs support (requires whylogs package)
try:
    from .data_analysis.log_whylogs import log_whylogs_profile
except ImportError:
    log_whylogs_profile = None

from .resource_monitor.resource import start_tracker, stop_tracker

from .train_monitor.log_metrics import log_metrics, log_search_space
from .train_monitor.log_model import (
    log_model_info,
    log_model_architecture,
    log_model_hyperparameters,
    log_model_signature,
)

__all__ = [
    # Data Analysis
    "log_dataset",
    "log_data_techniques",
    "timestamp_analysis",
    "log_whylogs_profile",
    # Resource Monitor
    "start_tracker",
    "stop_tracker",
    # Train Monitor
    "log_metrics",
    "log_search_space",
    "log_model_info",
    "log_model_architecture",
    "log_model_hyperparameters",
    "log_model_signature",
]
