'''
Visualization Package - Clinical Trial Failure Prediction

This package provides visualization utilities for EDA and model evaluation

- plot_utils: Standalone functions for static plots
- dashboard: InteractiveDashboard class for interactive visualizations
'''

# Import main classes and modules
from .dashboard import InteractiveDashboard
from . import plot_utils

# Define public API
__all__ = [
    'InteractiveDashboard',
    'plot_utils',
]
