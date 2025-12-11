# Import main classes
from .plot_utils import PlotUtils
from .dashboard import InteractiveDashboard

# Import convenience functions
from .plot_utils import (
    quick_distribution,
    quick_correlation,
    quick_comparison
)

from .dashboard import create_quick_dashboard

# Define public API
__all__ = [
    # Main classes
    'PlotUtils',
    'InteractiveDashboard',
    'EDAVisualizer',
    'QuestionVisualizer',
    
    # Convenience functions
    'quick_distribution',
    'quick_correlation',
    'quick_comparison',
    'create_quick_dashboard',
    'quick_eda',
    'answer_all_questions',
]

# Version info
__version__ = '1.0.0'
__author__ = 'Clinical Trial Analysis Team'
__description__ = 'Comprehensive visualization module for clinical trial failure prediction'
