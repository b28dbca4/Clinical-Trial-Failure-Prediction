'''
Utils Package - Clinical Trial Failure Prediction

This package provides configuration and utility functions for the project

Modules:
- config: Project paths, constants, and configuration settings
'''

from .config import (
    # Paths
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    FINAL_DIR,
    MODELING_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    EDA_FIGURES_DIR,
    MODEL_FIGURES_DIR,
    # Label definitions
    SUCCESS_STATUSES,
    FAIL_STATUSES,
    # EDA configuration
    SEED,
    DEFAULT_FIGSIZE,
    DEFAULT_DPI,
    OUTCOME_COLORS,
    SPONSOR_COLORS,
    ALPHA_LEVEL,
    CRAMERS_V_THRESHOLDS,
    ENROLLMENT_BINS,
    ENROLLMENT_LABELS,
    # API configuration
    CTGOV_BASE_URL,
    # Classes
    CollectConfig,
)

__all__ = [
    # Paths
    'PROJECT_ROOT',
    'DATA_DIR',
    'RAW_DIR',
    'PROCESSED_DIR',
    'FINAL_DIR',
    'MODELING_DIR',
    'REPORTS_DIR',
    'FIGURES_DIR',
    'EDA_FIGURES_DIR',
    'MODEL_FIGURES_DIR',
    # Label definitions
    'SUCCESS_STATUSES',
    'FAIL_STATUSES',
    # EDA configuration
    'SEED',
    'DEFAULT_FIGSIZE',
    'DEFAULT_DPI',
    'OUTCOME_COLORS',
    'SPONSOR_COLORS',
    'ALPHA_LEVEL',
    'CRAMERS_V_THRESHOLDS',
    'ENROLLMENT_BINS',
    'ENROLLMENT_LABELS',
    # API configuration
    'CTGOV_BASE_URL',
    # Classes
    'CollectConfig',
]
