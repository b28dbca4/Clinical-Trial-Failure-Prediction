"""
Data Module for Clinical Trial Failure Prediction

1. Introduction

1.1 Purpose
This module provides a comprehensive data processing pipeline for the Clinical Trial Failure Prediction project. It includes utilities for data loading, cleaning, feature engineering, and validation to support machine learning models predicting clinical trial outcomes.

1.2 Scope
The module encompasses the following components:
- Data Loader: Handles API integration and data collection from clinical trial databases.
- Data Cleaner: Performs data cleaning, standardization, and preprocessing.
- Feature Engineering: Creates advanced features for model training.
- Data Validation: Ensures data quality and integrity.

1.3 Definitions, Acronyms, and Abbreviations
- API: Application Programming Interface
- SRS: Software Requirements Specification
- ML: Machine Learning

1.4 References
- IEEE 830-1998: Recommended Practice for Software Requirements Specifications
- Project README.md for overall project documentation

1.5 Overview
This __init__.py file initializes the data package, providing lazy-loaded imports for efficient module usage. It exposes key functions and classes from submodules while maintaining a clean API.

2. Overall Description

2.1 Product Perspective
This module is part of a larger data science project focused on predicting clinical trial failures using historical data.

2.2 Product Functions
- Load and preprocess clinical trial data
- Engineer features for predictive modeling
- Validate data quality

2.3 User Characteristics
Intended for data scientists and researchers working on clinical trial analysis.

2.4 Constraints
- Requires Python 3.8+
- Depends on external libraries listed in requirements.txt

3. Specific Requirements

3.1 External Interface Requirements
- Imports from submodules: data_loader, data_cleaner, feature_engineering

3.2 Functional Requirements
- Lazy loading of module components to optimize import time
- Logging setup for debugging and monitoring
"""

from __future__ import annotations

# Public symbol lists (used for lazy imports)
_data_loader_exports = [
    "build_query_params",
    "api_smoke_test",
    "flatten_study",
    "get_flat_columns",
    "iter_studies_from_api",
]

_data_cleaner_exports = [
    "PreprocessConfig",
    "load_raw_data",
    "standardize_dtypes",
    "analyze_missing",
    "sanity_checks",
    "split_data",
    "create_data_dictionary",
]

# Feature engineering module may be empty; import lazily if available
try:
    from . import feature_engineering as feature_engineering
except Exception:
    feature_engineering = None

__all__ = [
    # data_loader
    "build_query_params",
    "api_smoke_test",
    "flatten_study",
    "get_flat_columns",
    "iter_studies_from_api",

    # data_cleaner
    "PreprocessConfig",
    "load_raw_data",
    "standardize_dtypes",
    "analyze_missing",
    "sanity_checks",
    "split_data",
    "create_data_dictionary",
]