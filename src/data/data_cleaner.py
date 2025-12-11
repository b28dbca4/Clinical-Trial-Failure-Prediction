# src/data/data_cleaner.py
"""
Data Cleaning & Preprocessing Module for Clinical Trial Failure Prediction

This module handles:
1. Cohort filtering (Interventional + Phase 2-3 + Cancer)
2. Label creation (Fail=1, Success=0)
3. Anti-leakage protection (drop post-t0 columns)
4. Data type standardization (date, numeric, boolean, categorical)
5. Missing value analysis & handling (impute, drop, indicator)
6. Data splitting (stratified or time-based)
7. Reproducibility artifacts generation

Author: Data Science Team
Date: December 2025
Version: 2.0 - Enhanced with dtype standardization and missing handling
"""

from __future__ import annotations

import json
import hashlib
import logging
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Setup logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# =============================================================================
# Data Type Schema Definition
# =============================================================================

DTYPE_SCHEMA = {
    # Date columns → datetime64[ns]
    "date_columns": [
        "start_date", "completion_date", "primary_completion_date",
        "study_first_post_date", "last_update_post_date", "results_first_post_date"
    ],
    
    # Numeric columns → float64 (with coercion)
    "numeric_columns": [
        "enrollment_count", "num_arms", "num_interventions", 
        "num_conditions", "locations_count", "outcomes_count",
        "min_age_years", "max_age_years", "number_of_arms", "number_of_groups"
    ],
    
    # Boolean columns → bool (with mapping)
    "boolean_columns": [
        "has_results", "has_dmc", "fda_regulated_drug", "fda_regulated_device",
        "is_fda_regulated_drug", "is_fda_regulated_device"
    ],
    
    # Categorical columns → string (strip, upper)
    "categorical_columns": [
        "study_type", "phases", "lead_sponsor_class", "allocation",
        "intervention_model", "masking", "gender", "sampling_method",
        "overall_status", "primary_purpose"
    ],
    
    # Text columns → string (strip, preserve case)
    "text_columns": [
        "brief_title", "official_title", "conditions", "interventions",
        "lead_sponsor_name", "why_stopped", "eligibility_criteria"
    ]
}

# Missing tokens to standardize
MISSING_TOKENS = [
    '', ' ', 'NA', 'N/A', 'na', 'n/a', 'NULL', 'null', 'Null',
    'None', 'none', 'NONE', 'Unknown', 'unknown', 'UNKNOWN',
    'Not Available', 'Not Provided', 'Not Applicable',
    '--', '-', '.', 'NaN', 'nan', 'NAN'
]

# Boolean mapping
BOOL_MAP = {
    True: True, False: False,
    "TRUE": True, "FALSE": False,
    "true": True, "false": False,
    "True": True, "False": False,
    "1": True, "0": False,
    1: True, 0: False,
    1.0: True, 0.0: False,
    "Yes": True, "No": False,
    "yes": True, "no": False,
    "Y": True, "N": False,
    "y": True, "n": False
}


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class PreprocessConfig:
    """Configuration for preprocessing pipeline."""
    
    # Cohort filters
    study_type: str = "INTERVENTIONAL"
    phases: List[str] = field(default_factory=lambda: ["PHASE2", "PHASE3", "PHASE2|PHASE3"])
    cancer_keywords: List[str] = field(default_factory=lambda: [
        "cancer", "tumor", "tumour", "carcinoma", "neoplasm", 
        "malignant", "oncology", "lymphoma", "leukemia", "melanoma"
    ])
    
    # Label definition
    fail_statuses: List[str] = field(default_factory=lambda: ["WITHDRAWN", "TERMINATED"])
    success_statuses: List[str] = field(default_factory=lambda: ["COMPLETED"])
    
    # Anti-leakage: columns that are post-t0 (t0 = start_date)
    post_t0_columns: List[str] = field(default_factory=lambda: [
        "completion_date", "primary_completion_date", "why_stopped",
        "results_first_post_date", "last_update_post_date", 
        "study_first_post_date", "has_results", "overall_status"
    ])
    
    # Columns to keep for QC but not for modeling
    qc_only_columns: List[str] = field(default_factory=lambda: [
        "completion_date", "has_results", "overall_status"
    ])
    
    # Core columns (must have low missing rate)
    core_columns: List[str] = field(default_factory=lambda: [
        "nct_id", "start_date", "study_type", "phases", 
        "enrollment_count", "lead_sponsor_class"
    ])
    
    # Split configuration
    test_size: float = 0.2
    val_size: float = 0.1  # From remaining after test
    random_seed: int = 42
    split_method: str = "stratified"  # "stratified" or "time"
    time_split_date: Optional[str] = None  # For time-based split
    
    # Paths
    output_dir: Path = field(default_factory=lambda: Path("data/processed/modeling"))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        d = asdict(self)
        d["output_dir"] = str(d["output_dir"])
        return d


@dataclass
class PreprocessingReport:
    """Report of preprocessing steps and outcomes."""
    
    timestamp: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Counts at each stage
    raw_count: int = 0
    after_interventional: int = 0
    after_phase_filter: int = 0
    after_cancer_filter: int = 0
    after_status_filter: int = 0
    final_count: int = 0
    
    # Label distribution
    label_counts: Dict[str, int] = field(default_factory=dict)
    label_rates: Dict[str, float] = field(default_factory=dict)
    
    # Missing data summary
    columns_dropped_missing: List[str] = field(default_factory=list)
    missing_rates: Dict[str, float] = field(default_factory=dict)
    
    # Anti-leakage
    columns_dropped_leakage: List[str] = field(default_factory=list)
    
    # Split info
    train_size: int = 0
    val_size: int = 0
    test_size: int = 0
    train_label_ratio: float = 0.0
    val_label_ratio: float = 0.0
    test_label_ratio: float = 0.0
    
    # Sanity checks
    date_anomalies: int = 0
    enrollment_outliers: int = 0
    duplicate_nct_ids: int = 0
    
    # File paths
    output_files: Dict[str, str] = field(default_factory=dict)
    data_checksum: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: Path) -> None:
        """Save report to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved to {path}")


# =============================================================================
# Core Preprocessing Functions
# =============================================================================

def load_raw_data(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load raw CSV data with proper type handling.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame with loaded data
    """
    logger.info(f"Loading data from {csv_path}")
    
    df = pd.read_csv(csv_path, low_memory=False)
    
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def standardize_missing_tokens(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize all missing token variants to NaN.
    
    Converts '', 'NA', 'N/A', 'null', 'None', 'Unknown', etc. to NaN.
    
    Args:
        df: Input DataFrame
        
    Returns:
        (df_standardized, conversion_report)
    """
    df = df.copy()
    conversion_log = []
    
    for col in df.columns:
        if df[col].dtype == 'object':
            before_null = df[col].isna().sum()
            
            # Strip whitespace
            df[col] = df[col].astype(str).str.strip()
            
            # Count tokens that will become NaN
            token_mask = df[col].isin(MISSING_TOKENS)
            tokens_found = token_mask.sum()
            
            # Replace missing tokens with NaN
            df[col] = df[col].replace(MISSING_TOKENS, np.nan)
            df[col] = df[col].replace('nan', np.nan)
            
            after_null = df[col].isna().sum()
            
            if tokens_found > 0:
                conversion_log.append({
                    "column": col,
                    "missing_before": int(before_null),
                    "tokens_converted": int(tokens_found),
                    "missing_after": int(after_null)
                })
    
    report_df = pd.DataFrame(conversion_log) if conversion_log else pd.DataFrame()
    
    if len(conversion_log) > 0:
        total_tokens = sum([x["tokens_converted"] for x in conversion_log])
        logger.info(f"Missing token standardization: {total_tokens:,} tokens → NaN across {len(conversion_log)} columns")
    
    return df, report_df


def standardize_dtypes(
    df: pd.DataFrame,
    schema: Optional[Dict[str, List[str]]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize column data types with coercion report.
    
    Args:
        df: Input DataFrame
        schema: Schema definition (uses DTYPE_SCHEMA if None)
        
    Returns:
        (df_standardized, dtype_report_df)
    """
    if schema is None:
        schema = DTYPE_SCHEMA
    
    df = df.copy()
    conversion_log = []
    
    # 1. Date columns → datetime64[ns]
    for col in schema.get("date_columns", []):
        if col in df.columns:
            before_dtype = str(df[col].dtype)
            before_null = df[col].isna().sum()
            
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
            after_null = df[col].isna().sum()
            coerced = after_null - before_null
            
            conversion_log.append({
                "column": col,
                "category": "date",
                "dtype_before": before_dtype,
                "dtype_after": str(df[col].dtype),
                "coerced_to_null": int(max(0, coerced))
            })
    
    # 2. Numeric columns → float64
    for col in schema.get("numeric_columns", []):
        if col in df.columns:
            before_dtype = str(df[col].dtype)
            before_null = df[col].isna().sum()
            
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            after_null = df[col].isna().sum()
            coerced = after_null - before_null
            
            conversion_log.append({
                "column": col,
                "category": "numeric",
                "dtype_before": before_dtype,
                "dtype_after": str(df[col].dtype),
                "coerced_to_null": int(max(0, coerced))
            })
    
    # 3. Boolean columns → bool
    for col in schema.get("boolean_columns", []):
        if col in df.columns:
            before_dtype = str(df[col].dtype)
            before_null = df[col].isna().sum()
            
            df[col] = df[col].map(BOOL_MAP)
            
            after_null = df[col].isna().sum()
            
            conversion_log.append({
                "column": col,
                "category": "boolean",
                "dtype_before": before_dtype,
                "dtype_after": str(df[col].dtype),
                "coerced_to_null": int(after_null - before_null) if after_null > before_null else 0
            })
    
    # 4. Categorical columns → string (strip, upper)
    for col in schema.get("categorical_columns", []):
        if col in df.columns:
            before_dtype = str(df[col].dtype)
            
            df[col] = df[col].astype(str).str.strip().str.upper()
            df[col] = df[col].replace(["NAN", "NONE", ""], pd.NA)
            
            conversion_log.append({
                "column": col,
                "category": "categorical",
                "dtype_before": before_dtype,
                "dtype_after": str(df[col].dtype),
                "coerced_to_null": 0
            })
    
    # 5. Text columns → string (strip, preserve case)
    for col in schema.get("text_columns", []):
        if col in df.columns:
            before_dtype = str(df[col].dtype)
            
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(["nan", "None", ""], pd.NA)
            
            conversion_log.append({
                "column": col,
                "category": "text",
                "dtype_before": before_dtype,
                "dtype_after": str(df[col].dtype),
                "coerced_to_null": 0
            })
    
    report_df = pd.DataFrame(conversion_log) if conversion_log else pd.DataFrame()
    logger.info(f"Dtype standardization complete: {len(conversion_log)} columns processed")
    
    return df, report_df


def filter_cohort(
    df: pd.DataFrame, 
    cfg: PreprocessConfig,
    return_attrition: bool = True
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Apply cohort filters: Interventional + Phase 2-3 + Cancer keywords.
    
    Args:
        df: Input DataFrame
        cfg: Preprocessing configuration
        return_attrition: Whether to return attrition counts
        
    Returns:
        Filtered DataFrame and attrition dictionary
    """
    attrition = {"raw": len(df)}
    
    # Normalize phases column first
    if "phases" in df.columns:
        df["phases"] = df["phases"].astype(str).str.strip().str.upper()
    
    # Step 1: Filter by study type (INTERVENTIONAL)
    if "study_type" in df.columns:
        mask_interventional = df["study_type"].str.upper().str.contains(
            cfg.study_type, na=False, regex=False
        )
        df = df[mask_interventional].copy()
    attrition["interventional"] = len(df)
    logger.info(f"After INTERVENTIONAL filter: {len(df):,} rows")
    
    # Step 2: Filter by phase (PHASE2 or PHASE3)
    if "phases" in df.columns:
        # Escape special regex characters and create pattern
        phases_escaped = [re.escape(p) for p in cfg.phases]
        phase_pattern = "|".join(phases_escaped)
        mask_phase = df["phases"].str.contains(
            phase_pattern, na=False, regex=True
        )
        df = df[mask_phase].copy()
    attrition["phase_2_3"] = len(df)
    logger.info(f"After Phase 2-3 filter: {len(df):,} rows")
    
    # Step 3: Filter by cancer keywords in conditions or title
    # Escape keywords for regex safety
    keywords_escaped = [re.escape(kw) for kw in cfg.cancer_keywords]
    cancer_pattern = "|".join(keywords_escaped)
    mask_cancer = pd.Series(False, index=df.index)
    
    cancer_match_counts = {}

    for col in ["conditions", "brief_title"]:
        if col in df.columns:
            mask_cancer |= df[col].str.lower().str.contains(
                cancer_pattern, na=False, regex=True
            )
    
    df = df[mask_cancer].copy()
    attrition["cancer"] = len(df)
    logger.info(f"After cancer keyword filter: {len(df):,} rows")
    
    # Step 4: Filter by overall status (only keep labeled outcomes)
    if "overall_status" in df.columns:
        valid_statuses = cfg.fail_statuses + cfg.success_statuses
        status_pattern = "|".join(valid_statuses)
        mask_status = df["overall_status"].str.upper().str.contains(
            status_pattern, na=False, regex=True
        )
        df = df[mask_status].copy()
    attrition["labeled_status"] = len(df)
    logger.info(f"After status filter (Completed/Terminated/Withdrawn): {len(df):,} rows")
    
    return df, attrition


def create_labels(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    """Create binary labels: Fail=1, Success=0.
    
    Args:
        df: DataFrame with overall_status column
        cfg: Configuration with fail/success status definitions
        
    Returns:
        DataFrame with 'label' column added
    """
    if "overall_status" not in df.columns:
        raise ValueError("Column 'overall_status' not found in DataFrame")
    
    df = df.copy()
    
    # Create label based on overall_status
    fail_pattern = "|".join(cfg.fail_statuses)
    success_pattern = "|".join(cfg.success_statuses)
    
    is_fail = df["overall_status"].str.upper().str.contains(fail_pattern, na=False, regex=True)
    is_success = df["overall_status"].str.upper().str.contains(success_pattern, na=False, regex=True)
    
    df["label"] = np.nan
    df.loc[is_fail, "label"] = 1
    df.loc[is_success, "label"] = 0
    
    # Drop rows with no valid label
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    
    n_fail = (df["label"] == 1).sum()
    n_success = (df["label"] == 0).sum()
    logger.info(f"Labels created: Fail={n_fail:,} (1), Success={n_success:,} (0)")
    
    return df


def analyze_missing(df: pd.DataFrame, threshold_drop: float = 0.7) -> Dict[str, Any]:
    """Analyze missing values and categorize columns.
    
    Args:
        df: DataFrame to analyze
        threshold_drop: Missing rate threshold for dropping columns
        
    Returns:
        Dictionary with missing analysis results
    """
    missing_counts = df.isnull().sum()
    missing_rates = df.isnull().mean()
    
    # Sort by missing rate
    missing_df = pd.DataFrame({
        "column": missing_counts.index,
        "missing_count": missing_counts.values,
        "missing_rate": missing_rates.values
    }).sort_values("missing_rate", ascending=False)
    
    # Categorize columns
    core = []
    optional = []
    drop = []
    
    for _, row in missing_df.iterrows():
        col = row["column"]
        rate = row["missing_rate"]
        
        if rate >= threshold_drop:
            drop.append(col)
        elif rate >= 0.3:
            optional.append(col)
        else:
            core.append(col)
    
    return {
        "missing_df": missing_df,
        "core_columns": core,
        "optional_columns": optional,
        "drop_columns": drop,
        "rates": missing_rates.to_dict()
    }


def handle_missing(
    df: pd.DataFrame,
    cfg: PreprocessConfig,
    drop_threshold: float = 0.7,
    indicator_threshold: float = 0.05,
    numeric_strategy: str = "median",
    categorical_fill: str = "UNKNOWN"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Handle missing values with clear policy.
    
    Policy:
    - Critical columns (nct_id, start_date, study_type, label): DROP ROWS
    - Numeric columns: Median impute + *_is_missing indicator (if > indicator_threshold)
    - Categorical columns: Fill with 'UNKNOWN' + indicator (if > indicator_threshold)
    - Columns with missing > drop_threshold: DROP COLUMN
    - Special: allocation → fill 'NOT_APPLICABLE' for single-arm studies
    
    Args:
        df: Input DataFrame
        cfg: Preprocessing configuration
        drop_threshold: Missing rate threshold for dropping columns (default 0.7)
        indicator_threshold: Missing rate threshold for creating indicators (default 0.05)
        numeric_strategy: Imputation strategy for numeric ('median', 'mean')
        categorical_fill: Fill value for categorical columns
        
    Returns:
        (df_handled, missing_report_df)
    """
    df = df.copy()
    missing_log = []
    
    # Critical columns: drop rows if missing
    critical_cols = ["nct_id", "start_date", "study_type", "label"]
    for col in critical_cols:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                df = df.dropna(subset=[col])
                missing_log.append({
                    "column": col,
                    "category": "critical",
                    "missing_rate_before": n_missing / (len(df) + n_missing),
                    "action": "drop_rows",
                    "rows_affected": int(n_missing),
                    "impute_value": None,
                    "indicator_created": False
                })
                logger.info(f"Dropped {n_missing:,} rows with missing {col}")
    
    # Identify numeric and categorical columns
    numeric_cols = [col for col in DTYPE_SCHEMA.get("numeric_columns", []) if col in df.columns]
    categorical_cols = [col for col in DTYPE_SCHEMA.get("categorical_columns", []) if col in df.columns]
    
    # Handle numeric columns
    for col in numeric_cols:
        if col not in df.columns:
            continue
        
        missing_rate = df[col].isna().mean()
        
        if missing_rate >= drop_threshold:
            df = df.drop(columns=[col])
            missing_log.append({
                "column": col,
                "category": "numeric",
                "missing_rate_before": float(missing_rate),
                "action": "drop_column",
                "rows_affected": int(df[col].isna().sum()) if col in df.columns else 0,
                "impute_value": None,
                "indicator_created": False
            })
            logger.info(f"Dropped column {col} (missing rate: {missing_rate:.1%})")
        elif missing_rate > 0:
            # Compute imputation value
            if numeric_strategy == "median":
                impute_val = df[col].median()
            else:
                impute_val = df[col].mean()
            
            # Create indicator if significant missing
            indicator_created = False
            if missing_rate >= indicator_threshold:
                indicator_col = f"{col}_is_missing"
                df[indicator_col] = df[col].isna().astype(int)
                indicator_created = True
            
            # Impute
            n_imputed = df[col].isna().sum()
            df[col] = df[col].fillna(impute_val)
            
            missing_log.append({
                "column": col,
                "category": "numeric",
                "missing_rate_before": float(missing_rate),
                "action": f"{numeric_strategy}_impute",
                "rows_affected": int(n_imputed),
                "impute_value": float(impute_val),
                "indicator_created": indicator_created
            })
    
    # Handle categorical columns
    for col in categorical_cols:
        if col not in df.columns:
            continue
        
        missing_rate = df[col].isna().mean()
        
        if missing_rate >= drop_threshold:
            df = df.drop(columns=[col])
            missing_log.append({
                "column": col,
                "category": "categorical",
                "missing_rate_before": float(missing_rate),
                "action": "drop_column",
                "rows_affected": 0,
                "impute_value": None,
                "indicator_created": False
            })
            logger.info(f"Dropped column {col} (missing rate: {missing_rate:.1%})")
        elif missing_rate > 0:
            # Special case: allocation (often missing for single-arm)
            if col == "allocation":
                fill_val = "NOT_APPLICABLE"
            else:
                fill_val = categorical_fill
            
            # Create indicator if significant missing
            indicator_created = False
            if missing_rate >= indicator_threshold:
                indicator_col = f"{col}_is_missing"
                df[indicator_col] = df[col].isna().astype(int)
                indicator_created = True
            
            # Fill missing
            n_filled = df[col].isna().sum()
            df[col] = df[col].fillna(fill_val)
            
            missing_log.append({
                "column": col,
                "category": "categorical",
                "missing_rate_before": float(missing_rate),
                "action": f"fill_{fill_val.lower()}",
                "rows_affected": int(n_filled),
                "impute_value": fill_val,
                "indicator_created": indicator_created
            })
    
    report_df = pd.DataFrame(missing_log) if missing_log else pd.DataFrame()
    
    n_indicators = sum([1 for x in missing_log if x.get("indicator_created", False)])
    logger.info(f"Missing handling complete: {len(missing_log)} columns processed, {n_indicators} indicators created")
    
    return df, report_df


def get_feature_columns(
    df: pd.DataFrame,
    cfg: PreprocessConfig
) -> Dict[str, Any]:
    """Get explicit column lists for modeling with anti-leakage guarantee.
    
    Categorizes columns into:
    - id_columns: Identifiers (nct_id)
    - label_columns: Target variable (label)
    - qc_only_columns: For analysis only, not modeling
    - feature_columns: Safe to use at t0 for modeling
    
    Args:
        df: DataFrame to analyze
        cfg: Preprocessing configuration
        
    Returns:
        Dictionary with column categorizations
    """
    all_cols = set(df.columns)
    
    # ID columns
    id_columns = ["nct_id"] if "nct_id" in all_cols else []
    
    # Label columns
    label_columns = ["label"] if "label" in all_cols else []
    
    # QC-only columns (keep for analysis, not for modeling)
    qc_columns = [c for c in cfg.qc_only_columns if c in all_cols]
    
    # Post-t0 columns to exclude from features
    post_t0 = set(cfg.post_t0_columns)
    
    # Feature columns = all - id - label - qc - post_t0
    exclude = set(id_columns) | set(label_columns) | set(qc_columns) | post_t0
    feature_columns = [c for c in df.columns if c not in exclude]
    
    # Separate base features and indicators
    base_features = [c for c in feature_columns if not c.endswith("_is_missing")]
    indicator_features = [c for c in feature_columns if c.endswith("_is_missing")]
    
    result = {
        "id_columns": id_columns,
        "label_columns": label_columns,
        "qc_only_columns": qc_columns,
        "feature_columns": feature_columns,
        "base_features": base_features,
        "indicator_features": indicator_features,
        "total_features": len(feature_columns),
        "post_t0_excluded": list(post_t0 & all_cols),
        "allowed_at_t0": True
    }
    
    logger.info(f"Feature columns: {len(base_features)} base + {len(indicator_features)} indicators = {len(feature_columns)} total")
    
    return result


def identify_post_t0_columns(df: pd.DataFrame, cfg: PreprocessConfig) -> List[str]:
    """Identify columns that contain information only available after t0 (start_date).
    
    These columns cause data leakage and must be excluded from features.
    
    Args:
        df: DataFrame to analyze
        cfg: Configuration with post-t0 column definitions
        
    Returns:
        List of post-t0 column names present in DataFrame
    """
    post_t0_present = [col for col in cfg.post_t0_columns if col in df.columns]
    logger.info(f"Post-t0 columns identified for removal: {post_t0_present}")
    return post_t0_present


def remove_leakage_columns(
    df: pd.DataFrame, 
    cfg: PreprocessConfig,
    keep_for_qc: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """Remove post-t0 columns to prevent data leakage.
    
    Args:
        df: DataFrame with potential leakage columns
        cfg: Configuration
        keep_for_qc: If True, keep some columns for QC analysis only
        
    Returns:
        DataFrame with leakage columns removed and list of removed columns
    """
    df = df.copy()
    post_t0_cols = identify_post_t0_columns(df, cfg)
    
    if keep_for_qc:
        # Keep QC columns but mark them
        cols_to_drop = [c for c in post_t0_cols if c not in cfg.qc_only_columns]
    else:
        cols_to_drop = post_t0_cols
    
    # Don't drop 'label' or 'overall_status' if needed for modeling
    cols_to_drop = [c for c in cols_to_drop if c != "label"]
    
    df_clean = df.drop(columns=cols_to_drop, errors="ignore")
    
    logger.info(f"Removed {len(cols_to_drop)} leakage columns: {cols_to_drop}")
    
    return df_clean, cols_to_drop


def sanity_checks(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform sanity checks on the data.
    
    Checks:
    1. completion_date < start_date (date anomalies)
    2. Enrollment count outliers
    3. Duplicate NCT IDs
    
    Args:
        df: DataFrame to check
        
    Returns:
        Dictionary with sanity check results
    """
    results = {}
    
    # Check 1: Date anomalies (completion < start)
    if "start_date" in df.columns and "completion_date" in df.columns:
        mask_valid = df["start_date"].notna() & df["completion_date"].notna()
        if mask_valid.any():
            date_anomaly = df.loc[mask_valid, "completion_date"] < df.loc[mask_valid, "start_date"]
            results["date_anomalies_count"] = date_anomaly.sum()
            results["date_anomalies_rate"] = date_anomaly.mean()
            results["date_anomaly_nct_ids"] = df.loc[mask_valid][date_anomaly]["nct_id"].tolist()[:10]
        else:
            results["date_anomalies_count"] = 0
            results["date_anomalies_rate"] = 0.0
    else:
        results["date_anomalies_count"] = "N/A"
    
    # Check 2: Enrollment outliers (using IQR method)
    if "enrollment_count" in df.columns:
        enrollment = df["enrollment_count"].dropna()
        if len(enrollment) > 0:
            Q1 = enrollment.quantile(0.25)
            Q3 = enrollment.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            outliers = enrollment[(enrollment < lower) | (enrollment > upper)]
            results["enrollment_outliers_count"] = len(outliers)
            results["enrollment_outliers_rate"] = len(outliers) / len(enrollment)
            results["enrollment_stats"] = {
                "mean": enrollment.mean(),
                "median": enrollment.median(),
                "std": enrollment.std(),
                "min": enrollment.min(),
                "max": enrollment.max(),
                "Q1": Q1,
                "Q3": Q3
            }
            # Top 5 outliers
            results["top_outliers"] = enrollment.nlargest(5).tolist()
        else:
            results["enrollment_outliers_count"] = "N/A"
    else:
        results["enrollment_outliers_count"] = "N/A"
    
    # Check 3: Duplicate NCT IDs
    if "nct_id" in df.columns:
        duplicates = df["nct_id"].duplicated().sum()
        results["duplicate_nct_ids"] = duplicates
    else:
        results["duplicate_nct_ids"] = "N/A"
    
    return results


def split_data(
    df: pd.DataFrame,
    cfg: PreprocessConfig,
    label_col: str = "label"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Split data into train/validation/test sets.
    
    Supports stratified or time-based splitting.
    
    Args:
        df: DataFrame to split
        cfg: Configuration with split parameters
        label_col: Name of label column for stratification
        
    Returns:
        (train_df, val_df, test_df, split_info)
    """
    split_info = {
        "method": cfg.split_method,
        "random_seed": cfg.random_seed,
        "test_size": cfg.test_size,
        "val_size": cfg.val_size
    }
    
    if cfg.split_method == "time" and cfg.time_split_date and "start_date" in df.columns:
        # Time-based split
        split_date = pd.to_datetime(cfg.time_split_date)
        
        train_val = df[df["start_date"] < split_date].copy()
        test = df[df["start_date"] >= split_date].copy()
        
        # Further split train_val into train and validation
        if len(train_val) > 0:
            val_frac = cfg.val_size / (1 - cfg.test_size)
            train, val = train_test_split(
                train_val, 
                test_size=val_frac,
                random_state=cfg.random_seed,
                stratify=train_val[label_col] if label_col in train_val.columns else None
            )
        else:
            train, val = train_val, pd.DataFrame()
        
        split_info["time_split_date"] = cfg.time_split_date
        
    else:
        # Stratified split
        # First split: train+val vs test
        train_val, test = train_test_split(
            df,
            test_size=cfg.test_size,
            random_state=cfg.random_seed,
            stratify=df[label_col] if label_col in df.columns else None
        )
        
        # Second split: train vs val
        val_frac = cfg.val_size / (1 - cfg.test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_frac,
            random_state=cfg.random_seed,
            stratify=train_val[label_col] if label_col in train_val.columns else None
        )
    
    # Calculate label ratios
    for name, data in [("train", train), ("val", val), ("test", test)]:
        if label_col in data.columns and len(data) > 0:
            split_info[f"{name}_size"] = len(data)
            split_info[f"{name}_label_ratio"] = data[label_col].mean()
        else:
            split_info[f"{name}_size"] = len(data)
            split_info[f"{name}_label_ratio"] = None
    
    # Verify no overlap
    train_ids = set(train["nct_id"]) if "nct_id" in train.columns else set()
    val_ids = set(val["nct_id"]) if "nct_id" in val.columns else set()
    test_ids = set(test["nct_id"]) if "nct_id" in test.columns else set()
    
    overlap_train_val = len(train_ids & val_ids)
    overlap_train_test = len(train_ids & test_ids)
    overlap_val_test = len(val_ids & test_ids)
    
    split_info["overlap_train_val"] = overlap_train_val
    split_info["overlap_train_test"] = overlap_train_test
    split_info["overlap_val_test"] = overlap_val_test
    
    if overlap_train_val + overlap_train_test + overlap_val_test > 0:
        logger.warning(f"⚠️ Data leakage detected! Overlapping NCT IDs between splits.")
    else:
        logger.info("✅ No overlap between train/val/test splits")
    
    logger.info(f"Split complete: train={len(train):,}, val={len(val):,}, test={len(test):,}")
    
    return train, val, test, split_info


def create_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """Create a data dictionary describing all columns.
    
    Args:
        df: DataFrame to document
        
    Returns:
        DataFrame with column documentation
    """
    dict_rows = []
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].notna().sum()
        null_rate = df[col].isnull().mean()
        n_unique = df[col].nunique()
        
        # Sample values
        sample_values = df[col].dropna().head(3).tolist()
        sample_str = str(sample_values)[:100]
        
        dict_rows.append({
            "column": col,
            "dtype": dtype,
            "non_null_count": non_null,
            "null_rate": round(null_rate, 4),
            "n_unique": n_unique,
            "sample_values": sample_str
        })
    
    return pd.DataFrame(dict_rows)


def compute_checksum(df: pd.DataFrame, algorithm: str = "sha256") -> str:
    """Compute checksum of DataFrame for reproducibility.
    
    Args:
        df: DataFrame to checksum
        algorithm: Hash algorithm ('md5' or 'sha256')
        
    Returns:
        Hash string (first 16 characters)
    """
    hash_values = pd.util.hash_pandas_object(df, index=True).values
    
    if algorithm == "sha256":
        return hashlib.sha256(hash_values).hexdigest()[:16]
    else:
        return hashlib.md5(hash_values).hexdigest()[:16]


def save_artifacts(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cfg: PreprocessConfig,
    feature_info: Dict[str, Any],
    attrition: Dict[str, int],
    reports: Optional[Dict[str, pd.DataFrame]] = None
) -> Dict[str, str]:
    """Save all preprocessing artifacts.
    
    Creates both model-ready files (features only) and QC files (with analysis columns).
    
    Args:
        train: Training DataFrame
        val: Validation DataFrame
        test: Test DataFrame
        cfg: Preprocessing configuration
        feature_info: Feature column information
        attrition: Attrition counts dictionary
        reports: Optional dict of report DataFrames (dtype_report, missing_report, etc.)
        
    Returns:
        Dictionary of output file paths
    """
    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_files = {}
    
    # Define model-only columns
    model_cols = (
        feature_info.get("id_columns", []) + 
        feature_info.get("label_columns", []) + 
        feature_info.get("feature_columns", [])
    )
    
    # Define QC columns (model + qc_only)
    qc_extra = feature_info.get("qc_only_columns", [])
    
    # Save model files (features only)
    for name, data in [("train", train), ("val", val), ("test", test)]:
        # Model-ready file
        available_model_cols = [c for c in model_cols if c in data.columns]
        model_path = output_dir / f"{name}_model.csv"
        data[available_model_cols].to_csv(model_path, index=False)
        output_files[f"{name}_model"] = str(model_path)
        
        # QC file (includes analysis columns)
        qc_cols = available_model_cols + [c for c in qc_extra if c in data.columns]
        qc_path = output_dir / f"{name}_qc.csv"
        data[[c for c in qc_cols if c in data.columns]].to_csv(qc_path, index=False)
        output_files[f"{name}_qc"] = str(qc_path)
        
        # Also save standard name
        std_path = output_dir / f"{name}_data.csv"
        data.to_csv(std_path, index=False)
        output_files[name] = str(std_path)
    
    # Save full processed data
    full_df = pd.concat([train, val, test], ignore_index=True)
    full_path = output_dir / "full_processed_data.csv"
    full_df.to_csv(full_path, index=False)
    output_files["full"] = str(full_path)
    
    # Save attrition report
    attrition_path = output_dir / "attrition_report.json"
    with open(attrition_path, "w", encoding="utf-8") as f:
        json.dump(attrition, f, indent=2)
    output_files["attrition"] = str(attrition_path)
    
    # Save split manifest
    manifest = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "seed": cfg.random_seed,
            "split_method": cfg.split_method,
            "test_size": cfg.test_size,
            "val_size": cfg.val_size
        },
        "sizes": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "total": len(full_df)
        },
        "label_ratios": {
            "train": float(train["label"].mean()) if "label" in train.columns else None,
            "val": float(val["label"].mean()) if "label" in val.columns and len(val) > 0 else None,
            "test": float(test["label"].mean()) if "label" in test.columns else None
        },
        "checksums": {
            "train": compute_checksum(train),
            "val": compute_checksum(val) if len(val) > 0 else None,
            "test": compute_checksum(test),
            "full": compute_checksum(full_df)
        },
        "anti_leakage": {
            "t0_definition": "start_date",
            "post_t0_columns": cfg.post_t0_columns,
            "feature_count": feature_info.get("total_features", 0)
        }
    }
    manifest_path = output_dir / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    output_files["manifest"] = str(manifest_path)
    
    # Save optional reports
    if reports:
        for name, report_df in reports.items():
            if isinstance(report_df, pd.DataFrame) and len(report_df) > 0:
                report_path = output_dir / f"{name}.csv"
                report_df.to_csv(report_path, index=False)
                output_files[name] = str(report_path)
    
    # Save feature info
    feature_path = output_dir / "feature_info.json"
    with open(feature_path, "w", encoding="utf-8") as f:
        # Convert lists to ensure JSON serializable
        feature_info_clean = {k: list(v) if isinstance(v, (list, set)) else v 
                             for k, v in feature_info.items()}
        json.dump(feature_info_clean, f, indent=2)
    output_files["feature_info"] = str(feature_path)
    
    # Save data dictionary with allowed_at_t0 flag
    dict_df = create_data_dictionary(full_df)
    dict_df["allowed_at_t0"] = dict_df["column"].apply(
        lambda c: "No" if c in cfg.post_t0_columns or c in ["label", "overall_status"] else "Yes"
    )
    dict_df["is_feature"] = dict_df["column"].apply(
        lambda c: "Yes" if c in feature_info.get("feature_columns", []) else "No"
    )
    dict_path = output_dir / "data_dictionary.csv"
    dict_df.to_csv(dict_path, index=False)
    output_files["data_dictionary"] = str(dict_path)
    
    logger.info(f"Saved {len(output_files)} artifacts to {output_dir}")
    
    return output_files


# =============================================================================
# Main Pipeline Function
# =============================================================================

def run_preprocessing_pipeline(
    input_csv: Union[str, Path],
    cfg: Optional[PreprocessConfig] = None,
    save_outputs: bool = True,
    handle_missing_data: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PreprocessingReport]:
    """Run the complete preprocessing pipeline (v2 - Enhanced).
    
    Pipeline Steps:
    1. Load raw data
    2. Standardize missing tokens → NaN
    3. Standardize data types (date, numeric, boolean, categorical)
    4. Filter cohort (Interventional + Phase 2-3 + Cancer)
    5. Create labels (Fail=1, Success=0)
    6. Handle missing values (impute, drop, indicators)
    7. Remove leakage columns
    8. Get feature columns (explicit t0-safe list)
    9. Sanity checks
    10. Split data (stratified/time)
    11. Save artifacts
    
    Args:
        input_csv: Path to raw CSV file
        cfg: Preprocessing configuration (uses defaults if None)
        save_outputs: Whether to save output files
        handle_missing_data: Whether to handle missing values (True) or just analyze (False)
        
    Returns:
        (train_df, val_df, test_df, report)
    """
    if cfg is None:
        cfg = PreprocessConfig()
    
    report = PreprocessingReport(
        timestamp=datetime.now().isoformat(),
        config=cfg.to_dict()
    )
    
    # Collect reports for saving
    all_reports = {}
    
    # Step 1: Load data
    logger.info("=" * 60)
    logger.info("STEP 1: Loading raw data")
    logger.info("=" * 60)
    df = load_raw_data(input_csv)
    report.raw_count = len(df)
    
    # Step 2: Standardize missing tokens
    logger.info("=" * 60)
    logger.info("STEP 2: Standardizing missing tokens → NaN")
    logger.info("=" * 60)
    df, missing_token_report = standardize_missing_tokens(df)
    if len(missing_token_report) > 0:
        all_reports["missing_token_report"] = missing_token_report
    
    # Step 3: Standardize data types
    logger.info("=" * 60)
    logger.info("STEP 3: Standardizing data types")
    logger.info("=" * 60)
    df, dtype_report = standardize_dtypes(df)
    if len(dtype_report) > 0:
        all_reports["dtype_conversion_report"] = dtype_report
    
    # Step 4: Filter cohort
    logger.info("=" * 60)
    logger.info("STEP 4: Filtering cohort")
    logger.info("=" * 60)
    df, attrition = filter_cohort(df, cfg)
    report.after_interventional = attrition.get("interventional", 0)
    report.after_phase_filter = attrition.get("phase_2_3", 0)
    report.after_cancer_filter = attrition.get("cancer", 0)
    report.after_status_filter = attrition.get("labeled_status", 0)
    
    # Step 5: Create labels
    logger.info("=" * 60)
    logger.info("STEP 5: Creating labels")
    logger.info("=" * 60)
    df = create_labels(df, cfg)
    report.label_counts = {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()}
    report.label_rates = {
        "fail_rate": float((df["label"] == 1).mean()),
        "success_rate": float((df["label"] == 0).mean())
    }
    
    # Step 6: Handle missing data
    logger.info("=" * 60)
    logger.info("STEP 6: Handling missing data")
    logger.info("=" * 60)
    
    # First analyze
    missing_analysis = analyze_missing(df)
    report.missing_rates = {k: float(v) for k, v in missing_analysis["rates"].items()}
    
    if handle_missing_data:
        df, missing_report = handle_missing(df, cfg)
        if len(missing_report) > 0:
            all_reports["missing_handling_report"] = missing_report
            report.columns_dropped_missing = [
                row["column"] for _, row in missing_report.iterrows() 
                if row.get("action") == "drop_column"
            ]
    else:
        report.columns_dropped_missing = missing_analysis["drop_columns"]
    
    # Step 7: Remove leakage columns
    logger.info("=" * 60)
    logger.info("STEP 7: Removing leakage columns")
    logger.info("=" * 60)
    df_clean, dropped_cols = remove_leakage_columns(df, cfg, keep_for_qc=True)
    report.columns_dropped_leakage = dropped_cols
    
    # Step 8: Get feature columns
    logger.info("=" * 60)
    logger.info("STEP 8: Getting feature columns (t0-safe)")
    logger.info("=" * 60)
    feature_info = get_feature_columns(df_clean, cfg)
    
    # Step 9: Sanity checks
    logger.info("=" * 60)
    logger.info("STEP 9: Running sanity checks")
    logger.info("=" * 60)
    sanity = sanity_checks(df)
    report.date_anomalies = sanity.get("date_anomalies_count", 0) if isinstance(sanity.get("date_anomalies_count"), int) else 0
    report.enrollment_outliers = sanity.get("enrollment_outliers_count", 0) if isinstance(sanity.get("enrollment_outliers_count"), int) else 0
    report.duplicate_nct_ids = sanity.get("duplicate_nct_ids", 0) if isinstance(sanity.get("duplicate_nct_ids"), int) else 0
    
    # Step 10: Split data
    logger.info("=" * 60)
    logger.info("STEP 10: Splitting data")
    logger.info("=" * 60)
    train, val, test, split_info = split_data(df_clean, cfg)
    report.train_size = split_info["train_size"]
    report.val_size = split_info["val_size"]
    report.test_size = split_info["test_size"]
    report.train_label_ratio = split_info.get("train_label_ratio", 0) or 0
    report.val_label_ratio = split_info.get("val_label_ratio", 0) or 0
    report.test_label_ratio = split_info.get("test_label_ratio", 0) or 0
    
    report.final_count = len(train) + len(val) + len(test)
    
    # Step 11: Save outputs
    if save_outputs:
        logger.info("=" * 60)
        logger.info("STEP 11: Saving artifacts")
        logger.info("=" * 60)
        
        output_files = save_artifacts(
            train=train,
            val=val,
            test=test,
            cfg=cfg,
            feature_info=feature_info,
            attrition=attrition,
            reports=all_reports
        )
        
        report.output_files = output_files
        report.data_checksum = compute_checksum(pd.concat([train, val, test], ignore_index=True))
        
        # Save final report
        report_path = cfg.output_dir / "preprocessing_report.json"
        report.save(report_path)
        report.output_files["report"] = str(report_path)
        
        logger.info(f"All outputs saved to {cfg.output_dir}")
    
    logger.info("=" * 60)
    logger.info("✅ PREPROCESSING PIPELINE COMPLETE (v2)")
    logger.info(f"   Features: {feature_info['total_features']} (base: {len(feature_info['base_features'])}, indicators: {len(feature_info['indicator_features'])})")
    logger.info(f"   Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    logger.info("=" * 60)
    
    return train, val, test, report


# =============================================================================
# Utility Functions for Visualization
# =============================================================================

def get_attrition_data(report: PreprocessingReport) -> pd.DataFrame:
    """Extract attrition data from report for visualization.
    
    Args:
        report: Preprocessing report
        
    Returns:
        DataFrame with attrition stages and counts
    """
    stages = [
        ("Raw Data", report.raw_count),
        ("Interventional", report.after_interventional),
        ("Phase 2-3", report.after_phase_filter),
        ("Cancer Keyword", report.after_cancer_filter),
        ("Labeled Status", report.after_status_filter),
        ("Final (after cleaning)", report.final_count)
    ]
    
    return pd.DataFrame(stages, columns=["Stage", "Count"])


def get_split_summary(report: PreprocessingReport) -> pd.DataFrame:
    """Extract split summary from report.
    
    Args:
        report: Preprocessing report
        
    Returns:
        DataFrame with split statistics
    """
    data = {
        "Split": ["Train", "Validation", "Test"],
        "Size": [report.train_size, report.val_size, report.test_size],
        "Fail Rate": [report.train_label_ratio, report.val_label_ratio, report.test_label_ratio]
    }
    
    df = pd.DataFrame(data)
    df["Success Rate"] = 1 - df["Fail Rate"]
    df["% of Total"] = df["Size"] / df["Size"].sum() * 100
    
    return df


# =============================================================================
# Enhanced Missing Value & Feature Processing Functions (Domain-specific)
# =============================================================================

def parse_dates_and_create_duration(
    df: pd.DataFrame,
    start_col: str = "start_date",
    completion_col: str = "completion_date"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Parse date columns and create duration_days feature.
    
    IMPORTANT: completion_date is NOT imputed (anti-leakage).
    
    Args:
        df: Input DataFrame
        start_col: Name of start date column
        completion_col: Name of completion date column
        
    Returns:
        (df_processed, parse_report)
    """
    df = df.copy()
    report = {}
    
    # Parse start_date
    if start_col in df.columns:
        before_null = df[start_col].isna().sum()
        df[start_col] = pd.to_datetime(df[start_col], errors='coerce')
        after_null = df[start_col].isna().sum()
        report[f"{start_col}_parsed"] = {
            "before_null": int(before_null),
            "after_null": int(after_null),
            "coerced": int(after_null - before_null)
        }
        logger.info(f"Parsed {start_col}: {after_null - before_null} values coerced to NaT")
    
    # Parse completion_date
    if completion_col in df.columns:
        before_null = df[completion_col].isna().sum()
        df[completion_col] = pd.to_datetime(df[completion_col], errors='coerce')
        after_null = df[completion_col].isna().sum()
        report[f"{completion_col}_parsed"] = {
            "before_null": int(before_null),
            "after_null": int(after_null),
            "coerced": int(after_null - before_null)
        }
        logger.info(f"Parsed {completion_col}: {after_null - before_null} values coerced to NaT")
        
        # Create completion_date_missing flag (BEFORE any imputation)
        df["completion_date_missing"] = df[completion_col].isna().astype(int)
        report["completion_date_missing_count"] = int(df["completion_date_missing"].sum())
        logger.info(f"Created completion_date_missing flag: {df['completion_date_missing'].sum()} missing")
    
    # Create duration_days = completion_date - start_date
    if start_col in df.columns and completion_col in df.columns:
        df["duration_days"] = (df[completion_col] - df[start_col]).dt.days
        
        # Handle negative durations (data quality issue)
        negative_mask = df["duration_days"] < 0
        n_negative = negative_mask.sum()
        if n_negative > 0:
            logger.warning(f"Found {n_negative} trials with negative duration (completion < start)")
            report["negative_duration_count"] = int(n_negative)
            # Set negative durations to NaN
            df.loc[negative_mask, "duration_days"] = np.nan
        
        report["duration_days_valid"] = int(df["duration_days"].notna().sum())
        report["duration_days_missing"] = int(df["duration_days"].isna().sum())
        logger.info(f"Created duration_days: {df['duration_days'].notna().sum()} valid values")
    
    return df, report


def handle_categorical_missing(
    df: pd.DataFrame,
    categorical_cols: List[str] = None,
    fill_value: str = "NOT_REPORTED"
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Handle missing values for categorical columns with missing flags.
    
    Creates <col>_missing = 1/0 flag, then fills with NOT_REPORTED.
    Does NOT use mode imputation (avoids "bịa" data).
    
    Args:
        df: Input DataFrame
        categorical_cols: List of categorical columns to process
        fill_value: Value to fill missing (default "NOT_REPORTED")
        
    Returns:
        (df_processed, missing_log)
    """
    if categorical_cols is None:
        categorical_cols = ["allocation", "intervention_model", "enrollment_type", "masking"]
    
    df = df.copy()
    missing_log = []
    
    for col in categorical_cols:
        if col not in df.columns:
            continue
        
        missing_count = df[col].isna().sum()
        missing_rate = df[col].isna().mean()
        
        if missing_count > 0:
            # Create missing indicator flag FIRST
            indicator_col = f"{col}_missing"
            df[indicator_col] = df[col].isna().astype(int)
            
            # Fill missing with NOT_REPORTED (not mode)
            df[col] = df[col].fillna(fill_value)
            
            missing_log.append({
                "column": col,
                "missing_count": int(missing_count),
                "missing_rate": float(missing_rate),
                "indicator_created": indicator_col,
                "fill_value": fill_value
            })
            
            logger.info(f"Categorical {col}: {missing_count} missing → filled with '{fill_value}', created {indicator_col}")
    
    return df, missing_log


def handle_has_dmc_missing(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Handle has_dmc missing values with UNKNOWN (not False).
    
    Rationale: Missing DMC info doesn't mean no DMC - it's unknown.
    
    Args:
        df: Input DataFrame
        
    Returns:
        (df_processed, report)
    """
    df = df.copy()
    report = {}
    
    if "has_dmc" not in df.columns:
        return df, report
    
    missing_count = df["has_dmc"].isna().sum()
    missing_rate = df["has_dmc"].isna().mean()
    
    # Create missing indicator
    df["has_dmc_missing"] = df["has_dmc"].isna().astype(int)
    
    # Convert to string and fill with "UNKNOWN" (not False)
    df["has_dmc"] = df["has_dmc"].astype(str)
    df.loc[df["has_dmc"].isin(["nan", "None", ""]), "has_dmc"] = "UNKNOWN"
    
    report = {
        "column": "has_dmc",
        "missing_count": int(missing_count),
        "missing_rate": float(missing_rate),
        "fill_value": "UNKNOWN",
        "indicator_created": "has_dmc_missing"
    }
    
    logger.info(f"has_dmc: {missing_count} missing → filled with 'UNKNOWN', created has_dmc_missing")
    
    return df, report


def fix_invalid_num_arms(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fix invalid num_arms values for INTERVENTIONAL studies.
    
    Rule: If study_type == INTERVENTIONAL and num_arms == 0, set to NaN.
    
    Args:
        df: Input DataFrame
        
    Returns:
        (df_fixed, invalid_rows_df)
    """
    df = df.copy()
    
    # Identify invalid rows
    if "num_arms" not in df.columns or "study_type" not in df.columns:
        return df, pd.DataFrame()
    
    invalid_mask = (
        (df["study_type"].str.upper().str.contains("INTERVENTIONAL", na=False)) &
        (df["num_arms"] == 0)
    )
    
    n_invalid = invalid_mask.sum()
    
    # Extract invalid rows for reporting
    invalid_rows = df.loc[invalid_mask, ["nct_id", "study_type", "num_arms"]].copy() if "nct_id" in df.columns else pd.DataFrame()
    
    if n_invalid > 0:
        # Set invalid num_arms to NaN
        df.loc[invalid_mask, "num_arms"] = np.nan
        logger.info(f"Fixed {n_invalid} rows with invalid num_arms=0 for INTERVENTIONAL studies → NaN")
    
    # Create missing indicator after fix
    df["num_arms_missing"] = df["num_arms"].isna().astype(int)
    
    return df, invalid_rows


def impute_numeric_group_median(
    df: pd.DataFrame,
    col: str,
    group_cols: List[str],
    fallback_median: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Impute numeric column using group median with fallback.
    
    Strategy:
    1. Compute median per group (group_cols)
    2. Fill missing with group median
    3. If still NaN, fallback to global median
    
    Args:
        df: Input DataFrame
        col: Column to impute
        group_cols: Columns for grouping
        fallback_median: Whether to fallback to global median
        
    Returns:
        (df_imputed, impute_report)
    """
    df = df.copy()
    report = {
        "column": col,
        "group_cols": group_cols,
        "missing_before": 0,
        "missing_after": 0,
        "imputed_by_group": 0,
        "imputed_by_fallback": 0,
        "global_median": None
    }
    
    if col not in df.columns:
        return df, report
    
    # Create missing indicator BEFORE imputation
    missing_indicator = f"{col}_missing"
    if missing_indicator not in df.columns:
        df[missing_indicator] = df[col].isna().astype(int)
    
    missing_before = df[col].isna().sum()
    report["missing_before"] = int(missing_before)
    
    if missing_before == 0:
        return df, report
    
    # Filter valid group columns
    valid_group_cols = [c for c in group_cols if c in df.columns]
    
    if valid_group_cols:
        # Compute group medians
        group_medians = df.groupby(valid_group_cols)[col].transform('median')
        
        # Count how many will be filled by group
        group_fill_mask = df[col].isna() & group_medians.notna()
        n_group_fill = group_fill_mask.sum()
        
        # Fill with group median
        df[col] = df[col].fillna(group_medians)
        report["imputed_by_group"] = int(n_group_fill)
    
    # Fallback to global median
    if fallback_median:
        global_median = df[col].median()
        report["global_median"] = float(global_median) if pd.notna(global_median) else None
        
        still_missing = df[col].isna().sum()
        if still_missing > 0 and pd.notna(global_median):
            df[col] = df[col].fillna(global_median)
            report["imputed_by_fallback"] = int(still_missing)
    
    report["missing_after"] = int(df[col].isna().sum())
    
    logger.info(f"Imputed {col}: {report['imputed_by_group']} by group, {report['imputed_by_fallback']} by fallback")
    
    return df, report


def impute_enrollment_count(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Impute enrollment_count using group median.
    
    Groups: (study_type, phases, overall_status)
    Fallback: Global median
    
    Also creates enrollment_missing flag and log_enrollment feature.
    
    Args:
        df: Input DataFrame
        
    Returns:
        (df_imputed, report)
    """
    df = df.copy()
    
    # Create missing indicator BEFORE imputation
    if "enrollment_count" in df.columns:
        df["enrollment_missing"] = df["enrollment_count"].isna().astype(int)
    
    # Impute with group median
    group_cols = ["study_type", "phases", "overall_status"]
    df, report = impute_numeric_group_median(
        df, "enrollment_count", group_cols, fallback_median=True
    )
    
    # Create log_enrollment for modeling (handles outliers)
    if "enrollment_count" in df.columns:
        df["log_enrollment"] = np.log1p(df["enrollment_count"])
        report["log_enrollment_created"] = True
    
    return df, report


def impute_num_arms(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Impute num_arms using group median.
    
    Groups: (intervention_model, allocation, phases)
    Fallback: Global median
    
    Args:
        df: Input DataFrame
        
    Returns:
        (df_imputed, report)
    """
    group_cols = ["intervention_model", "allocation", "phases"]
    return impute_numeric_group_median(df, "num_arms", group_cols, fallback_median=True)


def compute_outlier_stats(
    df: pd.DataFrame,
    col: str,
    percentiles: List[float] = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
) -> Dict[str, Any]:
    """Compute outlier statistics for a numeric column.
    
    Reports percentiles without dropping rows.
    
    Args:
        df: Input DataFrame
        col: Column to analyze
        percentiles: Percentiles to compute
        
    Returns:
        Dictionary with outlier statistics
    """
    if col not in df.columns:
        return {}
    
    data = df[col].dropna()
    
    if len(data) == 0:
        return {}
    
    stats = {
        "column": col,
        "count": int(len(data)),
        "mean": float(data.mean()),
        "std": float(data.std()),
        "min": float(data.min()),
        "max": float(data.max()),
        "percentiles": {}
    }
    
    for p in percentiles:
        pct_name = f"p{int(p*100)}" if p*100 == int(p*100) else f"p{p*100:.1f}"
        stats["percentiles"][pct_name] = float(data.quantile(p))
    
    # IQR-based outlier detection (for info only, no dropping)
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_low = (data < lower_bound).sum()
    outliers_high = (data > upper_bound).sum()
    
    stats["outliers"] = {
        "IQR": float(IQR),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "count_below": int(outliers_low),
        "count_above": int(outliers_high),
        "total_outliers": int(outliers_low + outliers_high),
        "outlier_rate": float((outliers_low + outliers_high) / len(data))
    }
    
    return stats


def get_missing_summary(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    columns: List[str] = None
) -> pd.DataFrame:
    """Generate missing rate before/after summary.
    
    Args:
        df_before: DataFrame before processing
        df_after: DataFrame after processing
        columns: Columns to compare (None = all common)
        
    Returns:
        DataFrame with missing comparison
    """
    if columns is None:
        columns = list(set(df_before.columns) & set(df_after.columns))
    
    summary = []
    for col in columns:
        before_rate = df_before[col].isna().mean() if col in df_before.columns else np.nan
        after_rate = df_after[col].isna().mean() if col in df_after.columns else np.nan
        
        summary.append({
            "column": col,
            "missing_before": int(df_before[col].isna().sum()) if col in df_before.columns else 0,
            "missing_after": int(df_after[col].isna().sum()) if col in df_after.columns else 0,
            "rate_before": f"{before_rate*100:.2f}%" if pd.notna(before_rate) else "N/A",
            "rate_after": f"{after_rate*100:.2f}%" if pd.notna(after_rate) else "N/A",
            "change": f"{(after_rate - before_rate)*100:.2f}pp" if pd.notna(before_rate) and pd.notna(after_rate) else "N/A"
        })
    
    return pd.DataFrame(summary).sort_values("missing_before", ascending=False)
