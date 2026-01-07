# src/data/data_cleaner.py
"""
Data Cleaning & Preprocessing Module for Clinical Trial Failure Prediction
"""

from __future__ import annotations

import json
import hashlib
import logging
from pathlib import Path
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

 
# Configuration Class

@dataclass
class PreprocessConfig:
    """Configuration for preprocessing pipeline."""
    
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

 
# Core Functions Used in Notebook 

def load_raw_data(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load raw CSV data.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame with loaded data
    """
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def standardize_dtypes(
    df: pd.DataFrame,
    schema: Optional[Dict[str, List[str]]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize column data types with coercion report.
    
    Args:
        df: Input DataFrame
        schema: Schema definition with keys:
                - date_columns: List of date column names
                - numeric_columns: List of numeric column names
                - boolean_columns: List of boolean column names
                - categorical_columns: List of categorical column names
                - text_columns: List of text column names
        
    Returns:
        (df_standardized, dtype_report_df)
    """
    df = df.copy()
    conversion_log = []
    
    if schema is None:
        schema = {}
    
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


def analyze_missing(df: pd.DataFrame, threshold_drop: float = 0.7) -> Dict[str, Any]:
    """Analyze missing values and categorize columns.
    
    Args:
        df: DataFrame to analyze
        threshold_drop: Missing rate threshold for dropping columns
        
    Returns:
        Dictionary with:
            - missing_df: DataFrame with missing stats per column
            - core_columns: Columns with < 30% missing
            - optional_columns: Columns with 30-70% missing
            - drop_columns: Columns with > threshold_drop missing
            - rates: Dict of missing rates per column
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
            results["date_anomalies_count"] = int(date_anomaly.sum())
        else:
            results["date_anomalies_count"] = 0
    else:
        results["date_anomalies_count"] = 0
    
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
            results["enrollment_outliers_count"] = int(len(outliers))
        else:
            results["enrollment_outliers_count"] = 0
    else:
        results["enrollment_outliers_count"] = 0
    
    # Check 3: Duplicate NCT IDs
    if "nct_id" in df.columns:
        duplicates = df["nct_id"].duplicated().sum()
        results["duplicate_nct_ids"] = int(duplicates)
    else:
        results["duplicate_nct_ids"] = 0
    
    return results


def split_data(
    df: pd.DataFrame,
    cfg: PreprocessConfig,
    label_col: str = "label_binary"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Split data into train/validation/test sets with stratification.
    
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
            split_info[f"{name}_label_ratio"] = float(data[label_col].mean())
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
        logger.warning(f"Data leakage detected! Overlapping NCT IDs between splits.")
    else:
        logger.info("No overlap between train/val/test splits")
    
    logger.info(f"Split complete: train={len(train):,}, val={len(val):,}, test={len(test):,}")
    
    return train, val, test, split_info


def create_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """Create a data dictionary describing all columns.
    
    Args:
        df: DataFrame to document
        
    Returns:
        DataFrame with column documentation (column, dtype, null_rate, n_unique, sample_values)
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
        Hash string
    """
    hash_values = pd.util.hash_pandas_object(df, index=True).values
    
    if algorithm == "sha256":
        return hashlib.sha256(hash_values).hexdigest()
    else:
        return hashlib.md5(hash_values).hexdigest()
