import pandas as pd
import numpy as np
import re

# -------------------------------------------------------------
# Cleaning
# -------------------------------------------------------------
def clean_dataframe(df):
    """
    Fix columns with inappropriate data types:
    - parse dates
    - phases: convert to categorical + extract numeric phase
    - healthy_volunteers: convert to boolean
    - sex_eligibility: convert to categorical
    - min_age, max_age: convert age strings -> numeric values
    """

    df = df.copy()
    
    # ---------------------------------------------------------
    # 1. Convert all date-like columns to datetime
    # ---------------------------------------------------------
    date_cols = [c for c in df.columns if "date" in c]

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    
    # ---------------------------------------------------------
    # 2. Replace empty strings in all object columns with NaN
    # ---------------------------------------------------------
    obj_cols = df.select_dtypes(include="object").columns

    df[obj_cols] = df[obj_cols].replace(
        to_replace=[r"^\s*$", r"^NA$", r"^N/A$", r"^na$", r"^n/a$"],
        value=pd.NA,
        regex=True
    )

    # ---------------------------------------------------------
    # 3. Clean phases (object -> category + numeric extraction)
    # ---------------------------------------------------------
    df["phases"] = df["phases"].astype("category")

    def extract_phase_num(p):
        if pd.isna(p):
            return np.nan
        nums = re.findall(r'\d+', str(p))
        if nums:
            return float(max(map(int, nums)))
        return np.nan

    df["phase_num"] = df["phases"].apply(extract_phase_num)

    # ---------------------------------------------------------
    # 4. Clean healthy_volunteers (object -> boolean)
    # ---------------------------------------------------------
    df["healthy_volunteers"] = df["healthy_volunteers"].astype("boolean")

    # ---------------------------------------------------------
    # 5. sex_eligibility (object -> category)
    # ---------------------------------------------------------
    df["sex_eligibility"] = df["sex_eligibility"].astype("category")

    # ---------------------------------------------------------
    # 6. Clean age fields (min_age, max_age)
    # ---------------------------------------------------------
    def age_to_int(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip()

        # Numeric age like "18 Years"
        if "Year" in x:
            try:
                return int(x.split()[0])
            except:
                return np.nan

        mapping = {
            "Child": 0,
            "Adult": 18,
            "Older Adult": 65,
            "N/A": np.nan
        }
        return mapping.get(x, np.nan)

    df["min_age"] = df["min_age"].apply(age_to_int)
    df["max_age"] = df["max_age"].apply(age_to_int)

    return df

# -------------------------------------------------------------
# Basic validation
# -------------------------------------------------------------
def validate_values(df):
    """
    Validate numeric columns only.
    Provides missing value counts, negative values, min/max.
    """
    numeric_df = df.select_dtypes(include=["number"])

    X = numeric_df.values

    return {
        "num_numeric_columns": numeric_df.shape[1],
        "nan_count": np.isnan(X).sum(),
        "inf_count": np.isinf(X).sum(),
        "negative_count": (X < 0).sum(),
        "min_value": float(np.nanmin(X)),
        "max_value": float(np.nanmax(X)),
    }
# -------------------------------------------------------------
# Outliers (IQR)
# -------------------------------------------------------------
def remove_outliers_iqr(df, col):
    """
    Remove outliers from a numeric column using IQR rule.
    Returns a filtered DataFrame.
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

# -------------------------------------------------------------
# Missing value handling
# -------------------------------------------------------------
def fill_missing_mean(df, columns=None):
    """
    Fill numeric missing values using column mean.
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=["number"]).columns
    for col in columns:
        df[col] = df[col].fillna(df[col].mean())
    return df


def fill_missing_median(df, columns=None):
    """
    Fill numeric missing values using column median.
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=["number"]).columns
    for col in columns:
        df[col] = df[col].fillna(df[col].median())
    return df


def fill_missing_constant(df, value=0):
    """
    Fill numeric missing values using a constant value.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(value)
    return df