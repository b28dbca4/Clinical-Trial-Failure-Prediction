# src/utils/config.py
"""
Configuration module for Clinical Trial Failure Prediction project.
Contains paths, constants, and settings for data processing, EDA, and modeling.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# PATH CONFIGURATION

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FINAL_DIR = DATA_DIR / "final"
MODELING_DIR = PROCESSED_DIR / "modeling"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EDA_FIGURES_DIR = FIGURES_DIR / "eda_plots"
MODEL_FIGURES_DIR = FIGURES_DIR / "model_results"

# Ensure directories exist
for dir_path in [PROCESSED_DIR, FINAL_DIR, MODELING_DIR, EDA_FIGURES_DIR, MODEL_FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# LABEL DEFINITIONS

SUCCESS_STATUSES: List[str] = ['COMPLETED']
FAIL_STATUSES: List[str] = ['WITHDRAWN', 'TERMINATED']

# EDA CONFIGURATION

SEED: int = 42

# Visualization style
DEFAULT_FIGSIZE: Tuple[int, int] = (12, 6)
DEFAULT_DPI: int = 150

# Color palettes
OUTCOME_COLORS: Dict[int, str] = {
    0: '#e74c3c',  # Fail - Red
    1: '#27ae60'   # Success - Green
}

SPONSOR_COLORS: Dict[str, str] = {
    'INDUSTRY': '#3498db',      # Blue
    'OTHER': '#95a5a6',         # Gray
    'NIH': '#9b59b6',           # Purple
    'FED': '#e67e22',           # Orange
    'NETWORK': '#1abc9c',       # Teal
    'INDIV': '#f1c40f'          # Yellow
}

# Statistical thresholds
ALPHA_LEVEL: float = 0.05

# Effect size interpretation (Cohen's guidelines for Cramér's V)
CRAMERS_V_THRESHOLDS: Dict[str, Tuple[float, str]] = {
    'negligible': (0.0, 'không đáng kể'),
    'weak': (0.1, 'yếu'),
    'moderate': (0.2, 'trung bình'),
    'strong': (0.4, 'mạnh')
}

# Enrollment categories
ENROLLMENT_BINS: List[int] = [0, 50, 200, 500, float('inf')]
ENROLLMENT_LABELS: List[str] = ['Nhỏ (<50)', 'Trung bình (50-199)', 'Lớn (200-499)', 'Rất lớn (≥500)']

# API CONFIGURATION

CTGOV_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


@dataclass(frozen=True)
class CollectConfig:
    base_url: str = CTGOV_BASE_URL

    # ClinicalTrials.gov v2 commonly uses query.term
    query_term: str = "cancer"

    # Optional structured filters (you can apply them via extra_params if API supports)
    # Keep these as "intent" fields for reproducibility even if you filter later in preprocessing.
    study_type_filter: Optional[str] = "INTERVENTIONAL"  # e.g., "INTERVENTIONAL" or None
    phase_filter: Optional[List[str]] = field(default_factory=lambda: ["PHASE2", "PHASE3"])  # or []
    overall_status_filter: Optional[List[str]] = None
    # e.g. overall_status_filter=["COMPLETED","TERMINATED","WITHDRAWN"] to fetch only labeled outcomes

    # Additional query params passed to requests.get(..., params=...)
    # Use this to store ALL params you want logged for reproducibility.
    extra_params: Dict[str, Any] = field(default_factory=dict)

    # ---- Collection limits
    max_studies: int = 100000
    page_size: int = 1000

    # ---- Throttling / reliability
    sleep_s: float = 0.3
    timeout_s: int = 30

    max_retries: int = 3
    retry_delay_s: float = 2.0
    max_retry_delay_s: float = 30.0
    retry_jitter_s: float = 0.2  # small jitter to avoid thundering herd

    # ---- Pagination/resume (optional)
    start_page_token: Optional[str] = None   # resume from a specific nextPageToken if needed
    max_pages: Optional[int] = None          # safety guard

    # ---- Versioning / outputs
    raw_dir: Path = RAW_DIR
    processed_dir: Path = PROCESSED_DIR
    snapshot_strategy: str = "timestamped"   # "timestamped" (recommended) or "overwrite"

    # ---- Project-level reproducibility notes (write into metadata.json)
    prediction_time: str = "t0 = StartDate"
    label_definition: str = "Hard: Fail={WITHDRAWN, TERMINATED}; Success={COMPLETED}"
    allowed_features_note: str = (
        "Use only features available at/before StartDate; drop completion dates, results-posted dates, whyStopped, etc."
    )
