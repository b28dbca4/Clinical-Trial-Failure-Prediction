# src/utils/config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FINAL_DIR = DATA_DIR / "final"
REPORTS_DIR = PROJECT_ROOT / "reports"

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
    max_studies: int = 1000
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
