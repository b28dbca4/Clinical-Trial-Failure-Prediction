# src/data/data_loader.py
from __future__ import annotations

import os
import sys
import csv
import json
import time
import hashlib
import logging
import requests
import subprocess
from dataclasses import asdict
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
from urllib.parse import urlencode
from typing import Any, Dict, Iterator, List, Optional, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from src.utils.config import CollectConfig

# Setup logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Helpers
def _get_nested(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    """Safely extract nested value from dictionary.
    
    Args:
        d: Source dictionary
        path: List of keys representing nested path
        default: Default value if path not found
        
    Returns:
        Nested value or default
    """
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _to_pipe_str(x: Any) -> str:
    """Convert list or value to pipe-separated string.
    
    Args:
        x: Value to convert (can be list, str, or other)
        
    Returns:
        Pipe-separated string or empty string
    """
    if x is None:
        return ""
    if isinstance(x, list):
        parts = [str(v).strip() for v in x if v is not None and str(v).strip()]
        return "|".join(parts)
    return str(x).strip()

def _safe_int(x: Any) -> Optional[int]:
    """Safely convert value to integer.
    
    Args:
        x: Value to convert
        
    Returns:
        Integer value or None if conversion fails
    """
    if x is None or x == "":
        return None
    try:
        return int(float(x))  # Handle float strings like "100.0"
    except (ValueError, TypeError):
        return None

def _normalize_ym_or_ymd(s: Any) -> str:
    """Normalize date string to YYYY-MM-DD format.
    
    Args:
        s: Date string in YYYY-MM or YYYY-MM-DD format
        
    Returns:
        Normalized date string or empty string
    """
    if s is None:
        return ""
    s = str(s).strip()
    # YYYY-MM format
    if len(s) == 7 and s[4] == "-":
        return s + "-01"
    # YYYY-MM-DD format
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return s
    return ""

def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate SHA256 hash of file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()

def _utc_now_iso() -> str:
    """Get current UTC time in ISO format."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _get_git_commit_hash() -> str:
    """Get current git commit hash for reproducibility.
    
    Returns:
        Git commit hash or empty string if not in git repo
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5
        )
        return out.decode("utf-8").strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return ""

def _safe_jsonl_write_line(fp, obj: Any) -> None:
    """Write object as JSON line to file."""
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _q_escape(s: str) -> str:
    """Quote string if it contains special characters."""
    s = s.strip()
    if not s:
        return s
    if any(ch.isspace() for ch in s) or any(ch in s for ch in ['"', ":", "(", ")", "{", "}", "[", "]"]):
        return f'"{s.replace(chr(34), r"\"")}"'
    return s

def _or_group(values: List[str]) -> str:
    """Create OR group from list of values."""
    vals = [v.strip() for v in values if v and v.strip()]
    if not vals:
        return ""
    if len(vals) == 1:
        return _q_escape(vals[0])
    return "(" + " OR ".join(_q_escape(v) for v in vals) + ")"

def build_query_params(cfg: CollectConfig) -> Dict[str, Any]:
    """Build ClinicalTrials.gov API v2 query parameters with reproducibility.
    
    Constructs query using AREA[FieldName]value syntax for filters.
    
    Args:
        cfg: Collection configuration
        
    Returns:
        Dictionary of query parameters
        
    Raises:
        ValueError: If query_term is invalid
    """
    if not cfg.query_term:
        logger.warning("Empty query_term, using wildcard '*'")
        
    base_term = (cfg.query_term or "").strip()
    clauses: List[str] = []
    if base_term:
        clauses.append(f"({base_term})")
    else:
        clauses.append("*")

    # Study type filter
    study_type = getattr(cfg, "study_type_filter", None)
    if study_type:
        clauses.append(f"AREA[StudyType]{_q_escape(study_type)}")

    # Phase filter
    phases = getattr(cfg, "phase_filter", None)
    if phases:
        phase_terms = [f"AREA[Phase]{_q_escape(p)}" for p in phases if p and str(p).strip()]
        if phase_terms:
            clauses.append(f"({' OR '.join(phase_terms)})")

    # Overall status filter
    statuses = getattr(cfg, "overall_status_filter", None)
    if statuses:
        status_terms = [f"AREA[OverallStatus]{_q_escape(s)}" for s in statuses if s and str(s).strip()]
        if status_terms:
            clauses.append(f"({' OR '.join(status_terms)})")

    # Date range
    start_from = getattr(cfg, "start_date_from", None)
    start_to = getattr(cfg, "start_date_to", None)
    if start_from and start_to:
        clauses.append(f"startDate:[{_q_escape(start_from)} TO {_q_escape(start_to)}]")

    query_term = " AND ".join(clauses)

    params: Dict[str, Any] = {
        "query.term": query_term,
        "pageSize": cfg.page_size,
    }

    # Extra params
    extra_params = getattr(cfg, "extra_params", None)
    if isinstance(extra_params, dict):
        for k, v in extra_params.items():
            if v is not None:
                params[k] = v

    return params

# Step 1: Smoke test
def api_smoke_test(cfg: CollectConfig) -> Tuple[bool, Dict[str, Any]]:
    """Test API connectivity and query validity.
    
    Args:
        cfg: Collection configuration
        
    Returns:
        Tuple of (success: bool, info: dict)
    """
    logger.info("Running API smoke test...")
    params = build_query_params(cfg)
    params["pageSize"] = min(int(params.get("pageSize", 5)), 5)

    try:
        resp = requests.get(cfg.base_url, params=params, timeout=cfg.timeout_s)
        status_code = resp.status_code
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API smoke test failed: {e}")
        return False, {"error": str(e), "error_type": type(e).__name__}
    except Exception as e:
        logger.error(f"Unexpected error in smoke test: {e}")
        return False, {"error": repr(e), "error_type": type(e).__name__}

    info = {
        "status_code": status_code,
        "endpoint": cfg.base_url,
        "params": params,
        "has_studies_key": isinstance(data, dict) and "studies" in data,
        "studies_len": len((data.get("studies") or []) if isinstance(data, dict) else []),
        "has_nextPageToken": bool(data.get("nextPageToken")) if isinstance(data, dict) else False,
        "top_keys": list(data.keys())[:20] if isinstance(data, dict) else [],
    }
    ok = bool(200 <= status_code < 300 and info["has_studies_key"])
    
    if ok:
        logger.info(f"✅ Smoke test passed: {info['studies_len']} studies found")
    else:
        logger.warning(f"⚠️ Smoke test failed or no studies found")
    
    return ok, info

# Step 2: Flatten
def flatten_study(study: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested trial JSON into flat dictionary.
    
    Args:
        study: Nested study JSON from API
        
    Returns:
        Flattened dictionary with key trial fields
    """
    ps = study.get("protocolSection", {}) or {}
    ident = ps.get("identificationModule", {}) or {}
    status = ps.get("statusModule", {}) or {}
    design = ps.get("designModule", {}) or {}
    sponsor = ps.get("sponsorCollaboratorsModule", {}) or {}
    oversight = ps.get("oversightModule", {}) or {}
    cond = ps.get("conditionsModule", {}) or {}
    arms = ps.get("armsInterventionsModule", {}) or {}

    design_info = design.get("designInfo", {}) or {}
    enrollment_info = design.get("enrollmentInfo", {}) or {}
    lead = sponsor.get("leadSponsor", {}) or {}

    start_date = _normalize_ym_or_ymd(_get_nested(status, ["startDateStruct", "date"]))
    completion_date = _normalize_ym_or_ymd(_get_nested(status, ["completionDateStruct", "date"]))
    masking = _get_nested(design_info, ["maskingInfo", "masking"]) or ""

    phases = design.get("phases", []) or []
    conditions = cond.get("conditions", []) or []
    arm_groups = arms.get("armGroups", []) or []
    interventions = arms.get("interventions", []) or []

    intervention_types = sorted({
        (i.get("type") or "").strip()
        for i in interventions
        if isinstance(i, dict) and i.get("type")
    })
    intervention_names = [
        i.get("name")
        for i in interventions
        if isinstance(i, dict) and i.get("name")
    ]

    return {
        "nct_id": ident.get("nctId", ""),
        "brief_title": ident.get("briefTitle", ""),
        "overall_status": status.get("overallStatus", ""),
        "start_date": start_date,
        "completion_date": completion_date,
        "study_type": design.get("studyType", ""),
        "phases": _to_pipe_str(phases),
        "allocation": design_info.get("allocation", ""),
        "intervention_model": design_info.get("interventionModel", ""),
        "primary_purpose": design_info.get("primaryPurpose", ""),
        "masking": masking,
        "enrollment_count": _safe_int(enrollment_info.get("count")),
        "enrollment_type": enrollment_info.get("type", ""),
        "lead_sponsor_name": lead.get("name", ""),
        "lead_sponsor_class": lead.get("class", ""),
        "has_dmc": oversight.get("oversightHasDmc", ""),
        "conditions": _to_pipe_str(conditions),
        "num_conditions": len(conditions),
        "intervention_types": _to_pipe_str(intervention_types),
        "intervention_names": _to_pipe_str(intervention_names),
        "num_interventions": len(interventions),
        "num_arms": len(arm_groups),
        "has_results": bool(study.get("hasResults", False)),
    }


def get_flat_columns() -> List[str]:
    """Get list of column names for flattened data."""
    return list(flatten_study({"protocolSection": {}}).keys())


# Step 3: API iterator 
def _create_session_with_retry() -> requests.Session:
    """Create requests session with automatic retry on failures.
    
    Returns:
        Configured requests.Session
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def iter_studies_from_api(cfg: CollectConfig) -> Iterator[Dict[str, Any]]:
    """Iterator yielding studies from ClinicalTrials.gov API with pagination.
    
    Args:
        cfg: Collection configuration
        
    Yields:
        Study dictionaries from API
        
    Raises:
        RuntimeError: If requests fail after all retries
    """
    session = _create_session_with_retry()
    params = build_query_params(cfg)
    
    logger.info(f"Starting API collection with query: {params.get('query.term', 'N/A')[:100]}...")

    # Optional resume token
    next_token: Optional[str] = getattr(cfg, "start_page_token", None)

    total_yielded = 0
    pages_fetched = 0
    request_failed = 0
    request_retried = 0

    max_pages = getattr(cfg, "max_pages", None)

    # Backoff settings
    base_delay = float(getattr(cfg, "retry_delay_s", 1.0))
    max_delay = float(getattr(cfg, "max_retry_delay_s", 30.0))
    jitter = float(getattr(cfg, "retry_jitter_s", 0.2))

    while True:
        if next_token:
            params["pageToken"] = next_token
        else:
            params.pop("pageToken", None)

        data: Optional[Dict[str, Any]] = None
        last_err: Optional[Exception] = None

        for attempt in range(1, cfg.max_retries + 1):
            try:
                resp = session.get(cfg.base_url, params=params, timeout=cfg.timeout_s)

                # Handle rate limit explicitly
                if resp.status_code == 429:
                    request_retried += 1
                    retry_after = resp.headers.get("Retry-After")
                    wait_s = float(retry_after) if retry_after and retry_after.isdigit() else min(max_delay, base_delay * (2 ** (attempt - 1)))
                    logger.warning(f"Rate limited (429). Waiting {wait_s:.1f}s before retry {attempt}/{cfg.max_retries}")
                    time.sleep(wait_s + (jitter * attempt))
                    continue

                resp.raise_for_status()
                data = resp.json()
                last_err = None
                break
            except requests.exceptions.Timeout as e:
                last_err = e
                request_retried += 1
                wait_s = min(max_delay, base_delay * (2 ** (attempt - 1)))
                logger.warning(f"Request timeout. Retry {attempt}/{cfg.max_retries} after {wait_s:.1f}s")
                time.sleep(wait_s + (jitter * attempt))
            except requests.exceptions.RequestException as e:
                last_err = e
                request_retried += 1
                wait_s = min(max_delay, base_delay * (2 ** (attempt - 1)))
                logger.warning(f"Request failed: {e}. Retry {attempt}/{cfg.max_retries} after {wait_s:.1f}s")
                time.sleep(wait_s + (jitter * attempt))
            except Exception as e:
                last_err = e
                request_retried += 1
                wait_s = min(max_delay, base_delay * (2 ** (attempt - 1)))
                logger.error(f"Unexpected error: {e}. Retry {attempt}/{cfg.max_retries}")
                time.sleep(wait_s + (jitter * attempt))

        if data is None:
            request_failed += 1
            error_msg = f"Failed after {cfg.max_retries} retries. Last error: {repr(last_err)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        pages_fetched += 1
        if max_pages is not None and pages_fetched >= int(max_pages):
            return

        studies = data.get("studies", []) or []
        if not studies:
            return

        for st in studies:
            yield st
            total_yielded += 1
            if total_yielded >= cfg.max_studies:
                return

        next_token = data.get("nextPageToken")
        if not next_token:
            return

        time.sleep(cfg.sleep_s)

# Step 4: Collect -> raw jsonl + processed csv + log
def collect_raw_jsonl(cfg: CollectConfig) -> Tuple[Path, Path]:
    """Collect studies from ClinicalTrials.gov API and save as JSONL.
    
    Creates three files:
    - Raw JSONL with all study records
    - Log file with collection events
    - Metadata JSON with reproducibility info
    
    Args:
        cfg: Collection configuration
        
    Returns:
        Tuple of (raw_jsonl_path, log_path)
        
    Raises:
        RuntimeError: If API collection fails
        IOError: If file writing fails
    """
    logger.info("Starting data collection from ClinicalTrials.gov API")
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = cfg.raw_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_term = "".join(ch if ch.isalnum() else "_" for ch in cfg.query_term.strip().lower())
    
    # Versioning strategy: snapshot per run
    raw_path = cfg.raw_dir / f"ctgov_raw.jsonl"
    log_path = logs_dir / f"ctgov_collect.log"
    meta_path = cfg.raw_dir / f"ctgov_metadata.json"

    params = build_query_params(cfg)
    endpoint = cfg.base_url
    git_hash = _get_git_commit_hash()

    prediction_time = getattr(cfg, "prediction_time", "t0 = StartDate")
    label_def = getattr(cfg, "label_definition", "Fail={Withdrawn,Terminated}; Success={Completed}")

    run_meta = {
        "run_started_at": _utc_now_iso(),
        "run_tag": run_tag,
        "endpoint": endpoint,
        "params": params,
        "timeout_s": cfg.timeout_s,
        "max_retries": cfg.max_retries,
        "retry_delay_s": getattr(cfg, "retry_delay_s", None),
        "sleep_s": cfg.sleep_s,
        "max_studies": cfg.max_studies,
        "git_commit": git_hash,
        "prediction_time": prediction_time,
        "label_definition": label_def,
        "raw_path": str(raw_path),
        "log_path": str(log_path),
    }

    total_raw = 0
    json_parse_errors = 0
    
    logger.info(f"Output files:")
    logger.info(f"  Raw JSONL: {raw_path}")
    logger.info(f"  Log: {log_path}")
    logger.info(f"  Metadata: {meta_path}")

    with open(raw_path, "w", encoding="utf-8") as f_raw, open(log_path, "w", encoding="utf-8") as f_log:
        _safe_jsonl_write_line(f_log, {"event": "run_start", **run_meta})

        pbar = tqdm(
            total=cfg.max_studies,
            desc="Downloading studies",
            unit="study",
            leave=True,
            file=sys.stdout,
            miniters=1,
            mininterval=0.2
        )

        try:
            for st in iter_studies_from_api(cfg):
                try:
                    f_raw.write(json.dumps(st, ensure_ascii=False) + "\n")
                    total_raw += 1
                except Exception as e:
                    json_parse_errors += 1
                    logger.debug(f"JSON serialization error: {e}")

                pbar.update(1)
                if total_raw >= cfg.max_studies:
                    break
        except KeyboardInterrupt:
            logger.warning("Collection interrupted by user")
            raise
        except Exception as e:
            logger.error(f"Collection failed: {e}")
            raise
        finally:
            pbar.close()

        raw_sha256 = _sha256_file(raw_path)

        run_end = {
            "event": "run_finish",
            "run_finished_at": _utc_now_iso(),
            "total_raw_records": total_raw,
            "json_write_errors": json_parse_errors,
            "raw_sha256": raw_sha256,
        }
        _safe_jsonl_write_line(f_log, run_end)

    # Write metadata.json (reproducibility)
    run_meta_out = {**run_meta, **run_end}
    with open(meta_path, "w", encoding="utf-8") as f_meta:
        json.dump(run_meta_out, f_meta, ensure_ascii=False, indent=2)
    
    logger.info(f"Collection complete: {total_raw} studies collected")
    logger.info(f"   SHA256: {raw_sha256[:16]}...")
    if json_parse_errors > 0:
        logger.warning(f"   ⚠️ {json_parse_errors} JSON write errors")

    return raw_path, log_path

# Step 5: Flatten from RAW
def flatten_raw_jsonl_to_csv(
    raw_jsonl_path: Path,
    csv_out_path: Optional[Path] = None,
    dedup_by_nct: bool = True,
    keep_latest_by: str = "last_update_post_date"
) -> Path:
    """Flatten raw JSONL to CSV with optional deduplication.
    
    Args:
        raw_jsonl_path: Path to raw JSONL file
        csv_out_path: Output CSV path (auto-generated if None)
        dedup_by_nct: Whether to deduplicate by NCT ID
        keep_latest_by: Field to use for keeping latest record
        
    Returns:
        Path to output CSV file
        
    Raises:
        FileNotFoundError: If raw_jsonl_path doesn't exist
        IOError: If file reading/writing fails
    """
    if not raw_jsonl_path.exists():
        raise FileNotFoundError(f"Raw JSONL not found: {raw_jsonl_path}")
    
    logger.info(f"Flattening {raw_jsonl_path.name} to CSV...")
    
    # Pass 1: read all records and deduplicate
    best: Dict[str, Dict[str, Any]] = {}
    best_key: Dict[str, str] = {}

    parse_errors = 0
    total_lines = 0

    with open(raw_jsonl_path, "r", encoding="utf-8") as f_in:
        for line in f_in:
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                st = json.loads(line)
            except Exception:
                parse_errors += 1
                continue

            row = flatten_study(st)
            nct = row.get("nct_id", "")
            if not nct:
                continue

            if not dedup_by_nct:
                # If no dedup, store with unique key
                best[f"{nct}__{total_lines}"] = row
                continue

            k = (row.get(keep_latest_by) or "")
            if nct not in best or k > (best_key.get(nct) or ""):
                best[nct] = row
                best_key[nct] = k

    # Pass 2: write CSV
    processed_dir = raw_jsonl_path.parents[1] / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    if csv_out_path is None:
        csv_out_path = processed_dir / f"ctgov_flat.csv"

    fieldnames = get_flat_columns()
    with open(csv_out_path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in best.values():
            writer.writerow(row)
    
    kept_records = len(best)
    dup_skipped = total_lines - parse_errors - kept_records
    
    logger.info(f"✅ Flattening complete: {csv_out_path.name}")
    logger.info(f"   Total lines: {total_lines}")
    logger.info(f"   Kept records: {kept_records}")
    logger.info(f"   Parse errors: {parse_errors}")
    if dedup_by_nct and dup_skipped > 0:
        logger.info(f"   Duplicates skipped: {dup_skipped}")

    return csv_out_path

# Step 6: Raw quality report
def raw_quality_report(raw_jsonl_path: Path, max_lines: Optional[int] = None) -> Dict[str, Any]:
    """Generate quality report for raw JSONL data.
    
    Analyzes:
    - JSON parse error rate
    - Missing critical fields rate
    - Top-level key coverage (schema drift detection)
    
    Args:
        raw_jsonl_path: Path to raw JSONL file
        max_lines: Maximum lines to analyze (None = all)
        
    Returns:
        Dictionary with quality metrics
    """
    logger.info(f"Generating quality report for {raw_jsonl_path.name}...")
    
    critical_paths = {
        "nct_id": ["protocolSection", "identificationModule", "nctId"],
        "overall_status": ["protocolSection", "statusModule", "overallStatus"],
        "start_date": ["protocolSection", "statusModule", "startDateStruct", "date"],
        "study_type": ["protocolSection", "designModule", "studyType"],
    }

    total = 0
    parse_errors = 0
    missing_counts = {k: 0 for k in critical_paths}
    top_keys_seen: Dict[str, int] = {}

    with open(raw_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if max_lines is not None and total >= max_lines:
                break
            total += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                parse_errors += 1
                continue

            if isinstance(obj, dict):
                for k in obj.keys():
                    top_keys_seen[k] = top_keys_seen.get(k, 0) + 1

            for name, path in critical_paths.items():
                val = _get_nested(obj, path)
                if val in (None, "", []):
                    missing_counts[name] += 1

    report = {
        "raw_path": str(raw_jsonl_path),
        "total_lines_read": total,
        "json_parse_errors": parse_errors,
        "parse_error_rate": parse_errors / max(total, 1),
        "missing_critical_counts": missing_counts,
        "missing_critical_rates": {k: round(missing_counts[k] / max(total, 1), 4) for k in missing_counts},
        "top_level_keys_coverage": dict(sorted(top_keys_seen.items(), key=lambda x: -x[1])[:20]),
    }
    
    logger.info(f"Quality Report Summary:")
    logger.info(f"  Lines analyzed: {total}")
    logger.info(f"  Parse errors: {parse_errors} ({report['parse_error_rate']:.2%})")
    for field, rate in report['missing_critical_rates'].items():
        status = "Complete" if rate < 0.1 else "Error" if rate < 0.5 else "Fail"
        logger.info(f"  {status} {field}: {rate:.1%} missing")
    
    return report
