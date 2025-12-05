import os
import time
import json
import requests
import pandas as pd
import numpy as np

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

class ClinicalTrialsAPI:
    """
    Loader for ClinicalTrials.gov API v2.
    Fetches study records using the nextPageToken-based pagination system.
    """

    def __init__(self, base_url=BASE_URL, page_size=1000, sleep=0.2, max_retries=3):
        self.base_url = base_url
        self.page_size = page_size         
        self.sleep = sleep                  # Delay between pages to avoid rate limits
        self.max_retries = max_retries      # Retry attempts for failed requests

    def _request(self, params):
        """
        Send a GET request with retry logic.
        Returns JSON if successful, otherwise raises exception.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            
            except requests.exceptions.Timeout:
                print(f"[ERROR] Timeout on attempt {attempt}/{self.max_retries}")
                time.sleep(1.5 * attempt)

            except Exception as e:
                print(f"[ERROR] Attempt {attempt}/{self.max_retries} failed: {e}")
                time.sleep(1.5 * attempt)

        raise RuntimeError("[FATAL] All API retries failed.")

    def fetch_batch(self, page_token=None):
        """
        Fetch one page of studies.
        Uses pageSize + optional nextPageToken.
        """
        params = {
            "pageSize": 1000,
            "format": "json",
            "fields": "protocolSection,derivedSection,hasResults"
        }

        if page_token is not None:
            params["pageToken"] = page_token

        return self._request(params)

    def fetch_n_studies(self, n=5000):
        """
        Collect approximately N studies using API v2 pagination.
        Stops when it reaches N or when no nextPageToken is returned.
        """
        print(f"[INFO] Fetching up to {n} studies...")

        studies = []
        page_token = None

        while len(studies) < n:
            batch = self.fetch_batch(page_token=page_token)
            batch_studies = batch.get("studies", [])
            studies.extend(batch_studies)

            print(
                f"[INFO] Retrieved batch with {len(batch_studies)} studies "
                f"(total: {len(studies)})"
            )

            # Move to next page
            page_token = batch.get("nextPageToken")
            if not page_token:
                print("[INFO] No nextPageToken returned. Reached last page.")
                break

            time.sleep(self.sleep)

        print(f"[SUCCESS] Total collected: {len(studies)}")
        return studies[:n]
    
# -------------------------------------------------------------
# Flatten one JSON study object
# -------------------------------------------------------------
def flatten_study(study):
    """
    Convert one ClinicalTrials.gov study JSON object
    into a flat dictionary suitable for pandas DataFrame.
    Only essential features for risk analysis are extracted.
    """
    protocol = study.get("protocolSection", {})
    derived = study.get("derivedSection", {}) 

    row = {}

    # Identification
    ident = protocol.get("identificationModule", {})
    row["nct_id"] = ident.get("nctId")

    # Status
    status = protocol.get("statusModule", {})
    row["overall_status"] = status.get("overallStatus")
    row["start_date"] = status.get("startDateStruct", {}).get("date")
    row["completion_date"] = status.get("completionDateStruct", {}).get("date")
    row["primary_completion_date"] = status.get("primaryCompletionDateStruct", {}).get("date")


    # Design
    design = protocol.get("designModule", {})
    design_info = design.get("designInfo", {}) or {}
    phases =  design.get("phases", []) or []
    masking_info = design_info.get("maskingInfo", {}) or {}

    row["study_type"] = design.get("studyType")
    row["phases"] = ", ".join(phases) if phases else None
    row["allocation"] = design_info.get("allocation")
    row["intervention_model"] = design_info.get("interventionModel")
    row["masking"] = masking_info.get("masking")
    row["primary_purpose"] = design_info.get("primaryPurpose")

    # Conditions
    cond = protocol.get("conditionsModule", {})
    conditions = cond.get("conditions", []) or []

    row["conditions"] = conditions 
    row["num_conditions"] = len(conditions)

    # Arms & Interventions
    arms_mod = protocol.get("armsInterventionsModule", {})
    arm_groups = arms_mod.get("armGroups", []) or []
    interventions = arms_mod.get("interventions", []) or []

    row["num_arms"] = len(arm_groups)
    row["num_interventions"] = len(interventions)

    # Outcomes
    outcomes = protocol.get("outcomesModule", {}) or {}
    row["num_primary_outcomes"] = len(outcomes.get("primaryOutcomes", []) or [])
    row["num_secondary_outcomes"] = len(outcomes.get("secondaryOutcomes", []) or [])

    # Sponsors
    sponsor_mod = protocol.get("sponsorCollaboratorsModule", {})
    resp = sponsor_mod.get("responsibleParty", {}) or {}
    lead = sponsor_mod.get("leadSponsor", {}) or {}
    collabs = sponsor_mod.get("collaborators", []) or []

    row["responsible_party_type"] = resp.get("type", "")
    row["lead_sponsor_class"] = lead.get("class", "")
    row["num_collaborators"] = len(collabs)

    # Eligibility
    elig_mod = protocol.get("eligibilityModule", {})

    row["healthy_volunteers"] = elig_mod.get("healthyVolunteers")
    row["sex_eligibility"] = elig_mod.get("sex", "")
    row["min_age"] = elig_mod.get("minimumAge")
    row["max_age"] = elig_mod.get("maximumAge")

    # Locations
    clm = protocol.get("contactsLocationsModule", {})
    locs = clm.get("locations", []) or []

    row["num_locations"] = len(locs)
    row["num_countries"] = len({l.get("country") for l in locs if l.get("country")})

    # MeSH terms
    cond_browse = derived.get("conditionBrowseModule", {}) or {}
    interv_browse = derived.get("interventionBrowseModule", {}) or {}

    row["num_condition_mesh_terms"] = len(cond_browse.get("meshes", []) or [])
    row["num_intervention_mesh_terms"] = len(interv_browse.get("meshes", []) or [])

    # Results flag
    row["has_results"] = study.get("hasResults", False)

    return row