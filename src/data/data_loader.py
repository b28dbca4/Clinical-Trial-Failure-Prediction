import os
import time
import json
import requests


BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


class ClinicalTrialsAPI:
    """
    Loader for ClinicalTrials.gov API v2.
    Fetches study records using the nextPageToken-based pagination system.
    """

    def __init__(self, base_url=BASE_URL, page_size=100, sleep=0.2, max_retries=3):
        self.base_url = base_url
        self.page_size = page_size          # Maximum allowed is 100
        self.sleep = sleep                  # Delay between pages to avoid rate limits
        self.max_retries = max_retries      # Retry attempts for failed requests

    def _request(self, params):
        """
        Send a GET request with retry logic.
        Returns JSON if successful, otherwise raises exception.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                return response.json()

            except Exception as e:
                print(f"[ERROR] Attempt {attempt}/{self.max_retries} failed: {e}")
                time.sleep(1.5 * attempt)

        raise RuntimeError("[FATAL] All API retries failed.")

    def fetch_batch(self, page_token=None):
        """
        Fetch one page of studies.
        Uses pageSize + optional nextPageToken.
        """
        params = {"pageSize": self.page_size}

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

    @staticmethod
    def save_json(data, path):
        """
        Save collected studies into a JSON file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"[INFO] JSON saved to: {path}")
