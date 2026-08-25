"""
EIA (U.S. Energy Information Administration) API client.

Pulls weekly U.S. commercial crude oil inventory data (the "storage" side
of the cost-of-carry model). Requires a free API key from:
    https://www.eia.gov/opendata/register.php

EIA API v2 docs: https://www.eia.gov/opendata/documentation.php

Series used:
    PET.WCESTUS1.W  -- Weekly U.S. Ending Stocks of Crude Oil (thousand barrels)
"""
from __future__ import annotations

import os
import requests
import pandas as pd

EIA_BASE_URL = "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"
CRUDE_STORAGE_SERIES_ID = "WCESTUS1"  # weekly U.S. ending stocks, crude oil


class EIAClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("EIA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No EIA API key found. Set the EIA_API_KEY environment variable "
                "or pass api_key= explicitly. Get a free key at "
                "https://www.eia.gov/opendata/register.php"
            )

    def get_crude_storage(self, start: str, end: str) -> pd.DataFrame:
        """
        Fetch weekly U.S. crude oil storage levels.

        Parameters
        ----------
        start, end : str
            Dates in 'YYYY-MM-DD' format.

        Returns
        -------
        pd.DataFrame with columns ['date', 'storage_kbbl']
        """
        params = {
            "api_key": self.api_key,
            "frequency": "weekly",
            "data[0]": "value",
            "facets[series][]": CRUDE_STORAGE_SERIES_ID,
            "start": start,
            "end": end,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
        }
        resp = requests.get(EIA_BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        rows = payload.get("response", {}).get("data", [])
        if not rows:
            raise RuntimeError(
                f"EIA API returned no data for series {CRUDE_STORAGE_SERIES_ID} "
                f"between {start} and {end}. Check your API key and date range."
            )

        df = pd.DataFrame(rows)
        df = df.rename(columns={"period": "date", "value": "storage_kbbl"})
        df["date"] = pd.to_datetime(df["date"])
        df["storage_kbbl"] = pd.to_numeric(df["storage_kbbl"])
        return df[["date", "storage_kbbl"]].sort_values("date").reset_index(drop=True)


def load_storage_data_cached(start: str, end: str, cache_path: str = "data/crude_storage.csv") -> pd.DataFrame:
    """
    Convenience wrapper: load from local cache if present, else hit the API
    and cache the result. Keeps you from burning API calls during development.
    """
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["date"])
        return df

    client = EIAClient()
    df = client.get_crude_storage(start, end)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df
