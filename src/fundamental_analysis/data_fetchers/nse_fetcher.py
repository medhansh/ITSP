"""NIFTY 500 constituent list fetcher.

NOTE: requires outbound internet access, which this scaffold's build
environment did not have (verified: NSE endpoints were unreachable from the
sandbox). Run this from your own machine to populate
data/universe/nifty500_list.csv.
"""
from __future__ import annotations

import pandas as pd
import requests

from src.common.logging_utils import get_logger

logger = get_logger(__name__)

NIFTY500_CSV_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"


def fetch_nifty500_list(url: str = NIFTY500_CSV_URL, timeout: int = 15) -> pd.DataFrame:
    """Download and normalize the official NIFTY 500 constituent list.

    Returns columns: symbol, name, sector, industry, series — matching
    data/universe/nifty500_list.csv's schema.
    """
    headers = {"User-Agent": "Mozilla/5.0"}  # NSE blocks requests without a UA
    logger.info("Fetching NIFTY500 list from %s", url)
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    from io import StringIO

    raw = pd.read_csv(StringIO(resp.text))
    # NSE's raw column names vary by export; normalize the common ones.
    rename_map = {
        "Symbol": "symbol",
        "Company Name": "name",
        "Industry": "sector",
        "Series": "series",
    }
    df = raw.rename(columns=rename_map)
    df["industry"] = df.get("industry", df.get("sector"))
    keep = ["symbol", "name", "sector", "industry", "series"]
    return df[[c for c in keep if c in df.columns]]


def save_universe_list(df: pd.DataFrame, out_path: str = "data/universe/nifty500_list.csv") -> None:
    df.to_csv(out_path, index=False)
    logger.info("Saved %d symbols to %s", len(df), out_path)
