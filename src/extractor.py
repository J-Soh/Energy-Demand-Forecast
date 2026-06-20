"""
Data extraction module for fetching Electrical Energy market data from NEMS API.
"""

from __future__ import annotations

import logging
from io import StringIO

import pandas as pd
import requests

logger = logging.getLogger(__name__)


def _process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and normalise raw API response into a consistent schema.

    Args:
        df: Raw DataFrame from NEMS data

    Returns:
        Processed DataFrame with "timestamp", "Demand (MW)", "Solar (MW)", "USEP ($/MWh)" columns
    """

    df.columns = [c.strip() for c in df.columns]

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    df["start_time"] = df["Period"].str.split("-").str[0]

    df["timestamp"] = pd.to_datetime(
    df["Date"].dt.strftime("%Y-%m-%d") + " " + df["start_time"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    df = df[[
        "timestamp",
        "Demand (MW)",
        "Solar (MW)",
        "USEP ($/MWh)"
    ]].copy()

    df.columns = [
        "timestamp",
        "demand",
        "solar",
        "usep"
        ]

    df = df.dropna()

    return df


def _download_data(
    target_date: str
) -> pd.DataFrame:
    """
    Download data from Singapore NEMS.

    Arg:
        target_date: Yesterday date to extract data from.

    Returns:
        Processed DataFrame or None if extraction fails.
    """
    url = "https://www.nems.emcsg.com/api/sitecore/DataSync/DataDownload"
    # (Optional) Manually downloaded from https://www.nems.emcsg.com/NEMS-Market-Trading-Reports.

    params = {"value": 10, "fromDate": target_date, "toDate": target_date, "tpcValue": 1}

    # Crucial step: Mirror a real browser to bypass security blocks
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    print(f"Requesting data for {target_date}...")
    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        # Check if the site returned an error page disguised as success
        if "No data found" in response.text or "Error" in response.text:
            print(f"No market data available yet for {target_date}.")
            return None

        # Convert raw text string directly into a Pandas Dataframe
        df = pd.read_csv(StringIO(response.text))
        print("Data downloaded successfully!")
        return df
    else:
        print(f"Request failed with status code: {response.status_code}")
        return None

    return pd.DataFrame()


def extract_data(target_date: str) -> pd.DataFrame:
    """
    Download and normalise NEMS data for a single day.

    Args:
        target_date: Yesterday date to extract data from.

    Returns:
        DataFrame with columns: ['timestamp', 'demand', 'solar', 'usep']
    """
    all_frames: list[pd.DataFrame] = []

    raw_df = _download_data(target_date)

    processed = _process_dataframe(raw_df)
    all_frames.append(processed)

    if not all_frames:
        return pd.DataFrame()

    nems_df = pd.concat(all_frames, ignore_index=True)

    logger.info("Extracted %d unique data", len(nems_df))
    return nems_df
