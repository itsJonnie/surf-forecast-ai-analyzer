"""
Core surf forecasting logic for the Surf Forecast AI Analyzer.

This module loads a small dataset of sample surf conditions and implements
functions to retrieve forecasts, compute a quality score, and generate
natural‑language summaries.  It is deliberately simple so it can run on
commodity hardware without any paid APIs.
"""

from __future__ import annotations

import pandas as pd
import math
from functools import lru_cache
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "surf_data.csv"


@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    """Load the surf forecast CSV into a pandas DataFrame.

    The result is cached so repeated calls are cheap.
    Returns a DataFrame with string dates parsed as strings (not datetime).
    """
    df = pd.read_csv(DATA_PATH)
    # Normalize location names for case‑insensitive matching
    df["location_norm"] = df["location"].str.strip().str.lower()
    return df


def retrieve_forecast(location: str, date: str) -> dict | None:
    """Retrieve the forecast row for a given location and date.

    Args:
        location: Name of the surf spot (case‑insensitive).
        date: Date string in ``YYYY‑MM‑DD`` format.

    Returns:
        A dictionary of the matching row's values (excluding the normalized column),
        or ``None`` if no match exists.
    """
    df = load_data()
    loc_norm = location.strip().lower()
    match = df[(df["location_norm"] == loc_norm) & (df["date"] == date)]
    if match.empty:
        return None
    row = match.iloc[0].to_dict()
    row.pop("location_norm", None)
    return row


def compute_quality_score(row: dict) -> float:
    """Compute a surf quality score from 1 (poor) to 10 (epic).

    This heuristic combines wave height, period, wind speed, and tide.  It
    punishes strong onshore winds and low wave height, and rewards longer
    periods and moderate tides.  The score is capped between 1 and 10.
    """
    height = float(row["wave_height_ft"])
    period = float(row["wave_period_s"])
    wind_speed = float(row["wind_speed_mph"])
    tide = float(row["tide_ft"])

    # Base score from wave height (1–6 ft good, above 6 excellent)
    base = min(height * 1.5, 15)
    # Add bonus for wave period (longer periods mean more power)
    base += period * 0.8
    # Penalise high wind speeds; slower winds are good
    base -= wind_speed * 0.5
    # Tides: assume 2–3 ft is ideal; deviate adds penalty
    if 2.0 <= tide <= 3.0:
        base += 2.0
    else:
        base -= abs(tide - 2.5) * 1.0
    # Normalise and clip between 1 and 10
    score = max(1.0, min(10.0, base / 3.0))
    return round(score, 1)


def score_to_description(score: float) -> str:
    """Map a numeric score to a qualitative description."""
    if score >= 8:
        return "epic"
    elif score >= 6:
        return "good"
    elif score >= 4:
        return "fair"
    else:
        return "poor"


def generate_summary(row: dict) -> str:
    """Generate a natural‑language summary of the surf forecast for a row."""
    score = compute_quality_score(row)
    quality = score_to_description(score)
    return (
        f"Surf forecast for {row['location']} on {row['date']}: "
        f"waves around {row['wave_height_ft']} ft with a {row['wave_period_s']} second period, "
        f"winds {row['wind_speed_mph']} mph from {row['wind_direction']}, tide {row['tide_ft']} ft. "
        f"Overall conditions look {quality} (score {score})."
    )


def outlook_for_location(location: str) -> pd.DataFrame:
    """Return the outlook DataFrame for the given location sorted by date.

    Returns the subset of the dataset for the location, adding a ``quality`` column.
    """
    df = load_data()
    loc_norm = location.strip().lower()
    subset = df[df["location_norm"] == loc_norm].copy()
    if subset.empty:
        return subset
    subset.sort_values("date", inplace=True)
    subset["quality"] = subset.apply(lambda r: compute_quality_score(r), axis=1)
    return subset