"""
Core surf forecasting logic for the Surf Forecast AI Analyzer.

This module loads a small dataset of sample surf conditions and implements
functions to retrieve forecasts, compute a quality score, and generate
natural‑language summaries.  It is deliberately simple so it can run on
commodity hardware without any paid APIs.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
import math
import pandas as pd
import requests

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "surf_data.csv"

# Coordinates for known surf spots (latitude, longitude).
# These coordinates are approximate and can be adjusted if needed.
LOCATION_COORDS: dict[str, tuple[float, float]] = {
    "malibu": (34.0259, -118.7798),
    "huntington beach": (33.6595, -117.9988),
    "santa monica": (34.0100, -118.4960),
}

def _degrees_to_cardinal(deg: float) -> str:
    """Convert degrees into one of the eight cardinal directions.

    Args:
        deg: A direction in degrees (0–360).

    Returns:
        A string representing the cardinal direction (e.g. "N", "NE").
    """
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    # Normalize degrees and map to 0–7
    idx = int((deg % 360) / 45.0 + 0.5) % 8
    return dirs[idx]

def fetch_live_forecast(location: str) -> dict:
    """Fetch live surf conditions for the given location using the Stormglass API.

    This function looks up the latitude and longitude for the provided location,
    sends a request to the Stormglass API using the `STORMGLASS_API_KEY` environment
    variable, and transforms the response into a dictionary matching the
    expected forecast format.

    Note:
        The Stormglass free tier allows a limited number of requests per day. This
        function will raise an exception if the API key is missing or the
        request fails. Callers should catch exceptions and fall back to
        historical data when necessary.

    Args:
        location: Name of the surf spot (case‑insensitive).

    Returns:
        A dictionary with keys matching the forecast CSV columns (except date),
        containing the latest forecast values.
    """
    coords = LOCATION_COORDS.get(location.strip().lower())
    if coords is None:
        raise KeyError(f"Unknown location {location}")
    api_key = os.getenv("STORMGLASS_API_KEY")
    if not api_key:
        raise RuntimeError("STORMGLASS_API_KEY not set in the environment")
    lat, lon = coords
    params = {
        "lat": lat,
        "lng": lon,
        "params": "waveHeight,wavePeriod,windSpeed,windDirection,tide",
        "source": "noaa",
    }
    headers = {"Authorization": api_key}
    response = requests.get(
        "https://api.stormglass.io/v2/weather/point",
        params=params,
        headers=headers,
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    if not data.get("hours"):
        raise ValueError("No hourly data returned from Stormglass API")
    hour = data["hours"][0]
    # Extract parameters, converting units where necessary. Some values may be None.
    wave_height_m = hour.get("waveHeight", {}).get("noaa")
    wave_period_s = hour.get("wavePeriod", {}).get("noaa")
    wind_speed_ms = hour.get("windSpeed", {}).get("noaa")
    wind_dir_deg = hour.get("windDirection", {}).get("noaa")
    tide_m = hour.get("tide", {}).get("sg")
    return {
        "location": location.title(),
        "wave_height_ft": round(wave_height_m * 3.28084, 1) if wave_height_m is not None else 0.0,
        "wave_period_s": wave_period_s if wave_period_s is not None else 0.0,
        "wind_speed_mph": round(wind_speed_ms * 2.23694, 1) if wind_speed_ms is not None else 0.0,
        "wind_direction": _degrees_to_cardinal(wind_dir_deg) if wind_dir_deg is not None else "N",
        "tide_ft": round(tide_m * 3.28084, 1) if tide_m is not None else 0.0,
    }


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
    """Retrieve a surf forecast for the given location and date.

    This function first attempts to fetch live data from the Stormglass API
    (if an API key is configured). If the live fetch fails for any reason
    (unknown location, missing API key, network error, etc.), it falls back
    to the static CSV dataset. When falling back, the date must match one of
    the entries in the dataset; otherwise ``None`` is returned.

    Args:
        location: Name of the surf spot (case‑insensitive).
        date: Date string in ``YYYY‑MM‑DD`` format.

    Returns:
        A dictionary of forecast values. When live data is used, the ``date``
        field of the returned dict will be set to the requested date. When
        static data is used, the date comes from the CSV row.
    """
    # Attempt live fetch first
    try:
        live_row = fetch_live_forecast(location)
        # Insert the requested date so downstream functions can reference it
        live_row["date"] = date
        return live_row
    except Exception:
        # Ignore any errors and fall back to CSV
        pass
    # Look up in the static dataset
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