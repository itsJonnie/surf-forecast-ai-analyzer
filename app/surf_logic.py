"""
Core surf forecasting logic for the Surf Forecast AI Analyzer.

This module loads a small dataset of sample surf conditions and implements
functions to retrieve forecasts, compute a quality score, and generate
natural‑language summaries.  It is deliberately simple so it can run on
commodity hardware without any paid APIs.
"""

from __future__ import annotations

import os
import json
from functools import lru_cache
from pathlib import Path
import math
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "surf_data.csv"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"

# Coordinates for known surf spots (latitude, longitude).
# These coordinates are approximate and can be adjusted if needed.
# Note: use runtime typing compatible with Python 3.9 in new code.
LOCATION_COORDS: Dict[str, tuple] = {
    "malibu": (34.0259, -118.7798),
    "huntington beach": (33.6595, -117.9988),
    "santa monica": (34.0100, -118.4960),
    "venice": (33.9941, -118.4527),
    "will rogers state beach": (34.0351, -118.5367),
}

# Spot orientation: degrees waves travel toward beach (approx).
# Offshore ~ orientation + 180 (+/- 45 deg)
SPOT_ORIENTATION: Dict[str, int] = {
    "Malibu": 220,
    "Huntington Beach": 225,
    "Santa Monica": 220,
    "Venice": 220,
}

# Domain tuning constants
SIZE_TIERS = {
    "micro": 1.5,     # < 1.5 ft
    "small": 3.0,     # 1.5–3 ft
    "chest": 5.0,     # 3–5 ft
    "head": 7.0,      # 5–7 ft
}
PERIOD_TIERS = {
    "weak": 9.0,      # < 9 s
    "medium": 12.0,   # 9–12 s
}
WIND_THRESH = {
    "ideal": 7.0,     # offshore <= 7 mph
    "side": 10.0,     # sideshore tolerable <= 10 mph
}


def _load_env_from_dotenv() -> None:
    """Lightweight .env loader.

    Loads KEY=VALUE pairs from a .env file at the project root, only setting
    variables that are not already present in the process environment. Lines
    starting with '#' are ignored. Quotes around values are stripped.
    """
    root = Path(__file__).resolve().parent.parent
    dotenv_path = root / ".env"
    if not dotenv_path.exists():
        return
    try:
        for line in dotenv_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        # Silently ignore .env parse errors; env vars can still be set normally
        pass


# Load .env on import so env vars like STORMGLASS_API_KEY are available
_load_env_from_dotenv()


def _ensure_cache_dir() -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If cache directory cannot be created, we will simply skip file caching
        pass


def _cache_path(location: str, date: str) -> Path:
    safe_loc = "".join(c for c in location.strip().lower() if c.isalnum() or c in ("-", "_"))
    return CACHE_DIR / f"live_{safe_loc}_{date}.json"


def _read_live_cache(location: str, date: str) -> dict | None:
    p = _cache_path(location, date)
    try:
        if not p.exists():
            return None
        obj = json.loads(p.read_text())
        exp = obj.get("_expires")
        if not exp:
            return None
        if datetime.now(timezone.utc) >= datetime.fromisoformat(exp):
            return None
        data = obj.get("data")
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _write_live_cache(location: str, date: str, data: dict) -> None:
    _ensure_cache_dir()
    try:
        # Expire at end of the requested UTC day, or in 6 hours, whichever is sooner
        now_utc = datetime.now(timezone.utc)
        try:
            day_end = datetime.fromisoformat(f"{date}T23:59:59+00:00")
        except Exception:
            day_end = now_utc + timedelta(hours=6)
        expires_at = min(day_end, now_utc + timedelta(hours=6))
        payload = {"_expires": expires_at.isoformat(), "data": data}
        _cache_path(location, date).write_text(json.dumps(payload))
    except Exception:
        # Best-effort cache write; ignore failures
        pass

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

def _cardinal_to_degrees(card: str) -> Optional[float]:
    """Approximate degrees from a cardinal like 'N', 'NE', etc."""
    if not isinstance(card, str):
        return None
    m = {
        "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
        "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
        "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
        "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
    }
    return m.get(card.strip().upper())

def _angular_distance(a: float, b: float) -> float:
    """Smallest absolute angular distance in degrees between two bearings."""
    diff = abs((a - b) % 360.0)
    return min(diff, 360.0 - diff)

def _classify_wind_quality(spot_name: str, wind_dir_input: Any, wind_speed_mph: float) -> Dict[str, Any]:
    """Classify wind relative to spot: offshore/side/onshore and quality label.

    Returns a dict { kind: 'offshore'|'side'|'onshore', label: 'ideal'|'tolerable'|'bad' }.
    """
    # Resolve wind direction degrees
    wind_deg: Optional[float] = None
    if isinstance(wind_dir_input, (int, float)):
        wind_deg = float(wind_dir_input)
    elif isinstance(wind_dir_input, str):
        # Try cardinal -> degrees first; else attempt parsing numeric string
        wind_deg = _cardinal_to_degrees(wind_dir_input)
        if wind_deg is None:
            try:
                wind_deg = float(wind_dir_input)
            except Exception:
                wind_deg = None
    # Default orientation for unknown spots: use 220 (west‑southwest LA beaches)
    orientation = SPOT_ORIENTATION.get(spot_name, 220)
    offshore_bearing = (orientation + 180) % 360
    kind = "onshore"
    if wind_deg is not None:
        dist = _angular_distance(wind_deg, offshore_bearing)
        if dist <= 45:
            kind = "offshore"
        elif dist <= 80:
            kind = "side"
        else:
            kind = "onshore"
    # Label by speed and orientation
    if kind == "offshore" and wind_speed_mph <= WIND_THRESH["ideal"]:
        label = "ideal"
    elif kind == "side" and wind_speed_mph <= WIND_THRESH["side"]:
        label = "tolerable"
    else:
        label = "bad"
    return {"kind": kind, "label": label, "wind_deg": wind_deg}

def _tier_size(height_ft: float) -> str:
    if height_ft < SIZE_TIERS["micro"]:
        return "micro"
    if height_ft < SIZE_TIERS["small"]:
        return "small"
    if height_ft < SIZE_TIERS["chest"]:
        return "chest"
    if height_ft < SIZE_TIERS["head"]:
        return "head"
    return "overhead"

def _tier_period(period_s: float) -> str:
    if period_s < PERIOD_TIERS["weak"]:
        return "weak"
    if period_s <= PERIOD_TIERS["medium"]:
        return "medium"
    return "powerful"

def _pick_board(size_tier: str, period_tier: str, skill: str) -> str:
    # Board: foamie/longboard for tiny or weak; fish for chest+ medium; shortboard for head+ or powerful
    if size_tier in ("micro", "small") or period_tier == "weak":
        return "foamie" if skill == "Beginner" else "longboard"
    if size_tier in ("chest",) and period_tier in ("medium",):
        return "fish"
    if size_tier in ("head", "overhead") or period_tier == "powerful":
        return "shortboard"
    return "longboard"

def _session_window(wind_kind: str, wind_speed_mph: float, period_tier: str, tide_ft: Optional[float]) -> str:
    if wind_kind == "onshore" and wind_speed_mph > 8.0:
        return "dawn"
    # Suggest midday for weak swell at higher tide (simple heuristic)
    if period_tier == "weak" and (tide_ft is not None) and tide_ft >= 3.5:
        return "midday"
    return "dawn"

def _spot_type(spot_name: str) -> str:
    # Very simple: Malibu is a point; others listed are beach breaks
    return "point" if spot_name.strip().lower() == "malibu" else "beach"

def score_to_text(result: Dict[str, Any], spot_name: str, skill: str) -> Dict[str, Any]:
    """Translate numeric/meteorological inputs into surfer‑friendly guidance.

    Args:
        result: Dict with at least wave_height_ft, wave_period_s, wind_speed_mph,
            wind_direction (deg or cardinal), optional tide_ft, as_of, source, score (0–100).
        spot_name: Human readable spot name (e.g., "Malibu").
        skill: One of "Beginner", "Intermediate", or "Advanced".

    Returns:
        Dict with headline/detail/advice/hazards/board/session_window/confidence.
    """
    # Extract metrics safely
    h = float(result.get("wave_height_ft", 0.0) or 0.0)
    p = float(result.get("wave_period_s", 0.0) or 0.0)
    w = float(result.get("wind_speed_mph", 0.0) or 0.0)
    wd_raw = result.get("wind_direction")
    tide_val = result.get("tide_ft")
    tide = float(tide_val) if tide_val is not None else None
    q100 = float(result.get("score", 0.0) or 0.0)
    source = (result.get("source") or "").lower()

    size_t = _tier_size(h)
    period_t = _tier_period(p)
    wind_info = _classify_wind_quality(spot_name, wd_raw, w)
    board = _pick_board(size_t, period_t, skill)
    session = _session_window(wind_info["kind"], w, period_t, tide)

    # Confidence: live -> high; csv -> low; missing metrics -> low
    critical_missing = any(x is None for x in [h, p, w]) or wind_info.get("wind_deg") is None
    if source == "live" and not critical_missing:
        confidence = "high"
    elif source == "csv" or critical_missing:
        confidence = "low"
    else:
        confidence = "medium"

    # Tide preferences by spot type
    spot_kind = _spot_type(spot_name)
    tide_note = None
    hazards: List[str] = []
    if tide is not None:
        if spot_kind == "point":
            if tide < 1.5:
                tide_note = "low tide can make sections fast at this point break"
            else:
                tide_note = "mid to high tide usually shapes Malibu best"
        else:  # beach
            if tide >= 3.5 and (size_t in ("micro", "small") or period_t == "weak"):
                tide_note = "high tide may swamp weak beach‑break swell"
                hazards.append("High tide swamping and shorebreak closeouts")

    # Skill‑specific framing
    wind_phrase = {
        ("offshore", "ideal"): "clean",
        ("offshore", "bad"): "groomed but brisk",
        ("side", "tolerable"): "ripply",
        ("side", "bad"): "bumpy",
        ("onshore", "bad"): "blown out",
        ("onshore", "tolerable"): "crumbly",
    }.get((wind_info["kind"], wind_info["label"]), "mixed")

    size_phrase = {
        "micro": "ankle‑slappers",
        "small": "knee‑to‑waist",
        "chest": "waist‑to‑chest",
        "head": "head‑high",
        "overhead": "overhead",
    }[size_t]

    # Headline
    headline = f"{size_phrase} {wind_phrase} {('lines' if period_t != 'weak' else 'windswell')} for {skill.lower()}s"

    # Hazards from wind/size/skill
    if wind_info["kind"] == "onshore" and w > 10:
        hazards.append("Strong onshore wind and chop")
    if size_t in ("head", "overhead"):
        if skill == "Beginner":
            hazards.append("Powerful surf — not recommended for beginners")
        else:
            hazards.append("Powerful sets and possible hold‑downs")
    if period_t == "powerful" and wind_info["kind"] in ("offshore", "side") and size_t in ("chest", "head", "overhead"):
        hazards.append("Hollow sections and fast takeoffs (barrels/closeouts)")
    if skill == "Beginner" and wind_info["kind"] in ("onshore", "side") and w >= 8:
        hazards.append("Watch for rip currents, especially near jetties")

    # Detail and advice
    period_desc = {"weak": "short‑period wind swell", "medium": "punchy mid‑period swell", "powerful": "long‑period groundswell"}[period_t]
    wind_desc = f"{wind_info['kind']} winds at {w:.0f} mph ({wind_info['label']})"
    pieces: List[str] = [
        f"About {h:.1f} ft, {period_desc}",
        wind_desc,
    ]
    if tide_note:
        pieces.append(tide_note)
    detail = ", ".join(pieces) + "."

    # Skill‑tuned advice
    if skill == "Beginner":
        if size_t in ("micro", "small") and period_t in ("weak", "medium") and wind_info["kind"] != "onshore":
            learn_line = "great for learning and working on pop‑ups"
        else:
            learn_line = "pick the mellowest peaks and stay on the inside"
        advice = f"Ride a {board}; aim for {session}. {learn_line}."
    elif skill == "Intermediate":
        advice = f"{('Find protected corners' if wind_info['kind']=='onshore' else 'Look for the best sandbars')} and ride a {board}. Aim for {session}."
    else:  # Advanced
        if period_t == "powerful" and wind_info["kind"] in ("offshore", "side"):
            extra = "Expect punchy walls and the odd barrel."
        else:
            extra = "Hunt for the steeper banks to avoid closeouts."
        advice = f"Grab a {board} and hit {session}. {extra}"

    # Live vs CSV badge text is handled in template; expose confidence only
    return {
        "headline": headline,
        "detail": detail,
        "advice": advice,
        "hazards": hazards,
        "board": board,
        "session_window": session,
        "confidence": confidence,
    }

def _pick_hour_for_date(hours: list[dict], date: str) -> dict | None:
    """Select a representative hour for the given YYYY-MM-DD date.

    Prefers 12:00Z if available; otherwise returns the first hour matching
    the date; returns None if no hour matches.
    """
    target_prefix = f"{date}T"
    noon_idx = None
    first_idx = None
    for i, h in enumerate(hours):
        t = h.get("time", "")
        if not t.startswith(target_prefix):
            continue
        if first_idx is None:
            first_idx = i
        # look for 12:00 specifically
        if t.startswith(f"{date}T12:00"):
            noon_idx = i
            break
    idx = noon_idx if noon_idx is not None else first_idx
    return hours[idx] if idx is not None else None


@lru_cache(maxsize=256)
def _fetch_tide_meters(lat: float, lon: float, date: str, api_key: str) -> float | None:
    """Fetch tide sea-level (meters) for a specific date and location.

    Uses the Stormglass /tide/sea-level/point endpoint and picks a
    representative hour for the given date (preferring 12:00Z).
    Returns None if unavailable.
    """
    start_iso = f"{date}T00:00:00Z"
    end_iso = f"{date}T23:59:59Z"
    headers = {"Authorization": api_key}
    try:
        resp = requests.get(
            "https://api.stormglass.io/v2/tide/sea-level/point",
            params={"lat": lat, "lng": lon, "start": start_iso, "end": end_iso},
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        entries = data.get("data", [])
        if not entries:
            return None
        entry = _pick_hour_for_date(entries, date) or entries[0]
        val = entry.get("sg")
        return float(val) if val is not None else None
    except Exception:
        return None


def fetch_live_forecast(location: str, date: str) -> dict:
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
        date: Date string (YYYY-MM-DD) for which to fetch data.

    Returns:
        A dictionary with keys matching the forecast CSV columns (except date),
        containing the latest forecast values.
    """
    # Try on-disk cache first to avoid unnecessary API calls
    cached = _read_live_cache(location, date)
    if cached is not None:
        return cached

    coords = LOCATION_COORDS.get(location.strip().lower())
    if coords is None:
        raise KeyError(f"Unknown location {location}")
    api_key = os.getenv("STORMGLASS_API_KEY")
    if not api_key:
        raise RuntimeError("STORMGLASS_API_KEY not set in the environment")
    lat, lon = coords
    # Build a UTC date range that covers the target day
    start_iso = f"{date}T00:00:00Z"
    end_iso = f"{date}T23:59:59Z"
    params = {
        "lat": lat,
        "lng": lon,
        "params": "waveHeight,wavePeriod,windSpeed,windDirection",
        "source": "noaa",
        "start": start_iso,
        "end": end_iso,
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
    hours = data.get("hours", [])
    if not hours:
        raise ValueError("No hourly data returned from Stormglass API")
    hour = _pick_hour_for_date(hours, date) or hours[0]
    # Extract parameters, converting units where necessary. Some values may be None.
    wave_height_m = hour.get("waveHeight", {}).get("noaa")
    wave_period_s = hour.get("wavePeriod", {}).get("noaa")
    wind_speed_ms = hour.get("windSpeed", {}).get("noaa")
    wind_dir_deg = hour.get("windDirection", {}).get("noaa")
    # Tide via sea-level endpoint (meters). Gracefully handle absence.
    tide_m = _fetch_tide_meters(lat, lon, date, api_key)
    result = {
        "location": location.title(),
        "wave_height_ft": round(wave_height_m * 3.28084, 1) if wave_height_m is not None else 0.0,
        "wave_period_s": wave_period_s if wave_period_s is not None else 0.0,
        "wind_speed_mph": round(wind_speed_ms * 2.23694, 1) if wind_speed_ms is not None else 0.0,
        "wind_direction": _degrees_to_cardinal(wind_dir_deg) if wind_dir_deg is not None else "N",
        "tide_ft": round(tide_m * 3.28084, 1) if tide_m is not None else 0.0,
        "source": "stormglass",
    }
    # Write to cache (best-effort)
    _write_live_cache(location, date, result)
    return result


def _openmeteo_pick_index(times: list[str], date: str) -> int | None:
    target_prefix = f"{date}T"
    noon = f"{date}T12:00"
    first_idx = None
    for i, t in enumerate(times):
        if not isinstance(t, str) or not t.startswith(target_prefix):
            continue
        if first_idx is None:
            first_idx = i
        if t.startswith(noon):
            return i
    return first_idx


def fetch_live_forecast_open_meteo(location: str, date: str) -> dict:
    """Fallback live forecast using Open‑Meteo Marine API (no API key).

    Provides wave height/period and wind; tide is not available (set to 0.0).
    """
    coords = LOCATION_COORDS.get(location.strip().lower())
    if coords is None:
        raise KeyError(f"Unknown location {location}")
    lat, lon = coords
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wave_height,wave_period,wind_speed_10m,wind_direction_10m",
        "start_date": date,
        "end_date": date,
        "timeformat": "iso8601",
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }
    resp = requests.get(
        "https://marine-api.open-meteo.com/v1/marine",
        params=params,
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    idx = _openmeteo_pick_index(times, date)
    if idx is None:
        raise ValueError("No hourly data returned from Open‑Meteo")
    # Extract values safely
    wave_h_list = hourly.get("wave_height") or []
    wave_p_list = hourly.get("wave_period") or []
    wind_s_list = hourly.get("wind_speed_10m") or hourly.get("windspeed_10m") or []
    wind_d_list = hourly.get("wind_direction_10m") or hourly.get("winddirection_10m") or []

    wave_height_m = wave_h_list[idx] if idx < len(wave_h_list) else None
    wave_period_s = wave_p_list[idx] if idx < len(wave_p_list) else None
    wind_speed_ms = wind_s_list[idx] if idx < len(wind_s_list) else None
    wind_dir_deg = wind_d_list[idx] if idx < len(wind_d_list) else None

    result = {
        "location": location.title(),
        "wave_height_ft": round(wave_height_m * 3.28084, 1) if wave_height_m is not None else 0.0,
        "wave_period_s": wave_period_s if wave_period_s is not None else 0.0,
        "wind_speed_mph": round(wind_speed_ms * 2.23694, 1) if wind_speed_ms is not None else 0.0,
        "wind_direction": _degrees_to_cardinal(wind_dir_deg) if wind_dir_deg is not None else "N",
        "tide_ft": 0.0,
        "source": "open-meteo",
    }
    _write_live_cache(location, date, result)
    return result


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


def retrieve_forecast(location: str, date: str) -> tuple[dict | None, str | None]:
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
    # Attempt live fetch first (Stormglass), then fallback to Open‑Meteo
    live_error: str | None = None
    tried_openmeteo = False
    try:
        live_row = fetch_live_forecast(location, date)
        # If cache returned an Open‑Meteo result, surface notice
        if isinstance(live_row, dict) and live_row.get("source") == "open-meteo":
            live_row["_notice"] = "Using Open‑Meteo fallback (no tide)."
        live_row["date"] = date
        return live_row, None
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status == 429:
            live_error = (
                "Stormglass rate limit reached (HTTP 429). "
                "Attempted Open‑Meteo fallback."
            )
        elif status:
            live_error = f"Stormglass HTTP error {status}. Attempted Open‑Meteo fallback."
        else:
            live_error = "Stormglass HTTP error. Attempted Open‑Meteo fallback."
        # Try Open‑Meteo fallback
        try:
            tried_openmeteo = True
            om_row = fetch_live_forecast_open_meteo(location, date)
            # Surface a non-fatal notice to callers/UI
            om_row["_notice"] = "Using Open‑Meteo fallback (no tide)."
            om_row["date"] = date
            return om_row, None
        except Exception as om_e:
            live_error += f" Open‑Meteo error: {om_e}"
    except (requests.RequestException, KeyError, RuntimeError, Exception) as e:
        # Non-HTTP errors (network, missing key, unknown location, etc.)
        base_msg = str(e)
        # Try Open‑Meteo fallback
        try:
            tried_openmeteo = True
            om_row = fetch_live_forecast_open_meteo(location, date)
            om_row["_notice"] = "Using Open‑Meteo fallback (no tide)."
            om_row["date"] = date
            return om_row, None
        except Exception as om_e:
            live_error = f"{base_msg}. Open‑Meteo error: {om_e}"
    # Look up in the static dataset
    df = load_data()
    loc_norm = location.strip().lower()
    match = df[(df["location_norm"] == loc_norm) & (df["date"] == date)]
    if match.empty:
        # Nothing in CSV; return detailed message including live fetch reason
        if live_error:
            if "429" in live_error or "rate limit" in live_error.lower():
                msg = (
                    f"Live data unavailable for {location.title()} on {date}: "
                    f"Stormglass daily limit reached. "
                    f"{('Fallback also failed.' if tried_openmeteo else '')} "
                    f"Try again later or use a cached date from the CSV. Details: {live_error}."
                )
            else:
                msg = (
                    f"No forecast found for {location.title()} on {date}. Live fetch failed. "
                    f"Details: {live_error}."
                )
        else:
            msg = (
                f"No forecast found for {location.title()} on {date}."
            )
        return None, msg
    row = match.iloc[0].to_dict()
    row.pop("location_norm", None)
    return row, None


def _window_score(val: float, ideal_min: float, ideal_max: float, hard_min: float, hard_max: float) -> float:
    """Return a 0–10 score for how well `val` fits within an ideal window.

    - 10 inside [ideal_min, ideal_max]
    - Linearly decreases to 0 at hard_min/hard_max outside the ideal window
    - Clipped to [0, 10]
    """
    if hard_max <= hard_min:
        return 0.0
    if ideal_min > ideal_max:
        ideal_min, ideal_max = ideal_max, ideal_min
    if val < hard_min or val > hard_max:
        return 0.0
    if ideal_min <= val <= ideal_max:
        return 10.0
    if val < ideal_min:
        # Map [hard_min, ideal_min] -> [0, 10]
        return max(0.0, 10.0 * (val - hard_min) / (ideal_min - hard_min))
    # val > ideal_max
    return max(0.0, 10.0 * (hard_max - val) / (hard_max - ideal_max))


def compute_quality_score(row: dict, skill: Optional[str] = None, spot_name: Optional[str] = None) -> float:
    """Compute a surf quality score from 1 (poor) to 10 (epic).

    The base heuristic combines wave height, period, wind speed, and tide.
    If ``skill`` is provided ("Beginner", "Intermediate", "Advanced"), the
    height/period contributions are gently biased so that smaller surf tends
    to rate higher for beginners, while larger/punchier surf gets a slight
    boost for advanced surfers.
    """
    height = float(row["wave_height_ft"])
    period = float(row["wave_period_s"])
    wind_speed = float(row["wind_speed_mph"])
    tide = float(row["tide_ft"])
    spot_name = spot_name or row.get("location") or ""

    # Normalize skill
    s = (skill or "").title()
    if s not in {"Beginner", "Intermediate", "Advanced"}:
        s = None

    # Pro-informed windows for size and period by skill (ft, s)
    if s == "Beginner":
        size_params = dict(ideal_min=1.5, ideal_max=3.0, hard_min=0.5, hard_max=4.0)
        period_params = dict(ideal_min=8.0, ideal_max=11.0, hard_min=6.0, hard_max=13.0)
    elif s == "Advanced":
        size_params = dict(ideal_min=4.0, ideal_max=8.0, hard_min=3.0, hard_max=12.0)
        period_params = dict(ideal_min=12.0, ideal_max=17.0, hard_min=10.0, hard_max=20.0)
    else:  # Intermediate or None
        size_params = dict(ideal_min=3.0, ideal_max=5.0, hard_min=2.0, hard_max=6.5)
        period_params = dict(ideal_min=10.0, ideal_max=14.0, hard_min=8.0, hard_max=16.0)

    size_score = _window_score(height, **size_params)
    period_score = _window_score(period, **period_params)

    # Wind: use existing classification and map to 0–10 with skill nuance
    wind_info = _classify_wind_quality(spot_name or "", row.get("wind_direction"), wind_speed)
    if wind_info["kind"] == "offshore" and wind_info["label"] == "ideal":
        wind_score = 10.0
    elif wind_info["kind"] == "side" and wind_info["label"] == "tolerable":
        wind_score = 6.5
    elif wind_info["kind"] == "offshore":
        wind_score = 8.0
    elif wind_info["kind"] == "side":
        wind_score = 5.0
    else:  # onshore
        wind_score = 3.0
    # Skill nuance: beginners suffer more in side/onshore, advanced cope better
    if s == "Beginner":
        if wind_info["kind"] == "side":
            wind_score -= 1.0
        elif wind_info["kind"] == "onshore":
            wind_score -= 1.5
    elif s == "Advanced":
        if wind_info["kind"] == "side":
            wind_score += 0.5
        elif wind_info["kind"] == "onshore":
            wind_score += 0.5  # still not great, but manageability
    wind_score = max(0.0, min(10.0, wind_score))

    # Tide: tailor by spot type and size
    spot_kind = _spot_type(spot_name or "")
    if spot_kind == "point":
        # Malibu: mid to high generally better
        tide_ideal_min, tide_ideal_max = 2.0, 4.0
        tide_hard_min, tide_hard_max = 1.0, 5.0
    else:  # beach breaks
        if height <= 3.0:
            tide_ideal_min, tide_ideal_max = 1.0, 2.5
            tide_hard_min, tide_hard_max = 0.5, 3.5
        else:
            tide_ideal_min, tide_ideal_max = 1.5, 3.2
            tide_hard_min, tide_hard_max = 0.5, 4.0
    tide_score = _window_score(tide, tide_ideal_min, tide_ideal_max, tide_hard_min, tide_hard_max)

    # Additional interactions from pro heuristics
    if s == "Beginner" and period >= 14.0 and height >= 3.5:
        period_score = max(0.0, period_score - 2.0)  # steep, powerful
    if s == "Advanced" and period >= 14.0 and height >= 5.0 and wind_info["kind"] in ("offshore", "side"):
        size_score = min(10.0, size_score + 1.0)

    # Weighted blend to 0–10
    # Emphasize size and wind, then period, then tide
    weights = dict(size=0.4, period=0.25, wind=0.25, tide=0.10)
    composite = (
        size_score * weights["size"]
        + period_score * weights["period"]
        + wind_score * weights["wind"]
        + tide_score * weights["tide"]
    )

    # Clip 1–10 to keep UI semantics (avoid 0s which read like broken data)
    score = max(1.0, min(10.0, composite))
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


def generate_summary(row: dict, skill: Optional[str] = None) -> str:
    """Generate a natural‑language summary of the surf forecast for a row.

    If ``skill`` is provided, the score will reflect skill-aware biases.
    """
    score = compute_quality_score(row, skill, row.get("location"))
    quality = score_to_description(score)
    return (
        f"Surf forecast for {row['location']} on {row['date']}: "
        f"waves around {row['wave_height_ft']} ft with a {row['wave_period_s']} second period, "
        f"winds {row['wind_speed_mph']} mph from {row['wind_direction']}, tide {row['tide_ft']} ft. "
        f"Overall conditions look {quality} (score {score})."
    )


def outlook_for_location(location: str, skill: Optional[str] = None) -> pd.DataFrame:
    """Return the outlook DataFrame for the given location sorted by date.

    Returns the subset of the dataset for the location, adding a ``quality`` column.
    """
    df = load_data()
    loc_norm = location.strip().lower()
    subset = df[df["location_norm"] == loc_norm].copy()
    if subset.empty:
        return subset
    subset.sort_values("date", inplace=True)
    subset["quality"] = subset.apply(lambda r: compute_quality_score(r, skill, r.get("location")), axis=1)
    return subset
