"""
FastAPI application for the Surf Forecast AI Analyzer.

This file defines the web server and HTML routes.  It uses Jinja2
templates for the user interface and serves a simple chart of surf
quality over time for a chosen location.
"""

from __future__ import annotations

import io
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import matplotlib

# Use a non‑interactive backend for server
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import pandas as pd

from . import surf_logic
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone


app = FastAPI(title="Surf Forecast AI Analyzer")

# Set up templates directory
templates = Jinja2Templates(directory=str((__file__).rsplit("/", 1)[0] + "/templates"))

# Optionally mount static directory if needed
app.mount("/static", StaticFiles(directory=str((__file__).rsplit("/", 1)[0] + "/static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    location: Optional[str] = None,
    date: Optional[str] = None,
    skill: Optional[str] = "Beginner",
):
    """Render the main page.  Optionally display results if query params are provided."""
    context = {"request": request}
    summary = None
    error = None
    notice: Optional[str] = None
    source: Optional[str] = None
    source_label: Optional[str] = None

    # Prepare known spots (title-cased) and validate selection
    known_spots: List[str] = [name.title() for name in surf_logic.LOCATION_COORDS.keys()]
    known_spots_sorted = sorted(known_spots)
    # Validate skill
    allowed_skills = ["Beginner", "Intermediate", "Advanced"]
    skill_sel = skill if (skill and skill.title() in allowed_skills) else "Beginner"
    if skill_sel != (skill or ""):
        skill_sel = skill_sel  # normalize casing

    # Chart data defaults
    chart_labels: List[str] = []
    chart_values: List[float] = []
    wave_ft: List[float] = []
    period_s: List[float] = []
    wind_mph: List[float] = []
    wind_dir: List[str] = []
    result: Optional[Dict[str, Any]] = None

    # Validate location against known spots (case-insensitive) for processing
    sel_location_label: Optional[str] = None
    if location:
        for label in known_spots_sorted:
            if label.lower() == location.strip().lower():
                sel_location_label = label
                break
        if not sel_location_label:
            error = (
                "Unknown location. Please select a known surf spot from the dropdown."
            )

    if sel_location_label and date:
        row, err = surf_logic.retrieve_forecast(sel_location_label, date)
        if row:
            # Extract optional non-fatal notice from live fallback
            notice = row.pop("_notice", None)
            source = row.get("source")  # don't pop yet; used for chart logic

            # Decide chart mode:
            # - If live source present -> single snapshot bar
            # - Else -> use CSV time series for the location
            if source in ("stormglass", "open-meteo"):
                # Live snapshot
                as_of = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
                quality_10 = surf_logic.compute_quality_score({**row, "date": date}, skill_sel, sel_location_label)
                chart_labels = [f"As of {as_of}"]
                chart_values = [min(100.0, max(0.0, quality_10 * 10.0))]
                wave_ft = [float(row.get("wave_height_ft", 0.0))]
                period_s = [float(row.get("wave_period_s", 0.0))]
                wind_mph = [float(row.get("wind_speed_mph", 0.0))]
                wind_dir = [str(row.get("wind_direction", "N"))]
                # Friendly source label for footer
                source_label = "Stormglass" if source == "stormglass" else "Open‑Meteo"
                # Set normalized source for status chip
                source = "live"
                # Build result payload for domain text
                result = {
                    "source": source,
                    "as_of": as_of,
                    "wave_height_ft": float(row.get("wave_height_ft", 0.0)),
                    "wave_period_s": float(row.get("wave_period_s", 0.0)),
                    "wind_speed_mph": float(row.get("wind_speed_mph", 0.0)),
                    "wind_direction": row.get("wind_direction", "N"),
                    "tide_ft": float(row.get("tide_ft", 0.0)),
                    "score": float(quality_10 * 10.0),
                }
            else:
                # CSV time series for this location
                df = surf_logic.outlook_for_location(sel_location_label, skill_sel)
                if not df.empty:
                    chart_labels = list(df["date"].astype(str).values)
                    chart_values = [min(100.0, max(0.0, float(v) * 10.0)) for v in df["quality"].values]
                    wave_ft = [float(v) for v in df["wave_height_ft"].values]
                    period_s = [float(v) for v in df["wave_period_s"].values]
                    wind_mph = [float(v) for v in df["wind_speed_mph"].values]
                    wind_dir = [str(v) for v in df["wind_direction"].values]
                    source = "csv"
                    # Build single-row based result for guidance using the retrieved row
                    q10 = surf_logic.compute_quality_score({**row, "date": row.get("date", date)}, skill_sel, sel_location_label)
                    result = {
                        "source": source,
                        "wave_height_ft": float(row.get("wave_height_ft", 0.0)),
                        "wave_period_s": float(row.get("wave_period_s", 0.0)),
                        "wind_speed_mph": float(row.get("wind_speed_mph", 0.0)),
                        "wind_direction": row.get("wind_direction", "N"),
                        "tide_ft": float(row.get("tide_ft", 0.0)),
                        "score": float(q10 * 10.0),
                    }
                # Footer is hidden for CSV by leaving source_label None

            # Now generate textual summary; drop helper keys from row
            row.pop("source", None)
            summary = surf_logic.generate_summary({**row, "date": date}, skill_sel)

            # Domain model text enrichment
            if result:
                enriched = surf_logic.score_to_text(result, sel_location_label, skill_sel)
            else:
                enriched = {}
        else:
            error = err or (
                f"No forecast found for {sel_location_label} on {date}. If this date is outside the sample dataset, "
                f"set STORMGLASS_API_KEY to enable live forecasts."
            )

    context.update({
        "summary": summary,
        "error": error,
        "notice": notice,
        "location": sel_location_label or (location or ""),
        "date": date or "",
        "source": source,
        "source_label": source_label,
        "spots": known_spots_sorted,
        "result": result,
        # Skill selection
        "skill": skill_sel,
        # Enriched domain guidance
        "headline": (enriched.get("headline") if 'enriched' in locals() else None),
        "detail": (enriched.get("detail") if 'enriched' in locals() else None),
        "advice": (enriched.get("advice") if 'enriched' in locals() else None),
        "hazards": (enriched.get("hazards") if 'enriched' in locals() else []),
        "board": (enriched.get("board") if 'enriched' in locals() else None),
        "session_window": (enriched.get("session_window") if 'enriched' in locals() else None),
        "confidence": (enriched.get("confidence") if 'enriched' in locals() else None),
        # Chart data
        "chart_labels": chart_labels,
        "chart_values": chart_values,
        "wave_ft": wave_ft,
        "period_s": period_s,
        "wind_mph": wind_mph,
        "wind_dir": wind_dir,
    })
    return templates.TemplateResponse("index.html", context)


@app.get("/chart", response_class=StreamingResponse)
async def chart(location: str):
    """Return a PNG chart of surf quality for the given location."""
    df = surf_logic.outlook_for_location(location)
    if df.empty:
        # Return a blank image if no data
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(pd.to_datetime(df["date"]), df["quality"], marker="o")
        ax.set_title(f"Surf quality forecast for {location.title()}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Quality (1-10)")
        ax.set_ylim(0, 10)
        fig.autofmt_xdate()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
