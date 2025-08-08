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


app = FastAPI(title="Surf Forecast AI Analyzer")

# Set up templates directory
templates = Jinja2Templates(directory=str((__file__).rsplit("/", 1)[0] + "/templates"))

# Optionally mount static directory if needed
app.mount("/static", StaticFiles(directory=str((__file__).rsplit("/", 1)[0] + "/static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, location: str | None = None, date: str | None = None):
    """Render the main page.  Optionally display results if query params are provided."""
    context = {"request": request}
    summary = None
    error = None
    if location and date:
        row = surf_logic.retrieve_forecast(location, date)
        if row:
            summary = surf_logic.generate_summary(row)
        else:
            error = f"No forecast found for {location.title()} on {date}."
    context.update({"summary": summary, "error": error, "location": location or "", "date": date or ""})
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