---
Surf Forecast AI Analyzer
=========================

AI‑assisted surf forecast dashboard built with FastAPI. It scores surf quality on a 0–100 scale, adds surfer‑friendly guidance (headline, hazards, board choice, session timing), and renders a simple chart for popular LA/OC spots. It can use live data (Stormglass with Open‑Meteo fallback) or a bundled CSV for offline demos.

Features
- Skill‑aware scoring: 0–100 quality tuned for Beginner, Intermediate, Advanced.
- Live data: Stormglass (waves, wind, tide) with Open‑Meteo fallback (waves, wind).
- Offline demo data: CSV time series for a few sample dates/locations.
- Helpful text: headline, details, hazards, board suggestion, session window, confidence.
- Simple UI: Tailwind + Chart.js (no build step), responsive and fast.
- Caching: On‑disk cache for live requests and in‑process memoization for tides.

Quick Start
- Python: 3.10+ (tested on 3.11)
- Install deps: `pip install -r requirements.txt`
- Run dev server: `uvicorn app.main:app --reload`
- Open: http://localhost:8000

Live Forecasts (Stormglass + Fallback)
- Configure an API key to enable arbitrary dates beyond the sample CSV:
  - macOS/Linux: `export STORMGLASS_API_KEY=your_key_here`
  - Windows (PowerShell): `$Env:STORMGLASS_API_KEY='your_key_here'`
  - Or create a `.env` file at the project root:
    - `STORMGLASS_API_KEY=your_key_here`
- Behavior:
  - If Stormglass is available, the app fetches live data (including tide) and marks results as “Live”.
  - If Stormglass is unavailable or rate‑limited (HTTP 429), it falls back to Open‑Meteo Marine (no key; no tide) and surfaces a notice.
  - If both live providers fail, it falls back to the bundled CSV and only supports the CSV dates.

Caching and Rate Limits
- On‑disk cache: Each live (location, date) result is cached under `.cache/` and expires after 6 hours or end‑of‑day (UTC), whichever comes first.
- Tide memoization: Tide lookups are memoized in‑process for repeat requests.
- Clear errors: If Stormglass returns 429, the app tries Open‑Meteo and provides a helpful message if both fail.

Usage
1) Open the app and select a location, date (YYYY‑MM‑DD), and skill level.
2) Click “Get Forecast” to see:
   - A 0–100 quality score (single snapshot for live data; multi‑day if from CSV).
   - Headline, detail, hazards, board suggestion, and best session window.
   - A chart with score and tooltip metrics (wave ft, period s, wind mph/dir).

Supported Demo Locations
- Malibu
- Huntington Beach
- Santa Monica
- Venice
- Will Rogers State Beach

API Endpoints
- `GET /` (HTML): Main page. Optional query parameters: `location`, `date`, `skill` (Beginner|Intermediate|Advanced).
- `GET /chart?location=Malibu` (PNG): Static chart endpoint for CSV time series (used internally/for debugging).

Project Structure
- `app/main.py`: FastAPI app, routes, request→UI wiring, chart data.
- `app/surf_logic.py`: Core forecast logic, live fetch, scoring, text guidance, caching.
- `app/templates/index.html`: Tailwind + Chart.js UI template.
- `app/static/style.css`: Optional custom styles.
- `data/surf_data.csv`: Sample dataset for offline/demo use.
- `.cache/`: On‑disk cache for live responses (auto‑created).

Environment Variables
- `STORMGLASS_API_KEY` (optional): Enables live forecasts with tide data via Stormglass.
- `.env` support: A `.env` file at repo root is auto‑loaded if present.

Docker
The included Dockerfile targets an environment where the project is provided as a zip (`surf_forecast_ai.zip`). If you want a simpler local image, you can use the following alternative Dockerfile:

```
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Add or Edit Locations
- Coordinates: Update `LOCATION_COORDS` in `app/surf_logic.py` with new spot names and `(lat, lon)`.
- Wind orientation (optional): Update `SPOT_ORIENTATION` with a spot’s approximate wave approach bearing to improve wind quality classification.

Notes
- CSV dates are examples only. For arbitrary dates, enable a Stormglass key.
- When aggregating hourly live data, the app prefers 12:00 UTC as the representative snapshot if available.
- Open‑Meteo fallback sets `tide_ft` to 0.0 (no tide info provided).

Contributing
Issues and pull requests are welcome. Please keep changes focused and include a short explanation/rationale.
