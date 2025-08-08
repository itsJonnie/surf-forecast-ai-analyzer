# Surf Forecast AI Analyzer

This project is a small **AI‑powered surf forecast dashboard**.  It runs completely for free — there are no external paid APIs or cloud services required.  It demonstrates how to combine **retrieval‑augmented analysis**, a bit of **NLP**, and simple **data visualisation** into a web application you can deploy to a free Hugging Face Space.

## What it does

* **Retrieves surf conditions**: Uses a small bundled dataset of sample surf forecasts for several Southern California spots over a week.  When you query a location and date, the app looks up the matching row.  If a match isn't found, it shows a helpful message.
* **Generates a natural‑language summary**:  The app performs light‐weight NLP on the conditions (wave height, period, wind speed/direction, tide) to assign a quality rating and produces a text summary describing how good the surf will be.
* **Visualises the outlook**:  It plots a simple line chart of the forecasted quality ratings over the next few days for the selected spot using matplotlib.  The chart is served as a PNG image.

## Running locally

1.  Install the dependencies into a virtual environment:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2.  Start the FastAPI server:

    ```bash
    uvicorn app.main:app --reload
    ```

3.  Open your browser to `http://127.0.0.1:8000` to use the app.

The `--reload` flag causes Uvicorn to automatically restart when you edit your code.

## Project structure

```
surf_forecast_ai/
├── app/
│   ├── main.py         # FastAPI application
│   ├── surf_logic.py   # Core retrieval and NLP functions
│   ├── templates/
│   │   └── index.html  # Minimal HTMX/Tailwind UI
│   └── static/
│       └── style.css   # Extra CSS (optional)
├── data/
│   └── surf_data.csv   # Sample surf forecasts (dates, spots, conditions)
├── tests/
│   └── test_app.py     # Simple pytest tests
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container definition for deployment
├── .github/workflows/ci.yml # GitHub Actions workflow
├── push.sh             # Script to create and push a GitHub repo via gh cli
└── push_manual.sh      # Script to push to an existing remote via git
```

## Deployment to Hugging Face Spaces

1.  Create a new **Docker** Space on [Hugging Face Spaces](https://huggingface.co/spaces).  Name it something like `yourname/surf-forecast-ai`.
2.  Clone the empty Space repository locally, or use the provided `push_manual.sh` to push this code directly once you have a Space created.  The `Dockerfile` is set up to install dependencies and run the app using Gunicorn.
3.  Commit and push.  Hugging Face will build your container and make the app available at `https://yourname-surf-forecast-ai.hf.space`.

## License

This project is provided under the **MIT License**.  See the [LICENSE](LICENSE) file for details.