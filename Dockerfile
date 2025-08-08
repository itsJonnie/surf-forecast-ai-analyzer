FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy the zipped project into the container
COPY surf_forecast_ai.zip .

# Install unzip to extract the archive
RUN apt-get update && apt-get install -y unzip \
    && unzip surf_forecast_ai.zip \
    && mv surf_forecast_ai/* . \
    && rm -rf surf_forecast_ai surf_forecast_ai.zip \
    && pip install --no-cache-dir -r requirements.txt

# Expose port and run the FastAPI application
ENV PORT=7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]