FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source
COPY . .

# Hugging Face Spaces will set $PORT; default to 7860 locally
ENV PORT=7860

# Use shell form so $PORT is expanded at runtime
CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
