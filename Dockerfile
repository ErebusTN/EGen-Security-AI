FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=5000 \
    HOST=0.0.0.0

# Expose the API port
EXPOSE 5000

# Command to run the application
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "5000"] 