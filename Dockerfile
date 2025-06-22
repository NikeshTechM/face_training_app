# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for face_recognition, dlib, PIL, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libjpeg-dev \
    zlib1g-dev \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libx11-dev \
    libgtk-3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip (optional but recommended)
RUN pip install --upgrade pip

# Copy and install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Explicitly install face_recognition_models from GitHub (if not in requirements.txt)
RUN pip install --no-cache-dir git+https://github.com/ageitgey/face_recognition_models

# Copy application source code
COPY app.py .

# Create a default output directory
RUN mkdir -p /app/shared

# Set UTF-8 encoding env (recommended)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8

# Optionally: Add a basic health check
# HEALTHCHECK CMD curl --fail http://localhost:5000/health || exit 1

# Define entrypoint
ENTRYPOINT ["python", "app.py"]
