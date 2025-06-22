# Use official Python slim image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements file first to leverage caching
COPY requirements.txt .

# Install system dependencies needed for dlib, face_recognition, and PIL
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

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code into the container
COPY . .

# Ensure the shared_data folder exists
RUN mkdir -p /app/shared_data

# Set the default command to run your app
ENTRYPOINT ["python", "app.py"]
