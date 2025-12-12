# Base Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies dependencies for OpenCV and other tools
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Default command (can be overridden by docker-compose)
CMD ["python", "src/main.py"]
