# Base Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV, LDView, and other tools
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ldview \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Download and install LDraw parts library
RUN mkdir -p /usr/share/ldraw && \
    cd /tmp && \
    wget -q https://library.ldraw.org/library/updates/complete.zip && \
    unzip -q complete.zip -d /usr/share/ldraw && \
    rm complete.zip

# Set LDraw directory environment variable
ENV LDRAWDIR=/usr/share/ldraw

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --timeout=1000 --retries=10 -r requirements.txt

# Copy source code
COPY src/ ./src/

# Default command (can be overridden by docker-compose)
# Expose API port
EXPOSE 8000

# Default command
CMD ["python", "src/main_api.py"]
