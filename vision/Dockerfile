# Use Ubuntu 24.04 for LDView 4.6 compatibility
FROM ubuntu:24.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-dri \
    libosmesa6 \
    libglx0 \
    xvfb \
    wget \
    unzip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Download and install LDView osmesa (headless) version from GitHub releases
RUN cd /tmp && \
    wget -q https://github.com/tcobbs/ldview/releases/download/v4.6/ldview-osmesa-4.6-ubuntu-24.04.amd64.deb && \
    apt-get update && \
    apt-get install -y ./ldview-osmesa-4.6-ubuntu-24.04.amd64.deb && \
    rm ldview-osmesa-4.6-ubuntu-24.04.amd64.deb && \
    rm -rf /var/lib/apt/lists/*

# Download and install LDraw parts library
RUN mkdir -p /usr/share/ldraw && \
    cd /tmp && \
    wget -q https://library.ldraw.org/library/updates/complete.zip && \
    unzip -q complete.zip -d /usr/share/ldraw && \
    rm complete.zip

# Set environment variables
ENV LDRAWDIR=/usr/share/ldraw/ldraw
ENV PYTHONUNBUFFERED=1

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --timeout=1000 --retries=10 --break-system-packages --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Expose API port
EXPOSE 8000

# Default command
CMD ["python3", "src/main_api.py"]
