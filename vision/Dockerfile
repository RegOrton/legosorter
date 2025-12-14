# Use Ubuntu for better graphics library support
FROM ubuntu:22.04

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
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libosmesa6 \
    libfuse2 \
    xvfb \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Download and install LDView from GitHub releases (AppImage)
RUN cd /tmp && \
    wget https://github.com/tcobbs/ldview/releases/download/v4.6/LDView-4.6-Linux-x86_64.AppImage -O ldview.AppImage 2>&1 && \
    ls -lah ldview.AppImage && \
    chmod +x ldview.AppImage && \
    ./ldview.AppImage --appimage-extract && \
    mv squashfs-root /opt/ldview && \
    ln -s /opt/ldview/AppRun /usr/local/bin/ldview && \
    rm ldview.AppImage

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
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir --timeout=1000 --retries=10 -r requirements.txt

# Copy source code
COPY src/ ./src/

# Expose API port
EXPOSE 8000

# Default command
CMD ["python3", "src/main_api.py"]
