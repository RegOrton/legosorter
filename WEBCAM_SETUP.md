# Webcam Setup for Docker Container

This guide explains how to access your Windows webcam from inside a Docker container using HTTP streaming.

## Overview

Since Docker on Windows runs containers in a Linux VM, direct webcam access via device mapping doesn't work reliably. Instead, we use an HTTP-based approach:

1. **Webcam Server** runs on Windows host and streams webcam frames via HTTP
2. **Webcam Client** runs in the Docker container and fetches frames from the server

## Prerequisites

- Python 3.7+ installed on Windows
- Docker Desktop installed and running
- A working webcam

## Setup Instructions

### Step 1: Install Dependencies on Windows

Open PowerShell or Command Prompt and install the required packages:

```bash
pip install flask opencv-python numpy
```

### Step 2: Start the Webcam Server

From the project root directory, run:

```bash
python webcam_server.py
```

You should see output like:
```
INFO - Starting webcam server on 0.0.0.0:5000
INFO - Camera 0 initialized successfully
```

**Keep this terminal window open** - the server needs to keep running.

### Step 3: Build and Start the Container

In a new terminal, navigate to the `vision` directory and run:

```bash
cd vision
docker-compose up --build
```

### Step 4: Test Webcam Access

Once the container is running, execute the test script:

```bash
docker-compose exec vision python src/test_webcam.py
```

You should see output confirming successful webcam access and a test frame saved to `vision/output/test_webcam_frame.jpg`.

## Usage in Your Code

### Option 1: Using WebcamClient Directly

```python
from webcam_client import WebcamClient

# Create webcam client (uses HTTP by default)
cap = WebcamClient("http://host.docker.internal:5000")

if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Process frame
        print(f"Frame shape: {frame.shape}")
    
    cap.release()
```

### Option 2: Using the Factory Function

```python
from webcam_client import create_webcam_capture
import os

# Automatically use HTTP webcam if USE_HTTP_WEBCAM env var is set
use_http = os.getenv('USE_HTTP_WEBCAM', 'false').lower() == 'true'
cap = create_webcam_capture(use_http=use_http)

ret, frame = cap.read()
cap.release()
```

### Option 3: Drop-in Replacement

The `WebcamClient` class is designed to be compatible with `cv2.VideoCapture`:

```python
from webcam_client import WebcamClient
import cv2

# Instead of: cap = cv2.VideoCapture(0)
cap = WebcamClient()

# Use the same API
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
```

## Server Options

The webcam server supports several command-line options:

```bash
# Use a different port
python webcam_server.py --port 8080

# Use a different camera (if you have multiple)
python webcam_server.py --camera 1

# Bind to a specific host
python webcam_server.py --host 192.168.1.100
```

## API Endpoints

The webcam server exposes the following endpoints:

- `GET /` - Server information
- `GET /status` - Camera status (returns JSON)
- `GET /frame` - Get a single frame as JPEG
- `GET /stream` - MJPEG stream for continuous video

## Troubleshooting

### Container can't connect to server

**Problem**: `Failed to connect to webcam server`

**Solutions**:
1. Make sure `webcam_server.py` is running on the host
2. Check that Docker Desktop is using the correct network mode
3. Try accessing `http://host.docker.internal:5000/status` from inside the container:
   ```bash
   docker-compose exec vision curl http://host.docker.internal:5000/status
   ```

### Camera not opening on Windows

**Problem**: `Failed to open camera 0`

**Solutions**:
1. Close any other applications using the webcam (Teams, Zoom, etc.)
2. Try a different camera index: `python webcam_server.py --camera 1`
3. Check Windows privacy settings (Settings → Privacy → Camera)

### Firewall blocking connections

**Problem**: Connection timeout or refused

**Solutions**:
1. Allow Python through Windows Firewall
2. Temporarily disable firewall to test
3. Add an inbound rule for port 5000

### Poor performance or lag

**Problem**: Frames are slow or choppy

**Solutions**:
1. Reduce frame quality in `webcam_server.py` (lower JPEG quality)
2. Reduce resolution in the server initialization
3. Check network performance between container and host

## Environment Variables

The following environment variables are available in `docker-compose.yml`:

- `USE_HTTP_WEBCAM` - Set to `true` to use HTTP webcam (default: true)
- `WEBCAM_SERVER_URL` - URL of the webcam server (default: http://host.docker.internal:5000)

## Advanced: Running Server as a Service

To run the webcam server automatically on Windows startup, you can:

1. Create a batch file `start_webcam_server.bat`:
   ```batch
   @echo off
   python "C:\Users\Reg\Lego Sorter\webcam_server.py"
   ```

2. Add it to Windows Startup folder:
   - Press `Win + R`
   - Type `shell:startup`
   - Copy the batch file to this folder

## Security Notes

- The server runs on `0.0.0.0` by default, making it accessible from your local network
- For production use, consider adding authentication
- Use `--host 127.0.0.1` to restrict access to localhost only

## Next Steps

- Integrate `WebcamClient` into your existing vision pipeline
- Modify `LegoSynthesizer` or other components to use HTTP webcam
- Test with your actual Lego sorting workflow
