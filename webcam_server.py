"""
Webcam HTTP Server for Docker Container Access

This server runs on the Windows host and streams webcam frames via HTTP.
The Docker container can access these frames using the webcam_client.py wrapper.

Usage:
    python webcam_server.py [--port PORT] [--camera CAMERA_INDEX]

Example:
    python webcam_server.py --port 5000 --camera 0
"""

import cv2
import numpy as np
from flask import Flask, Response, jsonify
import argparse
import logging
from threading import Lock
import time

app = Flask(__name__)

# Global variables
camera = None
camera_lock = Lock()
last_frame = None
last_frame_time = 0

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def initialize_camera(camera_index=0):
    """Initialize the webcam."""
    global camera
    try:
        camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Use DirectShow on Windows
        if not camera.isOpened():
            logger.error(f"Failed to open camera {camera_index}")
            return False
        
        # Set camera properties for better performance
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info(f"Camera {camera_index} initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing camera: {e}")
        return False


def get_frame():
    """Capture a frame from the webcam."""
    global camera, last_frame, last_frame_time
    
    with camera_lock:
        if camera is None or not camera.isOpened():
            return None
        
        success, frame = camera.read()
        if success:
            last_frame = frame
            last_frame_time = time.time()
            return frame
        else:
            logger.warning("Failed to read frame from camera")
            return None


def generate_mjpeg_stream():
    """Generate MJPEG stream for continuous video."""
    while True:
        frame = get_frame()
        if frame is not None:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)  # ~30 FPS


@app.route('/')
def index():
    """Root endpoint with server information."""
    return jsonify({
        'status': 'running',
        'endpoints': {
            '/frame': 'Get single frame as JPEG',
            '/stream': 'MJPEG stream',
            '/status': 'Server status'
        }
    })


@app.route('/status')
def status():
    """Return server status."""
    global camera, last_frame_time
    
    is_camera_open = camera is not None and camera.isOpened()
    time_since_last_frame = time.time() - last_frame_time if last_frame_time > 0 else -1
    
    return jsonify({
        'camera_open': is_camera_open,
        'last_frame_age_seconds': time_since_last_frame,
        'status': 'ok' if is_camera_open else 'error'
    })


@app.route('/frame')
def frame():
    """Return a single frame as JPEG."""
    frame = get_frame()
    if frame is not None:
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            return Response(buffer.tobytes(), mimetype='image/jpeg')
    
    return jsonify({'error': 'Failed to capture frame'}), 500


@app.route('/stream')
def stream():
    """Return MJPEG stream."""
    return Response(generate_mjpeg_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def main():
    parser = argparse.ArgumentParser(description='Webcam HTTP Server')
    parser.add_argument('--port', type=int, default=5000, help='Server port (default: 5000)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    args = parser.parse_args()
    
    logger.info(f"Starting webcam server on {args.host}:{args.port}")
    
    if not initialize_camera(args.camera):
        logger.error("Failed to initialize camera. Exiting.")
        return
    
    try:
        app.run(host=args.host, port=args.port, threaded=True)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    finally:
        if camera is not None:
            camera.release()
            logger.info("Camera released")


if __name__ == '__main__':
    main()
