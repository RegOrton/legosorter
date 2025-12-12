"""
Webcam Client for Docker Container

This module provides a WebcamClient class that acts as a drop-in replacement
for cv2.VideoCapture when accessing a webcam via HTTP from the host machine.

Usage:
    from webcam_client import WebcamClient
    
    # Use like cv2.VideoCapture
    cap = WebcamClient("http://host.docker.internal:5000")
    ret, frame = cap.read()
    cap.release()
"""

import cv2
import numpy as np
import requests
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebcamClient:
    """
    A client for accessing webcam frames via HTTP.
    
    Provides an interface compatible with cv2.VideoCapture for easy integration.
    """
    
    def __init__(self, server_url: str = "http://host.docker.internal:5000", timeout: int = 5):
        """
        Initialize the webcam client.
        
        Args:
            server_url: URL of the webcam server (default: http://host.docker.internal:5000)
            timeout: Request timeout in seconds (default: 5)
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.is_open = False
        
        # Test connection
        try:
            response = requests.get(f"{self.server_url}/status", timeout=self.timeout)
            if response.status_code == 200:
                status = response.json()
                if status.get('camera_open', False):
                    self.is_open = True
                    logger.info(f"Connected to webcam server at {self.server_url}")
                else:
                    logger.error("Webcam server camera is not open")
            else:
                logger.error(f"Failed to connect to webcam server: HTTP {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to webcam server: {e}")
    
    def isOpened(self) -> bool:
        """
        Check if the webcam connection is open.
        
        Returns:
            True if connected to the server and camera is open, False otherwise
        """
        return self.is_open
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the webcam.
        
        Returns:
            Tuple of (success, frame) where:
                - success: True if frame was successfully captured
                - frame: numpy array containing the frame, or None if failed
        """
        if not self.is_open:
            return False, None
        
        try:
            response = requests.get(f"{self.server_url}/frame", timeout=self.timeout)
            if response.status_code == 200:
                # Decode JPEG image
                img_array = np.frombuffer(response.content, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    return True, frame
                else:
                    logger.warning("Failed to decode frame")
                    return False, None
            else:
                logger.warning(f"Failed to get frame: HTTP {response.status_code}")
                return False, None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error reading frame: {e}")
            return False, None
    
    def release(self):
        """
        Release the webcam connection.
        
        Note: This doesn't actually close the server-side camera,
        it just marks this client as closed.
        """
        self.is_open = False
        logger.info("Webcam client released")
    
    def get(self, prop_id: int) -> float:
        """
        Get a property value (limited support).
        
        Args:
            prop_id: OpenCV property ID (e.g., cv2.CAP_PROP_FRAME_WIDTH)
        
        Returns:
            Property value, or 0 if not supported
        """
        # Limited property support - return defaults
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return 640.0
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return 480.0
        elif prop_id == cv2.CAP_PROP_FPS:
            return 30.0
        else:
            logger.warning(f"Property {prop_id} not supported by WebcamClient")
            return 0.0
    
    def set(self, prop_id: int, value: float) -> bool:
        """
        Set a property value (not supported).
        
        Args:
            prop_id: OpenCV property ID
            value: Value to set
        
        Returns:
            False (setting properties not supported)
        """
        logger.warning("Setting properties not supported by WebcamClient")
        return False


def create_webcam_capture(use_http: bool = True, server_url: str = "http://host.docker.internal:5000") -> object:
    """
    Factory function to create a webcam capture object.
    
    Args:
        use_http: If True, use WebcamClient; if False, use cv2.VideoCapture(0)
        server_url: URL of the webcam server (only used if use_http=True)
    
    Returns:
        WebcamClient or cv2.VideoCapture object
    """
    if use_http:
        return WebcamClient(server_url)
    else:
        return cv2.VideoCapture(0)


if __name__ == "__main__":
    # Test the webcam client
    print("Testing WebcamClient...")
    
    client = WebcamClient()
    
    if client.isOpened():
        print("Successfully connected to webcam server")
        
        # Read a few frames
        for i in range(5):
            ret, frame = client.read()
            if ret:
                print(f"Frame {i+1}: {frame.shape}")
            else:
                print(f"Frame {i+1}: Failed to read")
        
        client.release()
        print("Test complete")
    else:
        print("Failed to connect to webcam server")
        print("Make sure the server is running on the host machine:")
        print("  python webcam_server.py")
