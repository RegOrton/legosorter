import cv2
import time
import logging
import os
from webcam_client import create_webcam_capture

# Camera type constants
CAMERA_USB = "usb"
CAMERA_CSI = "csi"
CAMERA_HTTP = "http"

class Camera:
    def __init__(self, source=0, width=1280, height=720, camera_type=None):
        """
        Initialize the Camera.
        :param source: Camera index (int) or video file path (str).
        :param width: Desired frame width
        :param height: Desired frame height
        :param camera_type: Camera type: "usb", "csi", or "http". If None, auto-detect from environment.
        """
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        self.logger = logging.getLogger(__name__)

        # Determine camera type
        if camera_type is not None:
            self.camera_type = camera_type
        elif os.getenv('USE_HTTP_WEBCAM', 'false').lower() == 'true':
            self.camera_type = CAMERA_HTTP
        else:
            self.camera_type = CAMERA_USB

    def set_camera_type(self, camera_type: str):
        """
        Switch camera type and restart the camera.
        :param camera_type: "usb", "csi", or "http"
        """
        self.logger.info(f"Switching camera type from {self.camera_type} to {camera_type}")
        self.release()
        self.camera_type = camera_type
        self.start()

    def start(self):
        """Opens the video source."""
        self.logger.info(f"Opening camera (type: {self.camera_type}, source: {self.source})")

        if self.camera_type == CAMERA_HTTP:
            self.logger.info("Using HTTP webcam client")
            self.cap = create_webcam_capture(use_http=True)
        elif self.camera_type == CAMERA_CSI:
            self.logger.info("Using CSI camera (Raspberry Pi camera module)")
            # For Raspberry Pi camera module, use libcamera-vid backend or gstreamer
            # This attempts to use libcamera via OpenCV's CAP_V4L2 backend
            gstreamer_pipeline = (
                "libcamerasrc ! "
                "video/x-raw,width={},height={},framerate=30/1 ! "
                "videoconvert ! "
                "appsink"
            ).format(self.width, self.height)

            # Try gstreamer pipeline first
            self.cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

            # Fallback to V4L2 with /dev/video0
            if not self.cap.isOpened():
                self.logger.warning("GStreamer pipeline failed, trying V4L2...")
                self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        else:  # CAMERA_USB or default
            self.logger.info("Using USB camera (direct video capture)")
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            self.logger.error(f"Failed to open camera (type: {self.camera_type})")
            raise RuntimeError(f"Could not open camera (type: {self.camera_type})")

        # Try to set resolution (only works for webcams, ignored for HTTP/files)
        if self.camera_type in [CAMERA_USB, CAMERA_CSI] and isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.logger.info(f"Camera resolution set to: {actual_w}x{actual_h}")

    def get_frame(self):
        """
        Captures a single frame.
        :return: (success, frame)
        """
        if not self.cap or not self.cap.isOpened():
            self.logger.warning("Camera not opened, attempting to restart...")
            self.start()

        ret, frame = self.cap.read()
        if not ret:
            self.logger.warning("Failed to read frame")
            return False, None
        return True, frame

    def release(self):
        """Releases the camera resource."""
        if self.cap:
            self.cap.release()
        self.logger.info("Camera released")

if __name__ == "__main__":
    # Test stub
    logging.basicConfig(level=logging.INFO)
    cam = Camera(source=0) # Will fail in Docker w/o proper flags, but good for code structure
    try:
        cam.start()
        ret, frame = cam.get_frame()
        if ret:
            print(f"Captured frame shape: {frame.shape}")
    except Exception as e:
        print(f"Camera test skipped: {e}")
