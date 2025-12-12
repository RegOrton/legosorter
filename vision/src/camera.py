import cv2
import time
import logging

class Camera:
    def __init__(self, source=0, width=1280, height=720):
        """
        Initialize the Camera.
        :param source: Camera index (int) or video file path (str).
        """
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        self.logger = logging.getLogger(__name__)

    def start(self):
        """Opens the video source."""
        self.logger.info(f"Opening camera source: {self.source}")
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            self.logger.error(f"Failed to open camera source: {self.source}")
            raise RuntimeError(f"Could not open camera {self.source}")

        # Try to set resolution (only works for webcams, ignored for files)
        if isinstance(self.source, int):
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
