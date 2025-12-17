import cv2
import time
import logging
import os
import threading
from webcam_client import create_webcam_capture

# Camera type constants
CAMERA_USB = "usb"
CAMERA_CSI = "csi"
CAMERA_HTTP = "http"
CAMERA_VIDEO_FILE = "video_file"

class Camera:
    def __init__(self, source=0, width=1280, height=720, camera_type=None, video_file=None, playback_speed=1.0):
        """
        Initialize the Camera.
        :param source: Camera index (int) or video file path (str).
        :param width: Desired frame width
        :param height: Desired frame height
        :param camera_type: Camera type: "usb", "csi", "http", or "video_file". If None, auto-detect from environment.
        :param video_file: Path to video file (required if camera_type is "video_file")
        :param playback_speed: Playback speed multiplier (1.0 = normal speed, 2.0 = 2x, 0.5 = half speed)
        """
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        self.logger = logging.getLogger(__name__)
        self.video_file = video_file
        self.playback_speed = playback_speed
        self._lock = threading.Lock()  # Thread-safe access to video capture
        self.frame_delay = None  # Will be calculated based on video FPS
        self.last_frame_time = 0

        # Determine camera type
        if camera_type is not None:
            self.camera_type = camera_type
        elif os.getenv('USE_HTTP_WEBCAM', 'false').lower() == 'true':
            self.camera_type = CAMERA_HTTP
        else:
            self.camera_type = CAMERA_USB

    def set_camera_type(self, camera_type: str, video_file: str = None):
        """
        Switch camera type and restart the camera.
        :param camera_type: "usb", "csi", "http", or "video_file"
        :param video_file: Path to video file (required if camera_type is "video_file")
        """
        self.logger.info(f"Switching camera type from {self.camera_type} to {camera_type}")
        self.release()
        self.camera_type = camera_type
        if video_file:
            self.video_file = video_file
        self.start()

    def set_playback_speed(self, speed: float):
        """
        Set video playback speed.
        :param speed: Playback speed multiplier (1.0 = normal, 2.0 = 2x, 0.5 = half speed)
        """
        if speed <= 0:
            self.logger.warning(f"Invalid playback speed {speed}, must be > 0")
            return
        self.logger.info(f"Setting playback speed to {speed}x")
        self.playback_speed = speed
        if self.frame_delay is not None:
            # Recalculate delay based on new speed
            fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 30
            self.frame_delay = (1.0 / fps) / self.playback_speed

    def start(self):
        """Opens the video source."""
        self.logger.info(f"Opening camera (type: {self.camera_type}, source: {self.source})")

        if self.camera_type == CAMERA_VIDEO_FILE:
            if not self.video_file:
                self.logger.error("Video file path not specified for video_file camera type")
                raise RuntimeError("Video file path required for video_file camera type")

            self.logger.info(f"Using video file: {self.video_file} (speed: {self.playback_speed}x)")
            self.cap = cv2.VideoCapture(self.video_file)

            if not self.cap.isOpened():
                self.logger.error(f"Failed to open video file: {self.video_file}")
                raise RuntimeError(f"Could not open video file: {self.video_file}")

            # Calculate frame delay based on video FPS
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # Default fallback
            self.frame_delay = (1.0 / fps) / self.playback_speed
            self.last_frame_time = time.time()

            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.logger.info(f"Video opened: {fps} FPS, {total_frames} frames, delay: {self.frame_delay:.4f}s")

        elif self.camera_type == CAMERA_HTTP:
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
        Captures a single frame (thread-safe).
        :return: (success, frame)
        """
        with self._lock:
            if not self.cap or not self.cap.isOpened():
                self.logger.warning("Camera not opened, attempting to restart...")
                self.start()

            try:
                ret, frame = self.cap.read()
            except Exception as e:
                self.logger.error(f"Exception while reading frame: {e}")
                return False, None

            # Handle video file looping
            if not ret and self.camera_type == CAMERA_VIDEO_FILE:
                self.logger.info("Video file reached end, looping back to start")
                try:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    self.last_frame_time = time.time()
                except Exception as e:
                    self.logger.error(f"Exception while looping video: {e}")
                    return False, None

            if not ret:
                self.logger.warning("Failed to read frame")
                return False, None

            # Validate frame data
            if frame is None or frame.size == 0:
                self.logger.warning("Invalid frame data received")
                return False, None

            # Handle playback speed timing for video files
            if self.camera_type == CAMERA_VIDEO_FILE and self.frame_delay:
                current_time = time.time()
                elapsed = current_time - self.last_frame_time

                # If we're reading too fast, sleep to match desired playback speed
                if elapsed < self.frame_delay:
                    time.sleep(self.frame_delay - elapsed)

                self.last_frame_time = time.time()

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
