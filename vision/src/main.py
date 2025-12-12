import cv2
import time
import logging
import os
from camera import Camera
from preprocessor import Preprocessor

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Main")

def main():
    logger.info("Initializing Lego Sorter Vision System...")
    
    # Configuration
    # If running in Docker validation, we might not have a cam, so check for a mock file
    # For now, we'll default to 0 (Webcam) but wrap in try/catch loop
    SOURCE = 0 
    if os.environ.get("USE_MOCK_VIDEO"):
        SOURCE = "/app/input/mock_conveyor.mp4"

    cam = Camera(source=SOURCE)
    processor = Preprocessor(debug_output_dir="/app/output")

    try:
        cam.start()
    except Exception as e:
        logger.error(f"Could not start camera: {e}")
        return

    logger.info("Starting Main Loop. Press Q to quit (if interactive).")
    
    frame_count = 0
    
    # Run a limited loop for verification if not interactive
    MAX_FRAMES = 100 
    
    while frame_count < MAX_FRAMES:
        ret, frame = cam.get_frame()
        if not ret:
            logger.warning("No frame retrieved.")
            break

        roi, contour = processor.isolate_brick(frame)

        if roi is not None:
            logger.info(f"Brick Detected! ROI Shape: {roi.shape}")
            # Save the first detected brick for verification
            cv2.imwrite(f"/app/output/brick_{frame_count}.png", roi)
        else:
            logger.debug("No brick detected.")

        frame_count += 1
        time.sleep(0.1)

    cam.release()
    logger.info("Vision System Shutdown.")

if __name__ == "__main__":
    main()
