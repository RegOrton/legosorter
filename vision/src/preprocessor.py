import cv2
import numpy as np
import logging

class Preprocessor:
    def __init__(self, debug_output_dir=None):
        self.logger = logging.getLogger(__name__)
        self.debug_dir = debug_output_dir
        # Thresholds for brick detection
        self.min_area = 1000  # Minimum pixel area to be considered a brick
        self.threshold_val = 50 # Threshold for black conveyor belt vs colorful brick

    def isolate_brick(self, frame):
        """
        Isolates the largest object in the center of the frame.
        Assumes a dark/black background (conveyor belt).
        """
        if frame is None:
            return None, None

        # 1. Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. Thresholding (Assumes dark background, light-ish object)
        # Using simple binary threshold. Adjust 'threshold_val' based on lighting.
        # Ideally, we'd use cv2.THRESH_OTSU if the contrast is high enough.
        _, thresh = cv2.threshold(blurred, self.threshold_val, 255, cv2.THRESH_BINARY)

        # 4. Find Contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None

        # 5. Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area < self.min_area:
            self.logger.debug(f"Largest contour too small: {area}")
            return None, None

        # 6. Bounding Box & ROI
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Add a small padding
        pad = 10
        h_img, w_img = frame.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)

        roi = frame[y1:y2, x1:x2]
        mask_roi = thresh[y1:y2, x1:x2] # The binary mask of the object

        # Optional: Apply mask to ROI to black out background exactly
        # roi_masked = cv2.bitwise_and(roi, roi, mask=mask_roi)

        return roi, largest_contour

if __name__ == "__main__":
    # Test stub
    pass
