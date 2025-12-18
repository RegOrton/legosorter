"""
Fast background differencing detector for isolated object detection.

Uses frame differencing from a calibrated background reference to detect
objects in clean, controlled environments (solid background, single object).
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BackgroundDiffDetector:
    """Fast object detection using background frame differencing."""

    def __init__(
        self,
        min_area_percent: float = 0.001,
        max_area_percent: float = 0.15,
        diff_threshold: int = 30,
        center_tolerance: float = 0.15,
        edge_margin: int = 20,
        min_aspect_ratio: float = 0.3,
        max_aspect_ratio: float = 3.0,
    ):
        """
        Initialize the detector.

        Args:
            min_area_percent: Minimum contour area as fraction of frame (0.001 = 0.1% of frame, filters noise)
            max_area_percent: Maximum contour area as fraction of frame (0.15 = 15% of frame)
            diff_threshold: Brightness difference threshold for detecting changes
            center_tolerance: Fraction of frame center to consider "centered" (0.15 = ±15%)
            edge_margin: Pixels from frame edge where object is considered "touching"
            min_aspect_ratio: Minimum width/height ratio for valid objects
            max_aspect_ratio: Maximum width/height ratio for valid objects
        """
        self.min_area_percent = min_area_percent
        self.max_area_percent = max_area_percent
        self.diff_threshold = diff_threshold
        self.center_tolerance = center_tolerance
        self.edge_margin = edge_margin
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

        self.bg_frame = None  # Reference background frame
        self.is_calibrated = False

        # Morphological kernel for noise reduction
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def calibrate(self, frame: np.ndarray, save_to_disk: bool = True) -> bool:
        """
        Calibrate background by storing reference frame.

        Call this when the frame is empty and shows only the background.

        Args:
            frame: Input frame (BGR)
            save_to_disk: If True, saves calibration to disk for persistence

        Returns:
            True if calibration succeeded
        """
        if frame is None or frame.size == 0:
            logger.error("Cannot calibrate with empty frame")
            return False

        # Store a copy of the frame as reference
        self.bg_frame = frame.copy().astype(np.float32)
        self.is_calibrated = True
        logger.info(f"Background calibrated: {frame.shape}")

        # Save to disk for persistence
        if save_to_disk:
            self._save_calibration()

        return True

    def reset_calibration(self) -> None:
        """Reset calibration state."""
        self.bg_frame = None
        self.is_calibrated = False
        logger.info("Calibration reset")

    def detect(self, frame: np.ndarray) -> Tuple[List[Dict], bool, str]:
        """
        Detect objects in frame using background differencing.

        Args:
            frame: Input frame (BGR)

        Returns:
            Tuple of (bounding_boxes, center_detected, status_message)
            - bounding_boxes: List of detected bounding box dicts
            - center_detected: True if any bbox is centered and stable
            - status_message: Human-readable status (for debugging)
        """
        if not self.is_calibrated or self.bg_frame is None:
            return [], False, "Not calibrated"

        if frame is None or frame.size == 0:
            return [], False, "Invalid frame"

        h, w = frame.shape[:2]
        frame_center_x, frame_center_y = w // 2, h // 2

        # Calculate min_area and max_area from percentage of frame
        frame_area = h * w
        min_area = int(frame_area * self.min_area_percent)
        max_area = int(frame_area * self.max_area_percent)

        # Convert to float for differencing
        current_frame = frame.astype(np.float32)

        # Compute absolute difference from background
        diff = cv2.absdiff(current_frame, self.bg_frame)

        # Convert to grayscale for threshold
        diff_gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)

        # Threshold to binary mask
        _, fg_mask = cv2.threshold(diff_gray, self.diff_threshold, 255, cv2.THRESH_BINARY)

        # Morphological operations to reduce noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []
        center_detected = False

        # Count valid contours for validation
        valid_contour_count = 0

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if area < min_area or area > max_area:
                continue

            valid_contour_count += 1

            # Get bounding box
            x, y, box_w, box_h = cv2.boundingRect(contour)

            # Calculate bbox center
            bbox_center_x = x + box_w // 2
            bbox_center_y = y + box_h // 2

            # Check if bbox is centered
            center_threshold_x = w * self.center_tolerance
            center_threshold_y = h * self.center_tolerance

            is_centered = (
                abs(bbox_center_x - frame_center_x) < center_threshold_x and
                abs(bbox_center_y - frame_center_y) < center_threshold_y
            )

            # Check if bbox touches frame edges
            touches_edge = (
                x < self.edge_margin or
                y < self.edge_margin or
                x + box_w > w - self.edge_margin or
                y + box_h > h - self.edge_margin
            )

            # Calculate aspect ratio
            aspect_ratio = box_w / box_h if box_h > 0 else 0
            aspect_ratio_valid = (
                self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio
            )

            # Determine validation status
            validation_status = "valid"
            if touches_edge:
                validation_status = "touching_edge"
            elif not aspect_ratio_valid:
                validation_status = "bad_aspect_ratio"
            elif not is_centered:
                validation_status = "not_centered"

            if is_centered and not touches_edge and aspect_ratio_valid:
                center_detected = True

            bbox_dict = {
                'x': int(x),
                'y': int(y),
                'width': int(box_w),
                'height': int(box_h),
                'center_x': int(bbox_center_x),
                'center_y': int(bbox_center_y),
                'area': int(area),
                'is_centered': is_centered,
                'touches_edge': touches_edge,
                'aspect_ratio': float(aspect_ratio),
                'aspect_ratio_valid': aspect_ratio_valid,
                'validation_status': validation_status,
            }

            bounding_boxes.append(bbox_dict)

        # Determine status message
        if valid_contour_count == 0:
            status = "No objects detected"
        elif valid_contour_count == 1:
            status = "Single object detected"
            if center_detected:
                status += " (centered and valid)"
        else:
            status = f"Multiple objects detected ({valid_contour_count})"

        return bounding_boxes, center_detected, status

    def debug_detect(self, frame: np.ndarray) -> Tuple[List[Dict], bool, str, np.ndarray]:
        """
        Detect with debug visualization overlay.

        Returns:
            Tuple of (bounding_boxes, center_detected, status, debug_frame)
            - debug_frame: Frame with overlays for debugging
        """
        bounding_boxes, center_detected, status = self.detect(frame)

        debug_frame = frame.copy()
        h, w = frame.shape[:2]
        frame_center_x, frame_center_y = w // 2, h // 2

        # Draw frame center point
        cv2.circle(debug_frame, (frame_center_x, frame_center_y), 5, (0, 255, 255), -1)

        # Draw center tolerance zone
        center_tolerance_x = int(w * self.center_tolerance)
        center_tolerance_y = int(h * self.center_tolerance)
        cv2.rectangle(
            debug_frame,
            (frame_center_x - center_tolerance_x, frame_center_y - center_tolerance_y),
            (frame_center_x + center_tolerance_x, frame_center_y + center_tolerance_y),
            (255, 255, 0),  # Cyan
            2
        )

        # Draw edge margin zones
        cv2.rectangle(debug_frame, (0, 0), (self.edge_margin, h), (0, 0, 255), 1)  # Red
        cv2.rectangle(debug_frame, (w - self.edge_margin, 0), (w, h), (0, 0, 255), 1)
        cv2.rectangle(debug_frame, (0, 0), (w, self.edge_margin), (0, 0, 255), 1)
        cv2.rectangle(debug_frame, (0, h - self.edge_margin), (w, h), (0, 0, 255), 1)

        # Draw bounding boxes with color coding
        frame_area = h * w
        for bbox in bounding_boxes:
            x, y, width, height = bbox['x'], bbox['y'], bbox['width'], bbox['height']

            # Color based on validation status
            if bbox['validation_status'] == 'valid' and bbox['is_centered']:
                color = (0, 255, 0)  # Green - ready
            elif bbox['validation_status'] == 'valid':
                color = (255, 255, 0)  # Cyan - valid but not centered
            else:
                color = (0, 0, 255)  # Red - invalid

            cv2.rectangle(debug_frame, (x, y), (x + width, y + height), color, 2)

            # Draw center point
            cv2.circle(debug_frame, (bbox['center_x'], bbox['center_y']), 3, color, -1)

            # Add text label with area percentage
            area_percent = (bbox['area'] / frame_area) * 100
            label = f"{area_percent:.1f}% AR:{bbox['aspect_ratio']:.2f}"
            cv2.putText(
                debug_frame,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )

        # Add status text
        cv2.putText(
            debug_frame,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        return bounding_boxes, center_detected, status, debug_frame

    def get_calibration_status(self) -> Dict:
        """Get calibration status info."""
        return {
            'is_calibrated': self.is_calibrated,
            'bg_frame_shape': tuple(self.bg_frame.shape) if self.bg_frame is not None else None,
            'params': {
                'min_area_percent': self.min_area_percent,
                'max_area_percent': self.max_area_percent,
                'diff_threshold': self.diff_threshold,
                'center_tolerance': self.center_tolerance,
                'edge_margin': self.edge_margin,
                'min_aspect_ratio': self.min_aspect_ratio,
                'max_aspect_ratio': self.max_aspect_ratio,
            }
        }

    def _save_calibration(self) -> bool:
        """Save calibration background frame to disk."""
        if self.bg_frame is None:
            return False

        try:
            # Determine save path
            base_path = Path("/app/output") if Path("/app/output").exists() else Path(__file__).parent.parent / "output"
            base_path.mkdir(parents=True, exist_ok=True)
            calib_path = base_path / "calibration_bg.npy"

            # Save as numpy array to preserve float32 precision
            np.save(str(calib_path), self.bg_frame)
            logger.info(f"Saved calibration to {calib_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
            return False

    def load_calibration(self) -> bool:
        """Load calibration background frame from disk."""
        try:
            # Determine load path
            base_path = Path("/app/output") if Path("/app/output").exists() else Path(__file__).parent.parent / "output"
            calib_path = base_path / "calibration_bg.npy"

            if not calib_path.exists():
                logger.info("No saved calibration found")
                return False

            # Load numpy array
            self.bg_frame = np.load(str(calib_path))
            self.is_calibrated = True
            logger.info(f"Loaded calibration from {calib_path}, shape: {self.bg_frame.shape}")
            return True
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return False
