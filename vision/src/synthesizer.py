import cv2
import numpy as np
import random
from pathlib import Path

class LegoSynthesizer:
    def __init__(self, background_path, output_size=(224, 224)):
        """
        Args:
            background_path (str or Path): Path to the background image (conveyor belt).
            output_size (tuple): Target size for the output model input (width, height).
        """
        self.output_size = output_size
        self.background_path = Path(background_path)
        self.background = None
        self._load_background()

    def _load_background(self):
        if not self.background_path.exists():
            raise FileNotFoundError(f"Background not found: {self.background_path}")
        
        bg = cv2.imread(str(self.background_path))
        if bg is None:
            raise ValueError(f"Failed to load background image: {self.background_path}")
        self.background = bg

    def _get_random_crop_background(self):
        """Returns a random crop of the background resized to output_size."""
        h, w = self.background.shape[:2]
        th, tw = self.output_size

        # If background is smaller than target, resize it up
        if h < th or w < tw:
            self.background = cv2.resize(self.background, (max(w, tw), max(h, th)))
            h, w = self.background.shape[:2]

        # Random top-left corner
        x = random.randint(0, w - tw)
        y = random.randint(0, h - th)
        
        return self.background[y:y+th, x:x+tw].copy()

    def _rotate_image(self, image, angle):
        """Rotates an image (with alpha channel) around its center."""
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # Compute new bounding dimensions
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # Adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    def generate_sample(self, part_image_path):
        """
        Generates a synthetic training sample.
        Args:
            part_image_path (str or Path): Path to the single lego part image.
        Returns:
            np.array: The synthesised image (BGR).
        """
        # 1. Load Part Image
        # Rebrickable images are usually JPEGs on white backgrounds. 
        # For better synthesis, we ideally want transparent PNGs or we need to mask the white background.
        # Since we downloaded JPEGs, we'll implement a simple white-removal threshold.
        
        part_img = cv2.imread(str(part_image_path))
        if part_img is None:
            return None # Or raise

        # Create Alpha Channel (Masking white background)
        # Convert to gray
        gray = cv2.cvtColor(part_img, cv2.COLOR_BGR2GRAY)
        
        # Threshold: distinct white background usually > 240
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Add alpha channel
        b, g, r = cv2.split(part_img)
        rgba = [b, g, r, mask]
        part_img_rgba = cv2.merge(rgba, 4)

        # 2. Augmentations
        
        # Scale (0.5 to 1.0 relative to target size?)
        # Actually Rebrickable images vary in size. Let's ensure it fits within 80% of output size.
        target_max_dim = int(min(self.output_size) * 0.8)
        
        # Current scale
        h, w = part_img_rgba.shape[:2]
        max_dim = max(h, w)
        scale_factor = target_max_dim / max_dim
        
        # Add random variation to scale
        scale_factor *= random.uniform(0.7, 1.2)
        
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        part_img_rgba = cv2.resize(part_img_rgba, (new_w, new_h))

        # Rotate
        angle = random.randint(0, 360)
        part_img_rgba = self._rotate_image(part_img_rgba, angle)

        # 3. Composite
        background = self._get_random_crop_background()
        
        # Calculate center position on background
        bg_h, bg_w = background.shape[:2]
        fg_h, fg_w = part_img_rgba.shape[:2]
        
        # Center coordinates
        x_offset = (bg_w - fg_w) // 2
        y_offset = (bg_h - fg_h) // 2
        
        # Add random jitter to position
        jitter_x = random.randint(-20, 20)
        jitter_y = random.randint(-20, 20)
        x_offset += jitter_x
        y_offset += jitter_y
        
        # Ensure bounds
        x1 = max(0, x_offset)
        y1 = max(0, y_offset)
        x2 = min(bg_w, x_offset + fg_w)
        y2 = min(bg_h, y_offset + fg_h)
        
        # Source crop coordinates (if offset was negative)
        fg_x1 = max(0, -x_offset)
        fg_y1 = max(0, -y_offset)
        fg_x2 = fg_x1 + (x2 - x1)
        fg_y2 = fg_y1 + (y2 - y1)
        
        if fg_x2 > fg_w or fg_y2 > fg_h:
             # Safety clip
             fg_x2 = min(fg_x2, fg_w)
             fg_y2 = min(fg_y2, fg_h)

        # Alpha Blending
        foreground_roi = part_img_rgba[fg_y1:fg_y2, fg_x1:fg_x2]
        
        alpha_s = foreground_roi[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            background[y1:y2, x1:x2, c] = (alpha_s * foreground_roi[:, :, c] +
                                          alpha_l * background[y1:y2, x1:x2, c])
            
        # 4. Post-processing (Noise/Blur)
        # Quick Gaussian Blur to simulate camera focus/motion
        ksize = random.choice([1, 1, 3])
        if ksize > 1:
            background = cv2.GaussianBlur(background, (ksize, ksize), 0)

        return background
