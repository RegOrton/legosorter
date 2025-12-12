import cv2
import numpy as np
import random
from pathlib import Path


class LegoSynthesizer:
    def __init__(self, background_path, output_size=(224, 224)):
        """
        Enhanced synthesizer for generating realistic training samples.
        Addresses sim-to-real domain gap with perspective transforms, shadows,
        and camera effects.

        Args:
            background_path (str or Path): Path to background image OR directory of backgrounds.
            output_size (tuple): Target size for the output model input (width, height).
        """
        self.output_size = output_size
        self.background_path = Path(background_path)
        self.backgrounds = []
        self._load_backgrounds()

    def _load_backgrounds(self):
        """Load single background or multiple backgrounds from directory."""
        if self.background_path.is_dir():
            # Load all images from directory
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                self.backgrounds.extend(list(self.background_path.glob(ext)))
            if len(self.backgrounds) == 0:
                raise ValueError(f"No background images found in {self.background_path}")
        else:
            # Single background file
            if not self.background_path.exists():
                raise FileNotFoundError(f"Background not found: {self.background_path}")
            self.backgrounds = [self.background_path]

    def _get_random_crop_background(self):
        """Returns a random crop from a randomly selected background."""
        # Select random background
        bg_path = random.choice(self.backgrounds)
        bg = cv2.imread(str(bg_path))
        if bg is None:
            raise ValueError(f"Failed to load background image: {bg_path}")

        h, w = bg.shape[:2]
        th, tw = self.output_size

        # If background is smaller than target, resize it up
        if h < th or w < tw:
            scale = max(th / h, tw / w) * 1.1
            bg = cv2.resize(bg, (int(w * scale), int(h * scale)))
            h, w = bg.shape[:2]

        # Random top-left corner
        x = random.randint(0, w - tw)
        y = random.randint(0, h - th)

        return bg[y:y+th, x:x+tw].copy()

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

        return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    def _apply_perspective(self, image, intensity=0.15):
        """
        Apply perspective transform to simulate different viewing angles.
        This helps bridge the gap between single-viewpoint CGI and real-world
        multi-angle scenarios.

        Args:
            image: RGBA image
            intensity: How extreme the perspective shift can be (0-1)
        """
        h, w = image.shape[:2]

        # Original corners
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

        # Randomly perturb corners to create perspective effect
        max_shift = int(min(h, w) * intensity)
        pts2 = pts1.copy()

        # Apply random shifts to each corner
        for i in range(4):
            pts2[i, 0] += random.randint(-max_shift, max_shift)
            pts2[i, 1] += random.randint(-max_shift, max_shift)

        # Ensure we don't invert the quad (keep corners in reasonable positions)
        # Top-left should stay top-left-ish, etc.
        pts2[0] = np.clip(pts2[0], [0, 0], [w * 0.4, h * 0.4])
        pts2[1] = np.clip(pts2[1], [w * 0.6, 0], [w, h * 0.4])
        pts2[2] = np.clip(pts2[2], [0, h * 0.6], [w * 0.4, h])
        pts2[3] = np.clip(pts2[3], [w * 0.6, h * 0.6], [w, h])

        M = cv2.getPerspectiveTransform(pts1, pts2)

        # Calculate new bounding box size
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, M)
        x_coords = transformed_corners[:, 0, 0]
        y_coords = transformed_corners[:, 0, 1]

        new_w = int(np.ceil(x_coords.max() - x_coords.min()))
        new_h = int(np.ceil(y_coords.max() - y_coords.min()))

        # Adjust transform to keep image in frame
        M[0, 2] -= x_coords.min()
        M[1, 2] -= y_coords.min()

        return cv2.warpPerspective(
            image, M, (new_w, new_h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

    def _add_shadow(self, composite, mask, x1, y1, x2, y2):
        """
        Add a synthetic drop shadow to simulate directional lighting.
        This is critical for sim-to-real transfer as real scenes have shadows.

        Args:
            composite: The background image being composited onto
            mask: Alpha mask of the foreground object
            x1, y1, x2, y2: Bounding box of where the object is placed
        """
        if random.random() > 0.7:  # 70% chance of shadow
            return composite

        # Random light direction (angle in degrees)
        light_angle = random.uniform(30, 150)

        # Shadow offset based on light direction
        shadow_distance = random.randint(5, 15)
        offset_x = int(shadow_distance * np.cos(np.radians(light_angle)))
        offset_y = int(shadow_distance * np.sin(np.radians(light_angle)))

        # Create shadow from mask
        shadow = mask.astype(np.float32) / 255.0

        # Blur shadow for soft edges
        blur_size = random.choice([11, 15, 21, 25])
        shadow = cv2.GaussianBlur(shadow, (blur_size, blur_size), 0)

        # Shadow intensity
        shadow_intensity = random.uniform(0.2, 0.5)

        # Calculate shadow position
        sh, sw = shadow.shape
        bg_h, bg_w = composite.shape[:2]

        # Shadow destination
        sx1 = max(0, x1 + offset_x)
        sy1 = max(0, y1 + offset_y)
        sx2 = min(bg_w, x2 + offset_x)
        sy2 = min(bg_h, y2 + offset_y)

        # Shadow source (handle clipping)
        src_x1 = max(0, -offset_x) if x1 + offset_x < 0 else 0
        src_y1 = max(0, -offset_y) if y1 + offset_y < 0 else 0
        src_x2 = src_x1 + (sx2 - sx1)
        src_y2 = src_y1 + (sy2 - sy1)

        # Ensure we don't exceed shadow dimensions
        src_x2 = min(src_x2, sw)
        src_y2 = min(src_y2, sh)
        sx2 = sx1 + (src_x2 - src_x1)
        sy2 = sy1 + (src_y2 - src_y1)

        if sx2 > sx1 and sy2 > sy1 and src_x2 > src_x1 and src_y2 > src_y1:
            shadow_roi = shadow[src_y1:src_y2, src_x1:src_x2]
            for c in range(3):
                composite[sy1:sy2, sx1:sx2, c] = (
                    composite[sy1:sy2, sx1:sx2, c] * (1 - shadow_intensity * shadow_roi)
                ).astype(np.uint8)

        return composite

    def _apply_camera_effects(self, image):
        """
        Apply realistic camera effects to bridge sim-to-real gap.
        Includes noise, brightness/contrast variation, and blur.
        """
        result = image.copy()

        # Brightness and contrast adjustment (simulates exposure variation)
        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)  # Contrast
            beta = random.randint(-25, 25)     # Brightness
            result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)

        # Add sensor noise
        if random.random() < 0.4:
            noise_sigma = random.uniform(3, 12)
            noise = np.random.normal(0, noise_sigma, result.shape).astype(np.int16)
            result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Gaussian blur (simulates focus/motion blur)
        if random.random() < 0.3:
            ksize = random.choice([3, 5])
            result = cv2.GaussianBlur(result, (ksize, ksize), 0)

        # Slight color temperature shift (simulates different lighting)
        if random.random() < 0.3:
            # Warm or cool shift
            if random.random() < 0.5:
                # Warm (more red/yellow)
                result[:, :, 2] = np.clip(result[:, :, 2].astype(np.int16) + random.randint(5, 15), 0, 255).astype(np.uint8)
                result[:, :, 0] = np.clip(result[:, :, 0].astype(np.int16) - random.randint(3, 10), 0, 255).astype(np.uint8)
            else:
                # Cool (more blue)
                result[:, :, 0] = np.clip(result[:, :, 0].astype(np.int16) + random.randint(5, 15), 0, 255).astype(np.uint8)
                result[:, :, 2] = np.clip(result[:, :, 2].astype(np.int16) - random.randint(3, 10), 0, 255).astype(np.uint8)

        return result

    def generate_sample(self, part_image_path):
        """
        Generates a synthetic training sample with enhanced augmentation.

        Args:
            part_image_path (str or Path): Path to the single lego part image.
        Returns:
            np.array: The synthesized image (BGR).
        """
        # 1. Load Part Image
        part_img = cv2.imread(str(part_image_path))
        if part_img is None:
            return None

        # Create Alpha Channel (Masking white background)
        gray = cv2.cvtColor(part_img, cv2.COLOR_BGR2GRAY)

        # Threshold: distinct white background usually > 240
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        # Morphological operations to clean up mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Add alpha channel
        b, g, r = cv2.split(part_img)
        part_img_rgba = cv2.merge([b, g, r, mask])

        # 2. Geometric Augmentations

        # Scale - ensure it fits within 80% of output size with variation
        target_max_dim = int(min(self.output_size) * 0.8)
        h, w = part_img_rgba.shape[:2]
        max_dim = max(h, w)
        scale_factor = target_max_dim / max_dim
        scale_factor *= random.uniform(0.6, 1.3)

        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        part_img_rgba = cv2.resize(part_img_rgba, (new_w, new_h))

        # Perspective transform (simulates viewing angle changes)
        if random.random() < 0.7:  # 70% chance
            part_img_rgba = self._apply_perspective(part_img_rgba, intensity=0.12)

        # Rotate (2D rotation around camera axis)
        angle = random.randint(0, 360)
        part_img_rgba = self._rotate_image(part_img_rgba, angle)

        # 3. Composite onto background
        background = self._get_random_crop_background()

        bg_h, bg_w = background.shape[:2]
        fg_h, fg_w = part_img_rgba.shape[:2]

        # Center coordinates with jitter
        x_offset = (bg_w - fg_w) // 2 + random.randint(-25, 25)
        y_offset = (bg_h - fg_h) // 2 + random.randint(-25, 25)

        # Ensure bounds
        x1 = max(0, x_offset)
        y1 = max(0, y_offset)
        x2 = min(bg_w, x_offset + fg_w)
        y2 = min(bg_h, y_offset + fg_h)

        # Source crop coordinates
        fg_x1 = max(0, -x_offset)
        fg_y1 = max(0, -y_offset)
        fg_x2 = fg_x1 + (x2 - x1)
        fg_y2 = fg_y1 + (y2 - y1)

        # Safety clip
        fg_x2 = min(fg_x2, fg_w)
        fg_y2 = min(fg_y2, fg_h)

        # Get the mask for shadow generation (before alpha blending)
        if fg_y2 > fg_y1 and fg_x2 > fg_x1:
            shadow_mask = part_img_rgba[fg_y1:fg_y2, fg_x1:fg_x2, 3].copy()

            # Add shadow before the part
            background = self._add_shadow(background, shadow_mask, x1, y1, x2, y2)

            # Alpha Blending
            foreground_roi = part_img_rgba[fg_y1:fg_y2, fg_x1:fg_x2]
            alpha_s = foreground_roi[:, :, 3:4] / 255.0
            alpha_l = 1.0 - alpha_s

            background[y1:y2, x1:x2] = (
                alpha_s * foreground_roi[:, :, :3] +
                alpha_l * background[y1:y2, x1:x2]
            ).astype(np.uint8)

        # 4. Apply camera effects
        background = self._apply_camera_effects(background)

        return background
