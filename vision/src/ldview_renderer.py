import subprocess
import cv2
import numpy as np
import random
import tempfile
import os
from pathlib import Path
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class LDViewRenderer:
    """
    Renderer for LEGO .dat files using LDView command-line tool.
    Generates realistic 3D renders for training data with various augmentations.
    """

    def __init__(
        self,
        ldview_path: str = "ldview",
        ldraw_dir: Optional[str] = None,
        output_size: Tuple[int, int] = (224, 224),
        background_path: Optional[str] = None
    ):
        """
        Initialize LDView renderer.

        Args:
            ldview_path: Path to LDView executable (default: "ldview" assumes it's in PATH)
            ldraw_dir: Path to LDraw parts library directory (optional, auto-detects from LDRAWDIR env var)
            output_size: Output image size (width, height)
            background_path: Optional path to background image for compositing
        """
        self.ldview_path = ldview_path

        # Auto-detect LDraw directory from environment variable if not provided
        if ldraw_dir is None:
            ldraw_dir = os.environ.get('LDRAWDIR')
            if ldraw_dir:
                logger.info(f"Using LDraw directory from LDRAWDIR env var: {ldraw_dir}")

        self.ldraw_dir = ldraw_dir
        self.output_size = output_size
        self.background_path = background_path
        self.background = None

        if background_path:
            self._load_background()

        # Verify LDView is available
        self._verify_ldview()

    def _verify_ldview(self):
        """Check if LDView is available."""
        try:
            result = subprocess.run(
                [self.ldview_path, "-v"],
                capture_output=True,
                timeout=5
            )
            logger.info(f"LDView found: {result.stdout.decode('utf-8', errors='ignore')}")
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            raise RuntimeError(
                f"LDView not found at '{self.ldview_path}'. "
                f"Please install LDView and ensure it's in your PATH or specify the full path. "
                f"Error: {e}"
            )

    def _load_background(self):
        """Load background image if specified."""
        if not self.background_path:
            return

        bg_path = Path(self.background_path)
        if not bg_path.exists():
            raise FileNotFoundError(f"Background not found: {self.background_path}")

        bg = cv2.imread(str(bg_path))
        if bg is None:
            raise ValueError(f"Failed to load background: {self.background_path}")
        self.background = bg
        logger.info(f"Loaded background: {self.background_path}")

    def _get_random_crop_background(self) -> np.ndarray:
        """Returns a random crop of the background resized to output_size."""
        if self.background is None:
            # Generate solid color background
            color = np.random.randint(200, 255, 3, dtype=np.uint8)
            return np.full((*self.output_size[::-1], 3), color, dtype=np.uint8)

        h, w = self.background.shape[:2]
        th, tw = self.output_size[::-1]  # height, width

        # If background is smaller than target, resize it up
        if h < th or w < tw:
            self.background = cv2.resize(self.background, (max(w, tw), max(h, th)))
            h, w = self.background.shape[:2]

        # Random crop
        x = random.randint(0, w - tw)
        y = random.randint(0, h - th)

        return self.background[y:y+th, x:x+tw].copy()

    def render(
        self,
        dat_file_path: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        distance: Optional[float] = None,
        use_transparency: bool = True,
        lighting: str = "default"
    ) -> np.ndarray:
        """
        Render a .dat file using LDView.

        Args:
            dat_file_path: Path to the .dat file
            latitude: Camera latitude angle (default: random between -30 and 30)
            longitude: Camera longitude angle (default: random between 0 and 360)
            distance: Camera distance multiplier (default: random between 0.8 and 1.5)
            use_transparency: Enable transparent background
            lighting: Lighting preset ("default", "bright", "dim")

        Returns:
            Rendered image as numpy array (BGR format)
        """
        dat_path = Path(dat_file_path)
        if not dat_path.exists():
            raise FileNotFoundError(f"DAT file not found: {dat_file_path}")

        # Random camera angles if not specified
        if latitude is None:
            latitude = random.uniform(-30, 30)
        if longitude is None:
            longitude = random.uniform(0, 360)
        if distance is None:
            distance = random.uniform(0.8, 1.5)

        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            # Build LDView command with balanced quality/speed
            cmd = [
                self.ldview_path,
                str(dat_path),
                f"-SaveSnapshot={output_path}",
                f"-SaveWidth={self.output_size[0]}",
                f"-SaveHeight={self.output_size[1]}",
                f"-cg{latitude},{longitude}",  # Camera globe (lat, long)
                f"-ca0.1",  # Camera FOV
                "-SaveAlpha=1" if use_transparency else "-SaveAlpha=0",
                "-SaveActualSize=0",  # Don't save at actual size
                "-AutoCrop=0",  # Don't auto-crop
                # Balanced rendering options (faster than high quality)
                "-EdgeOnly=0",  # Show edges
                "-Lighting=1",  # Enable lighting
                "-Shading=1",  # Enable shading
            ]

            # Add LDraw directory if specified
            if self.ldraw_dir:
                cmd.append(f"-LDrawDir={self.ldraw_dir}")

            # Lighting adjustments
            if lighting == "bright":
                cmd.extend(["-LightVector=0,1,1"])
            elif lighting == "dim":
                cmd.extend(["-LightVector=0,0.5,0.5"])

            # Run LDView
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30,
                cwd=str(dat_path.parent)  # Run in the directory containing the .dat file
            )

            if result.returncode != 0:
                error_msg = result.stderr.decode('utf-8', errors='ignore')
                logger.warning(f"LDView warning: {error_msg}")

            # Load rendered image
            img = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Failed to load rendered image from {output_path}")

            return img

        finally:
            # Cleanup temporary file
            Path(output_path).unlink(missing_ok=True)

    def generate_sample(
        self,
        dat_file_path: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        distance: Optional[float] = None,
        apply_augmentations: bool = True
    ) -> np.ndarray:
        """
        Generate a training sample with random augmentations.

        Args:
            dat_file_path: Path to the .dat file
            latitude: Camera latitude (randomized if None)
            longitude: Camera longitude (randomized if None)
            distance: Camera distance (randomized if None)
            apply_augmentations: Apply post-processing augmentations

        Returns:
            Augmented training image (BGR format)
        """
        # Render with transparency
        rendered = self.render(
            dat_file_path,
            latitude=latitude,
            longitude=longitude,
            distance=distance,
            use_transparency=True,
            lighting=random.choice(["default", "bright", "dim"])
        )

        # Get background
        background = self._get_random_crop_background()

        # Composite if we have alpha channel
        if rendered.shape[2] == 4:
            # Extract alpha channel
            alpha = rendered[:, :, 3] / 255.0
            alpha = alpha[:, :, np.newaxis]

            # Resize if needed
            if rendered.shape[:2] != background.shape[:2]:
                rendered_rgb = cv2.resize(
                    rendered[:, :, :3],
                    (background.shape[1], background.shape[0])
                )
                alpha = cv2.resize(
                    alpha,
                    (background.shape[1], background.shape[0])
                )[:, :, np.newaxis]
            else:
                rendered_rgb = rendered[:, :, :3]

            # Alpha blend
            result = (alpha * rendered_rgb + (1 - alpha) * background).astype(np.uint8)
        else:
            # No alpha channel, just resize
            result = cv2.resize(rendered, (background.shape[1], background.shape[0]))

        # Apply augmentations
        if apply_augmentations:
            result = self._apply_augmentations(result)

        return result

    def _apply_augmentations(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentations to the image."""
        # Random blur (simulate motion/focus)
        if random.random() < 0.3:
            ksize = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)

        # Random brightness adjustment
        if random.random() < 0.5:
            brightness = random.uniform(0.8, 1.2)
            image = np.clip(image * brightness, 0, 255).astype(np.uint8)

        # Random noise
        if random.random() < 0.2:
            noise = np.random.normal(0, 5, image.shape).astype(np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return image

    def batch_generate(
        self,
        dat_file_paths: List[str],
        samples_per_part: int = 10,
        progress_callback: Optional[callable] = None
    ) -> List[Tuple[np.ndarray, str]]:
        """
        Generate multiple training samples from a list of .dat files.

        Args:
            dat_file_paths: List of paths to .dat files
            samples_per_part: Number of samples to generate per part
            progress_callback: Optional callback function(current, total, part_name)

        Returns:
            List of (image, part_id) tuples
        """
        samples = []
        total = len(dat_file_paths) * samples_per_part
        current = 0

        for dat_path in dat_file_paths:
            part_id = Path(dat_path).stem

            for i in range(samples_per_part):
                try:
                    img = self.generate_sample(dat_path)
                    samples.append((img, part_id))
                    current += 1

                    if progress_callback:
                        progress_callback(current, total, part_id)

                except Exception as e:
                    logger.error(f"Failed to render {dat_path}: {e}")
                    current += 1

        return samples


def save_samples(samples: List[Tuple[np.ndarray, str]], output_dir: str):
    """
    Save generated samples to disk.

    Args:
        samples: List of (image, part_id) tuples
        output_dir: Directory to save images
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Group by part_id
    part_counts = {}

    for img, part_id in samples:
        if part_id not in part_counts:
            part_counts[part_id] = 0

        part_dir = output_path / part_id
        part_dir.mkdir(exist_ok=True)

        count = part_counts[part_id]
        save_path = part_dir / f"{part_id}_{count:04d}.jpg"

        cv2.imwrite(str(save_path), img)
        part_counts[part_id] += 1

    logger.info(f"Saved {len(samples)} samples to {output_dir}")
