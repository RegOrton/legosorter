"""
Test high-resolution, realistic LDView rendering.
"""

import sys
from pathlib import Path
import cv2
import logging

sys.path.insert(0, str(Path(__file__).parent))

from ldview_renderer import LDViewRenderer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_high_res_rendering():
    """Test high-resolution realistic rendering."""

    base_dir = Path(__file__).parent.parent
    dat_dir = base_dir / "input" / "dat_files"
    output_dir = base_dir / "output" / "debug" / "high_res_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find .dat files
    dat_files = list(dat_dir.glob("*.dat"))[:3]  # Test first 3 parts

    if not dat_files:
        logger.error(f"No .dat files found in {dat_dir}")
        return

    logger.info(f"Testing high-res rendering with {len(dat_files)} parts")

    # Initialize renderer with higher resolution
    renderer = LDViewRenderer(
        ldview_path="ldview",
        output_size=(448, 448)  # Higher resolution
    )

    for dat_file in dat_files:
        part_name = dat_file.stem
        logger.info(f"\nRendering part: {part_name}")

        part_output_dir = output_dir / part_name
        part_output_dir.mkdir(exist_ok=True)

        # Test different angles with realistic rendering
        angles = [
            {"lat": 20, "lon": 45, "name": "angle1"},
            {"lat": -15, "lon": 135, "name": "angle2"},
            {"lat": 30, "lon": 225, "name": "angle3"},
        ]

        for angle in angles:
            try:
                logger.info(f"  Rendering {angle['name']}...")

                # Generate with augmentations
                img = renderer.generate_sample(
                    str(dat_file),
                    latitude=angle['lat'],
                    longitude=angle['lon'],
                    apply_augmentations=True
                )

                output_path = part_output_dir / f"{part_name}_{angle['name']}_448.jpg"
                cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                logger.info(f"    Saved: {output_path}")

            except Exception as e:
                logger.error(f"    Failed: {e}")

    logger.info(f"\n✅ Test complete! Check high-res images in: {output_dir}")


if __name__ == "__main__":
    test_high_res_rendering()
