"""
Test script to verify LDView rendering is working correctly.
Generates sample renders from available .dat files.
"""

import sys
from pathlib import Path
import cv2
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ldview_renderer import LDViewRenderer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_ldview_rendering():
    """Test LDView rendering with available .dat files."""

    # Setup paths
    base_dir = Path(__file__).parent.parent
    dat_dir = base_dir / "input" / "dat_files"
    output_dir = base_dir / "output" / "debug" / "ldview_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Looking for .dat files in: {dat_dir}")

    # Find .dat files
    dat_files = list(dat_dir.glob("*.dat"))

    if not dat_files:
        logger.error(f"No .dat files found in {dat_dir}")
        return

    logger.info(f"Found {len(dat_files)} .dat files")

    # Initialize renderer
    try:
        renderer = LDViewRenderer(
            ldview_path="ldview",
            output_size=(224, 224)
        )
        logger.info("LDView renderer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize LDView renderer: {e}")
        return

    # Test rendering first 5 .dat files with multiple viewpoints
    test_files = dat_files[:5]

    for dat_file in test_files:
        part_name = dat_file.stem
        logger.info(f"\nTesting part: {part_name}")

        # Create output directory for this part
        part_output_dir = output_dir / part_name
        part_output_dir.mkdir(exist_ok=True)

        # Generate 3 renders with different viewpoints
        viewpoints = [
            {"latitude": 0, "longitude": 0, "name": "front"},
            {"latitude": 30, "longitude": 45, "name": "angle_1"},
            {"latitude": -20, "longitude": 135, "name": "angle_2"},
        ]

        for i, viewpoint in enumerate(viewpoints):
            try:
                logger.info(f"  Rendering viewpoint {i+1}/3: {viewpoint['name']}")

                # Render with specific viewpoint
                img = renderer.generate_sample(
                    str(dat_file),
                    latitude=viewpoint['latitude'],
                    longitude=viewpoint['longitude'],
                    apply_augmentations=False  # No augmentations for testing
                )

                # Save image
                output_path = part_output_dir / f"{part_name}_{viewpoint['name']}.jpg"
                cv2.imwrite(str(output_path), img)
                logger.info(f"    Saved: {output_path}")

            except Exception as e:
                logger.error(f"    Failed to render {viewpoint['name']}: {e}")
                continue

        # Generate one sample with augmentations
        try:
            logger.info(f"  Rendering with augmentations...")
            img_aug = renderer.generate_sample(
                str(dat_file),
                apply_augmentations=True
            )
            output_path = part_output_dir / f"{part_name}_augmented.jpg"
            cv2.imwrite(str(output_path), img_aug)
            logger.info(f"    Saved: {output_path}")
        except Exception as e:
            logger.error(f"    Failed to render augmented: {e}")

    logger.info(f"\n✅ Test complete! Check images in: {output_dir}")
    logger.info(f"Generated test renders for {len(test_files)} parts")


if __name__ == "__main__":
    test_ldview_rendering()
