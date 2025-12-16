"""
Detailed LDView rendering test with multiple configurations.
"""

import subprocess
import cv2
import tempfile
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_raw_ldview():
    """Test raw LDView rendering with various parameters."""

    base_dir = Path(__file__).parent.parent
    dat_dir = base_dir / "input" / "dat_files"
    output_dir = base_dir / "output" / "debug" / "ldview_raw_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test with first .dat file
    dat_files = list(dat_dir.glob("*.dat"))
    if not dat_files:
        logger.error("No .dat files found")
        return

    test_file = dat_files[0]
    logger.info(f"Testing with: {test_file}")

    # Test different camera configurations
    test_configs = [
        {
            "name": "default",
            "args": []
        },
        {
            "name": "close_up",
            "args": ["-cg0,0", "-ca0.5"]
        },
        {
            "name": "angled",
            "args": ["-cg30,45", "-ca0.1"]
        },
        {
            "name": "no_transparency",
            "args": ["-SaveAlpha=0"]
        },
        {
            "name": "with_edge_lines",
            "args": ["-EdgeOnly=0", "-ConditionalLines=1"]
        }
    ]

    for config in test_configs:
        output_path = output_dir / f"{test_file.stem}_{config['name']}.png"

        import os
        ldraw_dir = os.environ.get('LDRAWDIR', '/usr/share/ldraw/ldraw')

        cmd = [
            "ldview",
            str(test_file),
            f"-SaveSnapshot={output_path}",
            "-SaveWidth=224",
            "-SaveHeight=224",
            "-SaveAlpha=1",
            "-SaveActualSize=0",
            "-AutoCrop=0",
            f"-LDrawDir={ldraw_dir}"
        ]

        cmd.extend(config['args'])

        logger.info(f"\nTesting config: {config['name']}")
        logger.info(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30,
                cwd=str(test_file.parent)
            )

            if result.returncode != 0:
                logger.error(f"LDView error: {result.stderr.decode('utf-8', errors='ignore')}")
            else:
                logger.info(f"LDView output: {result.stdout.decode('utf-8', errors='ignore')}")

            if output_path.exists():
                # Load and check image
                img = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    logger.info(f"  ✓ Image saved: {output_path}")
                    logger.info(f"    Shape: {img.shape}")
                    logger.info(f"    Min/Max pixel values: {img.min()}/{img.max()}")

                    # Check if image has content (not all background)
                    if len(img.shape) == 3 and img.shape[2] == 4:
                        alpha = img[:, :, 3]
                        non_transparent_pixels = (alpha > 0).sum()
                        total_pixels = alpha.size
                        coverage = (non_transparent_pixels / total_pixels) * 100
                        logger.info(f"    Coverage: {coverage:.2f}% ({non_transparent_pixels}/{total_pixels} pixels)")
                else:
                    logger.error(f"  ✗ Failed to load image: {output_path}")
            else:
                logger.error(f"  ✗ Output file not created: {output_path}")

        except Exception as e:
            logger.error(f"Error: {e}")

    logger.info(f"\n✅ Test complete! Check images in: {output_dir}")


if __name__ == "__main__":
    test_raw_ldview()
