#!/usr/bin/env python3
"""
Test script for LDView renderer.

This script tests the LDView renderer with a single .dat file
and displays the results.

Usage:
    python test_ldview_renderer.py /path/to/part.dat
"""

import sys
import cv2
import argparse
from pathlib import Path
from ldview_renderer import LDViewRenderer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test LDView renderer")
    parser.add_argument("dat_file", help="Path to .dat file to render")
    parser.add_argument(
        "--output", "-o",
        default="test_render.jpg",
        help="Output file path (default: test_render.jpg)"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=5,
        help="Number of sample renders (default: 5)"
    )
    parser.add_argument(
        "--ldview-path",
        default="ldview",
        help="Path to LDView executable"
    )
    parser.add_argument(
        "--background",
        default=None,
        help="Background image path (optional)"
    )

    args = parser.parse_args()

    dat_path = Path(args.dat_file)
    if not dat_path.exists():
        logger.error(f"File not found: {args.dat_file}")
        sys.exit(1)

    logger.info(f"Testing LDView renderer with: {dat_path.name}")

    try:
        # Initialize renderer
        renderer = LDViewRenderer(
            ldview_path=args.ldview_path,
            output_size=(224, 224),
            background_path=args.background
        )
        logger.info("✓ Renderer initialized successfully")

        # Test basic render
        logger.info("Rendering with default settings...")
        img = renderer.render(str(dat_path))
        logger.info(f"✓ Basic render successful: {img.shape}")

        # Save basic render
        basic_output = Path(args.output).stem + "_basic.jpg"
        cv2.imwrite(basic_output, img)
        logger.info(f"✓ Saved basic render to: {basic_output}")

        # Test augmented samples
        logger.info(f"Generating {args.samples} augmented samples...")
        for i in range(args.samples):
            img = renderer.generate_sample(str(dat_path))
            output = Path(args.output).stem + f"_sample_{i}.jpg"
            cv2.imwrite(output, img)
            logger.info(f"  Sample {i+1}/{args.samples} -> {output}")

        logger.info("✓ All tests passed!")
        logger.info("\nGenerated files:")
        logger.info(f"  - {basic_output} (basic render)")
        for i in range(args.samples):
            logger.info(f"  - {Path(args.output).stem}_sample_{i}.jpg")

    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
