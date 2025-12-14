#!/usr/bin/env python3
"""
Generate training data using LDView renderer.

This script renders LEGO .dat files (LDraw format) using LDView to create
realistic 3D training images with various camera angles and augmentations.

Usage:
    python generate_ldview_training_data.py --dat-dir /path/to/dat/files --output-dir ./training_data

Requirements:
    - LDView must be installed and in PATH (or specify with --ldview-path)
    - .dat files in LDraw format
    - Optional: background images for compositing
"""

import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from ldview_renderer import LDViewRenderer, save_samples

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_dat_files(dat_dir: str, recursive: bool = True) -> list:
    """Find all .dat files in a directory."""
    dat_path = Path(dat_dir)

    if not dat_path.exists():
        raise FileNotFoundError(f"Directory not found: {dat_dir}")

    if recursive:
        dat_files = list(dat_path.rglob("*.dat"))
    else:
        dat_files = list(dat_path.glob("*.dat"))

    logger.info(f"Found {len(dat_files)} .dat files in {dat_dir}")
    return [str(f) for f in dat_files]


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data from LEGO .dat files using LDView"
    )

    # Input/Output
    parser.add_argument(
        "--dat-dir",
        type=str,
        required=True,
        help="Directory containing .dat files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./training_data",
        help="Directory to save generated images (default: ./training_data)"
    )

    # LDView settings
    parser.add_argument(
        "--ldview-path",
        type=str,
        default="ldview",
        help="Path to LDView executable (default: ldview)"
    )
    parser.add_argument(
        "--ldraw-dir",
        type=str,
        default=None,
        help="Path to LDraw parts library (optional, LDView may auto-detect)"
    )

    # Generation settings
    parser.add_argument(
        "--samples-per-part",
        type=int,
        default=20,
        help="Number of samples to generate per part (default: 20)"
    )
    parser.add_argument(
        "--output-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Output image size as width height (default: 224 224)"
    )
    parser.add_argument(
        "--background",
        type=str,
        default=None,
        help="Path to background image for compositing (optional)"
    )

    # Filtering
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of .dat files to process (for testing)"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Search for .dat files recursively (default: True)"
    )

    args = parser.parse_args()

    # Find .dat files
    logger.info("Searching for .dat files...")
    dat_files = find_dat_files(args.dat_dir, recursive=args.recursive)

    if not dat_files:
        logger.error("No .dat files found!")
        return

    # Apply limit if specified
    if args.limit:
        dat_files = dat_files[:args.limit]
        logger.info(f"Limited to {len(dat_files)} files")

    # Initialize renderer
    logger.info("Initializing LDView renderer...")
    try:
        renderer = LDViewRenderer(
            ldview_path=args.ldview_path,
            ldraw_dir=args.ldraw_dir,
            output_size=tuple(args.output_size),
            background_path=args.background
        )
    except RuntimeError as e:
        logger.error(f"Failed to initialize renderer: {e}")
        return

    # Generate samples with progress bar
    logger.info(f"Generating {args.samples_per_part} samples per part...")
    total_samples = len(dat_files) * args.samples_per_part

    with tqdm(total=total_samples, desc="Rendering") as pbar:
        def progress_callback(current, total, part_id):
            pbar.set_postfix({"part": part_id})
            pbar.update(1)

        samples = renderer.batch_generate(
            dat_files,
            samples_per_part=args.samples_per_part,
            progress_callback=progress_callback
        )

    # Save samples
    logger.info(f"Saving {len(samples)} samples to {args.output_dir}...")
    save_samples(samples, args.output_dir)

    logger.info("✓ Complete!")
    logger.info(f"Generated {len(samples)} training images")
    logger.info(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
