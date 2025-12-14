#!/usr/bin/env python3
"""
Docker-optimized script for generating training data with LDView.

This script is designed to run inside the Docker container where LDView
and the LDraw library are pre-installed.

Usage (inside container):
    python src/docker_generate_training.py

Usage (from host with docker-compose):
    docker-compose run --rm training
"""

import os
import sys
from pathlib import Path
import logging
from ldview_renderer import LDViewRenderer, save_samples
from tqdm import tqdm

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
    # Configuration (can be overridden by environment variables)
    dat_dir = os.environ.get('DAT_DIR', '/app/input/dat_files')
    output_dir = os.environ.get('OUTPUT_DIR', '/app/data/training')
    samples_per_part = int(os.environ.get('SAMPLES_PER_PART', '20'))
    output_size = int(os.environ.get('OUTPUT_SIZE', '224'))
    limit = os.environ.get('LIMIT')
    background = os.environ.get('BACKGROUND_IMAGE')

    if limit:
        limit = int(limit)

    logger.info("=" * 60)
    logger.info("LDView Training Data Generator (Docker)")
    logger.info("=" * 60)
    logger.info(f"DAT directory: {dat_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Samples per part: {samples_per_part}")
    logger.info(f"Output size: {output_size}x{output_size}")
    logger.info(f"Background: {background or 'Random solid colors'}")
    logger.info(f"Limit: {limit or 'No limit'}")
    logger.info(f"LDraw directory: {os.environ.get('LDRAWDIR', 'Auto-detect')}")
    logger.info("=" * 60)

    # Verify DAT directory exists
    if not Path(dat_dir).exists():
        logger.error(f"DAT directory not found: {dat_dir}")
        logger.error("Please mount your .dat files to /app/input/dat_files")
        logger.error("Example: docker-compose run -v /path/to/dats:/app/input/dat_files training")
        sys.exit(1)

    # Find .dat files
    logger.info("Searching for .dat files...")
    dat_files = find_dat_files(dat_dir, recursive=True)

    if not dat_files:
        logger.error("No .dat files found!")
        logger.error(f"Searched in: {dat_dir}")
        sys.exit(1)

    # Apply limit if specified
    if limit:
        dat_files = dat_files[:limit]
        logger.info(f"Limited to {len(dat_files)} files")

    # Initialize renderer
    logger.info("Initializing LDView renderer...")
    try:
        renderer = LDViewRenderer(
            ldview_path="ldview",
            ldraw_dir=None,  # Will auto-detect from LDRAWDIR env var
            output_size=(output_size, output_size),
            background_path=background
        )
        logger.info("✓ Renderer initialized successfully")
    except RuntimeError as e:
        logger.error(f"Failed to initialize renderer: {e}")
        sys.exit(1)

    # Generate samples with progress bar
    logger.info(f"Generating {samples_per_part} samples per part...")
    total_samples = len(dat_files) * samples_per_part

    with tqdm(total=total_samples, desc="Rendering") as pbar:
        def progress_callback(current, total, part_id):
            pbar.set_postfix({"part": part_id[:20]})  # Truncate long part names
            pbar.update(1)

        samples = renderer.batch_generate(
            dat_files,
            samples_per_part=samples_per_part,
            progress_callback=progress_callback
        )

    # Save samples
    logger.info(f"Saving {len(samples)} samples to {output_dir}...")
    save_samples(samples, output_dir)

    logger.info("=" * 60)
    logger.info("✓ Complete!")
    logger.info(f"Generated {len(samples)} training images")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
