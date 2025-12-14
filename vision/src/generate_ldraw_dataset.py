"""
Generate multi-view training dataset from LDraw 3D models.
This provides true 3D viewpoint variation for training.
"""
import argparse
import logging
from pathlib import Path
from ldraw_renderer import LDrawRenderer, generate_training_images, POPULAR_PARTS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Common LEGO colors for training diversity
TRAINING_COLORS = [
    4,   # Red
    1,   # Blue
    14,  # Yellow
    2,   # Green
    0,   # Black
    15,  # White
    25,  # Orange
    70,  # Reddish Brown
    71,  # Light Bluish Grey
    72,  # Dark Bluish Grey
    19,  # Tan
    28,  # Dark Tan
    27,  # Lime
    26,  # Magenta
    22,  # Purple
]


def main():
    parser = argparse.ArgumentParser(description='Generate LDraw training dataset')
    parser.add_argument('--num-parts', type=int, default=100,
                        help='Number of popular parts to render')
    parser.add_argument('--views-per-color', type=int, default=16,
                        help='Number of views per part/color combination')
    parser.add_argument('--num-colors', type=int, default=10,
                        help='Number of colors per part')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for images')

    args = parser.parse_args()

    # Paths
    base_dir = Path(__file__).resolve().parent.parent
    ldraw_path = base_dir / "data" / "ldraw"
    output_path = Path(args.output_dir) if args.output_dir else base_dir / "data" / "ldraw_renders"

    if not ldraw_path.exists():
        logger.error(f"LDraw library not found at {ldraw_path}")
        logger.error("Download from https://library.ldraw.org/library/updates/complete.zip")
        return

    # Select parts and colors
    parts = POPULAR_PARTS[:args.num_parts]
    colors = TRAINING_COLORS[:args.num_colors]

    logger.info(f"Generating dataset:")
    logger.info(f"  Parts: {len(parts)}")
    logger.info(f"  Colors: {len(colors)}")
    logger.info(f"  Views per color: {args.views_per_color}")
    logger.info(f"  Total images: ~{len(parts) * len(colors) * args.views_per_color}")
    logger.info(f"  Output: {output_path}")

    # Generate
    generate_training_images(
        ldraw_path=ldraw_path,
        output_path=output_path,
        parts=parts,
        colors=colors,
        views_per_part=args.views_per_color
    )

    # Create metadata file
    metadata_path = output_path / "metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write("# LDraw Training Dataset Metadata\n")
        f.write(f"# Parts: {len(parts)}\n")
        f.write(f"# Colors: {len(colors)}\n")
        f.write(f"# Views per color: {args.views_per_color}\n")
        f.write("\n# Parts included:\n")
        for p in parts:
            f.write(f"{p}\n")
        f.write("\n# Color codes:\n")
        for c in colors:
            f.write(f"{c}\n")

    logger.info(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
