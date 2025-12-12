"""
Test the enhanced synthesizer to verify all augmentations work correctly.
Generates sample images showing the variety of augmentations.
"""
import cv2
import numpy as np
from pathlib import Path
from synthesizer import LegoSynthesizer
import random


def test_synthesis():
    """Test basic synthesis functionality with new enhancements."""
    base_dir = Path(__file__).resolve().parent.parent  # vision/
    bg_dir = base_dir / "data" / "backgrounds"
    images_dir = base_dir / "data" / "rebrickable" / "images" / "elements"
    output_dir = base_dir / "output" / "debug"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using backgrounds directory: {bg_dir}")
    if not bg_dir.exists():
        print("Backgrounds directory not found!")
        return

    # List available backgrounds
    bg_files = list(bg_dir.glob("*.jpg")) + list(bg_dir.glob("*.png"))
    print(f"Found {len(bg_files)} background images:")
    for bg in bg_files[:5]:
        print(f"  - {bg.name}")
    if len(bg_files) > 5:
        print(f"  ... and {len(bg_files) - 5} more")

    # Initialize synthesizer with backgrounds directory
    synth = LegoSynthesizer(bg_dir, output_size=(224, 224))

    # Get list of element images
    image_files = list(images_dir.glob("*.jpg"))
    if not image_files:
        print("No element images found! Run ingest_rebrickable.py first.")
        return

    print(f"\nFound {len(image_files)} element images")

    # Generate samples showing variation
    samples_per_part = 6
    parts_to_test = min(3, len(image_files))

    print(f"\nGenerating {parts_to_test * samples_per_part} samples to show augmentation variety...")

    for p in range(parts_to_test):
        part_img = image_files[p]
        part_name = part_img.stem
        print(f"\nPart {p+1}: {part_name}")

        samples = []
        for i in range(samples_per_part):
            result = synth.generate_sample(part_img)
            if result is not None:
                samples.append(result)
                # Save individual sample
                out_path = output_dir / f"sample_{part_name}_{i}.jpg"
                cv2.imwrite(str(out_path), result)

        # Create grid showing all variations of this part
        if samples:
            rows = 2
            cols = 3
            grid = np.zeros((224 * rows, 224 * cols, 3), dtype=np.uint8)

            for idx, sample in enumerate(samples[:rows * cols]):
                r = idx // cols
                c = idx % cols
                grid[r*224:(r+1)*224, c*224:(c+1)*224] = sample

            grid_path = output_dir / f"grid_{part_name}.jpg"
            cv2.imwrite(str(grid_path), grid)
            print(f"  Saved variation grid: {grid_path}")

    # Generate mixed samples (random parts)
    print("\n--- Generating mixed random samples ---")
    for i in range(10):
        part_img = random.choice(image_files)
        result = synth.generate_sample(part_img)

        if result is not None:
            out_path = output_dir / f"synth_{i}_{part_img.stem}.jpg"
            cv2.imwrite(str(out_path), result)
            print(f"Saved: {out_path.name}")
        else:
            print(f"Failed to generate sample for {part_img.name}")

    print(f"\n{'='*50}")
    print(f"All samples saved to: {output_dir}")
    print(f"{'='*50}")
    print("\nCheck the images to verify these augmentations:")
    print("  [x] Different backgrounds (solid, gradient, textured)")
    print("  [x] Perspective transforms (slight 3D viewpoint changes)")
    print("  [x] Drop shadows (directional lighting simulation)")
    print("  [x] Camera effects (noise, brightness, blur, color temp)")
    print("  [x] Rotation and scale variations")


if __name__ == "__main__":
    test_synthesis()
