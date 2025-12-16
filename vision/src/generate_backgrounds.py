"""
Generate synthetic background images for training diversity.
Helps prevent overfitting to a single conveyor belt texture.
"""
import cv2
import numpy as np
from pathlib import Path


def generate_solid_color(size, color, noise_level=10):
    """Generate solid color background with optional noise."""
    img = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def generate_gradient(size, color1, color2, direction='horizontal'):
    """Generate gradient background."""
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)

    if direction == 'horizontal':
        for i in range(size[0]):
            ratio = i / size[0]
            color = [int(color1[c] * (1 - ratio) + color2[c] * ratio) for c in range(3)]
            img[:, i] = color
    else:  # vertical
        for i in range(size[1]):
            ratio = i / size[1]
            color = [int(color1[c] * (1 - ratio) + color2[c] * ratio) for c in range(3)]
            img[i, :] = color

    return img


def generate_noise_texture(size, base_color, noise_level=30):
    """Generate noisy texture background."""
    img = generate_solid_color(size, base_color, noise_level=noise_level)
    # Add some blur for texture
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


def generate_grid_pattern(size, color1, color2, cell_size=20):
    """Generate checkerboard/grid pattern."""
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)

    for y in range(0, size[1], cell_size):
        for x in range(0, size[0], cell_size):
            color = color1 if ((x // cell_size) + (y // cell_size)) % 2 == 0 else color2
            img[y:y+cell_size, x:x+cell_size] = color

    return img


def main():
    output_dir = Path(__file__).resolve().parent.parent / "data" / "backgrounds"
    output_dir.mkdir(parents=True, exist_ok=True)

    size = (512, 512)  # Large enough to crop from

    backgrounds = []

    # Solid colors (common work surfaces)
    solids = [
        ("gray_dark", (60, 60, 60)),
        ("gray_medium", (128, 128, 128)),
        ("gray_light", (192, 192, 192)),
        ("white", (240, 240, 240)),
        ("black", (30, 30, 30)),
        ("blue_dark", (80, 60, 40)),  # Dark blue in BGR
        ("brown", (60, 80, 100)),     # Brown table
    ]

    for name, color in solids:
        img = generate_solid_color(size, color, noise_level=15)
        backgrounds.append((f"solid_{name}.jpg", img))

    # Gradients
    gradients = [
        ("gradient_gray_h", (80, 80, 80), (180, 180, 180), 'horizontal'),
        ("gradient_gray_v", (100, 100, 100), (160, 160, 160), 'vertical'),
        ("gradient_warm", (80, 90, 100), (120, 130, 140), 'horizontal'),
    ]

    for name, c1, c2, direction in gradients:
        img = generate_gradient(size, c1, c2, direction)
        # Add noise
        noise = np.random.normal(0, 8, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        backgrounds.append((f"{name}.jpg", img))

    # Textured surfaces
    textures = [
        ("texture_gray", (120, 120, 120), 25),
        ("texture_dark", (70, 70, 70), 20),
        ("texture_light", (200, 200, 200), 15),
    ]

    for name, color, noise in textures:
        img = generate_noise_texture(size, color, noise)
        backgrounds.append((f"{name}.jpg", img))

    # Grid patterns (for calibration-style backgrounds)
    grids = [
        ("grid_gray", (100, 100, 100), (140, 140, 140), 30),
    ]

    for name, c1, c2, cell in grids:
        img = generate_grid_pattern(size, c1, c2, cell)
        backgrounds.append((f"{name}.jpg", img))

    # Save all backgrounds
    for filename, img in backgrounds:
        path = output_dir / filename
        cv2.imwrite(str(path), img)
        print(f"Generated: {path}")

    print(f"\nGenerated {len(backgrounds)} background images in {output_dir}")
    print("These will be randomly selected during training for improved generalization.")


if __name__ == "__main__":
    main()
