# LDView Renderer for Training Data Generation

The LDView renderer generates realistic 3D training images from LEGO .dat files (LDraw format) using the LDView command-line tool.

## Features

- **3D Rendering**: Renders actual LEGO parts in 3D with proper geometry and shading
- **Camera Variation**: Random camera angles (latitude, longitude, distance) for diverse training data
- **Background Compositing**: Composite renders onto custom backgrounds or solid colors
- **Augmentations**: Random blur, brightness, and noise for realistic training conditions
- **Batch Processing**: Efficient batch generation with progress tracking
- **Flexible Output**: Configurable image sizes and multiple samples per part

## Installation

### 1. Install LDView

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install ldview
```

**macOS (using Homebrew):**
```bash
brew install ldview
```

**Windows:**
Download from: https://tcobbs.github.io/ldview/

Add LDView to your PATH or specify the full path when using the renderer.

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install LDraw Parts Library (Optional)

LDView may auto-detect the LDraw library, but you can manually install it:

1. Download from: https://www.ldraw.org/parts/latest-parts.html
2. Extract to a directory (e.g., `/usr/share/ldraw` or `C:\LDraw`)
3. Specify the path with `--ldraw-dir` when using the renderer

## Usage

### Basic Usage

Generate training data from a directory of .dat files:

```bash
python src/generate_ldview_training_data.py \
    --dat-dir /path/to/dat/files \
    --output-dir ./training_data \
    --samples-per-part 20
```

### With Custom Background

Use a custom background image for compositing:

```bash
python src/generate_ldview_training_data.py \
    --dat-dir /path/to/dat/files \
    --output-dir ./training_data \
    --background ./data/backgrounds/conveyor.jpg \
    --samples-per-part 30
```

### Custom LDView Path

If LDView is not in PATH:

```bash
python src/generate_ldview_training_data.py \
    --dat-dir /path/to/dat/files \
    --output-dir ./training_data \
    --ldview-path /usr/local/bin/ldview \
    --ldraw-dir /usr/share/ldraw
```

### All Options

```bash
python src/generate_ldview_training_data.py \
    --dat-dir /path/to/dat/files \
    --output-dir ./training_data \
    --ldview-path ldview \
    --ldraw-dir /usr/share/ldraw \
    --samples-per-part 20 \
    --output-size 224 224 \
    --background ./data/backgrounds/conveyor.jpg \
    --limit 10 \
    --recursive
```

## Programmatic Usage

You can also use the renderer directly in Python:

```python
from ldview_renderer import LDViewRenderer, save_samples

# Initialize renderer
renderer = LDViewRenderer(
    ldview_path="ldview",
    output_size=(224, 224),
    background_path="./data/backgrounds/conveyor.jpg"
)

# Generate single sample
image = renderer.generate_sample(
    dat_file_path="/path/to/part.dat",
    latitude=15.0,      # Optional: camera latitude (-30 to 30)
    longitude=45.0,     # Optional: camera longitude (0 to 360)
    distance=1.2,       # Optional: camera distance multiplier
)

# Generate batch
dat_files = ["/path/to/part1.dat", "/path/to/part2.dat"]
samples = renderer.batch_generate(
    dat_files,
    samples_per_part=10,
    progress_callback=lambda c, t, p: print(f"{c}/{t}: {p}")
)

# Save to disk
save_samples(samples, "./training_data")
```

## Render Parameters

### Camera Angles

- **Latitude**: Vertical angle (-30° to 30°)
  - Negative values: view from below
  - Positive values: view from above
  - 0: side view

- **Longitude**: Horizontal rotation (0° to 360°)
  - 0°: front view
  - 90°: right side view
  - 180°: back view
  - 270°: left side view

- **Distance**: Camera distance multiplier (0.8 to 1.5)
  - Lower values: closer/larger part
  - Higher values: farther/smaller part

### Lighting

The renderer automatically varies lighting:
- **default**: Standard LDView lighting
- **bright**: Increased light intensity
- **dim**: Reduced light intensity

### Augmentations

Post-processing augmentations applied randomly:
- **Gaussian Blur** (30% chance): Simulates motion blur or camera defocus
- **Brightness Adjustment** (50% chance): ±20% brightness variation
- **Noise** (20% chance): Gaussian noise to simulate sensor noise

## Output Structure

Generated images are organized by part ID:

```
training_data/
├── 3001/              # Part ID (from .dat filename)
│   ├── 3001_0000.jpg
│   ├── 3001_0001.jpg
│   └── ...
├── 3003/
│   ├── 3003_0000.jpg
│   └── ...
└── ...
```

This structure is compatible with the existing `LegoTripletDataset` class for training.

## Integration with Training Pipeline

The generated images can be used directly with the existing training pipeline:

```bash
# Generate training data
python src/generate_ldview_training_data.py \
    --dat-dir /path/to/dat/files \
    --output-dir ./data/ldview_training \
    --samples-per-part 50

# Train model
python src/train.py --data-dir ./data/ldview_training
```

## Troubleshooting

### LDView not found

**Error:** `LDView not found at 'ldview'`

**Solution:** Install LDView or specify the full path:
```bash
--ldview-path /usr/local/bin/ldview
```

### Missing LDraw library

**Error:** `LDView warning: Unable to find LDraw directory`

**Solution:** Install LDraw library and specify path:
```bash
--ldraw-dir /usr/share/ldraw
```

### Rendering fails for some parts

**Error:** `Failed to render /path/to/part.dat`

**Solution:**
- Ensure .dat file is valid LDraw format
- Check LDView can render the file manually: `ldview /path/to/part.dat`
- Some complex parts may require additional subparts in the LDraw library

### Slow rendering

**Tip:** LDView rendering can be slow for complex parts. Consider:
- Using `--limit` to test with fewer parts first
- Running on multiple cores (batch processing is single-threaded currently)
- Using simpler background images or solid colors

## Performance

- **Rendering Speed**: ~1-5 seconds per image (depends on part complexity)
- **Memory Usage**: ~100-500 MB (depends on batch size and image size)
- **Disk Space**: ~20-50 KB per JPG image at 224x224

For 100 parts with 50 samples each:
- Total images: 5,000
- Estimated time: 1-7 hours
- Disk space: ~100-250 MB

## Advanced: Custom Camera Presets

You can modify `ldview_renderer.py` to add custom camera presets:

```python
CAMERA_PRESETS = {
    "top_down": {"latitude": 30, "longitude": 0, "distance": 1.0},
    "side_view": {"latitude": 0, "longitude": 90, "distance": 1.0},
    "angled": {"latitude": 15, "longitude": 45, "distance": 1.2},
}

# Use in generate_sample:
img = renderer.generate_sample(dat_path, **CAMERA_PRESETS["angled"])
```

## References

- **LDView**: https://tcobbs.github.io/ldview/
- **LDraw**: https://www.ldraw.org/
- **LDraw File Format**: https://www.ldraw.org/article/218.html
