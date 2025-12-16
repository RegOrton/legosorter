# LDView Docker Container - Testing Guide

## Overview

The Docker container has been updated to include:
- **LDView 4.6** (osmesa headless version) - for rendering LEGO models
- **LDraw Library** (complete.zip) - with 23,000+ LEGO part definitions
- **Python rendering script** - LDViewRenderer for batch generation

## Building the Container

The Docker build may take 10-15 minutes due to:
1. Large Ubuntu 24.04 base image download
2. LDView 4.6 deb package installation
3. Complete LDraw library download and extraction (200+ MB)
4. Python dependencies installation

```bash
cd vision
docker-compose build
```

## Testing LDView Installation

Once the build completes, verify LDView is working:

```bash
# Check LDView version
docker-compose run --rm vision ldview -v

# Test the Python test script
docker-compose run --rm vision python src/test_ldview_simple.py
```

## Generating Test Images

### Option 1: Quick Test with a Single Part

First, you need a `.dat` file. Either:
- Place your own `.dat` files in `vision/input/dat_files/`
- Or download a sample part from LDraw

```bash
# Create input directory
mkdir -p vision/input/dat_files

# Test rendering a sample part from the LDraw library
docker-compose run --rm vision python src/test_ldview_renderer.py /usr/share/ldraw/ldraw/parts/3001.dat --output test_brick.jpg --samples 5
```

### Option 2: Batch Generation with docker-compose service

Set environment variables and run the training data generator:

```bash
# Using docker-compose with environment variables
docker-compose run \
  -e DAT_DIR=/app/input/dat_files \
  -e OUTPUT_DIR=/app/output/ldview_training \
  -e SAMPLES_PER_PART=10 \
  -e OUTPUT_SIZE=224 \
  -e LIMIT=5 \
  --rm training
```

This will:
1. Find all `.dat` files in `/app/input/dat_files`
2. Generate 10 samples per part
3. Output to `/app/output/ldview_training`
4. Limit to first 5 parts (remove LIMIT env var for all parts)

## Expected Outputs

Generated images will be saved to:
```
vision/output/ldview_training/
  ├── 3001/              # Part number (part ID)
  │   ├── 3001_0000.jpg  # Sample 0
  │   ├── 3001_0001.jpg  # Sample 1
  │   └── ...
  ├── 3002/
  │   └── ...
  └── ...
```

## Troubleshooting

### "LDView not found" Error

If you get `LDView not found` errors:

```bash
# Verify LDView is installed in the container
docker-compose run --rm vision which ldview
docker-compose run --rm vision ldview -h
```

### "LDraw directory not found" Error

The LDRAWDIR should be set to `/usr/share/ldraw/ldraw` inside the container:

```bash
# Check LDRAWDIR
docker-compose run --rm vision printenv LDRAWDIR

# List available parts
docker-compose run --rm vision ls /usr/share/ldraw/ldraw/parts | head -20
```

### Rendering Hangs or Takes Too Long

LDView rendering is single-threaded. Rendering large parts or high-quality samples can take time:
- Start with `--samples 1` or `SAMPLES_PER_PART=1`
- Reduce `OUTPUT_SIZE` (default: 224)
- Increase timeout if needed

### Out of Disk Space

The complete LDraw library + sample renders can use significant space. Check:

```bash
docker-compose run --rm vision du -sh /usr/share/ldraw
docker-compose run --rm vision du -sh /app/output
```

## Integration with Training

Once images are generated, update your training pipeline to use them:

```python
from pathlib import Path
from torchvision import transforms
from PIL import Image

# Point to generated images
ldview_output = Path("/app/output/ldview_training")

# Load images for training
for part_dir in ldview_output.iterdir():
    for img_path in part_dir.glob("*.jpg"):
        img = Image.open(img_path)
        # Use in your training pipeline
```

## Performance Tips

1. **Batch Generation**: Use `docker-compose run` with multiple `SAMPLES_PER_PART` values
2. **Parallel Processing**: You can run multiple container instances with different part ranges:
   ```bash
   # Terminal 1: First batch
   docker-compose run -e LIMIT=10 training

   # Terminal 2: Second batch
   docker-compose run -e LIMIT=20 training
   ```

3. **Monitor Progress**: Generated files appear in real-time in `vision/output/ldview_training`

## Next Steps

After generating test images:
1. Verify image quality in `vision/output/ldview_training/`
2. Update the training dataset loader to use LDView images
3. Configure training parameters (epochs, batch size, etc.)
4. Start training via the API or dashboard

For more details, see [CLAUDE.md](./CLAUDE.md) section on "Training Data Sources".
