# Docker Guide for LDView Training

This guide covers using the Docker container with LDView pre-installed for generating training data.

## What's Included

The Docker container includes:
- **LDView**: Command-line 3D renderer for LEGO parts
- **LDraw Library**: Complete parts library (~50MB)
- **Python Environment**: All required dependencies
- **Pre-configured**: LDRAWDIR environment variable set automatically

## Quick Start

### 1. Build the Container

```bash
docker-compose build
```

This will:
- Install LDView and dependencies
- Download the LDraw parts library
- Install Python packages
- Set up the rendering environment

**Note**: First build takes ~5-10 minutes due to downloading the LDraw library.

### 2. Prepare Your .dat Files

Place your LEGO .dat files in the input directory:

```bash
mkdir -p input/dat_files
cp /path/to/your/*.dat input/dat_files/
```

The directory structure should look like:
```
input/
└── dat_files/
    ├── 3001.dat
    ├── 3003.dat
    └── ...
```

### 3. Generate Training Data

Run the training data generation:

```bash
docker-compose run --rm training
```

This will:
- Find all .dat files in `input/dat_files/`
- Generate 20 samples per part (configurable)
- Save images to `data/training/`

### 4. Check the Output

Generated images will be in:
```
data/training/
├── 3001/
│   ├── 3001_0000.jpg
│   ├── 3001_0001.jpg
│   └── ...
├── 3003/
│   └── ...
```

## Configuration

### Environment Variables

Customize generation by setting environment variables:

```bash
# Generate 50 samples per part
docker-compose run --rm -e SAMPLES_PER_PART=50 training

# Use custom output directory
docker-compose run --rm -e OUTPUT_DIR=/app/data/custom training

# Limit to 10 parts (for testing)
docker-compose run --rm -e LIMIT=10 training

# Custom output size (512x512)
docker-compose run --rm -e OUTPUT_SIZE=512 training
```

### Available Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DAT_DIR` | `/app/input/dat_files` | Directory containing .dat files |
| `OUTPUT_DIR` | `/app/data/training` | Output directory for images |
| `SAMPLES_PER_PART` | `20` | Number of samples per part |
| `OUTPUT_SIZE` | `224` | Image size (width=height) |
| `LIMIT` | None | Limit number of parts to process |
| `BACKGROUND_IMAGE` | None | Path to background image |

### Using a Custom Background

To use a custom background image:

1. Place background in the data directory:
   ```bash
   cp conveyor.jpg data/backgrounds/
   ```

2. Run with background:
   ```bash
   docker-compose run --rm \
     -e BACKGROUND_IMAGE=/app/data/backgrounds/conveyor.jpg \
     training
   ```

### Edit docker-compose.yml

For persistent configuration, edit `docker-compose.yml`:

```yaml
training:
  environment:
    - SAMPLES_PER_PART=50        # Change from 20 to 50
    - OUTPUT_SIZE=512             # Change from 224 to 512
    - LIMIT=100                   # Add a limit
```

Then run:
```bash
docker-compose run --rm training
```

## Advanced Usage

### Custom DAT Directory

Mount a different directory with your .dat files:

```bash
docker-compose run --rm \
  -v /path/to/my/dats:/app/input/dat_files \
  training
```

### Interactive Shell

Access the container shell for debugging:

```bash
docker-compose run --rm training bash
```

Inside the container:
```bash
# Test LDView
ldview --version

# Check LDraw library
ls $LDRAWDIR

# Run Python directly
python src/test_ldview_renderer.py /app/input/dat_files/3001.dat

# Generate training data
python src/docker_generate_training.py
```

### Use generate_ldview_training_data.py Directly

You can also use the full-featured CLI script inside the container:

```bash
docker-compose run --rm training bash -c "
  python src/generate_ldview_training_data.py \
    --dat-dir /app/input/dat_files \
    --output-dir /app/data/training \
    --samples-per-part 30 \
    --background /app/data/backgrounds/conveyor.jpg
"
```

## Training the Model

After generating training data, train the model:

```bash
# Start training service with UI
docker-compose up vision

# Access training UI at http://localhost:8000
```

Or run training directly:

```bash
docker-compose run --rm vision python src/train.py
```

## Troubleshooting

### No .dat files found

**Error**: `No .dat files found!`

**Solution**: Ensure your .dat files are in `input/dat_files/`:
```bash
ls input/dat_files/
# Should show: 3001.dat, 3003.dat, etc.
```

### Permission issues

**Error**: `Permission denied` when writing files

**Solution**: Check directory permissions:
```bash
chmod -R 755 data/ output/ input/
```

### Container build fails

**Error**: `Failed to download LDraw library`

**Solution**: Check internet connection and retry:
```bash
docker-compose build --no-cache
```

### LDView rendering errors

**Error**: `Failed to render /app/input/dat_files/xxxx.dat`

**Solution**: Test the .dat file:
```bash
# Enter container
docker-compose run --rm training bash

# Test render
ldview /app/input/dat_files/xxxx.dat -SaveSnapshot=/tmp/test.png
```

### Slow rendering

**Tip**: Start with a small subset to test:
```bash
# Test with 5 parts, 3 samples each
docker-compose run --rm \
  -e LIMIT=5 \
  -e SAMPLES_PER_PART=3 \
  training
```

## Performance

### Build Time
- First build: ~5-10 minutes (downloads LDraw library)
- Subsequent builds: ~1-2 minutes (cached)

### Rendering Speed
- Simple parts: ~1-2 seconds per image
- Complex parts: ~3-5 seconds per image
- 100 parts × 20 samples: ~1-2 hours

### Resource Usage
- **Memory**: ~500 MB - 1 GB
- **Disk**:
  - Container image: ~2 GB
  - LDraw library: ~50 MB
  - Training data (100 parts × 20 samples): ~100-200 MB
- **CPU**: Single-threaded per render (LDView limitation)

## Tips

### Maximize Efficiency

1. **Use LIMIT for testing**: Test with a few parts first
   ```bash
   docker-compose run --rm -e LIMIT=5 training
   ```

2. **Adjust samples based on part variety**:
   - Many similar parts: fewer samples (10-20)
   - Diverse parts: more samples (30-50)

3. **Choose appropriate output size**:
   - Small models: 128×128 or 224×224
   - High-detail needs: 512×512 or larger

4. **Monitor progress**: The script shows a progress bar with part names

### Batch Processing

For large datasets, process in batches:

```bash
# Process first 100 parts
docker-compose run --rm -e LIMIT=100 -e OUTPUT_DIR=/app/data/batch1 training

# Process next 100 parts (manual file management needed)
# Move processed files, then run again
```

## Next Steps

After generating training data:

1. **Verify output**:
   ```bash
   ls data/training/
   ```

2. **Train the model**:
   ```bash
   docker-compose up vision
   # Visit http://localhost:8000
   ```

3. **Adjust parameters** if needed and regenerate

## Reference

- [LDView Documentation](https://tcobbs.github.io/ldview/)
- [LDraw File Format](https://www.ldraw.org/article/218.html)
- [Main LDView Renderer Guide](./LDVIEW_RENDERER.md)
- [Vision Module README](./README.md)
