# Vision Module

Computer vision module for the LEGO Sorter system. This module handles part identification using deep learning with metric learning (triplet loss).

## Features

- **Metric Learning**: Siamese network architecture with triplet loss for part similarity
- **Multiple Training Data Sources**:
  - **2D Images**: Composite Rebrickable images onto backgrounds (basic, fast)
  - **3D Rendering**: Generate realistic training data from .dat files using LDView (recommended)
- **REST API**: FastAPI server for real-time inference
- **Docker Support**: Containerized training and deployment

## Training Data Generation

### Option 1: LDView Renderer (Recommended)

Generate realistic 3D training data from LEGO .dat files:

```bash
python src/generate_ldview_training_data.py \
    --dat-dir /path/to/dat/files \
    --output-dir ./data/training \
    --samples-per-part 50 \
    --background ./data/backgrounds/conveyor.jpg
```

**Advantages:**
- Realistic 3D geometry and lighting
- Accurate part representation
- Better generalization to real-world conditions

**Requirements:**
- LDView installed ([Installation Guide](./LDVIEW_RENDERER.md))
- .dat files in LDraw format

See [LDVIEW_RENDERER.md](./LDVIEW_RENDERER.md) for detailed documentation.

### Option 2: 2D Image Synthesis

Composite 2D part images onto backgrounds:

```bash
python src/generate_mock.py
```

**Advantages:**
- Fast generation
- No external dependencies
- Good for quick prototyping

**Requirements:**
- Part images (e.g., from Rebrickable)
- Background images

## Training

### Local Training

```bash
# Install dependencies
pip install -r requirements.txt

# Generate training data (LDView)
python src/generate_ldview_training_data.py \
    --dat-dir /path/to/dat/files \
    --output-dir ./data/training

# Train model
python src/train.py --data-dir ./data/training
```

### Docker Training

```bash
# Build image
docker-compose build

# Start training
docker-compose up
```

The training UI will be available at http://localhost:8000

## API Server

Start the vision API server:

```bash
python src/main_api.py
```

Endpoints:
- `POST /identify`: Identify a LEGO part from an image
- `GET /metrics`: View training metrics
- `GET /health`: Health check

## Project Structure

```
vision/
├── src/
│   ├── ldview_renderer.py           # LDView 3D renderer
│   ├── generate_ldview_training_data.py  # Generate training data with LDView
│   ├── test_ldview_renderer.py      # Test LDView setup
│   ├── synthesizer.py               # 2D image synthesizer (legacy)
│   ├── dataset.py                   # PyTorch dataset for triplet learning
│   ├── model.py                     # Siamese network architecture
│   ├── train.py                     # Training loop
│   ├── main_api.py                  # FastAPI server
│   └── ...
├── data/
│   ├── backgrounds/                 # Background images
│   └── training/                    # Generated training data
├── output/
│   └── models/                      # Trained model checkpoints
├── LDVIEW_RENDERER.md              # LDView renderer documentation
├── requirements.txt
└── README.md                        # This file
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install LDView (recommended):**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install ldview

   # macOS
   brew install ldview
   ```

3. **Test LDView setup:**
   ```bash
   python src/test_ldview_renderer.py /path/to/part.dat
   ```

4. **Generate training data:**
   ```bash
   python src/generate_ldview_training_data.py \
       --dat-dir /path/to/dat/files \
       --output-dir ./data/training \
       --samples-per-part 50
   ```

5. **Train model:**
   ```bash
   python src/train.py --data-dir ./data/training
   ```

## Documentation

- [LDView Renderer Guide](./LDVIEW_RENDERER.md) - Detailed guide for 3D rendering
- [Architecture](../ARCHITECTURE.md) - System architecture
- [Development Plan](../DEVELOPMENT_PLAN.md) - Roadmap

## Troubleshooting

### LDView not found

Install LDView or specify path:
```bash
python src/generate_ldview_training_data.py \
    --ldview-path /usr/local/bin/ldview \
    --dat-dir /path/to/dat/files
```

See [LDVIEW_RENDERER.md](./LDVIEW_RENDERER.md#troubleshooting) for more troubleshooting.

### CUDA/GPU Issues

The model will automatically use GPU if available. To force CPU:
```python
# In train.py
device = torch.device("cpu")
```

## Performance

**Training:**
- 100 parts × 50 samples = 5,000 images
- Training time: ~2-4 hours on GPU, ~12-24 hours on CPU
- Model size: ~20 MB

**Inference:**
- CPU: ~50-100 ms per image
- GPU: ~10-20 ms per image

## License

See main repository LICENSE file.
