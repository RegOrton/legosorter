# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Lego Sorter** using "Chaotic Storage" methodology - an automated storage and retrieval system (ASRS) inspired by Amazon's robotic warehouses. Unlike traditional sorters that separate bricks by type, this system stores any mix of bricks in boxes (30 bricks/box) and relies on computer vision identification + database tracking for retrieval.

**Core Concept**: The system doesn't care what's in a container, only where it is. A webcam identifies each brick, the system drops it into the current "Fill Box", and the database records `{BrickID, BoxID}`. For retrieval, the user requests a part, and the system queries the DB for the nearest box containing it.

## Architecture

The system has 4 main components:

1. **The Brain (Core Logic & Database)**: Python (FastAPI) + SQLite - tracks inventory, manages box states, provides API
2. **The Face (User Interface)**: Next.js (React) - visualization and user input
3. **The Eyes (Computer Vision)**: Python (OpenCV + PyTorch) - identifies bricks from webcam stream using metric learning
4. **The Body (Hardware Interface)**: Raspberry Pi + GPIO - controls conveyor, servo, gantry (planned for Phase 4)

**Data Flow**: User ↔ Next.js Frontend ↔ Python Backend API ↔ SQLite DB. Separately: Webcam → Computer Vision → Backend API → Hardware Interface.

## Development Environment

### System Requirements

- **OS**: Development on Windows, Production on Raspberry Pi
- **Docker**: Vision service runs in Docker container
- **Python**: 3.7+ (for webcam server on Windows host)
- **Node.js**: Required for Next.js frontend

### Webcam Setup (Windows + Docker)

Docker on Windows cannot directly access webcam. The solution:

1. **Webcam Server** runs on Windows host (port 5000) - streams frames via HTTP
2. **Webcam Client** in Docker container fetches frames from `http://host.docker.internal:5000`

**Start webcam server** (must be running before Docker container):
```bash
python webcam_server.py
```

**Environment variables** (in `vision/docker-compose.yml`):
- `USE_HTTP_WEBCAM=true` - enables HTTP webcam mode
- `WEBCAM_SERVER_URL=http://host.docker.internal:5000` - webcam server URL

## Common Commands

### Frontend (Next.js)

```bash
cd frontend
npm install              # Install dependencies
npm run dev              # Start dev server (http://localhost:3000)
npm run build            # Production build
npm start                # Start production server
npm run lint             # Run ESLint
```

### Vision Service (Docker)

```bash
cd vision

# Start vision API with webcam support
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Execute commands in container
docker-compose exec vision python src/test_webcam.py

# Stop container
docker-compose down
```

**Vision API** runs on `http://localhost:8000` when container is running.

### Training the Model

Training uses **Metric Learning** (Triplet Loss) with MobileNetV3 backbone to generate embeddings for LEGO brick classification.

**Start training via API**:
```bash
curl -X POST "http://localhost:8000/train/start?epochs=10&batch_size=32"
```

**Check training status**:
```bash
curl http://localhost:8000/train/status
```

**Stop training**:
```bash
curl -X POST "http://localhost:8000/train/stop"
```

**Or use the Frontend Dashboard**: Navigate to `http://localhost:3000/training` for live training visualization with triplet images and logs.

### Inference

**Start real-time inference**:
```bash
curl -X POST "http://localhost:8000/inference/start"
```

**One-shot classification** (classify current frame):
```bash
curl -X POST "http://localhost:8000/inference/classify_now"
```

**Get inference status**:
```bash
curl http://localhost:8000/inference/status
```

## Code Structure

### Vision Service (`vision/src/`)

- **`main_api.py`** - FastAPI server with endpoints for training, inference, video streaming
- **`train.py`** - Training loop with threading, uses TrainingState for live updates
- **`model.py`** - LegoEmbeddingNet (MobileNetV3 + metric learning)
- **`dataset.py`** - LegoTripletDataset for triplet loss training
- **`synthesizer.py`** - LegoSynthesizer generates synthetic training samples with perspective transforms, shadows, and camera effects
- **`generate_backgrounds.py`** - Generates synthetic background images (solid colors, gradients, textures) for training diversity
- **`camera.py`** - Camera wrapper that supports both direct cv2.VideoCapture and HTTP webcam client
- **`webcam_client.py`** - WebcamClient class for fetching frames from Windows host
- **`inference.py`** - Real-time inference engine (planned)
- **`preprocessor.py`** - Image preprocessing utilities
- **`ingest_rebrickable.py`** - Downloads LEGO part images from Rebrickable API

### Frontend (`frontend/app/`)

- **`page.tsx`** - Main dashboard (planned: inventory explorer, retrieval interface)
- **`training/page.tsx`** - Training dashboard with live triplet visualization and logs
- **`layout.tsx`** - Root layout
- **`globals.css`** - Tailwind CSS styles

### Data Directories

- **`vision/data/`** - Training data (clean brick images, backgrounds, Rebrickable database)
- **`vision/input/`** - Input images for processing
- **`vision/output/`** - Generated outputs (models saved to `vision/output/models/`)
- **`vision/output/debug/`** - Debug outputs (test frames, etc.)

## Key Technical Details

### Metric Learning Pipeline

The vision system uses **Triplet Loss** to learn embeddings:
- **Anchor**: A synthetic image of a LEGO brick
- **Positive**: Another synthetic image of the SAME brick (different rotation/position/background)
- **Negative**: A synthetic image of a DIFFERENT brick

The model (MobileNetV3) learns to place similar bricks close in embedding space and dissimilar bricks far apart.

### Synthetic Data Generation

Since real training data is limited (single-viewpoint CGI renders from Rebrickable), `LegoSynthesizer` applies aggressive augmentation to bridge the sim-to-real domain gap:

1. **Background Masking**: Removes white background from CGI renders using thresholding + morphological cleanup
2. **Perspective Transform** (70% chance): Simulates viewing angle changes to compensate for single-viewpoint source data
3. **2D Rotation**: Full 360° rotation around camera axis
4. **Scale Variation**: 0.6x to 1.3x random scaling
5. **Multi-Background Compositing**: Randomly selects from 15+ backgrounds (solid colors, gradients, textures)
6. **Synthetic Shadows** (70% chance): Directional drop shadows with random light angle
7. **Camera Effects**:
   - Brightness/contrast adjustment (simulates exposure)
   - Sensor noise injection
   - Gaussian blur (focus/motion simulation)
   - Color temperature shifts (warm/cool lighting)

The `dataset.py` adds additional PyTorch transforms:
- **ColorJitter** (80% chance): brightness, contrast, saturation, hue variation
- **RandomGrayscale** (10% chance): forces shape-based learning
- **GaussianBlur** (30% chance): additional blur augmentation
- **RandomErasing** (10% chance): simulates occlusion

Run `python vision/src/generate_backgrounds.py` to regenerate synthetic backgrounds.
Run `python vision/src/test_synthesizer.py` to visualize augmentation samples in `vision/output/debug/`.

### Camera Abstraction

The `Camera` class in `camera.py` automatically switches between:
- **Direct capture**: `cv2.VideoCapture(0)` when running on Pi or native Linux
- **HTTP webcam**: `WebcamClient` when `USE_HTTP_WEBCAM=true` (for Docker on Windows)

Code using `Camera` doesn't need to know which mode is active.

## Important Conventions

### API Port Mapping
- **Vision API**: `8000` (Docker container port 8000 → host port 8000)
- **Webcam Server**: `5000` (Windows host only)
- **Frontend**: `3000` (Next.js dev server)

### Frontend API Configuration
The frontend hardcodes the Vision API URL in `frontend/app/training/page.tsx`:
```typescript
const API_URL = "http://localhost:8000";
```

If running Vision API on a different host/port, update this variable.

### Docker Host Networking
The vision container uses `extra_hosts` to access the Windows host:
```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

This allows the container to reach `http://host.docker.internal:5000` for the webcam server.

### Model Checkpointing
Trained models are saved to `/app/output/models/` inside the container, which maps to `vision/output/models/` on the host via Docker volume mount.

Final model: `lego_embedder_final.pth`

### Environment Variables
The vision service reads:
- `PYTHONUNBUFFERED=1` - ensures logs appear immediately
- `USE_HTTP_WEBCAM=true` - enables HTTP webcam mode
- `WEBCAM_SERVER_URL` - webcam server URL

## Development Workflow

### Starting from Scratch

1. **Start webcam server** (Windows terminal 1):
   ```bash
   python webcam_server.py
   ```

2. **Start vision API** (terminal 2):
   ```bash
   cd vision
   docker-compose up --build
   ```

3. **Start frontend** (terminal 3):
   ```bash
   cd frontend
   npm run dev
   ```

4. **Access dashboards**:
   - Main: http://localhost:3000
   - Training: http://localhost:3000/training
   - Vision API docs: http://localhost:8000/docs

### Testing Webcam Access

Inside the vision container:
```bash
docker-compose exec vision python src/test_webcam.py
```

This captures a test frame and saves to `vision/output/test_webcam_frame.jpg`.

## Project Status

**Current Phase**: Phase 1-2 (Computer Vision + Initial Frontend)

Completed:
- ✅ Docker-based vision service with FastAPI
- ✅ HTTP webcam streaming for Windows development
- ✅ Metric learning model architecture (MobileNetV3)
- ✅ Synthetic data generation pipeline
- ✅ Training API with live status updates
- ✅ Frontend training dashboard with live visualization
- ✅ Video streaming endpoints

In Progress:
- 🚧 Inference engine refinement
- 🚧 Main dashboard (inventory explorer)

Planned:
- Phase 3: Complete UI (retrieval interface, developer mode)
- Phase 4: Hardware interface (Raspberry Pi GPIO, motors, servos)
- Database integration (SQLite for inventory tracking)

## Troubleshooting

### "Failed to connect to webcam server"
- Ensure `webcam_server.py` is running on Windows host
- Check firewall isn't blocking port 5000
- Test: `curl http://localhost:5000/status` from host

### "Camera not opening on Windows"
- Close other apps using webcam (Teams, Zoom, etc.)
- Try different camera index: `python webcam_server.py --camera 1`
- Check Windows Privacy Settings → Camera

### Frontend can't reach Vision API
- Ensure vision container is running: `docker-compose ps`
- Check API is responding: `curl http://localhost:8000/health`
- Verify port 8000 is mapped in `docker-compose.yml`

### Training fails with "No images found"
- Check `vision/data/` contains brick images
- Run `ingest_rebrickable.py` to download training images
- Verify background image exists in `vision/data/backgrounds/`

## References

See project documentation:
- [ARCHITECTURE.md](./ARCHITECTURE.md) - System architecture details
- [BACKGROUND.md](./BACKGROUND.md) - Chaotic storage methodology explained
- [PRD.md](./PRD.md) - Product requirements and KPIs
- [WEBCAM_SETUP.md](./WEBCAM_SETUP.md) - Detailed webcam setup guide
- [DEVELOPMENT_PLAN.md](./DEVELOPMENT_PLAN.md) - Development roadmap
