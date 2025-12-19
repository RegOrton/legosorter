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
- **Docker**: Both vision service and frontend run in Docker containers (recommended)
- **Python**: 3.7+ (for webcam server on Windows host)
- **Node.js**: Optional - only required if running frontend locally (outside Docker)

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

#### Docker Container (Recommended)

```bash
cd frontend

# Start frontend in Docker container
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop container
docker-compose down
```

**Frontend dashboard** runs on `http://localhost:3000` when container is running.

#### Local Development (Alternative)

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

### Nearest-Neighbor Classification System

The vision system uses **embedding-based nearest-neighbor search** for classification instead of a traditional classification head.

#### How It Works

1. **Training**: Model learns embeddings via triplet loss (anchor, positive, negative)
2. **Reference Database**: Pre-computed embeddings for known parts stored in `reference_db.npz`
3. **Classification**: At inference, compute embedding for detected object and find closest match using cosine similarity
4. **Result**: Returns actual LEGO part ID (e.g., "3001", "3005") with similarity score (0-1)

#### Building the Reference Database

After training, generate reference embeddings:

```bash
# In vision container
docker-compose exec vision python src/build_reference_db.py

# Or directly if running locally
python vision/src/build_reference_db.py
```

This creates `vision/output/reference_db.npz` containing:
- `part_ids`: Array of part IDs (e.g., ["3001", "3003", "3004"])
- `embeddings`: Averaged embeddings for each part (shape: [num_parts, 128])
- `metadata`: Optional metadata for each part

**Default settings**: 10 views per part, uses calibration background if available

#### Auto-Classification State Machine

When inference runs in AUTO mode, it follows a 3-state pipeline:

| State | Description | Duration |
|-------|-------------|----------|
| **WAITING** | Waiting for object to appear and center | Until object detected |
| **STABILIZING** | Object centered, classification in progress | 5 frames (~166ms) |
| **CLASSIFIED** | Classification complete, waiting for object to leave | Until object exits |

**Flow**: Object appears → becomes stable (8 frames) → STABILIZING (classify immediately) → wait 5 frames → CLASSIFIED → object leaves → WAITING

The dashboard displays this pipeline visually when running in AUTO mode at http://localhost:3000.

**Switch modes**:
```bash
# Auto mode (detects centered objects automatically)
curl -X POST "http://localhost:8000/inference/mode?mode=auto"

# Manual mode (classify on demand via button)
curl -X POST "http://localhost:8000/inference/mode?mode=manual"
```

## Code Structure

### Vision Service (`vision/src/`)

- **`main_api.py`** - FastAPI server with endpoints for training, inference, video streaming, settings, and camera control
- **`train.py`** - Training loop with threading, uses TrainingState for live updates, supports 3 dataset types
- **`settings_manager.py`** - Persistent settings storage (dataset, epochs, batch_size, camera_type) saved to JSON
- **`model.py`** - LegoEmbeddingNet (MobileNetV3 + metric learning)
- **`dataset.py`** - LegoTripletDataset for triplet loss training (Rebrickable on-the-fly synthesis)
- **`synthesizer.py`** - LegoSynthesizer generates synthetic training samples with perspective transforms, shadows, and camera effects (auto-loads calibration background)
- **`generate_backgrounds.py`** - Generates synthetic background images (solid colors, gradients, textures) for training diversity
- **`ldraw_parser.py`** - Parses LDraw .dat files and resolves subfile references to build 3D geometry
- **`ldraw_renderer.py`** - Software 3D renderer for generating multi-view training images from LDraw models
- **`ldraw_dataset.py`** - PyTorch dataset for training with multi-view LDraw renders
- **`generate_ldraw_dataset.py`** - CLI tool to generate training images from LDraw library
- **`ldview_renderer.py`** - LDView-based renderer for realistic 3D renders
- **`generate_ldview_training_data.py`** - CLI to generate training data using LDView
- **`build_reference_db.py`** - Generates reference embedding database from trained model for nearest-neighbor classification
- **`camera.py`** - Camera wrapper supporting USB, CSI (Raspberry Pi), HTTP webcam, and video file modes
- **`webcam_client.py`** - WebcamClient class for fetching frames from Windows host
- **`inference.py`** - Real-time inference engine with auto-classification state machine and nearest-neighbor search
- **`background_diff_detector.py`** - Background subtraction detector for object detection with calibration persistence
- **`preprocessor.py`** - Image preprocessing utilities
- **`ingest_rebrickable.py`** - Downloads LEGO part images from Rebrickable API

### Frontend (`frontend/app/`)

- **`page.tsx`** - Main dashboard with live video feed, bounding box visualization, auto-classification pipeline indicator, calibration controls, and machine controls
- **`training/page.tsx`** - Training dashboard with live triplet visualization, logs, calibration background display, and training controls
- **`settings/page.tsx`** - Settings page for configuring dataset, training parameters, camera source, and video file upload
- **`layout.tsx`** - Root layout
- **`globals.css`** - Tailwind CSS styles

### Data Directories

- **`vision/data/`** - Training data (clean brick images, backgrounds, Rebrickable database)
- **`vision/data/ldraw_renders/`** - LDraw Python software renderer output
- **`vision/data/ldview_training/`** - LDView realistic 3D renders
- **`vision/input/`** - Input images for processing (includes .dat files for LDView training)
- **`vision/output/`** - Generated outputs
- **`vision/output/models/`** - Trained models (lego_embedder_final.pth)
- **`vision/output/reference_db.npz`** - Reference embedding database for nearest-neighbor classification
- **`vision/output/calibration_bg.npy`** - Saved calibration background (auto-loaded on startup)
- **`vision/output/settings.json`** - Persistent settings (dataset, epochs, batch_size, camera_type, video_file)
- **`vision/output/video_uploads/`** - Uploaded video files for video file mode
- **`vision/output/debug/`** - Debug outputs (test frames, detector visualizations)

## Key Technical Details

### Metric Learning Pipeline

The vision system uses **Triplet Loss** to learn embeddings:
- **Anchor**: A synthetic image of a LEGO brick
- **Positive**: Another synthetic image of the SAME brick (different rotation/position/background)
- **Negative**: A synthetic image of a DIFFERENT brick

The model (MobileNetV3) learns to place similar bricks close in embedding space and dissimilar bricks far apart.

### Training Data Sources

The system supports three training data sources (configurable via Settings page or API):

#### 1. LDraw Python Software Renderer (Default)

Uses the [LDraw](https://www.ldraw.org/) 3D parts library to generate true multi-view training images:

**Setup:**
```bash
# Download complete.zip from https://library.ldraw.org/library/updates/complete.zip
# Place in vision/ directory, then extract:
cd vision && unzip complete.zip -d data/

# Generate training renders (94 parts × 5 colors × 8 views = 3,760 images):
python vision/src/generate_ldraw_dataset.py --num-parts 100 --views-per-color 8 --num-colors 5
```

**Key files:**
- `ldraw_parser.py` - Parses .dat files, resolves subfile references, extracts 3D geometry
- `ldraw_renderer.py` - Software 3D renderer with orthographic projection, shading, multi-view generation
- `ldraw_dataset.py` - PyTorch dataset for triplet/classification training
- `generate_ldraw_dataset.py` - CLI to generate training images

**Advantages:**
- True 3D viewpoint variation (elevation 20-70°, azimuth 0-360°)
- Consistent geometry across all angles
- 23,000+ parts available in LDraw library

#### 2. LDView Realistic 3D Renders

Uses LDView command-line tool for photorealistic renders with proper lighting and shading:
- Pre-generate training images from .dat files
- Realistic lighting, shadows, and textures
- Multiple camera angles per part

**Setup:**
```bash
# Generate LDView training data
python vision/src/generate_ldview_training_data.py \
    --dat-dir vision/input/dat_files \
    --output-dir vision/data/ldview_training \
    --samples-per-part 20
```

See `vision/LDVIEW_RENDERER.md` for detailed LDView setup instructions.

#### 3. Rebrickable On-the-Fly Synthesis

Uses single-viewpoint CGI renders from Rebrickable with aggressive real-time augmentation:
- Perspective transforms to simulate angle changes
- Synthetic shadows and camera effects
- Multiple backgrounds generated on-the-fly

**Generate backgrounds:** `python vision/src/generate_backgrounds.py`
**Test augmentations:** `python vision/src/test_synthesizer.py`

### Persistent Settings System

Settings are stored in `vision/output/settings.json` and persist across restarts:

**Default Settings:**
- Dataset: `ldraw` (LDraw Python renderer)
- Epochs: `10`
- Batch Size: `8`
- Camera Type: `usb`
- Video File: `null` (none selected)
- Video Playback Speed: `1.0` (normal speed)

**Configure via Settings Page:**
Navigate to `http://localhost:3000/settings` to configure:
- Dataset source (ldraw, ldview, rebrickable)
- Training parameters (epochs, batch size)
- Camera source (usb, csi, http, video_file)
- Video file selection and upload
- Video playback speed

**Configure via API:**
```bash
# Get current settings
curl http://localhost:8000/settings

# Update settings
curl -X POST http://localhost:8000/settings \
  -H "Content-Type: application/json" \
  -d '{"dataset": "ldview", "epochs": 20, "batch_size": 16, "camera_type": "http"}'

# Reset to defaults
curl -X POST http://localhost:8000/settings/reset
```

### Starting Training

**Via Settings + Frontend (Recommended):**
1. Configure settings at `http://localhost:3000/settings`
2. Click "Save Settings"
3. Navigate to `http://localhost:3000/training`
4. Click "Start Training" (uses saved settings)

**Via API:**
```bash
# Uses saved settings from settings.json
curl -X POST "http://localhost:8000/train/start?epochs=20&batch_size=16&dataset=ldraw"

# Or specify parameters directly
curl -X POST "http://localhost:8000/train/start?epochs=20&batch_size=16&dataset=ldview"
```

### Camera System

The `Camera` class in [camera.py](vision/src/camera.py) supports four modes (switchable via Settings page or API):

- **USB Camera** (`usb`): Direct USB webcam via `cv2.VideoCapture(0)`
- **CSI Camera** (`csi`): Raspberry Pi camera module via GStreamer or V4L2
- **HTTP Camera** (`http`): Remote webcam via `WebcamClient` (for Docker on Windows)
- **Video File** (`video_file`): Pre-recorded MP4/AVI/MOV video that loops continuously with configurable playback speed

**Switch camera via API:**
```bash
curl -X POST "http://localhost:8000/camera/type?camera_type=video_file"
```

**Switch camera via Settings page:**
Navigate to `http://localhost:3000/settings` and select camera source.

#### Video File Mode

The video file mode allows you to use pre-recorded videos for testing and development without a physical camera:

**Upload a video:**
1. Navigate to Settings page: `http://localhost:3000/settings`
2. Select "Video File" as camera source
3. Click "Upload Video" and select an MP4, AVI, MOV, MKV, or WebM file
4. Select the uploaded video from the list
5. Adjust playback speed (0.1x to 5.0x)
6. Click "Save Settings"

**Video file storage:**
- Uploaded videos are stored in `vision/output/video_uploads/`
- Videos loop automatically when playback reaches the end
- Playback speed can be adjusted in real-time

**Video management API:**
```bash
# Upload video
curl -X POST -F "file=@myvideo.mp4" http://localhost:8000/video/upload

# List uploaded videos
curl http://localhost:8000/video/list

# Delete video
curl -X DELETE http://localhost:8000/video/{filename}

# Set playback speed
curl -X POST "http://localhost:8000/video/playback_speed?speed=2.0"
```

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

### Starting from Scratch (Containerized - Recommended)

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
   docker-compose up --build
   ```

4. **Access dashboards**:
   - Main: http://localhost:3000
   - Training: http://localhost:3000/training
   - Settings: http://localhost:3000/settings
   - Vision API docs: http://localhost:8000/docs

### Local Development (Alternative)

If you prefer to run the frontend locally without Docker:

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
   npm install  # First time only
   npm run dev
   ```

**Note:** See [DASHBOARD_SETUP.md](./DASHBOARD_SETUP.md) for additional deployment options.

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
- ✅ Three dataset options: LDraw Python, LDView, Rebrickable
- ✅ Synthetic data generation pipeline
- ✅ Training API with live status updates
- ✅ Frontend training dashboard with live visualization
- ✅ Dedicated Settings page with persistent storage
- ✅ Camera switching (USB, CSI, HTTP)
- ✅ Video streaming endpoints
- ✅ Settings management API

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
