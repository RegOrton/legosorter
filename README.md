# LEGO Sorter with Chaotic Storage

An automated LEGO sorting and retrieval system using computer vision and "Chaotic Storage" methodology inspired by Amazon's robotic warehouses.

## Core Concept: Chaotic Storage

Unlike traditional sorters that separate bricks by type into specific bins, this system uses **Chaotic Storage**:
- **No Fixed Bins**: Any box can hold any mix of bricks
- **High Density**: Each box holds exactly 30 bricks
- **Database Tracking**: Computer vision identifies each brick, database tracks `{BrickID, BoxID}`
- **Retrieval**: System queries DB for nearest box containing requested part

## System Architecture

**4 Main Components:**
1. **Vision System** (Python + PyTorch) - Identifies LEGO bricks using metric learning and nearest-neighbor classification
2. **Frontend Dashboard** (Next.js + React) - Live video feed, training interface, settings, machine controls
3. **Backend API** (FastAPI) - Coordinates inference, training, camera, and hardware
4. **Hardware Interface** (Raspberry Pi + GPIO) - Controls conveyor, servo, gantry (Phase 4)

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.7+ (for webcam server on Windows)
- Git

### 1. Clone Repository
```bash
git clone <repository-url>
cd legosorter
```

### 2. Start Webcam Server (Windows only)
```bash
python webcam_server.py
```

### 3. Start Vision Service
```bash
cd vision
docker-compose up -d
```

### 4. Start Frontend Dashboard
```bash
cd frontend
docker-compose up -d
```

### 5. Access Dashboards
- **Main Dashboard**: http://localhost:3000
- **Training Dashboard**: http://localhost:3000/training
- **Settings**: http://localhost:3000/settings
- **Vision API Docs**: http://localhost:8000/docs

## Training & Classification Workflow

### Step 1: Prepare Training Data

**Option A: LDView (Recommended)**
```bash
# Download LDraw library from https://library.ldraw.org/
# Place .dat files in vision/input/dat_files/
docker-compose exec vision python src/generate_ldview_training_data.py
```

**Option B: LDraw Python Renderer**
```bash
cd vision && unzip complete.zip -d data/
python src/generate_ldraw_dataset.py --num-parts 100
```

### Step 2: Calibrate Background
1. Navigate to http://localhost:3000
2. Start inference
3. Clear the camera view (no objects)
4. Click "Calibrate Now"
5. Background saved to `calibration_bg.npy`

### Step 3: Train Model
1. Go to http://localhost:3000/settings
2. Select dataset (ldview recommended)
3. Set epochs (10-20) and batch size (8-16)
4. Save settings
5. Go to http://localhost:3000/training
6. Click "Start Training"
7. Wait for completion (~10-20 min)

### Step 4: Build Reference Database
```bash
docker-compose exec vision python src/build_reference_db.py
```

This generates `reference_db.npz` with embeddings for each part.

### Step 5: Run Auto-Classification
1. Go to http://localhost:3000
2. Start inference (should auto-start)
3. Switch to AUTO mode
4. Place LEGO brick in center of frame
5. System detects, stabilizes, classifies, and displays part ID

**Classification Results**: Shows actual part ID (e.g., "3001") with confidence score (0-100%)

## Key Features

### Computer Vision
- **Metric Learning**: Triplet loss training for robust embeddings
- **Nearest-Neighbor Classification**: Cosine similarity search in embedding space
- **Background Subtraction**: Frame differencing for object detection
- **Calibration Persistence**: Background auto-loaded on startup
- **Multi-Dataset Support**: LDraw, LDView, or Rebrickable sources

### Auto-Classification Pipeline
Three-state workflow displayed on dashboard:
1. **Waiting for Part** - Place part in center
2. **Classifying** - Object detected, processing (5 frames / ~166ms)
3. **Remove Part** - Classification complete, ready for next

### Training Features
- Live triplet visualization
- Real-time loss graphs
- Calibration background preview
- Training logs streaming
- Settings persistence

## Documentation

- **[CLAUDE.md](./CLAUDE.md)** - Comprehensive development guide and API reference
- **[BACKGROUND.md](./BACKGROUND.md)** - Chaotic storage methodology and operational logic
- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - System design and component architecture
- **[vision/LDVIEW_RENDERER.md](./vision/LDVIEW_RENDERER.md)** - LDView setup and usage

## Current Status

**Completed (Phase 1-2)**:
- ✅ Docker-based vision service + frontend
- ✅ Metric learning training pipeline
- ✅ Nearest-neighbor classification system
- ✅ Auto-classification state machine
- ✅ Background calibration with persistence
- ✅ Live dashboards (main, training, settings)
- ✅ Video file mode for testing
- ✅ Reference database generation

**In Progress**:
- 🚧 Inference refinement and accuracy improvements
- 🚧 Main dashboard inventory explorer

**Planned (Phase 3-4)**:
- Database integration (SQLite)
- Hardware interface (Raspberry Pi GPIO)
- Retrieval system UI
- Multi-part tracking and batch processing

## Tech Stack

- **Backend**: Python, FastAPI, PyTorch, OpenCV
- **Frontend**: Next.js 14, React, TypeScript, Tailwind CSS
- **Model**: MobileNetV3 + triplet loss (128-dim embeddings)
- **Database**: SQLite (planned)
- **Hardware**: Raspberry Pi 4, stepper motors, servos (planned)

## License

See LICENSE file for details.
