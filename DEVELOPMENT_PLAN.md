# Master Development Plan: Lego Sorter

| **Project** | Lego Sorter (Chaotic Storage) |
| :--- | :--- |
| **Status** | PLANNING |
| **Strategy** | Riskiest Assumption Test (RAT) - Vision First |

---

## Phase 1: The Eyes (Computer Vision & Environment)
**Objective:** robustly identify LEGO parts from a video feed.
**Constraint:** Must run in Docker (Dev on Windows, Prod on Pi).

### 1.1 Development Environment & Docker
- [ ] **Docker Setup**:
    -   Create `input/` and `output/` directories for testing.
    -   Write `Dockerfile` based on `python:3.11-slim`.
    -   Install `opencv-python-headless`, `numpy`, `requests`.
    -   Configure `docker-compose.yml` to mount local source code.
-   **Verification**: Run `hello_world.py` inside the container via Docker Compose.

### 1.2 Image Acquisition & Preprocessing
- [ ] **Capture Service**:
    -   Implement `Camera` class ensuring abstraction (Webcam vs. File Stream).
    -   Create a "Dataset Generator" script to capture labeled images of parts for training/testing.
- [ ] **Preprocessing Pipeline**:
    -   Implement `isolate_brick(frame)`:
        -   Convert to Grayscale -> Gaussian Blur -> Thresholding.
        -   Find Contours -> Filter by Area (ignore noise).
        -   **Output**: Cropped image of the brick on a black background.

### 1.3 Reference Database (The Knowledge)
- [ ] **Rebrickable Integration**:
    -   Get API Key.
    -   Script to fetch common parts (e.g., top 100 parts) with images.
    -   Store feature vectors (e.g., ORB descriptors) of reference images in a local file (`features.pkl`).

### 1.4 Recognition Engine
- [ ] **Feature Matching**:
    -   Implement `Inspector` class.
    -   Compute ORB/SIFT descriptors for the captured brick.
    -   Match against `features.pkl` using BFMatcher.
    -   Filter matches (Lowe's ratio test).
    -   **Metric**: If good matches > N, return `PartID`.

---

## Phase 2: The Brain (Backend & Database)
**Objective:** Manage inventory and make decisions.
**Stack:** Python (FastAPI), SQLite.

### 2.1 Database Schema
- [ ] **Models**:
    -   `Box`: `id` (PK, int), `capacity` (int), `current_count` (int), `status` (enum: FILLING, STORAGE).
    -   `Part`: `id` (PK, str), `name` (str), `image_url` (str).
    -   `Item`: `id` (PK), `part_id` (FK), `box_id` (FK), `color` (str), `timestamp`.
- [ ] **Migration**: Setup Alembic for schema migrations.

### 2.2 Core Logic (The "Chaotic" Engine)
- [ ] **Ingestion Logic**:
    -   `add_item(part_id, color)`:
        -   Look up active "FILLING" box.
        -   If none or full, trigger `swap_box()`.
        -   Insert `Item`.
        -   Increment box count.
- [ ] **Retrieval Logic**:
    -   `find_part(part_id, color)`:
        -   Query `Item` table.
        -   Group by `box_id`.
        -   Return list of boxes sorted by "retrieval cost" (placeholder for physical distance).

### 2.3 API Layer
- [ ] **Endpoints**:
    -   `GET /status`: System health.
    -   `POST /identify`: Upload image -> Return Part ID (Connects to Vision).
    -   `GET /inventory`: Search parts.
    -   `POST /retrieve`: Request a part.

---

## Phase 3: The Face (Frontend UI)
**Objective:** User interaction and monitoring.
**Stack:** Next.js (Local).

### 3.1 Project Setup
- [ ] Initialize Next.js 14 (App Router).
- [ ] Setup TailwindCSS.
- [ ] Setup ShadcnUI (optional) for polished components.

### 3.2 Dashboard
- [ ] **Live Feed**: Canvas element drawing images sent from Backend (or MJPEG stream).
- [ ] **Status Panel**: Display "Current Box" status (e.g., 12/30 items).

### 3.3 Inventory Viewer
- [ ] **Data Grid**: Searchable table of parts.
- [ ] **Retrieval Action**: "Get This" button triggering the API.

---

## Phase 4: The Body (Hardware Interface)
**Objective:** Physical control.
**Stack:** Raspberry Pi GPIO.

### 4.1 Hardware Abstraction Layer (HAL)
- [ ] Create `Motor` abstract base class.
- [ ] Create `MockMotor` for Windows dev (logs actions to console).
- [ ] Create `GpioMotor` using `RPi.GPIO` or `gpiozero`.

### 4.2 Subsystems
- [ ] **Conveyor**: Simple On/Off/Speed control.
- [ ] **Diverter**: Servo control (Angle A = Pass, Angle B = Storage).
- [ ] **Gantry**: X/Y/Z coordinate system mapper.

### 4.3 Control Loop
- [ ] **Main State Machine**:
    -   `IDLE` -> `FEEDING` -> `DETECTED` -> `SORTING` -> `IDLE`.
    -   Interrupt handling for `FULL_BOX` or `JAM`.
