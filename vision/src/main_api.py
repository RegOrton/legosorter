from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from train import training_manager, state
from settings_manager import get_settings_manager
import uvicorn
import logging
import cv2
import os
from pathlib import Path
from camera import Camera
import torchvision.transforms as transforms
from typing import Dict, Any, List
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

app = FastAPI(title="Lego Sorter Vision API")

# Video file storage directory
VIDEO_UPLOAD_DIR = Path("/app/output/video_uploads")
VIDEO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global camera instance
camera = None

def get_camera():
    """Get or create camera instance."""
    global camera
    if camera is None:
        # Get camera type from settings
        settings_manager = get_settings_manager()
        camera_type = settings_manager.get("camera_type")
        video_file = settings_manager.get("video_file")
        playback_speed = settings_manager.get("video_playback_speed", 1.0)

        logger.info(f"Initializing camera with type: {camera_type}")

        # Handle video file camera type
        if camera_type == "video_file":
            if video_file:
                video_path = VIDEO_UPLOAD_DIR / video_file
                logger.info(f"Using video file: {video_path}")
                camera = Camera(
                    source=0,
                    width=640,
                    height=480,
                    camera_type=camera_type,
                    video_file=str(video_path),
                    playback_speed=playback_speed
                )
            else:
                logger.error("Video file camera type selected but no video file specified")
                return None
        else:
            camera = Camera(source=0, width=640, height=480, camera_type=camera_type)

        try:
            camera.start()
            logger.info(f"Camera initialized for streaming (type: {camera_type})")
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            camera = None
    return camera

def generate_video_stream():
    """Generate MJPEG video stream."""
    cam = get_camera()
    if cam is None:
        logger.error("Camera not available for streaming")
        return
    
    while True:
        ret, frame = cam.get_frame()
        if ret and frame is not None:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            logger.warning("Failed to get frame from camera")
            break

@app.get("/")
def root():
    return {"status": "ok", "service": "Lego Sorter Vision API"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/video/stream")
def video_stream():
    """MJPEG video stream endpoint."""
    return StreamingResponse(
        generate_video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/video/frame")
def video_frame():
    """Get a single frame as JPEG."""
    cam = get_camera()
    if cam is None:
        raise HTTPException(status_code=500, detail="Camera not available")
    
    ret, frame = cam.get_frame()
    if ret and frame is not None:
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            return StreamingResponse(
                iter([buffer.tobytes()]),
                media_type="image/jpeg"
            )
    
    raise HTTPException(status_code=500, detail="Failed to capture frame")

@app.get("/camera/type")
def get_camera_type():
    """Get current camera type."""
    cam = get_camera()
    if cam is None:
        return {"camera_type": None, "error": "Camera not initialized"}

    return {
        "camera_type": cam.camera_type,
        "available_types": ["usb", "csi", "http", "video_file"]
    }

@app.get("/camera/resolution")
def get_camera_resolution():
    """Get current camera frame resolution."""
    cam = get_camera()
    if cam is None:
        return {"width": 640, "height": 480, "error": "Camera not initialized"}

    ret, frame = cam.get_frame()
    if ret and frame is not None:
        h, w = frame.shape[:2]
        return {"width": w, "height": h}

    # Return defaults if unable to get frame
    return {"width": 640, "height": 480}

@app.post("/camera/type")
def set_camera_type(camera_type: str):
    """
    Set camera type and restart camera.

    Args:
        camera_type: Camera type - "usb", "csi", "http", or "video_file"
    """
    from camera import CAMERA_USB, CAMERA_CSI, CAMERA_HTTP, CAMERA_VIDEO_FILE

    valid_types = [CAMERA_USB, CAMERA_CSI, CAMERA_HTTP, CAMERA_VIDEO_FILE]
    if camera_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid camera type. Must be one of: {valid_types}"
        )

    cam = get_camera()
    if cam is None:
        raise HTTPException(status_code=500, detail="Camera not initialized")

    try:
        cam.set_camera_type(camera_type)
        logger.info(f"Camera type changed to: {camera_type}")
        return {
            "message": f"Camera type set to {camera_type}",
            "camera_type": camera_type
        }
    except Exception as e:
        logger.error(f"Failed to set camera type: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set camera type: {str(e)}")

@app.get("/settings")
def get_settings():
    """Get all current settings."""
    settings_manager = get_settings_manager()
    return settings_manager.get_all()

@app.post("/settings")
def update_settings(settings: Dict[str, Any]):
    """
    Update settings and persist to disk.

    Args:
        settings: Dictionary of settings to update (dataset, epochs, batch_size, camera_type, video_file, video_playback_speed)
    """
    settings_manager = get_settings_manager()

    # Validate settings
    valid_datasets = ["ldraw", "ldview", "rebrickable"]
    valid_cameras = ["usb", "csi", "http", "video_file"]

    if "dataset" in settings and settings["dataset"] not in valid_datasets:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dataset. Must be one of: {valid_datasets}"
        )

    if "camera_type" in settings and settings["camera_type"] not in valid_cameras:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid camera_type. Must be one of: {valid_cameras}"
        )

    if "epochs" in settings:
        try:
            epochs = int(settings["epochs"])
            if epochs < 1 or epochs > 100:
                raise ValueError()
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="epochs must be an integer between 1 and 100")

    if "batch_size" in settings:
        try:
            batch_size = int(settings["batch_size"])
            if batch_size < 1 or batch_size > 64:
                raise ValueError()
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="batch_size must be an integer between 1 and 64")

    if "video_playback_speed" in settings:
        try:
            speed = float(settings["video_playback_speed"])
            if speed <= 0 or speed > 10:
                raise ValueError()
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="video_playback_speed must be a number between 0.1 and 10")

    # Update settings
    success = settings_manager.update(settings)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to save settings")

    # If camera_type, video_file, or playback speed was changed, reset the global camera instance
    if "camera_type" in settings or "video_file" in settings or "video_playback_speed" in settings:
        global camera
        if camera is not None:
            logger.info("Camera settings changed, resetting camera instance")
            camera.release()
            camera = None

    logger.info(f"Settings updated: {settings}")
    return {"message": "Settings updated successfully", "settings": settings_manager.get_all()}

@app.post("/settings/reset")
def reset_settings():
    """Reset all settings to defaults."""
    settings_manager = get_settings_manager()
    success = settings_manager.reset()

    if not success:
        raise HTTPException(status_code=500, detail="Failed to reset settings")

    logger.info("Settings reset to defaults")
    return {"message": "Settings reset to defaults", "settings": settings_manager.get_all()}

@app.post("/train/start")
def start_training(epochs: int = 10, batch_size: int = 32, dataset: str = "ldraw"):
    """
    Start training with specified dataset.

    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        dataset: Dataset type - "ldraw" (Python software renderer), "ldview" (LDView realistic renders), or "rebrickable" (Rebrickable on-the-fly synthesis)
    """
    from train import DATASET_LDRAW, DATASET_REBRICKABLE, DATASET_LDVIEW

    # Map dataset string to constant
    dataset_map = {
        "ldraw": DATASET_LDRAW,
        "ldview": DATASET_LDVIEW,
        "rebrickable": DATASET_REBRICKABLE
    }

    dataset_type = dataset_map.get(dataset, DATASET_LDRAW)

    success, msg = training_manager.start_training(
        epochs=epochs,
        batch_size=batch_size,
        dataset_type=dataset_type
    )
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"message": msg}

@app.post("/train/stop")
def stop_training():
    success, msg = training_manager.stop_training()
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"message": msg}

@app.get("/train/status")
def get_status():
    # Format parts_stats for frontend
    parts_stats_list = []
    if state.parts_stats:
        for part_id, stats in state.parts_stats.items():
            avg_loss = (sum(stats['loss_values']) / len(stats['loss_values'])) if stats['loss_values'] else 0.0
            parts_stats_list.append({
                'part_id': part_id,
                'views': stats['views'],
                'epochs': round(stats['epochs'], 2),
                'samples': stats['samples'],
                'avg_loss': round(avg_loss, 4)
            })

    # Sort by part_id for consistent ordering
    parts_stats_list.sort(key=lambda x: x['part_id'])

    return {
        "is_running": state.is_running,
        "epoch": state.epoch,
        "total_epochs": state.total_epochs,
        "loss": state.loss,
        "logs": state.logs[-10:], # Return last 10 logs
        "images": state.latest_images,
        "loss_history": state.loss_history,
        "timing_stats": state.timing_stats,
        "batch_number": state.batch_number,
        "total_batches": state.total_batches,
        "parts_stats": parts_stats_list,
        "checkpoint_loaded": state.checkpoint_loaded,
        "checkpoint_path": state.checkpoint_path
    }

# Inference endpoints
@app.post("/inference/start")
def start_inference(
    model_path: str = "/app/output/models/lego_embedder_final.pth",
    mode: str = "auto"
):
    """
    Start real-time inference on webcam stream.
    
    Args:
        model_path: Path to model checkpoint
        mode: Inference mode - "auto" or "manual"
    """
    from inference import get_inference_engine
    
    cam = get_camera()
    if cam is None:
        raise HTTPException(status_code=500, detail="Camera not available")
    
    # Define class names (update these based on your actual classes)
    class_names = [
        "Brick 2x4",
        "Brick 2x2", 
        "Plate 2x4",
        "Plate 2x2",
        "Slope 45°",
        "Tile 1x2",
        "Round Brick",
        "Technic Brick",
        "Window",
        "Door"
    ]
    
    engine = get_inference_engine(
        model_path=model_path if Path(model_path).exists() else None,
        num_classes=len(class_names),
        class_names=class_names
    )
    
    # Set initial mode
    success, msg = engine.set_mode(mode)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    
    success, msg = engine.start(cam)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    
    return {"message": f"{msg} (Mode: {mode})"}

@app.post("/inference/stop")
def stop_inference():
    """Stop real-time inference."""
    from inference import get_inference_engine
    
    engine = get_inference_engine()
    success, msg = engine.stop()
    
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    
    return {"message": msg}

@app.post("/inference/mode")
def set_inference_mode(mode: str):
    """Switch inference mode (auto/manual)."""
    from inference import get_inference_engine
    
    engine = get_inference_engine()
    success, msg = engine.set_mode(mode)
    
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    
    return {"message": msg, "mode": engine.state.mode}

@app.get("/inference/status")
def get_inference_status():
    """Get current inference results."""
    from inference import get_inference_engine
    
    engine = get_inference_engine()
    
    return {
        "is_running": engine.state.is_running,
        "mode": engine.state.mode,
        "auto_state": engine.state.auto_state,
        "fps": engine.state.fps,
        "frame_count": engine.state.frame_count,
        "current_prediction": engine.state.current_prediction,
        "recent_predictions": engine.state.predictions[-10:] if engine.state.predictions else [],
        "error": engine.state.error,
        "bounding_boxes": engine.state.bounding_boxes,
        "center_detected": engine.state.center_detected
    }

@app.get("/inference/history")
def get_classification_history():
    """Get classification history with thumbnails."""
    from inference import get_inference_engine

    engine = get_inference_engine()

    return {
        "history": engine.state.classification_history
    }

@app.post("/inference/classify_now")
def classify_now():
    """
    Perform one-shot classification on current frame.
    In Manual mode, this triggers the inference loop.
    """
    from inference import get_inference_engine
    import time

    engine = get_inference_engine()
    
    if not engine.state.is_running:
         raise HTTPException(status_code=400, detail="Inference is not running. Start inference first.")
    
    # Trigger classification
    engine.trigger()
    
    # Wait briefly for result (optional, or just return 'triggered')
    # Since it runs in a separate thread, we can't easily return the exact result *here* 
    # without a sync mechanism, but for UI responsiveness it's often better to just say "Triggered"
    # and let the status poll pick it up.
    # However, to be nice to the legacy API contract, we could try to wait a split second.
    
    return {"message": "Classification triggered", "status": "ok"}

@app.post("/inference/calibrate")
def calibrate_background():
    """
    Calibrate background detection from current frame.

    Call this when the frame shows only the background (no objects).
    """
    from inference import get_inference_engine

    engine = get_inference_engine()
    camera = get_camera()

    if not engine.state.is_running:
        raise HTTPException(status_code=400, detail="Inference is not running. Start inference first.")

    # Get current frame from camera
    ret, frame = camera.get_frame()
    if not ret or frame is None:
        raise HTTPException(status_code=400, detail="Could not capture frame from camera")

    # Calibrate
    success, message = engine.calibrate_background(frame)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {
        "message": message,
        "is_calibrated": engine.state.is_calibrated,
        "calibration_status": engine.state.calibration_status
    }

@app.post("/inference/recalibrate")
def recalibrate_background():
    """
    Reset background calibration.
    """
    from inference import get_inference_engine

    engine = get_inference_engine()
    success, message = engine.recalibrate_background()

    return {
        "message": message,
        "is_calibrated": engine.state.is_calibrated,
        "calibration_status": engine.state.calibration_status
    }

@app.get("/inference/calibration_status")
def get_calibration_status():
    """
    Get detailed calibration status.
    """
    from inference import get_inference_engine

    engine = get_inference_engine()
    calibration_info = engine.get_calibration_status()

    return {
        "is_calibrated": engine.state.is_calibrated,
        "calibration_status": engine.state.calibration_status,
        "detector_info": calibration_info
    }

@app.post("/inference/detector/params")
def update_detector_params(
    min_area_percent: float = None,
    max_area_percent: float = None,
    diff_threshold: int = None,
    center_tolerance: float = None,
    edge_margin: int = None
):
    """
    Update detector parameters for fine-tuning object detection.

    Args:
        min_area_percent: Minimum object area as % of frame (default: 0.001 = 0.1% of frame)
        max_area_percent: Maximum object area as % of frame (default: 0.15 = 15% of frame)
        diff_threshold: Difference threshold for background subtraction (default: 30, lower = more sensitive)
        center_tolerance: Fraction of frame center for "centered" detection (default: 0.15)
        edge_margin: Pixels from edge where object is considered touching (default: 20)
    """
    from inference import get_inference_engine

    engine = get_inference_engine()

    if engine.detector is None:
        raise HTTPException(status_code=400, detail="Detector not initialized")

    updated_params = {}

    if min_area_percent is not None:
        engine.detector.min_area_percent = min_area_percent
        updated_params['min_area_percent'] = min_area_percent

    if max_area_percent is not None:
        engine.detector.max_area_percent = max_area_percent
        updated_params['max_area_percent'] = max_area_percent

    if diff_threshold is not None:
        engine.detector.diff_threshold = diff_threshold
        updated_params['diff_threshold'] = diff_threshold

    if center_tolerance is not None:
        engine.detector.center_tolerance = center_tolerance
        updated_params['center_tolerance'] = center_tolerance

    if edge_margin is not None:
        engine.detector.edge_margin = edge_margin
        updated_params['edge_margin'] = edge_margin

    # Save updated params to settings for persistence
    if updated_params:
        from settings_manager import get_settings_manager
        settings_manager = get_settings_manager()

        # Get current detector settings
        current_detector = settings_manager.get("detector", {})
        current_detector.update(updated_params)

        # Save back to settings
        settings_manager.set("detector", current_detector)
        logger.info(f"Saved detector params to settings: {updated_params}")

    logger.info(f"Updated detector params: {updated_params}")

    return {
        "message": "Detector parameters updated",
        "updated": updated_params,
        "current_params": {
            "min_area_percent": engine.detector.min_area_percent,
            "max_area_percent": engine.detector.max_area_percent,
            "diff_threshold": engine.detector.diff_threshold,
            "center_tolerance": engine.detector.center_tolerance,
            "edge_margin": engine.detector.edge_margin,
        }
    }

@app.get("/inference/detector/debug")
def get_detector_debug():
    """
    Get debug visualization showing what the detector sees.
    Returns base64-encoded image showing background difference.
    """
    from inference import get_inference_engine
    import base64

    engine = get_inference_engine()

    if not engine.state.is_calibrated:
        raise HTTPException(status_code=400, detail="Detector not calibrated. Calibrate first.")

    camera = get_camera()
    if camera is None:
        raise HTTPException(status_code=400, detail="Camera not available")

    ret, frame = camera.get_frame()
    if not ret or frame is None:
        raise HTTPException(status_code=400, detail="Could not capture frame")

    # Get debug visualization
    bounding_boxes, center_detected, status, debug_img = engine.detector.debug_detect(frame)

    # Encode debug image as JPEG
    _, buffer = cv2.imencode('.jpg', debug_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "status": status,
        "bounding_boxes": bounding_boxes,
        "center_detected": center_detected,
        "debug_image": img_base64,
        "detector_params": {
            "min_area_percent": engine.detector.min_area_percent,
            "max_area_percent": engine.detector.max_area_percent,
            "diff_threshold": engine.detector.diff_threshold,
        }
    }

@app.get("/inference/detector/calibration_bg")
def get_calibration_background():
    """
    Get the calibration background image.

    Returns base64-encoded image of calibration background.
    """
    import base64
    import numpy as np
    from pathlib import Path

    # Determine calibration path
    base_path = Path("/app/output") if Path("/app/output").exists() else Path(__file__).parent.parent / "output"
    calib_path = base_path / "calibration_bg.npy"

    if not calib_path.exists():
        raise HTTPException(status_code=404, detail="No calibration background found. Calibrate detector first.")

    try:
        # Load numpy array (float32)
        bg_frame = np.load(str(calib_path))
        # Convert to uint8 BGR for encoding
        bg_frame = bg_frame.astype(np.uint8)

        # Get resolution
        h, w = bg_frame.shape[:2]

        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', bg_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "calibration_image": img_base64,
            "resolution": {"width": w, "height": h},
            "exists": True
        }
    except Exception as e:
        logger.error(f"Failed to load calibration background: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load calibration: {str(e)}")

# Video file management endpoints
@app.post("/video/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file for use as camera input.

    Args:
        file: Video file (MP4, AVI, MOV, etc.)
    """
    # Validate file type
    allowed_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_extensions}"
        )

    # Generate unique filename
    filename = f"{Path(file.filename).stem}{file_ext}"
    file_path = VIDEO_UPLOAD_DIR / filename

    # Save file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Video file uploaded: {filename}")

        return {
            "message": "Video uploaded successfully",
            "filename": filename,
            "path": str(file_path)
        }
    except Exception as e:
        logger.error(f"Failed to upload video: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload video: {str(e)}")

@app.get("/video/list")
def list_videos():
    """List all uploaded video files."""
    try:
        video_files = []
        for file_path in VIDEO_UPLOAD_DIR.iterdir():
            if file_path.is_file():
                video_files.append({
                    "filename": file_path.name,
                    "size": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime
                })

        return {"videos": video_files}
    except Exception as e:
        logger.error(f"Failed to list videos: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list videos: {str(e)}")

@app.delete("/video/{filename}")
def delete_video(filename: str):
    """
    Delete a video file.

    Args:
        filename: Name of the video file to delete
    """
    file_path = VIDEO_UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    try:
        # Check if this video is currently in use
        settings_manager = get_settings_manager()
        current_video = settings_manager.get("video_file")

        if current_video == filename:
            raise HTTPException(
                status_code=400,
                detail="Cannot delete video file that is currently in use. Please select a different camera source first."
            )

        file_path.unlink()
        logger.info(f"Video file deleted: {filename}")

        return {"message": f"Video file '{filename}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete video: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete video: {str(e)}")

@app.post("/video/playback_speed")
def set_playback_speed(speed: float):
    """
    Set video playback speed.

    Args:
        speed: Playback speed multiplier (0.1 to 10.0)
    """
    if speed <= 0 or speed > 10:
        raise HTTPException(
            status_code=400,
            detail="Playback speed must be between 0.1 and 10.0"
        )

    global camera
    if camera is not None and camera.camera_type == "video_file":
        camera.set_playback_speed(speed)

    # Update settings
    settings_manager = get_settings_manager()
    settings_manager.set("video_playback_speed", speed)

    logger.info(f"Playback speed set to {speed}x")

    return {
        "message": f"Playback speed set to {speed}x",
        "speed": speed
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
