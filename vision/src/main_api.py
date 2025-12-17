from fastapi import FastAPI, HTTPException
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
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

app = FastAPI(title="Lego Sorter Vision API")

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

        logger.info(f"Initializing camera with type: {camera_type}")
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
        "available_types": ["usb", "csi", "http"]
    }

@app.post("/camera/type")
def set_camera_type(camera_type: str):
    """
    Set camera type and restart camera.

    Args:
        camera_type: Camera type - "usb", "csi", or "http"
    """
    from camera import CAMERA_USB, CAMERA_CSI, CAMERA_HTTP

    valid_types = [CAMERA_USB, CAMERA_CSI, CAMERA_HTTP]
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
        settings: Dictionary of settings to update (dataset, epochs, batch_size, camera_type)
    """
    settings_manager = get_settings_manager()

    # Validate settings
    valid_datasets = ["ldraw", "ldview", "rebrickable"]
    valid_cameras = ["usb", "csi", "http"]

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

    # Update settings
    success = settings_manager.update(settings)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to save settings")

    # If camera_type was changed, reset the global camera instance
    if "camera_type" in settings:
        global camera
        if camera is not None:
            logger.info("Camera type changed, resetting camera instance")
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
        "total_batches": state.total_batches
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
