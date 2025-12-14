from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from train import training_manager, state
import uvicorn
import logging
import cv2
import os
from pathlib import Path
from camera import Camera
import torchvision.transforms as transforms

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
        camera = Camera(source=0, width=640, height=480)
        try:
            camera.start()
            logger.info("Camera initialized for streaming")
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

@app.post("/train/start")
def start_training(epochs: int = 10, batch_size: int = 32, dataset: str = "ldraw"):
    """
    Start training with specified dataset.

    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        dataset: Dataset type - "ldraw" (multi-view 3D renders) or "rebrickable" (single-view CGI)
    """
    from train import DATASET_LDRAW, DATASET_REBRICKABLE
    dataset_type = DATASET_LDRAW if dataset == "ldraw" else DATASET_REBRICKABLE
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
        "images": state.latest_images
    }

# Inference endpoints
@app.post("/inference/start")
def start_inference(model_path: str = "/app/output/models/lego_embedder_final.pth"):
    """Start real-time inference on webcam stream."""
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
    
    success, msg = engine.start(cam)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    
    return {"message": msg}

@app.post("/inference/stop")
def stop_inference():
    """Stop real-time inference."""
    from inference import get_inference_engine
    
    engine = get_inference_engine()
    success, msg = engine.stop()
    
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    
    return {"message": msg}

@app.get("/inference/status")
def get_inference_status():
    """Get current inference results."""
    from inference import get_inference_engine
    
    engine = get_inference_engine()
    
    return {
        "is_running": engine.state.is_running,
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
    """Perform one-shot classification on current frame."""
    from inference import get_inference_engine
    import torch
    import torchvision.transforms as transforms
    
    cam = get_camera()
    if cam is None:
        raise HTTPException(status_code=500, detail="Camera not available")
    
    # Get current frame
    ret, frame = cam.get_frame()
    if not ret or frame is None:
        raise HTTPException(status_code=500, detail="Failed to capture frame")
    
    # Initialize or get inference engine
    class_names = [
        "Brick 2x4", "Brick 2x2", "Plate 2x4", "Plate 2x2", "Slope 45°",
        "Tile 1x2", "Round Brick", "Technic Brick", "Window", "Door"
    ]
    
    engine = get_inference_engine(
        model_path="/app/output/models/lego_embedder_final.pth" if Path("/app/output/models/lego_embedder_final.pth").exists() else None,
        num_classes=len(class_names),
        class_names=class_names
    )
    
    # Preprocess frame
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(frame_rgb).unsqueeze(0).to(engine.device)
        
        # Run inference
        with torch.no_grad():
            logits, embeddings = engine.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)
        
        # Return results
        return {
            "class_id": predicted_class.item(),
            "class_name": class_names[predicted_class.item()],
            "confidence": confidence.item(),
            "probabilities": probabilities[0].cpu().numpy().tolist(),
            "all_classes": [
                {
                    "name": class_names[i],
                    "probability": probabilities[0][i].item()
                }
                for i in range(len(class_names))
            ]
        }
    
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
