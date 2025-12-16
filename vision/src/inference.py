"""
Real-time inference engine for Lego piece classification.

This module provides real-time classification of Lego pieces from the webcam stream.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model import LegoEmbeddingNet
import cv2
import numpy as np
from pathlib import Path
import logging
import threading
import time
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LegoClassifier(nn.Module):
    """Classifier built on top of the embedding model."""
    
    def __init__(self, embedding_dim=128, num_classes=10):
        super(LegoClassifier, self).__init__()
        self.embedder = LegoEmbeddingNet(embedding_dim=embedding_dim, pretrained=False)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        embeddings = self.embedder(x)
        logits = self.classifier(embeddings)
        return logits, embeddings


class InferenceState:
    """Shared state for inference results."""
    
    def __init__(self):
        self.is_running = False
        self.predictions = []  # List of recent predictions
        self.current_prediction = None  # Latest prediction
        self.fps = 0.0
        self.frame_count = 0
        self.error = None
        self.bounding_boxes = []  # List of detected bounding boxes
        self.center_detected = False  # True if object detected in center


class InferenceEngine:
    """Real-time inference engine for webcam stream."""
    
    def __init__(self, model_path=None, num_classes=10, class_names=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
        # Load model
        self.model = LegoClassifier(embedding_dim=128, num_classes=num_classes).to(self.device)
        
        if model_path and Path(model_path).exists():
            try:
                # Load embedding weights
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.embedder.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}. Using untrained model.")
        
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.state = InferenceState()
        self.stop_event = threading.Event()
        self.thread = None
        self.camera = None
        
        # FPS tracking
        self.frame_times = deque(maxlen=30)
        
        # Background subtractor for object detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
        self.frame_buffer = deque(maxlen=10)  # Buffer for background learning
    
    def start(self, camera):
        """Start inference on the camera stream."""
        if self.thread and self.thread.is_alive():
            return False, "Inference already running"
        
        self.camera = camera
        self.stop_event.clear()
        self.state.is_running = True
        self.state.error = None
        
        self.thread = threading.Thread(target=self._inference_loop)
        self.thread.daemon = True
        self.thread.start()
        
        return True, "Inference started"
    
    def stop(self):
        """Stop inference."""
        if not self.thread or not self.thread.is_alive():
            return False, "Inference not running"
        
        self.stop_event.set()
        self.thread.join(timeout=5)
        self.state.is_running = False
        
        return True, "Inference stopped"
    
    def _detect_objects(self, frame):
        """Detect objects in frame using background subtraction and contour detection."""
        h, w = frame.shape[:2]
        frame_center_x, frame_center_y = w // 2, h // 2
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bounding_boxes = []
        center_detected = False
        
        # Process each contour
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter small contours (noise) and very large ones (background)
            if area < 500 or area > (w * h * 0.5):
                continue
            
            # Get bounding box
            x, y, box_w, box_h = cv2.boundingRect(contour)
            
            # Calculate bbox center
            bbox_center_x = x + box_w // 2
            bbox_center_y = y + box_h // 2
            
            # Check if bbox center is within 20% of frame center
            center_threshold_x = w * 0.2
            center_threshold_y = h * 0.2
            
            is_centered = (
                abs(bbox_center_x - frame_center_x) < center_threshold_x and
                abs(bbox_center_y - frame_center_y) < center_threshold_y
            )
            
            if is_centered:
                center_detected = True
            
            bounding_boxes.append({
                'x': int(x),
                'y': int(y),
                'width': int(box_w),
                'height': int(box_h),
                'center_x': int(bbox_center_x),
                'center_y': int(bbox_center_y),
                'area': int(area),
                'is_centered': is_centered
            })
        
        return bounding_boxes, center_detected
    
    def _inference_loop(self):
        """Main inference loop."""
        logger.info("Starting inference loop")
        
        try:
            while not self.stop_event.is_set():
                start_time = time.time()
                
                # Get frame from camera
                ret, frame = self.camera.get_frame()
                if not ret or frame is None:
                    logger.warning("Failed to get frame")
                    time.sleep(0.1)
                    continue
                
                # Preprocess frame
                try:
                    # Detect objects and bounding boxes
                    bounding_boxes, center_detected = self._detect_objects(frame)
                    self.state.bounding_boxes = bounding_boxes
                    self.state.center_detected = center_detected
                    
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Transform
                    input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
                    
                    # Run inference
                    with torch.no_grad():
                        logits, embeddings = self.model(input_tensor)
                        probabilities = torch.softmax(logits, dim=1)
                        confidence, predicted_class = torch.max(probabilities, dim=1)
                    
                    # Store results
                    prediction = {
                        'class_id': predicted_class.item(),
                        'class_name': self.class_names[predicted_class.item()],
                        'confidence': confidence.item(),
                        'probabilities': probabilities[0].cpu().numpy().tolist(),
                        'timestamp': time.time(),
                        'bounding_boxes': bounding_boxes,
                        'center_detected': center_detected
                    }
                    
                    self.state.current_prediction = prediction
                    self.state.predictions.append(prediction)
                    
                    # Keep only last 100 predictions
                    if len(self.state.predictions) > 100:
                        self.state.predictions.pop(0)
                    
                    self.state.frame_count += 1
                    
                except Exception as e:
                    logger.error(f"Inference error: {e}")
                    self.state.error = str(e)
                
                # Calculate FPS
                elapsed = time.time() - start_time
                self.frame_times.append(elapsed)
                if len(self.frame_times) > 0:
                    avg_time = sum(self.frame_times) / len(self.frame_times)
                    self.state.fps = 1.0 / avg_time if avg_time > 0 else 0
                
                # Limit to ~30 FPS max
                if elapsed < 0.033:
                    time.sleep(0.033 - elapsed)
        
        except Exception as e:
            logger.error(f"Inference loop crashed: {e}")
            self.state.error = str(e)
        finally:
            self.state.is_running = False
            logger.info("Inference loop stopped")


# Global inference engine instance
inference_engine = None

def get_inference_engine(model_path=None, num_classes=10, class_names=None):
    """Get or create the global inference engine."""
    global inference_engine
    if inference_engine is None:
        inference_engine = InferenceEngine(
            model_path=model_path,
            num_classes=num_classes,
            class_names=class_names
        )
    return inference_engine
