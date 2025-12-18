"""
Real-time inference engine for Lego piece classification.

This module provides real-time classification of Lego pieces from the webcam stream.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model import LegoEmbeddingNet
from background_diff_detector import BackgroundDiffDetector
from settings_manager import get_settings_manager
import cv2
import numpy as np
from pathlib import Path
import logging
import threading
import time
from collections import deque
from enum import Enum

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


class InferenceMode(Enum):
    MANUAL = "manual"
    AUTO = "auto"

class AutoState(Enum):
    WAITING = "waiting"       # Waiting for object to appear/center
    STABILIZING = "stabilizing" # Object is centered, checking stability
    CLASSIFIED = "classified"   # Object classified, waiting for it to leave
    COOLDOWN = "cooldown"     # Short cooldown after classification

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

        # New State Machine fields
        self.mode = InferenceMode.AUTO.value
        self.auto_state = AutoState.WAITING.value
        self.last_classification_time = 0.0

        # Calibration state
        self.is_calibrated = False
        self.calibration_status = "Not calibrated"

        # Stability tracking for better detection
        self.last_bbox_center = None
        self.stability_frame_count = 0


class InferenceEngine:
    """Real-time inference engine for webcam stream."""

    def __init__(self, model_path=None, reference_db_path=None, num_classes=10, class_names=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]

        # Load embedding model (no classification head needed for nearest-neighbor)
        from model import LegoEmbeddingNet
        self.model = LegoEmbeddingNet(embedding_dim=128).to(self.device)

        if model_path and Path(model_path).exists():
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded embedding model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}. Using untrained model.")

        self.model.eval()

        # Load reference database for nearest-neighbor classification
        self.reference_db = None
        if reference_db_path and Path(reference_db_path).exists():
            self._load_reference_db(reference_db_path)
        else:
            logger.warning("No reference database found. Classification will use generic class names.")

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
        self.trigger_event = threading.Event() # For manual trigger

        self.thread = None
        self.camera = None

        # FPS tracking
        self.frame_times = deque(maxlen=30)

        # Load detector settings from settings manager
        settings_manager = get_settings_manager()
        detector_settings = settings_manager.get("detector", {})

        # Fast background differencing detector (replaces MOG2)
        self.detector = BackgroundDiffDetector(
            min_area_percent=detector_settings.get("min_area_percent", 0.001),
            max_area_percent=detector_settings.get("max_area_percent", 0.15),
            diff_threshold=detector_settings.get("diff_threshold", 30),
            center_tolerance=detector_settings.get("center_tolerance", 0.15),
            edge_margin=detector_settings.get("edge_margin", 20),
            min_aspect_ratio=detector_settings.get("min_aspect_ratio", 0.3),
            max_aspect_ratio=detector_settings.get("max_aspect_ratio", 3.0)
        )

        # Try to load saved calibration
        if self.detector.load_calibration():
            self.state.is_calibrated = True
            self.state.calibration_status = "Loaded from disk"
            logger.info("Loaded saved calibration on startup")
        else:
            logger.info("No saved calibration found - calibration required")

        # Stability params
        self.stability_counter = 0
        self.stability_threshold = 8  # frames required for stability
        self.position_tolerance = 5  # pixels of movement allowed

    def _load_reference_db(self, db_path):
        """Load reference embedding database for nearest-neighbor classification."""
        try:
            data = np.load(str(db_path), allow_pickle=True)
            self.reference_db = {
                'part_ids': data['part_ids'].tolist(),
                'embeddings': data['embeddings'],
                'metadata': data['metadata'].tolist() if 'metadata' in data else []
            }
            logger.info(f"Loaded reference database with {len(self.reference_db['part_ids'])} parts")
            logger.info(f"Parts: {', '.join(self.reference_db['part_ids'][:10])}...")
        except Exception as e:
            logger.error(f"Failed to load reference database: {e}")
            self.reference_db = None

    def _find_nearest_neighbor(self, embedding):
        """Find nearest neighbor in reference database using cosine similarity."""
        if self.reference_db is None:
            return None, 0.0

        # Normalize query embedding
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)

        # Normalize reference embeddings
        ref_embeddings = self.reference_db['embeddings']
        ref_norms = np.linalg.norm(ref_embeddings, axis=1, keepdims=True) + 1e-8
        ref_embeddings_norm = ref_embeddings / ref_norms

        # Compute cosine similarity
        similarities = np.dot(ref_embeddings_norm, embedding_norm)

        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        best_part = self.reference_db['part_ids'][best_idx]

        return best_part, float(best_similarity)

    def calibrate_background(self, frame):
        """
        Calibrate background from current frame.

        Call this when frame shows only the background (no objects).

        Args:
            frame: Input frame (BGR)

        Returns:
            Tuple of (success, message)
        """
        success = self.detector.calibrate(frame)
        self.state.is_calibrated = success

        if success:
            self.state.calibration_status = "Calibrated"
            logger.info("Background calibration successful")
            return True, "Background calibrated successfully"
        else:
            self.state.calibration_status = "Calibration failed"
            return False, "Failed to calibrate background"

    def recalibrate_background(self):
        """Reset calibration."""
        self.detector.reset_calibration()
        self.state.is_calibrated = False
        self.state.calibration_status = "Not calibrated"
        self.stability_counter = 0
        logger.info("Background calibration reset")
        return True, "Calibration reset"

    def get_calibration_status(self):
        """Get detailed calibration status."""
        return self.detector.get_calibration_status()

    def set_mode(self, mode_str):
        """Set inference mode (auto/manual)."""
        try:
            new_mode = InferenceMode(mode_str)
            self.state.mode = new_mode.value
            logger.info(f"Switched inference mode to: {new_mode.value}")

            # Reset internal state on mode switch
            self.state.auto_state = AutoState.WAITING.value
            self.stability_counter = 0
            self.state.last_bbox_center = None
            self.state.stability_frame_count = 0
            return True, f"Mode set to {mode_str}"
        except ValueError:
            return False, f"Invalid mode: {mode_str}"

    def trigger(self):
        """Manually trigger classification."""
        if self.state.mode != InferenceMode.MANUAL.value:
             # Even in auto mode, we might want to force a trigger? 
             # For now let's allow it but log a warning if it's weird.
             logger.info("Manual trigger received (in Auto mode)")
        else:
             logger.info("Manual trigger received")
        
        self.trigger_event.set()
        return True

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
        """Detect objects in frame using fast background differencing."""
        if not self.state.is_calibrated:
            return [], False

        bounding_boxes, center_detected, status = self.detector.detect(frame)

        # Add stability tracking for better filtering
        for bbox in bounding_boxes:
            if bbox['is_centered'] and not bbox['touches_edge'] and bbox['aspect_ratio_valid']:
                # Track stability of centered object
                current_center = (bbox['center_x'], bbox['center_y'])

                if self.state.last_bbox_center is None:
                    self.state.last_bbox_center = current_center
                    self.state.stability_frame_count = 1
                else:
                    # Check if position is stable
                    dx = abs(current_center[0] - self.state.last_bbox_center[0])
                    dy = abs(current_center[1] - self.state.last_bbox_center[1])

                    if dx <= self.position_tolerance and dy <= self.position_tolerance:
                        self.state.stability_frame_count += 1
                    else:
                        self.state.stability_frame_count = 1
                        self.state.last_bbox_center = current_center

                bbox['is_stable'] = self.state.stability_frame_count >= self.stability_threshold
                bbox['stability_counter'] = self.state.stability_frame_count
            else:
                bbox['is_stable'] = False
                bbox['stability_counter'] = 0
                self.state.last_bbox_center = None
                self.state.stability_frame_count = 0

        return bounding_boxes, center_detected
    
    def _run_classifier(self, frame):
        """Run classification using nearest-neighbor search in embedding space."""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Transform
            input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)

            # Get embedding
            with torch.no_grad():
                embedding = self.model(input_tensor)
                embedding = embedding.cpu().numpy().flatten()

            # Find nearest neighbor in reference database
            part_id, similarity = self._find_nearest_neighbor(embedding)

            if part_id is not None:
                # Use nearest-neighbor result
                prediction = {
                    'class_id': -1,  # Not used for nearest-neighbor
                    'class_name': part_id,
                    'confidence': similarity,
                    'probabilities': [],  # Not applicable for nearest-neighbor
                    'timestamp': time.time()
                }
            else:
                # No reference database, use generic class name
                prediction = {
                    'class_id': 0,
                    'class_name': "Unknown",
                    'confidence': 0.0,
                    'probabilities': [],
                    'timestamp': time.time()
                }

            self.state.current_prediction = prediction
            self.state.predictions.append(prediction)
            
            # Keep only last 100 predictions
            if len(self.state.predictions) > 100:
                self.state.predictions.pop(0)
            
            self.state.last_classification_time = time.time()
            logger.info(f"Classified: {prediction['class_name']} ({prediction['confidence']:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"Classifier error: {e}")
            self.state.error = str(e)
            return False

    def _process_frame(self, frame):
        """Process a single frame for inference logic."""
        try:
            # Preprocess / Detect objects (ALWAYS RUN for UI feedback)
            bounding_boxes, center_detected = self._detect_objects(frame)
            self.state.bounding_boxes = bounding_boxes
            self.state.center_detected = center_detected
            self.state.frame_count += 1

            # --- STATE MACHINE ---
            if self.state.mode == InferenceMode.MANUAL.value:
                if self.trigger_event.is_set():
                    logger.info("Executing manual trigger...")
                    self._run_classifier(frame)
                    self.trigger_event.clear()

            elif self.state.mode == InferenceMode.AUTO.value:
                # Auto Mode Logic - classify when object is stable and centered

                # Check if any bbox is stable and centered
                is_stable_and_centered = any(
                    bbox.get('is_stable', False) and
                    bbox.get('is_centered', False) and
                    not bbox.get('touches_edge', False) and
                    bbox.get('aspect_ratio_valid', True)
                    for bbox in bounding_boxes
                )

                if is_stable_and_centered:
                    if self.state.auto_state == AutoState.WAITING.value:
                        self.state.auto_state = AutoState.STABILIZING.value
                        self.stability_counter = 1
                    elif self.state.auto_state == AutoState.STABILIZING.value:
                        # Already stabilizing, just run classifier
                        self._run_classifier(frame)
                        self.state.auto_state = AutoState.CLASSIFIED.value
                        logger.info("Object stable and centered - classifying")
                else:
                    # No stable centered object
                    self.stability_counter = 0
                    if self.state.auto_state == AutoState.CLASSIFIED.value:
                        # Object left, re-arm
                        self.state.auto_state = AutoState.WAITING.value
                        logger.info("Object removed, re-armed.")
                    elif self.state.auto_state == AutoState.STABILIZING.value:
                        # Lost tracking before stable
                        self.state.auto_state = AutoState.WAITING.value

            return True

        except Exception as e:
            logger.error(f"Inference error: {e}")
            self.state.error = str(e)
            return False

    def _inference_loop(self):
        """Main inference loop."""
        logger.info("Starting inference loop")
        
        try:
            while not self.stop_event.is_set():
                start_time = time.time()
                
                # Get frame from camera
                ret, frame = self.camera.get_frame()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                
                # Process frame
                self._process_frame(frame)
                
                # Calculate FPS
                elapsed = time.time() - start_time
                self.frame_times.append(elapsed)
                if len(self.frame_times) > 0:
                    avg_time = sum(self.frame_times) / len(self.frame_times)
                    self.state.fps = 1.0 / avg_time if avg_time > 0 else 0
                
                # Limit to ~30 FPS max to save CPU
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

def get_inference_engine(model_path=None, reference_db_path=None, num_classes=10, class_names=None):
    """Get or create the global inference engine."""
    global inference_engine
    if inference_engine is None:
        # Default paths if not provided
        if model_path is None:
            base_path = Path("/app/output") if Path("/app/output").exists() else Path(__file__).parent.parent / "output"
            model_path = base_path / "models" / "lego_embedder_final.pth"

        if reference_db_path is None:
            base_path = Path("/app/output") if Path("/app/output").exists() else Path(__file__).parent.parent / "output"
            reference_db_path = base_path / "reference_db.npz"

        inference_engine = InferenceEngine(
            model_path=model_path,
            reference_db_path=reference_db_path,
            num_classes=num_classes,
            class_names=class_names
        )
    return inference_engine
