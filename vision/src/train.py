import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader
from model import LegoEmbeddingNet
import argparse
from pathlib import Path
import logging
import threading
import time
import base64
import cv2
import numpy as np

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingState:
    def __init__(self):
        self.is_running = False
        self.epoch = 0
        self.total_epochs = 0
        self.loss = 0.0
        self.logs = []
        self.latest_images = None # { 'anchor': b64, 'positive': b64, 'negative': b64 }

    def log(self, message):
        self.logs.append(message)
        if len(self.logs) > 100:
            self.logs.pop(0)

state = TrainingState()

def tensor_to_b64(tensor):
    # tensor: (3, 224, 224)
    # Un-normalize
    # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    
    img = tensor.cpu().numpy()
    img = img * std + mean # Denormalize
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0)) # CHW -> HWC
    
    # Convert RGB to BGR for OpenCV encoding if needed, but we output base64 for web (RGB usually assumed or PNG)
    # OpenCV encodes BGR by default, so if we want RGB in web, we should swap if using cv2.imencode.
    # Actually, let's keep it simple. cv2 expects BGR. 
    # Our input was RGB. So img is RGB. 
    # cv2.imencode expects BGR. So we convert RGB -> BGR.
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    _, buffer = cv2.imencode('.jpg', img)
    b64 = base64.b64encode(buffer).decode('utf-8')
    return b64

class Trainer:
    def __init__(self):
        self.stop_event = threading.Event()
        self.thread = None

    def start_training(self, epochs=10, batch_size=32, limit=None):
        if self.thread and self.thread.is_alive():
            return False, "Training already running"
        
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._train_loop, args=(epochs, batch_size, limit))
        self.thread.start()
        return True, "Training started"

    def stop_training(self):
        if not self.thread or not self.thread.is_alive():
            return False, "Training not running"
        
        self.stop_event.set()
        self.thread.join(timeout=5)
        return True, "Stopping training..."

    def _train_loop(self, epochs, batch_size, limit):
        state.is_running = True
        state.total_epochs = epochs
        state.epoch = 0
        state.loss = 0.0
        state.log("Initializing training...")
        
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state.log(f"Device: {device}")
            
            data_dir = "/app/data" 
            if not Path(data_dir).exists():
                data_dir = str(Path(__file__).resolve().parent.parent / "data")
            
            dataloader = get_dataloader(data_dir, batch_size=batch_size, limit=limit)
            model = LegoEmbeddingNet(embedding_dim=128).to(device)
            state.log("Model loaded.")
            
            criterion = nn.TripletMarginLoss(margin=1.0, p=2)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            save_path = Path(data_dir).parent / "output" / "models"
            save_path.mkdir(parents=True, exist_ok=True)
            
            model.train()
            
            for epoch in range(epochs):
                if self.stop_event.is_set():
                    state.log("Training interrupted by user.")
                    break
                    
                state.epoch = epoch + 1
                total_loss = 0.0
                batches = 0
                
                for i, (anchor, positive, negative) in enumerate(dataloader):
                    if self.stop_event.is_set(): 
                        break

                    anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                    
                    optimizer.zero_grad()
                    emb_a = model(anchor)
                    emb_p = model(positive)
                    emb_n = model(negative)
                    loss = criterion(emb_a, emb_p, emb_n)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batches += 1
                    
                    # Update metrics and image preview
                    if i % 2 == 0: # Update frequently
                        state.loss = loss.item()
                        try:
                            # Capture the first item in the batch for preview
                            state.latest_images = {
                                'anchor': tensor_to_b64(anchor[0]),
                                'positive': tensor_to_b64(positive[0]),
                                'negative': tensor_to_b64(negative[0])
                            }
                        except Exception as e:
                            # Don't crash training loop if preview fails
                            print(f"Preview error: {e}")
                
                avg_loss = total_loss / batches if batches > 0 else 0
                state.log(f"Epoch {epoch+1} done. Avg Loss: {avg_loss:.4f}")
                
                # Checkpoint
                torch.save(model.state_dict(), save_path / f"lego_embedder_epoch_{epoch+1}.pth")
            
            if not self.stop_event.is_set():
                state.log("Training Complete.")
                torch.save(model.state_dict(), save_path / "lego_embedder_final.pth")
                
        except Exception as e:
            logger.error(f"Training crashed: {e}")
            state.log(f"Error: {e}")
        finally:
            state.is_running = False

# Global instance
training_manager = Trainer()
