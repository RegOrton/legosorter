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

    def log(self, message):
        self.logs.append(message)
        if len(self.logs) > 100:
            self.logs.pop(0)

state = TrainingState()

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
                    
                    # Update state more frequently for UI
                    if i % 5 == 0:
                        state.loss = loss.item()
                
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
