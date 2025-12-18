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

# Dataset types
DATASET_REBRICKABLE = "rebrickable"
DATASET_LDRAW = "ldraw"
DATASET_LDVIEW = "ldview"

class TrainingState:
    def __init__(self):
        self.is_running = False
        self.epoch = 0
        self.total_epochs = 0
        self.loss = 0.0
        self.logs = []
        self.latest_images = None # { 'anchor': b64, 'positive': b64, 'negative': b64 }
        self.loss_history = []  # List of (epoch, loss) tuples
        self.timing_stats = {
            'data_generation': 0.0,
            'forward_pass': 0.0,
            'backward_pass': 0.0,
            'total_time': 0.0
        }
        self.batch_number = 0
        self.total_batches = 0

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

    def start_training(self, epochs=10, batch_size=32, limit=None, dataset_type=DATASET_LDRAW):
        if self.thread and self.thread.is_alive():
            return False, "Training already running"

        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._train_loop,
            args=(epochs, batch_size, limit, dataset_type)
        )
        self.thread.start()
        return True, f"Training started with {dataset_type} dataset"

    def stop_training(self):
        if not self.thread or not self.thread.is_alive():
            return False, "Training not running"

        self.stop_event.set()
        self.thread.join(timeout=5)
        return True, "Stopping training..."

    def _train_loop(self, epochs, batch_size, limit, dataset_type):
        state.is_running = True
        state.total_epochs = epochs
        state.epoch = 0
        state.loss = 0.0
        state.log(f"Initializing training with {dataset_type} dataset...")

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state.log(f"Device: {device}")

            data_dir = "/app/data"
            if not Path(data_dir).exists():
                data_dir = str(Path(__file__).resolve().parent.parent / "data")

            # Load appropriate dataset
            if dataset_type == DATASET_LDRAW:
                from ldraw_dataset import get_ldraw_dataloader
                ldraw_renders_dir = Path(data_dir) / "ldraw_renders"
                if not ldraw_renders_dir.exists():
                    state.log("ERROR: LDraw renders not found. Run generate_ldraw_dataset.py first.")
                    state.is_running = False
                    return
                dataloader = get_ldraw_dataloader(ldraw_renders_dir, batch_size=batch_size, limit=limit)
                state.log(f"Loaded LDraw dataset with {len(dataloader.dataset.parts)} parts")
            elif dataset_type == DATASET_LDVIEW:
                # LDView generates images on-the-fly during training
                from ldview_dataset import get_ldview_dataloader
                dat_dir = Path(data_dir).parent / "input" / "dat_files"
                if not dat_dir.exists():
                    state.log(f"ERROR: .dat files directory not found at {dat_dir}")
                    state.log("Please create the directory and add .dat files, or use a different dataset.")
                    state.is_running = False
                    return
                dat_files_count = len(list(dat_dir.glob("*.dat")))
                if dat_files_count == 0:
                    state.log(f"ERROR: No .dat files found in {dat_dir}")
                    state.log("Please add .dat files to the directory, or use a different dataset.")
                    state.is_running = False
                    return

                # Load calibration background if available
                base_path = Path("/app/output") if Path("/app/output").exists() else Path(__file__).parent.parent / "output"
                calib_path = base_path / "calibration_bg.npy"
                background_path = None

                if calib_path.exists():
                    # Save calibration as temporary image file for LDView
                    import numpy as np
                    bg_frame = np.load(str(calib_path))
                    temp_bg_path = base_path / "calibration_bg_temp.jpg"
                    cv2.imwrite(str(temp_bg_path), bg_frame.astype(np.uint8))
                    background_path = str(temp_bg_path)
                    state.log(f"Using calibration background for training: {bg_frame.shape}")
                else:
                    state.log("No calibration found - using default backgrounds")

                # Generate 1000 triplets per epoch on-the-fly
                samples_per_epoch = limit if limit else 1000
                dataloader = get_ldview_dataloader(
                    dat_dir,
                    batch_size=batch_size,
                    samples_per_epoch=samples_per_epoch,
                    background_path=background_path
                )
                state.log(f"Loaded LDView on-the-fly renderer with {dat_files_count} .dat files")
                state.log(f"Will generate {samples_per_epoch} triplets per epoch in real-time")
            else:
                # Rebrickable uses the synthesizer which automatically loads calibration background
                dataloader = get_dataloader(data_dir, batch_size=batch_size, limit=limit)
                # Check if calibration background was loaded by synthesizer
                base_path = Path("/app/output") if Path("/app/output").exists() else Path(__file__).parent.parent / "output"
                calib_path = base_path / "calibration_bg.npy"
                if calib_path.exists():
                    state.log("Loaded Rebrickable dataset with calibration background")
                else:
                    state.log("Loaded Rebrickable dataset with synthetic backgrounds")

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
                epoch_start = time.time()

                # Reset timing stats for this epoch
                epoch_data_gen = 0.0
                epoch_forward = 0.0
                epoch_backward = 0.0
                state.total_batches = len(dataloader)

                for i, (anchor, positive, negative) in enumerate(dataloader):
                    state.batch_number = i
                    if self.stop_event.is_set():
                        break

                    batch_start = time.time()

                    # Timing: Data loading (done by dataloader before this point)
                    data_load_time = time.time() - batch_start

                    # Timing: GPU transfer
                    transfer_start = time.time()
                    anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                    transfer_time = time.time() - transfer_start

                    # Timing: Forward pass
                    forward_start = time.time()
                    optimizer.zero_grad()
                    emb_a = model(anchor)
                    emb_p = model(positive)
                    emb_n = model(negative)
                    loss = criterion(emb_a, emb_p, emb_n)
                    forward_time = time.time() - forward_start

                    # Timing: Backward pass
                    backward_start = time.time()
                    loss.backward()
                    optimizer.step()
                    backward_time = time.time() - backward_start

                    batch_total_time = time.time() - batch_start

                    total_loss += loss.item()
                    batches += 1

                    # Accumulate timing stats
                    epoch_data_gen += data_load_time
                    epoch_forward += forward_time
                    epoch_backward += backward_time

                    # Log detailed timing every 5 batches
                    if i % 5 == 0:
                        state.log(f"Batch {i}: Total={batch_total_time:.3f}s (Transfer={transfer_time:.3f}s, Forward={forward_time:.3f}s, Backward={backward_time:.3f}s) Loss={loss.item():.4f}")

                    # Update metrics and timing stats
                    state.loss = loss.item()

                    # Update timing stats with percentages
                    total_time_so_far = epoch_data_gen + epoch_forward + epoch_backward
                    if total_time_so_far > 0:
                        state.timing_stats = {
                            'data_generation': (epoch_data_gen / total_time_so_far) * 100,
                            'forward_pass': (epoch_forward / total_time_so_far) * 100,
                            'backward_pass': (epoch_backward / total_time_so_far) * 100,
                            'total_time': total_time_so_far
                        }

                    # Cache batch tensors for continuous display
                    try:
                        # Convert all samples in batch to base64 once
                        batch_images = []
                        for sample_idx in range(anchor.size(0)):
                            batch_images.append({
                                'anchor': tensor_to_b64(anchor[sample_idx]),
                                'positive': tensor_to_b64(positive[sample_idx]),
                                'negative': tensor_to_b64(negative[sample_idx])
                            })

                        # Start background thread to continuously loop through samples
                        # This keeps UI updating while next batch is being generated
                        import threading
                        import time as time_module

                        def update_display():
                            while state.is_running:
                                for imgs in batch_images:
                                    if not state.is_running:
                                        break
                                    state.latest_images = imgs
                                    time_module.sleep(0.2)  # 200ms per sample

                        # Stop previous display thread if exists
                        if hasattr(state, 'display_thread') and state.display_thread and state.display_thread.is_alive():
                            pass  # Will be replaced by new batch

                        # Start new display thread
                        state.display_thread = threading.Thread(target=update_display, daemon=True)
                        state.display_thread.start()

                    except Exception as e:
                        # Don't crash training loop if preview fails
                        print(f"Preview error: {e}")
                
                epoch_time = time.time() - epoch_start
                avg_loss = total_loss / batches if batches > 0 else 0
                time_per_batch = epoch_time / batches if batches > 0 else 0
                state.log(f"Epoch {epoch+1} done in {epoch_time:.2f}s. Avg Loss: {avg_loss:.4f}, Time/Batch: {time_per_batch:.3f}s, Batches: {batches}")

                # Add to loss history
                state.loss_history.append({
                    'epoch': epoch + 1,
                    'loss': avg_loss
                })

                # Checkpoint
                checkpoint_start = time.time()
                torch.save(model.state_dict(), save_path / f"lego_embedder_epoch_{epoch+1}.pth")
                checkpoint_time = time.time() - checkpoint_start
                state.log(f"Checkpoint saved in {checkpoint_time:.3f}s")
            
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
