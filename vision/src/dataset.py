import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import random
from pathlib import Path
from synthesizer import LegoSynthesizer
import cv2
import numpy as np

class LegoTripletDataset(Dataset):
    def __init__(self, images_dir, background_path, synthesizer=None, transform=None, limit=None):
        """
        Args:
            images_dir (Path): Directory containing clean element images.
            background_path (Path): Path to background texture.
            synthesizer (LegoSynthesizer): Optional pre-initialized synthesizer.
            transform (callable): Transform to apply to images.
            limit (int): Limit dataset size for testing.
        """
        self.images_dir = Path(images_dir)
        self.image_paths = list(self.images_dir.glob("*.jpg"))
        
        if limit:
            self.image_paths = self.image_paths[:limit]
            
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {images_dir}")
            
        if synthesizer:
            self.synth = synthesizer
        else:
            self.synth = LegoSynthesizer(background_path, output_size=(224, 224))
            
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def _generate_synthetic(self, image_path):
        """Generates a synthetic image from the source path."""
        # Synthesizer returns BGR numpy array
        img = self.synth.generate_sample(image_path)
        if img is None:
            # Fallback for some reason, maybe return black or try another?
            # ideally shouldn't happen if path exists
            return np.zeros((224, 224, 3), dtype=np.uint8)
            
        # Convert BGR to RGB for PyTorch transformations
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        # Anchor
        anchor_path = self.image_paths[idx]
        anchor_img = self._generate_synthetic(anchor_path)
        
        # Positive (Same class, different synthesis)
        positive_img = self._generate_synthetic(anchor_path)
        
        # Negative (Different class)
        # Avoid selecting the same class
        negative_idx = idx
        while negative_idx == idx:
            negative_idx = random.randint(0, len(self.image_paths) - 1)
        
        negative_path = self.image_paths[negative_idx]
        negative_img = self._generate_synthetic(negative_path)
        
        # Apply transforms
        if self.transform:
            anchor = self.transform(anchor_img)
            positive = self.transform(positive_img)
            negative = self.transform(negative_img)
            
        return anchor, positive, negative

def get_dataloader(data_dir, batch_size=32, limit=None):
    base = Path(data_dir)
    images_dir = base / "rebrickable" / "images" / "elements"
    bg_path = base / "backgrounds" / "conveyor_belt.jpg"
    
    dataset = LegoTripletDataset(images_dir, bg_path, limit=limit)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) # workers=0 avoids multiprocessing issues in some envs
