import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import random
from pathlib import Path
from synthesizer import LegoSynthesizer
import cv2
import numpy as np

def get_training_transforms():
    """
    Returns augmentation pipeline optimized for sim-to-real transfer.
    Addresses domain gap between CGI renders and real webcam images.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        # Color jitter to handle lighting variations and color temperature shifts
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1  # Small hue shift - LEGO colors are specific
            )
        ], p=0.8),
        # Occasionally convert to grayscale to force shape-based learning
        transforms.RandomGrayscale(p=0.1),
        # Simulate camera focus/motion blur
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        ], p=0.3),
        # Random erasing to simulate occlusion/debris
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        # Normalize with ImageNet stats (required for pretrained backbone)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class LegoTripletDataset(Dataset):
    def __init__(self, images_dir, background_path, synthesizer=None, transform=None, limit=None):
        """
        Args:
            images_dir (Path): Directory containing clean element images.
            background_path (Path): Path to background texture OR directory of backgrounds.
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

        # Use enhanced transforms by default for better sim-to-real transfer
        self.transform = transform or get_training_transforms()

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
    bg_dir = base / "backgrounds"

    # Use backgrounds directory (supports multiple backgrounds)
    # Falls back to single file if directory doesn't exist
    if bg_dir.is_dir():
        bg_path = bg_dir
    else:
        bg_path = base / "backgrounds" / "conveyor_belt.jpg"

    dataset = LegoTripletDataset(images_dir, bg_path, limit=limit)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # workers=0 avoids multiprocessing issues in some envs
