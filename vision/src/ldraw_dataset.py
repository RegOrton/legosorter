"""
Dataset for training with multi-view LDraw renders.
Uses true 3D viewpoint variation for robust angle-invariant recognition.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_ldraw_training_transforms():
    """
    Returns augmentation pipeline for LDraw renders.
    Since LDraw renders are already multi-view, we focus on sim-to-real augmentation.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        # Color jitter for lighting/color variation
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.08
            )
        ], p=0.8),
        # Grayscale to force shape learning
        transforms.RandomGrayscale(p=0.1),
        # Blur for camera simulation
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        ], p=0.2),
        # Normalize for pretrained backbone
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class LDrawTripletDataset(Dataset):
    """
    Dataset that creates triplets from multi-view LDraw renders.

    Image naming convention: {part_id}_c{color}_v{view}.jpg
    Example: 3001_c4_v05.jpg (part 3001, color 4, view 5)

    Triplet formation:
    - Anchor: Image of part X
    - Positive: Different image of SAME part X (different view/color)
    - Negative: Image of DIFFERENT part Y
    """

    def __init__(self, images_dir: Path, transform=None, limit: int = None):
        """
        Args:
            images_dir: Directory containing LDraw render images
            transform: Transforms to apply
            limit: Limit number of parts (for testing)
        """
        self.images_dir = Path(images_dir)
        self.transform = transform or get_ldraw_training_transforms()

        # Parse all images and group by part ID
        self.part_images = defaultdict(list)
        self._parse_images()

        # Filter to parts with multiple images (needed for positive sampling)
        self.parts = [p for p in self.part_images.keys()
                      if len(self.part_images[p]) >= 2]

        if limit:
            self.parts = self.parts[:limit]

        if len(self.parts) < 2:
            raise ValueError(f"Need at least 2 parts with multiple images. Found {len(self.parts)}")

        logger.info(f"Loaded {len(self.parts)} parts with {sum(len(self.part_images[p]) for p in self.parts)} images")

    def _parse_images(self):
        """Parse image filenames to extract part IDs."""
        # Pattern: {part_id}_c{color}_v{view}.jpg
        pattern = re.compile(r'^(.+?)_c\d+_v\d+\.jpg$')

        for img_path in self.images_dir.glob("*.jpg"):
            match = pattern.match(img_path.name)
            if match:
                part_id = match.group(1)
                self.part_images[part_id].append(img_path)

    def __len__(self):
        return len(self.parts) * 10  # Virtual expansion for training

    def _load_image(self, path: Path) -> np.ndarray:
        """Load and preprocess an image."""
        img = cv2.imread(str(path))
        if img is None:
            # Return placeholder if image fails to load
            return np.zeros((224, 224, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        # Map index to actual part
        part_idx = idx % len(self.parts)
        anchor_part = self.parts[part_idx]

        # Get images for anchor part
        anchor_images = self.part_images[anchor_part]

        # Select anchor and positive (different images of same part)
        anchor_path, positive_path = random.sample(anchor_images, 2)

        # Select negative (different part)
        negative_part = anchor_part
        while negative_part == anchor_part:
            negative_part = random.choice(self.parts)
        negative_path = random.choice(self.part_images[negative_part])

        # Load images
        anchor_img = self._load_image(anchor_path)
        positive_img = self._load_image(positive_path)
        negative_img = self._load_image(negative_path)

        # Apply transforms
        anchor = self.transform(anchor_img)
        positive = self.transform(positive_img)
        negative = self.transform(negative_img)

        return anchor, positive, negative


class LDrawClassificationDataset(Dataset):
    """
    Dataset for classification training (cross-entropy loss).
    Returns (image, label) pairs.
    """

    def __init__(self, images_dir: Path, transform=None, limit: int = None):
        self.images_dir = Path(images_dir)
        self.transform = transform or get_ldraw_training_transforms()

        # Parse images and group by part
        self.part_images = defaultdict(list)
        self._parse_images()

        # Create part to label mapping
        self.parts = sorted(self.part_images.keys())
        if limit:
            self.parts = self.parts[:limit]

        self.part_to_label = {p: i for i, p in enumerate(self.parts)}

        # Flatten all images
        self.samples = []
        for part in self.parts:
            for img_path in self.part_images[part]:
                self.samples.append((img_path, self.part_to_label[part]))

        logger.info(f"Loaded {len(self.samples)} images for {len(self.parts)} classes")

    def _parse_images(self):
        pattern = re.compile(r'^(.+?)_c\d+_v\d+\.jpg$')
        for img_path in self.images_dir.glob("*.jpg"):
            match = pattern.match(img_path.name)
            if match:
                part_id = match.group(1)
                self.part_images[part_id].append(img_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        return img, label

    @property
    def num_classes(self):
        return len(self.parts)

    @property
    def class_names(self):
        return self.parts


def get_ldraw_dataloader(data_dir: Path, batch_size: int = 32,
                         mode: str = 'triplet', limit: int = None) -> DataLoader:
    """
    Create a dataloader for LDraw training.

    Args:
        data_dir: Path to ldraw_renders directory
        batch_size: Batch size
        mode: 'triplet' for metric learning, 'classification' for cross-entropy
        limit: Limit number of parts

    Returns:
        DataLoader
    """
    if mode == 'triplet':
        dataset = LDrawTripletDataset(data_dir, limit=limit)
    else:
        dataset = LDrawClassificationDataset(data_dir, limit=limit)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )


if __name__ == "__main__":
    # Test the dataset
    data_dir = Path(__file__).resolve().parent.parent / "data" / "ldraw_renders"

    if not data_dir.exists():
        print(f"LDraw renders not found at {data_dir}")
        print("Run: python generate_ldraw_dataset.py first")
        exit(1)

    print("Testing LDrawTripletDataset...")
    dataset = LDrawTripletDataset(data_dir)
    print(f"Dataset size: {len(dataset)}")
    print(f"Parts: {len(dataset.parts)}")

    # Get a sample
    anchor, positive, negative = dataset[0]
    print(f"Anchor shape: {anchor.shape}")
    print(f"Positive shape: {positive.shape}")
    print(f"Negative shape: {negative.shape}")

    print("\nTesting DataLoader...")
    dataloader = get_ldraw_dataloader(data_dir, batch_size=8)
    batch = next(iter(dataloader))
    print(f"Batch anchor shape: {batch[0].shape}")

    print("\nTesting ClassificationDataset...")
    cls_dataset = LDrawClassificationDataset(data_dir)
    print(f"Classes: {cls_dataset.num_classes}")
    img, label = cls_dataset[0]
    print(f"Image shape: {img.shape}, Label: {label}")

    print("\nAll tests passed!")
