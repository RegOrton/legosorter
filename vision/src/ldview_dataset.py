"""
On-the-fly LDView dataset for triplet learning.
Generates images in real-time during training instead of pre-generating them.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from pathlib import Path
import logging
import os

from ldview_renderer import LDViewRenderer

logger = logging.getLogger(__name__)


class LDViewTripletDataset(Dataset):
    """
    Dataset that generates LDView renders on-the-fly during training.
    Each sample returns a triplet: (anchor, positive, negative)
    """

    def __init__(
        self,
        dat_dir: Path,
        samples_per_epoch: int = 1000,
        background_path: str = None,
        output_size: tuple = (336, 336)  # Balanced resolution: 1.5x final size for quality
    ):
        """
        Args:
            dat_dir: Directory containing .dat files
            samples_per_epoch: Number of triplets to generate per epoch
            background_path: Optional background image path
            output_size: Output image size
        """
        self.dat_dir = Path(dat_dir)
        self.samples_per_epoch = samples_per_epoch
        self.output_size = output_size

        # Find all .dat files
        self.dat_files = list(self.dat_dir.glob("*.dat"))

        if len(self.dat_files) == 0:
            raise ValueError(f"No .dat files found in {dat_dir}")

        logger.info(f"Found {len(self.dat_files)} .dat files in {dat_dir}")

        # Initialize LDView renderer
        self.renderer = LDViewRenderer(
            ldview_path="ldview",
            ldraw_dir=os.environ.get('LDRAWDIR'),
            output_size=output_size,
            background_path=background_path
        )

        # Image transforms (resize to 224x224 for MobileNetV3, with normalization)
        # We render at higher res (448x448) then downsample for better quality
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),  # Downsample with antialiasing
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        """
        Generate a triplet on-the-fly.

        Returns:
            (anchor, positive, negative) tuple of tensors
        """
        import time
        triplet_start = time.time()

        # Select random .dat file for anchor and positive
        anchor_dat = random.choice(self.dat_files)

        # Generate anchor image with random viewpoint
        anchor_start = time.time()
        anchor_img = self.renderer.generate_sample(
            str(anchor_dat),
            apply_augmentations=True
        )
        anchor_time = time.time() - anchor_start

        # Generate positive image (same part, different viewpoint/augmentation)
        positive_start = time.time()
        positive_img = self.renderer.generate_sample(
            str(anchor_dat),
            apply_augmentations=True
        )
        positive_time = time.time() - positive_start

        # Select different .dat file for negative
        negative_dat = random.choice([f for f in self.dat_files if f != anchor_dat])

        # Generate negative image
        negative_start = time.time()
        negative_img = self.renderer.generate_sample(
            str(negative_dat),
            apply_augmentations=True
        )
        negative_time = time.time() - negative_start

        # Convert BGR to RGB (OpenCV returns BGR, torchvision expects RGB)
        transform_start = time.time()
        anchor_img = anchor_img[:, :, ::-1].copy()
        positive_img = positive_img[:, :, ::-1].copy()
        negative_img = negative_img[:, :, ::-1].copy()

        # Apply transforms
        anchor = self.transform(anchor_img)
        positive = self.transform(positive_img)
        negative = self.transform(negative_img)
        transform_time = time.time() - transform_start

        total_time = time.time() - triplet_start

        # Log timing every 10 samples
        if idx % 10 == 0:
            logger.info(f"[DataGen {idx}] Total={total_time:.3f}s (Anchor={anchor_time:.3f}s, Pos={positive_time:.3f}s, Neg={negative_time:.3f}s, Transform={transform_time:.3f}s)")

        return anchor, positive, negative


def get_ldview_dataloader(
    dat_dir: Path,
    batch_size: int = 32,
    samples_per_epoch: int = 1000,
    background_path: str = None,
    num_workers: int = 0,  # 0 for single-threaded (safer with LDView subprocess calls)
    limit: int = None
):
    """
    Create a DataLoader for on-the-fly LDView rendering.

    Args:
        dat_dir: Directory containing .dat files
        batch_size: Batch size
        samples_per_epoch: Number of triplets per epoch
        background_path: Optional background image
        num_workers: Number of worker processes (recommend 0 for LDView)
        limit: Optional limit on number of samples

    Returns:
        DataLoader instance
    """
    if limit:
        samples_per_epoch = min(samples_per_epoch, limit)

    dataset = LDViewTripletDataset(
        dat_dir=dat_dir,
        samples_per_epoch=samples_per_epoch,
        background_path=background_path
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Not needed since we're randomly generating
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader
