"""
Build reference embedding database for nearest-neighbor classification.

This script generates embeddings for each LEGO part and saves them as a reference database.
"""

import torch
import numpy as np
from pathlib import Path
from model import LegoEmbeddingNet
from ldview_renderer import LDViewRenderer
from torchvision import transforms
import cv2
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_reference_database(
    dat_dir: Path,
    model_path: Path,
    output_path: Path,
    samples_per_part: int = 10,
    use_calibration_bg: bool = True
):
    """
    Build reference embedding database from .dat files.

    Args:
        dat_dir: Directory containing .dat files
        model_path: Path to trained embedding model
        output_path: Where to save the reference database
        samples_per_part: Number of views to generate per part
        use_calibration_bg: Use calibration background if available
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained embedding model
    model = LegoEmbeddingNet(embedding_dim=128).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Loaded model from {model_path}")

    # Image preprocessing (same as inference)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Get calibration background path if available
    background_path = None
    if use_calibration_bg:
        base_path = Path("/app/output") if Path("/app/output").exists() else Path(__file__).parent.parent / "output"
        calib_path = base_path / "calibration_bg.npy"
        if calib_path.exists():
            # Save as temp JPEG for renderer
            bg_frame = np.load(str(calib_path))
            temp_bg_path = base_path / "calibration_bg_temp.jpg"
            cv2.imwrite(str(temp_bg_path), bg_frame.astype(np.uint8))
            background_path = str(temp_bg_path)
            logger.info(f"Using calibration background: {bg_frame.shape}")

    # Initialize renderer
    renderer = LDViewRenderer(
        ldview_path="ldview",
        ldraw_dir=os.environ.get('LDRAWDIR'),
        output_size=(224, 224),
        background_path=background_path
    )

    # Get all .dat files
    dat_files = list(dat_dir.glob("*.dat"))
    logger.info(f"Found {len(dat_files)} parts")

    # Build reference database
    reference_db = {
        'part_ids': [],
        'embeddings': [],
        'metadata': []
    }

    for dat_file in dat_files:
        part_id = dat_file.stem
        logger.info(f"Processing {part_id}...")

        part_embeddings = []

        # Generate multiple views for each part
        for i in range(samples_per_part):
            try:
                # Render with random viewpoint
                img = renderer.generate_sample(str(dat_file), apply_augmentations=True)

                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Resize to 224x224
                img_resized = cv2.resize(img_rgb, (224, 224))

                # Transform and get embedding
                input_tensor = transform(img_resized).unsqueeze(0).to(device)

                with torch.no_grad():
                    embedding = model(input_tensor)
                    embedding = embedding.cpu().numpy().flatten()

                part_embeddings.append(embedding)

            except Exception as e:
                logger.warning(f"Failed to process {part_id} view {i}: {e}")
                continue

        if len(part_embeddings) > 0:
            # Average embeddings for this part
            avg_embedding = np.mean(part_embeddings, axis=0)

            reference_db['part_ids'].append(part_id)
            reference_db['embeddings'].append(avg_embedding)
            reference_db['metadata'].append({
                'num_views': len(part_embeddings),
                'dat_file': str(dat_file)
            })
            logger.info(f"  Added {part_id} ({len(part_embeddings)} views)")

    # Convert to numpy arrays
    reference_db['embeddings'] = np.array(reference_db['embeddings'])

    # Save database
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(output_path),
        part_ids=reference_db['part_ids'],
        embeddings=reference_db['embeddings'],
        metadata=reference_db['metadata']
    )

    logger.info(f"Saved reference database to {output_path}")
    logger.info(f"Database contains {len(reference_db['part_ids'])} parts")
    logger.info(f"Embedding shape: {reference_db['embeddings'].shape}")

    return reference_db


if __name__ == "__main__":
    # Paths
    dat_dir = Path("/app/data/../input/dat_files") if Path("/app/data").exists() else Path(__file__).parent.parent / "input" / "dat_files"
    model_path = Path("/app/output/models/lego_embedder_final.pth") if Path("/app/output").exists() else Path(__file__).parent.parent / "output" / "models" / "lego_embedder_final.pth"
    output_path = Path("/app/output/reference_db.npz") if Path("/app/output").exists() else Path(__file__).parent.parent / "output" / "reference_db.npz"

    # Build database
    build_reference_database(
        dat_dir=dat_dir,
        model_path=model_path,
        output_path=output_path,
        samples_per_part=10,
        use_calibration_bg=True
    )
