import os
import requests
import gzip
import shutil
import logging
from pathlib import Path

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
BASE_URL = "https://cdn.rebrickable.com/media/downloads"
TABLES = [
    "themes",
    "colors",
    "part_categories",
    "parts",
    "part_relationships",
    "elements",
    "minifigs",
    "inventories",
    "inventory_parts",
    "inventory_sets",
    "inventory_minifigs"
]

# Define data directory relative to this script
# vision/src/ingest_rebrickable.py -> vision/data/rebrickable
CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = CURRENT_DIR.parent / "data" / "rebrickable"

def download_file(url, dest_path):
    """Downloads a file from a URL to a destination path."""
    try:
        logger.info(f"Downloading {url}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False

def decompress_file(gzip_path, output_path):
    """Decompresses a GZIP file."""
    try:
        logger.info(f"Extracting {gzip_path} to {output_path}...")
        with gzip.open(gzip_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return True
    except Exception as e:
        logger.error(f"Failed to decompress {gzip_path}: {e}")
        return False

import csv
import time

def download_element_images(limit=50):
    """Downloads element images from Rebrickable CDN."""
    elements_file = DATA_DIR / "elements.csv"
    images_dir = DATA_DIR / "images" / "elements"
    
    if not elements_file.exists():
        logger.error(f"Elements file not found at {elements_file}. Cannot download images.")
        return

    images_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting image download (limit={limit})...")
    
    count = 0
    with open(elements_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if count >= limit:
                break
                
            element_id = row['element_id']
            image_url = f"https://cdn.rebrickable.com/media/parts/elements/{element_id}.jpg"
            save_path = images_dir / f"{element_id}.jpg"
            
            if save_path.exists():
                count += 1
                continue
                
            try:
                # logger.info(f"Downloading image for element {element_id}...")
                with requests.get(image_url, stream=True) as r:
                    if r.status_code == 200:
                        with open(save_path, 'wb') as f_out:
                            for chunk in r.iter_content(chunk_size=8192):
                                f_out.write(chunk)
                        count += 1
                        time.sleep(0.1)  # Rate limiting
                    else:
                        # Some elements might not have images or URL is different
                        # logger.warning(f"Image not found for {element_id} (Status {r.status_code})")
                        pass
            except Exception as e:
                logger.error(f"Failed to download image {element_id}: {e}")

    logger.info(f"Downloaded {count} images to {images_dir}")

def main():
    # Ensure output directory exists
    if not DATA_DIR.exists():
        logger.info(f"Creating directory {DATA_DIR}")
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    for table in TABLES:
        filename = f"{table}.csv.gz"
        url = f"{BASE_URL}/{filename}"
        gz_file_path = DATA_DIR / filename
        csv_file_path = DATA_DIR / f"{table}.csv"
        
        # Check if CSV already exists to avoid re-downloading every time
        if csv_file_path.exists():
             logger.info(f"{table}.csv already exists. Skipping download.")
             continue

        # 1. Download
        if download_file(url, gz_file_path):
            # 2. Decompress
            if decompress_file(gz_file_path, csv_file_path):
                pass
            else:
                logger.error(f"Skipping decompression for {table}")
        else:
            logger.error(f"Skipping {table}")

    # Download Images
    # Warning: Downloading ALL images is huge. Set a limit appropriately.
    # There are >50k elements.
    download_element_images(limit=50)

    logger.info("Ingestion Complete. Files saved to " + str(DATA_DIR))

if __name__ == "__main__":
    main()
