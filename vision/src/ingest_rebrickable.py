import os
import requests
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load Environment Variables
load_dotenv()
API_KEY = os.getenv("REBRICKABLE_API_KEY")
BASE_URL = "https://rebrickable.com/api/v3/lego"
DATA_DIR = Path("/app/data/reference")

def check_api_key():
    if not API_KEY or API_KEY == "paste_your_key_here":
        logger.error("REBRICKABLE_API_KEY is missing or invalid. Please check your .env file.")
        return False
    return True

def get_headers():
    return {"Authorization": f"key {API_KEY}"}

def fetch_common_parts(limit=20):
    """
    Fetches a list of common parts. 
    In a real scenario, we'd maybe fetch based on a specific set or year.
    Here we just fetch the 'elements' endpoint or 'parts' endpoint.
    """
    url = f"{BASE_URL}/parts/"
    params = {
        "page_size": limit,
        "ordering": "-year_from" # Get newer parts first? or generic?
    }
    logger.info(f"Fetching top {limit} parts from {url}...")
    
    try:
        response = requests.get(url, headers=get_headers(), params=params)
        response.raise_for_status()
        return response.json()['results']
    except Exception as e:
        logger.error(f"Failed to fetch parts: {e}")
        return []

def download_image(part_num, image_url):
    if not image_url:
        logger.warning(f"No image URL for part {part_num}")
        return

    filename = DATA_DIR / f"{part_num}.jpg"
    if filename.exists():
        logger.info(f"Image for {part_num} already exists.")
        return

    try:
        logger.info(f"Downloading {part_num} from {image_url}")
        img_data = requests.get(image_url).content
        with open(filename, 'wb') as handler:
            handler.write(img_data)
        time.sleep(0.5) # Be nice to the API
    except Exception as e:
        logger.error(f"Failed to download image for {part_num}: {e}")

def main():
    if not check_api_key():
        return

    # Ensure output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Fetch Parts
    parts = fetch_common_parts(limit=50)
    
    # 2. Download Images
    for part in parts:
        part_num = part['part_num']
        name = part['name']
        img_url = part['part_img_url']
        
        logger.info(f"Processing {part_num}: {name}")
        download_image(part_num, img_url)

    logger.info("Ingestion Complete.")

if __name__ == "__main__":
    main()
