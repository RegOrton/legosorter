import cv2
import random
from pathlib import Path
from synthesizer import LegoSynthesizer

def test_synthesis():
    base_dir = Path(__file__).resolve().parent.parent # vision/
    bg_path = base_dir / "data" / "backgrounds" / "conveyor_belt.jpg"
    images_dir = base_dir / "data" / "rebrickable" / "images" / "elements"
    output_dir = base_dir / "output" / "debug"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Initializing Synthesizer with background: {bg_path}")
    if not bg_path.exists():
        print("Background not found! Make sure you created it.")
        return

    synth = LegoSynthesizer(bg_path, output_size=(224, 224))
    
    # Get list of images
    image_files = list(images_dir.glob("*.jpg"))
    if not image_files:
        print("No images found! Run ingest_rebrickable.py first.")
        return
    
    # Generate 10 samples
    samples_to_gen = 10
    print(f"Generating {samples_to_gen} samples...")
    
    for i in range(samples_to_gen):
        part_img = random.choice(image_files)
        print(f"Processing {part_img.name}...")
        
        result = synth.generate_sample(part_img)
        
        if result is not None:
             out_path = output_dir / f"synth_{i}_{part_img.name}"
             cv2.imwrite(str(out_path), result)
             print(f"Saved {out_path}")
        else:
             print("Failed to generate sample")

if __name__ == "__main__":
    test_synthesis()
