"""
Test script to verify Camera class integration with HTTP webcam.
"""

import logging
import sys
import os

# Set environment variable for HTTP webcam
os.environ['USE_HTTP_WEBCAM'] = 'true'

from camera import Camera

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_camera_http():
    """Test Camera class with HTTP webcam."""
    print("=" * 60)
    print("Testing Camera Class with HTTP Webcam")
    print("=" * 60)
    
    try:
        # Create camera instance
        print("\n1. Creating Camera instance...")
        cam = Camera(source=0, width=640, height=480)
        
        # Start camera
        print("\n2. Starting camera...")
        cam.start()
        
        # Get a few frames
        print("\n3. Capturing frames...")
        success_count = 0
        for i in range(5):
            ret, frame = cam.get_frame()
            if ret and frame is not None:
                success_count += 1
                print(f"   Frame {i+1}: ✓ {frame.shape}")
            else:
                print(f"   Frame {i+1}: ✗ Failed")
        
        # Release camera
        print("\n4. Releasing camera...")
        cam.release()
        
        print("\n" + "=" * 60)
        if success_count >= 4:
            print("✓ CAMERA HTTP INTEGRATION TEST PASSED")
            print("=" * 60)
            return True
        else:
            print("❌ CAMERA HTTP INTEGRATION TEST FAILED")
            print("=" * 60)
            return False
            
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_camera_http()
    sys.exit(0 if success else 1)
