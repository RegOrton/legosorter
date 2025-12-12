"""
Test script to verify webcam access from Docker container.

This script tests the WebcamClient to ensure it can successfully
connect to the webcam server running on the host machine.
"""

import cv2
import sys
import os
from webcam_client import WebcamClient

def test_webcam():
    """Test webcam access via HTTP."""
    print("=" * 60)
    print("Testing Webcam Access from Docker Container")
    print("=" * 60)
    
    # Create webcam client
    print("\n1. Connecting to webcam server...")
    cap = WebcamClient("http://host.docker.internal:5000")
    
    if not cap.isOpened():
        print("❌ Failed to connect to webcam server")
        print("\nTroubleshooting:")
        print("  1. Make sure webcam_server.py is running on the host:")
        print("     python webcam_server.py")
        print("  2. Check that the server is accessible from the container")
        print("  3. Verify firewall settings allow connections on port 5000")
        return False
    
    print("✓ Successfully connected to webcam server")
    
    # Test reading frames
    print("\n2. Testing frame capture...")
    success_count = 0
    fail_count = 0
    
    for i in range(10):
        ret, frame = cap.read()
        if ret and frame is not None:
            success_count += 1
            if i == 0:  # Print info for first frame
                print(f"✓ Frame captured successfully")
                print(f"  - Shape: {frame.shape}")
                print(f"  - Data type: {frame.dtype}")
                print(f"  - Size: {frame.nbytes} bytes")
        else:
            fail_count += 1
    
    print(f"\nCapture results: {success_count}/10 successful, {fail_count}/10 failed")
    
    # Save a test frame
    if success_count > 0:
        print("\n3. Saving test frame...")
        ret, frame = cap.read()
        if ret:
            output_path = "/app/output/test_webcam_frame.jpg"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, frame)
            print(f"✓ Test frame saved to {output_path}")
        else:
            print("❌ Failed to capture frame for saving")
    
    # Clean up
    cap.release()
    print("\n4. Webcam client released")
    
    print("\n" + "=" * 60)
    if success_count >= 8:
        print("✓ WEBCAM TEST PASSED")
        print("=" * 60)
        return True
    else:
        print("❌ WEBCAM TEST FAILED")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = test_webcam()
    sys.exit(0 if success else 1)
