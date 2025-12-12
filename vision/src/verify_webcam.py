import cv2
import sys
import os

def check_camera(index):
    print(f"Checking camera index {index}...")
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"FAILED: Could not open camera {index}")
        return False
    
    ret, frame = cap.read()
    if not ret:
        print(f"FAILED: Could not read frame from camera {index}")
        cap.release()
        return False
    
    output_path = f"/app/output/webcam_test_{index}.png"
    cv2.imwrite(output_path, frame)
    print(f"SUCCESS: Captured frame from camera {index} to {output_path}")
    
    cap.release()
    return True

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs("/app/output", exist_ok=True)
    
    success = False
    # Check index 0 and 1
    if check_camera(0):
        success = True
    if check_camera(1):
        success = True
        
    if success:
        print("VERIFICATION SUCCESSFUL: At least one camera working.")
        sys.exit(0)
    else:
        print("VERIFICATION FAILED: No cameras working.")
        sys.exit(1)
