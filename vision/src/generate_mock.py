import cv2
import numpy as np
import os

def create_mock_video(filename="vision/input/mock_conveyor.mp4", width=640, height=480, duration_sec=5, fps=30):
    """
    Creates a video of a blue rectangle moving across a black background.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    frames = duration_sec * fps
    
    # Brick properties
    brick_w, brick_h = 100, 60
    color = (255, 255, 255) # White (B,G,R) for high contrast
    
    start_x = -brick_w
    end_x = width + brick_w
    step = (end_x - start_x) / frames
    
    current_x = start_x
    y = (height - brick_h) // 2

    print(f"Generating mock video: {filename} ({frames} frames)")

    for _ in range(frames):
        # Black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw Brick
        top_left = (int(current_x), int(y))
        bottom_right = (int(current_x + brick_w), int(y + brick_h))
        cv2.rectangle(frame, top_left, bottom_right, color, -1)
        
        out.write(frame)
        current_x += step

    out.release()
    print("Video generation complete.")

if __name__ == "__main__":
    # Ensure directory exists
    filename = "/app/input/mock_conveyor.mp4"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    create_mock_video(filename=filename)
