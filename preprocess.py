import cv2
import numpy as np

def clean_frame(frame, scale=1.0):
    """
    Cleans and optimizes the raw frame before pushing to detection models.
    Important for processing heavy footage on lightweight devices (Raspberry Pi).
    """
    # 1. Resize if required to save compute
    if scale != 1.0:
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        
    # 2. Extract grayscale for motion tracking and brightness analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    
    # 3. Handle low-light scenarios ("Night Mode")
    if avg_brightness < 60:
        # Applying CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # to boost edges and textures without increasing global noise.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # We can also equalize the color frame's luminance channel if YOLO struggles
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        # frame = cv2.cvtColor(hsv, cv2.HSV2BGR)

    return frame, gray, avg_brightness
