import numpy as np

# -------------------------------------------------------------
# Configuration Settings
# -------------------------------------------------------------

# File Paths
MODEL_PATH = "weights/best.pt"
VIDEO_SOURCE = "ITS/d1.mp4"
TRACKER_CONFIG = "bytetrack.yaml"

# Timing Thresholds (in seconds)
PARKED_SECONDS = 5.0
STALE_TRACK_SECONDS = 2.0
MIN_VISIBLE_SECONDS = 0.6

# Motion Calibration
CALIBRATION_FRAMES = 60
MIN_STATIONARY_THRESHOLD = 3.0
MAX_STATIONARY_THRESHOLD = 20.0

# Detection Bounding Box constraints
MIN_BBOX_HEIGHT = 120   # Avoid too far (pixel motion near zero)
MAX_BBOX_HEIGHT = 800   # Avoid too near (huge distorted boxes)

# Center Smoothing & Tracking Tolerances
CENTROID_EMA_ALPHA = 0.35
FORGIVENESS_FRAMES = 10

# Optical Flow / Lane Settings
MAX_LANE_FEATURES = 2500
MIN_LANE_FEATURES = 15

# HSV constraints to detect white lanes (lower V for shadows)
WHITE_HSV_LO = np.array([0, 0, 100], dtype=np.uint8)
WHITE_HSV_HI = np.array([180, 60, 255], dtype=np.uint8)

# Target Objects
VEHICLE_LABELS = {
    "motorbike", "motorcycle", "scooter", "car", "suv", "van",
    "pickup", "truck", "bus", "minibus", "tractor", "trailer",
    "rickshaw", "autorickshaw", "ambulance", "firetruck"
}
