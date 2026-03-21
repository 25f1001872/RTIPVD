import cv2
import numpy as np
from config import *

class ParkingAnalyzer:
    def __init__(self, fps):
        self.parked_frames = int(PARKED_SECONDS * fps)
        self.min_visible_frames = int(MIN_VISIBLE_SECONDS * fps)
        self.stale_track_frames = int(STALE_TRACK_SECONDS * fps)
        
        self.track_states = {}
        
        # Adaptive Thresholding state
        self.stationary_threshold = MIN_STATIONARY_THRESHOLD
        self.calibration_samples = []

    def analyze_vehicle(self, track_id, cx, cy, bbox_h, frame_idx, H_ego, lane_dx, lane_dy):
        """
        Determines if a tracked vehicle is MOVING, PARKED, or OUT_OF_RANGE.
        """
        if bbox_h < MIN_BBOX_HEIGHT or bbox_h > MAX_BBOX_HEIGHT:
            return "OUT_OF_RANGE", 0.0

        state = self.track_states.get(
            track_id, 
            {
                "scx": cx, "scy": cy, 
                "stationary_f": 0, "jitter_f": 0, 
                "visible_f": 0, "last_seen": frame_idx
            }
        )

        # Exponential Moving Average for Centers (dampens YOLO bounding box jitter)
        scx = CENTROID_EMA_ALPHA * cx + (1 - CENTROID_EMA_ALPHA) * state["scx"]
        scy = CENTROID_EMA_ALPHA * cy + (1 - CENTROID_EMA_ALPHA) * state["scy"]

        # Calculate difference comparing against camera ego-motion 
        if H_ego is not None:
            pt = np.array([[[state["scx"], state["scy"]]]], dtype=np.float32)
            expected = cv2.perspectiveTransform(pt, H_ego)[0][0]
            dx = scx - float(expected[0])
            dy = scy - float(expected[1])
        else:
            dx = (scx - state["scx"]) - lane_dx
            dy = (scy - state["scy"]) - lane_dy

        motion_magnitude = float(np.hypot(dx, dy))

        # Update jitter/stationary tolerance counters
        if motion_magnitude < self.stationary_threshold:
            state["stationary_f"] += 1
            state["jitter_f"] = 0
        else:
            state["jitter_f"] += 1
            if state["jitter_f"] > FORGIVENESS_FRAMES:
                state["stationary_f"] = 0
                state["jitter_f"] = 0

        state["scx"] = scx
        state["scy"] = scy
        state["visible_f"] += 1
        state["last_seen"] = frame_idx

        # Adaptive Threshold Calibration 
        if frame_idx <= CALIBRATION_FRAMES:
            self.calibration_samples.append(motion_magnitude)
        elif frame_idx == CALIBRATION_FRAMES + 1 and self.calibration_samples:
            p80 = float(np.percentile(self.calibration_samples, 80))
            self.stationary_threshold = float(np.clip(p80 * 1.5, MIN_STATIONARY_THRESHOLD, MAX_STATIONARY_THRESHOLD))

        self.track_states[track_id] = state

        # Check thresholds
        is_parked = (state["visible_f"] >= self.min_visible_frames and 
                     state["stationary_f"] >= self.parked_frames)

        return "PARKED" if is_parked else "MOVING", motion_magnitude

    def purge_stale_tracks(self, frame_idx):
        """
        Cleans memory of vehicles that haven't been seen recently.
        """
        stale_ids = [tid for tid, state in self.track_states.items() 
                     if frame_idx - state["last_seen"] > self.stale_track_frames]
        for tid in stale_ids:
            del self.track_states[tid]
