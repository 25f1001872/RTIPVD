import cv2
import numpy as np
from config import WHITE_HSV_LO, WHITE_HSV_HI

class LaneTracker:
    def __init__(self, max_features=2500, min_features=15):
        self.max_features = max_features
        self.min_features = min_features
        
        self.prev_gray = None
        self.prev_pts = None

    def _get_lane_mask(self, frame, boxes):
        """
        Creates a tightly filtered binary mask targeting ONLY road lines.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, WHITE_HSV_LO, WHITE_HSV_HI)

        # Remove upper half containing sky and buildings
        h, w = frame.shape[:2]
        horizon = int(h * 0.5)
        mask[0:horizon, :] = 0

        # Enforce edges (prevent flat sun glares from being lanes)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        mask = cv2.bitwise_and(mask, edges)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Erase areas covered by car bounding boxes
        if boxes is not None:
            for b in boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                pad = 12
                mask[
                    max(0, y1 - pad): min(h, y2 + pad),
                    max(0, x1 - pad): min(w, x2 + pad)
                ] = 0
                
        return mask

    def compute_ego_motion(self, frame, current_gray, boxes):
        """
        Calculates homography shift (camera roll on road surface) by optical flow of lane marks.
        """
        mask = self._get_lane_mask(frame, boxes)
        lane_px_count = np.count_nonzero(mask)

        H_ego, dx, dy = None, 0.0, 0.0

        if self.prev_gray is not None:
            # Re-seed if points are lost
            if self.prev_pts is None or len(self.prev_pts) < self.min_features:
                self.prev_pts = cv2.goodFeaturesToTrack(
                    self.prev_gray,
                    maxCorners=self.max_features,
                    qualityLevel=0.01,
                    minDistance=10,
                    mask=mask if lane_px_count >= self.min_features else None
                )

            # Perform Lucas-Kanade optical flow
            if self.prev_pts is not None and len(self.prev_pts) >= 4:
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, current_gray, self.prev_pts, None)
                
                good_prev = self.prev_pts[status.ravel() == 1]
                good_curr = curr_pts[status.ravel() == 1]

                if len(good_prev) >= 4:
                    H_ego, _ = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 3.0)
                    if H_ego is not None:
                        dx = float(H_ego[0, 2])
                        dy = float(H_ego[1, 2])

                # Carry over good points to next frame
                self.prev_pts = good_curr.reshape(-1, 1, 2) if len(good_curr) >= self.min_features else None
            else:
                self.prev_pts = None

        self.prev_gray = current_gray.copy()
        
        return H_ego, dx, dy, lane_px_count
