import cv2
import numpy as np
from ultralytics import YOLO

# -------------------------
# Config
# -------------------------
MODEL_PATH = "weights/best.pt"
VIDEO_SOURCE = "ITS/d1.mp4"
TRACKER_CONFIG = "bytetrack.yaml"

PARKED_SECONDS = 5.0
STALE_TRACK_SECONDS = 2.0
MIN_VISIBLE_SECONDS = 0.6

# Auto calibration from early frames (lane-relative motion samples).
CALIBRATION_FRAMES = 60
MIN_STATIONARY_THRESHOLD = 3.0
MAX_STATIONARY_THRESHOLD = 20.0

# EMA smoothing on centroid — dampens YOLO bbox jitter before motion is used.
CENTROID_EMA_ALPHA = 0.35

# Forgiveness: tolerate this many consecutive "moving" frames before resetting
# the stationary counter.  Handles single-frame jitter spikes.
FORGIVENESS_FRAMES = 10

# ---- Lane-based ego-motion ----
# LK feature tracking is seeded ONLY from white/yellow lane pixels.
# Their motion == road-surface motion == exact ego-motion reference.
MAX_LANE_FEATURES = 2500
MIN_LANE_FEATURES = 15      # below this, fall back to generic bg features

# HSV ranges for white lane markings. 
# V threshold lowered to 100 to catch lanes in deep shadows. Add slight S tolerance.
WHITE_HSV_LO = np.array([0,   0, 100], dtype=np.uint8)
WHITE_HSV_HI = np.array([180, 60, 255], dtype=np.uint8)

# HSV ranges for yellow lane markings.
#YELLOW_HSV_LO = np.array([15,  80, 100], dtype=np.uint8)
#YELLOW_HSV_HI = np.array([35, 255, 255], dtype=np.uint8)

# Show lane mask overlay (green tint) — set False for clean output.
DEBUG_LANE_OVERLAY = True

# ---- Distance filter ----
# Vehicles far from the camera have tiny bounding boxes and their centroid
# motion in pixels drops to near-zero even when they are actually driving.
# Vehicles too close to the camera can have very large bounding boxes and skewed perspective.
# We skip the parked logic for any bbox whose height is outside this range.
MIN_BBOX_HEIGHT = 120   # pixels (too far)
MAX_BBOX_HEIGHT = 800   # pixels (too near)

# Broad vehicle coverage; also supports custom dataset names via keyword matching.
VEHICLE_EXACT_LABELS = {
    "motorbike",
    "motorcycle",
    "scooter",
    "car",
    "suv",
    "van",
    "pickup",
    "truck",
    "bus",
    "minibus",
    "tractor",
    "trailer",
    "rickshaw",
    "autorickshaw",
    "ambulance",
    "firetruck",
}
VEHICLE_KEYWORDS = (
    "moto",
    "scooter",
    "car",
    "suv",
    "van",
    "truck",
    "bus",
    "pickup",
    "tractor",
    "trailer",
    "rickshaw",
    "vehicle",
)


def is_vehicle_label(label: str) -> bool:
    name = label.lower().replace("-", "").replace("_", "").replace(" ", "")
    if name in VEHICLE_EXACT_LABELS:
        return True
    return any(token in name for token in VEHICLE_KEYWORDS)


def get_lane_mask(frame: np.ndarray, vehicle_boxes) -> np.ndarray:
    """
    Returns a binary mask of road features, heavily filtered to avoid
    buildings, sky, and sunlight glare while catching lanes in shadows.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_white  = cv2.inRange(hsv, WHITE_HSV_LO,  WHITE_HSV_HI)
    lane_mask = mask_white

    # 1. Horizon Mask: Eliminate sky, buildings, trees (ignore top 50%)
    h, w = frame.shape[:2]
    horizon = int(h * 0.5)
    lane_mask[0:horizon, :] = 0

    # 2. Edge Intersection: Eliminate flat blobs of sunlight/glare
    # Lane lines have sharp gradients. Glare/sun reflection is mostly flat inside.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1) # Thicken edges to match color mask
    
    lane_mask = cv2.bitwise_and(lane_mask, edges)

    # Light morphological clean-up: remove tiny noise blobs.
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)

    # Blank out vehicle bbox regions so features come from road only.
    if vehicle_boxes is not None:
        for _b in vehicle_boxes:
            _bx1, _by1, _bx2, _by2 = map(int, _b.xyxy[0])
            pad = 12
            lane_mask[
                max(0, _by1 - pad): min(lane_mask.shape[0], _by2 + pad),
                max(0, _bx1 - pad): min(lane_mask.shape[1], _bx2 + pad),
            ] = 0
    return lane_mask


model = YOLO(MODEL_PATH)

# Read FPS once for time-based thresholds.
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")

fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
if not fps or np.isnan(fps):
    fps = 30.0

parked_frames = int(PARKED_SECONDS * fps)
stale_track_frames = int(STALE_TRACK_SECONDS * fps)
min_visible_frames = int(MIN_VISIBLE_SECONDS * fps)

results = model.track(
    source=VIDEO_SOURCE,
    tracker=TRACKER_CONFIG,
    stream=True,
    persist=True,
    conf=0.3,
    iou=0.5,
)

prev_gray = None
frame_idx = 0
track_state = {}
stationary_threshold = 10.0   # generous default until calibration fires
calibration_samples = []
prev_lane_pts = None           # LK points seeded from lane pixels

for r in results:
    frame = r.orig_img.copy()
    boxes = r.boxes
    frame_idx += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---- Lane mask (white + yellow road markings) ----
    lane_mask = get_lane_mask(frame, boxes)
    lane_pixel_count = int(np.count_nonzero(lane_mask))

    # ---- Road-surface ego-motion via LK on lane pixels ----
    # Because lane markings are PAINTED ON THE ROAD, their optical flow
    # vector = exact road-surface motion in the camera frame.
    # vehicle_motion - lane_motion = absolute motion relative to road.
    H_ego = None
    lane_dx, lane_dy = 0.0, 0.0

    if prev_gray is not None:
        # Re-seed when we run low on tracked points.
        if prev_lane_pts is None or len(prev_lane_pts) < MIN_LANE_FEATURES:
            prev_lane_pts = cv2.goodFeaturesToTrack(
                prev_gray,
                maxCorners=MAX_LANE_FEATURES,
                qualityLevel=0.01,
                minDistance=10,
                mask=lane_mask if lane_pixel_count >= MIN_LANE_FEATURES else None,
            )

        if prev_lane_pts is not None and len(prev_lane_pts) >= 4:
            curr_pts, st, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, prev_lane_pts, None
            )
            good_prev = prev_lane_pts[st.ravel() == 1]
            good_curr = curr_pts[st.ravel() == 1]

            if len(good_prev) >= 4:
                # Homography absorbs rotation + perspective, not just translation.
                H_ego, _ = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 3.0)
                if H_ego is not None:
                    lane_dx = float(H_ego[0, 2])
                    lane_dy = float(H_ego[1, 2])

            prev_lane_pts = good_curr.reshape(-1, 1, 2) if len(good_curr) >= MIN_LANE_FEATURES else None
        else:
            prev_lane_pts = None

    prev_gray = gray

    # ---- Optional lane debug overlay ----
    if DEBUG_LANE_OVERLAY and lane_pixel_count > 0:
        green_tint = np.zeros_like(frame)
        green_tint[:, :, 1] = lane_mask  # green channel = lane pixels
        frame = cv2.addWeighted(frame, 1.0, green_tint, 0.25, 0)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            bbox_h = y2 - y1

            compensated_motion = 0.0
            parked = False
            status = "NA"

            # Skip parked logic for vehicles that are out of expected range (too far or too near).
            # Their pixel-motion might be too small or distorted by perspective.
            out_of_range = bbox_h < MIN_BBOX_HEIGHT or bbox_h > MAX_BBOX_HEIGHT
            if out_of_range:
                status = "OUT_OF_RANGE"

            if track_id >= 0 and is_vehicle_label(label) and not out_of_range:
                state = track_state.get(
                    track_id,
                    {
                        "smoothed_cx": cx,
                        "smoothed_cy": cy,
                        "stationary_frames": 0,
                        "jitter_frames": 0,
                        "visible_frames": 0,
                        "last_seen": frame_idx,
                    },
                )

                # --- EMA centroid smoothing ---
                # Reduces bbox jitter noise before motion is measured.
                scx = CENTROID_EMA_ALPHA * cx + (1 - CENTROID_EMA_ALPHA) * state["smoothed_cx"]
                scy = CENTROID_EMA_ALPHA * cy + (1 - CENTROID_EMA_ALPHA) * state["smoothed_cy"]

                # --- Lane-relative motion ---
                # Project the previous smoothed centroid through the lane homography
                # to get where it SHOULD be this frame if it moved WITH the road
                # (i.e. if it is parked).
                # Residual distance from that expected position = true absolute motion.
                if H_ego is not None:
                    pt = np.array(
                        [[[state["smoothed_cx"], state["smoothed_cy"]]]],
                        dtype=np.float32,
                    )
                    expected = cv2.perspectiveTransform(pt, H_ego)[0][0]
                    compensated_dx = scx - float(expected[0])
                    compensated_dy = scy - float(expected[1])
                else:
                    # Fallback: simple translation using lane flow averages.
                    compensated_dx = (scx - state["smoothed_cx"]) - lane_dx
                    compensated_dy = (scy - state["smoothed_cy"]) - lane_dy

                compensated_motion = float(np.hypot(compensated_dx, compensated_dy))

                if frame_idx <= CALIBRATION_FRAMES:
                    calibration_samples.append(compensated_motion)

                # --- Forgiveness frames ---
                # A single jitter spike won't wipe out accumulated history.
                if compensated_motion < stationary_threshold:
                    state["stationary_frames"] += 1
                    state["jitter_frames"] = 0
                else:
                    state["jitter_frames"] += 1
                    if state["jitter_frames"] > FORGIVENESS_FRAMES:
                        state["stationary_frames"] = 0
                        state["jitter_frames"] = 0

                state["visible_frames"] += 1
                state["smoothed_cx"] = scx
                state["smoothed_cy"] = scy
                state["last_seen"] = frame_idx

                parked = (
                    state["visible_frames"] >= min_visible_frames
                    and state["stationary_frames"] >= parked_frames
                )
                status = "PARKED" if parked else "MOVING"
                track_state[track_id] = state

            color = (0, 0, 255) if parked else (0, 255, 255) if out_of_range else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"ID:{track_id} {label} {conf:.2f} {status} d={compensated_motion:.1f} h={bbox_h}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), (0, 255, 255), -1)
            cv2.putText(
                frame,
                text,
                (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
            )

    # One-time threshold calibration after initial frames.
    # P80 of real compensated motions + margin gives a robust scene-specific threshold:
    # low enough to catch parked vehicles, high enough to absorb residual noise.
    if frame_idx == CALIBRATION_FRAMES and calibration_samples:
        p80 = float(np.percentile(calibration_samples, 80))
        stationary_threshold = float(
            np.clip(p80 * 1.5, MIN_STATIONARY_THRESHOLD, MAX_STATIONARY_THRESHOLD)
        )

    stale_ids = [
        tid
        for tid, state in track_state.items()
        if frame_idx - state["last_seen"] > stale_track_frames
    ]
    for tid in stale_ids:
        del track_state[tid]

    top_info = (
        f"lane_flow=({lane_dx:.2f},{lane_dy:.2f}) "
        f"lane_px={lane_pixel_count} "
        f"thr={stationary_threshold:.2f}px parked_after={parked_frames}f"
    )
    cv2.putText(
        frame,
        top_info,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    cv2.imshow("Parking Detection (Compensated)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
