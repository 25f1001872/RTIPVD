import cv2
from ultralytics import YOLO

# Load model
model = YOLO("weights/best.pt")

VIDEO_SOURCE = "data_01.mp4"

# Run ByteTrack tracking — assigns a persistent ID to every detection
results = model.track(
    source=VIDEO_SOURCE,
    tracker="bytetrack.yaml",   # built-in ByteTrack config
    stream=True,
    persist=True,               # keep track state across frames
    conf=0.3,
    iou=0.5,
)

for r in results:
    frame = r.orig_img.copy()
    boxes = r.boxes

    if boxes is not None:
        for box in boxes:
            # Bounding box (xyxy)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Track ID (None if tracker lost the object momentarily)
            track_id = int(box.id[0]) if box.id is not None else -1

            # Confidence & class
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = model.names[cls]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # Draw label with track ID
            text = f"ID:{track_id} {label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), (0, 255, 255), -1)
            cv2.putText(frame, text, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow("ByteTrack — RTIPVD", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()