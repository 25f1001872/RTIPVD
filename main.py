import cv2
from config import VIDEO_SOURCE, MODEL_PATH, TRACKER_CONFIG
from preprocess import clean_frame
from vehicle_tracker import VehicleYOLOTracker
from lane_tracker import LaneTracker
from parking_analyzer import ParkingAnalyzer
from license_plate_reader import LicensePlateReader

def main():
    # 1. Initialize Video
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Cannot open video source: {VIDEO_SOURCE}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
        
    # 2. Initialize Models & Trackers
    yolo_tracker = VehicleYOLOTracker(MODEL_PATH, TRACKER_CONFIG)
    lane_tracker = LaneTracker()
    parking_analyzer = ParkingAnalyzer(fps=fps)
    plate_reader = LicensePlateReader(use_mock=False) # Turned ON real OCR
    
    # Run YOLO streaming
    results_generator = yolo_tracker.track_stream(VIDEO_SOURCE)
    frame_idx = 0
    
    # 3. Main processing loop
    for r in results_generator:
        frame_idx += 1
        orig_frame = r.orig_img.copy()
        boxes = r.boxes
        
        # State memory for plates read this frame so we don't spam OCR
        read_plates = {}
        
        # --- PREPROCESSING ---
        # Note: YOLO processes the original stream, but we use the cleaned
        # version for optical-flow and final display.
        cleaned_frame, gray, avg_brightness = clean_frame(orig_frame)
        display_frame = cleaned_frame.copy()

        # --- EGO MOTION (Optical Flow on Lanes) ---
        H_ego, lane_dx, lane_dy, lane_px_count = lane_tracker.compute_ego_motion(cleaned_frame, gray, boxes)
        
        # --- VEHICLE LOGIC ---
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else -1
                cls_id = int(box.cls[0])
                
                # Filter classes
                if track_id < 0 or not yolo_tracker.is_vehicle(cls_id):
                    continue
                    
                # Centers and heights
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                bbox_h = y2 - y1
                
                # Analyze Status
                status, motion_mag = parking_analyzer.analyze_vehicle(
                    track_id, cx, cy, bbox_h, 
                    frame_idx, H_ego, lane_dx, lane_dy
                )
                
                # --- DRAWING ---
                plate_text = ""
                if status == "PARKED":
                    color = (0, 0, 255)      # Red for parked/illegal
                    
                    # Only read plate if parked to save CPU
                    if track_id not in read_plates:
                         plate_text = plate_reader.read_plate(orig_frame, x1, y1, x2, y2, track_id)
                         read_plates[track_id] = plate_text
                    else:
                         plate_text = read_plates[track_id]
                         
                elif status == "OUT_OF_RANGE":
                    color = (0, 255, 255)    # Yellow to notify ignoring
                else: 
                    color = (0, 255, 0)      # Green for moving cleanly
                    
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Format the text with the license plate if available
                text = f"ID:{track_id} {status} d={motion_mag:.1f}"
                if plate_text:
                    text += f" [{plate_text}]"
                    
                cv2.putText(display_frame, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                
        # --- CLEANUP MEMORY ---
        parking_analyzer.purge_stale_tracks(frame_idx)
        
        # Overlay Top Details
        top_info = f"Brightness: {avg_brightness:.1f} | Thr: {parking_analyzer.stationary_threshold:.1f}px"
        cv2.putText(display_frame, top_info, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Show Output
        cv2.imshow("Parking Detection System", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Teardown
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

