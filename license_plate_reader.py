import cv2
import numpy as np
import easyocr
import re
from collections import Counter

class LicensePlateReader:
    def __init__(self, use_mock=False, max_history=7):
        """
        Initializes the OCR engine, regex, and voting history.
        """
        self.use_mock = use_mock
        self.max_history = max_history
        self.plate_history = {} # {track_id: [list of valid reads]}
        
        # Extremely robust Indian License Plate Regex
        # Matches: MH12AB1234, MH121234, RJ01K456, etc.
        self.plate_pattern = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{1,4}$")

        if not self.use_mock:
            print("Loading EasyOCR model...")
            self.reader = easyocr.Reader(['en'], gpu=False)
            print("EasyOCR loaded.")

    def extract_plate_region(self, frame, x1, y1, x2, y2):
        """
        Heuristic: The license plate is usually in the bottom half or center of the vehicle bounding box.
        This simply crops the lower portion of the car to send to the OCR engine.
        For production, a secondary lightweight YOLO specifically for plates is much better.
        """
        h = y2 - y1
        w = x2 - x1
        
        # Crop the bottom 40% of the bounding box where plates usually live, 
        # while keeping some margin from the edges.
        plate_y1 = int(y1 + (h * 0.6))
        plate_y2 = y2
        plate_x1 = int(x1 + (w * 0.1))
        plate_x2 = int(x2 - (w * 0.1))
        
        # Ensure coordinates are within frame bounds
        plate_y1, plate_y2 = max(0, plate_y1), min(frame.shape[0], plate_y2)
        plate_x1, plate_x2 = max(0, plate_x1), min(frame.shape[1], plate_x2)

        if plate_y2 <= plate_y1 or plate_x2 <= plate_x1:
            return None

        return frame[plate_y1:plate_y2, plate_x1:plate_x2]

    def preprocess_crop(self, crop):
        """
        Enhances the plate crop specifically for OCR.
        Applies grayscale, blur to remove pixel noise, and CLAHE to pop text.
        """
        # 1. Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # 2. Gaussian blur to remove high-frequency noise/sensor grain
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. CLAHE to normalize lighting (important for shadows or bright glares)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blur)
        
        return enhanced

    def read_plate(self, frame, x1, y1, x2, y2, track_id):
        """
        Crops the vehicle, enhances image, reads text, validates via Regex, 
        and applies temporal voting.
        """
        plate_crop = self.extract_plate_region(frame, x1, y1, x2, y2)
        
        # Initialize memory for this track_id if not present
        if track_id not in self.plate_history:
            self.plate_history[track_id] = []
            
        def _get_best_vote():
            if not self.plate_history[track_id]:
                return "DETECTING..."
            # Return the most common string in the history window
            return Counter(self.plate_history[track_id]).most_common(1)[0][0]

        if plate_crop is None:
            return _get_best_vote()

        if self.use_mock:
            self.plate_history[track_id].append("MH12AB1234")
            return _get_best_vote()
        
        # --- Preprocess and Read ---
        enhanced_crop = self.preprocess_crop(plate_crop)
        results = self.reader.readtext(enhanced_crop, detail=0)
        
        if results:
            # Clean string
            raw_text = "".join(results).replace(" ", "").upper()
            # Clean out common OCR hallucinations (like dashes, underscores)
            clean_text = re.sub(r'[^A-Z0-9]', '', raw_text)
            
            # Regex validation
            if self.plate_pattern.match(clean_text):
                self.plate_history[track_id].append(clean_text)
                
                # Keep rolling window size to max_history bounds
                if len(self.plate_history[track_id]) > self.max_history:
                    self.plate_history[track_id].pop(0)

        return _get_best_vote()
