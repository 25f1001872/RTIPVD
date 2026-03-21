from ultralytics import YOLO
from config import VEHICLE_LABELS

class VehicleYOLOTracker:
    def __init__(self, model_path, tracker_config):
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        
    def track_stream(self, source, conf=0.3, iou=0.5):
        """
        Creates a generator yielding track results frame-by-frame.
        """
        return self.model.track(
            source=source,
            tracker=self.tracker_config,
            stream=True,
            persist=True,
            conf=conf,
            iou=iou
        )
        
    def is_vehicle(self, cls_id):
        """
        Checks if the YOLO class ID maps to our targeted vehicle labels.
        """
        label = self.model.names[cls_id].lower().replace("-", "").replace("_", "").replace(" ", "")
        
        if label in VEHICLE_LABELS:
            return True
            
        # Fallback keyword match
        keywords = ["moto", "scooter", "car", "suv", "van", "truck", "bus", "pickup"]
        return any(k in label for k in keywords)
