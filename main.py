from ultralytics import YOLO

# Load model
model = YOLO("weights/best.pt")

# Run detection on video
results = model.predict(
    source="data_01.mp4",
    show=True,
    stream=True
)

for r in results:
    boxes = r.boxes