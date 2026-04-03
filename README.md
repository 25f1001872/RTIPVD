# 🚗 RTIPVD — Real-Time Illegal Parking Vehicle Detection

> **IIT Roorkee | 2025**

A real-time system for detecting illegally parked vehicles from **moving cameras** (dashcam, drone, patrol vehicle) using computer vision and deep learning.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎥 **Moving Camera Support** | Works on dashcams, drones, patrol vehicles — not limited to fixed CCTV |
| 🛣️ **Ego-Motion Compensation** | Lane-anchored optical flow + RANSAC homography separates camera motion from vehicle motion |
| 🎯 **YOLOv8 + ByteTrack** | Custom-trained vehicle detection with persistent multi-object tracking |
| 📊 **Auto-Calibrating Threshold** | P80-based adaptive threshold self-tunes to any video in first 60 frames |
| 🔍 **License Plate OCR** | EasyOCR with Indian plate regex validation and temporal majority voting |
| 🛡️ **Jitter-Resilient** | EMA smoothing + forgiveness frames prevent false classifications |
| 🌙 **Night Mode** | CLAHE-based automatic enhancement for low-light scenarios |

---

## 🏗️ Architecture

Video Input → Preprocessing → Dual Pipeline:
├── Vehicle Detection (YOLOv8) + Tracking (ByteTrack)
└── Lane Detection (HSV) + Ego-Motion (LK + Homography)
↓
Motion Analysis (ego-compensated) → Parking Decision → Plate OCR → Display


---

## 📁 Project Structure
RTIPVD/
├── main.py # Entry point
├── config/ # Configuration
│ ├── config.py # All parameters
│ └── bytetrack.yaml # Tracker config
├── src/ # Source code
│ ├── preprocessing/ # Frame enhancement
│ ├── detection/ # YOLOv8 + ByteTrack
│ ├── ego_motion/ # Lane detection + optical flow
│ ├── analyzer/ # Parking decision + calibration
│ ├── ocr/ # License plate reading
│ ├── visualization/ # Rendering + stats overlay
│ ├── evidence/ # [Phase 2] Screenshots, GPS, maps
│ ├── database/ # [Phase 2] Violation storage
│ └── utils/ # Logging, timing, validators
├── weights/ # Model weights (.gitignored)
├── data/ # Videos and datasets
├── output/ # Generated output
├── docs/ # Documentation
└── scripts/ # Utility scripts


---

## 🚀 Quick Start

### Prerequisites
- Python 3.11.9
- NVIDIA GPU with CUDA 12.1 (tested on RTX 4050)
- NVIDIA drivers installed

### Installation

```powershell
# Clone the repository
git clone https://github.com/YOUR_USERNAME/RTIPVD.git
cd RTIPVD

# Create virtual environment
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Place your model weights
# Copy best.pt to weights/best.pt

# Place your test video
# Copy your video to data/videos/d1.mp4

# Verify setup
python scripts/verify_setup.py

# Run
python main.py

Controls
Press q to quit
🎯 Tech Stack
Component	Technology
Detection	YOLOv8 (custom trained)
Tracking	ByteTrack
Ego-Motion	Lucas-Kanade Optical Flow + RANSAC Homography
Lane Detection	HSV Color Filtering + Canny Edge
OCR	EasyOCR
Preprocessing	OpenCV CLAHE
Language	Python 3.11.9
GPU	NVIDIA CUDA 12.1

📋 Configuration
All parameters are in config/config.py:

Parameter	Default	Description
PARKED_SECONDS	5.0	Seconds stationary before flagging
CALIBRATION_FRAMES	60	Frames for auto-threshold calibration
CENTROID_EMA_ALPHA	0.35	Smoothing factor for centroid jitter
FORGIVENESS_FRAMES	10	Jitter tolerance before counter reset
MIN_BBOX_HEIGHT	120	Minimum bbox height (skip far vehicles)
MAX_BBOX_HEIGHT	800	Maximum bbox height (skip close vehicles)

🔜 Roadmap (Phase 2)
 Database integration (SQLite — plate as primary key)
 Frame screenshot as violation evidence
 GPS coordinates from recording camera
 Google Maps / OSM overlay for zone verification
 Duration logging (how long vehicle was parked)
 Web dashboard for traffic authorities
 Edge deployment (Jetson Nano / Raspberry Pi)
 Automated e-challan generation
