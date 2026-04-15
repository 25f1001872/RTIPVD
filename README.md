# RTIPVD - Real-Time Illegal Parking Vehicle Detection

> IIT Roorkee | 2025

RTIPVD detects illegally parked vehicles from moving-camera video using YOLOv8 + ByteTrack, ego-motion compensation, and plate OCR.

## New Deployment Direction (Recommended)

Raspberry Pi now acts as a sender node:

1. Pi sends compressed video frames over network.
2. Pi sends synchronized camera GPS telemetry with each frame.
3. Laptop acts as processing server (detection + geospatial projection).

Key scripts:

- Pi sender: `deploy/raspberry_pi/send_video_and_gps.py`
- Pi wrapper: `deploy/raspberry_pi/send_stream.sh`
- Laptop stream server: `scripts/laptop_stream_server.py`
- Laptop wrapper: `deploy/laptop/start_stream_server.ps1`
- Geospatial utility: `scripts/calculate_vehicle_geocoords.py`

## Features

- Moving camera support (dashcam/drone/patrol videos)
- Ego-motion compensation from lane optical flow
- Vehicle detection + persistent tracking
- Adaptive parked/moving threshold calibration
- Plate OCR with regex validation and temporal voting
- Local violation database (SQLite)
- Optional GPS tagging (ESP32 + NEO-6M serial)
- Optional backend sync + dashboard API

## Architecture

Video input -> preprocessing -> detection/tracking + lane ego-motion -> parked decision -> OCR -> violation service

Violation service:
- tags GPS coordinates
- saves/updates SQLite rows
- optionally posts to backend API

## Project Structure

- main pipeline: `main.py`
- config: `config/config.py`
- source modules: `src/`
- backend API: `dashboard/backend/`
- dashboard frontend: `dashboard/frontend/`
- deployment profiles:
	- laptop: `deploy/laptop/`
	- raspberry pi: `deploy/raspberry_pi/`

## Quick Start (Laptop)

1. Clone repo and enter folder.
2. Create virtual env and activate.
3. Install dependencies.
4. Copy model to `weights/best.pt`.
5. Copy test video to `data/videos/d1.mp4`.
6. Run setup verifier.
7. Run backend and pipeline.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -r dashboard/backend/requirements.txt
python scripts/verify_setup.py
python dashboard/backend/app.py
python main.py
```

## Quick Start (Raspberry Pi)

Use the Raspberry Pi deployment profile:

```bash
bash deploy/raspberry_pi/setup.sh
bash deploy/raspberry_pi/run_pi.sh
```

For full beginner steps, see:
- `docs/BEGINNER_GUIDE.md`
- `docs/STREAMING_ARCHITECTURE.md`
- `deploy/laptop/README.md`
- `deploy/raspberry_pi/README.md`

## Backend API

- `GET /api/health`
- `GET /api/violations?limit=100`
- `POST /api/violations`

## Notes

- Config supports environment variables (`RTIPVD_*`) for profile-based runs.
- CPU-only mode is supported for Raspberry Pi.
- If backend is disabled, all violations are still stored locally in SQLite.
