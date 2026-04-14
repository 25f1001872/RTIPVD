"""
Laptop stream server for RTIPVD.

Receives frame+GPS packets from Raspberry Pi, runs vehicle detection,
and estimates geospatial coordinates for each detected vehicle.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from flask import Flask, jsonify, request

from src.database.models import GPSFix
from src.detection.vehicle_detector import VehicleDetector
from src.geospatial.vehicle_geo_mapper import VehicleGeoMapper
from src.streaming.packet import FrameTelemetryPacket
from src.streaming.sync import GPSSyncBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RTIPVD laptop stream processor")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8088, help="Bind port")
    parser.add_argument("--model", default="weights/best.pt", help="YOLO model path")
    parser.add_argument("--device", default="cuda:0", help="Inference device")
    parser.add_argument("--det-conf", type=float, default=0.30, help="Detection confidence")
    parser.add_argument("--hfov", type=float, default=78.0, help="Camera horizontal FOV")
    parser.add_argument("--default-heading", type=float, default=0.0, help="Fallback heading")
    parser.add_argument("--log-csv", default="output/results/stream_geocoords.csv", help="Output CSV for detections")
    return parser.parse_args()


def _build_app(args: argparse.Namespace) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

    detector = VehicleDetector(model_path=args.model, device=args.device)
    model = detector.get_model()
    mapper = VehicleGeoMapper(horizontal_fov_deg=args.hfov)
    gps_sync = GPSSyncBuffer(max_size=1024)

    csv_path = Path(args.log_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, int] = {
        "packets_received": 0,
        "frames_processed": 0,
        "detections_logged": 0,
    }

    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                [
                    "sequence_id",
                    "frame_timestamp_utc",
                    "class_id",
                    "class_label",
                    "bbox_x1",
                    "bbox_y1",
                    "bbox_x2",
                    "bbox_y2",
                    "camera_lat",
                    "camera_lon",
                    "camera_heading",
                    "vehicle_lat",
                    "vehicle_lon",
                    "distance_m",
                    "bearing_deg",
                    "geo_confidence",
                    "det_confidence",
                ]
            )

    def _resolve_gps(packet: FrameTelemetryPacket) -> GPSFix:
        parsed_fix = gps_sync.parse_fix(packet.gps)
        if parsed_fix.fix and parsed_fix.latitude is not None and parsed_fix.longitude is not None:
            gps_sync.add_fix(parsed_fix)
            return parsed_fix

        matched = gps_sync.get_closest(packet.frame_timestamp_utc)
        if matched is not None:
            return matched

        return GPSFix(
            latitude=None,
            longitude=None,
            heading_deg=None,
            speed_mps=None,
            satellites=None,
            fix=False,
            source="unsynced",
            timestamp=parsed_fix.timestamp,
        )

    def _process_packet(packet: FrameTelemetryPacket) -> List[dict]:
        frame = packet.decode_frame()
        if frame is None:
            return []

        stats["frames_processed"] += 1

        gps_fix = _resolve_gps(packet)
        if gps_fix.latitude is None or gps_fix.longitude is None:
            return []

        heading = gps_fix.heading_deg if gps_fix.heading_deg is not None else args.default_heading
        frame_h, frame_w = frame.shape[:2]

        results = model.predict(
            source=frame,
            conf=args.det_conf,
            verbose=False,
            device=args.device,
        )
        if not results or results[0].boxes is None:
            return []

        detections: List[dict] = []

        with csv_path.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                if not detector.is_vehicle(cls_id):
                    continue

                x1, y1, x2, y2 = map(float, box.xyxy[0])
                det_conf = float(box.conf[0]) if box.conf is not None else 0.0

                estimate = mapper.estimate_from_bbox(
                    camera_lat=float(gps_fix.latitude),
                    camera_lon=float(gps_fix.longitude),
                    camera_heading_deg=float(heading),
                    bbox_xyxy=(x1, y1, x2, y2),
                    frame_shape=(frame_h, frame_w),
                )
                if estimate is None:
                    continue

                row = {
                    "sequence_id": packet.sequence_id,
                    "frame_timestamp_utc": packet.frame_timestamp_utc,
                    "class_id": cls_id,
                    "class_label": detector.get_label(cls_id),
                    "bbox": [x1, y1, x2, y2],
                    "camera_lat": gps_fix.latitude,
                    "camera_lon": gps_fix.longitude,
                    "camera_heading": heading,
                    "vehicle_lat": estimate.latitude,
                    "vehicle_lon": estimate.longitude,
                    "distance_m": estimate.distance_m,
                    "bearing_deg": estimate.bearing_deg,
                    "geo_confidence": estimate.confidence,
                    "det_confidence": det_conf,
                }
                detections.append(row)
                stats["detections_logged"] += 1

                writer.writerow(
                    [
                        packet.sequence_id,
                        packet.frame_timestamp_utc,
                        cls_id,
                        detector.get_label(cls_id),
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2),
                        gps_fix.latitude,
                        gps_fix.longitude,
                        round(float(heading), 3),
                        round(estimate.latitude, 8),
                        round(estimate.longitude, 8),
                        round(estimate.distance_m, 3),
                        round(estimate.bearing_deg, 3),
                        round(estimate.confidence, 3),
                        round(det_conf, 3),
                    ]
                )

        return detections

    @app.get("/health")
    def health():
        return jsonify(
            {
                "ok": True,
                "stats": stats,
                "model": args.model,
                "device": args.device,
                "csv_log": str(csv_path),
            }
        )

    @app.post("/ingest/frame")
    def ingest_frame():
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"ok": False, "error": "Empty or invalid JSON"}), 400

        packet = FrameTelemetryPacket.from_json(payload)
        stats["packets_received"] += 1

        detections = _process_packet(packet)
        return jsonify(
            {
                "ok": True,
                "sequence_id": packet.sequence_id,
                "detections": detections,
                "detection_count": len(detections),
            }
        )

    return app


def main() -> int:
    args = parse_args()
    app = _build_app(args)
    print(f"[STREAM SERVER] Listening on http://{args.host}:{args.port}")
    print(f"[STREAM SERVER] CSV log: {args.log_csv}")
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
