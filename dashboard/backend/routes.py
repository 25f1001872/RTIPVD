"""REST API routes for the RTIPVD backend."""

from flask import Blueprint, Flask, current_app, jsonify, request

from src.database.models import ViolationRecord, parse_timestamp, utc_now


def _require_api_key() -> bool:
	"""Validate API key header when backend API key is configured."""
	expected = current_app.config.get("API_KEY", "")
	if not expected:
		return True

	provided = request.headers.get("X-API-Key", "")
	return provided == expected


def _to_float(value, default=None):
	if value is None:
		return default
	try:
		return float(value)
	except (TypeError, ValueError):
		return default


def register_routes(app: Flask) -> None:
	"""Register all API endpoints on the provided Flask app."""
	api = Blueprint("api", __name__, url_prefix="/api")

	@api.get("/health")
	def health():
		db_manager = current_app.config["DB_MANAGER"]
		return jsonify(
			{
				"ok": True,
				"db_enabled": db_manager.enabled,
				"db_path": str(db_manager.db_path),
			}
		)

	@api.get("/violations")
	def list_violations():
		db_manager = current_app.config["DB_MANAGER"]

		try:
			limit = int(request.args.get("limit", 100))
		except ValueError:
			limit = 100
		limit = max(1, min(limit, 1000))

		records = db_manager.list_recent(limit=limit)
		return jsonify({"count": len(records), "items": records})

	@api.post("/violations")
	def upsert_violation():
		if not _require_api_key():
			return jsonify({"ok": False, "error": "Unauthorized"}), 401

		payload = request.get_json(silent=True) or {}
		plate = str(payload.get("license_plate", "")).strip().upper()
		if not plate:
			return jsonify({"ok": False, "error": "license_plate is required"}), 400

		first_seen = parse_timestamp(payload.get("first_seen")) or utc_now()
		last_seen = parse_timestamp(payload.get("last_seen")) or first_seen
		if last_seen < first_seen:
			last_seen = first_seen

		duration = _to_float(payload.get("duration_sec"), None)
		if duration is None:
			duration = max((last_seen - first_seen).total_seconds(), 0.0)

		record = ViolationRecord(
			license_plate=plate,
			first_seen=first_seen,
			last_seen=last_seen,
			duration_sec=duration,
			latitude=_to_float(payload.get("latitude"), None),
			longitude=_to_float(payload.get("longitude"), None),
			screenshot_path=payload.get("screenshot_path"),
			video_source=payload.get("video_source"),
			confidence=_to_float(payload.get("confidence"), None),
		)

		db_manager = current_app.config["DB_MANAGER"]
		violation_id, inserted = db_manager.upsert_violation(record)

		status_code = 201 if inserted else 200
		return (
			jsonify(
				{
					"ok": True,
					"id": violation_id,
					"event_type": payload.get("event_type", "updated"),
					"inserted": inserted,
				}
			),
			status_code,
		)

	app.register_blueprint(api)
