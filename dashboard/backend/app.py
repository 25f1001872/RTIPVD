"""
RTIPVD backend server entry point.

Exposes REST endpoints for violation ingestion and listing,
and serves a simple frontend dashboard for operators.
"""

import os
import sys
from pathlib import Path

from flask import Flask, send_from_directory
from flask_cors import CORS

# Ensure project root is importable when app is launched directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from config.config import DASHBOARD_DEBUG, DASHBOARD_HOST, DASHBOARD_PORT
from dashboard.backend.routes import register_routes
from src.database.db_manager import DatabaseManager


def create_app() -> Flask:
	"""Build and configure Flask app instance."""
	frontend_dir = PROJECT_ROOT / "dashboard" / "frontend"

	app = Flask(
		__name__,
		static_folder=str(frontend_dir),
		static_url_path="/dashboard/static",
	)
	CORS(app)

	db_manager = DatabaseManager()
	app.config["DB_MANAGER"] = db_manager
	app.config["API_KEY"] = os.getenv("RTIPVD_BACKEND_API_KEY", "")

	register_routes(app)

	@app.get("/")
	def index():
		return send_from_directory(frontend_dir, "index.html")

	@app.get("/dashboard")
	def dashboard():
		return send_from_directory(frontend_dir, "index.html")

	return app


app = create_app()


if __name__ == "__main__":
	app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=DASHBOARD_DEBUG)
