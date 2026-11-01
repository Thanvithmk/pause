import logging
import os
import time

from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, g, jsonify, request
from flask_cors import CORS

from config import get_config
from modules.database import init_db

_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv()


def create_app(config_name=None):
    app = Flask(__name__)
    app.config.from_object(get_config(config_name))

    _configure_logging(app)
    _configure_cors(app)
    init_db(app.config["SQLALCHEMY_DATABASE_URI"])

    register_blueprints(app)
    register_error_handlers(app)
    register_request_logging(app)

    return app


def _configure_logging(app):
    level = logging.DEBUG if app.config.get("DEBUG") else logging.INFO
    app.logger.setLevel(level)
    if not app.logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        app.logger.addHandler(handler)


def _configure_cors(app):
    CORS(
        app,
        origins=app.config["CORS_ORIGINS"],
        supports_credentials=app.config["CORS_SUPPORTS_CREDENTIALS"],
        allow_headers=app.config["CORS_ALLOW_HEADERS"],
        methods=app.config["CORS_METHODS"],
    )


def register_blueprints(app):
    from api.routes import ApiError, api_bp
    from modules.auth import AuthError, auth_bp
    from modules.send_gateway import SendError

    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(api_bp, url_prefix="/api")

    @app.errorhandler(AuthError)
    def handle_auth_error(error):
        return jsonify({"error": "auth_error", "message": error.message}), error.status_code

    @app.errorhandler(ApiError)
    def handle_api_error(error):
        return jsonify({"error": "api_error", "message": error.message}), error.status_code

    @app.errorhandler(SendError)
    def handle_send_error(error):
        return jsonify({"error": "send_error", "message": error.message}), error.status_code


def register_error_handlers(app):
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({"error": "bad_request", "message": str(error)}), 400

    @app.errorhandler(401)
    def unauthorized(error):
        return jsonify({"error": "unauthorized", "message": str(error)}), 401

    @app.errorhandler(403)
    def forbidden(error):
        return jsonify({"error": "forbidden", "message": str(error)}), 403

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "not_found", "message": str(error)}), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({"error": "method_not_allowed", "message": str(error)}), 405

    @app.errorhandler(500)
    def internal_error(error):
        app.logger.exception("Internal server error")
        return jsonify({"error": "internal_server_error", "message": str(error)}), 500

    @app.errorhandler(Exception)
    def unhandled_exception(error):
        app.logger.exception("Unhandled exception")
        return jsonify({"error": "internal_server_error", "message": str(error)}), 500


def register_request_logging(app):
    @app.before_request
    def log_request():
        g.request_start = time.perf_counter()
        app.logger.info("request %s %s", request.method, request.path)

    @app.after_request
    def log_response(response):
        duration = time.perf_counter() - getattr(g, "request_start", time.perf_counter())
        app.logger.info(
            "response %s %s %s %.3fs",
            request.method,
            request.path,
            response.status_code,
            duration,
        )
        return response


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "5000")),
        debug=app.config.get("DEBUG", False),
    )
