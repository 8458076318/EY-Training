# src/app/__init__.py
from flask import Flask
from .config import get_settings
from .routes import api_bp
from .extensions import db

def create_app(config=None) -> Flask:
    app = Flask(__name__)
    settings = get_settings()
    app.config.from_mapping(
        SECRET_KEY=settings.secret_key,
        SQLALCHEMY_DATABASE_URI=settings.db_url,
        DEBUG=settings.debug)
    if config:
        app.config.update(config)

    db.init_app(app)
    app.register_blueprint(api_bp,
        url_prefix="/api/v1")

    @app.errorhandler(404)
    def not_found(e):
        return {"error": "Not found"}, 404

    @app.errorhandler(422)
    def validation_error(e):
        return {"error": str(e)}, 422

    @app.errorhandler(500)
    def server_error(e):
        app.logger.error("Unhandled", exc_info=e)
        return {"error": "Internal error"}, 500

    return app

# run.py
if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8080)
