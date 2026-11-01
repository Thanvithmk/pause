import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


def _database_uri():
    uri = os.getenv("DATABASE_URL")
    if uri:
        if uri.startswith("postgres://"):
            uri = uri.replace("postgres://", "postgresql://", 1)
        return uri

    use_postgres = os.getenv("USE_POSTGRES", "false").lower() in ("1", "true", "yes")
    if use_postgres:
        user = os.getenv("POSTGRES_USER", "pptis")
        password = os.getenv("POSTGRES_PASSWORD", "pptis")
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        name = os.getenv("POSTGRES_DB", "pptis")
        return f"postgresql://{user}:{password}@{host}:{port}/{name}"

    sqlite_path = os.getenv("SQLITE_PATH", str(BASE_DIR / "pptis.db"))
    return f"sqlite:///{sqlite_path}"


def _cors_origins():
    raw = os.getenv("CORS_ORIGINS", "http://localhost:3000")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


class Config:
    ENV = os.getenv("FLASK_ENV", "production")
    DEBUG = False
    TESTING = False

    SECRET_KEY = os.getenv("SECRET_KEY", "change-me")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me-jwt")
    JWT_ACCESS_TOKEN_EXPIRES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRES", "3600"))

    SQLALCHEMY_DATABASE_URI = _database_uri()
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    CORS_ORIGINS = _cors_origins()
    CORS_SUPPORTS_CREDENTIALS = True
    CORS_ALLOW_HEADERS = ["Content-Type", "Authorization"]
    CORS_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]

    PACL_LAMBDA = 0.5
    FINAL_RISK_ALPHA = 0.5
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")

    THRESHOLD_ALLOW = 0.30
    THRESHOLD_WARN = 0.55
    THRESHOLD_REWRITE = 0.80
    DELAYED_SEND_DURATION = 60

    # CEL weight vector W (6 dimensions)
    CEL_WEIGHT_VECTOR = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    # Logistic regression: P_esc = σ(b + w_E * E + w_P * PACL)
    # TRAINED in Phase 6; these are first-run defaults.
    LOGISTIC_INTERCEPT = 0.0
    LOGISTIC_COEFFICIENTS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    LOGISTIC_B = float(os.getenv("LOGISTIC_B", "0.0"))
    LOGISTIC_W_E = float(os.getenv("LOGISTIC_W_E", "1.0"))
    LOGISTIC_W_P = float(os.getenv("LOGISTIC_W_P", "1.0"))


class DevelopmentConfig(Config):
    ENV = "development"
    DEBUG = True


class ProductionConfig(Config):
    ENV = "production"
    DEBUG = False


class TestingConfig(Config):
    ENV = "testing"
    DEBUG = True
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"


config_by_name = {
    "development": DevelopmentConfig,
    "debug": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}


def get_config(name=None):
    key = (name or os.getenv("FLASK_ENV", "production")).lower()
    return config_by_name.get(key, ProductionConfig)
