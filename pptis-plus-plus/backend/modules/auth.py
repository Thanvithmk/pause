import os
import re
import smtplib
import string
import time
import uuid
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from functools import wraps

import jwt
from flask import Blueprint, g, jsonify, request
from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError
from werkzeug.security import check_password_hash, generate_password_hash

from modules.database import User, close_session, get_session

USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9]{3,20}$")
EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
TOKEN_TTL = timedelta(days=7)
RESET_TOKEN_TTL = timedelta(hours=1)
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 15 * 60
ALGORITHM = "HS256"

_token_blacklist = {}
_failed_attempts = {}

auth_bp = Blueprint("auth", __name__)


class AuthError(Exception):
    def __init__(self, message, status_code=401):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _jwt_secret():
    try:
        from config import get_config

        return get_config().JWT_SECRET_KEY
    except ImportError:
        return os.getenv("JWT_SECRET_KEY", "change-me-jwt")


def _purge_expired(store, now=None):
    now = now or datetime.now(timezone.utc)
    expired = [
        key
        for key, expires_at in store.items()
        if isinstance(expires_at, datetime) and expires_at <= now
    ]
    for key in expired:
        store.pop(key, None)


def encode_token(user_id, token_type="access", expires_delta=None):
    expires_delta = expires_delta or TOKEN_TTL
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "type": token_type,
        "iat": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(payload, _jwt_secret(), algorithm=ALGORITHM)


def decode_token(token, expected_type="access"):
    if not token:
        raise AuthError("Missing token", 401)
    try:
        payload = jwt.decode(token, _jwt_secret(), algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("Token has expired", 401) from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError("Invalid token", 401) from exc

    _purge_expired(_token_blacklist)
    jti = payload.get("jti")
    if jti in _token_blacklist:
        raise AuthError("Token has been revoked", 401)
    if expected_type and payload.get("type") != expected_type:
        raise AuthError("Invalid token type", 401)
    return payload


def blacklist_token(payload):
    jti = payload.get("jti")
    exp = payload.get("exp")
    if not jti or not exp:
        return
    _token_blacklist[jti] = datetime.fromtimestamp(exp, tz=timezone.utc)


def extract_bearer_token():
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        return header[7:].strip()
    return None


def validate_username(username):
    if not username or not USERNAME_PATTERN.fullmatch(username):
        raise AuthError("Username must be alphanumeric and 3-20 characters", 400)
    return username


def validate_email(email):
    if not email or not EMAIL_PATTERN.fullmatch(email.strip()):
        raise AuthError("Invalid email format", 400)
    return email.strip().lower()


def validate_password(password):
    if not password or len(password) < 8:
        raise AuthError("Password must be at least 8 characters", 400)
    if not re.search(r"[A-Z]", password):
        raise AuthError("Password must contain an uppercase letter", 400)
    if not re.search(r"\d", password):
        raise AuthError("Password must contain a digit", 400)
    if not any(char in string.punctuation for char in password):
        raise AuthError("Password must contain a special character", 400)
    return password


def _attempt_key(identifier):
    return f"{request.remote_addr}:{identifier.lower()}"


def check_rate_limit(identifier):
    _purge_expired(_failed_attempts)
    key = _attempt_key(identifier)
    expires_at = _failed_attempts.get(key)
    if expires_at:
        raise AuthError("Too many failed attempts. Try again later.", 429)


def record_failed_attempt(identifier):
    key = _attempt_key(identifier)
    count_key = f"{key}:count"
    count = int(_failed_attempts.get(count_key, 0)) + 1
    _failed_attempts[count_key] = count
    if count >= MAX_FAILED_ATTEMPTS:
        _failed_attempts[key] = datetime.now(timezone.utc) + timedelta(seconds=LOCKOUT_SECONDS)
        _failed_attempts.pop(count_key, None)


def clear_failed_attempts(identifier):
    key = _attempt_key(identifier)
    _failed_attempts.pop(key, None)
    _failed_attempts.pop(f"{key}:count", None)


def _user_payload(user, token, refresh_token=None):
    data = {
        "user_id": user.user_id,
        "username": user.username,
        "email": user.email,
        "token": token,
        "user": user.to_dict(),
    }
    if refresh_token is not None:
        data["refresh_token"] = refresh_token
    return data


def register_user(username, email, password):
    username = validate_username(username)
    email = validate_email(email)
    validate_password(password)

    session = get_session()
    try:
        existing = (
            session.query(User)
            .filter(or_(User.username == username, User.email == email))
            .first()
        )
        if existing:
            if existing.username == username:
                raise AuthError("Username already exists", 409)
            raise AuthError("Email already exists", 409)

        user = User(username=username, email=email)
        user.password_hash = generate_password_hash(password)
        session.add(user)
        session.commit()
        session.refresh(user)
        token = encode_token(user.user_id, "access")
        return {"user_id": user.user_id, "token": token, "user": user.to_dict()}
    except IntegrityError as exc:
        session.rollback()
        raise AuthError("Username or email already exists", 409) from exc
    except AuthError:
        session.rollback()
        raise
    finally:
        close_session()


def login_user(identifier, password):
    if not identifier or not password:
        raise AuthError("Invalid credentials", 401)

    check_rate_limit(identifier)
    session = get_session()
    try:
        lookup = identifier.strip()
        query = session.query(User)
        if "@" in lookup:
            user = query.filter(User.email == lookup.lower()).first()
        else:
            user = query.filter(User.username == lookup).first()
            if user is None:
                user = query.filter(User.email == lookup.lower()).first()

        if user is None or not check_password_hash(user.password_hash, password):
            record_failed_attempt(identifier)
            raise AuthError("Invalid credentials", 401)

        clear_failed_attempts(identifier)
        access_token = encode_token(user.user_id, "access")
        refresh_token = encode_token(user.user_id, "refresh")
        return _user_payload(user, access_token, refresh_token)
    finally:
        close_session()


def logout_user(token=None):
    token = token or extract_bearer_token()
    payload = decode_token(token, expected_type=None)
    blacklist_token(payload)
    return {"message": "Logged out"}


def refresh_access_token(refresh_token=None):
    refresh_token = refresh_token or extract_bearer_token()
    payload = decode_token(refresh_token, expected_type="refresh")
    user_id = int(payload["sub"])
    blacklist_token(payload)

    session = get_session()
    try:
        user = User.get_by_id(session, user_id)
        if user is None:
            raise AuthError("Invalid credentials", 401)
        access_token = encode_token(user.user_id, "access")
        new_refresh = encode_token(user.user_id, "refresh")
        return _user_payload(user, access_token, new_refresh)
    finally:
        close_session()


def verify_token(token=None):
    token = token or extract_bearer_token()
    payload = decode_token(token, expected_type="access")
    session = get_session()
    try:
        user = User.get_by_id(session, int(payload["sub"]))
        if user is None:
            raise AuthError("Invalid credentials", 401)
        return user, payload
    finally:
        close_session()


def require_auth(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        user, payload = verify_token()
        g.current_user = user
        g.current_user_id = user.user_id
        g.token_payload = payload
        return view(*args, **kwargs)

    return wrapped


def _send_reset_email(email, token):
    host = os.getenv("SMTP_HOST")
    if not host:
        return False

    message = EmailMessage()
    message["Subject"] = "PPTIS++ password reset"
    message["From"] = os.getenv("SMTP_FROM", "noreply@pptis.local")
    message["To"] = email
    message.set_content(
        f"Use this token to reset your password. It expires in 1 hour.\n\n{token}"
    )

    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    use_tls = os.getenv("SMTP_TLS", "true").lower() in ("1", "true", "yes")

    with smtplib.SMTP(host, port, timeout=10) as smtp:
        if use_tls:
            smtp.starttls()
        if username and password:
            smtp.login(username, password)
        smtp.send_message(message)
    return True


def request_password_reset(email):
    email = validate_email(email)
    session = get_session()
    try:
        user = session.query(User).filter(User.email == email).first()
        if user is None:
            return {"message": "If the account exists, a reset email was sent"}
        token = encode_token(user.user_id, "reset", expires_delta=RESET_TOKEN_TTL)
        _send_reset_email(user.email, token)
        result = {"message": "If the account exists, a reset email was sent"}
        if os.getenv("FLASK_ENV") == "development":
            result["reset_token"] = token
        return result
    finally:
        close_session()


def reset_password(token, new_password):
    validate_password(new_password)
    payload = decode_token(token, expected_type="reset")
    session = get_session()
    try:
        user = User.get_by_id(session, int(payload["sub"]))
        if user is None:
            raise AuthError("Invalid credentials", 401)
        user.password_hash = generate_password_hash(new_password)
        session.commit()
        blacklist_token(payload)
        return {"message": "Password updated"}
    finally:
        close_session()


@auth_bp.errorhandler(AuthError)
def handle_auth_error(error):
    return jsonify({"error": "auth_error", "message": error.message}), error.status_code


@auth_bp.route("/register", methods=["POST"])
def register_route():
    body = request.get_json(silent=True) or {}
    result = register_user(body.get("username"), body.get("email"), body.get("password"))
    return jsonify(result), 201


@auth_bp.route("/login", methods=["POST"])
def login_route():
    body = request.get_json(silent=True) or {}
    identifier = body.get("email") or body.get("username") or body.get("identifier")
    result = login_user(identifier, body.get("password"))
    return jsonify(result), 200


@auth_bp.route("/logout", methods=["POST"])
def logout_route():
    return jsonify(logout_user()), 200


@auth_bp.route("/refresh", methods=["POST"])
def refresh_route():
    body = request.get_json(silent=True) or {}
    token = body.get("refresh_token") or extract_bearer_token()
    return jsonify(refresh_access_token(token)), 200


@auth_bp.route("/password-reset", methods=["POST"])
def password_reset_request_route():
    body = request.get_json(silent=True) or {}
    return jsonify(request_password_reset(body.get("email"))), 200


@auth_bp.route("/password-reset/confirm", methods=["POST"])
def password_reset_confirm_route():
    body = request.get_json(silent=True) or {}
    return jsonify(reset_password(body.get("token"), body.get("password"))), 200
