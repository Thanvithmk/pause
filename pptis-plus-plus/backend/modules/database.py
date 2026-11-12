import os
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.orm import (
    declarative_base,
    relationship,
    scoped_session,
    sessionmaker,
    validates,
)
from werkzeug.security import check_password_hash, generate_password_hash

INPUT_TYPES = ("text", "voice")
DECISION_ACTIONS = ("ALLOW", "WARN", "REWRITE", "DELAYED-SEND")
USER_RESPONSES = ("accepted", "rejected", "edited")
SEND_STATUSES = ("draft", "sending", "sent", "blocked")
OUTCOMES = ("positive", "neutral", "negative")
REVIEW_STATUSES = ("pending", "reviewed", "approved", "rejected")
TOXICITY_FIELDS = (
    "toxic",
    "threat",
    "insult",
    "severe_toxic",
    "obscene",
    "identity_hate",
)

Base = declarative_base()

engine = None
SessionLocal = None

input_type_enum = Enum(*INPUT_TYPES, name="input_type_enum")
decision_action_enum = Enum(*DECISION_ACTIONS, name="decision_action_enum")
user_response_enum = Enum(*USER_RESPONSES, name="user_response_enum")
outcome_enum = Enum(*OUTCOMES, name="outcome_enum")
review_status_enum = Enum(*REVIEW_STATUSES, name="review_status_enum")


def utcnow():
    return datetime.now(timezone.utc)


def _database_uri(database_uri=None):
    if database_uri:
        return database_uri
    try:
        from config import get_config

        return get_config().SQLALCHEMY_DATABASE_URI
    except ImportError as exc:
        raise RuntimeError("database: PostgreSQL configuration could not be loaded") from exc


def init_db(database_uri=None, create_tables=True):
    global engine, SessionLocal

    uri = _database_uri(database_uri)
    kwargs = {}
    if uri.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}

    engine = create_engine(uri, **kwargs)
    SessionLocal = scoped_session(
        sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)
    )
    Base.query = SessionLocal.query_property()

    if create_tables:
        Base.metadata.create_all(bind=engine)
        ensure_optional_columns()

    try:
        from modules.behavior_profile import clear_profile_cache

        clear_profile_cache()
    except Exception:
        logger = __import__("logging").getLogger(__name__)
        logger.debug("Behavior profile cache was not cleared", exc_info=True)

    return SessionLocal


def ensure_optional_columns():
    if engine is None:
        return
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    timestamp_type = "TIMESTAMP" if engine.dialect.name == "postgresql" else "DATETIME"
    json_type = "JSON"
    flag_type = "BOOLEAN DEFAULT FALSE" if engine.dialect.name == "postgresql" else "INTEGER DEFAULT 0"
    desired = {
        "messages": {
            "current_text": "TEXT",
            "analyzed_text": "TEXT",
            "send_status": "VARCHAR(32) DEFAULT 'draft'",
            "sent_at": timestamp_type,
            "analyzed_at": timestamp_type,
            "profile_updated": flag_type,
            "slang_detections": json_type,
        },
        "delayed_send_locks": {
            "conversation_id": "INTEGER",
        },
        "intervention_logs": {
            "user_action": "VARCHAR(64)",
        },
        "risk_results": {
            "previous_vector_6d": json_type,
            "mathematical_decision": "VARCHAR(32)",
            "guardrail_reason": "VARCHAR(64)",
            "model_toxicity": json_type,
        },
    }
    with engine.begin() as connection:
        for table, columns in desired.items():
            if table not in tables:
                continue
            existing = {column["name"] for column in inspect(engine).get_columns(table)}
            for name, ddl in columns.items():
                if name in existing:
                    continue
                connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))


def get_session():
    if SessionLocal is None:
        init_db()
    return SessionLocal()


def get_engine():
    if engine is None:
        init_db()
    return engine


def close_session():
    if SessionLocal is not None:
        SessionLocal.remove()


class CRUDMixin:
    @classmethod
    def create(cls, session, **kwargs):
        instance = cls(**kwargs)
        session.add(instance)
        session.commit()
        session.refresh(instance)
        return instance

    def update(self, session, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(f"{self.__class__.__name__} has no column '{key}'")
            setattr(self, key, value)
        session.commit()
        session.refresh(self)
        return self

    def delete(self, session):
        session.delete(self)
        session.commit()

    @classmethod
    def get_by_id(cls, session, record_id):
        return session.get(cls, record_id)

    @classmethod
    def query_filter(cls, session, **filters):
        query = session.query(cls)
        for key, value in filters.items():
            column = getattr(cls, key, None)
            if column is None:
                raise AttributeError(f"{cls.__name__} has no column '{key}'")
            query = query.filter(column == value)
        return query.all()

    def to_dict(self):
        data = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            if isinstance(value, datetime):
                value = value.isoformat()
            data[column.name] = value
        return data


class User(CRUDMixin, Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True)
    username = Column(String(150), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    conversations = relationship(
        "Conversation", back_populates="user", cascade="all, delete-orphan"
    )
    messages = relationship("Message", back_populates="user", cascade="all, delete-orphan")
    behavior_profile = relationship(
        "BehaviorProfile", back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    unknown_slang = relationship(
        "UnknownSlang", back_populates="user", cascade="all, delete-orphan"
    )

    @validates("username")
    def validate_username(self, key, value):
        if not value or not value.strip():
            raise ValueError("username is required")
        return value.strip()

    @validates("email")
    def validate_email(self, key, value):
        if not value or "@" not in value:
            raise ValueError("email must be a valid address")
        return value.strip().lower()

    def set_password(self, password):
        if not password:
            raise ValueError("password is required")
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @classmethod
    def create(cls, session, username, email, password, **kwargs):
        user = cls(username=username, email=email, **kwargs)
        user.set_password(password)
        session.add(user)
        session.commit()
        session.refresh(user)
        return user

    def to_dict(self):
        data = super().to_dict()
        data.pop("password_hash", None)
        return data


class Conversation(CRUDMixin, Base):
    __tablename__ = "conversations"

    conversation_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )

    user = relationship("User", back_populates="conversations")
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.timestamp",
    )

    def get_recent_messages(self, session, limit=50):
        return Message.get_recent_from_conversation(
            session, self.conversation_id, limit=limit
        )


class Message(CRUDMixin, Base):
    __tablename__ = "messages"

    message_id = Column(Integer, primary_key=True)
    conversation_id = Column(
        Integer, ForeignKey("conversations.conversation_id", ondelete="CASCADE"), nullable=False
    )
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    original_text = Column(Text, nullable=False)
    current_text = Column(Text)
    analyzed_text = Column(Text)
    processed_text = Column(Text)
    input_type = Column(input_type_enum, nullable=False)
    transcription_confidence = Column(Float)
    send_status = Column(String(32), nullable=False, default="draft")
    sent_at = Column(DateTime(timezone=True))
    analyzed_at = Column(DateTime(timezone=True))
    profile_updated = Column(Boolean, nullable=False, default=False)
    slang_detections = Column(JSON)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    conversation = relationship("Conversation", back_populates="messages")
    user = relationship("User", back_populates="messages")
    toxicity_result = relationship(
        "ToxicityResult", back_populates="message", uselist=False, cascade="all, delete-orphan"
    )
    risk_result = relationship(
        "RiskResult", back_populates="message", uselist=False, cascade="all, delete-orphan"
    )
    intervention_logs = relationship(
        "InterventionLog", back_populates="message", cascade="all, delete-orphan"
    )
    feedback = relationship(
        "Feedback", back_populates="message", cascade="all, delete-orphan"
    )

    @validates("original_text")
    def validate_original_text(self, key, value):
        if value is None or not str(value).strip():
            raise ValueError("original_text is required")
        return value

    @validates("input_type")
    def validate_input_type(self, key, value):
        if value not in INPUT_TYPES:
            raise ValueError(f"input_type must be one of {INPUT_TYPES}")
        return value

    @validates("transcription_confidence")
    def validate_transcription_confidence(self, key, value):
        if value is None:
            return value
        if not 0 <= float(value) <= 1:
            raise ValueError("transcription_confidence must be between 0 and 1")
        return float(value)

    @classmethod
    def get_recent_from_conversation(cls, session, conversation_id, limit=50):
        return (
            session.query(cls)
            .filter(cls.conversation_id == conversation_id)
            .order_by(cls.timestamp.desc())
            .limit(limit)
            .all()
        )


class ToxicityResult(CRUDMixin, Base):
    __tablename__ = "toxicity_results"
    __table_args__ = tuple(
        CheckConstraint(f"{field} >= 0 AND {field} <= 1", name=f"ck_toxicity_{field}_range")
        for field in TOXICITY_FIELDS
    )

    result_id = Column(Integer, primary_key=True)
    message_id = Column(
        Integer, ForeignKey("messages.message_id", ondelete="CASCADE"), nullable=False
    )
    toxic = Column(Float, nullable=False)
    threat = Column(Float, nullable=False)
    insult = Column(Float, nullable=False)
    severe_toxic = Column(Float, nullable=False)
    obscene = Column(Float, nullable=False)
    identity_hate = Column(Float, nullable=False)

    message = relationship("Message", back_populates="toxicity_result")

    @validates(*TOXICITY_FIELDS)
    def validate_score(self, key, value):
        if value is None or not 0 <= float(value) <= 1:
            raise ValueError(f"{key} must be between 0 and 1")
        return float(value)


class BehaviorProfile(CRUDMixin, Base):
    __tablename__ = "behavior_profiles"

    profile_id = Column(Integer, primary_key=True)
    user_id = Column(
        Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, unique=True
    )
    avg_toxic = Column(Float, nullable=False, default=0)
    avg_threat = Column(Float, nullable=False, default=0)
    avg_insult = Column(Float, nullable=False, default=0)
    avg_severe_toxic = Column(Float, nullable=False, default=0)
    avg_obscene = Column(Float, nullable=False, default=0)
    avg_identity_hate = Column(Float, nullable=False, default=0)
    message_count = Column(Integer, nullable=False, default=0)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )

    user = relationship("User", back_populates="behavior_profile")

    @validates(
        "avg_toxic",
        "avg_threat",
        "avg_insult",
        "avg_severe_toxic",
        "avg_obscene",
        "avg_identity_hate",
    )
    def validate_average(self, key, value):
        if value is None or not 0 <= float(value) <= 1:
            raise ValueError(f"{key} must be between 0 and 1")
        return float(value)

    @validates("message_count")
    def validate_message_count(self, key, value):
        if value is None or int(value) < 0:
            raise ValueError("message_count must be >= 0")
        return int(value)


class RiskResult(CRUDMixin, Base):
    __tablename__ = "risk_results"

    risk_id = Column(Integer, primary_key=True)
    message_id = Column(
        Integer, ForeignKey("messages.message_id", ondelete="CASCADE"), nullable=False
    )
    pacl = Column(Float, nullable=False)
    delta_vector_6d = Column(JSON, nullable=False)
    escalation_feature = Column(Float, nullable=False)
    escalation_probability = Column(Float, nullable=False)
    final_risk = Column(Float, nullable=False)
    decision = Column(decision_action_enum, nullable=False)
    previous_vector_6d = Column(JSON)
    mathematical_decision = Column(String(32))
    guardrail_reason = Column(String(64))
    model_toxicity = Column(JSON)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    message = relationship("Message", back_populates="risk_result")

    @validates("delta_vector_6d")
    def validate_delta_vector(self, key, value):
        if not isinstance(value, (list, tuple)) or len(value) != 6:
            raise ValueError("delta_vector_6d must be a 6-dimensional array")
        return list(value)

    @validates("previous_vector_6d")
    def validate_previous_vector(self, key, value):
        if value is None:
            return value
        if not isinstance(value, (list, tuple)) or len(value) != 6:
            raise ValueError("previous_vector_6d must be a 6-dimensional array")
        return list(value)

    @validates("mathematical_decision")
    def validate_mathematical_decision(self, key, value):
        if value is None:
            return value
        if value not in DECISION_ACTIONS:
            raise ValueError(f"mathematical_decision must be one of {DECISION_ACTIONS}")
        return value

    @validates("decision")
    def validate_decision(self, key, value):
        if value not in DECISION_ACTIONS:
            raise ValueError(f"decision must be one of {DECISION_ACTIONS}")
        return value

    @validates("escalation_probability", "final_risk", "pacl")
    def validate_unit_interval(self, key, value):
        if value is None:
            raise ValueError(f"{key} is required")
        return float(value)


class InterventionLog(CRUDMixin, Base):
    __tablename__ = "intervention_logs"

    id = Column(Integer, primary_key=True)
    message_id = Column(
        Integer, ForeignKey("messages.message_id", ondelete="CASCADE"), nullable=False
    )
    action = Column(decision_action_enum, nullable=False)
    suggested_text = Column(Text)
    context_ids = Column(JSON)
    user_response = Column(user_response_enum)
    user_action = Column(String(64))
    delay_seconds = Column(Integer)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    message = relationship("Message", back_populates="intervention_logs")

    @validates("action")
    def validate_action(self, key, value):
        if value not in DECISION_ACTIONS:
            raise ValueError(f"action must be one of {DECISION_ACTIONS}")
        return value

    @validates("user_response")
    def validate_user_response(self, key, value):
        if value is not None and value not in USER_RESPONSES:
            raise ValueError(f"user_response must be one of {USER_RESPONSES}")
        return value

    @validates("delay_seconds")
    def validate_delay_seconds(self, key, value):
        if value is not None and int(value) < 0:
            raise ValueError("delay_seconds must be >= 0")
        return value

    @validates("context_ids")
    def validate_context_ids(self, key, value):
        if value is None:
            return value
        if not isinstance(value, (list, tuple)):
            raise ValueError("context_ids must be a JSON array")
        return list(value)


class SlangDictionary(CRUDMixin, Base):
    __tablename__ = "slang_dictionary"

    slang_id = Column(Integer, primary_key=True)
    slang_term = Column(String(255), nullable=False)
    meaning = Column(Text, nullable=False)
    category = Column(String(50))
    confidence = Column(Float)

    @validates("slang_term", "meaning")
    def validate_required_text(self, key, value):
        if not value or not str(value).strip():
            raise ValueError(f"{key} is required")
        return value.strip() if key == "slang_term" else value

    @validates("confidence")
    def validate_confidence(self, key, value):
        if value is None:
            return value
        if not 0 <= float(value) <= 1:
            raise ValueError("confidence must be between 0 and 1")
        return float(value)


class UnknownSlang(CRUDMixin, Base):
    __tablename__ = "unknown_slang"

    id = Column(Integer, primary_key=True)
    term = Column(String(255), nullable=False)
    sentence = Column(Text, nullable=False)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    review_status = Column(
        review_status_enum,
        nullable=False,
        default="pending",
    )

    user = relationship("User", back_populates="unknown_slang")

    @validates("term", "sentence")
    def validate_required_text(self, key, value):
        if not value or not str(value).strip():
            raise ValueError(f"{key} is required")
        return value

    @validates("review_status")
    def validate_review_status(self, key, value):
        if value not in REVIEW_STATUSES:
            raise ValueError(f"review_status must be one of {REVIEW_STATUSES}")
        return value


class Feedback(CRUDMixin, Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True)
    message_id = Column(
        Integer, ForeignKey("messages.message_id", ondelete="CASCADE"), nullable=False
    )
    intervention = Column(decision_action_enum, nullable=False)
    user_response = Column(user_response_enum)
    outcome = Column(outcome_enum)
    reward = Column(Float)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    message = relationship("Message", back_populates="feedback")

    @validates("intervention")
    def validate_intervention(self, key, value):
        if value not in DECISION_ACTIONS:
            raise ValueError(f"intervention must be one of {DECISION_ACTIONS}")
        return value

    @validates("user_response")
    def validate_user_response(self, key, value):
        if value is not None and value not in USER_RESPONSES:
            raise ValueError(f"user_response must be one of {USER_RESPONSES}")
        return value

    @validates("outcome")
    def validate_outcome(self, key, value):
        if value is not None and value not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}")
        return value
