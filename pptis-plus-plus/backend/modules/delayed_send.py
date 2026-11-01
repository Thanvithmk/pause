import logging
import math
from datetime import timedelta, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, Text

from modules.database import (
    Base,
    CRUDMixin,
    Message,
    ToxicityResult,
    close_session,
    get_engine,
    get_session,
    utcnow,
)
from modules.risk_calculation import RiskCalculator

logger = logging.getLogger(__name__)

CALMING_EXERCISES = (
    {
        "exercise": "Slow breathing",
        "duration": "20-30 sec",
        "duration_seconds": 25,
        "instructions": "Inhale through your nose for 4 seconds, hold for 2, exhale through your mouth for 6. Repeat until the timer eases.",
    },
    {
        "exercise": "5-4-3-2-1 Grounding",
        "duration": "30-45 sec",
        "duration_seconds": 40,
        "instructions": "Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, and 1 you taste or remember fondly.",
    },
    {
        "exercise": "Shoulder release",
        "duration": "15-20 sec",
        "duration_seconds": 18,
        "instructions": "Shrug your shoulders up to your ears, hold, then drop them and unclench your jaw and hands.",
    },
    {
        "exercise": "Pause & reread",
        "duration": "20 sec",
        "duration_seconds": 20,
        "instructions": "Read the message once as if you were the recipient. Ask whether you still want these exact words to land.",
    },
)


class DelayedSendLock(CRUDMixin, Base):
    __tablename__ = "delayed_send_locks"

    lock_id = Column(Integer, primary_key=True)
    message_id = Column(
        Integer, ForeignKey("messages.message_id", ondelete="CASCADE"), nullable=False
    )
    conversation_id = Column(Integer, ForeignKey("conversations.conversation_id", ondelete="CASCADE"))
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    original_text = Column(Text, nullable=False)
    final_risk = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    released_at = Column(DateTime(timezone=True))
    is_active = Column(Boolean, nullable=False, default=True)


class DelayedSendManager:
    def __init__(self, delay_duration=60):
        self.delay_duration = self._load_delay(delay_duration)
        self.risk_calculator = RiskCalculator()
        self.toxicity_analyzer = self._init_toxicity_analyzer()
        self.profile_manager = self._init_profile_manager()
        try:
            get_session()
            Base.metadata.create_all(bind=get_engine())
            from modules.database import ensure_optional_columns

            ensure_optional_columns()
        except Exception as exc:
            logger.warning("Delayed-send database init warning: %s", exc)
        finally:
            close_session()
        logger.info("DelayedSendManager initialized delay=%ss", self.delay_duration)

    def create_delayed_send_state(
        self, message_id, user_id, original_text, final_risk, conversation_id=None
    ):
        now = utcnow()
        expires_at = now + timedelta(seconds=self.delay_duration)
        session = get_session()
        try:
            if conversation_id is None:
                message = session.get(Message, message_id)
                conversation_id = None if message is None else message.conversation_id
            self._deactivate_user_locks(session, user_id, conversation_id)
            lock = DelayedSendLock(
                message_id=message_id,
                conversation_id=conversation_id,
                user_id=user_id,
                original_text=original_text or "",
                final_risk=float(final_risk),
                created_at=now,
                expires_at=expires_at,
                is_active=True,
            )
            session.add(lock)
            session.commit()
            session.refresh(lock)
            payload = {
                "lock_id": lock.lock_id,
                "countdown_seconds": self.delay_duration,
                "message_id": message_id,
                "conversation_id": conversation_id,
                "expires_at": lock.expires_at.isoformat(),
                "allow_send_anyway": False,
                "exercises": self.get_calming_exercises(),
            }
            logger.info(
                "Created delayed-send lock_id=%s message_id=%s expires_at=%s",
                lock.lock_id,
                message_id,
                lock.expires_at,
            )
            return payload
        except Exception:
            session.rollback()
            logger.exception("Failed to create delayed-send state message_id=%s", message_id)
            raise
        finally:
            close_session()

    def get_delayed_send_status(self, message_id):
        session = get_session()
        try:
            lock = self._active_lock(session, message_id)
            if lock is None:
                return None
            remaining = self._remaining_seconds(lock)
            status = {
                "is_locked": True,
                "remaining_seconds": remaining,
                "lock_id": lock.lock_id,
                "expired": remaining == 0,
                "allow_send_anyway": False,
            }
            logger.info(
                "Delayed-send status message_id=%s remaining=%s",
                message_id,
                remaining,
            )
            return status
        except Exception:
            logger.exception("Failed to read delayed-send status message_id=%s", message_id)
            raise
        finally:
            close_session()

    def attempt_send(self, message_id, user_id, conversation_id=None):
        session = get_session()
        try:
            message = session.get(Message, message_id)
            if conversation_id is None and message is not None:
                conversation_id = message.conversation_id
            lock = self._blocking_lock(session, user_id, conversation_id)
            if lock is None:
                logger.info("Send allowed; no delayed-send lock message_id=%s", message_id)
                return {
                    "allowed": True,
                    "remaining_seconds": 0,
                    "next_action": "send",
                    "allow_send_anyway": False,
                }

            remaining = self._remaining_seconds(lock)
            if remaining > 0:
                logger.info(
                    "Send blocked; wait %ss message_id=%s lock_id=%s",
                    remaining,
                    message_id,
                    lock.lock_id,
                )
                return {
                    "allowed": False,
                    "remaining_seconds": remaining,
                    "next_action": "wait",
                    "error": f"Wait {remaining} seconds",
                    "status_code": 403,
                    "allow_send_anyway": False,
                }

            logger.info(
                "Timer expired; re-analysis required message_id=%s lock_id=%s",
                message_id,
                lock.lock_id,
            )
            return {
                "allowed": False,
                "remaining_seconds": 0,
                "next_action": "reanalyze",
                "error": "Timer expired. Edit and re-analyze before sending.",
                "status_code": 403,
                "allow_send_anyway": False,
            }
        except Exception:
            logger.exception("attempt_send failed message_id=%s", message_id)
            raise
        finally:
            close_session()

    def unlock_after_timer(self, message_id):
        session = get_session()
        try:
            lock = self._active_lock(session, message_id)
            if lock is None:
                logger.info("No active lock to release message_id=%s", message_id)
                return {"unlocked": False, "sent": False, "message_id": message_id}
            lock.is_active = False
            lock.released_at = utcnow()
            session.commit()
            logger.info(
                "Unlocked delayed-send message_id=%s lock_id=%s sent=False",
                message_id,
                lock.lock_id,
            )
            return {
                "unlocked": True,
                "sent": False,
                "message_id": message_id,
                "lock_id": lock.lock_id,
                "next_action": "edit_or_reanalyze",
                "allow_send_anyway": False,
            }
        except Exception:
            session.rollback()
            logger.exception("Failed to unlock message_id=%s", message_id)
            raise
        finally:
            close_session()

    def handle_re_analysis_after_delay(self, message_id, edited_text):
        session = get_session()
        try:
            lock = self._active_lock(session, message_id)
            message = session.query(Message).filter(Message.message_id == message_id).one_or_none()
            if message is None:
                raise ValueError(f"Message {message_id} not found")
            user_id = message.user_id
            text = edited_text if edited_text is not None else (lock.original_text if lock else message.original_text)
        finally:
            close_session()

        logger.info("Re-analyzing after delay message_id=%s", message_id)
        toxicity = self._analyze_text(text)
        baseline = [0.0] * 6
        if self.profile_manager is not None:
            try:
                baseline = self.profile_manager.get_profile(user_id)
            except Exception as exc:
                logger.warning("Baseline unavailable: %s", exc)

        pacl = self.risk_calculator.calculate_pacl(toxicity[0], baseline[0] if baseline else 0.0)
        previous = self._previous_toxicity_vector(user_id, message_id)
        if previous is None:
            cel = 0.0
        else:
            delta = self.risk_calculator.calculate_delta_x(toxicity, previous)
            cel = self.risk_calculator.calculate_cel(delta)
        p_esc = self.risk_calculator.calculate_escalation_probability(cel, pacl)
        final_risk = self.risk_calculator.calculate_final_risk(pacl, p_esc)
        category, color, description = self.risk_calculator.get_risk_category(final_risk)

        if category == "ALLOW":
            self.unlock_after_timer(message_id)
            logger.info("Re-analysis allowed send message_id=%s risk=%.4f", message_id, final_risk)
            return {
                "allowed": True,
                "sent": False,
                "lock_cleared": True,
                "final_risk": final_risk,
                "decision": category,
                "next_action": "send",
                "color_for_ui": color,
                "description": description,
                "allow_send_anyway": False,
            }

        if category == "DELAYED-SEND":
            new_lock = self.create_delayed_send_state(message_id, user_id, text, final_risk)
            logger.info("Re-analysis still high-risk; chained delay message_id=%s", message_id)
            return {
                "allowed": False,
                "sent": False,
                "lock_cleared": False,
                "final_risk": final_risk,
                "decision": category,
                "next_action": "wait",
                "new_lock": new_lock,
                "color_for_ui": color,
                "description": description,
                "allow_send_anyway": False,
            }

        self.unlock_after_timer(message_id)
        logger.info(
            "Re-analysis requires %s not send message_id=%s risk=%.4f",
            category,
            message_id,
            final_risk,
        )
        return {
            "allowed": False,
            "sent": False,
            "lock_cleared": True,
            "final_risk": final_risk,
            "decision": category,
            "next_action": "rewrite" if category == "REWRITE" else "warn",
            "color_for_ui": color,
            "description": description,
            "allow_send_anyway": False,
        }

    def get_calming_exercises(self):
        exercises = [dict(item) for item in CALMING_EXERCISES]
        logger.info("Returned %s calming exercises", len(exercises))
        return exercises

    def _previous_toxicity_vector(self, user_id, message_id):
        session = get_session()
        try:
            current = session.query(Message).filter(Message.message_id == message_id).one_or_none()
            if current is None:
                return None
            previous = (
                session.query(Message)
                .filter(
                    Message.user_id == user_id,
                    Message.conversation_id == current.conversation_id,
                    Message.timestamp < current.timestamp,
                )
                .order_by(Message.timestamp.desc())
                .first()
            )
            if previous is None:
                return None
            result = (
                session.query(ToxicityResult)
                .filter(ToxicityResult.message_id == previous.message_id)
                .one_or_none()
            )
            if result is None:
                return None
            return [
                float(result.toxic),
                float(result.threat),
                float(result.insult),
                float(result.severe_toxic),
                float(result.obscene),
                float(result.identity_hate),
            ]
        except Exception as exc:
            logger.warning("Previous toxicity unavailable: %s", exc)
            return None
        finally:
            close_session()

    def _analyze_text(self, text):
        labels = ("toxic", "threat", "insult", "severe_toxic", "obscene", "identity_hate")
        if self.toxicity_analyzer is None:
            return [0.0] * 6
        try:
            scores = self.toxicity_analyzer.analyze(text or "")
            return [float(scores.get(label, 0.0)) for label in labels]
        except Exception as exc:
            logger.exception("Toxicity analysis failed in delayed-send: %s", exc)
            return [0.0] * 6

    @staticmethod
    def _active_lock(session, message_id):
        return (
            session.query(DelayedSendLock)
            .filter(
                DelayedSendLock.message_id == message_id,
                DelayedSendLock.is_active.is_(True),
            )
            .order_by(DelayedSendLock.lock_id.desc())
            .first()
        )

    def get_user_lock_status(self, user_id, conversation_id=None):
        session = get_session()
        try:
            lock = self._blocking_lock(session, user_id, conversation_id)
            if lock is None:
                return {
                    "is_locked": False,
                    "remaining_seconds": 0,
                    "allow_send_anyway": False,
                }
            remaining = self._remaining_seconds(lock)
            return {
                "is_locked": True,
                "remaining_seconds": remaining,
                "lock_id": lock.lock_id,
                "message_id": lock.message_id,
                "conversation_id": lock.conversation_id,
                "expired": remaining == 0,
                "exercises": self.get_calming_exercises(),
                "allow_send_anyway": False,
            }
        finally:
            close_session()

    @staticmethod
    def _blocking_lock(session, user_id, conversation_id=None):
        query = session.query(DelayedSendLock).filter(
            DelayedSendLock.user_id == user_id,
            DelayedSendLock.is_active.is_(True),
        )
        if conversation_id is not None:
            query = query.filter(
                (DelayedSendLock.conversation_id == conversation_id)
                | (DelayedSendLock.conversation_id.is_(None))
            )
        return query.order_by(DelayedSendLock.lock_id.desc()).first()

    @staticmethod
    def _deactivate_user_locks(session, user_id, conversation_id=None):
        now = utcnow()
        query = session.query(DelayedSendLock).filter(
            DelayedSendLock.user_id == user_id,
            DelayedSendLock.is_active.is_(True),
        )
        if conversation_id is not None:
            query = query.filter(
                (DelayedSendLock.conversation_id == conversation_id)
                | (DelayedSendLock.conversation_id.is_(None))
            )
        for row in query.all():
            row.is_active = False
            row.released_at = now

    @staticmethod
    def _deactivate_locks(session, message_id):
        now = utcnow()
        rows = (
            session.query(DelayedSendLock)
            .filter(
                DelayedSendLock.message_id == message_id,
                DelayedSendLock.is_active.is_(True),
            )
            .all()
        )
        for row in rows:
            row.is_active = False
            row.released_at = now

    @staticmethod
    def _as_aware(value):
        if value is None:
            return utcnow()
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @classmethod
    def _remaining_seconds(cls, lock):
        remaining = (cls._as_aware(lock.expires_at) - utcnow()).total_seconds()
        return max(0, int(math.ceil(remaining)))

    @staticmethod
    def _load_delay(delay_duration):
        try:
            from config import get_config

            configured = int(get_config().DELAYED_SEND_DURATION)
            if delay_duration == 60:
                return configured
        except Exception as exc:
            logger.warning("Using provided delayed-send duration: %s", exc)
        return int(delay_duration)

    @staticmethod
    def _init_toxicity_analyzer():
        try:
            from modules.toxicity_model import ToxicityAnalyzer

            return ToxicityAnalyzer()
        except Exception as exc:
            logger.warning("ToxicityAnalyzer unavailable: %s", exc)
            return None

    @staticmethod
    def _init_profile_manager():
        try:
            from modules.behavior_profile import BehaviorProfileManager

            return BehaviorProfileManager()
        except Exception as exc:
            logger.warning("BehaviorProfileManager unavailable: %s", exc)
            return None
