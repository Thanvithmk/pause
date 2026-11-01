import logging
from collections import OrderedDict
from datetime import datetime, timedelta, timezone

from sqlalchemy import func

from modules.database import (
    BehaviorProfile,
    InterventionLog,
    Message,
    RiskResult,
    ToxicityResult,
    close_session,
    get_session,
)

logger = logging.getLogger(__name__)

BASELINE_ORDER = (
    ("avg_toxic", "toxic"),
    ("avg_threat", "threat"),
    ("avg_insult", "insult"),
    ("avg_severe_toxic", "severe_toxic"),
    ("avg_obscene", "obscene"),
    ("avg_identity_hate", "identity_hate"),
)
NEUTRAL_BASELINE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
CACHE_SIZE = 256
ESCALATION_DECISIONS = ("WARN", "REWRITE", "DELAYED-SEND")


class BehaviorProfileManager:
    def __init__(self):
        self._cache = OrderedDict()
        get_session()
        logger.info("BehaviorProfileManager initialized")

    def get_or_create_profile(self, user_id):
        session = get_session()
        try:
            profile = (
                session.query(BehaviorProfile)
                .filter(BehaviorProfile.user_id == user_id)
                .one_or_none()
            )
            if profile is None:
                logger.info("Creating neutral behavior profile user_id=%s", user_id)
                profile = BehaviorProfile.create(
                    session,
                    user_id=user_id,
                    avg_toxic=0.0,
                    avg_threat=0.0,
                    avg_insult=0.0,
                    avg_severe_toxic=0.0,
                    avg_obscene=0.0,
                    avg_identity_hate=0.0,
                    message_count=0,
                )
            data = self._to_dict(profile)
            self._cache_put(user_id, data)
            return data
        except Exception as exc:
            session.rollback()
            logger.exception("Failed to get or create profile user_id=%s error=%s", user_id, exc)
            raise
        finally:
            close_session()

    def get_profile(self, user_id):
        cached = self._cache_get(user_id)
        if cached is not None:
            return self._vector_from_dict(cached)

        session = get_session()
        try:
            profile = (
                session.query(BehaviorProfile)
                .filter(BehaviorProfile.user_id == user_id)
                .one_or_none()
            )
            if profile is None:
                logger.info("New user baseline returned user_id=%s", user_id)
                return list(NEUTRAL_BASELINE)
            data = self._to_dict(profile)
            self._cache_put(user_id, data)
            return self._vector_from_dict(data)
        except Exception as exc:
            logger.exception("Failed to get profile user_id=%s error=%s", user_id, exc)
            return list(NEUTRAL_BASELINE)
        finally:
            close_session()

    def update_profile(self, user_id, new_toxicity_vector):
        vector = self._normalize_vector(new_toxicity_vector)
        session = get_session()
        try:
            profile = (
                session.query(BehaviorProfile)
                .filter(BehaviorProfile.user_id == user_id)
                .one_or_none()
            )
            if profile is None:
                profile = BehaviorProfile(
                    user_id=user_id,
                    avg_toxic=0.0,
                    avg_threat=0.0,
                    avg_insult=0.0,
                    avg_severe_toxic=0.0,
                    avg_obscene=0.0,
                    avg_identity_hate=0.0,
                    message_count=0,
                )
                session.add(profile)
                session.flush()

            n = profile.message_count or 0
            for column, key in BASELINE_ORDER:
                previous = float(getattr(profile, column) or 0.0)
                incoming = vector[key]
                updated = incoming if n == 0 else (n * previous + incoming) / (n + 1)
                setattr(profile, column, max(0.0, min(1.0, updated)))
            profile.message_count = n + 1
            session.commit()
            session.refresh(profile)

            data = self._to_dict(profile)
            self._cache_put(user_id, data)
            logger.info(
                "Updated behavior profile user_id=%s message_count=%s",
                user_id,
                profile.message_count,
            )
            return data
        except Exception as exc:
            session.rollback()
            logger.exception("Failed to update profile user_id=%s error=%s", user_id, exc)
            raise
        finally:
            close_session()

    def calculate_averages(self, user_id):
        session = get_session()
        try:
            rows = (
                session.query(ToxicityResult)
                .join(Message, ToxicityResult.message_id == Message.message_id)
                .filter(Message.user_id == user_id)
                .all()
            )
            if not rows:
                logger.info("No toxicity history user_id=%s; using neutral baseline", user_id)
                return list(NEUTRAL_BASELINE)

            totals = {key: 0.0 for _, key in BASELINE_ORDER}
            for row in rows:
                for _, key in BASELINE_ORDER:
                    totals[key] += float(getattr(row, key) or 0.0)
            count = len(rows)
            baseline = [totals[key] / count for _, key in BASELINE_ORDER]
            logger.info("Recalculated averages user_id=%s n=%s", user_id, count)
            return baseline
        except Exception as exc:
            logger.exception("Failed to calculate averages user_id=%s error=%s", user_id, exc)
            return list(NEUTRAL_BASELINE)
        finally:
            close_session()

    def get_user_statistics(self, user_id, days=30):
        cutoff = datetime.now(timezone.utc) - timedelta(days=int(days))
        session = get_session()
        try:
            messages_count = (
                session.query(func.count(Message.message_id))
                .filter(Message.user_id == user_id, Message.timestamp >= cutoff)
                .scalar()
            ) or 0

            avg_toxicity = (
                session.query(func.avg(ToxicityResult.toxic))
                .join(Message, ToxicityResult.message_id == Message.message_id)
                .filter(Message.user_id == user_id, Message.timestamp >= cutoff)
                .scalar()
            )
            avg_toxicity = float(avg_toxicity) if avg_toxicity is not None else 0.0

            escalations_count = (
                session.query(func.count(RiskResult.risk_id))
                .join(Message, RiskResult.message_id == Message.message_id)
                .filter(
                    Message.user_id == user_id,
                    RiskResult.timestamp >= cutoff,
                    RiskResult.decision.in_(ESCALATION_DECISIONS),
                )
                .scalar()
            ) or 0

            interventions_count = (
                session.query(func.count(InterventionLog.id))
                .join(Message, InterventionLog.message_id == Message.message_id)
                .filter(
                    Message.user_id == user_id,
                    InterventionLog.timestamp >= cutoff,
                    InterventionLog.action.in_(ESCALATION_DECISIONS),
                )
                .scalar()
            ) or 0

            stats = {
                "messages_count": int(messages_count),
                "avg_toxicity": avg_toxicity,
                "escalations_count": int(escalations_count),
                "interventions_count": int(interventions_count),
            }
            logger.info("User statistics user_id=%s days=%s stats=%s", user_id, days, stats)
            return stats
        except Exception as exc:
            logger.exception("Failed to get user statistics user_id=%s error=%s", user_id, exc)
            return {
                "messages_count": 0,
                "avg_toxicity": 0.0,
                "escalations_count": 0,
                "interventions_count": 0,
            }
        finally:
            close_session()

    def _to_dict(self, profile):
        return {
            "user_id": profile.user_id,
            "avg_toxic": float(profile.avg_toxic or 0.0),
            "avg_threat": float(profile.avg_threat or 0.0),
            "avg_insult": float(profile.avg_insult or 0.0),
            "avg_severe_toxic": float(profile.avg_severe_toxic or 0.0),
            "avg_obscene": float(profile.avg_obscene or 0.0),
            "avg_identity_hate": float(profile.avg_identity_hate or 0.0),
            "message_count": int(profile.message_count or 0),
        }

    @staticmethod
    def _vector_from_dict(data):
        return [float(data[column]) for column, _ in BASELINE_ORDER]

    @staticmethod
    def _normalize_vector(new_toxicity_vector):
        if isinstance(new_toxicity_vector, dict):
            source = new_toxicity_vector
        elif isinstance(new_toxicity_vector, (list, tuple)) and len(new_toxicity_vector) == 6:
            source = {key: new_toxicity_vector[index] for index, (_, key) in enumerate(BASELINE_ORDER)}
        else:
            raise ValueError("new_toxicity_vector must be a dict or 6-dimensional list")
        return {key: float(source.get(key, 0.0)) for _, key in BASELINE_ORDER}

    def _cache_get(self, user_id):
        if user_id not in self._cache:
            return None
        self._cache.move_to_end(user_id)
        return dict(self._cache[user_id])

    def _cache_put(self, user_id, data):
        self._cache[user_id] = dict(data)
        self._cache.move_to_end(user_id)
        while len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
