import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer
from sqlalchemy import func

from modules.database import (
    Base,
    CRUDMixin,
    Feedback,
    InterventionLog,
    Message,
    RiskResult,
    ToxicityResult,
    close_session,
    get_engine,
    get_session,
    utcnow,
)
from modules.risk_calculation import RiskCalculator

logger = logging.getLogger(__name__)

INTERVENTION_TYPES = ("WARN", "REWRITE", "DELAYED-SEND")
RESPONSE_VALUES = ("accepted", "rejected", "edited")
OUTCOME_VALUES = ("positive", "neutral", "negative")
REWARD_MAP = {
    ("accepted", "positive"): 1.0,
    ("accepted", "neutral"): 0.6,
    ("accepted", "negative"): 0.2,
    ("edited", "positive"): 0.7,
    ("edited", "neutral"): 0.4,
    ("edited", "negative"): 0.1,
    ("rejected", "positive"): 0.3,
    ("rejected", "neutral"): 0.1,
    ("rejected", "negative"): 0.0,
}


class InterventionEffectiveness(CRUDMixin, Base):
    __tablename__ = "intervention_effectiveness"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, unique=True)
    warn_success = Column(Integer, nullable=False, default=0)
    warn_fail = Column(Integer, nullable=False, default=0)
    rewrite_success = Column(Integer, nullable=False, default=0)
    rewrite_fail = Column(Integer, nullable=False, default=0)
    delayed_success = Column(Integer, nullable=False, default=0)
    delayed_fail = Column(Integer, nullable=False, default=0)
    warn_eff = Column(Float, nullable=False, default=0)
    rewrite_eff = Column(Float, nullable=False, default=0)
    delayed_send_eff = Column(Float, nullable=False, default=0)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class BehavioralLearning:
    def __init__(self):
        try:
            get_session()
            Base.metadata.create_all(bind=get_engine())
        except Exception as exc:
            logger.warning("RIL database init warning: %s", exc)
        finally:
            close_session()
        self.risk_calculator = RiskCalculator()
        self.statistics = self._load_global_statistics()
        logger.info("BehavioralLearning initialized stats=%s", self.statistics)

    def record_intervention_outcome(
        self, message_id, intervention_type, user_response, outcome
    ):
        if intervention_type not in INTERVENTION_TYPES:
            raise ValueError(f"intervention_type must be one of {INTERVENTION_TYPES}")
        if user_response not in RESPONSE_VALUES:
            raise ValueError(f"user_response must be one of {RESPONSE_VALUES}")
        if outcome not in OUTCOME_VALUES:
            raise ValueError(f"outcome must be one of {OUTCOME_VALUES}")

        reward = REWARD_MAP[(user_response, outcome)]
        session = get_session()
        try:
            feedback = (
                session.query(Feedback)
                .filter(
                    Feedback.message_id == message_id,
                    Feedback.intervention == intervention_type,
                )
                .order_by(Feedback.id.desc())
                .first()
            )
            if feedback is None:
                feedback = Feedback(
                    message_id=message_id,
                    intervention=intervention_type,
                    user_response=user_response,
                    outcome=outcome,
                    reward=reward,
                )
                session.add(feedback)
            else:
                feedback.user_response = user_response
                feedback.outcome = outcome
                feedback.reward = reward
            log = (
                session.query(InterventionLog)
                .filter(
                    InterventionLog.message_id == message_id,
                    InterventionLog.action == intervention_type,
                )
                .order_by(InterventionLog.id.desc())
                .first()
            )
            if log is not None:
                log.user_response = user_response
            session.commit()
            session.refresh(feedback)

            message = session.query(Message).filter(Message.message_id == message_id).one_or_none()
            user_id = message.user_id if message is not None else None
            logger.info(
                "Recorded RIL outcome message_id=%s type=%s response=%s outcome=%s reward=%.2f",
                message_id,
                intervention_type,
                user_response,
                outcome,
                reward,
            )
        except Exception:
            session.rollback()
            logger.exception("Failed to record intervention outcome message_id=%s", message_id)
            raise
        finally:
            close_session()

        effectiveness = None
        if user_id is not None:
            effectiveness = self.update_ril_statistics(user_id)
        return {
            "feedback_id": feedback.id,
            "reward": reward,
            "effectiveness": effectiveness,
        }

    def update_ril_statistics(self, user_id):
        session = get_session()
        try:
            rows = (
                session.query(Feedback, Message)
                .join(Message, Feedback.message_id == Message.message_id)
                .filter(Message.user_id == user_id)
                .all()
            )
            counts = {
                kind: {"success": 0, "fail": 0}
                for kind in INTERVENTION_TYPES
            }
            for feedback, _message in rows:
                kind = feedback.intervention
                if kind not in counts:
                    continue
                # Heeding the intervention means the original was not sent.
                if feedback.user_response in ("edited", "rejected"):
                    counts[kind]["success"] += 1
                else:
                    counts[kind]["fail"] += 1

            stats = {
                "WARN_eff": self._effectiveness(counts["WARN"]),
                "REWRITE_eff": self._effectiveness(counts["REWRITE"]),
                "DELAYED_SEND_eff": self._effectiveness(counts["DELAYED-SEND"]),
            }
            record = (
                session.query(InterventionEffectiveness)
                .filter(InterventionEffectiveness.user_id == user_id)
                .one_or_none()
            )
            payload = {
                "warn_success": counts["WARN"]["success"],
                "warn_fail": counts["WARN"]["fail"],
                "rewrite_success": counts["REWRITE"]["success"],
                "rewrite_fail": counts["REWRITE"]["fail"],
                "delayed_success": counts["DELAYED-SEND"]["success"],
                "delayed_fail": counts["DELAYED-SEND"]["fail"],
                "warn_eff": stats["WARN_eff"],
                "rewrite_eff": stats["REWRITE_eff"],
                "delayed_send_eff": stats["DELAYED_SEND_eff"],
                "updated_at": utcnow(),
            }
            if record is None:
                InterventionEffectiveness.create(session, user_id=user_id, **payload)
            else:
                record.update(session, **payload)
            logger.info("Updated RIL statistics user_id=%s %s", user_id, stats)
            return stats
        except Exception:
            session.rollback()
            logger.exception("Failed to update RIL statistics user_id=%s", user_id)
            raise
        finally:
            close_session()

    def get_intervention_preferences(self, user_id):
        stats = self.update_ril_statistics(user_id)
        heeded = self._heeded_counts(user_id)
        ranked = sorted(
            stats.items(),
            key=lambda item: (item[1], heeded.get(item[0], 0)),
            reverse=True,
        )
        preferred_key, preferred_score = ranked[0]
        preferred = preferred_key.replace("_eff", "").replace("DELAYED_SEND", "DELAYED-SEND")
        result = {
            "preferred": preferred if preferred_score > 0 else None,
            "scores": stats,
            "ranking": [
                {
                    "intervention": key.replace("_eff", "").replace("DELAYED_SEND", "DELAYED-SEND"),
                    "effectiveness": value,
                }
                for key, value in ranked
            ],
        }
        logger.info("Intervention preference user_id=%s preferred=%s", user_id, result["preferred"])
        return result

    def train_escalation_model_with_feedback(self, training_data):
        logger.info("Retraining escalation model from feedback n=%s", len(training_data or []))
        previous_alpha = self.risk_calculator.alpha
        previous_lambda = self.risk_calculator.lambda_const
        metrics = self.risk_calculator.train_escalation_model(training_data)
        self.risk_calculator.alpha = previous_alpha
        self.risk_calculator.lambda_const = previous_lambda
        metrics["alpha_unchanged"] = self.risk_calculator.alpha == previous_alpha
        metrics["lambda_unchanged"] = self.risk_calculator.lambda_const == previous_lambda
        logger.info(
            "Escalation model retrained b=%s w_E=%s w_P=%s alpha=%s lambda=%s",
            metrics.get("b"),
            metrics.get("w_E"),
            metrics.get("w_P"),
            self.risk_calculator.alpha,
            self.risk_calculator.lambda_const,
        )
        return metrics

    def generate_user_insights(self, user_id):
        this_month = self._window_stats(user_id, 0, 30)
        last_month = self._window_stats(user_id, 30, 60)
        insights = []

        if last_month["avg_toxicity"] > 0:
            delta = (this_month["avg_toxicity"] - last_month["avg_toxicity"]) / last_month["avg_toxicity"]
            percent = abs(delta) * 100
            if delta < 0:
                insights.append(f"Your average toxicity decreased {percent:.0f}% this month")
            elif delta > 0:
                insights.append(f"Your average toxicity increased {percent:.0f}% this month")
            else:
                insights.append("Your average toxicity held steady this month")
        if last_month["warnings"] > 0:
            delta = (this_month["warnings"] - last_month["warnings"]) / last_month["warnings"]
            percent = abs(delta) * 100
            if delta < 0:
                insights.append(f"Warnings reduced by {percent:.0f}% compared to last month")
            else:
                insights.append(f"Warnings increased by {percent:.0f}% compared to last month")
        elif this_month["warnings"] == 0 and this_month["messages"] > 0 and last_month["messages"] > 0:
            insights.append("No warnings this month")

        if (
            last_month["messages"] > 0
            and this_month["avg_toxicity"] < last_month["avg_toxicity"]
        ):
            insights.append("Average toxicity is lower than the previous month")

        preferences = self.get_intervention_preferences(user_id)
        if preferences["preferred"]:
            insights.append(
                f"{preferences['preferred']} is the intervention this user follows most often"
            )

        logger.info("Generated %s insights for user_id=%s", len(insights), user_id)
        return insights

    def _window_stats(self, user_id, start_days_ago, end_days_ago):
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=end_days_ago)
        end = now - timedelta(days=start_days_ago)
        session = get_session()
        try:
            messages = (
                session.query(func.count(Message.message_id))
                .filter(
                    Message.user_id == user_id,
                    Message.timestamp >= start,
                    Message.timestamp < end,
                )
                .scalar()
            ) or 0
            avg_toxicity = (
                session.query(func.avg(ToxicityResult.toxic))
                .join(Message, ToxicityResult.message_id == Message.message_id)
                .filter(
                    Message.user_id == user_id,
                    Message.timestamp >= start,
                    Message.timestamp < end,
                )
                .scalar()
            )
            warnings = (
                session.query(func.count(RiskResult.risk_id))
                .join(Message, RiskResult.message_id == Message.message_id)
                .filter(
                    Message.user_id == user_id,
                    RiskResult.timestamp >= start,
                    RiskResult.timestamp < end,
                    RiskResult.decision == "WARN",
                )
                .scalar()
            ) or 0
            return {
                "messages": int(messages),
                "avg_toxicity": float(avg_toxicity or 0.0),
                "warnings": int(warnings),
            }
        finally:
            close_session()

    def _load_global_statistics(self):
        session = get_session()
        try:
            feedback_count = session.query(func.count(Feedback.id)).scalar() or 0
            users = session.query(func.count(InterventionEffectiveness.id)).scalar() or 0
            return {"feedback_count": int(feedback_count), "tracked_users": int(users)}
        except Exception as exc:
            logger.warning("Could not load RIL statistics: %s", exc)
            return {"feedback_count": 0, "tracked_users": 0}
        finally:
            close_session()

    @staticmethod
    def _effectiveness(counts):
        total = counts["success"] + counts["fail"]
        if total == 0:
            return 0.0
        return counts["success"] / total

    @staticmethod
    def _heeded_counts(user_id):
        session = get_session()
        try:
            record = (
                session.query(InterventionEffectiveness)
                .filter(InterventionEffectiveness.user_id == user_id)
                .one_or_none()
            )
            if record is None:
                return {"WARN_eff": 0, "REWRITE_eff": 0, "DELAYED_SEND_eff": 0}
            return {
                "WARN_eff": record.warn_success,
                "REWRITE_eff": record.rewrite_success,
                "DELAYED_SEND_eff": record.delayed_success,
            }
        finally:
            close_session()
