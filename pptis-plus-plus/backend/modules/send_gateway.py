import logging

from modules.database import (
    Conversation,
    InterventionLog,
    Message,
    RiskResult,
    ToxicityResult,
    close_session,
    get_session,
    utcnow,
)
from modules.delayed_send import DelayedSendManager

logger = logging.getLogger(__name__)

DECISION_ALLOW = "ALLOW"
DECISION_WARN = "WARN"
DECISION_REWRITE = "REWRITE"
DECISION_DELAYED = "DELAYED-SEND"


class SendError(Exception):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _normalize(text):
    return " ".join((text or "").strip().split())


def send_analyzed_message(
    user_id,
    message_id,
    conversation_id,
    text,
    user_action="send",
    delayed_manager=None,
):
    action = (user_action or "send").strip().lower()
    requested_text = _normalize(text)
    if not requested_text:
        raise SendError("text is required", 400)
    if message_id is None or conversation_id is None:
        raise SendError("message_id and conversation_id are required", 400)

    session = get_session()
    try:
        conversation = session.get(Conversation, int(conversation_id))
        if conversation is None:
            raise SendError("Conversation not found", 404)
        if conversation.user_id != user_id:
            raise SendError("You do not own this conversation", 403)

        message = session.get(Message, int(message_id))
        if message is None:
            raise SendError("Message not found", 404)
        if message.user_id != user_id or message.conversation_id != conversation.conversation_id:
            raise SendError("You do not own this message", 403)

        if getattr(message, "send_status", "draft") == "sent":
            raise SendError("Message already sent", 409)

        analyzed_text = _normalize(message.analyzed_text or message.original_text)
        if analyzed_text != requested_text:
            raise SendError("Message was edited after analysis. Re-analyze before sending.", 409)

        risk = (
            session.query(RiskResult)
            .filter(RiskResult.message_id == message.message_id)
            .order_by(RiskResult.timestamp.desc(), RiskResult.risk_id.desc())
            .first()
        )
        if risk is None:
            raise SendError("Message has not been analyzed", 422)

        decision = risk.decision
        lock = DelayedSendManager._blocking_lock(session, user_id, conversation.conversation_id)
        if lock is not None:
            remaining = DelayedSendManager._remaining_seconds(lock)
            if remaining > 0:
                raise SendError(f"Wait {remaining} seconds", 403)
            raise SendError("Timer expired. Edit and re-analyze before sending.", 403)

        if decision == DECISION_DELAYED:
            raise SendError("Delayed-send is active. Sending is not permitted.", 403)
        if decision == DECISION_REWRITE:
            raise SendError(
                "Rewrite required. Sending the original message is not permitted.",
                403,
            )
        if decision == DECISION_WARN and action != "send_anyway":
            raise SendError("Warning requires explicit send_anyway confirmation.", 409)
        if decision == DECISION_ALLOW and action not in ("send", "send_anyway"):
            raise SendError("This decision does not permit the requested action.", 422)
        if decision not in (DECISION_ALLOW, DECISION_WARN):
            raise SendError("Current decision does not permit sending.", 422)

        message.send_status = "sent"
        message.current_text = message.analyzed_text or message.original_text
        message.sent_at = utcnow()
        conversation.updated_at = utcnow()
        session.add(
            InterventionLog(
                message_id=message.message_id,
                action=decision,
                suggested_text=None,
                user_response="accepted",
                user_action=action,
            )
        )
        tox_row = (
            session.query(ToxicityResult)
            .filter(ToxicityResult.message_id == message.message_id)
            .order_by(ToxicityResult.result_id.desc())
            .first()
        )
        tox_vector = None
        if tox_row is not None:
            tox_vector = {
                "toxic": tox_row.toxic,
                "threat": tox_row.threat,
                "insult": tox_row.insult,
                "severe_toxic": tox_row.severe_toxic,
                "obscene": tox_row.obscene,
                "identity_hate": tox_row.identity_hate,
            }
        session.commit()
        session.refresh(message)
        logger.info(
            "Sent message_id=%s user_id=%s decision=%s action=%s",
            message.message_id,
            user_id,
            decision,
            action,
        )
        if tox_vector is not None:
            try:
                from modules.behavior_profile import BehaviorProfileManager

                BehaviorProfileManager().update_profile(user_id, tox_vector)
            except Exception:
                logger.exception("Behavior profile update failed after send message_id=%s", message.message_id)
        if decision == DECISION_WARN:
            try:
                from modules.ril import BehavioralLearning

                BehavioralLearning().record_intervention_outcome(
                    message.message_id,
                    "WARN",
                    "accepted",
                    "neutral",
                )
            except Exception:
                logger.exception("RIL feedback failed message_id=%s", message.message_id)
        return {
            "sent": True,
            "message_id": message.message_id,
            "conversation_id": conversation.conversation_id,
            "user_id": user_id,
            "text": message.current_text or message.original_text,
            "decision": decision,
            "final_risk": risk.final_risk,
            "risk_score": risk.final_risk,
            "pacl": risk.pacl,
            "escalation_feature": risk.escalation_feature,
            "escalation_probability": risk.escalation_probability,
            "send_status": message.send_status,
            "user_action": action,
        }
    except SendError:
        session.rollback()
        raise
    except Exception:
        session.rollback()
        logger.exception("Send failed message_id=%s", message_id)
        raise SendError("Could not send message", 500)
    finally:
        close_session()
