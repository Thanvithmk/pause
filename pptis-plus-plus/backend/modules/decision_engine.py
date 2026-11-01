import logging
import math

from modules.database import RiskResult, close_session, get_session
from modules.risk_calculation import RISK_CATEGORIES, RiskCalculator

logger = logging.getLogger(__name__)

DECISION_ORDER = ("ALLOW", "WARN", "REWRITE", "DELAYED-SEND")


def _load_decision_config():
    try:
        from config import get_config

        config = get_config()
        return {
            "T_allow": float(config.THRESHOLD_ALLOW),
            "T_warn": float(config.THRESHOLD_WARN),
            "T_rewrite": float(config.THRESHOLD_REWRITE),
            "delayed_send_duration": int(config.DELAYED_SEND_DURATION),
        }
    except Exception as exc:
        logger.warning("Using default decision thresholds: %s", exc)
        return {
            "T_allow": 0.30,
            "T_warn": 0.55,
            "T_rewrite": 0.80,
            "delayed_send_duration": 60,
        }


class DecisionEngine:
    def __init__(self):
        settings = _load_decision_config()
        self.T_allow = settings["T_allow"]
        self.T_warn = settings["T_warn"]
        self.T_rewrite = settings["T_rewrite"]
        self.delayed_send_duration = settings["delayed_send_duration"]
        self.risk_calculator = RiskCalculator()
        self.rewrite_engine = None
        logger.info(
            "DecisionEngine initialized T_allow=%.2f T_warn=%.2f T_rewrite=%.2f delay=%ss",
            self.T_allow,
            self.T_warn,
            self.T_rewrite,
            self.delayed_send_duration,
        )

    def make_decision(self, final_risk_score, toxicity_vector=None, safety_flags=None):
        score = self._validate_risk(final_risk_score)
        if score < self.T_allow:
            decision = "ALLOW"
        elif score < self.T_warn:
            decision = "WARN"
        elif score < self.T_rewrite:
            decision = "REWRITE"
        else:
            decision = "DELAYED-SEND"

        decision, guardrail = self._apply_safety_guardrail(
            decision, toxicity_vector, safety_flags
        )

        confidence = self._confidence(score, decision)
        ui = RISK_CATEGORIES[decision]
        result = {
            "decision": decision,
            "risk_score": score,
            "confidence": confidence,
            "color_for_ui": ui["color_for_ui"],
            "description": ui["description"],
            "guardrail": guardrail,
        }
        logger.info(
            "Decision %s risk=%.4f confidence=%.4f",
            decision,
            score,
            confidence,
        )
        return result

    def apply_decision(
        self, decision, message_id, user_id, conversation_id, original_text
    ):
        if decision not in DECISION_ORDER:
            raise ValueError(f"Unknown decision '{decision}'")
        logger.info(
            "Applying %s message_id=%s user_id=%s conversation_id=%s",
            decision,
            message_id,
            user_id,
            conversation_id,
        )
        handlers = {
            "ALLOW": self._apply_allow,
            "WARN": self._apply_warn,
            "REWRITE": self._apply_rewrite,
            "DELAYED-SEND": self._apply_delayed_send,
        }
        action = handlers[decision](
            message_id, user_id, conversation_id, original_text or ""
        )
        logger.info("Applied %s next_steps=%s", decision, action.get("next_steps"))
        return action

    def log_decision(self, message_id, decision, risk_score):
        if decision not in DECISION_ORDER:
            raise ValueError(f"Unknown decision '{decision}'")
        score = self._validate_risk(risk_score)
        session = get_session()
        try:
            risk = (
                session.query(RiskResult)
                .filter(RiskResult.message_id == message_id)
                .one_or_none()
            )
            if risk is None:
                risk = RiskResult(
                    message_id=message_id,
                    pacl=score,
                    delta_vector_6d=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    escalation_feature=0.0,
                    escalation_probability=0.0,
                    final_risk=score,
                    decision=decision,
                )
                session.add(risk)
            else:
                risk.final_risk = score
                risk.decision = decision
            session.commit()
            logger.info(
                "Logged decision message_id=%s decision=%s risk=%.4f",
                message_id,
                decision,
                score,
            )
            return risk.to_dict()
        except Exception:
            session.rollback()
            logger.exception("Failed to log decision message_id=%s", message_id)
            raise
        finally:
            close_session()

    def _apply_allow(self, message_id, user_id, conversation_id, original_text):
        return {
            "action": "ALLOW",
            "message_id": message_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "original_text": original_text,
            "send": True,
            "store": True,
            "ui": {
                "title": "Message allowed",
                "description": RISK_CATEGORIES["ALLOW"]["description"],
                "color_for_ui": RISK_CATEGORIES["ALLOW"]["color_for_ui"],
            },
            "next_steps": ["store_message", "deliver_message"],
        }

    def _apply_warn(self, message_id, user_id, conversation_id, original_text):
        warning = (
            "This message may come across as harsh. Consider pausing before you send it."
        )
        return {
            "action": "WARN",
            "message_id": message_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "original_text": original_text,
            "send": False,
            "warning_message": warning,
            "ui": {
                "title": "Warning",
                "description": RISK_CATEGORIES["WARN"]["description"],
                "color_for_ui": RISK_CATEGORIES["WARN"]["color_for_ui"],
                "prompt": warning,
            },
            "next_steps": ["show_warning", "await_user_confirmation"],
        }

    def _apply_rewrite(self, message_id, user_id, conversation_id, original_text):
        return {
            "action": "REWRITE",
            "message_id": message_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "original_text": original_text,
            "send": False,
            "suggested_text": None,
            "ui": {
                "title": "Suggested rewrite",
                "description": RISK_CATEGORIES["REWRITE"]["description"],
                "color_for_ui": RISK_CATEGORIES["REWRITE"]["color_for_ui"],
                "prompt": "A calmer rewrite is recommended before sending.",
            },
            "next_steps": ["call_rewrite_engine", "present_suggestion", "await_user_choice"],
        }

    def _apply_delayed_send(self, message_id, user_id, conversation_id, original_text):
        delay_seconds = self.delayed_send_duration
        return {
            "action": "DELAYED-SEND",
            "message_id": message_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "original_text": original_text,
            "send": False,
            "delayed_send": True,
            "delay_seconds": delay_seconds,
            "state": "pending_delay",
            "ui": {
                "title": "Send delayed",
                "description": RISK_CATEGORIES["DELAYED-SEND"]["description"],
                "color_for_ui": RISK_CATEGORIES["DELAYED-SEND"]["color_for_ui"],
                "prompt": f"This message will wait {delay_seconds} seconds. Use that time for a short reset exercise.",
            },
            "next_steps": [
                "create_delayed_send_state",
                "start_timer",
                "offer_exercises",
            ],
        }

    def _call_rewrite(self, original_text):
        for method_name in ("rewrite", "generate", "suggest"):
            method = getattr(self.rewrite_engine, method_name, None)
            if callable(method):
                result = method(original_text)
                if isinstance(result, dict):
                    return result.get("suggested_text") or result.get("text") or original_text
                return result
        raise RuntimeError("RewriteEngine has no rewrite method")

    def _confidence(self, score, decision):
        lower, upper = {
            "ALLOW": (0.0, self.T_allow),
            "WARN": (self.T_allow, self.T_warn),
            "REWRITE": (self.T_warn, self.T_rewrite),
            "DELAYED-SEND": (self.T_rewrite, 1.0),
        }[decision]
        width = max(upper - lower, 1e-9)
        center = (lower + upper) / 2.0
        return round(max(0.0, min(1.0, 1.0 - abs(score - center) / (width / 2.0))), 4)

    @staticmethod
    def _apply_safety_guardrail(decision, toxicity_vector, safety_flags):
        """Escalate from 6-D features. Does not change R.

        Explicit high-threat or identity-hate mass-harm signals may raise the
        intervention even when PACL is computed only from Tb. This never
        lowers a DELAYED-SEND decision and never uses user history to
        normalize a present threat.
        """
        flags = safety_flags or {}
        if flags.get("quoted") or flags.get("negated") or flags.get("playful_idiom"):
            return decision, None
        if not toxicity_vector:
            return decision, None
        threat = float(toxicity_vector.get("threat", 0.0) or 0.0)
        identity = float(toxicity_vector.get("identity_hate", 0.0) or 0.0)
        severe = float(toxicity_vector.get("severe_toxic", 0.0) or 0.0)
        rank = {name: index for index, name in enumerate(DECISION_ORDER)}
        escalated = decision
        reason = None
        if threat >= 0.75 or severe >= 0.70:
            escalated = "DELAYED-SEND"
            reason = "threat_vector_guardrail"
        elif identity >= 0.70:
            escalated = "DELAYED-SEND"
            reason = "identity_harm_guardrail"
        if rank[escalated] > rank[decision]:
            return escalated, reason
        return decision, None

    @staticmethod
    def _validate_risk(value):
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("final_risk_score must be a number") from exc
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("final_risk_score must be a finite number in [0, 1]")
        return score

    @staticmethod
    def _init_rewrite_engine():
        try:
            from modules import rewrite_engine

            engine_cls = getattr(rewrite_engine, "RewriteEngine", None)
            if engine_cls is None:
                logger.info("RewriteEngine not available yet")
                return None
            return engine_cls()
        except Exception as exc:
            logger.warning("Could not initialize RewriteEngine: %s", exc)
            return None
