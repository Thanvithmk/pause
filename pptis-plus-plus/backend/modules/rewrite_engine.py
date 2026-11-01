import logging
import os
import re
import time
from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, Text

from modules.database import (
    Base,
    CRUDMixin,
    InterventionLog,
    Message,
    close_session,
    get_engine,
    get_session,
    utcnow,
)
from modules.risk_calculation import RiskCalculator

logger = logging.getLogger(__name__)

REQUIRED_GROQ_MODEL = "llama-3.1-8b-instant"
_ENV_FILES = (
    Path(__file__).resolve().parents[2] / ".env",
    Path(__file__).resolve().parents[1] / ".env",
)
MAX_REWRITE_ATTEMPTS = 2

STRICT_SYSTEM_PROMPT = (
    "Rewrite the user's message so it is safer and less toxic while keeping the same intent. "
    "Do not add insults, threats, slurs, or sarcasm. Return only the rewritten message. "
    "Never tell the user to send the original message."
)

class RewriteServiceError(RuntimeError):
    """Groq rewrite failed. No substitute sentence is returned."""


class RewriteContext(CRUDMixin, Base):
    __tablename__ = "rewrite_context"

    id = Column(Integer, primary_key=True)
    message_id = Column(
        Integer, ForeignKey("messages.message_id", ondelete="CASCADE"), nullable=False
    )
    original_text = Column(Text, nullable=False)
    suggested_text = Column(Text, nullable=False)
    context_ids = Column(JSON)
    validation_score = Column(Float)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=utcnow)


def _setting(name):
    for path in _ENV_FILES:
        if not path.is_file():
            continue
        value = dotenv_values(path).get(name)
        if value is not None and str(value).strip():
            return str(value).strip().strip('"').strip("'")
    return (os.getenv(name) or "").strip().strip('"').strip("'")


class RewriteEngine:
    def __init__(self):
        self.groq_client = None
        self._groq_key = None
        self.groq_model = REQUIRED_GROQ_MODEL
        self.inference = "actual"
        self._ensure_groq()
        self.toxicity_analyzer = self._init_toxicity_analyzer()
        self.risk_calculator = RiskCalculator()
        try:
            Base.metadata.create_all(bind=get_engine())
        except Exception as exc:
            logger.warning("Could not ensure rewrite_context table: %s", exc)
        logger.info(
            "RewriteEngine ready model=%s device_note=remote",
            self.groq_model,
        )

    def rewrite(self, original_text, conversation_id=None, message_id=None, user_id=None):
        context_messages = []
        if conversation_id is not None and message_id is not None:
            context_messages = self.get_conversation_context(conversation_id, message_id)
        suggested = self.generate_candidate_rewrite(original_text, context_messages)
        original_scores = self._analyze_text(original_text)
        validation = self.validate_rewrite(suggested, original_scores)
        if not validation["is_safer"]:
            handled = self.handle_unsafe_rewrite(original_text, context_messages)
            suggested = handled["suggested_text"]
            validation = handled.get("validation") or self.validate_rewrite(
                suggested or "", original_scores
            )
        context_ids = [item["message_id"] for item in context_messages]
        suggestion_analysis = None
        if suggested and validation.get("new_risk") is not None:
            category, color, description = self.risk_calculator.get_risk_category(
                validation["new_risk"]
            )
            suggestion_analysis = {
                "decision": category,
                "final_risk": validation["new_risk"],
                "color_for_ui": color,
                "description": description,
                "approved": category in ("ALLOW", "WARN"),
            }
        approved = bool(suggested) and (
            suggestion_analysis is None or suggestion_analysis.get("approved")
        )
        if not approved:
            suggested = None
        if message_id is not None and suggested:
            self.store_rewrite_attempt(
                message_id,
                original_text,
                suggested,
                context_ids,
                validation.get("validation_score"),
            )
        return {
            "suggested_text": suggested,
            "allow_send_original": False,
            "edit_only": not approved,
            "context_ids": context_ids,
            "validation": validation,
            "suggestion_analysis": suggestion_analysis,
            "inference": self.inference,
            "model": self.groq_model,
        }

    def get_conversation_context(self, conversation_id, message_id, num_previous=5):
        session = get_session()
        try:
            current = (
                session.query(Message)
                .filter(
                    Message.message_id == message_id,
                    Message.conversation_id == conversation_id,
                )
                .one_or_none()
            )
            if current is None:
                raise RewriteServiceError(
                    "rewrite: current message was not found in this conversation"
                )

            rows = (
                session.query(Message)
                .filter(
                    Message.conversation_id == conversation_id,
                    Message.user_id == current.user_id,
                    Message.timestamp < current.timestamp,
                    Message.message_id != message_id,
                )
                .order_by(Message.timestamp.desc())
                .limit(num_previous)
                .all()
            )
            rows = list(reversed(rows))
            context = [
                {
                    "message_id": row.message_id,
                    "user_id": row.user_id,
                    "text": row.original_text,
                    "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                }
                for row in rows
            ]
            logger.info(
                "Loaded %s context messages conversation_id=%s message_id=%s",
                len(context),
                conversation_id,
                message_id,
            )
            return context
        except RewriteServiceError:
            raise
        except Exception as exc:
            logger.exception("Failed to load conversation context")
            raise RewriteServiceError("rewrite: conversation context could not be loaded") from exc
        finally:
            close_session()

    def generate_rewrite_prompt(self, original_text, context_messages):
        history_lines = []
        for item in context_messages or []:
            history_lines.append(f"- ({item.get('timestamp')}) {item.get('text')}")
        history = "\n".join(history_lines) if history_lines else "(no earlier messages)"
        prompt = (
            "You are helping a user rewrite a chat message so it is safer and less toxic.\n"
            "Keep the intended meaning. Do not add new accusations. Do not offer to send the original.\n\n"
            f"Conversation history (same user, chronological):\n{history}\n\n"
            f"Original message:\n{original_text}\n\n"
            "What is problematic: harsh tone, possible insult/threat, or escalating language.\n"
            "Write one safer alternative that a recipient could reasonably receive.\n"
            "Return only the rewritten message."
        )
        logger.info("Built rewrite prompt chars=%s", len(prompt))
        return prompt

    def generate_candidate_rewrite(self, original_text, context_messages):
        prompt = self.generate_rewrite_prompt(original_text, context_messages)
        suggested = self._complete(prompt)
        if suggested and not self._repeats_original(original_text, suggested):
            logger.info("Generated rewrite via Groq chars=%s model=%s", len(suggested), self.groq_model)
            return suggested
        retry = (
            STRICT_SYSTEM_PROMPT
            + "\n\n"
            + prompt
            + "\nDo not copy the original wording."
        )
        suggested = self._complete(retry)
        if not suggested or self._repeats_original(original_text, suggested):
            raise RewriteServiceError("rewrite: Groq did not return a distinct rewrite")
        logger.info("Generated rewrite via Groq chars=%s model=%s", len(suggested), self.groq_model)
        return suggested

    def validate_rewrite(self, suggested_text, original_toxicity_scores):
        original_vector = self._as_vector(original_toxicity_scores)
        new_vector = self._analyze_text(suggested_text)
        original_tb = original_vector[0]
        new_tb = new_vector[0]
        baseline = original_tb
        original_pacl = self.risk_calculator.calculate_pacl(original_tb, baseline)
        new_pacl = self.risk_calculator.calculate_pacl(new_tb, baseline)
        delta = self.risk_calculator.calculate_delta_x(new_vector, original_vector)
        cel = self.risk_calculator.calculate_cel(delta)
        original_p_esc = self.risk_calculator.calculate_escalation_probability(0.0, original_pacl)
        new_p_esc = self.risk_calculator.calculate_escalation_probability(cel, new_pacl)
        original_risk = self.risk_calculator.calculate_final_risk(original_pacl, original_p_esc)
        new_risk = self.risk_calculator.calculate_final_risk(new_pacl, new_p_esc)
        is_safer = new_risk < original_risk
        validation_score = original_risk - new_risk
        result = {
            "is_safer": is_safer,
            "new_risk": new_risk,
            "original_risk": original_risk,
            "validation_score": validation_score,
            "new_toxicity": dict(zip(
                ("toxic", "threat", "insult", "severe_toxic", "obscene", "identity_hate"),
                new_vector,
            )),
        }
        logger.info(
            "Rewrite validation is_safer=%s original_risk=%.4f new_risk=%.4f score=%.4f",
            is_safer,
            original_risk,
            new_risk,
            validation_score,
        )
        return result

    def store_rewrite_attempt(
        self, message_id, original_text, suggested_text, context_ids, validation_score
    ):
        session = get_session()
        try:
            RewriteContext.create(
                session,
                message_id=message_id,
                original_text=original_text,
                suggested_text=suggested_text,
                context_ids=list(context_ids or []),
                validation_score=None if validation_score is None else float(validation_score),
            )
            InterventionLog.create(
                session,
                message_id=message_id,
                action="REWRITE",
                suggested_text=suggested_text,
                context_ids=list(context_ids or []),
            )
            logger.info("Stored rewrite attempt message_id=%s", message_id)
        except Exception:
            session.rollback()
            logger.exception("Failed to store rewrite attempt message_id=%s", message_id)
            raise
        finally:
            close_session()

    def handle_unsafe_rewrite(self, original_text, context_messages):
        original_scores = self._analyze_text(original_text)
        for attempt in range(MAX_REWRITE_ATTEMPTS):
            logger.info("Retrying safer rewrite attempt=%s", attempt + 1)
            prompt = (
                STRICT_SYSTEM_PROMPT
                + "\n\n"
                + self.generate_rewrite_prompt(original_text, context_messages)
                + "\nThe previous rewrite was still too risky. Make it milder."
            )
            candidate = self._complete(prompt)
            if not candidate or self._repeats_original(original_text, candidate):
                continue
            validation = self.validate_rewrite(candidate, original_scores)
            if validation["is_safer"]:
                return {
                    "suggested_text": candidate,
                    "edit_only": False,
                    "allow_send_original": False,
                    "validation": validation,
                }

        logger.warning("Rewrite remained high-risk; switching to edit-only mode")
        return {
            "suggested_text": None,
            "edit_only": True,
            "allow_send_original": False,
            "validation": {
                "is_safer": False,
                "new_risk": None,
                "validation_score": 0.0,
            },
            "ui": {
                "prompt": "Please edit this message before sending. Sending the original is not available.",
            },
        }

    def _complete(self, prompt):
        self._ensure_groq()
        return self._complete_groq(prompt)

    def _complete_groq(self, prompt):
        last_error = None
        for attempt in range(2):
            try:
                response = self.groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=[
                        {"role": "system", "content": STRICT_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.4,
                    max_tokens=256,
                )
                text = (response.choices[0].message.content or "").strip()
                return text or None
            except RewriteServiceError:
                raise
            except Exception as exc:
                last_error = exc
                message = str(exc).lower()
                if "429" in message or "rate" in message:
                    logger.warning("Groq rate limit; retrying")
                    time.sleep(1.5 * (attempt + 1))
                    continue
                logger.exception("Groq request failed")
                break
        raise RewriteServiceError(f"rewrite: Groq request failed for {self.groq_model}") from last_error

    def _analyze_text(self, text):
        from modules.toxicity_model import ToxicityAnalysisError

        labels = ("toxic", "threat", "insult", "severe_toxic", "obscene", "identity_hate")
        if not text:
            raise ToxicityAnalysisError("toxicity analysis received empty rewrite text")
        if self.toxicity_analyzer is None:
            raise ToxicityAnalysisError(f"toxicity model is not loaded")
        scores = self.toxicity_analyzer.analyze(text)
        return [float(scores[label]) for label in labels]

    @staticmethod
    def _as_vector(scores):
        labels = ("toxic", "threat", "insult", "severe_toxic", "obscene", "identity_hate")
        if isinstance(scores, dict):
            return [float(scores.get(label, 0.0)) for label in labels]
        if isinstance(scores, (list, tuple)) and len(scores) == 6:
            return [float(value) for value in scores]
        raise RewriteServiceError("rewrite: toxicity scores are incomplete")

    def _ensure_groq(self):
        api_key = _setting("GROQ_API_KEY")
        configured = _setting("GROQ_MODEL") or REQUIRED_GROQ_MODEL
        if configured != REQUIRED_GROQ_MODEL:
            raise RewriteServiceError(
                f"rewrite: GROQ_MODEL must be {REQUIRED_GROQ_MODEL}, not {configured}"
            )
        self.groq_model = REQUIRED_GROQ_MODEL
        if not api_key:
            raise RewriteServiceError("rewrite: GROQ_API_KEY is not configured")
        if self.groq_client is not None and self._groq_key == api_key:
            return self.groq_client
        try:
            from groq import Groq

            self.groq_client = Groq(api_key=api_key)
            self._groq_key = api_key
            logger.info("Groq rewrite client ready model=%s", self.groq_model)
            return self.groq_client
        except Exception as exc:
            logger.exception("Groq client could not be created")
            raise RewriteServiceError("rewrite: Groq client could not be created") from exc

    @staticmethod
    def _repeats_original(original_text, candidate):
        original = re.sub(r"\s+", " ", (original_text or "").strip().lower())
        rewritten = re.sub(r"\s+", " ", (candidate or "").strip().lower())
        if not original or not rewritten:
            return False
        return original in rewritten or rewritten in original

    @staticmethod
    def _init_toxicity_analyzer():
        from modules.toxicity_model import ToxicityAnalyzer

        return ToxicityAnalyzer()
