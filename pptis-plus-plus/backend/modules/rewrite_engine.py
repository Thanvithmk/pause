import logging
import os
import re
import time

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

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "llama3")
MAX_REWRITE_ATTEMPTS = 2

REWRITE_TEMPLATES = (
    "I want to be honest, but more calmly: {softened}",
    "I'm frustrated about this and I'd rather say it this way: {softened}",
    "Can we talk about this? Here's what I mean without the heat: {softened}",
)

STRICT_SYSTEM_PROMPT = (
    "Rewrite the user's message so it is safer and less toxic while keeping the same intent. "
    "Do not add insults, threats, slurs, or sarcasm. Return only the rewritten message. "
    "Never tell the user to send the original message."
)

HARSH_WORDS = (
    "stupid",
    "idiot",
    "hate",
    "kill",
    "dumb",
    "shut up",
    "worthless",
    "pathetic",
    "loser",
    "moron",
    "disgusting",
    "incompetent",
    "clown",
    "hopeless",
)


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


class RewriteEngine:
    def __init__(self):
        self.groq_client = self._init_groq()
        self.templates = list(REWRITE_TEMPLATES)
        self.toxicity_analyzer = self._init_toxicity_analyzer()
        self.risk_calculator = RiskCalculator()
        try:
            Base.metadata.create_all(bind=get_engine())
        except Exception as exc:
            logger.warning("Could not ensure rewrite_context table: %s", exc)
        logger.info(
            "RewriteEngine ready groq=%s templates=%s toxicity=%s",
            self.groq_client is not None,
            len(self.templates),
            self.toxicity_analyzer is not None,
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
                logger.warning(
                    "Context skipped; message_id=%s not in conversation_id=%s",
                    message_id,
                    conversation_id,
                )
                return []

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
        except Exception:
            logger.exception("Failed to load conversation context")
            return []
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
        try:
            suggested = self._complete(prompt)
            if suggested:
                logger.info("Generated rewrite via LLM chars=%s", len(suggested))
                return suggested
        except Exception as exc:
            logger.exception("LLM rewrite failed, using template: %s", exc)
        suggested = self._template_rewrite(original_text)
        logger.info("Generated template rewrite")
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
            try:
                candidate = self._complete(prompt) or self._template_rewrite(original_text)
            except Exception:
                candidate = self._template_rewrite(original_text)
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
        if self.groq_client is not None:
            return self._complete_groq(prompt)
        if LOCAL_LLM_URL:
            return self._complete_local(prompt)
        return None

    def _complete_groq(self, prompt):
        last_error = None
        for attempt in range(2):
            try:
                response = self.groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": STRICT_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.4,
                    max_tokens=256,
                )
                text = (response.choices[0].message.content or "").strip()
                return text or None
            except Exception as exc:
                last_error = exc
                message = str(exc).lower()
                if "429" in message or "rate" in message:
                    logger.warning("Groq free-tier rate limit; retrying")
                    time.sleep(1.5 * (attempt + 1))
                    continue
                logger.exception("Groq request failed")
                break
        if last_error:
            raise last_error
        return None

    def _complete_local(self, prompt):
        import json
        from urllib import request

        payload = json.dumps(
            {
                "model": LOCAL_LLM_MODEL,
                "messages": [
                    {"role": "system", "content": STRICT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.4,
            }
        ).encode("utf-8")
        req = request.Request(
            LOCAL_LLM_URL.rstrip("/") + "/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
        return (body["choices"][0]["message"]["content"] or "").strip()

    def _template_rewrite(self, original_text):
        softened = original_text or ""
        for word in HARSH_WORDS:
            softened = re.sub(r"\b" + re.escape(word) + r"\b", "", softened, flags=re.IGNORECASE)
        softened = re.sub(r"\s+", " ", softened).strip()
        still_targeted = bool(
            re.search(r"\b(you|your|yourself|npc|cringe)\b", softened, re.I)
        )
        if still_targeted or not softened:
            softened = "I disagree with this approach and would like to handle the work differently."
        else:
            softened = softened[0].upper() + softened[1:]
        candidate = self.templates[0].format(softened=softened)
        try:
            from modules.safety_patterns import SafetyPatternAnalyzer

            if SafetyPatternAnalyzer().score(candidate).get("label") in ("REWRITE", "DELAYED-SEND"):
                return self.templates[0].format(
                    softened="I disagree with this approach and would like to handle the work differently."
                )
        except Exception:
            pass
        return candidate

    def _analyze_text(self, text):
        labels = ("toxic", "threat", "insult", "severe_toxic", "obscene", "identity_hate")
        if not text or self.toxicity_analyzer is None:
            return [0.0] * 6
        try:
            scores = self.toxicity_analyzer.analyze(text)
            return [float(scores.get(label, 0.0)) for label in labels]
        except Exception as exc:
            logger.exception("Toxicity analysis failed during rewrite: %s", exc)
            return [0.0] * 6

    @staticmethod
    def _as_vector(scores):
        labels = ("toxic", "threat", "insult", "severe_toxic", "obscene", "identity_hate")
        if isinstance(scores, dict):
            return [float(scores.get(label, 0.0)) for label in labels]
        if isinstance(scores, (list, tuple)) and len(scores) == 6:
            return [float(value) for value in scores]
        return [0.0] * 6

    @staticmethod
    def _init_groq():
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.info("GROQ_API_KEY not set; using local LLM or templates")
            return None
        try:
            from groq import Groq

            return Groq(api_key=api_key)
        except Exception as exc:
            logger.warning("Groq client unavailable: %s", exc)
            return None

    @staticmethod
    def _init_toxicity_analyzer():
        try:
            from modules.toxicity_model import ToxicityAnalyzer

            return ToxicityAnalyzer()
        except Exception as exc:
            logger.warning("ToxicityAnalyzer unavailable for rewrite validation: %s", exc)
            return None
