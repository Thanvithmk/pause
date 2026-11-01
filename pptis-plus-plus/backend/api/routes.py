import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone

from flask import Blueprint, g, jsonify, request
from sqlalchemy import func

from modules.auth import AuthError, login_user, register_user, require_auth
from modules.database import (
    Conversation,
    InterventionLog,
    Message,
    RiskResult,
    SlangDictionary,
    ToxicityResult,
    UnknownSlang,
    close_session,
    get_session,
    utcnow,
)
from modules.send_gateway import SendError, send_analyzed_message
from modules.toxicity_model import ToxicityAnalysisError

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)

CEL_LABELS = ("toxic", "threat", "insult", "severe_toxic", "obscene", "identity_hate")


class ApiError(Exception):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class PipelineServices:
    def __init__(self):
        self._cache = {}

    def reset(self):
        self._cache.clear()

    def _load(self, name, factory):
        if name not in self._cache:
            try:
                self._cache[name] = factory()
                logger.info("Loaded pipeline service %s", name)
            except Exception as exc:
                logger.exception("Failed to load service %s: %s", name, exc)
                self._cache[name] = None
        return self._cache[name]

    @property
    def preprocessor(self):
        from modules.preprocessing import Preprocessor

        return self._load("preprocessor", Preprocessor)

    @property
    def slang(self):
        from modules.slang_layer import SlangInterpreter

        return self._load("slang", SlangInterpreter)

    @property
    def toxicity(self):
        from modules.toxicity_model import ToxicityAnalyzer

        return self._load("toxicity", ToxicityAnalyzer)

    @property
    def profiles(self):
        from modules.behavior_profile import BehaviorProfileManager

        return self._load("profiles", BehaviorProfileManager)

    @property
    def risk(self):
        from modules.risk_calculation import RiskCalculator

        return self._load("risk", RiskCalculator)

    @property
    def decisions(self):
        from modules.decision_engine import DecisionEngine

        return self._load("decisions", DecisionEngine)

    @property
    def rewriter(self):
        from modules.rewrite_engine import RewriteEngine

        return self._load("rewriter", RewriteEngine)

    @property
    def delayed(self):
        from modules.delayed_send import DelayedSendManager

        return self._load("delayed", DelayedSendManager)

    @property
    def transcriber(self):
        from modules.speech_to_text import VoiceTranscriber

        return self._load("transcriber", VoiceTranscriber)

    @property
    def learning(self):
        from modules.ril import BehavioralLearning

        return self._load("learning", BehavioralLearning)


services = PipelineServices()


@api_bp.errorhandler(ApiError)
def handle_api_error(error):
    logger.warning("API error %s %s", error.status_code, error.message)
    return jsonify({"error": "api_error", "message": error.message}), error.status_code


@api_bp.errorhandler(AuthError)
def handle_auth_error(error):
    return jsonify({"error": "auth_error", "message": error.message}), error.status_code


@api_bp.route("/auth/register", methods=["POST"])
def register():
    body = request.get_json(silent=True) or {}
    logger.info("API register username=%s", body.get("username"))
    result = register_user(body.get("username"), body.get("email"), body.get("password"))
    return jsonify(result), 201


@api_bp.route("/auth/login", methods=["POST"])
def login():
    body = request.get_json(silent=True) or {}
    identifier = body.get("email") or body.get("username") or body.get("identifier")
    logger.info("API login identifier=%s", identifier)
    result = login_user(identifier, body.get("password"))
    return jsonify(result), 200


@api_bp.route("/auth/logout", methods=["POST"])
@require_auth
def logout():
    from modules.auth import logout_user

    return jsonify(logout_user()), 200


@api_bp.errorhandler(SendError)
def handle_send_error(error):
    logger.warning("Send rejected %s %s", error.status_code, error.message)
    return jsonify({"error": "send_error", "message": error.message}), error.status_code


@api_bp.route("/analyze", methods=["POST"])
@require_auth
def analyze():
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or body.get("original_text") or "").strip()
    input_type = (body.get("input_type") or "text").lower()
    if input_type not in ("text", "voice"):
        raise ApiError("input_type must be 'text' or 'voice'", 400)
    if input_type == "voice" and not text:
        raise ApiError("Voice analysis requires transcribed text or /api/analyze-voice", 400)
    if not text:
        raise ApiError("text is required", 400)
    logger.info("Analyze text user_id=%s chars=%s", g.current_user_id, len(text))
    try:
        payload = run_analysis_pipeline(
            user_id=g.current_user_id,
            text=text,
            input_type=input_type,
            conversation_id=body.get("conversation_id"),
            transcription_confidence=body.get("transcription_confidence"),
        )
    except ToxicityAnalysisError as exc:
        raise ApiError(str(exc) or "Toxicity analysis failed", 503) from exc
    return jsonify(payload), 200


@api_bp.route("/analyze-voice", methods=["POST"])
@require_auth
def analyze_voice():
    audio = request.files.get("audio")
    if audio is None or not audio.filename:
        raise ApiError("audio file is required", 400)
    conversation_id = request.form.get("conversation_id") or (request.get_json(silent=True) or {}).get(
        "conversation_id"
    )
    transcriber = services.transcriber
    if transcriber is None:
        raise ApiError("Speech-to-text is unavailable", 500)

    suffix = os.path.splitext(audio.filename)[1] or ".webm"
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        audio.save(handle.name)
        handle.close()
        logger.info("Transcribing voice user_id=%s file=%s", g.current_user_id, audio.filename)
        result = transcriber.transcribe_audio_file(handle.name, language="en")
    finally:
        try:
            os.unlink(handle.name)
        except OSError:
            pass

    if not result.get("success"):
        raise ApiError(result.get("error") or "Transcription failed", 400)
    text = (result.get("text") or "").strip()
    if not text:
        raise ApiError("No speech detected", 400)

    try:
        payload = run_analysis_pipeline(
            user_id=g.current_user_id,
            text=text,
            input_type="voice",
            conversation_id=int(conversation_id) if conversation_id else None,
            transcription_confidence=result.get("transcription_confidence", 1.0),
            transcription=result,
        )
    except ToxicityAnalysisError as exc:
        raise ApiError(str(exc) or "Toxicity analysis failed", 503) from exc
    payload["transcription"] = result
    payload["transcription_confidence"] = payload.get("transcription_confidence")
    return jsonify(payload), 200


@api_bp.route("/send", methods=["POST"])
@require_auth
def send_message():
    body = request.get_json(silent=True) or {}
    try:
        message_id = int(body.get("message_id"))
        conversation_id = int(body.get("conversation_id"))
    except (TypeError, ValueError) as exc:
        raise ApiError("message_id and conversation_id are required", 400) from exc
    text = (body.get("text") or body.get("original_text") or "").strip()
    user_action = (body.get("user_action") or "send").strip().lower()
    logger.info(
        "Send request user_id=%s message_id=%s action=%s",
        g.current_user_id,
        message_id,
        user_action,
    )
    result = send_analyzed_message(
        user_id=g.current_user_id,
        message_id=message_id,
        conversation_id=conversation_id,
        text=text,
        user_action=user_action,
        delayed_manager=services.delayed,
    )
    return jsonify(result), 200


@api_bp.route("/delayed-send/<int:message_id>", methods=["GET"])
@require_auth
def delayed_send_status(message_id):
    delayed = services.delayed
    if delayed is None:
        raise ApiError("Delayed-send service unavailable", 500)
    session = get_session()
    try:
        message = session.get(Message, message_id)
        if message is None:
            raise ApiError("Message not found", 404)
        if message.user_id != g.current_user_id:
            raise ApiError("You do not own this message", 403)
    finally:
        close_session()
    status = delayed.get_delayed_send_status(message_id) or delayed.get_user_lock_status(
        g.current_user_id
    )
    return jsonify(status or {"is_locked": False, "remaining_seconds": 0}), 200


@api_bp.route("/user/profile", methods=["GET"])
@require_auth
def user_profile():
    user_id = g.current_user_id
    logger.info("Profile user_id=%s", user_id)
    profiles = services.profiles
    learning = services.learning
    if profiles is None:
        raise ApiError("Profile service unavailable", 500)
    profile = profiles.get_or_create_profile(user_id)
    baseline = profiles.get_profile(user_id)
    stats = profiles.get_user_statistics(user_id, days=30)
    preferences = learning.get_intervention_preferences(user_id) if learning else {}
    insights = learning.generate_user_insights(user_id) if learning else []
    return jsonify(
        {
            "user": g.current_user.to_dict() if hasattr(g.current_user, "to_dict") else {"user_id": user_id},
            "profile": profile,
            "baseline": {
                "B_T": baseline[0],
                "B_H": baseline[1],
                "B_I": baseline[2],
                "B_S": baseline[3],
                "B_O": baseline[4],
                "B_IH": baseline[5],
                "vector": baseline,
            },
            "intervention_statistics": stats,
            "preferences": preferences,
            "insights": insights,
        }
    )


@api_bp.route("/dashboard", methods=["GET"])
@require_auth
def dashboard():
    user_id = g.current_user_id
    days = int(request.args.get("days", 30))
    logger.info("Dashboard user_id=%s days=%s", user_id, days)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    session = get_session()
    try:
        dialect = session.bind.dialect.name if session.bind is not None else "sqlite"
        day_expr = (
            func.strftime("%Y-%m-%d", Message.timestamp)
            if dialect == "sqlite"
            else func.date(Message.timestamp)
        )
        messages_count = (
            session.query(func.count(Message.message_id))
            .filter(Message.user_id == user_id, Message.timestamp >= cutoff)
            .scalar()
            or 0
        )
        sent_count = (
            session.query(func.count(Message.message_id))
            .filter(
                Message.user_id == user_id,
                Message.timestamp >= cutoff,
                Message.send_status == "sent",
            )
            .scalar()
            or 0
        )
        averages = session.query(
            func.avg(ToxicityResult.toxic),
            func.avg(ToxicityResult.threat),
            func.avg(ToxicityResult.insult),
            func.avg(ToxicityResult.severe_toxic),
            func.avg(ToxicityResult.obscene),
            func.avg(ToxicityResult.identity_hate),
        ).join(Message, ToxicityResult.message_id == Message.message_id).filter(
            Message.user_id == user_id, Message.timestamp >= cutoff
        ).one()
        average_risk = (
            session.query(func.avg(RiskResult.final_risk))
            .join(Message, RiskResult.message_id == Message.message_id)
            .filter(Message.user_id == user_id, RiskResult.timestamp >= cutoff)
            .scalar()
        )
        decision_counts = dict(
            session.query(RiskResult.decision, func.count(RiskResult.risk_id))
            .join(Message, RiskResult.message_id == Message.message_id)
            .filter(Message.user_id == user_id, RiskResult.timestamp >= cutoff)
            .group_by(RiskResult.decision)
            .all()
        )
        recent = (
            session.query(RiskResult, Message)
            .join(Message, RiskResult.message_id == Message.message_id)
            .filter(Message.user_id == user_id)
            .order_by(RiskResult.timestamp.desc())
            .limit(20)
            .all()
        )
        trends = (
            session.query(
                day_expr,
                func.avg(ToxicityResult.toxic),
            )
            .join(ToxicityResult, ToxicityResult.message_id == Message.message_id)
            .filter(Message.user_id == user_id, Message.timestamp >= cutoff)
            .group_by(day_expr)
            .order_by(day_expr)
            .all()
        )
        unknown_slang = (
            session.query(func.count(UnknownSlang.id))
            .filter(UnknownSlang.user_id == user_id)
            .scalar()
            or 0
        )
        dictionary_terms = session.query(func.count(SlangDictionary.slang_id)).scalar() or 0
    except Exception:
        logger.exception("Dashboard query failed user_id=%s", user_id)
        raise ApiError("Could not load dashboard", 500)
    finally:
        close_session()

    learning = services.learning
    insights = learning.generate_user_insights(user_id) if learning else []
    warn = int(decision_counts.get("WARN", 0))
    rewrite = int(decision_counts.get("REWRITE", 0))
    delayed = int(decision_counts.get("DELAYED-SEND", 0))
    allow = int(decision_counts.get("ALLOW", 0))
    payload = {
        "days": days,
        "messages_analyzed": int(messages_count),
        "messages_sent": int(sent_count),
        "average_risk": float(average_risk or 0.0),
        "slang_statistics": {
            "dictionary_terms": int(dictionary_terms),
            "unknown_terms_logged": int(unknown_slang),
        },
        "behavior_profile": profiles.get_or_create_profile(user_id) if (profiles := services.profiles) else {},
        "average_toxicity": float(averages[0] or 0.0),
        "average_threat": float(averages[1] or 0.0),
        "average_insult": float(averages[2] or 0.0),
        "average_severe_toxic": float(averages[3] or 0.0),
        "average_obscene": float(averages[4] or 0.0),
        "average_identity_hate": float(averages[5] or 0.0),
        "escalation_count": warn + rewrite + delayed,
        "warning_count": warn,
        "rewrite_count": rewrite,
        "delayed_send_count": delayed,
        "toxicity_trends": [{"date": str(row[0]), "toxic": float(row[1] or 0)} for row in trends],
        "message_outcomes": [
            {"name": "allow", "value": allow},
            {"name": "warn", "value": warn},
            {"name": "rewrite", "value": rewrite},
            {"name": "delayed", "value": delayed},
        ],
        "decision_breakdown": [
            {"name": key, "value": int(value)} for key, value in decision_counts.items()
        ],
        "escalation_history": [
            {
                "decision": risk.decision,
                "timestamp": risk.timestamp.isoformat() if risk.timestamp else None,
                "final_risk": risk.final_risk,
                "message_id": message.message_id,
            }
            for risk, message in recent
            if risk.decision in ("WARN", "REWRITE", "DELAYED-SEND")
        ],
        "recent_activity": [
            {
                "decision": risk.decision,
                "timestamp": risk.timestamp.isoformat() if risk.timestamp else None,
                "final_risk": risk.final_risk,
                "text": (message.original_text or "")[:160],
            }
            for risk, message in recent
        ],
        "insights": insights,
    }
    return jsonify(payload)


@api_bp.route("/conversations", methods=["POST"])
@require_auth
def create_conversation():
    conversation = _get_or_create_conversation(g.current_user_id, None)
    return jsonify({"conversation": conversation.to_dict()}), 201


@api_bp.route("/conversations", methods=["GET"])
@require_auth
def list_conversations():
    session = get_session()
    try:
        rows = (
            session.query(Conversation)
            .filter(Conversation.user_id == g.current_user_id)
            .order_by(Conversation.updated_at.desc())
            .all()
        )
        return jsonify({"conversations": [row.to_dict() for row in rows]}), 200
    finally:
        close_session()


@api_bp.route("/conversation/<int:conversation_id>", methods=["GET"])
@require_auth
def get_conversation(conversation_id):
    session = get_session()
    try:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            raise ApiError("Conversation not found", 404)
        if conversation.user_id != g.current_user_id:
            raise ApiError("You do not own this conversation", 403)
        messages = (
            session.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.timestamp.asc())
            .all()
        )
        thread = []
        for message in messages:
            toxicity = (
                session.query(ToxicityResult)
                .filter(ToxicityResult.message_id == message.message_id)
                .order_by(ToxicityResult.result_id.desc())
                .first()
            )
            risk = (
                session.query(RiskResult)
                .filter(RiskResult.message_id == message.message_id)
                .order_by(RiskResult.timestamp.desc(), RiskResult.risk_id.desc())
                .first()
            )
            interventions = (
                session.query(InterventionLog)
                .filter(InterventionLog.message_id == message.message_id)
                .all()
            )
            thread.append(
                {
                    "message": message.to_dict(),
                    "toxicity": toxicity.to_dict() if toxicity else None,
                    "decision": risk.to_dict() if risk else None,
                    "interventions": [item.to_dict() for item in interventions],
                }
            )
        logger.info(
            "Conversation thread conversation_id=%s messages=%s",
            conversation_id,
            len(thread),
        )
        return jsonify(
            {
                "conversation": conversation.to_dict(),
                "messages": thread,
            }
        )
    finally:
        close_session()


def run_analysis_pipeline(
    user_id,
    text,
    input_type="text",
    conversation_id=None,
    transcription_confidence=None,
    transcription=None,
):
    preprocessor = services.preprocessor
    slang = services.slang
    toxicity = services.toxicity
    profiles = services.profiles
    risk_calc = services.risk
    decisions = services.decisions
    if None in (preprocessor, slang, toxicity, profiles, risk_calc, decisions):
        raise ApiError("Analysis services are unavailable", 500)

    conversation = _get_or_create_conversation(user_id, conversation_id)
    prior_context = _recent_message_texts(conversation.conversation_id)
    preprocessed = preprocessor.preprocess(text)
    cleaned = preprocessed["cleaned_text"]
    slang_result = slang.interpret(cleaned, user_id=user_id)
    enriched = slang_result.get("enriched_text_with_context") or cleaned
    toxicity_scores = toxicity.analyze(enriched, context=prior_context)
    vector = [float(toxicity_scores[label]) for label in CEL_LABELS]

    session = get_session()
    try:
        message = Message(
            conversation_id=conversation.conversation_id,
            user_id=user_id,
            original_text=text,
            current_text=text,
            analyzed_text=text,
            processed_text=enriched,
            input_type=input_type,
            transcription_confidence=transcription_confidence,
            send_status="draft",
            analyzed_at=utcnow(),
        )
        session.add(message)
        conversation.updated_at = datetime.now(timezone.utc)
        session.commit()
        session.refresh(message)
        message_id = message.message_id
        previous = _previous_toxicity(session, user_id, conversation.conversation_id, message_id)
        tox_row = ToxicityResult(
            message_id=message_id,
            toxic=vector[0],
            threat=vector[1],
            insult=vector[2],
            severe_toxic=vector[3],
            obscene=vector[4],
            identity_hate=vector[5],
        )
        session.add(tox_row)
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Failed to store message/toxicity")
        raise ApiError("Could not store analysis", 500)
    finally:
        close_session()

    profile = profiles.get_or_create_profile(user_id)
    baseline = profiles.get_profile(user_id)
    pacl = risk_calc.calculate_pacl(vector[0], baseline[0])
    if previous is None:
        delta = [0.0] * 6
        cel = 0.0
    else:
        delta = risk_calc.calculate_delta_x(vector, previous)
        cel = risk_calc.calculate_cel(delta)
    p_esc = risk_calc.calculate_escalation_probability(cel, pacl)
    final_risk = risk_calc.calculate_final_risk(pacl, p_esc)
    decision = decisions.make_decision(final_risk, toxicity_vector=toxicity_scores)

    session = get_session()
    try:
        risk_row = RiskResult(
            message_id=message_id,
            pacl=pacl,
            delta_vector_6d=delta,
            escalation_feature=cel,
            escalation_probability=p_esc,
            final_risk=final_risk,
            decision=decision["decision"],
        )
        session.add(risk_row)
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Failed to store risk result")
        raise ApiError("Could not store risk result", 500)
    finally:
        close_session()

    action = decisions.apply_decision(
        decision["decision"],
        message_id,
        user_id,
        conversation.conversation_id,
        text,
    )
    rewrite_payload = None
    delay_payload = None
    if decision["decision"] == "REWRITE" and services.rewriter:
        rewrite_payload = services.rewriter.rewrite(
            text,
            conversation_id=conversation.conversation_id,
            message_id=message_id,
            user_id=user_id,
        )
        action["suggested_text"] = rewrite_payload.get("suggested_text")
        action["allow_send_original"] = False
    if decision["decision"] == "DELAYED-SEND" and services.delayed:
        delay_payload = services.delayed.create_delayed_send_state(
            message_id,
            user_id,
            text,
            final_risk,
            conversation_id=conversation.conversation_id,
        )
        action["delayed_send"] = delay_payload
        action["allow_send_anyway"] = False

    session = get_session()
    try:
        InterventionLog.create(
            session,
            message_id=message_id,
            action=decision["decision"],
            suggested_text=(rewrite_payload or {}).get("suggested_text"),
            context_ids=[item.get("term") for item in slang.detect_candidates(cleaned)],
            delay_seconds=(delay_payload or {}).get("countdown_seconds"),
        )
    except Exception:
        logger.exception("Failed to store intervention log")
    finally:
        close_session()

    logger.info(
        "Analysis complete user_id=%s message_id=%s decision=%s R=%.4f",
        user_id,
        message_id,
        decision["decision"],
        final_risk,
    )
    return {
        "message_id": message_id,
        "conversation_id": conversation.conversation_id,
        "original_text": text,
        "processed_text": cleaned,
        "enriched_text": enriched,
        "input_type": input_type,
        "transcription_confidence": transcription_confidence,
        "preprocessed": preprocessed,
        "slang": slang_result,
        "toxicity": toxicity_scores,
        "profile": profile,
        "pacl": pacl,
        "delta_vector_6d": delta,
        "escalation_feature": cel,
        "escalation_probability": p_esc,
        "final_risk": final_risk,
        "risk_score": final_risk,
        "decision": decision["decision"],
        "confidence": decision.get("confidence"),
        "color_for_ui": decision.get("color_for_ui"),
        "description": decision.get("description"),
        "action": action,
        "rewrite": rewrite_payload,
        "delayed_send": delay_payload,
        "allow_send_original": False,
        "allow_send_anyway": decision["decision"] == "WARN",
        "can_send": decision["decision"] == "ALLOW",
        "send_status": "draft",
        "suggested_text": (rewrite_payload or {}).get("suggested_text") or action.get(
            "suggested_text"
        ),
        "warning_message": action.get("warning_message"),
    }


def _get_or_create_conversation(user_id, conversation_id):
    session = get_session()
    try:
        if conversation_id:
            conversation = session.get(Conversation, int(conversation_id))
            if conversation is None:
                raise ApiError("Conversation not found", 404)
            if conversation.user_id != user_id:
                raise ApiError("You do not own this conversation", 403)
            return conversation
        conversation = Conversation.create(session, user_id=user_id)
        logger.info("Created conversation_id=%s user_id=%s", conversation.conversation_id, user_id)
        return conversation
    except ApiError:
        raise
    except Exception:
        session.rollback()
        logger.exception("Conversation lookup failed")
        raise ApiError("Could not load conversation", 500)
    finally:
        close_session()


def _recent_message_texts(conversation_id, limit=4):
    session = get_session()
    try:
        rows = (
            session.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.timestamp.desc())
            .limit(limit)
            .all()
        )
        texts = []
        for row in reversed(rows):
            piece = row.current_text or row.original_text or ""
            if piece.strip():
                texts.append(piece.strip())
        return "\n".join(texts)
    except Exception:
        logger.exception("Could not load prior messages conversation_id=%s", conversation_id)
        return ""
    finally:
        close_session()


def _previous_toxicity(session, user_id, conversation_id, message_id):
    previous = (
        session.query(Message)
        .filter(
            Message.user_id == user_id,
            Message.conversation_id == conversation_id,
            Message.message_id != message_id,
        )
        .order_by(Message.timestamp.desc(), Message.message_id.desc())
        .first()
    )
    if previous is None:
        return None
    row = (
        session.query(ToxicityResult)
        .filter(ToxicityResult.message_id == previous.message_id)
        .one_or_none()
    )
    if row is None:
        return None
    return [float(getattr(row, label)) for label in CEL_LABELS]
