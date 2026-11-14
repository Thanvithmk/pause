import logging
from datetime import datetime, timedelta, timezone

from flask import Blueprint, current_app, g, jsonify, request
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
from modules.contextual_threat import assess
from modules.linguistic_features import (
    annotate_slang_usage,
    emoji_annotations,
    has_playful_laughter,
    recognize_in_parallel,
)
from modules.rewrite_engine import REQUIRED_GROQ_MODEL, RewriteServiceError
from modules.send_gateway import SendError, send_analyzed_message
from modules.slang_layer import SlangModelError
from modules.toxicity_model import ToxicityAnalysisError

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)

CEL_LABELS = ("toxic", "threat", "insult", "severe_toxic", "obscene", "identity_hate")


def _fill_trigger_categories(session, recent):
    from modules.trigger_category import CATEGORIES, categorize_many

    stopped = [
        (risk, message)
        for risk, message in recent
        if risk.decision in ("WARN", "REWRITE", "DELAYED-SEND")
    ]
    missing = [
        (risk, message)
        for risk, message in stopped
        if not risk.trigger_category
    ]
    try:
        categories = categorize_many(
            [(risk.risk_id, message.original_text or "") for risk, message in missing]
        )
    except Exception:
        logger.exception("Trigger categories could not be loaded")
        categories = {}
    changed = False
    for risk, _message in missing:
        category = categories.get(risk.risk_id)
        if category:
            risk.trigger_category = category
            changed = True
    if changed:
        session.commit()
    counts = {}
    for risk, _message in stopped:
        if risk.trigger_category:
            counts[risk.trigger_category] = counts.get(risk.trigger_category, 0) + 1
    return [
        {"category": name, "count": counts[name]}
        for name in CATEGORIES
        if counts.get(name)
    ]


def _baseline_view(profile):
    source = profile if isinstance(profile, dict) else {}
    view = {}
    for label in CEL_LABELS:
        raw = source.get(label, source.get(f"avg_{label}", 0.0))
        try:
            view[label] = float(raw or 0.0)
        except (TypeError, ValueError):
            view[label] = 0.0
    return view


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
        from modules.behavior_profile import clear_profile_cache

        clear_profile_cache()

    def _load(self, name, factory):
        if name not in self._cache:
            try:
                self._cache[name] = factory()
                logger.info("Loaded pipeline service %s", name)
            except Exception as exc:
                logger.exception("Failed to load service %s: %s", name, exc)
                raise
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
    def slang_recognition(self):
        from modules.slang_recognition import SlangRecognizer

        return self._load("slang_recognition", SlangRecognizer)

    @property
    def emoji_recognition(self):
        from modules.emoji_recognition import EmojiRecognizer

        return self._load("emoji_recognition", EmojiRecognizer)

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


@api_bp.route("/health", methods=["GET"])
def health():
    report = current_app.extensions.get("primary_models")
    if not report:
        return jsonify({"ready": False, "production_ready": False, "error": "primary models were not verified"}), 503
    local_ready = all(
        (report.get(name) or {}).get("inference") == "actual"
        for name in ("slang", "toxicity")
    )
    return jsonify({
        "ready": local_ready,
        "production_ready": bool(local_ready and report.get("production_ready")),
        **report,
    }), 200 if local_ready else 503


@api_bp.route("/analyze", methods=["POST"])
@require_auth
def analyze():
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or body.get("original_text") or "").strip()
    input_type = (body.get("input_type") or "text").lower()
    if input_type != "text":
        raise ApiError("input_type must be 'text'", 400)
    if not text:
        raise ApiError("text is required", 400)
    logger.info("Analyze text user_id=%s chars=%s", g.current_user_id, len(text))
    try:
        payload = run_analysis_pipeline(
            user_id=g.current_user_id,
            text=text,
            conversation_id=body.get("conversation_id"),
        )
    except (ToxicityAnalysisError, SlangModelError, RewriteServiceError) as exc:
        raise ApiError(str(exc) or "Analysis failed", 503) from exc
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
            .filter(Message.user_id == user_id, RiskResult.timestamp >= cutoff)
            .order_by(RiskResult.timestamp.desc())
            .limit(50)
            .all()
        )
        categorized = (
            session.query(RiskResult, Message)
            .join(Message, RiskResult.message_id == Message.message_id)
            .filter(
                Message.user_id == user_id,
                RiskResult.timestamp >= cutoff,
                RiskResult.decision.in_(("WARN", "REWRITE", "DELAYED-SEND")),
            )
            .order_by(RiskResult.timestamp.desc())
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
        trigger_categories = _fill_trigger_categories(session, categorized)
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
        "trigger_categories": trigger_categories,
        "recent_activity": [
            {
                "decision": risk.decision,
                "timestamp": risk.timestamp.isoformat() if risk.timestamp else None,
                "final_risk": risk.final_risk,
                "text": (message.original_text or "")[:160],
                "send_status": message.send_status,
            }
            for risk, message in recent
        ],
        "insights": insights,
        "intervention_effectiveness": _effectiveness_view(user_id),
    }
    return jsonify(payload)


@api_bp.route("/intervention-response", methods=["POST"])
@require_auth
def intervention_response():
    body = request.get_json(silent=True) or {}
    try:
        message_id = int(body.get("message_id"))
    except (TypeError, ValueError) as exc:
        raise ApiError("message_id is required", 400) from exc
    requested = (body.get("user_response") or "").strip().lower()
    mapped = {
        "cancel": ("rejected", "neutral", "cancel"),
        "rejected": ("rejected", "neutral", "cancel"),
        "edit": ("edited", "neutral", "edit"),
        "edited": ("edited", "neutral", "edit"),
        "accepted": ("accepted", "neutral", "accepted"),
    }
    if requested not in mapped:
        raise ApiError("user_response must be cancel, edit, or accepted", 400)
    user_response, outcome, user_action = mapped[requested]
    if body.get("outcome") in ("positive", "neutral", "negative"):
        outcome = body["outcome"]

    session = get_session()
    try:
        message = session.get(Message, message_id)
        if message is None:
            raise ApiError("Message not found", 404)
        if message.user_id != g.current_user_id:
            raise ApiError("You do not own this message", 403)
        if message.send_status == "sent":
            raise ApiError("Message already sent", 409)
        risk = (
            session.query(RiskResult)
            .filter(RiskResult.message_id == message.message_id)
            .order_by(RiskResult.timestamp.desc(), RiskResult.risk_id.desc())
            .first()
        )
        if risk is None or risk.decision not in ("WARN", "REWRITE", "DELAYED-SEND"):
            raise ApiError("This message has no intervention to record", 422)
        decision = risk.decision
        profile_count = message.profile_updated
    finally:
        close_session()

    learning = services.learning
    if learning is None:
        raise ApiError("Feedback learning is unavailable", 500)
    recorded = learning.record_intervention_outcome(message_id, decision, user_response, outcome)
    session = get_session()
    try:
        log = (
            session.query(InterventionLog)
            .filter(InterventionLog.message_id == message_id, InterventionLog.action == decision)
            .order_by(InterventionLog.id.desc())
            .first()
        )
        if log is not None:
            log.user_action = user_action
        if user_action == "cancel":
            message = session.get(Message, message_id)
            if message is not None and message.send_status != "sent":
                message.send_status = "cancelled"
        session.commit()
    finally:
        close_session()
    return jsonify({
        "recorded": True,
        "sent": False,
        "message_id": message_id,
        "decision": decision,
        "user_response": user_response,
        "outcome": outcome,
        "profile_updated": bool(profile_count),
        "effectiveness": recorded.get("effectiveness"),
    }), 200


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
    conversation_id=None,
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
    prior_messages = _recent_message_texts(conversation.conversation_id)
    prior_context = "\n".join(prior_messages)
    preprocessed = preprocessor.preprocess(text)
    cleaned = preprocessed["cleaned_text"]
    linguistic_features = recognize_in_parallel(
        cleaned,
        services.slang_recognition,
        services.emoji_recognition,
    )
    slang_result = slang.interpret(cleaned, user_id=user_id)
    linguistic_features["slang"] = annotate_slang_usage(
        linguistic_features["slang"],
        slang_result.get("detections"),
    )
    contextual = assess(cleaned)
    enriched = slang_result.get("enriched_text_with_context") or cleaned
    notes = emoji_annotations(linguistic_features.get("emoji"))
    if notes:
        enriched = f"{enriched} {' '.join(notes)}"
    toxicity_detail = toxicity.inspect(
        enriched,
        context=prior_context,
        context_messages=prior_messages,
    )
    model_scores = dict(toxicity_detail["model"])
    # Safety flags can raise the decision later. They do not replace Xt.
    toxicity_scores = model_scores
    vector = [float(toxicity_scores[label]) for label in CEL_LABELS]
    detections = slang_result.get("detections") or []

    profile = profiles.get_or_create_profile(user_id)
    baseline = profiles.get_profile(user_id)
    session = get_session()
    try:
        locked_conversation = _lock_row(
            session, Conversation, Conversation.conversation_id == conversation.conversation_id
        )
        message = Message(
            conversation_id=conversation.conversation_id,
            user_id=user_id,
            original_text=text,
            current_text=text,
            analyzed_text=text,
            processed_text=enriched,
            input_type="text",
            send_status="draft",
            analyzed_at=utcnow(),
            profile_updated=False,
            slang_detections=detections,
        )
        session.add(message)
        locked_conversation.updated_at = datetime.now(timezone.utc)
        session.flush()
        message_id = message.message_id
        previous = _previous_toxicity(session, user_id, conversation.conversation_id, message_id)
        session.add(ToxicityResult(
            message_id=message_id,
            toxic=vector[0],
            threat=vector[1],
            insult=vector[2],
            severe_toxic=vector[3],
            obscene=vector[4],
            identity_hate=vector[5],
        ))
        if previous is None:
            delta = [0.0] * 6
            cel = 0.0
        else:
            delta = risk_calc.calculate_delta_x(vector, previous)
            cel = risk_calc.calculate_cel(delta)
        pacl = risk_calc.calculate_pacl(vector[0], baseline[0])
        p_esc = risk_calc.calculate_escalation_probability(cel, pacl)
        final_risk = risk_calc.calculate_final_risk(pacl, p_esc)
        safety_flags = {
            "quoted": toxicity_detail.get("quoted"),
            "negated": toxicity_detail.get("negated"),
            "playful_idiom": toxicity_detail.get("playful_idiom"),
            "playful_emoji": has_playful_laughter(linguistic_features.get("emoji")),
            "contextual_threat": contextual,
        }
        mathematical_decision = risk_calc.get_risk_category(final_risk)[0]
        decision = decisions.make_decision(
            final_risk,
            toxicity_vector=toxicity_scores,
            safety_flags=safety_flags,
            safety_label=toxicity_detail.get("safety_label"),
        )
        send_status = {
            "ALLOW": "approved",
            "WARN": "pending_warning",
            "REWRITE": "pending_rewrite",
            "DELAYED-SEND": "delayed",
        }[decision["decision"]]
        message.send_status = send_status
        session.add(RiskResult(
            message_id=message_id,
            pacl=pacl,
            delta_vector_6d=delta,
            escalation_feature=cel,
            escalation_probability=p_esc,
            final_risk=final_risk,
            decision=decision["decision"],
            previous_vector_6d=previous,
            mathematical_decision=mathematical_decision,
            guardrail_reason=decision.get("guardrail"),
            model_toxicity=model_scores,
        ))
        session.commit()
    except ApiError:
        session.rollback()
        raise
    except Exception:
        session.rollback()
        logger.exception("Failed to store analysis")
        raise ApiError("Could not store analysis", 500)
    finally:
        close_session()

    action = decisions.apply_decision(
        decision["decision"],
        message_id,
        user_id,
        conversation.conversation_id,
        text,
    )
    rewrite_payload = _rewrite_payload(decision["decision"], conversation.conversation_id, message_id, user_id, text)
    delay_payload = None
    if decision["decision"] == "REWRITE":
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
            suggested_text=rewrite_payload.get("suggested_text"),
            context_ids=[item.get("term") for item in detections],
            delay_seconds=(delay_payload or {}).get("countdown_seconds"),
            user_action="rewrite_unavailable" if rewrite_payload.get("status") == "unavailable" else None,
        )
    except Exception:
        logger.exception("Failed to store intervention log message_id=%s", message_id)
        raise ApiError("Could not store intervention log", 500)
    finally:
        close_session()

    logger.info(
        "Analysis complete user_id=%s message_id=%s decision=%s mathematical=%s R=%.4f guardrail=%s contextual=%s",
        user_id,
        message_id,
        decision["decision"],
        mathematical_decision,
        final_risk,
        decision.get("guardrail"),
        contextual.get("category"),
    )
    return {
        "message_id": message_id,
        "conversation_id": conversation.conversation_id,
        "original_text": text,
        "context_messages": prior_messages,
        "processed_text": cleaned,
        "enriched_text": enriched,
        "input_type": "text",
        "preprocessed": preprocessed,
        "slang": detections,
        "linguistic_features": linguistic_features,
        "toxicity": toxicity_scores,
        "model_toxicity": model_scores,
        "safety": {
            "label": toxicity_detail.get("safety_label"),
            "quoted": safety_flags["quoted"],
            "negated": safety_flags["negated"],
            "playful_idiom": safety_flags["playful_idiom"],
        },
        "contextual_threat": contextual,
        "toxicity_truncated": bool(toxicity_detail.get("truncated")),
        "behavioral_baseline": _baseline_view(profile),
        "profile": profile,
        "previous_vector": previous,
        "pacl": pacl,
        "delta_vector": delta,
        "delta_vector_6d": delta,
        "escalation_feature": cel,
        "escalation_probability": p_esc,
        "final_risk": final_risk,
        "risk_score": final_risk,
        "mathematical_decision": mathematical_decision,
        "decision": decision["decision"],
        "guardrail": decision.get("guardrail"),
        "confidence": decision.get("confidence"),
        "color_for_ui": decision.get("color_for_ui"),
        "description": decision.get("description"),
        "action": action,
        "rewrite": rewrite_payload,
        "delayed_send": delay_payload,
        "allow_send_original": False,
        "allow_send_anyway": decision["decision"] == "WARN",
        "can_send": decision["decision"] == "ALLOW",
        "send_status": send_status,
        "suggested_text": rewrite_payload.get("suggested_text"),
        "warning_message": action.get("warning_message"),
        "analysis_complete": True,
        "inference": {
            "slang": "actual",
            "toxicity": "actual",
            "rewrite": rewrite_payload.get("inference"),
        },
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


def _recent_message_texts(conversation_id, limit=5):
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
        return texts[-limit:]
    except Exception:
        logger.exception("Could not load prior messages conversation_id=%s", conversation_id)
        return []
    finally:
        close_session()


def _previous_toxicity(session, user_id, conversation_id, message_id):
    row = (
        session.query(ToxicityResult)
        .join(Message, ToxicityResult.message_id == Message.message_id)
        .filter(
            Message.user_id == user_id,
            Message.conversation_id == conversation_id,
            Message.message_id < message_id,
        )
        .order_by(Message.message_id.desc())
        .first()
    )
    if row is None:
        return None
    return [float(getattr(row, label)) for label in CEL_LABELS]


def _lock_row(session, model, criterion):
    query = session.query(model).filter(criterion)
    bind = session.get_bind()
    if bind is not None and bind.dialect.name == "postgresql":
        query = query.with_for_update()
    return query.one()


def _rewrite_payload(decision_name, conversation_id, message_id, user_id, text):
    if decision_name != "REWRITE":
        return {
            "suggested_text": None,
            "available": False,
            "status": "not_required",
            "inference": None,
            "model": None,
            "error": None,
            "allow_send_original": False,
        }
    rewriter = services.rewriter
    try:
        if rewriter is None:
            raise RewriteServiceError("rewrite: Groq rewrite service is unavailable")
        produced = rewriter.rewrite(
            text,
            conversation_id=conversation_id,
            message_id=message_id,
            user_id=user_id,
        )
    except RewriteServiceError as exc:
        logger.warning("Rewrite unavailable message_id=%s error=%s", message_id, exc)
        return {
            "suggested_text": None,
            "available": False,
            "status": "unavailable",
            "inference": "unavailable",
            "model": REQUIRED_GROQ_MODEL,
            "error": str(exc),
            "allow_send_original": False,
        }
    if produced.get("inference") != "actual" or produced.get("model") != REQUIRED_GROQ_MODEL:
        return {
            "suggested_text": None,
            "available": False,
            "status": "unavailable",
            "inference": "unavailable",
            "model": REQUIRED_GROQ_MODEL,
            "error": f"rewrite: Groq rewrite did not use {REQUIRED_GROQ_MODEL}",
            "allow_send_original": False,
        }
    suggestion = produced.get("suggested_text")
    return {
        **produced,
        "suggested_text": suggestion,
        "available": bool(suggestion),
        "status": "ready" if suggestion else "edit_only",
        "error": None,
        "allow_send_original": False,
    }


def _effectiveness_view(user_id):
    learning = services.learning
    if learning is None:
        return None
    stats = learning.update_ril_statistics(user_id)
    session = get_session()
    try:
        from modules.ril import InterventionEffectiveness

        record = (
            session.query(InterventionEffectiveness)
            .filter(InterventionEffectiveness.user_id == user_id)
            .one_or_none()
        )
        counts = record.to_dict() if record is not None else {}
    finally:
        close_session()
    return {"scores": stats, "counts": counts}
