"""Personalized Adaptive Contextual Liability (PACL) and
Conversational Escalation Layer (CEL).

PACL = clip(Tb + λ (Tb - B_T), 0, 1)

Tb  : current overall toxicity (Toxic dimension)
B_T : user's historical average toxic score
λ   : personalization factor (FIXED, not tuned)

CEL:
    ΔX = X_t - X_{t-1}   (6-D element-wise difference)
    E  = W · ΔX         (dot product; escalation feature)

Weight meaning (W = [w_T, w_H, w_I, w_S, w_O, w_IH]):
    w_T  : toxic
    w_H  : threat
    w_I  : insult
    w_S  : severe_toxic
    w_O  : obscene
    w_IH : identity_hate

E > 0 escalation; E < 0 de-escalation. First message has no previous vector, so E = 0.

Escalation probability:
    P_esc = 1 / (1 + e^{-z})
    z = b + w_E * E + w_P * PACL
    b, w_E, w_P are TRAINED (Phase 6), not fixed.

Final risk:
    R = α * PACL + (1 - α) * P_esc
    α is FIXED (not tuned). α = 0.5 weights current and future risk equally.
    Higher α emphasizes current PACL; lower α emphasizes escalation probability.
"""

import json
import logging
import math
from collections import deque
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from modules.behavior_profile import BehaviorProfileManager
from modules.database import (
    Message,
    RiskResult,
    ToxicityResult,
    close_session,
    get_session,
)

logger = logging.getLogger(__name__)

WITHIN_BASELINE_MAX = 0.40
HIGHER_THAN_BASELINE_MAX = 0.70
CEL_LABELS = ("toxic", "threat", "insult", "severe_toxic", "obscene", "identity_hate")
CEL_WEIGHT_MEANING = {
    "w_T": "toxic",
    "w_H": "threat",
    "w_I": "insult",
    "w_S": "severe_toxic",
    "w_O": "obscene",
    "w_IH": "identity_hate",
}
DEFAULT_WEIGHT_VECTOR = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
ZERO_DELTA = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
DEFAULT_LOGISTIC_B = 0.0
DEFAULT_LOGISTIC_W_E = 1.0
DEFAULT_LOGISTIC_W_P = 1.0
ESCALATION_PARAM_PATH = Path(__file__).resolve().parent.parent / "escalation_logistic.json"
DEFAULT_ALPHA = 0.5
DEFAULT_THRESHOLD_ALLOW = 0.30
DEFAULT_THRESHOLD_WARN = 0.55
DEFAULT_THRESHOLD_REWRITE = 0.80
RISK_HISTORY_SIZE = 100
RISK_CATEGORIES = {
    "ALLOW": {
        "color_for_ui": "#22c55e",
        "description": "Low risk, send normally",
    },
    "WARN": {
        "color_for_ui": "#eab308",
        "description": "Moderate risk, show warning",
    },
    "REWRITE": {
        "color_for_ui": "#f97316",
        "description": "High risk, suggest rewrite",
    },
    "DELAYED-SEND": {
        "color_for_ui": "#ef4444",
        "description": "Very high risk, delay send and offer exercises",
    },
}


def _load_risk_settings():
    try:
        from config import get_config

        config = get_config()
        weights = tuple(float(value) for value in config.CEL_WEIGHT_VECTOR)
        if len(weights) != 6:
            raise ValueError("CEL_WEIGHT_VECTOR must have 6 dimensions")
        logistic = (
            float(getattr(config, "LOGISTIC_B", DEFAULT_LOGISTIC_B)),
            float(getattr(config, "LOGISTIC_W_E", DEFAULT_LOGISTIC_W_E)),
            float(getattr(config, "LOGISTIC_W_P", DEFAULT_LOGISTIC_W_P)),
        )
        alpha = float(getattr(config, "FINAL_RISK_ALPHA", DEFAULT_ALPHA))
        thresholds = (
            float(getattr(config, "THRESHOLD_ALLOW", DEFAULT_THRESHOLD_ALLOW)),
            float(getattr(config, "THRESHOLD_WARN", DEFAULT_THRESHOLD_WARN)),
            float(getattr(config, "THRESHOLD_REWRITE", DEFAULT_THRESHOLD_REWRITE)),
        )
        return float(config.PACL_LAMBDA), float(config.THRESHOLD_WARN), weights, logistic, alpha, thresholds
    except Exception as exc:
        logger.warning("Using default PACL/CEL settings: %s", exc)
        return (
            0.5,
            0.55,
            DEFAULT_WEIGHT_VECTOR,
            (DEFAULT_LOGISTIC_B, DEFAULT_LOGISTIC_W_E, DEFAULT_LOGISTIC_W_P),
            DEFAULT_ALPHA,
            (DEFAULT_THRESHOLD_ALLOW, DEFAULT_THRESHOLD_WARN, DEFAULT_THRESHOLD_REWRITE),
        )


class RiskCalculator:
    def __init__(self, lambda_const=0.5):
        config_lambda, pacl_threshold, weight_vector, logistic, alpha, thresholds = _load_risk_settings()
        self.lambda_const = float(config_lambda if lambda_const is None else lambda_const)
        self.pacl_threshold = pacl_threshold
        self.weight_vector = list(weight_vector)
        self.b, self.w_E, self.w_P = logistic
        self.alpha = alpha
        self.threshold_allow, self.threshold_warn, self.threshold_rewrite = thresholds
        self.risk_history = deque(maxlen=RISK_HISTORY_SIZE)
        self.load_escalation_parameters()
        try:
            self.profile_manager = BehaviorProfileManager()
        except Exception as exc:
            logger.exception("BehaviorProfileManager unavailable: %s", exc)
            self.profile_manager = None
        logger.info(
            "RiskCalculator initialized lambda=%s alpha=%s pacl_threshold=%s W=%s b=%s w_E=%s w_P=%s",
            self.lambda_const,
            self.alpha,
            self.pacl_threshold,
            self.weight_vector,
            self.b,
            self.w_E,
            self.w_P,
        )

    def calculate_pacl(self, current_toxicity_score, baseline_toxic, lambda_const=None):
        tb = self._validate_unit("Tb", current_toxicity_score)
        baseline = self._validate_unit("B_T", baseline_toxic)
        lam = self.lambda_const if lambda_const is None else float(lambda_const)
        if lam < 0:
            raise ValueError("lambda must be >= 0")

        # PACL = clip[Tb + λ(Tb - B_T), 0, 1]
        raw = tb + lam * (tb - baseline)
        pacl = min(1.0, max(0.0, raw))
        logger.info(
            "PACL Tb=%.4f B_T=%.4f lambda=%.4f raw=%.4f clipped=%.4f",
            tb,
            baseline,
            lam,
            raw,
            pacl,
        )
        return pacl

    def interpret_pacl(self, pacl_score):
        score = self._validate_unit("PACL", pacl_score)
        if score <= WITHIN_BASELINE_MAX:
            label = "Within baseline"
        elif score <= HIGHER_THAN_BASELINE_MAX:
            label = "Higher than baseline"
        else:
            label = "Significantly elevated"
        logger.info("PACL interpretation score=%.4f label=%s", score, label)
        return label

    def calculate_final_risk(self, pacl_score, escalation_probability, alpha=None):
        pacl = self._validate_unit("PACL", pacl_score)
        p_esc = self._validate_unit("P_esc", escalation_probability)
        weight = self.alpha if alpha is None else float(alpha)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("alpha must be in [0, 1]")

        # R = α * PACL + (1 - α) * P_esc
        final_risk = min(1.0, max(0.0, weight * pacl + (1.0 - weight) * p_esc))
        entry = {
            "pacl": pacl,
            "p_esc": p_esc,
            "alpha": weight,
            "final_risk": final_risk,
        }
        self.risk_history.append(entry)
        logger.info(
            "Final risk PACL=%.4f P_esc=%.4f alpha=%.4f R=%.4f",
            pacl,
            p_esc,
            weight,
            final_risk,
        )
        return final_risk

    def get_risk_category(self, final_risk):
        score = self._validate_unit("R", final_risk)
        if score < self.threshold_allow:
            category = "ALLOW"
        elif score < self.threshold_warn:
            category = "WARN"
        elif score < self.threshold_rewrite:
            category = "REWRITE"
        else:
            category = "DELAYED-SEND"
        meta = RISK_CATEGORIES[category]
        logger.info("Risk category R=%.4f category=%s", score, category)
        return (category, meta["color_for_ui"], meta["description"])

    def get_risk_history(self):
        return list(self.risk_history)

    def calculate_escalation_probability(
        self, escalation_feature, pacl_score, b=None, w_E=None, w_P=None
    ):
        e_score = self._validate_number("E", escalation_feature)
        pacl = self._validate_unit("PACL", pacl_score)
        intercept = self.b if b is None else float(b)
        weight_e = self.w_E if w_E is None else float(w_E)
        weight_p = self.w_P if w_P is None else float(w_P)

        # P_esc = 1 / (1 + e^(-z)), z = b + w_E * E + w_P * PACL
        z_score = intercept + weight_e * e_score + weight_p * pacl
        probability = self._sigmoid(z_score)
        logger.info(
            "P_esc z=%.4f E=%.4f PACL=%.4f b=%.4f w_E=%.4f w_P=%.4f P=%.4f",
            z_score,
            e_score,
            pacl,
            intercept,
            weight_e,
            weight_p,
            probability,
        )
        return probability

    def interpret_escalation_probability(self, p_esc):
        score = self._validate_unit("P_esc", p_esc)
        if score < 0.3:
            label = "Low risk"
        elif score <= 0.7:
            label = "Moderate risk"
        else:
            label = "High escalation risk"
        logger.info("P_esc interpretation score=%.4f label=%s", score, label)
        return label

    def train_escalation_model(self, training_data, test_size=0.2, random_state=42):
        features, labels = self._training_matrix(training_data)
        if len(np.unique(labels)) < 2:
            raise ValueError("training_data must include both escalation classes 0 and 1")

        stratify = labels if min(np.bincount(labels)) >= 2 and len(labels) >= 5 else None
        if len(labels) >= 5 and test_size > 0:
            x_train, x_test, y_train, y_test = train_test_split(
                features,
                labels,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
            )
        else:
            x_train, y_train = features, labels
            x_test, y_test = features, labels
            logger.warning("Small training set; skipping held-out validation split")

        model = LogisticRegression(solver="lbfgs", max_iter=1000)
        model.fit(x_train, y_train)

        self.b = float(model.intercept_[0])
        self.w_E = float(model.coef_[0][0])
        self.w_P = float(model.coef_[0][1])

        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]
        accuracy = float(accuracy_score(y_test, y_pred))
        try:
            auc = float(roc_auc_score(y_test, y_prob))
        except ValueError:
            auc = None

        metrics = {
            "accuracy": accuracy,
            "auc": auc,
            "b": self.b,
            "w_E": self.w_E,
            "w_P": self.w_P,
            "n_samples": int(len(labels)),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
        }
        self.save_escalation_parameters(metrics)
        logger.info("Trained escalation logistic model metrics=%s", metrics)
        return metrics

    def save_escalation_parameters(self, extra=None, path=None):
        path = Path(path or ESCALATION_PARAM_PATH)
        payload = {
            "b": self.b,
            "w_E": self.w_E,
            "w_P": self.w_P,
        }
        if extra:
            payload.update(extra)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved escalation parameters to %s", path)
        return payload

    def load_escalation_parameters(self, path=None):
        path = Path(path or ESCALATION_PARAM_PATH)
        if not path.is_file():
            logger.info(
                "No saved escalation parameters; using defaults b=%s w_E=%s w_P=%s",
                self.b,
                self.w_E,
                self.w_P,
            )
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.b = float(payload["b"])
        self.w_E = float(payload["w_E"])
        self.w_P = float(payload["w_P"])
        logger.info("Loaded escalation parameters from %s", path)
        return True

    def _training_matrix(self, training_data):
        if not training_data:
            raise ValueError("training_data is required")
        rows = []
        labels = []
        for item in training_data:
            e_score, pacl, label = self._training_row(item)
            rows.append([e_score, pacl])
            labels.append(label)
        return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int)

    def _training_row(self, item):
        if not isinstance(item, dict):
            raise ValueError("Each training row must be a dict")
        label = item.get("label", item.get("escalation"))
        if label not in (0, 1):
            raise ValueError("Each conversation must have escalation label 0 or 1")

        if "escalation_feature" in item or "E" in item:
            e_score = self._validate_number("E", item.get("escalation_feature", item.get("E")))
        elif "delta_x" in item:
            e_score = self.calculate_cel(item["delta_x"], item.get("weight_vector"))
        elif "current_toxicity_vector" in item:
            previous = item.get("previous_toxicity_vector") or ZERO_DELTA
            delta = self.calculate_delta_x(item["current_toxicity_vector"], previous)
            e_score = self.calculate_cel(delta, item.get("weight_vector"))
        else:
            raise ValueError("Training row missing CEL inputs (E or toxicity vectors)")

        if "pacl" in item or "PACL" in item or "pacl_score" in item:
            pacl = self._validate_unit(
                "PACL", item.get("pacl", item.get("PACL", item.get("pacl_score")))
            )
        elif "current_toxicity_score" in item and "baseline_toxic" in item:
            pacl = self.calculate_pacl(
                item["current_toxicity_score"],
                item["baseline_toxic"],
                item.get("lambda_const"),
            )
        else:
            raise ValueError("Training row missing PACL inputs")
        return e_score, pacl, int(label)

    @staticmethod
    def _sigmoid(z_score):
        clipped = max(-60.0, min(60.0, float(z_score)))
        return 1.0 / (1.0 + math.exp(-clipped))

    def calculate_delta_x(self, current_toxicity_vector, previous_toxicity_vector):
        current = self._as_toxicity_vector(current_toxicity_vector, "X_t")
        previous = self._as_toxicity_vector(previous_toxicity_vector, "X_t-1")
        delta = [round(curr - prev, 10) for curr, prev in zip(current, previous)]
        logger.info("CEL ΔX=%s", delta)
        return delta

    def calculate_cel(self, delta_x, weight_vector=None):
        delta = self._as_delta_vector(delta_x)
        weights = self._as_weight_vector(weight_vector)
        escalation = sum(weight * change for weight, change in zip(weights, delta))
        logger.info("CEL E=W·ΔX W=%s ΔX=%s E=%.6f", weights, delta, escalation)
        return float(escalation)

    def get_escalation_feature(self, user_id, message_id):
        session = get_session()
        try:
            message = (
                session.query(Message)
                .filter(Message.message_id == message_id, Message.user_id == user_id)
                .one_or_none()
            )
            if message is None:
                raise ValueError(f"Message {message_id} not found for user {user_id}")

            current_toxicity = (
                session.query(ToxicityResult)
                .filter(ToxicityResult.message_id == message_id)
                .one_or_none()
            )
            if current_toxicity is None:
                raise ValueError(f"No toxicity result for message {message_id}")

            previous_message = (
                session.query(Message)
                .filter(
                    Message.user_id == user_id,
                    Message.conversation_id == message.conversation_id,
                    Message.timestamp < message.timestamp,
                    Message.message_id != message.message_id,
                )
                .order_by(Message.timestamp.desc())
                .first()
            )
            previous_toxicity = None
            if previous_message is not None:
                previous_toxicity = (
                    session.query(ToxicityResult)
                    .filter(ToxicityResult.message_id == previous_message.message_id)
                    .one_or_none()
                )

            if previous_toxicity is None:
                delta = list(ZERO_DELTA)
                escalation = 0.0
                logger.info(
                    "CEL first message user_id=%s message_id=%s E=0",
                    user_id,
                    message_id,
                )
            else:
                delta = self.calculate_delta_x(
                    self._toxicity_row_to_vector(current_toxicity),
                    self._toxicity_row_to_vector(previous_toxicity),
                )
                escalation = self.calculate_cel(delta)

            self._store_escalation_feature(session, message_id, delta, escalation)
            session.commit()
            return escalation
        except Exception:
            session.rollback()
            logger.exception(
                "CEL failed user_id=%s message_id=%s", user_id, message_id
            )
            raise
        finally:
            close_session()

    def _store_escalation_feature(self, session, message_id, delta, escalation):
        risk = (
            session.query(RiskResult)
            .filter(RiskResult.message_id == message_id)
            .one_or_none()
        )
        if risk is None:
            risk = RiskResult(
                message_id=message_id,
                pacl=0.0,
                delta_vector_6d=list(delta),
                escalation_feature=float(escalation),
                escalation_probability=0.0,
                final_risk=0.0,
                decision="ALLOW",
            )
            session.add(risk)
        else:
            risk.delta_vector_6d = list(delta)
            risk.escalation_feature = float(escalation)
        logger.info(
            "Stored CEL message_id=%s E=%.6f ΔX=%s",
            message_id,
            escalation,
            delta,
        )

    @staticmethod
    def _toxicity_row_to_vector(row):
        return [float(getattr(row, label)) for label in CEL_LABELS]

    def _as_toxicity_vector(self, vector, name):
        values = self._coerce_six(vector, name)
        return [self._validate_unit(f"{name}[{label}]", value) for label, value in zip(CEL_LABELS, values)]

    def _as_delta_vector(self, vector):
        values = self._coerce_six(vector, "ΔX")
        return [self._validate_number("ΔX", value) for value in values]

    def _as_weight_vector(self, weight_vector):
        if weight_vector is None:
            values = list(self.weight_vector)
        else:
            values = self._coerce_six(weight_vector, "W")
            values = [self._validate_number("W", value) for value in values]
        return values

    @staticmethod
    def _coerce_six(vector, name):
        if isinstance(vector, dict):
            aliases = {
                "T": "toxic",
                "H": "threat",
                "I": "insult",
                "S": "severe_toxic",
                "O": "obscene",
                "IH": "identity_hate",
            }
            values = []
            for label in CEL_LABELS:
                short = next(key for key, value in aliases.items() if value == label)
                item = vector.get(label, vector.get(short))
                if item is None:
                    raise ValueError(f"{name} must include all 6 toxicity dimensions {CEL_LABELS}")
                values.append(item)
            return values
        if not isinstance(vector, (list, tuple)) or len(vector) != 6:
            raise ValueError(f"{name} must be a 6-dimensional vector [T, H, I, S, O, IH]")
        return list(vector)

    @staticmethod
    def _validate_number(name, value):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} must be a finite number")
        return number

    @staticmethod
    def _validate_unit(name, value):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number") from exc
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{name} must be a finite number in [0, 1]")
        return number
