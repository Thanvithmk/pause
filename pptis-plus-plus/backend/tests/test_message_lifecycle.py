import math
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["FLASK_ENV"] = "testing"

from api.routes import services
from app import create_app
from modules.database import (
    BehaviorProfile,
    InterventionLog,
    Message,
    RiskResult,
    close_session,
    get_session,
)
from modules.rewrite_engine import RewriteServiceError
from tests.fixtures import TEST_PASSWORD
from tests.flow_cases import FLOW_CASES

LABELS = ("toxic", "threat", "insult", "severe_toxic", "obscene", "identity_hate")
DECISIONS = ("ALLOW", "WARN", "REWRITE", "DELAYED-SEND")


def _band(score):
    if score < 0.30:
        return "ALLOW"
    if score < 0.55:
        return "WARN"
    if score < 0.80:
        return "REWRITE"
    return "DELAYED-SEND"


class MessageLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        services.reset()
        cls.app = create_app("testing")
        cls.client = cls.app.test_client()
        cls.rewrite_patch = patch(
            "modules.rewrite_engine.RewriteEngine._complete_groq",
            side_effect=RewriteServiceError(
                "rewrite: Groq request failed for openai/gpt-oss-120b"
            ),
        )
        cls.rewrite_patch.start()
        cls.headers = {
            "Authorization": f"Bearer {cls._register(cls.client, 'flowuser', 'flowuser@example.com')['token']}"
        }
        cls.metrics = cls._score_catalog()

    @classmethod
    def tearDownClass(cls):
        cls.rewrite_patch.stop()
        services.reset()
        close_session()

    @staticmethod
    def _register(client, username, email):
        response = client.post(
            "/api/auth/register",
            json={"username": username, "email": email, "password": TEST_PASSWORD},
        )
        if response.status_code != 201:
            raise AssertionError(response.get_json())
        return response.get_json()

    def _analyze(self, text, conversation_id=None, headers=None):
        payload = {"text": text}
        if conversation_id is not None:
            payload["conversation_id"] = conversation_id
        response = self.client.post(
            "/api/analyze",
            json=payload,
            headers=headers or self.headers,
        )
        return response.status_code, response.get_json()

    def _send(self, analysis, text=None, user_action="send", headers=None):
        response = self.client.post(
            "/api/send",
            json={
                "message_id": analysis["message_id"],
                "conversation_id": analysis["conversation_id"],
                "text": text if text is not None else analysis["original_text"],
                "user_action": user_action,
            },
            headers=headers or self.headers,
        )
        return response.status_code, response.get_json()

    @classmethod
    def _score_catalog(cls):
        preprocessor = services.preprocessor
        slang = services.slang
        toxicity = services.toxicity
        risk = services.risk
        decisions = services.decisions
        prepared = []
        enriched_texts = []
        for case in FLOW_CASES:
            cleaned = preprocessor.preprocess(case["text"])["cleaned_text"]
            interpreted = slang.interpret(cleaned)
            enriched = interpreted.get("enriched_text_with_context") or cleaned
            prepared.append((case, cleaned, interpreted, enriched))
            enriched_texts.append(enriched)
        details = toxicity.inspect_many(enriched_texts)
        rows = []
        matrix = {gold: {pred: 0 for pred in DECISIONS} for gold in DECISIONS}
        for (case, cleaned, interpreted, enriched), detail in zip(prepared, details):
            vector = detail["model"]
            values = [float(vector[label]) for label in LABELS]
            for value in values:
                if value < 0 or value > 1:
                    raise AssertionError(f"score out of range for {case['text']}")
            pacl = risk.calculate_pacl(values[0], 0.0)
            probability = risk.calculate_escalation_probability(0.0, pacl)
            final_risk = risk.calculate_final_risk(pacl, probability)
            mathematical = _band(final_risk)
            chosen = decisions.make_decision(
                final_risk,
                toxicity_vector=vector,
                safety_flags={
                    "quoted": detail["quoted"],
                    "negated": detail["negated"],
                    "playful_idiom": detail["playful_idiom"],
                },
                safety_label=detail["safety_label"],
            )
            if abs(final_risk - (0.5 * pacl + 0.5 * probability)) > 1e-9:
                raise AssertionError("final risk left the locked formula")
            if chosen["guardrail"] is None and chosen["decision"] != mathematical:
                raise AssertionError("decision left the locked thresholds")
            if chosen["guardrail"] is not None:
                if chosen["risk_score"] != final_risk:
                    raise AssertionError("guardrail changed the risk score")
                if DECISIONS.index(chosen["decision"]) <= DECISIONS.index(mathematical):
                    raise AssertionError("guardrail did not raise the intervention")
            if not cleaned.strip() or not enriched.strip():
                raise AssertionError("preprocessing removed the message")
            matrix[case["gold"]][chosen["decision"]] += 1
            rows.append({
                "case": case,
                "enriched": enriched,
                "detections": interpreted.get("detections") or [],
                "vector": vector,
                "decision": chosen["decision"],
                "guardrail": chosen["guardrail"],
                "final_risk": final_risk,
            })
        return {"rows": rows, "matrix": matrix}

    def test_catalog_has_at_least_250_authored_cases(self):
        self.assertGreaterEqual(len(FLOW_CASES), 250)
        texts = [case["text"] for case in FLOW_CASES]
        self.assertEqual(len(texts), len(set(texts)))

    def test_catalog_scores_follow_the_formula_and_report_metrics(self):
        matrix = self.metrics["matrix"]
        print("\nDECISION_CONFUSION gold rows, predicted columns ALLOW WARN REWRITE DELAYED-SEND")
        for gold in DECISIONS:
            print(gold, [matrix[gold][pred] for pred in DECISIONS])
        per_class = {}
        for label in DECISIONS:
            tp = matrix[label][label]
            fp = sum(matrix[other][label] for other in DECISIONS if other != label)
            fn = sum(matrix[label][other] for other in DECISIONS if other != label)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
        print("PER_CLASS", per_class)
        threats = [row for row in self.metrics["rows"] if row["case"]["threat"]]
        threat_hits = sum(1 for row in threats if row["decision"] == "DELAYED-SEND")
        threat_recall = threat_hits / len(threats) if threats else 0.0
        harmless = [row for row in self.metrics["rows"] if row["case"]["harmless"]]
        harmless_fp = sum(1 for row in harmless if row["decision"] != "ALLOW")
        harmless_fpr = harmless_fp / len(harmless) if harmless else 0.0
        print("THREAT_RECALL", threat_recall, "of", len(threats))
        print("HARMLESS_FPR", harmless_fpr, "of", len(harmless))
        slang_rows = [row for row in self.metrics["rows"] if row["case"].get("slang_usage")]
        slang_correct = 0
        for row in slang_rows:
            term = row["case"]["slang_term"].lower()
            found = next((item for item in row["detections"] if str(item.get("term", "")).lower() == term), None)
            if found and found.get("usage") == row["case"]["slang_usage"]:
                slang_correct += 1
            elif found is None and row["case"]["slang_usage"] == "unknown":
                slang_correct += 1
        slang_accuracy = slang_correct / len(slang_rows) if slang_rows else 0.0
        print("SLANG_USAGE_ACCURACY", slang_accuracy, "of", len(slang_rows))
        self.assertGreater(len(threats), 0)
        self.assertGreater(len(harmless), 0)
        self.assertGreater(len(slang_rows), 0)

    def test_enrichment_keeps_threat_words_and_does_not_score_unknown_slang_as_slang(self):
        threat = next(row for row in self.metrics["rows"] if row["case"]["text"] == "I will hurt you after class.")
        self.assertIn("hurt", threat["enriched"].lower())
        unknown = next(row for row in self.metrics["rows"] if "glorped" in row["case"]["text"])
        self.assertNotIn("[Contextual meaning:", unknown["enriched"])
        usages = [item.get("usage") for item in unknown["detections"]]
        self.assertIn("unknown", usages)

    def test_negated_sentence_keeps_model_scores_in_the_risk_formula(self):
        status, body = self._analyze("I am angry, but I do not want to hurt anyone.")
        self.assertEqual(status, 200, body)
        self.assertTrue(body["safety"]["negated"])
        self.assertEqual(body["toxicity"], body["model_toxicity"])
        self.assertGreater(body["toxicity"]["toxic"], 0.0)
        tb = body["toxicity"]["toxic"]
        pacl = min(1.0, max(0.0, tb + 0.5 * (tb - 0.0)))
        pesc = 1.0 / (1.0 + math.exp(-(0.0 + pacl)))
        risk = 0.5 * pacl + 0.5 * pesc
        self.assertAlmostEqual(body["pacl"], pacl, places=9)
        self.assertAlmostEqual(body["escalation_feature"], 0.0, places=9)
        self.assertAlmostEqual(body["escalation_probability"], pesc, places=9)
        self.assertAlmostEqual(body["final_risk"], risk, places=9)
        self.assertEqual(body["mathematical_decision"], _band(risk))
        self.assertNotEqual(body["send_status"], "sent")

    def test_known_decision_examples_and_send_restrictions(self):
        expectations = {
            "Thanks for reviewing my draft.": "ALLOW",
            "You never listen to what I'm saying.": "WARN",
            "This whole plan is trash. Let's start over.": "REWRITE",
            "I will hurt you after class.": "DELAYED-SEND",
        }
        for text, decision in expectations.items():
            status, body = self._analyze(text)
            self.assertEqual(status, 200, body)
            self.assertEqual(body["decision"], decision, body)
            self.assertEqual(body["original_text"], text)
            self.assertTrue(body["analysis_complete"])
            self.assertEqual(len(body["toxicity"]), 6)
            self.assertEqual(body["mathematical_decision"], _band(body["final_risk"]))
            if decision == "ALLOW":
                self.assertTrue(body["can_send"])
                self.assertEqual(self._send(body)[0], 200)
            elif decision == "WARN":
                blocked, payload = self._send(body)
                self.assertEqual(blocked, 409, payload)
                self.assertEqual(self._send(body, user_action="send_anyway")[0], 200)
            else:
                self.assertFalse(body["can_send"])
                self.assertIsNone(body.get("suggested_text"))
                blocked, payload = self._send(body)
                self.assertEqual(blocked, 403, payload)
                edited, edit_body = self._send(body, text=text + " edited")
                self.assertEqual(edited, 409, edit_body)

    def test_rewrite_unavailable_is_recorded_and_original_stays_blocked(self):
        status, body = self._analyze("This whole plan is trash. Let's start over.")
        self.assertEqual(status, 200, body)
        self.assertEqual(body["rewrite"]["status"], "unavailable")
        self.assertIn("openai/gpt-oss-120b", body["rewrite"]["error"])
        self.assertNotEqual(body["inference"]["rewrite"], "actual")
        session = get_session()
        try:
            log = (
                session.query(InterventionLog)
                .filter(InterventionLog.message_id == body["message_id"])
                .one()
            )
            self.assertEqual(log.action, "REWRITE")
            self.assertEqual(log.user_action, "rewrite_unavailable")
            self.assertIsNone(log.suggested_text)
        finally:
            close_session()

    def test_conversation_cel_uses_only_the_previous_message_in_that_conversation(self):
        created = self.client.post("/api/conversations", headers=self.headers)
        conversation_id = created.get_json()["conversation"]["conversation_id"]
        texts = [
            "I disagree.",
            "You never listen.",
            "You're completely useless.",
            "I will hurt you after class.",
        ]
        analyses = []
        for text in texts:
            status, body = self._analyze(text, conversation_id)
            self.assertEqual(status, 200, body)
            analyses.append(body)
        self.assertIsNone(analyses[0]["previous_vector"])
        self.assertEqual(analyses[0]["delta_vector"], [0.0] * 6)
        self.assertEqual(analyses[0]["escalation_feature"], 0.0)
        for previous, current in zip(analyses, analyses[1:]):
            expected_previous = [previous["toxicity"][label] for label in LABELS]
            self.assertEqual(current["previous_vector"], expected_previous)
            for index, label in enumerate(LABELS):
                self.assertAlmostEqual(
                    current["delta_vector"][index],
                    current["toxicity"][label] - expected_previous[index],
                )
            self.assertAlmostEqual(
                current["escalation_feature"],
                sum(current["delta_vector"]),
            )

    def test_conversations_and_users_do_not_share_previous_vectors(self):
        other = self._register(self.client, "flowother", "flowother@example.com")
        other_headers = {"Authorization": f"Bearer {other['token']}"}
        first_conv = self.client.post("/api/conversations", headers=self.headers).get_json()["conversation"]["conversation_id"]
        second_conv = self.client.post("/api/conversations", headers=self.headers).get_json()["conversation"]["conversation_id"]
        status, first = self._analyze("Thanks for the notes from yesterday.", first_conv)
        self.assertEqual(status, 200, first)
        status, foreign = self._analyze("I will hurt you after class.", headers=other_headers)
        self.assertEqual(status, 200, foreign)
        status, isolated = self._analyze("Thanks for explaining the diagram.", second_conv)
        self.assertEqual(status, 200, isolated)
        self.assertIsNone(isolated["previous_vector"])
        status, follow = self._analyze("I don't agree with this.", first_conv)
        self.assertEqual(status, 200, follow)
        self.assertEqual(follow["previous_vector"], [first["toxicity"][label] for label in LABELS])
        self.assertNotEqual(
            follow["previous_vector"],
            [foreign["toxicity"][label] for label in LABELS],
        )

    def test_profile_updates_once_on_send_and_not_on_analysis_or_cancel(self):
        fresh = self._register(self.client, "flowprof", "flowprof@example.com")
        headers = {"Authorization": f"Bearer {fresh['token']}"}
        status, body = self._analyze("You never listen to what I'm saying.", headers=headers)
        self.assertEqual(status, 200, body)
        self.assertEqual(body["decision"], "WARN", body)
        self.assertEqual(body["profile"]["message_count"], 0)
        self.assertEqual(body["behavioral_baseline"]["toxic"], 0.0)
        cancelled = self.client.post(
            "/api/intervention-response",
            json={"message_id": body["message_id"], "user_response": "cancel"},
            headers=headers,
        )
        self.assertEqual(cancelled.status_code, 200, cancelled.get_json())
        self.assertFalse(cancelled.get_json()["sent"])
        session = get_session()
        try:
            profile = session.query(BehaviorProfile).filter(BehaviorProfile.user_id == body["profile"]["user_id"]).one()
            self.assertEqual(profile.message_count, 0)
        finally:
            close_session()
        allowed, sent = self._send(body, user_action="send_anyway", headers=headers)
        self.assertEqual(allowed, 409, sent)
        status, fresh_body = self._analyze("Thanks for reviewing my draft.", headers=headers)
        self.assertEqual(status, 200, fresh_body)
        self.assertEqual(fresh_body["decision"], "ALLOW")
        allowed, sent = self._send(fresh_body, headers=headers)
        self.assertEqual(allowed, 200, sent)
        session = get_session()
        try:
            profile = session.query(BehaviorProfile).filter(BehaviorProfile.user_id == body["profile"]["user_id"]).one()
            self.assertEqual(profile.message_count, 1)
            message = session.get(Message, fresh_body["message_id"])
            self.assertTrue(message.profile_updated)
            cancelled_row = session.get(Message, body["message_id"])
            self.assertEqual(cancelled_row.send_status, "cancelled")
            self.assertFalse(cancelled_row.profile_updated)
        finally:
            close_session()
        again, payload = self._send(fresh_body, headers=headers)
        self.assertEqual(again, 409, payload)
        session = get_session()
        try:
            profile = session.query(BehaviorProfile).filter(BehaviorProfile.user_id == body["profile"]["user_id"]).one()
            self.assertEqual(profile.message_count, 1)
        finally:
            close_session()

    def test_slang_context_is_the_current_sentence_only(self):
        created = self.client.post("/api/conversations", headers=self.headers)
        conversation_id = created.get_json()["conversation"]["conversation_id"]
        self._analyze("The earlier note mentioned cooked chicken in the oven.", conversation_id)
        seen = {}
        original = services.slang.interpret

        def spy(text, user_id=None):
            seen["text"] = text
            return original(text, user_id=user_id)

        with patch.object(services.slang, "interpret", side_effect=spy):
            status, body = self._analyze("Thanks for explaining.", conversation_id)
        self.assertEqual(status, 200, body)
        self.assertNotIn("cooked", seen["text"].lower())

    def test_analysis_uses_the_last_five_messages(self):
        created = self.client.post("/api/conversations", headers=self.headers)
        conversation_id = created.get_json()["conversation"]["conversation_id"]
        earlier = [f"Earlier note {index}." for index in range(1, 7)]
        for text in earlier:
            status, body = self._analyze(text, conversation_id)
            self.assertEqual(status, 200, body)
        status, current = self._analyze("The current note.", conversation_id)
        self.assertEqual(status, 200, current)
        self.assertEqual(current["context_messages"], earlier[-5:])
        self.assertNotIn(earlier[0], current["context_messages"])

    def test_dashboard_effectiveness_comes_from_stored_feedback(self):
        status, body = self._analyze("You never listen to what I'm saying.")
        self.assertEqual(body["decision"], "WARN")
        self.assertEqual(self._send(body, user_action="send_anyway")[0], 200)
        dashboard = self.client.get("/api/dashboard", headers=self.headers)
        payload = dashboard.get_json()
        self.assertEqual(dashboard.status_code, 200, payload)
        self.assertIn("intervention_effectiveness", payload)
        counts = payload["intervention_effectiveness"]["counts"]
        self.assertGreaterEqual(counts["warn_fail"], 1)
        total = counts["warn_success"] + counts["warn_fail"]
        self.assertAlmostEqual(
            payload["intervention_effectiveness"]["scores"]["WARN_eff"],
            counts["warn_success"] / total,
        )
        self.assertLess(payload["intervention_effectiveness"]["scores"]["WARN_eff"], 1)
        session = get_session()
        try:
            risk = session.query(RiskResult).filter(RiskResult.message_id == body["message_id"]).one()
            self.assertEqual(risk.mathematical_decision, _band(risk.final_risk))
            self.assertEqual(len(risk.delta_vector_6d), 6)
        finally:
            close_session()


if __name__ == "__main__":
    unittest.main()
