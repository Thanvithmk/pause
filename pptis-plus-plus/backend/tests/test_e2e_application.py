import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ["FLASK_ENV"] = "testing"

# Slang and toxicity calls in this file use the loaded models.
# Groq completions are mocked so a remote API error cannot be stored as a rewrite.
REWRITE_INFERENCE = "mocked"

from api.routes import services
from app import create_app
from modules.database import Feedback, Message, RiskResult, close_session, get_session
from tests.fixtures import TEST_PASSWORD


class EndToEndApplicationTests(unittest.TestCase):
    def setUp(self):
        services.reset()
        self.app = create_app("testing")
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        self.user_a = self._register("e2ealpha", "e2ealpha@example.com")
        self.headers_a = {"Authorization": f"Bearer {self.user_a['token']}"}

    def tearDown(self):
        services.reset()
        close_session()

    def _register(self, username, email):
        response = self.client.post(
            "/api/auth/register",
            json={"username": username, "email": email, "password": TEST_PASSWORD},
        )
        self.assertEqual(response.status_code, 201, response.get_json())
        return response.get_json()

    def _analyze(self, text, conversation_id=None, headers=None):
        payload = {"text": text}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        with patch(
            "modules.rewrite_engine.RewriteEngine._complete_groq",
            return_value="I want to discuss this more calmly.",
        ):
            response = self.client.post(
                "/api/analyze", json=payload, headers=headers or self.headers_a
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
            headers=headers or self.headers_a,
        )
        return response.status_code, response.get_json()

    def test_unauthenticated_analyze_rejected(self):
        response = self.client.post("/api/analyze", json={"text": "hello"})
        self.assertEqual(response.status_code, 401)

    def test_register_conversation_allow_send_and_history(self):
        created = self.client.post("/api/conversations", headers=self.headers_a)
        self.assertEqual(created.status_code, 201, created.get_json())
        conversation_id = created.get_json()["conversation"]["conversation_id"]

        status, analysis = self._analyze("Thanks for reviewing my draft.", conversation_id)
        self.assertEqual(status, 200)
        self.assertEqual(analysis["decision"], "ALLOW")
        self.assertIn("slang", analysis)
        self.assertEqual(len(analysis["toxicity"]), 6)

        send_status, sent = self._send(analysis)
        self.assertEqual(send_status, 200, sent)
        self.assertTrue(sent["sent"])

        listed = self.client.get("/api/conversations", headers=self.headers_a)
        self.assertEqual(listed.status_code, 200)
        ids = [row["conversation_id"] for row in listed.get_json()["conversations"]]
        self.assertIn(conversation_id, ids)

        thread = self.client.get(f"/api/conversation/{conversation_id}", headers=self.headers_a)
        self.assertEqual(thread.status_code, 200)
        messages = thread.get_json()["messages"]
        self.assertGreaterEqual(len(messages), 1)
        self.assertEqual(messages[-1]["decision"]["decision"], "ALLOW")
        self.assertEqual(messages[-1]["message"]["send_status"], "sent")

    def test_warn_rewrite_delayed_and_edit(self):
        self.assertEqual(REWRITE_INFERENCE, "mocked")
        status, analysis = self._analyze("This deadline feels unrealistic given the remaining work.")
        self.assertEqual(status, 200, analysis)
        self.assertTrue(analysis["analysis_complete"])
        self.assertEqual(analysis["inference"]["toxicity"], "actual")
        self.assertEqual(analysis["mathematical_decision"], _band(analysis["final_risk"]))
        if analysis.get("guardrail"):
            self.assertGreater(
                ("ALLOW", "WARN", "REWRITE", "DELAYED-SEND").index(analysis["decision"]),
                ("ALLOW", "WARN", "REWRITE", "DELAYED-SEND").index(analysis["mathematical_decision"]),
            )
        else:
            self.assertEqual(analysis["decision"], _band(analysis["final_risk"]))
        if analysis["decision"] == "WARN":
            blocked, body = self._send(analysis)
            self.assertEqual(blocked, 409, body)
            allowed, sent = self._send(analysis, user_action="send_anyway")
            self.assertEqual(allowed, 200, sent)
        elif analysis["decision"] == "ALLOW":
            self.assertEqual(self._send(analysis)[0], 200)

        session = get_session()
        feedback = session.query(Feedback).filter(Feedback.message_id == analysis["message_id"]).first()
        close_session()
        if analysis["decision"] == "ALLOW":
            self.assertIsNone(feedback)
        else:
            self.assertIsNotNone(feedback)

        _, rewrite = self._analyze("You're a joke in this group.")
        self.assertEqual(rewrite["decision"], "REWRITE")
        rewrite_send, rewrite_body = self._send(rewrite)
        self.assertEqual(rewrite_send, 403, rewrite_body)

        _, delayed = self._analyze("I will hurt you after class.")
        self.assertEqual(delayed["decision"], "DELAYED-SEND")
        delayed_send, delayed_body = self._send(delayed)
        self.assertEqual(delayed_send, 403, delayed_body)

        edited_status, edited = self._send(delayed, text="I will hurt you after class now")
        self.assertEqual(edited_status, 409, edited)

    def test_slang_is_not_toxicity_by_itself(self):
        _, analysis = self._analyze("Bro, that movie was fire")
        self.assertEqual(analysis["decision"], "ALLOW")
        self.assertEqual(analysis["original_text"], "Bro, that movie was fire")
        self.assertEqual(analysis["slang"][0]["term"].lower(), "fire")
        self.assertEqual(analysis["slang"][0]["usage"], "slang")
        self.assertTrue(analysis["slang"][0]["meaning"])
        self.assertEqual(set(analysis["behavioral_baseline"]), {
            "toxic", "threat", "insult", "severe_toxic", "obscene", "identity_hate",
        })
        self.assertEqual(len(analysis["delta_vector"]), 6)
        self.assertIn("escalation_feature", analysis)
        self.assertIn("escalation_probability", analysis)
        self.assertIn("pacl", analysis)
        self.assertIn("final_risk", analysis)

    def test_cross_user_conversation_and_delayed_status(self):
        _, analysis = self._analyze("I will hurt you after class.")
        other = self._register("e2ebravo", "e2ebravo@example.com")
        headers_b = {"Authorization": f"Bearer {other['token']}"}
        stolen = self.client.get(
            f"/api/conversation/{analysis['conversation_id']}",
            headers=headers_b,
        )
        self.assertEqual(stolen.status_code, 403)
        lock = self.client.get(
            f"/api/delayed-send/{analysis['message_id']}",
            headers=headers_b,
        )
        self.assertEqual(lock.status_code, 403)
        send_other, body = self._send(analysis, headers=headers_b)
        self.assertIn(send_other, (403, 404), body)

    def test_dashboard_uses_stored_records(self):
        _, analysis = self._analyze("Thanks a lot for the notes.")
        self._send(analysis)
        dash = self.client.get("/api/dashboard", headers=self.headers_a)
        self.assertEqual(dash.status_code, 200, dash.get_json())
        payload = dash.get_json()
        self.assertGreaterEqual(payload["messages_analyzed"], 1)
        self.assertGreaterEqual(payload["messages_sent"], 1)
        self.assertIn("average_risk", payload)
        self.assertIn("slang_statistics", payload)
        self.assertIn("behavior_profile", payload)
        allow = next(item for item in payload["message_outcomes"] if item["name"] == "allow")
        self.assertGreaterEqual(len(payload["recent_activity"]), 1)
        self.assertIn("send_status", payload["recent_activity"][0])

    def test_logout_revokes_token(self):
        logout = self.client.post("/api/auth/logout", headers=self.headers_a)
        self.assertEqual(logout.status_code, 200, logout.get_json())
        after = self.client.get("/api/user/profile", headers=self.headers_a)
        self.assertEqual(after.status_code, 401)

    def test_voice_transcription_is_deferred(self):
        response = self.client.post("/api/analyze-voice", headers=self.headers_a)
        body = response.get_json()
        self.assertEqual(response.status_code, 503, body)
        self.assertIn("currently deferred", body["message"])

    def test_single_conversation_demonstration(self):
        created = self.client.post("/api/conversations", headers=self.headers_a)
        self.assertEqual(created.status_code, 201, created.get_json())
        conversation_id = created.get_json()["conversation"]["conversation_id"]

        status, first = self._analyze("I don't agree with this.", conversation_id)
        self.assertEqual(status, 200, first)
        self.assertEqual(first["decision"], "ALLOW")
        self.assertLess(first["toxicity"]["toxic"], 0.30)
        self.assertLess(first["pacl"], 0.30)
        self.assertLess(first["escalation_feature"], 0.30)
        self.assertLess(first["final_risk"], 0.30)
        self.assertEqual(len(first["delta_vector"]), 6)
        self.assertEqual(set(first["behavioral_baseline"]), {
            "toxic", "threat", "insult", "severe_toxic", "obscene", "identity_hate",
        })
        sent_status, sent = self._send(first)
        self.assertEqual(sent_status, 200, sent)

        status, second = self._analyze("Bro you're cooked 💀", conversation_id)
        self.assertEqual(status, 200, second)
        cooked = next(item for item in second["slang"] if item["term"].lower() == "cooked")
        self.assertEqual(cooked["usage"], "slang")
        self.assertIsInstance(cooked["confidence"], float)
        self.assertGreater(cooked["confidence"], 0.5)
        self.assertIn("trouble", cooked["meaning"])
        self.assertTrue(second["enriched_text"].startswith("Bro you're cooked 💀"))
        self.assertIn("[Contextual meaning:", second["enriched_text"])
        self.assertNotIn("Bro you're defeated", second["enriched_text"])
        self.assertEqual(second["decision"], _band(second["final_risk"]))
        if second["decision"] == "ALLOW":
            self.assertEqual(self._send(second)[0], 200)

        status, third = self._analyze("You're completely useless.", conversation_id)
        self.assertEqual(status, 200, third)
        self.assertGreater(third["toxicity"]["toxic"], second["toxicity"]["toxic"])
        self.assertGreater(third["pacl"], second["pacl"])
        self.assertGreater(third["escalation_feature"], 0)
        self.assertGreater(third["escalation_feature"], second["escalation_feature"])
        self.assertGreater(third["final_risk"], second["final_risk"])
        self.assertIn(third["decision"], ("REWRITE", "DELAYED-SEND"))
        self.assertEqual(third["decision"], _band(third["final_risk"]))
        blocked, blocked_body = self._send(third)
        self.assertEqual(blocked, 403, blocked_body)

        if third["decision"] != "DELAYED-SEND":
            return
        self.assertFalse(third["allow_send_anyway"])
        self.assertGreaterEqual(len(third["delayed_send"]["exercises"]), 1)

        from datetime import timedelta

        from modules.database import utcnow
        from modules.delayed_send import DelayedSendLock

        session = get_session()
        lock = session.query(DelayedSendLock).filter_by(message_id=third["message_id"]).one()
        lock.expires_at = utcnow() - timedelta(seconds=2)
        session.commit()
        close_session()

        still_blocked, still_body = self._send(third)
        self.assertEqual(still_blocked, 403, still_body)

        status, calm = self._analyze("I don't agree with this.", conversation_id)
        self.assertEqual(status, 200, calm)
        self.assertEqual(calm["decision"], "ALLOW")
        ok, sent_body = self._send(calm)
        self.assertEqual(ok, 200, sent_body)

        session = get_session()
        feedback = session.query(Feedback).filter(Feedback.message_id == third["message_id"]).one()
        close_session()
        self.assertEqual(feedback.intervention, "DELAYED-SEND")
        self.assertEqual(feedback.user_response, "edited")
        self.assertEqual(feedback.outcome, "positive")

    def test_invalid_message_id_send(self):
        response = self.client.post(
            "/api/send",
            json={"message_id": 999999, "conversation_id": 1, "text": "hi"},
            headers=self.headers_a,
        )
        self.assertIn(response.status_code, (403, 404, 422))


def _band(score):
    if score < 0.30:
        return "ALLOW"
    if score < 0.55:
        return "WARN"
    if score < 0.80:
        return "REWRITE"
    return "DELAYED-SEND"


if __name__ == "__main__":
    unittest.main()
