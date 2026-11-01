import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("PPTIS_LIGHT_MODE", "true")
os.environ["FLASK_ENV"] = "testing"

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
        _, analysis = self._analyze("This deadline feels unrealistic given the remaining work.")
        self.assertEqual(analysis["decision"], "WARN")
        blocked, body = self._send(analysis)
        self.assertEqual(blocked, 409, body)
        allowed, sent = self._send(analysis, user_action="send_anyway")
        self.assertEqual(allowed, 200, sent)

        session = get_session()
        feedback = session.query(Feedback).filter(Feedback.message_id == analysis["message_id"]).first()
        close_session()
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
        self.assertTrue(analysis.get("slang"))

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
        self.assertGreaterEqual(allow["value"], 1)

    def test_logout_revokes_token(self):
        logout = self.client.post("/api/auth/logout", headers=self.headers_a)
        self.assertEqual(logout.status_code, 200, logout.get_json())
        after = self.client.get("/api/user/profile", headers=self.headers_a)
        self.assertEqual(after.status_code, 401)

    def test_voice_without_audio_fails(self):
        response = self.client.post("/api/analyze-voice", headers=self.headers_a)
        self.assertEqual(response.status_code, 400)

    def test_invalid_message_id_send(self):
        response = self.client.post(
            "/api/send",
            json={"message_id": 999999, "conversation_id": 1, "text": "hi"},
            headers=self.headers_a,
        )
        self.assertIn(response.status_code, (403, 404, 422))


if __name__ == "__main__":
    unittest.main()
