import os
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ["FLASK_ENV"] = "testing"

from api.routes import services
from app import create_app
from modules.database import RiskResult, close_session, get_session
from tests.fixtures import TEST_PASSWORD


class SendPolicyTests(unittest.TestCase):
    def setUp(self):
        services.reset()
        self.app = create_app("testing")
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        register = self.client.post(
            "/api/auth/register",
            json={
                "username": "senduser",
                "email": "senduser@example.com",
                "password": TEST_PASSWORD,
            },
        )
        self.assertEqual(register.status_code, 201, register.get_json())
        body = register.get_json()
        self.token = body["token"]
        self.user_id = body["user_id"]
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def tearDown(self):
        services.reset()
        close_session()

    def _analyze(self, text, conversation_id=None):
        payload = {"text": text}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        response = self.client.post("/api/analyze", json=payload, headers=self.headers)
        return response.status_code, response.get_json()

    def _send(self, analysis, text=None, user_action="send"):
        response = self.client.post(
            "/api/send",
            json={
                "message_id": analysis["message_id"],
                "conversation_id": analysis["conversation_id"],
                "text": text if text is not None else analysis["original_text"],
                "user_action": user_action,
            },
            headers=self.headers,
        )
        return response.status_code, response.get_json()

    def _force_decision(self, message_id, decision, risk):
        session = get_session()
        row = session.query(RiskResult).filter(RiskResult.message_id == message_id).one()
        row.decision = decision
        row.final_risk = risk
        session.commit()
        close_session()

    def test_allow_send(self):
        status, analysis = self._analyze("I like you")
        self.assertEqual(status, 200)
        self._force_decision(analysis["message_id"], "ALLOW", 0.12)
        status, payload = self._send(analysis)
        self.assertEqual(status, 200, payload)
        self.assertTrue(payload["sent"])
        status, again = self._send(analysis)
        self.assertEqual(status, 409)

    def test_warn_requires_send_anyway(self):
        status, analysis = self._analyze("Thanks for the notes.")
        self.assertEqual(status, 200)
        self._force_decision(analysis["message_id"], "WARN", 0.4)
        status, payload = self._send(analysis)
        self.assertEqual(status, 409, payload)
        status, payload = self._send(analysis, user_action="send_anyway")
        self.assertEqual(status, 200, payload)
        self.assertTrue(payload["sent"])

    def test_rewrite_blocks_original(self):
        status, analysis = self._analyze("Thanks for the update.")
        self.assertEqual(status, 200)
        self._force_decision(analysis["message_id"], "REWRITE", 0.7)
        status, payload = self._send(analysis)
        self.assertEqual(status, 403, payload)
        status, payload = self._send(analysis, user_action="send_anyway")
        self.assertEqual(status, 403, payload)

    def test_delayed_send_blocks_and_other_message(self):
        status, analysis = self._analyze("I like you")
        self.assertEqual(status, 200)
        self._force_decision(analysis["message_id"], "DELAYED-SEND", 0.9)
        manager = services.delayed
        manager.create_delayed_send_state(
            analysis["message_id"],
            self.user_id,
            analysis["original_text"],
            0.9,
            conversation_id=analysis["conversation_id"],
        )
        status, payload = self._send(analysis)
        self.assertEqual(status, 403, payload)
        second_status, second = self._analyze("another try", analysis["conversation_id"])
        self.assertEqual(second_status, 200)
        self._force_decision(second["message_id"], "ALLOW", 0.1)
        status, payload = self._send(second)
        self.assertEqual(status, 403, payload)

    def test_edited_text_rejected(self):
        status, analysis = self._analyze("I like you")
        self.assertEqual(status, 200)
        self._force_decision(analysis["message_id"], "ALLOW", 0.1)
        status, payload = self._send(analysis, text="I like you a lot now")
        self.assertEqual(status, 409, payload)

    def test_unanalyzed_message_rejected(self):
        status, analysis = self._analyze("I like you")
        self.assertEqual(status, 200)
        session = get_session()
        session.query(RiskResult).filter(RiskResult.message_id == analysis["message_id"]).delete()
        session.commit()
        close_session()
        status, payload = self._send(analysis)
        self.assertEqual(status, 422, payload)


if __name__ == "__main__":
    unittest.main()
