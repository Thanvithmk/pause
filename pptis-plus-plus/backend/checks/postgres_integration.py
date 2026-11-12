"""Integration checks against the live PostgreSQL database pptis.

Inference for slang, toxicity, and Whisper is actual.
The Groq call is a real HTTP request to openai/gpt-oss-120b.
This file is not part of the SQLite unit suite.
"""

import io
import math
import os
import struct
import sys
import time
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["FLASK_ENV"] = "development"
os.environ["WERKZEUG_RUN_MAIN"] = "true"

from app import app
from modules.database import (
    BehaviorProfile,
    InterventionLog,
    Message,
    RiskResult,
    ToxicityResult,
    UnknownSlang,
    User,
    close_session,
    get_session,
)
from modules.delayed_send import DelayedSendLock
from modules.rewrite_engine import REQUIRED_GROQ_MODEL, RewriteContext, RewriteEngine, RewriteServiceError

PASSWORD = "AuditPass1!"
SUFFIX = str(int(time.time()))[-6:]
USER_A = f"audita{SUFFIX}"
USER_B = f"auditb{SUFFIX}"
EMAIL_A = f"{USER_A}@example.com"
EMAIL_B = f"{USER_B}@example.com"


def _tone_wav():
    rate = 16000
    frames = []
    for index in range(rate):
        sample = int(8000 * math.sin(2 * math.pi * 220 * index / rate))
        frames.append(struct.pack("<h", sample))
    handle = io.BytesIO()
    with wave.open(handle, "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"".join(frames))
    handle.seek(0)
    return handle


class PostgresApplicationTests(unittest.TestCase):
    """These tests write to PostgreSQL pptis and delete only the users they create."""

    @classmethod
    def setUpClass(cls):
        cls.app = app
        cls.client = app.test_client()
        cls.user_a = cls._register(USER_A, EMAIL_A)
        cls.user_b = cls._register(USER_B, EMAIL_B)
        cls.headers_a = {"Authorization": f"Bearer {cls.user_a['token']}"}
        cls.headers_b = {"Authorization": f"Bearer {cls.user_b['token']}"}

    @classmethod
    def tearDownClass(cls):
        session = get_session()
        try:
            session.query(User).filter(User.username.in_([USER_A, USER_B])).delete(synchronize_session=False)
            session.commit()
        finally:
            close_session()

    @classmethod
    def _register(cls, username, email):
        response = cls.client.post(
            "/api/auth/register",
            json={"username": username, "email": email, "password": PASSWORD},
        )
        if response.status_code != 201:
            raise RuntimeError(response.get_json())
        return response.get_json()

    def _analyze(self, text, headers=None, conversation_id=None):
        payload = {"text": text}
        if conversation_id is not None:
            payload["conversation_id"] = conversation_id
        response = self.client.post("/api/analyze", json=payload, headers=headers or self.headers_a)
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

    def test_database_is_postgresql(self):
        session = get_session()
        try:
            self.assertEqual(session.bind.dialect.name, "postgresql")
            self.assertIn("pptis", str(session.bind.url.database))
        finally:
            close_session()

    def test_login_and_profile(self):
        response = self.client.post(
            "/api/auth/login",
            json={"email": EMAIL_A, "password": PASSWORD},
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        token = response.get_json()["token"]
        profile = self.client.get("/api/user/profile", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(profile.status_code, 200, profile.get_json())
        self.assertEqual(profile.get_json()["user"]["username"], USER_A)

    def test_allow_persists_six_toxicity_scores_risk_and_profile(self):
        status, body = self._analyze("Thanks for reviewing my draft.")
        self.assertEqual(status, 200, body)
        self.assertTrue(body["analysis_complete"])
        self.assertEqual(body["inference"]["slang"], "actual")
        self.assertEqual(body["inference"]["toxicity"], "actual")
        self.assertEqual(body["decision"], "ALLOW")
        self.assertLess(body["final_risk"], 0.30)
        scores = body["toxicity"]
        self.assertEqual(
            set(scores),
            {"toxic", "threat", "insult", "severe_toxic", "obscene", "identity_hate"},
        )
        session = get_session()
        try:
            row = session.query(ToxicityResult).filter(ToxicityResult.message_id == body["message_id"]).one()
            for label in scores:
                self.assertAlmostEqual(float(getattr(row, label)), float(scores[label]), places=5)
            risk = session.query(RiskResult).filter(RiskResult.message_id == body["message_id"]).one()
            self.assertAlmostEqual(risk.pacl, body["pacl"], places=5)
            self.assertAlmostEqual(risk.escalation_probability, body["escalation_probability"], places=5)
            self.assertAlmostEqual(risk.final_risk, body["final_risk"], places=5)
            self.assertEqual(risk.decision, "ALLOW")
            log = session.query(InterventionLog).filter(InterventionLog.message_id == body["message_id"]).one()
            self.assertEqual(log.action, "ALLOW")
            profile = session.query(BehaviorProfile).filter(BehaviorProfile.user_id == self.user_a["user_id"]).one()
            self.assertEqual(profile.message_count, 0)
        finally:
            close_session()
        sent, payload = self._send(body)
        self.assertEqual(sent, 200, payload)
        session = get_session()
        try:
            profile = session.query(BehaviorProfile).filter(BehaviorProfile.user_id == self.user_a["user_id"]).one()
            self.assertGreaterEqual(profile.message_count, 1)
        finally:
            close_session()

    def test_conversation_ownership(self):
        status, body = self._analyze("Thanks for the notes today.")
        self.assertEqual(status, 200, body)
        foreign = self.client.get(
            f"/api/conversation/{body['conversation_id']}",
            headers=self.headers_b,
        )
        self.assertEqual(foreign.status_code, 403, foreign.get_json())

    def test_warn_requires_confirmation_and_edit_is_rejected(self):
        status, body = self._analyze("You never listen to what I'm saying.")
        self.assertEqual(status, 200, body)
        self.assertEqual(body["decision"], "WARN")
        self.assertEqual(body["send_status"], "pending_warning")
        self.assertNotEqual(body["send_status"], "sent")
        self.assertEqual(body["mathematical_decision"], "ALLOW")
        self.assertLess(body["final_risk"], 0.30)
        self.assertEqual(body["guardrail"], "safety_pattern:WARN")
        self.assertAlmostEqual(body["toxicity"]["toxic"], body["model_toxicity"]["toxic"])
        blocked, error = self._send(body)
        self.assertEqual(blocked, 409, error)
        edited, edit_error = self._send(body, text="A different message")
        self.assertEqual(edited, 409, edit_error)
        allowed, sent = self._send(body, user_action="send_anyway")
        self.assertEqual(allowed, 200, sent)

    def test_cancelled_warning_cannot_be_sent(self):
        status, body = self._analyze("You never listen to what I'm saying.")
        self.assertEqual(status, 200, body)
        self.assertEqual(body["send_status"], "pending_warning")
        cancelled = self.client.post(
            "/api/intervention-response",
            json={"message_id": body["message_id"], "user_response": "cancel"},
            headers=self.headers_a,
        )
        self.assertEqual(cancelled.status_code, 200, cancelled.get_json())
        self.assertFalse(cancelled.get_json()["sent"])
        blocked, error = self._send(body, user_action="send_anyway")
        self.assertEqual(blocked, 409, error)
        session = get_session()
        try:
            row = session.get(Message, body["message_id"])
            self.assertEqual(row.send_status, "cancelled")
        finally:
            close_session()

    def test_contextual_slang_uses_the_trained_head(self):
        status, body = self._analyze("Bro, that movie was fire")
        self.assertEqual(status, 200, body)
        fire = [item for item in body["slang"] if item.get("term") == "fire"]
        self.assertTrue(fire, body["slang"])
        self.assertEqual(fire[0]["source"], "xlm-roberta-classifier")
        self.assertIn(fire[0]["usage"], ("slang", "literal", "uncertain"))
        self.assertTrue(body["original_text"].startswith("Bro, that movie was fire"))

    def test_unknown_slang_is_stored(self):
        status, body = self._analyze("The group found a glorped note.")
        self.assertEqual(status, 200, body)
        self.assertTrue(body["analysis_complete"])
        session = get_session()
        try:
            row = (
                session.query(UnknownSlang)
                .filter(UnknownSlang.user_id == self.user_a["user_id"], UnknownSlang.term == "glorped")
                .one()
            )
            self.assertIn("glorped", row.sentence)
            self.assertEqual(row.review_status, "pending")
        finally:
            close_session()

    def test_negated_sentence_uses_model_scores_for_risk(self):
        status, body = self._analyze("I am angry, but I do not want to hurt anyone.")
        self.assertEqual(status, 200, body)
        self.assertTrue(body["safety"]["negated"])
        self.assertEqual(body["toxicity"], body["model_toxicity"])
        tb = body["toxicity"]["toxic"]
        self.assertGreater(tb, 0.0)
        pacl = min(1.0, max(0.0, tb + 0.5 * tb))
        pesc = 1.0 / (1.0 + math.exp(-pacl))
        risk = 0.5 * pacl + 0.5 * pesc
        self.assertAlmostEqual(body["final_risk"], risk, places=9)
        expected_status = {
            "ALLOW": "approved",
            "WARN": "pending_warning",
            "REWRITE": "pending_rewrite",
            "DELAYED-SEND": "delayed",
        }[body["mathematical_decision"]]
        self.assertEqual(body["decision"], body["mathematical_decision"])
        self.assertEqual(body["send_status"], expected_status)
        session = get_session()
        try:
            row = session.query(ToxicityResult).filter(ToxicityResult.message_id == body["message_id"]).one()
            self.assertAlmostEqual(row.toxic, tb, places=9)
            risk_row = session.query(RiskResult).filter(RiskResult.message_id == body["message_id"]).one()
            self.assertAlmostEqual(risk_row.final_risk, risk, places=9)
            self.assertNotEqual(session.get(Message, body["message_id"]).send_status, "sent")
        finally:
            close_session()

    def test_delayed_send_persists_and_blocks_sending(self):
        status, body = self._analyze("I will hurt you after class.")
        self.assertEqual(status, 200, body)
        self.assertEqual(body["decision"], "DELAYED-SEND")
        self.assertEqual(body["send_status"], "delayed")
        self.assertFalse(body["can_send"])
        session = get_session()
        try:
            lock = session.query(DelayedSendLock).filter(DelayedSendLock.message_id == body["message_id"]).one()
            self.assertTrue(lock.is_active)
            self.assertEqual(lock.user_id, self.user_a["user_id"])
        finally:
            close_session()
        blocked, error = self._send(body)
        self.assertEqual(blocked, 403, error)

    def test_rewrite_context_is_the_last_five_messages(self):
        first_status, first = self._analyze("Thanks for reviewing my draft.")
        self.assertEqual(first_status, 200, first)
        conversation_id = first["conversation_id"]
        message_ids = [first["message_id"]]
        for index in range(5):
            status, body = self._analyze(f"Thanks for note number {index}.", conversation_id=conversation_id)
            self.assertEqual(status, 200, body)
            message_ids.append(body["message_id"])
        engine = RewriteEngine()
        context = engine.get_conversation_context(conversation_id, message_ids[-1])
        self.assertEqual([item["message_id"] for item in context], message_ids[-6:-1])
        self.assertEqual(len(context), 5)

    def test_voice_transcription_uses_base_en(self):
        from modules.speech_to_text import VoiceTranscriber

        audio = _tone_wav()
        handle = __import__("tempfile").NamedTemporaryFile(delete=False, suffix=".wav")
        handle.write(audio.getvalue())
        handle.close()
        try:
            transcriber = VoiceTranscriber("base.en")
            transcript = transcriber.transcribe_audio_file(handle.name, language="en")
        finally:
            os.unlink(handle.name)
        self.assertEqual(transcriber.model_name, "base.en")
        self.assertTrue(transcript["success"], transcript)
        response = self.client.post(
            "/api/analyze-voice",
            data={"audio": (audio, "tone.wav")},
            headers=self.headers_a,
            content_type="multipart/form-data",
        )
        body = response.get_json()
        self.assertEqual(response.status_code, 503, body)
        self.assertIn("currently deferred", body["message"])


class RealGroqRewriteTests(unittest.TestCase):
    """A real request to the configured Groq model. No substitute model is called."""

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.user = PostgresApplicationTests._register(f"auditg{SUFFIX}", f"auditg{SUFFIX}@example.com")
        cls.headers = {"Authorization": f"Bearer {cls.user['token']}"}

    @classmethod
    def tearDownClass(cls):
        session = get_session()
        try:
            session.query(User).filter(User.username == f"auditg{SUFFIX}").delete(synchronize_session=False)
            session.commit()
        finally:
            close_session()

    def test_live_rewrite_call_and_original_remains_unsendable(self):
        engine = RewriteEngine()
        self.assertEqual(engine.groq_model, REQUIRED_GROQ_MODEL)
        try:
            suggestion = engine.generate_candidate_rewrite(
                "This whole plan is trash. Let's start over.",
                [{"message_id": 1, "text": "Can we review the plan?", "timestamp": "2026-11-02T00:00:00"}],
            )
        except RewriteServiceError as exc:
            self.fail(str(exc))
        self.assertTrue(suggestion)
        self.assertNotEqual(suggestion.strip().lower(), "this whole plan is trash. let's start over.")

    def test_rewrite_decision_stays_blocked_when_groq_is_unavailable(self):
        response = self.client.post(
            "/api/analyze",
            json={"text": "This whole plan is trash. Let's start over."},
            headers=self.headers,
        )
        body = response.get_json()
        self.assertEqual(response.status_code, 200, body)
        self.assertTrue(body["analysis_complete"])
        self.assertEqual(body["decision"], "REWRITE")
        self.assertFalse(body["can_send"])
        self.assertIsNone(body["suggested_text"])
        self.assertEqual(body["rewrite"]["status"], "unavailable")
        self.assertIn(REQUIRED_GROQ_MODEL, body["rewrite"]["error"])
        session = get_session()
        try:
            message = (
                session.query(Message)
                .filter(Message.user_id == self.user["user_id"])
                .order_by(Message.message_id.desc())
                .first()
            )
            self.assertIsNotNone(message)
            message_id = message.message_id
            conversation_id = message.conversation_id
            original_text = message.original_text
            risk = session.query(RiskResult).filter(RiskResult.message_id == message_id).one()
            self.assertEqual(risk.decision, "REWRITE")
            stored = session.query(RewriteContext).filter(RewriteContext.message_id == message_id).count()
            self.assertEqual(stored, 0)
        finally:
            close_session()
        send = self.client.post(
            "/api/send",
            json={
                "message_id": message_id,
                "conversation_id": conversation_id,
                "text": original_text,
                "user_action": "send",
            },
            headers=self.headers,
        )
        self.assertEqual(send.status_code, 403, send.get_json())


class HealthStatusTests(unittest.TestCase):
    def test_health_separates_local_models_from_groq(self):
        response = app.test_client().get("/api/health")
        body = response.get_json()
        self.assertEqual(response.status_code, 200, body)
        self.assertTrue(body["ready"])
        self.assertTrue(body["local_models_ready"])
        self.assertFalse(body["production_ready"])
        self.assertEqual(body["rewrite"]["model"], REQUIRED_GROQ_MODEL)
        self.assertFalse(body["rewrite"]["available"])
        self.assertEqual(body["slang"]["inference"], "actual")
        self.assertEqual(body["toxicity"]["inference"], "actual")
        self.assertEqual(body["voice"]["inference"], "actual")


if __name__ == "__main__":
    unittest.main()
