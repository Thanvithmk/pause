"""Primary-model checks.

Slang, toxicity, and Whisper tests below run the real checkpoints.
The Groq tests mock the HTTP client and are marked inference=\"mocked\".
The live Groq call is backend/checks/groq_live_check.py and is not part of this file.
"""

import os
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("FLASK_ENV", "testing")


def _tone_wav(path):
    rate = 16000
    frames = []
    for index in range(rate):
        sample = int(8000 * __import__("math").sin(2 * 3.14159265 * 220 * index / rate))
        frames.append(struct.pack("<h", sample))
    with wave.open(path, "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"".join(frames))


class ActualToxicityTests(unittest.TestCase):
    inference = "actual"

    def test_toxic_bert_six_dimensions(self):
        from modules.toxicity_model import PRIMARY_MODEL, TOXICITY_LABELS, ToxicityAnalyzer

        analyzer = ToxicityAnalyzer()
        scores = analyzer.analyze("You are stupid.")
        self.assertEqual(self.inference, "actual")
        self.assertEqual(analyzer.inference, "actual")
        self.assertEqual(analyzer.model_name, PRIMARY_MODEL)
        self.assertEqual(set(scores), set(TOXICITY_LABELS))
        self.assertGreater(scores["toxic"], scores["identity_hate"])
        safe = analyzer.analyze("Thanks for reviewing the notes.")
        self.assertLess(safe["toxic"], scores["toxic"])


class ActualWhisperTests(unittest.TestCase):
    inference = "actual"

    def test_base_en_transcribes_a_wav(self):
        from modules.speech_to_text import REQUIRED_MODEL, VoiceTranscriber

        transcriber = VoiceTranscriber(REQUIRED_MODEL)
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        handle.close()
        try:
            _tone_wav(handle.name)
            result = transcriber.transcribe_audio_file(handle.name, language="en")
        finally:
            os.unlink(handle.name)
        self.assertEqual(self.inference, "actual")
        self.assertEqual(transcriber.model_name, "base.en")
        self.assertTrue(result["success"], result)
        self.assertIn(transcriber.get_model_info()["device"], ("CPU", "CUDA"))

    def test_smaller_whisper_model_is_rejected(self):
        from modules.speech_to_text import VoiceModelError, VoiceTranscriber

        with self.assertRaises(VoiceModelError):
            VoiceTranscriber("tiny.en")


class MockedGroqRewriteTests(unittest.TestCase):
    inference = "mocked"

    def _engine(self):
        from modules.rewrite_engine import RewriteEngine

        engine = RewriteEngine.__new__(RewriteEngine)
        engine.groq_model = "llama-3.1-8b-instant"
        engine.inference = "mocked"
        engine.groq_client = MagicMock()
        engine._groq_key = "test-key"
        return engine

    def test_mocked_completion_is_the_suggestion(self):
        engine = self._engine()
        engine._complete = lambda prompt: "I want to talk about this differently."
        text = engine.generate_candidate_rewrite("You are awful.", [])
        self.assertEqual(self.inference, "mocked")
        self.assertEqual(text, "I want to talk about this differently.")
        self.assertNotIn("I want to be honest, but more calmly", text)

    def test_api_error_is_not_replaced(self):
        from modules.rewrite_engine import RewriteServiceError

        engine = self._engine()

        def explode(prompt):
            raise RewriteServiceError("rewrite: Groq request failed for llama-3.1-8b-instant")

        engine._complete = explode
        with self.assertRaises(RewriteServiceError) as caught:
            engine.generate_candidate_rewrite("You are awful.", [])
        self.assertIn("llama-3.1-8b-instant", str(caught.exception))
        self.assertNotIn("template", str(caught.exception).lower())


if __name__ == "__main__":
    unittest.main()
