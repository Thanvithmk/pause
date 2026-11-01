import sys
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from flask import Flask

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.auth import AuthError, decode_token, login_user, register_user
from modules.database import Conversation, Message, User, close_session, get_session
from modules.decision_engine import DecisionEngine
from modules.delayed_send import DelayedSendManager
from modules.preprocessing import Preprocessor
from modules.risk_calculation import RiskCalculator
from modules.slang_layer import SlangInterpreter, SlangModelError
from tests.fixtures import (
    CURRENT_VECTOR,
    DELTA_VECTOR,
    PREVIOUS_VECTOR,
    SAFE_VECTOR,
    SLANG_TEXT,
    TEST_PASSWORD,
    TOXIC_VECTOR,
    init_test_db,
    seed_conversation,
    seed_core_dataset,
    seed_message,
    seed_user,
)


class AuthModuleTests(unittest.TestCase):
    def setUp(self):
        init_test_db()
        self.app = Flask(__name__)

    def tearDown(self):
        close_session()

    def test_register_login_and_jwt(self):
        registered = register_user("alexsafe", "alex@example.com", TEST_PASSWORD)
        self.assertIn("user_id", registered)
        payload = decode_token(registered["token"], expected_type="access")
        self.assertEqual(str(registered["user_id"]), payload["sub"])
        with self.app.test_request_context():
            logged_in = login_user("alex@example.com", TEST_PASSWORD)
        self.assertEqual(logged_in["user_id"], registered["user_id"])
        self.assertTrue(logged_in["token"])

    def test_register_rejects_weak_password(self):
        with self.assertRaises(AuthError):
            register_user("baduser", "bad@example.com", "weak")


class PreprocessingModuleTests(unittest.TestCase):
    def setUp(self):
        self.preprocessor = Preprocessor()

    def test_preserves_emoji_and_contractions(self):
        result = self.preprocessor.preprocess("don't go 😀")
        self.assertIn("don't", result["cleaned_text"])
        self.assertTrue(result["emoji_list"])
        self.assertIn("don't", result["preserved_info"]["contractions_found"])


class SlangModuleTests(unittest.TestCase):
    def setUp(self):
        self.interpreter = SlangInterpreter.__new__(SlangInterpreter)
        self.interpreter.slang_dictionary = {
            "cooked": {
                "term": "cooked",
                "meaning": "defeated/finished/in serious trouble",
                "category": "gen-z",
                "confidence": 0.95,
            }
        }
        self.interpreter._terms_by_length = ("cooked",)
        self.interpreter.threshold = 0.7

    def test_detects_cooked_position(self):
        candidates = self.interpreter.detect_candidates(SLANG_TEXT)
        self.assertEqual(candidates[0]["term"], "cooked")
        self.assertEqual(candidates[0]["pos"], 11)

    def test_missing_encoder_does_not_invent_probabilities(self):
        """This case does not run model inference. It checks that a missing encoder raises."""
        self.interpreter.encoder = None
        self.interpreter._use_trained_head = False
        with self.assertRaises(SlangModelError):
            self.interpreter.disambiguate_slang_literal("cooked", "Bro you're cooked 💀")

    def test_interpret_appends_meaning_without_replacing_text(self):
        self.interpreter.disambiguate_slang_literal = lambda term, context: {
            "term": term,
            "p_slang": 0.8,
            "p_literal": 0.2,
            "classification": "slang" if "chicken" not in context else "literal",
            "threshold": 0.58,
            "source": "scripted",
        }
        slang = self.interpreter.interpret("Bro you're cooked 💀")
        self.assertTrue(slang["enriched_text_with_context"].startswith("Bro you're cooked 💀"))
        self.assertIn(
            "[Contextual meaning: defeated / finished / in serious trouble]",
            slang["enriched_text_with_context"],
        )
        self.assertNotIn("Bro you're defeated", slang["enriched_text_with_context"])
        self.assertEqual(slang["detections"][0]["term"], "cooked")
        self.assertEqual(slang["detections"][0]["usage"], "slang")
        self.assertIn("trouble", slang["detections"][0]["meaning"])
        literal = self.interpreter.interpret("The chicken is cooked.")
        self.assertEqual(literal["enriched_text_with_context"], "The chicken is cooked.")
        self.assertEqual(literal["detections"][0]["usage"], "literal")
        self.assertIsNone(literal["detections"][0]["meaning"])
        self.assertEqual(slang["detections"][0]["source"], "scripted")

    def test_enrich_preserves_original(self):
        self.interpreter.disambiguate_slang_literal = lambda term, context: {
            "term": term,
            "p_slang": 0.94,
            "p_literal": 0.06,
            "classification": "slang",
            "threshold": 0.7,
        }
        result = self.interpreter.enrich_with_context(SLANG_TEXT, [{"term": "cooked", "pos": 12}])
        self.assertEqual(result["original_text"], SLANG_TEXT)
        self.assertIn("[Contextual meaning:", result["enriched_text_with_context"])
        self.assertTrue(result["enriched_text_with_context"].startswith(SLANG_TEXT))

    def test_unknown_glorped_is_not_treated_as_toxic_slang(self):
        result = self.interpreter.interpret("Bro is completely glorped.")
        self.assertEqual(result["enriched_text_with_context"], "Bro is completely glorped.")
        unknown_terms = [item["term"].lower() for item in result["unknown_slang"]]
        self.assertIn("glorped", unknown_terms)
        self.assertNotIn("completely", unknown_terms)
        self.assertTrue(all(item.get("label") == "UNKNOWN_SLANG" for item in result["unknown_slang"]))
        self.assertTrue(all(item.get("stored") is False for item in result["unknown_slang"]))

    def test_missing_weights_do_not_invent_probabilities(self):
        """This case does not run model inference."""
        self.interpreter.encoder = None
        self.interpreter.classifier = None
        self.interpreter._use_trained_head = False
        with self.assertRaises(SlangModelError):
            self.interpreter.disambiguate_slang_literal("cooked", "Bro you're cooked 💀")


class ToxicityModuleTests(unittest.TestCase):
    def test_known_toxic_vs_safe_vectors(self):
        from modules.toxicity_model import ToxicityAnalyzer

        analyzer = ToxicityAnalyzer.__new__(ToxicityAnalyzer)
        analyzer.analyze = lambda text: (
            dict(zip(("toxic", "threat", "insult", "severe_toxic", "obscene", "identity_hate"), TOXIC_VECTOR))
            if "hate" in text
            else dict(zip(("toxic", "threat", "insult", "severe_toxic", "obscene", "identity_hate"), SAFE_VECTOR))
        )
        toxic = analyzer.analyze("You are stupid and I hate you.")
        safe = analyzer.analyze("Thanks for waiting.")
        self.assertGreater(toxic["toxic"], 0.8)
        self.assertLess(safe["toxic"], 0.2)


class RiskModuleTests(unittest.TestCase):
    def setUp(self):
        with patch("modules.risk_calculation.BehaviorProfileManager"):
            self.calc = RiskCalculator(lambda_const=0.5)
        self.calc.alpha = 0.5
        self.calc.b = 0.0
        self.calc.w_E = 1.0
        self.calc.w_P = 1.0

    def test_pacl_known_baseline(self):
        self.assertAlmostEqual(self.calc.calculate_pacl(0.6, 0.4, 0.5), 0.7)

    def test_cel_delta_vector(self):
        delta = self.calc.calculate_delta_x(CURRENT_VECTOR, PREVIOUS_VECTOR)
        for actual, expected in zip(delta, DELTA_VECTOR):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(self.calc.calculate_cel(delta, [1, 1, 1, 1, 1, 1]), 1.28)

    def test_escalation_probability(self):
        probability = self.calc.calculate_escalation_probability(0.0, 0.0, b=0.0, w_E=1.0, w_P=1.0)
        self.assertAlmostEqual(probability, 0.5)

    def test_final_risk(self):
        self.assertAlmostEqual(self.calc.calculate_final_risk(0.4, 0.8, alpha=0.5), 0.6)


class DecisionEngineTests(unittest.TestCase):
    def setUp(self):
        with patch("modules.decision_engine.RiskCalculator"), patch.object(
            DecisionEngine, "_init_rewrite_engine", return_value=None
        ):
            self.engine = DecisionEngine()
        self.engine.T_allow = 0.30
        self.engine.T_warn = 0.55
        self.engine.T_rewrite = 0.80

    def test_all_thresholds(self):
        self.assertEqual(self.engine.make_decision(0.10)["decision"], "ALLOW")
        self.assertEqual(self.engine.make_decision(0.40)["decision"], "WARN")
        self.assertEqual(self.engine.make_decision(0.70)["decision"], "REWRITE")
        self.assertEqual(self.engine.make_decision(0.90)["decision"], "DELAYED-SEND")

    def test_boundary_thresholds(self):
        self.assertEqual(self.engine.make_decision(0.29)["decision"], "ALLOW")
        self.assertEqual(self.engine.make_decision(0.30)["decision"], "WARN")
        self.assertEqual(self.engine.make_decision(0.54)["decision"], "WARN")
        self.assertEqual(self.engine.make_decision(0.55)["decision"], "REWRITE")
        self.assertEqual(self.engine.make_decision(0.79)["decision"], "REWRITE")
        self.assertEqual(self.engine.make_decision(0.80)["decision"], "DELAYED-SEND")
        self.assertEqual(self.engine.make_decision(1.00)["decision"], "DELAYED-SEND")

    def test_rejects_non_finite_risk(self):
        with self.assertRaises(ValueError):
            self.engine.make_decision(float("nan"))
        with self.assertRaises(ValueError):
            self.engine.make_decision(float("inf"))
        with self.assertRaises(ValueError):
            self.engine.make_decision(-0.1)
        with self.assertRaises(ValueError):
            self.engine.make_decision(1.01)
        with self.assertRaises(ValueError):
            self.engine.make_decision(None)


class RewriteEngineTests(unittest.TestCase):
    def test_validate_rewrite_is_safer(self):
        from modules.rewrite_engine import RewriteEngine

        with patch("modules.risk_calculation.BehaviorProfileManager"):
            calc = RiskCalculator(lambda_const=0.5)
        calc.b, calc.w_E, calc.w_P, calc.alpha = 0.0, 1.0, 1.0, 0.5
        engine = RewriteEngine.__new__(RewriteEngine)
        engine.risk_calculator = calc
        engine.toxicity_analyzer = MagicMock()
        engine.toxicity_analyzer.analyze.return_value = dict(
            zip(("toxic", "threat", "insult", "severe_toxic", "obscene", "identity_hate"), SAFE_VECTOR)
        )
        result = engine.validate_rewrite("Thanks for your time.", TOXIC_VECTOR)
        self.assertTrue(result["is_safer"])
        self.assertLess(result["new_risk"], result["original_risk"])


class DelayedSendTests(unittest.TestCase):
    def setUp(self):
        init_test_db()
        self.session = get_session()
        self.user = seed_user(self.session)
        self.conversation = seed_conversation(self.session, self.user)
        self.message = seed_message(
            self.session,
            self.conversation,
            self.user,
            {"original_text": "angry text", "input_type": "text"},
        )
        with patch("modules.delayed_send.RiskCalculator"), patch.object(
            DelayedSendManager, "_init_toxicity_analyzer", return_value=None
        ), patch.object(DelayedSendManager, "_init_profile_manager", return_value=None):
            self.manager = DelayedSendManager(delay_duration=60)
        self.manager.delay_duration = 60

    def tearDown(self):
        close_session()

    def test_lock_blocks_then_requires_reanalysis(self):
        lock = self.manager.create_delayed_send_state(
            self.message.message_id, self.user.user_id, "angry text", 0.9
        )
        blocked = self.manager.attempt_send(self.message.message_id, self.user.user_id)
        self.assertFalse(blocked["allowed"])
        self.assertEqual(blocked["next_action"], "wait")
        session = get_session()
        from modules.delayed_send import DelayedSendLock

        row = session.get(DelayedSendLock, lock["lock_id"])
        row.expires_at = row.expires_at - timedelta(seconds=120)
        session.commit()
        close_session()
        expired = self.manager.attempt_send(self.message.message_id, self.user.user_id)
        self.assertFalse(expired["allowed"])
        self.assertEqual(expired["next_action"], "reanalyze")
        self.assertFalse(expired["allow_send_anyway"])


class DatabaseCrudTests(unittest.TestCase):
    def setUp(self):
        init_test_db()
        self.session = get_session()

    def tearDown(self):
        close_session()

    def test_user_conversation_message_crud(self):
        user = seed_user(self.session)
        fetched = User.get_by_id(self.session, user.user_id)
        self.assertEqual(fetched.email, "alex@example.com")
        conversation = seed_conversation(self.session, user)
        self.assertEqual(conversation.user_id, user.user_id)
        message = seed_message(
            self.session,
            conversation,
            user,
            {"original_text": "hello", "input_type": "text"},
        )
        message.update(self.session, processed_text="hello")
        self.assertEqual(Message.get_by_id(self.session, message.message_id).processed_text, "hello")
        conversation.delete(self.session)
        self.assertIsNone(Conversation.get_by_id(self.session, conversation.conversation_id))


if __name__ == "__main__":
    unittest.main()
