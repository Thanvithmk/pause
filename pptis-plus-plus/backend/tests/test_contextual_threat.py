import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.contextual_threat import assess
from modules.decision_engine import DecisionEngine
from modules.safety_patterns import SafetyPatternAnalyzer


def _new_user_risk(toxic):
    pacl = min(1.0, max(0.0, toxic + 0.5 * toxic))
    p_esc = 1.0 / (1.0 + math.exp(-pacl))
    return 0.5 * pacl + 0.5 * p_esc


class ContextualReadingTests(unittest.TestCase):
    def test_grooming_and_chores_are_harmless_physical_actions(self):
        samples = (
            "I will gently cut your nails.",
            "I will gently cut your nails",
            "I will cut your hair.",
            "I will trim your nails.",
            "I will trim your beard.",
            "I will cut the vegetables.",
            "I will cut the rope.",
            "I will clean your nails.",
            "The nurse will clean the wound.",
            "Apply medicine to the cut.",
            "The doctor will give you an injection.",
        )
        for text in samples:
            with self.subTest(text=text):
                reading = assess(text)
                self.assertEqual(reading["category"], "harmless_physical")
                self.assertEqual(reading["effect"], "clear")
                self.assertEqual(reading["floor"], "ALLOW")

    def test_violent_objects_and_directed_harm_stay_explicit(self):
        samples = (
            "I will cut your head.",
            "I will gently cut your head.",
            "I will slash your throat.",
            "I will hurt you.",
            "I will hurt you after class.",
            "I will attack you.",
            "I will stab you.",
            "I will kill you.",
            "I will cut your nails and then I will cut your head.",
        )
        for text in samples:
            with self.subTest(text=text):
                reading = assess(text)
                self.assertEqual(reading["category"], "explicit_threat")
                self.assertEqual(reading["effect"], "hold")

    def test_ambiguous_idioms_are_not_cleared(self):
        for text in (
            "I will cut you.",
            "I will deal with you.",
            "You will regret this.",
            "I will finish you.",
            "I will make you pay.",
            "I will end you.",
        ):
            with self.subTest(text=text):
                reading = assess(text)
                self.assertEqual(reading["category"], "ambiguous")
                self.assertEqual(reading["effect"], "hold")

    def test_figurative_expressions_are_not_threats(self):
        for text in (
            "That joke killed me.",
            "This exam destroyed me.",
            "I am dead tired.",
            "I will kill this presentation.",
            "This deadline is killing me.",
        ):
            with self.subTest(text=text):
                reading = assess(text)
                self.assertEqual(reading["category"], "figurative")
                self.assertEqual(reading["effect"], "clear")

    def test_an_insult_keeps_a_rewrite_floor(self):
        reading = assess("I will cut your nails, you idiot.")
        self.assertEqual(reading["category"], "harmless_physical")
        self.assertEqual(reading["effect"], "clear")
        self.assertEqual(reading["floor"], "REWRITE")

    def test_safety_label_follows_the_object(self):
        layer = SafetyPatternAnalyzer()
        self.assertEqual(layer.score("I will gently cut your nails")["label"], "ALLOW")
        self.assertEqual(layer.score("I will cut your hair")["label"], "ALLOW")
        self.assertEqual(layer.score("I will cut your head")["label"], "DELAYED-SEND")
        self.assertEqual(layer.score("I will cut you")["label"], "DELAYED-SEND")
        self.assertEqual(layer.score("I will kill this presentation")["label"], "ALLOW")
        self.assertEqual(layer.score("I will hurt you after class")["label"], "DELAYED-SEND")


class ContextualThreatModelTests(unittest.TestCase):
    """The model scores stay in place. Only the final intervention can change."""

    @classmethod
    def setUpClass(cls):
        from modules.toxicity_model import ToxicityAnalyzer

        cls.analyzer = ToxicityAnalyzer()
        cls.safety = SafetyPatternAnalyzer()
        with patch("modules.decision_engine.RiskCalculator"), patch.object(
            DecisionEngine, "_init_rewrite_engine", return_value=None
        ):
            cls.engine = DecisionEngine()
        cls.engine.T_allow = 0.30
        cls.engine.T_warn = 0.55
        cls.engine.T_rewrite = 0.80

    def _decide(self, text):
        detail = self.analyzer.inspect(text)
        scores = dict(detail["model"])
        reading = assess(text)
        risk = _new_user_risk(scores["toxic"])
        result = self.engine.make_decision(
            risk,
            toxicity_vector=scores,
            safety_flags={
                "quoted": detail["quoted"],
                "negated": detail["negated"],
                "playful_idiom": detail["playful_idiom"],
                "contextual_threat": reading,
            },
            safety_label=self.safety.score(text)["label"],
        )
        return scores, risk, result

    def test_contrast_sentences_keep_model_scores_and_change_only_the_decision(self):
        expected = {
            "I will gently cut your nails.": "ALLOW",
            "I will gently cut your nails": "ALLOW",
            "I will cut your hair.": "ALLOW",
            "I will cut your head.": "DELAYED-SEND",
            "I will cut you.": "DELAYED-SEND",
            "I will cut the vegetables.": "ALLOW",
            "I will trim your nails.": "ALLOW",
            "I will hurt you after class.": "DELAYED-SEND",
        }
        for text, decision in expected.items():
            with self.subTest(text=text):
                scores, risk, result = self._decide(text)
                self.assertEqual(result["decision"], decision)
                self.assertAlmostEqual(result["risk_score"], risk, places=4)
                self.assertGreater(scores["toxic"], 0.0)
                if "head" in text or text.startswith("I will hurt") or text == "I will cut you.":
                    self.assertGreater(scores["threat"], 0.4)
                    self.assertNotIn("contextual_threat", result["guardrail"] or "")
                if "nails" in text and "trim" not in text and "head" not in text:
                    self.assertGreater(scores["toxic"], 0.5)
                    self.assertIn("contextual_threat:harmless_physical", result["guardrail"])


if __name__ == "__main__":
    unittest.main()
