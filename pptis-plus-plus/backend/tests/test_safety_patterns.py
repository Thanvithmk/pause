import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.decision_engine import DecisionEngine
from modules.risk_calculation import RiskCalculator
from modules.safety_patterns import SafetyPatternAnalyzer
from modules.toxicity_model import ToxicityAnalyzer


def _risk_decision(text):
    analyzer = ToxicityAnalyzer()
    scores = analyzer.analyze(text)
    calc = RiskCalculator(lambda_const=0.5)
    calc.b, calc.w_E, calc.w_P, calc.alpha = 0.0, 1.0, 1.0, 0.5
    pacl = calc.calculate_pacl(scores["toxic"], 0.0)
    p_esc = calc.calculate_escalation_probability(0.0, pacl)
    final_risk = calc.calculate_final_risk(pacl, p_esc)
    with patch("modules.decision_engine.RiskCalculator"), patch.object(
        DecisionEngine, "_init_rewrite_engine", return_value=None
    ):
        engine = DecisionEngine()
    engine.T_allow, engine.T_warn, engine.T_rewrite = 0.30, 0.55, 0.80
    decision = engine.make_decision(final_risk)["decision"]
    return scores, final_risk, decision


class SafetyPatternTests(unittest.TestCase):
    def setUp(self):
        self.layer = SafetyPatternAnalyzer()

    def test_targeted_dead_is_threat(self):
        result = self.layer.score("You're dead. I know where you'll be after class.")
        self.assertGreaterEqual(result["vector"]["threat"], 0.8)
        self.assertFalse(result["playful_idiom"])

    def test_playful_dead_is_not_threat(self):
        result = self.layer.score("Bro, you're dead 💀")
        self.assertTrue(result["playful_idiom"])
        self.assertEqual(result["vector"]["threat"], 0.0)

    def test_killing_me_idiom(self):
        result = self.layer.score("That meme is killing me 😂")
        self.assertTrue(result["playful_idiom"])


class ThreatRegressionTests(unittest.TestCase):
    CASES = (
        ("You're dead. I know where you'll be after class.", "DELAYED-SEND"),
        ("I'm going to find you after class and beat you up.", "DELAYED-SEND"),
        ("You better watch your back. I'm coming for you tonight.", "DELAYED-SEND"),
        ("I know where you live, and you'll regret what you did.", "DELAYED-SEND"),
        ("I'll wait outside your building and teach you a lesson.", "DELAYED-SEND"),
        ("I will hurt you if you come near me again.", "DELAYED-SEND"),
        ("I will gntly stab you", "DELAYED-SEND"),
        ("I will gently cut your head", "DELAYED-SEND"),
        ("Bro, you're dead 💀", "ALLOW"),
        ("You're dead 💀 that was the funniest thing I've seen today.", "ALLOW"),
    )

    def test_threat_and_humor_cases(self):
        for text, expected in self.CASES:
            with self.subTest(text=text):
                scores, risk, decision = _risk_decision(text)
                self.assertEqual(decision, expected, msg=f"{text} R={risk:.3f} tox={scores}")


class PrimaryToxicityTests(unittest.TestCase):
    """Inference: actual unitary/toxic-bert. These checks do not use a lexicon."""

    def test_real_model_returns_six_scores_and_the_formula_band(self):
        from modules.toxicity_model import PRIMARY_MODEL

        samples = (
            "Thanks for helping me with the assignment.",
            "I will hurt you if you come near me again.",
        )
        for text in samples:
            scores, risk, decision = _risk_decision(text)
            self.assertEqual(set(scores), {
                "toxic", "threat", "insult", "severe_toxic", "obscene", "identity_hate",
            })
            self.assertTrue(all(0.0 <= float(value) <= 1.0 for value in scores.values()))
            from modules.toxicity_model import ToxicityAnalyzer

            loaded = ToxicityAnalyzer()
            self.assertEqual(loaded.model_name, PRIMARY_MODEL)
            self.assertEqual(loaded.inference, "actual")
            if risk < 0.30:
                self.assertEqual(decision, "ALLOW")
            elif risk < 0.55:
                self.assertEqual(decision, "WARN")
            elif risk < 0.80:
                self.assertEqual(decision, "REWRITE")
            else:
                self.assertEqual(decision, "DELAYED-SEND")


class ContextualSafetyTests(unittest.TestCase):
    def setUp(self):
        self.layer = SafetyPatternAnalyzer()

    def test_quoted_watch_your_back_is_allow(self):
        result = self.layer.score("My friend was told 'watch your back tonight' by a stranger.")
        self.assertEqual(result["label"], "ALLOW")
        self.assertTrue(result["quoted"])

    def test_game_harm_is_warn_not_delayed(self):
        result = self.layer.score("I will hurt your ranking, not you.")
        self.assertEqual(result["label"], "WARN")

    def test_hostile_prior_escalates_come_over(self):
        result = self.layer.score(
            "Don't make me come over there.",
            context="I have a bat. You took my money.",
        )
        self.assertEqual(result["label"], "DELAYED-SEND")

    def test_model_threat_survives_when_regex_allows(self):
        base = {
            "toxic": 0.89,
            "threat": 0.87,
            "insult": 0.11,
            "severe_toxic": 0.11,
            "obscene": 0.08,
            "identity_hate": 0.05,
        }
        result = self.layer.merge(base, "Please send the notes when you can.")
        self.assertEqual(result["label"], "ALLOW")
        self.assertAlmostEqual(result["vector"]["threat"], 0.87)
        self.assertAlmostEqual(result["vector"]["toxic"], 0.89)

    def test_negated_harm_still_drops_model_scores(self):
        base = {"toxic": 0.9, "threat": 0.9, "insult": 0.0, "severe_toxic": 0.0, "obscene": 0.0, "identity_hate": 0.0}
        result = self.layer.merge(base, "I will not stab you.")
        self.assertEqual(result["label"], "ALLOW")
        self.assertEqual(result["vector"]["threat"], 0.0)
        self.assertEqual(result["vector"]["toxic"], 0.0)

    def test_family_come_over_is_warn(self):
        result = self.layer.score(
            "Don't make me come over there.",
            context="Kids, stop fighting over the remote.",
        )
        self.assertEqual(result["label"], "WARN")

    def test_dehumanizing_identity_is_delayed(self):
        result = self.layer.score("They're not even human if they dress like that.")
        self.assertEqual(result["label"], "DELAYED-SEND")


if __name__ == "__main__":
    unittest.main()
