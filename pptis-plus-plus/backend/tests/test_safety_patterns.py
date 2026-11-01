import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PPTIS_LIGHT_MODE", "true")

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
        ("Bro, you're dead 💀", "ALLOW"),
        ("You're dead 💀 that was the funniest thing I've seen today.", "ALLOW"),
    )

    def test_threat_and_humor_cases(self):
        for text, expected in self.CASES:
            with self.subTest(text=text):
                scores, risk, decision = _risk_decision(text)
                self.assertEqual(decision, expected, msg=f"{text} R={risk:.3f} tox={scores}")


SHEET_CASES = (
    ("T001", "Hey! Are we still meeting at 4 PM?", "ALLOW"),
    ("T002", "Thanks for helping me with the assignment. I appreciate it.", "ALLOW"),
    ("T003", "I see your point. Let me think about it and get back to you.", "ALLOW"),
    ("T004", "Could you please send me the notes from today's class?", "ALLOW"),
    ("T005", "That was a really good presentation. Well done!", "ALLOW"),
    ("T006", "I cannot attend today. Can we reschedule for tomorrow?", "ALLOW"),
    ("T007", "I disagree with this decision, but I understand your reasoning.", "ALLOW"),
    ("T008", "Please check the attached file and let me know if anything is missing.", "ALLOW"),
    ("T009", "Bro, that movie was fire 🔥", "ALLOW"),
    ("T010", "That meme is killing me 😂", "ALLOW"),
    ("T011", "You're cooked 💀", "ALLOW"),
    ("T012", "This outfit is sick!", "ALLOW"),
    ("T013", "Your explanation is confusing. Could you explain it another way?", "WARN"),
    ("T014", "You never listen to what I'm saying.", "WARN"),
    ("T015", "I'm really annoyed that you changed the plan without telling me.", "WARN"),
    ("T016", "This is a terrible idea, and we should reconsider it.", "WARN"),
    ("T017", "Stop interrupting me. I want to finish what I'm saying.", "WARN"),
    ("T018", "You completely ignored my message again.", "WARN"),
    ("T019", "I'm done dealing with this nonsense. We need to talk.", "WARN"),
    ("T020", "That answer was stupid. Please check it again.", "WARN"),
    ("T021", "You're being so extra right now 🙄", "WARN"),
    ("T022", "This whole plan is trash. Let's start over.", "WARN"),
    ("T023", "I'm disappointed in how you handled this.", "WARN"),
    ("T024", "Seriously? That was such a dumb move.", "WARN"),
    ("T025", "You're completely useless and nobody wants to work with you.", "REWRITE"),
    ("T026", "You are an idiot. I can't believe I trusted you.", "REWRITE"),
    ("T027", "Shut up and stop acting like you know everything.", "REWRITE"),
    ("T028", "You always ruin everything for everyone.", "REWRITE"),
    ("T029", "Your work is pathetic. Did you even try?", "REWRITE"),
    ("T030", "Nobody takes you seriously because you're so incompetent.", "REWRITE"),
    ("T031", "I hate working with you. You make every task miserable.", "REWRITE"),
    ("T032", "What is wrong with you? Are you incapable of doing one simple thing?", "REWRITE"),
    ("T033", "You're such a clown 🤡. This is embarrassing.", "REWRITE"),
    ("T034", "That presentation was cringe and you looked clueless.", "REWRITE"),
    ("T035", "You are a disgusting person and I never want to see you again.", "REWRITE"),
    ("T036", "Everyone in the group thinks you're worthless.", "REWRITE"),
    ("T037", "I'm going to find you after class and beat you up.", "DELAYED-SEND"),
    ("T038", "I will hurt you if you come near me again.", "DELAYED-SEND"),
    ("T039", "You better watch your back. I'm coming for you tonight.", "DELAYED-SEND"),
    ("T040", "I know where you live, and you'll regret what you did.", "DELAYED-SEND"),
    ("T041", "I am going to smash your face in tomorrow.", "DELAYED-SEND"),
    ("T042", "Bring your friends. We'll make sure you never walk again.", "DELAYED-SEND"),
    ("T043", "People like you should be attacked because of your religion.", "DELAYED-SEND"),
    ("T044", "Your entire community deserves to be wiped out.", "DELAYED-SEND"),
    ("T045", "If you report me, I'll make you pay. I know how to reach you.", "DELAYED-SEND"),
    ("T046", "I have a weapon and I'm coming to your house to use it.", "DELAYED-SEND"),
    ("T047", "I'll wait outside your building and teach you a lesson.", "DELAYED-SEND"),
    ("T048", "You deserve to get seriously injured for what you said.", "DELAYED-SEND"),
    ("T049", "Bro, you're dead 💀", "ALLOW"),
    ("T050", "You're dead. I know where you'll be after class.", "DELAYED-SEND"),
)


class SpreadsheetBenchmarkTests(unittest.TestCase):
    def test_all_fifty_sheet_cases(self):
        rows = []
        failures = []
        for test_id, text, expected in SHEET_CASES:
            scores, risk, actual = _risk_decision(text)
            passed = actual == expected
            rows.append((test_id, expected, actual, "Pass" if passed else "Fail", risk, scores["threat"]))
            if not passed:
                failures.append(f"{test_id} expected {expected} got {actual} R={risk:.3f} {scores}")
        if failures:
            table = "\n".join(f"{row[0]} {row[1]} {row[2]} {row[3]}" for row in rows)
            self.fail("Sheet mismatches:\n" + "\n".join(failures) + "\n\n" + table)


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
