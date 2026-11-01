import unittest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.risk_calculation import RiskCalculator


class RiskCalculatorPACLTests(unittest.TestCase):
    def setUp(self):
        with patch("modules.risk_calculation.BehaviorProfileManager"):
            self.calculator = RiskCalculator(lambda_const=0.5)

    def test_equal_to_baseline(self):
        pacl = self.calculator.calculate_pacl(0.5, 0.5, lambda_const=0.5)
        self.assertAlmostEqual(pacl, 0.5)

    def test_higher_than_baseline(self):
        # 0.6 + 0.5 * (0.6 - 0.4) = 0.7
        pacl = self.calculator.calculate_pacl(0.6, 0.4, lambda_const=0.5)
        self.assertAlmostEqual(pacl, 0.7)

    def test_lower_than_baseline(self):
        # 0.4 + 0.5 * (0.4 - 0.6) = 0.3
        pacl = self.calculator.calculate_pacl(0.4, 0.6, lambda_const=0.5)
        self.assertAlmostEqual(pacl, 0.3)

    def test_clips_high_to_one(self):
        # 0.8 + 0.5 * (0.8 - 0.2) = 1.1 -> 1.0
        pacl = self.calculator.calculate_pacl(0.8, 0.2, lambda_const=0.5)
        self.assertEqual(pacl, 1.0)

    def test_clips_low_to_zero(self):
        # 0.2 + 0.5 * (0.2 - 0.8) = -0.1 -> 0.0
        pacl = self.calculator.calculate_pacl(0.2, 0.8, lambda_const=0.5)
        self.assertEqual(pacl, 0.0)

    def test_rejects_out_of_range_inputs(self):
        with self.assertRaises(ValueError):
            self.calculator.calculate_pacl(1.2, 0.5)
        with self.assertRaises(ValueError):
            self.calculator.calculate_pacl(0.5, -0.1)

    def test_interpret_within_baseline(self):
        self.assertEqual(self.calculator.interpret_pacl(0.32), "Within baseline")

    def test_interpret_higher_than_baseline(self):
        self.assertEqual(self.calculator.interpret_pacl(0.55), "Higher than baseline")

    def test_interpret_significantly_elevated(self):
        self.assertEqual(self.calculator.interpret_pacl(0.91), "Significantly elevated")

    def test_delta_x_example(self):
        previous = [0.30, 0.20, 0.20, 0.05, 0.10, 0.02]
        current = [0.70, 0.60, 0.50, 0.10, 0.20, 0.05]
        delta = self.calculator.calculate_delta_x(current, previous)
        expected = [0.40, 0.40, 0.30, 0.05, 0.10, 0.03]
        for actual, value in zip(delta, expected):
            self.assertAlmostEqual(actual, value)

    def test_cel_dot_product(self):
        delta = [0.40, 0.40, 0.30, 0.05, 0.10, 0.03]
        escalation = self.calculator.calculate_cel(delta, weight_vector=[1, 1, 1, 1, 1, 1])
        self.assertAlmostEqual(escalation, 1.28)

    def test_cel_deescalation_is_negative(self):
        escalation = self.calculator.calculate_cel([-0.2, 0.0, 0.0, 0.0, 0.0, 0.0], [1, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(escalation, -0.2)

    def test_delta_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            self.calculator.calculate_delta_x([0.1, 0.2], [0.1, 0.2])

    def test_escalation_probability_sigmoid_at_zero(self):
        probability = self.calculator.calculate_escalation_probability(
            0.0, 0.0, b=0.0, w_E=1.0, w_P=1.0
        )
        self.assertAlmostEqual(probability, 0.5)

    def test_escalation_probability_range(self):
        probability = self.calculator.calculate_escalation_probability(
            0.4, 0.6, b=0.0, w_E=1.0, w_P=1.0
        )
        self.assertGreater(probability, 0.5)
        self.assertLessEqual(probability, 1.0)

    def test_interpret_escalation_probability(self):
        self.assertEqual(self.calculator.interpret_escalation_probability(0.2), "Low risk")
        self.assertEqual(self.calculator.interpret_escalation_probability(0.5), "Moderate risk")
        self.assertEqual(
            self.calculator.interpret_escalation_probability(0.8), "High escalation risk"
        )

    def test_final_risk_equal_weight(self):
        risk = self.calculator.calculate_final_risk(0.4, 0.8, alpha=0.5)
        self.assertAlmostEqual(risk, 0.6)

    def test_final_risk_emphasize_current(self):
        risk = self.calculator.calculate_final_risk(0.2, 0.8, alpha=1.0)
        self.assertAlmostEqual(risk, 0.2)

    def test_risk_category_bands(self):
        self.assertEqual(self.calculator.get_risk_category(0.10)[0], "ALLOW")
        self.assertEqual(self.calculator.get_risk_category(0.29)[0], "ALLOW")
        self.assertEqual(self.calculator.get_risk_category(0.30)[0], "WARN")
        self.assertEqual(self.calculator.get_risk_category(0.40)[0], "WARN")
        self.assertEqual(self.calculator.get_risk_category(0.54)[0], "WARN")
        self.assertEqual(self.calculator.get_risk_category(0.55)[0], "REWRITE")
        self.assertEqual(self.calculator.get_risk_category(0.70)[0], "REWRITE")
        self.assertEqual(self.calculator.get_risk_category(0.79)[0], "REWRITE")
        self.assertEqual(self.calculator.get_risk_category(0.80)[0], "DELAYED-SEND")
        self.assertEqual(self.calculator.get_risk_category(1.00)[0], "DELAYED-SEND")
        category, color, description = self.calculator.get_risk_category(0.90)
        self.assertEqual(category, "DELAYED-SEND")
        self.assertTrue(color.startswith("#"))
        self.assertIn("delay", description.lower())
