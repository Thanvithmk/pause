import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from flask import Flask

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.database import (
    BehaviorProfile,
    Feedback,
    Message,
    RiskResult,
    ToxicityResult,
    close_session,
    get_session,
)
from modules.ril import BehavioralLearning
from tests.fixtures import (
    CURRENT_VECTOR,
    PREVIOUS_VECTOR,
    SAFE_VECTOR,
    TEST_PASSWORD,
    TOXIC_VECTOR,
    init_test_db,
    seed_conversation,
    seed_core_dataset,
    seed_message,
    seed_toxicity,
    seed_user,
)


def _fake_pipeline_result(text, user_id, conversation_id, vector, decision="ALLOW"):
    return {
        "original_text": text,
        "user_id": user_id,
        "conversation_id": conversation_id,
        "toxicity": dict(
            zip(("toxic", "threat", "insult", "severe_toxic", "obscene", "identity_hate"), vector)
        ),
        "decision": decision,
        "final_risk": vector[0],
    }


class IntegrationFlowTests(unittest.TestCase):
    def setUp(self):
        init_test_db()
        self.session = get_session()
        self.data = seed_core_dataset(self.session)
        self.app = Flask(__name__)

    def tearDown(self):
        close_session()

    def test_text_input_decision_storage(self):
        user = self.data["user"]
        conversation = self.data["conversation"]
        message = seed_message(
            self.session,
            conversation,
            user,
            {"original_text": "I am disappointed but let's talk.", "input_type": "text"},
        )
        seed_toxicity(self.session, message, SAFE_VECTOR)
        RiskResult.create(
            self.session,
            message_id=message.message_id,
            pacl=0.08,
            delta_vector_6d=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            escalation_feature=0.0,
            escalation_probability=0.2,
            final_risk=0.12,
            decision="ALLOW",
        )
        stored = (
            self.session.query(RiskResult)
            .filter(RiskResult.message_id == message.message_id)
            .one()
        )
        self.assertEqual(stored.decision, "ALLOW")
        self.assertEqual(message.input_type, "text")

    def test_multiple_messages_update_profile(self):
        from modules.behavior_profile import BehaviorProfileManager

        manager = BehaviorProfileManager()
        user = self.data["user"]
        manager.get_or_create_profile(user.user_id)
        manager.update_profile(user.user_id, dict(zip(
            ("toxic", "threat", "insult", "severe_toxic", "obscene", "identity_hate"),
            SAFE_VECTOR,
        )))
        manager.update_profile(user.user_id, dict(zip(
            ("toxic", "threat", "insult", "severe_toxic", "obscene", "identity_hate"),
            TOXIC_VECTOR,
        )))
        session = get_session()
        profile = (
            session.query(BehaviorProfile)
            .filter(BehaviorProfile.user_id == user.user_id)
            .one()
        )
        self.assertEqual(profile.message_count, 2)
        self.assertGreater(profile.avg_toxic, SAFE_VECTOR[0])
        self.assertLess(profile.avg_toxic, TOXIC_VECTOR[0])

    def test_escalation_across_messages(self):
        with patch("modules.risk_calculation.BehaviorProfileManager"):
            from modules.risk_calculation import RiskCalculator

            calc = RiskCalculator(lambda_const=0.5)
        delta = calc.calculate_delta_x(CURRENT_VECTOR, PREVIOUS_VECTOR)
        cel = calc.calculate_cel(delta, [1, 1, 1, 1, 1, 1])
        self.assertGreater(cel, 0)
        first = self.data["messages"][0]
        second = self.data["messages"][1]
        self.assertLess(first.timestamp, second.timestamp)

    def test_ril_feedback_collection(self):
        with patch("modules.ril.RiskCalculator"):
            learner = BehavioralLearning()
        message = self.data["messages"][2]
        learner.record_intervention_outcome(
            message.message_id, "REWRITE", "accepted", "positive"
        )
        session = get_session()
        feedback = (
            session.query(Feedback)
            .filter(Feedback.message_id == message.message_id)
            .one()
        )
        self.assertEqual(feedback.user_response, "accepted")
        stats = learner.update_ril_statistics(self.data["user"].user_id)
        self.assertEqual(stats["REWRITE_eff"], 1.0)

    def test_behavioral_profile_update_from_history(self):
        from modules.behavior_profile import BehaviorProfileManager

        manager = BehaviorProfileManager()
        user = self.data["user"]
        averages = manager.calculate_averages(user.user_id)
        self.assertEqual(len(averages), 6)
        self.assertGreater(sum(averages), 0)


if __name__ == "__main__":
    unittest.main()
