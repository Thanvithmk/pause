import unittest

from modules.toxicity_model import attribute_current_scores, compose_model_input


class AnalysisContextTests(unittest.TestCase):
    def test_keeps_the_five_newest_messages_before_the_current_one(self):
        prior = [f"note {index}" for index in range(1, 7)]
        text, truncated = compose_model_input("current", prior, len, 1000)
        self.assertEqual(text, "note 2\nnote 3\nnote 4\nnote 5\nnote 6\ncurrent")
        self.assertFalse(truncated)

    def test_drops_older_context_when_the_window_is_full(self):
        def token_length(value):
            return 100 if "old" in value else len(value)

        text, truncated = compose_model_input("now", ["old", "new"], token_length, 10)
        self.assertEqual(text, "new\nnow")
        self.assertTrue(truncated)

    def test_current_message_remains_when_context_does_not_fit(self):
        def token_length(value):
            return 100 if "huge" in value else 1

        text, truncated = compose_model_input("current", ["huge"], token_length, 10)
        self.assertEqual(text, "current")
        self.assertTrue(truncated)

    def test_prior_toxicity_does_not_replace_the_current_sentence(self):
        mild = _scores(toxic=0.02, insult=0.01)
        window = _scores(toxic=0.71, insult=0.34)
        prior = _scores(toxic=0.68, insult=0.33)
        scored = attribute_current_scores(mild, window, prior)
        self.assertAlmostEqual(scored["toxic"], 0.03)
        self.assertAlmostEqual(scored["insult"], 0.01)

    def test_current_sentence_keeps_the_harm_it_adds(self):
        current = _scores(toxic=0.10, threat=0.04)
        window = _scores(toxic=0.80, threat=0.70)
        prior = _scores(toxic=0.20, threat=0.05)
        scored = attribute_current_scores(current, window, prior)
        self.assertAlmostEqual(scored["toxic"], 0.60)
        self.assertAlmostEqual(scored["threat"], 0.65)

    def test_current_sentence_keeps_its_own_score_when_the_window_is_saturated(self):
        current = _scores(toxic=0.74)
        window = _scores(toxic=0.76)
        prior = _scores(toxic=0.75)
        scored = attribute_current_scores(current, window, prior)
        self.assertAlmostEqual(scored["toxic"], 0.74)


def _scores(**overrides):
    base = {
        "toxic": 0.0,
        "threat": 0.0,
        "insult": 0.0,
        "severe_toxic": 0.0,
        "obscene": 0.0,
        "identity_hate": 0.0,
    }
    base.update(overrides)
    return base


if __name__ == "__main__":
    unittest.main()
