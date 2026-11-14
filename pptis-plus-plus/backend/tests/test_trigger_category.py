import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["FLASK_ENV"] = "testing"

from modules.trigger_category import _category_name, _parse_reply, categorize_many


class TriggerCategoryTests(unittest.TestCase):
    def test_numbered_reply_maps_topics(self):
        parsed = _parse_reply("1. Community\n2. Insult\n3. Harm", 3)
        self.assertEqual(parsed, {1: "Community", 2: "Insult", 3: "Harm"})

    def test_self_harm_is_not_read_as_harm(self):
        self.assertEqual(_category_name("Self-harm"), "Self-harm")

    def test_intervention_name_is_not_a_topic(self):
        self.assertIsNone(_category_name("DELAYED-SEND"))
        self.assertIsNone(_category_name("WARN"))

    def test_testing_does_not_call_groq(self):
        self.assertEqual(categorize_many([("current", "Keep talking and you'll regret it.")]), {})


if __name__ == "__main__":
    unittest.main()
