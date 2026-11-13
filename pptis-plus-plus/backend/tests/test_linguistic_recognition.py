import threading
import time
import unittest

from modules.emoji_recognition import EmojiRecognizer
from modules.linguistic_features import annotate_slang_usage, recognize_in_parallel
from modules.slang_recognition import SlangRecognizer


class SlangRecognitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recognizer = SlangRecognizer()

    def test_no_cap_is_one_expression(self):
        result = self.recognizer.recognize("no cap that was great")
        terms = [item["term"].lower() for item in result["terms"]]
        self.assertIn("no cap", terms)
        self.assertNotIn("cap", terms)

    def test_slay_keeps_both_meanings(self):
        result = self.recognizer.recognize("You slay")
        matches = [item for item in result["terms"] if item["term"].lower() == "slay"]
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0]["ambiguous"])
        definitions = {sense["definition"] for sense in matches[0]["senses"]}
        self.assertTrue({"do extremely well", "succeed stylishly"} <= definitions)

    def test_longer_phrase_wins_over_its_words(self):
        result = self.recognizer.recognize("cooked beyond repair")
        terms = [item["term"].lower() for item in result["terms"]]
        self.assertEqual(terms, ["cooked beyond repair"])

    def test_lookup_does_not_assign_a_decision(self):
        result = self.recognizer.recognize("no cap")
        self.assertNotIn("decision", result)
        self.assertNotIn("usage", result["terms"][0])


class EmojiRecognitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recognizer = EmojiRecognizer()

    def test_fire_keeps_its_character_and_description(self):
        message = "Bro cooked 🔥"
        result = self.recognizer.recognize(message)
        self.assertEqual(result["emojis"][0]["emoji"], "🔥")
        self.assertIn("fire", result["emojis"][0]["description"].lower())
        self.assertNotIn("text", result)
        self.assertIn("🔥", message)

    def test_adjacent_emoji_is_a_combination(self):
        result = self.recognizer.recognize("That's hilarious 😂💀")
        self.assertEqual([item["emoji"] for item in result["emojis"]], ["😂", "💀"])
        self.assertEqual(result["combinations"][0]["members"], ["😂", "💀"])
        self.assertEqual(result["combinations"][0]["emoji"], "😂💀")

    def test_heart_with_variation_selector_is_one_emoji(self):
        result = self.recognizer.recognize("I hate you ❤️")
        self.assertEqual(len(result["emojis"]), 1)
        self.assertEqual(result["emojis"][0]["emoji"], "❤️")
        self.assertTrue(result["emojis"][0]["known"])
        self.assertEqual(result["combinations"], [])

    def test_unknown_emoji_is_kept_without_a_description(self):
        result = self.recognizer.recognize("See you \U0001FAEA")
        self.assertEqual(result["emojis"][0]["emoji"], "\U0001FAEA")
        self.assertFalse(result["emojis"][0]["known"])
        self.assertEqual(result["emojis"][0]["description"], "")


class ParallelRecognitionTests(unittest.TestCase):
    def test_lookups_overlap(self):
        state = {"open": 0, "max": 0}
        lock = threading.Lock()

        class Slow:
            def __init__(self, payload):
                self.payload = payload

            def recognize(self, text):
                with lock:
                    state["open"] += 1
                    state["max"] = max(state["max"], state["open"])
                time.sleep(0.05)
                with lock:
                    state["open"] -= 1
                return self.payload

        recognize_in_parallel("hello", Slow({"terms": []}), Slow({"emojis": []}))
        self.assertEqual(state["max"], 2)

    def test_head_label_is_copied_beside_the_meanings(self):
        features = {
            "terms": [
                {"term": "cooked", "senses": [{"definition": "defeated/finished/in serious trouble"}]},
            ]
        }
        annotate_slang_usage(features, [{"term": "cooked", "usage": "slang", "confidence": 0.91}])
        self.assertEqual(features["terms"][0]["usage"], "slang")
        self.assertEqual(features["terms"][0]["confidence"], 0.91)
        self.assertNotIn("decision", features["terms"][0])

    def test_laughter_emoji_is_playful_and_a_weapon_emoji_is_not(self):
        from modules.linguistic_features import has_playful_laughter

        laughing = {"emojis": [{"emoji": "😂", "tags": ["laughing", "humor"]}]}
        armed = {"emojis": [{"emoji": "😂", "tags": ["laughing"]}, {"emoji": "🔪", "tags": ["object"]}]}
        heart = {"emojis": [{"emoji": "❤️", "tags": ["affection"]}]}
        self.assertTrue(has_playful_laughter(laughing))
        self.assertFalse(has_playful_laughter(armed))
        self.assertFalse(has_playful_laughter(heart))

    def test_emoji_note_keeps_the_original_character(self):
        from modules.linguistic_features import emoji_annotations

        notes = emoji_annotations(
            {"emojis": [{"emoji": "😂", "description": "FACE WITH TEARS OF JOY"}]}
        )
        self.assertEqual(notes, ["[Emoji meaning: 😂 FACE WITH TEARS OF JOY]"])


if __name__ == "__main__":
    unittest.main()
