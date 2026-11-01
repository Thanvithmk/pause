import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.preprocessing import Preprocessor


class PreprocessorTests(unittest.TestCase):
    def setUp(self):
        self.preprocessor = Preprocessor()

    def test_empty_text(self):
        result = self.preprocessor.preprocess("")
        self.assertEqual(result["original_text"], "")
        self.assertEqual(result["cleaned_text"], "")
        self.assertEqual(result["tokens"], [])
        self.assertEqual(result["emoji_list"], [])

    def test_none_text(self):
        result = self.preprocessor.preprocess(None)
        self.assertEqual(result["original_text"], "")
        self.assertEqual(result["cleaned_text"], "")

    def test_extra_whitespace(self):
        result = self.preprocessor.preprocess("hello    world\n\tthere")
        self.assertEqual(result["cleaned_text"], "hello world there")

    def test_url_normalization(self):
        result = self.preprocessor.preprocess("see https://example.com now")
        self.assertEqual(result["cleaned_text"], "see [URL] now")
        self.assertEqual(result["preserved_info"]["urls"], ["https://example.com"])

    def test_mention_and_hashtag(self):
        result = self.preprocessor.preprocess("hey @user check #news")
        self.assertEqual(result["cleaned_text"], "hey [MENTION] check [HASHTAG]")
        self.assertEqual(result["preserved_info"]["mentions"], ["@user"])
        self.assertEqual(result["preserved_info"]["hashtags"], ["#news"])

    def test_contractions_are_preserved(self):
        result = self.preprocessor.preprocess("don't won't I'm")
        self.assertIn("don't", result["cleaned_text"])
        self.assertIn("won't", result["cleaned_text"])
        self.assertNotIn("do not", result["cleaned_text"])
        self.assertIn("don't", result["preserved_info"]["contractions_found"])
        self.assertIn("won't", result["preserved_info"]["contractions_found"])
        self.assertIn("i'm", result["preserved_info"]["contractions_found"])

    def test_meaningful_punctuation_preserved(self):
        result = self.preprocessor.preprocess("What??? No!!!")
        self.assertIn("???", result["cleaned_text"])
        self.assertIn("!!!", result["cleaned_text"])
        self.assertEqual(result["punctuation_marks"]["question"], 3)
        self.assertEqual(result["punctuation_marks"]["exclamation"], 3)
        self.assertIn("???", result["punctuation_marks"]["meaningful_runs"])
        self.assertIn("!!!", result["punctuation_marks"]["meaningful_runs"])

    def test_emoji_extraction_and_preservation(self):
        result = self.preprocessor.preprocess("hi 😀 bye 🔥")
        self.assertIn("😀", result["cleaned_text"])
        self.assertTrue(result["emoji_list"])
        self.assertTrue(result["preserved_info"]["emojis"])

    def test_tokenize_preserves_punctuation(self):
        tokens = self.preprocessor.tokenize("don't go!!!")
        self.assertIn("don't", tokens)
        self.assertIn("!!!", tokens)

    def test_denormalize_restores_placeholders(self):
        result = self.preprocessor.preprocess("go to https://a.com @bob #fun")
        restored = self.preprocessor.denormalize_for_output(
            result["cleaned_text"], result["preserved_info"]
        )
        self.assertIn("https://a.com", restored)
        self.assertIn("@bob", restored)
        self.assertIn("#fun", restored)
        self.assertNotIn("[URL]", restored)

    def test_sentence_structure_kept(self):
        original = "I won't go. You can't make me!"
        result = self.preprocessor.preprocess(original)
        self.assertEqual(result["cleaned_text"], original)
        self.assertTrue(result["preserved_info"]["sentence_structure"])


if __name__ == "__main__":
    unittest.main()
